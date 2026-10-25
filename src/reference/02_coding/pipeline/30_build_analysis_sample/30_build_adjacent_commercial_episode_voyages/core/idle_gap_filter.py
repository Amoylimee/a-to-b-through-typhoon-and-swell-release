#!/usr/bin/env python3
"""Apply the frozen C2 idle-gap break rule to 2021 C1 candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CODING_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = PIPELINE_ROOT / "config/frozen/idle_gap_2021.json"
ARTIFACT_DIR = DATA_ROOT / "main/provenance/idle_gap_2021"
REPORT_DIR = DATA_ROOT / "main/validation/idle_gap_2021"
RUN_DIR = (
    CODING_ROOT
    / ".runs/build_multicall_first_passage_voyages/"
    "idle_gap_2021"
)
RETAINED_DIR = ARTIFACT_DIR / "candidate_passages_r4_retained_2021"
REJECTED_DIR = ARTIFACT_DIR / "candidate_passages_r4_rejected_2021"
SIDECAR_DIR = RUN_DIR / "parts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def verify_manifests(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config["apply_congestion_exemption"]:
        raise ValueError("C2.3 contract requires no congestion exemption")
    c1_path = CODING_ROOT / config["c1_output_manifest"]
    c2_path = CODING_ROOT / config["c2_threshold_output_manifest"]
    if sha256_file(c1_path) != config["c1_output_manifest_sha256"]:
        raise ValueError("C1 output manifest hash mismatch")
    if sha256_file(c2_path) != config["c2_threshold_output_manifest_sha256"]:
        raise ValueError("C2 threshold output manifest hash mismatch")
    c1 = load_json(c1_path)
    c2 = load_json(c2_path)
    if len(c1["candidate_parts"]) != config["task_count"]:
        raise ValueError("Unexpected C1 candidate part count")
    if len(c2["annotated_raw_stop_parts"]) != config["task_count"]:
        raise ValueError("Unexpected C2 annotated-stop part count")
    for candidate_item, stop_item in zip(
        c1["candidate_parts"], c2["annotated_raw_stop_parts"]
    ):
        candidate_path = CODING_ROOT / candidate_item["path"]
        stop_path = CODING_ROOT / stop_item["path"]
        if sha256_file(candidate_path) != candidate_item["sha256"]:
            raise ValueError(f"Candidate hash mismatch: {candidate_item['path']}")
        if sha256_file(stop_path) != stop_item["sha256"]:
            raise ValueError(f"Annotated-stop hash mismatch: {stop_item['path']}")
    return c1, c2


def annotate_candidate_breaks(
    candidates: pd.DataFrame,
    stops: pd.DataFrame,
) -> pd.DataFrame:
    result = candidates.copy()
    if result["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate_id in candidate part")
    if stops["stop_id"].duplicated().any():
        raise ValueError("Duplicate stop_id in annotated-stop part")
    expected_stop_count = (
        result["inbound_leg_sequence"] - result["outbound_leg_sequence"]
    )
    if not expected_stop_count.eq(result["n_intermediate_raw_stops"]).all():
        raise ValueError("Candidate-to-stop interval contract failed")
    if (expected_stop_count < 0).any():
        raise ValueError("Candidate contains a reversed stop interval")

    row_count = len(result)
    break_count = np.zeros(row_count, dtype=np.int64)
    first_stop_id = np.full(row_count, "", dtype=object)
    first_stop_sequence = np.full(row_count, -1, dtype=np.int64)
    first_stop_dwell = np.full(row_count, np.nan, dtype=np.float64)
    first_stop_threshold = np.full(row_count, np.nan, dtype=np.float64)
    first_stop_location = np.full(row_count, "", dtype=object)

    extreme = stops[
        stops["idle_gap_exceeds_unadjusted_threshold"]
    ].copy()
    candidate_groups = result.groupby("mmsi", sort=False).indices
    extreme_groups = extreme.groupby("mmsi", sort=False).indices
    for mmsi, candidate_positions in candidate_groups.items():
        positions = np.asarray(candidate_positions, dtype=np.int64)
        stop_positions = extreme_groups.get(mmsi)
        if stop_positions is None:
            continue
        vessel_breaks = extreme.iloc[np.asarray(stop_positions)].sort_values(
            "stop_sequence", kind="mergesort"
        )
        sequences = vessel_breaks["stop_sequence"].to_numpy(dtype=np.int64)
        if len(np.unique(sequences)) != len(sequences):
            raise ValueError(f"Duplicate break stop_sequence for MMSI {mmsi}")
        outbound = result.iloc[positions][
            "outbound_leg_sequence"
        ].to_numpy(dtype=np.int64)
        inbound = result.iloc[positions][
            "inbound_leg_sequence"
        ].to_numpy(dtype=np.int64)
        left = np.searchsorted(sequences, outbound, side="left")
        right = np.searchsorted(sequences, inbound, side="left")
        counts = right - left
        break_count[positions] = counts
        has_break = counts > 0
        if not has_break.any():
            continue
        target_positions = positions[has_break]
        first_indices = left[has_break]
        first_rows = vessel_breaks.iloc[first_indices]
        first_stop_id[target_positions] = first_rows["stop_id"].astype(str)
        first_stop_sequence[target_positions] = first_rows[
            "stop_sequence"
        ].to_numpy(dtype=np.int64)
        first_stop_dwell[target_positions] = first_rows[
            "dwell_days"
        ].to_numpy(dtype=float)
        first_stop_threshold[target_positions] = first_rows[
            "idle_gap_threshold_days"
        ].to_numpy(dtype=float)
        first_stop_location[target_positions] = first_rows[
            "resolved_topology_city_id"
        ].fillna("").astype(str)

    result["n_r4_idle_break_stops"] = break_count
    result["r4_idle_break"] = break_count > 0
    result["first_r4_break_stop_id"] = first_stop_id
    result["first_r4_break_stop_sequence"] = first_stop_sequence
    result["first_r4_break_dwell_days"] = first_stop_dwell
    result["first_r4_break_threshold_days"] = first_stop_threshold
    result["first_r4_break_location"] = first_stop_location
    result["congestion_exempt"] = False
    result["r4_rule_version"] = "c2_2021_p99_no_congestion_exemption"
    return result


def add_post_r4_support(retained: pd.DataFrame) -> pd.DataFrame:
    result = retained.copy()
    if result.empty:
        result["r3_recurrence_count_post_r4"] = pd.Series(dtype="int64")
        result["r3_singleton_post_r4"] = pd.Series(dtype="bool")
        return result
    recurrence = result.groupby(
        ["imo", "origin_city", "dest_city", "period_id"],
        sort=False,
        dropna=False,
    )["candidate_id"].transform("size")
    result["r3_recurrence_count_post_r4"] = recurrence.astype(np.int64)
    result["r3_singleton_post_r4"] = (
        result["r3_window_eligible"]
        & result["r3_recurrence_count_post_r4"].eq(1)
    )
    return result


def duration_breakdown(
    annotated: pd.DataFrame,
    edges_days: list[float],
    labels: list[str],
) -> list[dict[str, Any]]:
    bins = list(edges_days) + [np.inf]
    total_days = annotated["total_hours"] / 24.0
    category = pd.cut(
        total_days,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    frame = pd.DataFrame(
        {
            "duration_bin": category,
            "candidate_rows": 1,
            "rejected_candidates": annotated["r4_idle_break"].astype(int),
        }
    )
    summary = (
        frame.groupby("duration_bin", observed=False, sort=False)[
            ["candidate_rows", "rejected_candidates"]
        ]
        .sum()
        .reset_index()
    )
    return json.loads(summary.to_json(orient="records"))


def type_breakdown(annotated: pd.DataFrame) -> list[dict[str, Any]]:
    summary = (
        annotated.groupby(
            "detailed_vessel_type_normalized",
            observed=True,
            sort=True,
            dropna=False,
        )
        .agg(
            candidate_rows=("candidate_id", "size"),
            rejected_candidates=("r4_idle_break", "sum"),
        )
        .reset_index()
    )
    return json.loads(summary.to_json(orient="records"))


def part_is_complete(
    sidecar_path: Path,
    retained_path: Path,
    rejected_path: Path,
    contract_hash: str,
    candidate_source_hash: str,
    stop_source_hash: str,
) -> bool:
    if not (
        sidecar_path.is_file()
        and retained_path.is_file()
        and rejected_path.is_file()
    ):
        return False
    try:
        payload = load_json(sidecar_path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("contract_hash") == contract_hash
        and payload.get("candidate_source_sha256") == candidate_source_hash
        and payload.get("stop_source_sha256") == stop_source_hash
        and payload.get("retained_sha256") == sha256_file(retained_path)
        and payload.get("rejected_sha256") == sha256_file(rejected_path)
    )


def process_part(
    task_id: int,
    candidate_path_text: str,
    stop_path_text: str,
    retained_path_text: str,
    rejected_path_text: str,
    sidecar_path_text: str,
    contract_hash: str,
    candidate_source_hash: str,
    stop_source_hash: str,
    duration_edges: list[float],
    duration_labels: list[str],
) -> dict[str, Any]:
    candidate_path = Path(candidate_path_text)
    stop_path = Path(stop_path_text)
    retained_path = Path(retained_path_text)
    rejected_path = Path(rejected_path_text)
    sidecar_path = Path(sidecar_path_text)
    candidates = pd.read_feather(candidate_path)
    stops = pd.read_feather(stop_path)
    annotated = annotate_candidate_breaks(candidates, stops)
    retained = add_post_r4_support(
        annotated.loc[~annotated["r4_idle_break"]].reset_index(drop=True)
    )
    rejected = annotated.loc[
        annotated["r4_idle_break"]
    ].reset_index(drop=True)
    retained_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    retained.to_feather(
        retained_path, compression="zstd", compression_level=6
    )
    rejected.to_feather(
        rejected_path, compression="zstd", compression_level=6
    )
    payload = {
        "task_id": task_id,
        "contract_hash": contract_hash,
        "candidate_source_sha256": candidate_source_hash,
        "stop_source_sha256": stop_source_hash,
        "candidate_rows": int(len(annotated)),
        "retained_rows": int(len(retained)),
        "rejected_rows": int(len(rejected)),
        "candidates_with_multiple_breaks": int(
            annotated["n_r4_idle_break_stops"].gt(1).sum()
        ),
        "input_unique_pairs": int(annotated["pair_id"].nunique()),
        "retained_unique_pairs": int(retained["pair_id"].nunique()),
        "input_unique_vessels": int(annotated["mmsi"].nunique()),
        "retained_unique_vessels": int(retained["mmsi"].nunique()),
        "type_breakdown": type_breakdown(annotated),
        "duration_breakdown": duration_breakdown(
            annotated, duration_edges, duration_labels
        ),
        "retained_sha256": sha256_file(retained_path),
        "rejected_sha256": sha256_file(rejected_path),
    }
    write_json(sidecar_path, payload)
    return payload


def run_parts(
    config: dict[str, Any],
    c1_manifest: dict[str, Any],
    c2_manifest: dict[str, Any],
    contract_hash: str,
) -> tuple[list[dict[str, Any]], int]:
    RETAINED_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    for task_id, (candidate_item, stop_item) in enumerate(
        zip(
            c1_manifest["candidate_parts"],
            c2_manifest["annotated_raw_stop_parts"],
        )
    ):
        candidate_path = CODING_ROOT / candidate_item["path"]
        stop_path = CODING_ROOT / stop_item["path"]
        retained_path = RETAINED_DIR / f"part-{task_id:03d}.feather"
        rejected_path = REJECTED_DIR / f"part-{task_id:03d}.feather"
        sidecar_path = SIDECAR_DIR / f"part-{task_id:03d}.json"
        if part_is_complete(
            sidecar_path,
            retained_path,
            rejected_path,
            contract_hash,
            candidate_item["sha256"],
            stop_item["sha256"],
        ):
            completed.append(load_json(sidecar_path))
        else:
            pending.append(
                (
                    task_id,
                    str(candidate_path),
                    str(stop_path),
                    str(retained_path),
                    str(rejected_path),
                    str(sidecar_path),
                    contract_hash,
                    candidate_item["sha256"],
                    stop_item["sha256"],
                    config["duration_bin_edges_days"],
                    config["duration_bin_labels"],
                )
            )
    if pending:
        worker_count = min(config["worker_count"], len(pending), 16)
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as executor:
            futures = {
                executor.submit(process_part, *task): task[0]
                for task in pending
            }
            for future in as_completed(futures):
                payload = future.result()
                completed.append(payload)
                print(
                    f"Part {payload['task_id']:03d}: "
                    f"{payload['retained_rows']:,} retained, "
                    f"{payload['rejected_rows']:,} rejected",
                    flush=True,
                )
    return sorted(completed, key=lambda row: row["task_id"]), len(pending)


def aggregate_breakdown(
    part_payloads: list[dict[str, Any]],
    key: str,
    group_column: str,
) -> pd.DataFrame:
    frames = [pd.DataFrame(payload[key]) for payload in part_payloads]
    combined = pd.concat(frames, ignore_index=True)
    result = (
        combined.groupby(group_column, sort=False, dropna=False)[
            ["candidate_rows", "rejected_candidates"]
        ]
        .sum()
        .reset_index()
    )
    result["retained_candidates"] = (
        result["candidate_rows"] - result["rejected_candidates"]
    )
    result["rejection_rate"] = (
        result["rejected_candidates"] / result["candidate_rows"]
    )
    return result


def collect_unique_values(
    paths: list[Path], column: str
) -> set[str]:
    values: set[str] = set()
    for path in paths:
        frame = pd.read_feather(path, columns=[column])
        values.update(frame[column].dropna().astype(str).unique().tolist())
    return values


def collect_value_counts(
    paths: list[Path], column: str
) -> pd.Series:
    pieces: list[pd.Series] = []
    for path in paths:
        frame = pd.read_feather(path, columns=[column])
        pieces.append(frame[column].value_counts(sort=False))
    if not pieces:
        return pd.Series(dtype="int64")
    return (
        pd.concat(pieces, axis=1)
        .fillna(0)
        .sum(axis=1)
        .astype(np.int64)
    )


def pair_support_attrition(
    input_paths: list[Path],
    retained_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_counts = collect_value_counts(input_paths, "pair_id")
    retained_counts = collect_value_counts(retained_paths, "pair_id")
    detail = (
        input_counts.rename("input_candidates")
        .to_frame()
        .join(retained_counts.rename("retained_candidates"), how="left")
        .fillna(0)
    )
    detail["retained_candidates"] = detail[
        "retained_candidates"
    ].astype(np.int64)
    detail["lost_after_r4"] = detail["retained_candidates"].eq(0)
    detail["support_bin"] = pd.cut(
        detail["input_candidates"],
        bins=[0, 1, 2, 5, 10, np.inf],
        labels=["1", "2", "3-5", "6-10", "11+"],
        right=True,
    )
    summary = (
        detail.groupby("support_bin", observed=False, sort=False)
        .agg(
            input_pairs=("lost_after_r4", "size"),
            lost_pairs=("lost_after_r4", "sum"),
        )
        .reset_index()
    )
    summary["retained_pairs"] = (
        summary["input_pairs"] - summary["lost_pairs"]
    )
    summary["lost_pair_rate"] = (
        summary["lost_pairs"] / summary["input_pairs"]
    )
    support_order = ["1", "2", "3-5", "6-10", "11+"]
    summary["support_bin"] = pd.Categorical(
        summary["support_bin"], categories=support_order, ordered=True
    )
    summary = summary.sort_values("support_bin", kind="mergesort").reset_index(
        drop=True
    )
    lost = (
        detail.loc[detail["lost_after_r4"]]
        .reset_index(names="pair_id")
        .sort_values(
            ["input_candidates", "pair_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return detail, summary, lost


def write_report(
    summary: dict[str, Any],
    type_summary: pd.DataFrame,
    duration_summary: pd.DataFrame,
    pair_support_summary: pd.DataFrame,
) -> None:
    top_types = type_summary.sort_values(
        "rejected_candidates", ascending=False, kind="mergesort"
    ).head(10)
    type_lines = "\n".join(
        (
            f"- {row.detailed_vessel_type_normalized}: "
            f"{int(row.rejected_candidates):,} of "
            f"{int(row.candidate_rows):,} rejected "
            f"({row.rejection_rate:.3%})."
        )
        for row in top_types.itertuples(index=False)
    )
    duration_lines = "\n".join(
        (
            f"- {row.duration_bin}: {int(row.rejected_candidates):,} of "
            f"{int(row.candidate_rows):,} rejected "
            f"({row.rejection_rate:.3%})."
        )
        for row in duration_summary.itertuples(index=False)
    )
    support_lines = "\n".join(
        (
            f"- {row.support_bin} input candidate(s): "
            f"{int(row.lost_pairs):,} of {int(row.input_pairs):,} pairs lost "
            f"({row.lost_pair_rate:.3%})."
        )
        for row in pair_support_summary.itertuples(index=False)
    )
    body = f"""# C2.3 Candidate Idle-Gap Attrition

## Plain-Language Result

The filter cuts a candidate chain only when it crosses at least one frozen
extreme idle gap. Candidates before or after the same stop remain available.
No congestion exemption, R2 exclusion, R3 exclusion, or R5 rule was applied.

## Overall Attrition

- Input candidates: {summary['input_candidates']:,}.
- Retained candidates: {summary['retained_candidates']:,}.
- Rejected candidates: {summary['rejected_candidates']:,}
  ({summary['candidate_rejection_rate']:.3%}).
- Candidates crossing multiple break stops:
  {summary['candidates_with_multiple_breaks']:,}.
- Input directed city pairs: {summary['input_unique_pairs']:,}.
- Retained directed city pairs: {summary['retained_unique_pairs']:,}.
- Pairs with no retained candidate: {summary['lost_unique_pairs']:,}
  ({summary['lost_pair_share']:.3%}).
- Input vessels: {summary['input_unique_vessels']:,}.
- Retained vessels: {summary['retained_unique_vessels']:,}.
- Vessels with no retained candidate: {summary['lost_unique_vessels']:,}.
- Lost pairs that had only one input candidate:
  {summary['lost_singleton_pairs']:,}
  ({summary['lost_singleton_pair_share']:.3%} of lost pairs).

## Largest Ship-Type Rejection Counts

{type_lines}

## Duration Pattern

{duration_lines}

## Pair-Support Pattern

{support_lines}

## Interpretation

The rule is designed to remove candidate chains that bridge an exceptionally
long discontinuity, not to remove operational stops. Rejection should be
concentrated in longer candidate spans because those spans cross more
intermediate stops. The retained table preserves total elapsed time, including
ordinary loading, unloading, bunkering, waiting, and congestion.

## Boundary

C2 is complete after validation of these outputs. Do not start AIS work or C3
without explicit authorization.
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "c2_3_candidate_attrition.md").write_text(
        body, encoding="utf-8"
    )


def run(config_path: Path) -> None:
    config = load_json(config_path)
    c1_manifest, c2_manifest = verify_manifests(config)
    script_hash = sha256_file(Path(__file__).resolve())
    contract_payload = {
        "config": config,
        "script_sha256": script_hash,
    }
    contract_hash = hashlib.sha256(
        json.dumps(
            contract_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    part_payloads, pending_count = run_parts(
        config, c1_manifest, c2_manifest, contract_hash
    )
    input_count = sum(row["candidate_rows"] for row in part_payloads)
    retained_count = sum(row["retained_rows"] for row in part_payloads)
    rejected_count = sum(row["rejected_rows"] for row in part_payloads)
    if input_count != config["expected_candidate_count"]:
        raise ValueError(
            f"Expected {config['expected_candidate_count']} candidates, "
            f"got {input_count}"
        )
    if retained_count + rejected_count != input_count:
        raise ValueError("Retained and rejected counts do not reconstruct input")

    type_summary = aggregate_breakdown(
        part_payloads,
        "type_breakdown",
        "detailed_vessel_type_normalized",
    ).sort_values(
        ["candidate_rows", "detailed_vessel_type_normalized"],
        ascending=[False, True],
        kind="mergesort",
    )
    duration_summary = aggregate_breakdown(
        part_payloads, "duration_breakdown", "duration_bin"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    type_summary.to_csv(
        REPORT_DIR / "c2_3_attrition_by_ship_type.csv", index=False
    )
    duration_summary.to_csv(
        REPORT_DIR / "c2_3_attrition_by_duration.csv", index=False
    )

    input_paths = [
        CODING_ROOT / item["path"] for item in c1_manifest["candidate_parts"]
    ]
    retained_paths = [
        RETAINED_DIR / f"part-{task_id:03d}.feather"
        for task_id in range(config["task_count"])
    ]
    input_pairs = collect_unique_values(input_paths, "pair_id")
    retained_pairs = collect_unique_values(retained_paths, "pair_id")
    if len(input_pairs) != config["expected_pair_count"]:
        raise ValueError("Input pair count does not reproduce the C0 contract")
    _, pair_support_summary, lost_pair_detail = pair_support_attrition(
        input_paths, retained_paths
    )
    lost_pairs = set(lost_pair_detail["pair_id"])
    lost_pair_detail.to_csv(
        REPORT_DIR / "c2_3_lost_pairs.csv", index=False
    )
    lost_pair_detail.head(100).to_csv(
        REPORT_DIR / "c2_3_lost_pairs_top_support.csv", index=False
    )
    pair_support_summary.to_csv(
        REPORT_DIR / "c2_3_pair_support_attrition.csv", index=False
    )
    input_vessels = collect_unique_values(input_paths, "mmsi")
    retained_vessels = collect_unique_values(retained_paths, "mmsi")
    lost_vessels = sorted(input_vessels - retained_vessels)
    pd.DataFrame({"mmsi": lost_vessels}).to_csv(
        REPORT_DIR / "c2_3_lost_vessels.csv", index=False
    )

    candidate_break_incidences = 0
    direct_rejected_candidates = 0
    retained_max_time_residual = 0.0
    for task_id in range(config["task_count"]):
        rejected_path = REJECTED_DIR / f"part-{task_id:03d}.feather"
        retained_path = RETAINED_DIR / f"part-{task_id:03d}.feather"
        rejected_check = pd.read_feather(
            rejected_path,
            columns=["n_r4_idle_break_stops", "n_intermediate_raw_stops"],
        )
        candidate_break_incidences += int(
            rejected_check["n_r4_idle_break_stops"].sum()
        )
        direct_rejected_candidates += int(
            rejected_check["n_intermediate_raw_stops"].eq(0).sum()
        )
        retained_residual = pd.read_feather(
            retained_path, columns=["time_identity_residual_hours"]
        )["time_identity_residual_hours"].abs()
        retained_max_time_residual = max(
            retained_max_time_residual,
            float(retained_residual.max()) if len(retained_residual) else 0.0,
        )
    stop_break_incidences = 0
    extreme_stop_count = 0
    for stop_item in c2_manifest["annotated_raw_stop_parts"]:
        stops = pd.read_feather(
            CODING_ROOT / stop_item["path"],
            columns=[
                "idle_gap_exceeds_unadjusted_threshold",
                "candidate_intermediate_usage_count",
            ],
        )
        extreme = stops["idle_gap_exceeds_unadjusted_threshold"]
        extreme_stop_count += int(extreme.sum())
        stop_break_incidences += int(
            stops.loc[extreme, "candidate_intermediate_usage_count"].sum()
        )
    if candidate_break_incidences != stop_break_incidences:
        raise ValueError("Candidate/stop break-incidence identity failed")
    if direct_rejected_candidates:
        raise ValueError("A direct candidate was incorrectly rejected")

    lost_singleton_pairs = int(
        (
            lost_pair_detail["input_candidates"] == 1
        ).sum()
    )
    summary = {
        "analysis_year": config["analysis_year"],
        "r4_rule": config["r4_rule"],
        "congestion_exemption_applied": False,
        "input_candidates": input_count,
        "retained_candidates": retained_count,
        "rejected_candidates": rejected_count,
        "candidate_rejection_rate": rejected_count / input_count,
        "candidates_with_multiple_breaks": sum(
            row["candidates_with_multiple_breaks"] for row in part_payloads
        ),
        "input_unique_pairs": len(input_pairs),
        "retained_unique_pairs": len(retained_pairs),
        "lost_unique_pairs": len(lost_pairs),
        "lost_pair_share": len(lost_pairs) / len(input_pairs),
        "lost_singleton_pairs": lost_singleton_pairs,
        "lost_singleton_pair_share": (
            lost_singleton_pairs / len(lost_pairs) if lost_pairs else 0.0
        ),
        "input_unique_vessels": len(input_vessels),
        "retained_unique_vessels": len(retained_vessels),
        "lost_unique_vessels": len(lost_vessels),
        "r2_exclusion_applied": False,
        "r3_exclusion_applied": False,
        "r5_applied": False,
        "extreme_stop_count": extreme_stop_count,
        "candidate_break_incidences": candidate_break_incidences,
        "stop_break_incidences": stop_break_incidences,
        "direct_rejected_candidates": direct_rejected_candidates,
        "retained_max_time_identity_residual_hours": (
            retained_max_time_residual
        ),
    }
    write_json(REPORT_DIR / "c2_3_candidate_attrition_summary.json", summary)
    write_report(
        summary,
        type_summary,
        duration_summary,
        pair_support_summary,
    )
    validation = {
        "candidate_partition_identity_passed": (
            retained_count + rejected_count == input_count
        ),
        "candidate_stop_break_incidence_identity_passed": (
            candidate_break_incidences == stop_break_incidences
        ),
        "direct_candidate_rejection_check_passed": (
            direct_rejected_candidates == 0
        ),
        "input_pair_contract_passed": (
            len(input_pairs) == config["expected_pair_count"]
        ),
        "retained_time_identity_passed": (
            retained_max_time_residual <= 1e-6
        ),
        "candidate_break_incidences": candidate_break_incidences,
        "stop_break_incidences": stop_break_incidences,
        "retained_max_time_identity_residual_hours": (
            retained_max_time_residual
        ),
    }
    if not all(
        value
        for key, value in validation.items()
        if key.endswith("_passed")
    ):
        raise ValueError("C2.3 validation gate failed")
    write_json(REPORT_DIR / "c2_3_validation.json", validation)

    part_manifest: list[dict[str, Any]] = []
    for task_id, payload in enumerate(part_payloads):
        part_manifest.append(
            {
                "task_id": task_id,
                "retained_path": str(
                    (
                        RETAINED_DIR / f"part-{task_id:03d}.feather"
                    ).relative_to(CODING_ROOT)
                ),
                "retained_rows": payload["retained_rows"],
                "retained_sha256": payload["retained_sha256"],
                "rejected_path": str(
                    (
                        REJECTED_DIR / f"part-{task_id:03d}.feather"
                    ).relative_to(CODING_ROOT)
                ),
                "rejected_rows": payload["rejected_rows"],
                "rejected_sha256": payload["rejected_sha256"],
            }
        )
    tracked_reports = [
        REPORT_DIR / "c2_3_attrition_by_ship_type.csv",
        REPORT_DIR / "c2_3_attrition_by_duration.csv",
        REPORT_DIR / "c2_3_lost_pairs.csv",
        REPORT_DIR / "c2_3_lost_pairs_top_support.csv",
        REPORT_DIR / "c2_3_pair_support_attrition.csv",
        REPORT_DIR / "c2_3_lost_vessels.csv",
        REPORT_DIR / "c2_3_candidate_attrition_summary.json",
        REPORT_DIR / "c2_3_validation.json",
        REPORT_DIR / "c2_3_candidate_attrition.md",
    ]
    output_manifest = {
        "analysis_year": config["analysis_year"],
        "contract_hash": contract_hash,
        "c1_output_manifest_sha256": config["c1_output_manifest_sha256"],
        "c2_threshold_output_manifest_sha256": config[
            "c2_threshold_output_manifest_sha256"
        ],
        "parts": part_manifest,
        "reports": {
            str(path.relative_to(CODING_ROOT)): sha256_file(path)
            for path in tracked_reports
        },
    }
    write_json(ARTIFACT_DIR / "c2_3_output_manifest.json", output_manifest)
    write_json(
        RUN_DIR / "run_summary_latest.json",
        {
            "contract_hash": contract_hash,
            "pending_parts_this_run": pending_count,
            "completed_parts": len(part_payloads),
            "input_candidates": input_count,
            "retained_candidates": retained_count,
            "rejected_candidates": rejected_count,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config.resolve())
