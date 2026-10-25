#!/usr/bin/env python3
"""Rebuild 2021/2025 samples under the strict-location 30-day C2 rule."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from . import frozen_quality as c3
from . import idle_gap_filter as c2_filter
from . import main_sample as c4
from .validation_core import assign_suez_dependence_bin

CODING_ROOT = Path(__file__).resolve().parents[4]
C5_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = C5_ROOT / "config/frozen/main_rebuild_2021_2025.json"
DEFAULT_ARTIFACT_DIR = DATA_ROOT / "main/provenance/main_rebuild_2021_2025"
DEFAULT_REPORT_DIR = DATA_ROOT / "main/validation/main_rebuild_2021_2025"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verified_path(config: dict[str, Any], key: str) -> Path:
    path = CODING_ROOT / config[key]
    if sha256_file(path) != config[f"{key}_sha256"]:
        raise ValueError(f"Frozen input hash mismatch: {key}")
    return path


def verify_contract(config: dict[str, Any]) -> None:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=CODING_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != config["approved_branch"]:
        raise ValueError("Unexpected branch")
    if config["analysis_years"] != [2021, 2025]:
        raise ValueError("Only the two endpoint years are allowed")
    if config["hard_break_threshold_days"] != 30.0:
        raise ValueError("The C2 sensitivity threshold must be 30 days")
    if config["read_2022_2024"] or config["allow_baseline_overwrite"]:
        raise ValueError(
            "The sensitivity cannot read middle years or overwrite baseline"
        )
    frozen_values = {
        "route_circuity_threshold": 2.0,
        "duration_threshold_days": 30.0,
        "positive_route_speed_threshold_knots": 1.0,
        "same_node_route_backstop_km": 2000.0,
        "q_lower": 0.01,
        "q_upper": 0.99,
    }
    for key, value in frozen_values.items():
        if float(config[key]) != value:
            raise ValueError(f"Unexpected downstream rule: {key}")
    for key in (
        "c1_output_manifest_2021",
        "c2_stop_manifest_2021",
        "c5_candidate_manifest_2025",
        "c5_c2_manifest_2025",
        "yardstick_2021",
        "yardstick_2025",
        "transition_2021",
        "transition_2025",
        "frozen_2021_node_routes",
        "c3_aux_node_routes_2021",
        "c5_new_node_routes_2025",
        "p4_city_shiptype_exposure",
    ):
        verified_path(config, key)


def source_parts(
    year: int, config: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if year == 2021:
        candidates = load_json(CODING_ROOT / config["c1_output_manifest_2021"])[
            "candidate_parts"
        ]
        stops = load_json(CODING_ROOT / config["c2_stop_manifest_2021"])[
            "annotated_raw_stop_parts"
        ]
        return list(zip(candidates, stops, strict=True))
    candidates_manifest = load_json(CODING_ROOT / config["c5_candidate_manifest_2025"])
    candidates = [
        {"path": path, "sha256": record["sha256"]}
        for path, record in candidates_manifest["outputs"].items()
        if "/candidate_passages_2025/part-" in path
    ]
    candidates.sort(key=lambda item: item["path"])
    stop_manifest = load_json(CODING_ROOT / config["c5_c2_manifest_2025"])
    stops = [
        {
            "path": item["annotated_stop_path"],
            "sha256": item["annotated_stop_sha256"],
        }
        for item in sorted(stop_manifest["parts"], key=lambda item: item["task_id"])
    ]
    return list(zip(candidates, stops, strict=True))


def materialize_c2(
    year: int,
    config: dict[str, Any],
    artifact_dir: Path,
    config_hash: str,
) -> dict[str, Any]:
    records = []
    totals = Counter()
    for task_id, (candidate_item, stop_item) in enumerate(source_parts(year, config)):
        candidate_path = CODING_ROOT / candidate_item["path"]
        stop_path = CODING_ROOT / stop_item["path"]
        if sha256_file(candidate_path) != candidate_item["sha256"]:
            raise ValueError(f"Candidate source changed: {candidate_path}")
        if sha256_file(stop_path) != stop_item["sha256"]:
            raise ValueError(f"Stop source changed: {stop_path}")
        target = artifact_dir / f"c2_retained_{year}/part-{task_id:03d}.feather"
        sidecar = artifact_dir / f"c2_sidecars_{year}/part-{task_id:03d}.json"
        contract_hash = hashlib.sha256(
            (
                config_hash + candidate_item["sha256"] + stop_item["sha256"] + str(year)
            ).encode()
        ).hexdigest()
        if target.is_file() and sidecar.is_file():
            prior = load_json(sidecar)
            if prior.get("contract_hash") == contract_hash and prior.get(
                "output_sha256"
            ) == sha256_file(target):
                records.append(prior)
                totals.update(
                    {
                        "input_candidates": prior["input_candidates"],
                        "retained_candidates": prior["retained_candidates"],
                        "excluded_candidates": prior["excluded_candidates"],
                    }
                )
                continue
        candidates = pd.read_feather(candidate_path)
        stops = pd.read_feather(stop_path)
        stops["idle_gap_exceeds_unadjusted_threshold"] = stops[
            "is_strict_same_location"
        ].astype(bool) & stops["dwell_days"].gt(
            float(config["hard_break_threshold_days"])
        )
        stops["idle_gap_threshold_days"] = float(config["hard_break_threshold_days"])
        annotated = c2_filter.annotate_candidate_breaks(candidates, stops)
        annotated["r4_rule_version"] = "strict_same_location_gt_30_days"
        retained = c2_filter.add_post_r4_support(
            annotated.loc[~annotated["r4_idle_break"]].reset_index(drop=True)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        retained.to_feather(target, compression="zstd", compression_level=6)
        payload = {
            "task_id": task_id,
            "contract_hash": contract_hash,
            "source_candidate_path": candidate_item["path"],
            "source_stop_path": stop_item["path"],
            "input_candidates": int(len(candidates)),
            "retained_candidates": int(len(retained)),
            "excluded_candidates": int(len(candidates) - len(retained)),
            "retained_path": str(target.relative_to(CODING_ROOT)),
            "retained_sha256": sha256_file(target),
            "output_sha256": sha256_file(target),
        }
        write_json(sidecar, payload)
        records.append(payload)
        totals.update(
            {
                "input_candidates": len(candidates),
                "retained_candidates": len(retained),
                "excluded_candidates": len(candidates) - len(retained),
            }
        )
    manifest = {
        "status": "pass",
        "analysis_year": year,
        "rule": "strict_same_location_gt_30_days",
        "parts": records,
        "summary": {key: int(value) for key, value in totals.items()},
    }
    manifest_path = artifact_dir / f"c2_30_day_manifest_{year}.json"
    write_json(manifest_path, manifest)
    return manifest


def city_node_table(direct: pd.DataFrame) -> pd.DataFrame:
    origin = direct[
        [
            "origin_city",
            "origin_network_node_id",
        ]
    ].rename(
        columns={
            "origin_city": "city",
            "origin_network_node_id": "network_node_id",
        }
    )
    destination = direct[
        [
            "dest_city",
            "dest_network_node_id",
        ]
    ].rename(
        columns={
            "dest_city": "city",
            "dest_network_node_id": "network_node_id",
        }
    )
    combined = pd.concat([origin, destination], ignore_index=True)
    if combined.groupby("city")["network_node_id"].nunique().gt(1).any():
        raise ValueError("A city maps to multiple network nodes")
    return combined.drop_duplicates("city")


def node_coordinate_table(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_feather(
            path,
            columns=[
                "origin_network_node_id",
                "origin_network_lon",
                "origin_network_lat",
                "dest_network_node_id",
                "dest_network_lon",
                "dest_network_lat",
            ],
        )
        frames.extend(
            [
                frame[
                    [
                        "origin_network_node_id",
                        "origin_network_lon",
                        "origin_network_lat",
                    ]
                ].rename(
                    columns={
                        "origin_network_node_id": "network_node_id",
                        "origin_network_lon": "network_lon",
                        "origin_network_lat": "network_lat",
                    }
                ),
                frame[
                    [
                        "dest_network_node_id",
                        "dest_network_lon",
                        "dest_network_lat",
                    ]
                ].rename(
                    columns={
                        "dest_network_node_id": "network_node_id",
                        "dest_network_lon": "network_lon",
                        "dest_network_lat": "network_lat",
                    }
                ),
            ]
        )
    combined = pd.concat(frames, ignore_index=True)
    coordinate_counts = combined.groupby("network_node_id")[
        ["network_lon", "network_lat"]
    ].nunique()
    if coordinate_counts.gt(1).any().any():
        raise ValueError("A network node has conflicting coordinates")
    return combined.drop_duplicates("network_node_id")


def build_transition_support(
    year: int,
    config: dict[str, Any],
    c2_manifest: dict[str, Any],
    artifact_dir: Path,
) -> tuple[Path, dict[str, int]]:
    usage = c3.enumerate_transition_usage(
        c2_manifest, str(config["sequence_separator"])
    )
    direct = pd.read_feather(
        CODING_ROOT / config[f"yardstick_{year}"],
        columns=[
            "pair_id",
            "origin_city",
            "dest_city",
            "d_base",
            "same_network_node",
            "origin_network_node_id",
            "dest_network_node_id",
        ],
    )
    existing = pd.read_feather(
        CODING_ROOT / config[f"transition_{year}"],
        columns=["pair_id", "d_base", "same_network_node"],
    )
    lookup = (
        pd.concat(
            [
                direct[["pair_id", "d_base", "same_network_node"]],
                existing,
            ],
            ignore_index=True,
        )
        .drop_duplicates("pair_id")
        .reset_index(drop=True)
    )
    transition = usage.merge(lookup, on="pair_id", how="left", validate="one_to_one")
    missing = transition.loc[transition["d_base"].isna(), "pair_id"]
    extension = pd.DataFrame()
    new_node_routes = 0
    if not missing.empty:
        missing_frame = transition.loc[
            transition["d_base"].isna(),
            ["pair_id", "origin_city", "dest_city"],
        ].copy()
        nodes = city_node_table(direct)
        extra_key = (
            "c3_aux_node_routes_2021" if year == 2021 else "c5_new_node_routes_2025"
        )
        nodes = nodes.merge(
            node_coordinate_table(
                [
                    CODING_ROOT / config["frozen_2021_node_routes"],
                    CODING_ROOT / config[extra_key],
                ]
            ),
            on="network_node_id",
            how="left",
            validate="many_to_one",
        )
        if nodes[["network_lon", "network_lat"]].isna().any().any():
            raise ValueError("A required city node lacks coordinates")
        origin = nodes.rename(
            columns={
                "city": "origin_city",
                "network_node_id": "origin_network_node_id",
                "network_lon": "origin_network_lon",
                "network_lat": "origin_network_lat",
            }
        )
        destination = nodes.rename(
            columns={
                "city": "dest_city",
                "network_node_id": "dest_network_node_id",
                "network_lon": "dest_network_lon",
                "network_lat": "dest_network_lat",
            }
        )
        extension = missing_frame.merge(
            origin, on="origin_city", validate="many_to_one"
        ).merge(destination, on="dest_city", validate="many_to_one")
        extension["node_pair_id"] = (
            extension["origin_network_node_id"]
            + "__"
            + extension["dest_network_node_id"]
        )
        extension["same_network_node"] = extension["origin_network_node_id"].eq(
            extension["dest_network_node_id"]
        )
        node_sources = [
            direct.assign(
                node_pair_id=(
                    direct["origin_network_node_id"]
                    + "__"
                    + direct["dest_network_node_id"]
                )
            )[["node_pair_id", "d_base"]],
            pd.read_feather(
                CODING_ROOT / config["frozen_2021_node_routes"],
                columns=["node_pair_id", "d_base"],
            ),
        ]
        node_sources.append(
            pd.read_feather(
                CODING_ROOT / config[extra_key],
                columns=["node_pair_id", "d_base"],
            )
        )
        known = pd.concat(node_sources, ignore_index=True)
        if known.groupby("node_pair_id")["d_base"].nunique().gt(1).any():
            raise ValueError("Known node-route distances disagree")
        known = known.drop_duplicates("node_pair_id")
        extension = extension.merge(
            known, on="node_pair_id", how="left", validate="many_to_one"
        )
        extension.loc[extension["same_network_node"], "d_base"] = 0.0
        unresolved = (
            extension.loc[
                extension["d_base"].isna(),
                [
                    "node_pair_id",
                    "origin_network_node_id",
                    "origin_network_lon",
                    "origin_network_lat",
                    "dest_network_node_id",
                    "dest_network_lon",
                    "dest_network_lat",
                ],
            ]
            .drop_duplicates("node_pair_id")
            .reset_index(drop=True)
        )
        routed = c3.route_unresolved_nodes(
            unresolved, config, artifact_dir / f"new_transition_routes_{year}"
        )
        new_node_routes = len(unresolved)
        if not routed.empty:
            extension = extension.merge(
                routed[["node_pair_id", "d_base"]].rename(
                    columns={"d_base": "new_d_base"}
                ),
                on="node_pair_id",
                how="left",
                validate="many_to_one",
            )
            extension["d_base"] = extension["d_base"].fillna(extension["new_d_base"])
            extension = extension.drop(columns="new_d_base")
        if extension["d_base"].isna().any():
            raise ValueError("A new transition route remains unresolved")
        fill = extension[["pair_id", "d_base", "same_network_node"]].rename(
            columns={
                "d_base": "extension_d_base",
                "same_network_node": "extension_same_node",
            }
        )
        transition = transition.merge(
            fill, on="pair_id", how="left", validate="one_to_one"
        )
        transition["d_base"] = transition["d_base"].fillna(
            transition["extension_d_base"]
        )
        transition["same_network_node"] = (
            transition["same_network_node"]
            .where(
                transition["same_network_node"].notna(),
                transition["extension_same_node"],
            )
            .astype(bool)
        )
        transition = transition.drop(
            columns=["extension_d_base", "extension_same_node"]
        )
    if transition["d_base"].isna().any():
        raise ValueError("Transition support remains incomplete")
    target = artifact_dir / f"transition_yardstick_{year}.feather"
    target.parent.mkdir(parents=True, exist_ok=True)
    transition.to_feather(target, compression="zstd")
    if not extension.empty:
        extension.to_feather(
            artifact_dir / f"transition_extension_{year}.feather",
            compression="zstd",
        )
    return target, {
        "required_transition_pairs": int(len(usage)),
        "new_transition_city_pairs": int(len(missing)),
        "new_transition_node_routes": int(new_node_routes),
    }


def run_c3(
    year: int,
    config: dict[str, Any],
    c2_manifest: dict[str, Any],
    transition_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    direct_path = CODING_ROOT / config[f"yardstick_{year}"]
    contract_hash = hashlib.sha256(
        (
            sha256_file(transition_path)
            + sha256_file(direct_path)
            + json.dumps(config, sort_keys=True)
        ).encode()
    ).hexdigest()
    tasks = []
    for item in c2_manifest["parts"]:
        task_id = int(item["task_id"])
        source = CODING_ROOT / item["retained_path"]
        route = artifact_dir / f"route_decisions_{year}/part-{task_id:03d}.feather"
        final = artifact_dir / f"c3_decisions_{year}/part-{task_id:03d}.feather"
        retained = artifact_dir / f"c3_retained_{year}/part-{task_id:03d}.feather"
        sidecar = artifact_dir / f"c3_sidecars_{year}/part-{task_id:03d}.json"
        tasks.append(
            (
                task_id,
                str(source),
                str(route),
                str(final),
                str(retained),
                str(sidecar),
                contract_hash,
                item["retained_sha256"],
            )
        )
    results = []
    with ProcessPoolExecutor(
        max_workers=int(config["worker_count"]),
        mp_context=mp.get_context("spawn"),
        initializer=c3.initialize_worker,
        initargs=(str(transition_path), str(direct_path), config),
    ) as executor:
        futures = {
            executor.submit(c3.process_part, *task): task[0]
            for task in tasks
            if not c3.part_is_complete(
                Path(task[5]),
                Path(task[2]),
                Path(task[3]),
                Path(task[4]),
                task[6],
                task[7],
            )
        }
        for future in as_completed(futures):
            results.append(future.result())
    if not results:
        results = [load_json(Path(task[5])) for task in tasks]
    else:
        results = [load_json(Path(task[5])) for task in tasks]
    totals = Counter()
    for item in results:
        totals.update(
            {
                "c2_retained": item["input_candidates"],
                "route_excluded": item["route_circuity_exclusions"],
                "route_retained": item["route_retained_candidates"],
                "duration_speed_excluded": (
                    item["positive_route_slow_long_exclusions"]
                    + item["zero_route_long_exclusions"]
                ),
                "same_node_excluded": item["same_node_backstop_exclusions"],
                "c3_retained": item["final_retained_candidates"],
            }
        )
    if totals["c2_retained"] != (totals["route_excluded"] + totals["route_retained"]):
        raise ValueError(f"{year} route attrition does not balance")
    if totals["c3_retained"] != (
        totals["route_retained"]
        - totals["duration_speed_excluded"]
        - totals["same_node_excluded"]
    ):
        raise ValueError(f"{year} post-route C3 attrition does not balance")
    manifest = {
        "status": "pass",
        "analysis_year": year,
        "candidate_enumeration_mode": config["candidate_enumeration_mode"],
        "parts": results,
        "summary": {key: int(value) for key, value in totals.items()},
    }
    write_json(artifact_dir / f"c3_30_day_manifest_{year}.json", manifest)
    return manifest


def load_exposure(config: dict[str, Any]) -> pd.DataFrame:
    exposure = pd.read_feather(
        CODING_ROOT / config["p4_city_shiptype_exposure"],
        columns=["city_pair", "shiptype_group", "BaselineSuezDependence_g"],
    ).rename(columns={"city_pair": "pair_id"})
    if exposure.duplicated(["pair_id", "shiptype_group"]).any():
        raise ValueError("Route exposure duplicates pair by ship type")
    return exposure


def prepare_c4_part(source: Path, target: Path, yardstick_path: Path) -> Path:
    """Apply the unchanged C4 interface to one independently written source part."""
    yardstick = pd.read_feather(yardstick_path, columns=["pair_id", "d_yardstick"])
    frame = pd.read_feather(source)
    frame["shiptype_group"] = frame["detailed_vessel_type_normalized"].map(c4.map_shiptype_group)
    if frame["shiptype_group"].isna().any():
        raise ValueError("An analytical ship type is unmapped")
    frame = frame.merge(yardstick, on="pair_id", how="left", validate="many_to_one")
    if frame["d_yardstick"].isna().any():
        raise ValueError("A C3-retained candidate lacks a yardstick")
    frame["total_voyage_time_days"] = frame["total_hours"] / 24.0
    frame["departure_month"] = pd.to_datetime(frame["departure_time"], utc=True).dt.strftime("%Y-%m")
    frame["temporal_scope_eligible"] = frame["r3_window_eligible"].astype(bool)
    ratio = np.full(len(frame), np.nan, dtype=float)
    np.divide(frame["total_voyage_time_days"].to_numpy(float), frame["d_yardstick"].to_numpy(float),
              out=ratio, where=frame["d_yardstick"].to_numpy(float) > 0)
    frame["normalized_time_ratio"] = ratio
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_feather(target, compression="zstd", compression_level=6)
    return target


def prepare_c4_parts(
    year: int,
    config: dict[str, Any],
    artifact_dir: Path,
) -> list[Path]:
    sources = sorted((artifact_dir / f"c3_retained_{year}").glob("part-*.feather"))
    expected = int(config.get("candidate_part_count", 16))
    if expected < 1 or [p.name for p in sources] != [f"part-{i:03d}.feather" for i in range(expected)]:
        raise ValueError(f"Expected {expected} contiguous C4 source parts for {year}")
    yardstick_path = CODING_ROOT / config[f"yardstick_{year}"]
    outputs = [artifact_dir / f"c4_pretrim_{year}" / source.name for source in sources]
    workers = min(int(config.get("worker_count", 1)), expected)
    if workers < 1:
        raise ValueError("C4 worker_count must be positive")
    if workers == 1:
        for source, target in zip(sources, outputs):
            prepare_c4_part(source, target, yardstick_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(prepare_c4_part, source, target, yardstick_path): source
                       for source, target in zip(sources, outputs)}
            failures = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures.append((str(futures[future]), repr(error)))
            if failures:
                raise RuntimeError(f"C4 preparation failures: {failures}")
    return outputs


def fit_2021_thresholds(
    paths: list[Path], config: dict[str, Any], artifact_dir: Path
) -> pd.DataFrame:
    compact = pd.concat(
        [
            pd.read_feather(
                path,
                columns=[
                    "shiptype_group",
                    "normalized_time_ratio",
                    "temporal_scope_eligible",
                ],
            )
            for path in paths
        ],
        ignore_index=True,
    )
    eligible = compact.loc[
        compact["temporal_scope_eligible"]
        & np.isfinite(compact["normalized_time_ratio"])
    ]
    thresholds = (
        eligible.groupby("shiptype_group", observed=True)["normalized_time_ratio"]
        .quantile([config["q_lower"], config["q_upper"]])
        .unstack()
        .rename(
            columns={
                config["q_lower"]: "normalized_time_ratio_q01",
                config["q_upper"]: "normalized_time_ratio_q99",
            }
        )
        .reset_index()
    )
    counts = (
        eligible.groupby("shiptype_group", observed=True)
        .size()
        .rename("quantile_eligible_rows")
        .reset_index()
    )
    thresholds = thresholds.merge(counts, on="shiptype_group", validate="one_to_one")
    thresholds.to_csv(artifact_dir / "q01_q99_thresholds_2021.csv", index=False)
    return thresholds


def apply_c4(
    year: int,
    pretrim_paths: list[Path],
    thresholds: pd.DataFrame,
    exposure: pd.DataFrame,
    config: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    totals = Counter()
    outputs = []
    for source in pretrim_paths:
        frame = pd.read_feather(source).merge(
            thresholds[
                [
                    "shiptype_group",
                    "normalized_time_ratio_q01",
                    "normalized_time_ratio_q99",
                ]
            ],
            on="shiptype_group",
            how="left",
            validate="many_to_one",
        )
        eligible = (
            frame["temporal_scope_eligible"]
            & np.isfinite(frame["normalized_time_ratio"])
            & frame["d_yardstick"].gt(0)
        )
        frame["trimmed_below_q01"] = eligible & frame["normalized_time_ratio"].lt(
            frame["normalized_time_ratio_q01"]
        )
        frame["trimmed_above_q99"] = eligible & frame["normalized_time_ratio"].gt(
            frame["normalized_time_ratio_q99"]
        )
        frame["trimmed"] = frame["trimmed_below_q01"] | frame["trimmed_above_q99"]
        frame["retain_after_c4"] = frame["temporal_scope_eligible"] & ~frame["trimmed"]
        frame = frame.merge(
            exposure,
            on=["pair_id", "shiptype_group"],
            how="left",
            validate="many_to_one",
        )
        frame["suez_dependence_bin"] = (
            assign_suez_dependence_bin(frame["BaselineSuezDependence_g"])
            .astype("string")
            .fillna(config["missing_exposure_label"])
        )
        retained = frame.loc[frame["retain_after_c4"]].reset_index(drop=True)
        target = artifact_dir / f"final_sample_{year}" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        retained.to_feather(target, compression="zstd", compression_level=6)
        outputs.append(target)
        totals.update(
            {
                "c3_retained": len(frame),
                "temporal_scope_excluded": int(
                    (~frame["temporal_scope_eligible"]).sum()
                ),
                "q01_trimmed": int(frame["trimmed_below_q01"].sum()),
                "q99_trimmed": int(frame["trimmed_above_q99"].sum()),
                "final_retained": len(retained),
                "exposure_supported": int(
                    retained["BaselineSuezDependence_g"].notna().sum()
                ),
                "positive_exposure": int(
                    retained["BaselineSuezDependence_g"].gt(0).sum()
                ),
                "high_very_exposure": int(
                    retained["suez_dependence_bin"].isin(["high", "very_high"]).sum()
                ),
            }
        )
    if totals["final_retained"] != (
        totals["c3_retained"]
        - totals["temporal_scope_excluded"]
        - totals["q01_trimmed"]
        - totals["q99_trimmed"]
    ):
        raise ValueError(f"{year} C4 attrition does not balance")
    manifest = {
        "status": "pass",
        "analysis_year": year,
        "parts": [
            {
                "path": str(path.relative_to(CODING_ROOT)),
                "rows": int(len(pd.read_feather(path, columns=["candidate_id"]))),
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
        "summary": {key: int(value) for key, value in totals.items()},
    }
    write_json(artifact_dir / f"final_sample_manifest_{year}.json", manifest)
    return manifest


def load_final_sample(year: int, artifact_dir: Path) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "imo",
        "pair_id",
        "shiptype_group",
        "departure_month",
        "total_voyage_time_days",
        "BaselineSuezDependence_g",
        "suez_dependence_bin",
    ]
    return pd.concat(
        [
            pd.read_feather(path, columns=columns)
            for path in sorted(
                (artifact_dir / f"final_sample_{year}").glob("part-*.feather")
            )
        ],
        ignore_index=True,
    )


def build_analysis_reports(
    config: dict[str, Any],
    c2_manifests: dict[int, dict[str, Any]],
    c3_manifests: dict[int, dict[str, Any]],
    c4_manifests: dict[int, dict[str, Any]],
    transition_summaries: dict[int, dict[str, int]],
    artifact_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    samples = {year: load_final_sample(year, artifact_dir) for year in (2021, 2025)}
    for year, frame in samples.items():
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{year} final sample contains duplicate candidate IDs")
        expected = c4_manifests[year]["summary"]["final_retained"]
        if len(frame) != expected:
            raise ValueError(f"{year} final sample row count does not match manifest")
    attrition_rows = []
    for year in (2021, 2025):
        c2_summary = c2_manifests[year]["summary"]
        c3_summary = c3_manifests[year]["summary"]
        c4_summary = c4_manifests[year]["summary"]
        attrition_rows.extend(
            [
                {
                    "year": year,
                    "stage": "c2_input",
                    "candidate_rows": c2_summary["input_candidates"],
                },
                {
                    "year": year,
                    "stage": "c2_30_day_retained",
                    "candidate_rows": c2_summary["retained_candidates"],
                },
                {
                    "year": year,
                    "stage": "c3_route_retained",
                    "candidate_rows": c3_summary["route_retained"],
                },
                {
                    "year": year,
                    "stage": "c3_duration_speed_retained",
                    "candidate_rows": c3_summary["route_retained"]
                    - c3_summary["duration_speed_excluded"],
                },
                {
                    "year": year,
                    "stage": "c3_final_retained",
                    "candidate_rows": c3_summary["c3_retained"],
                },
                {
                    "year": year,
                    "stage": "c4_temporal_retained",
                    "candidate_rows": c4_summary["c3_retained"]
                    - c4_summary["temporal_scope_excluded"],
                },
                {
                    "year": year,
                    "stage": "c4_final_retained",
                    "candidate_rows": c4_summary["final_retained"],
                },
            ]
        )
    attrition = pd.DataFrame(attrition_rows)
    attrition.to_csv(report_dir / "stage_attrition.csv", index=False)

    prior_review_medians = {
        ("Antwerp-Shanghai", 2021): 40.3,
        ("Antwerp-Shanghai", 2025): 48.0,
        ("Port Said-Fujairah", 2021): 21.0,
        ("Port Said-Fujairah", 2025): 18.3,
        ("Port Said-Ningbo", 2021): 31.1,
        ("Port Said-Ningbo", 2025): 43.7,
        ("Port Said-Shanghai", 2021): 39.5,
        ("Port Said-Shanghai", 2025): 57.9,
        ("Port Said-Singapore", 2021): 19.8,
        ("Port Said-Singapore", 2025): 23.2,
        ("Suez Port-Singapore", 2021): 41.3,
        ("Suez Port-Singapore", 2025): 54.2,
        ("Algeciras-Tanjung Pelepas", 2021): 37.3,
        ("Algeciras-Tanjung Pelepas", 2025): 39.4,
        ("Tanger Med-Ningbo", 2021): 34.8,
        ("Tanger Med-Ningbo", 2025): 51.9,
        ("Tanger Med-Salalah", 2021): 14.3,
        ("Tanger Med-Salalah", 2025): 23.1,
        ("Rotterdam-Port Said", 2021): 12.9,
        ("Rotterdam-Port Said", 2025): 19.8,
        ("Jeddah-Jebel Ali", 2021): 10.5,
        ("Jeddah-Jebel Ali", 2025): 12.4,
        ("Jeddah-Ningbo", 2021): 26.7,
        ("Jeddah-Ningbo", 2025): 27.9,
        ("Jeddah-Mundra", 2021): 12.6,
        ("Jeddah-Mundra", 2025): 11.8,
        ("New York-Colombo", 2021): 49.8,
        ("New York-Colombo", 2025): 56.6,
        ("New York-Tanjung Pelepas", 2021): 35.8,
        ("New York-Tanjung Pelepas", 2025): 40.2,
    }
    key_rows = []
    for pair_id, shiptype_group, label in config["key_city_pairs"]:
        for year, frame in samples.items():
            selected = frame.loc[
                frame["pair_id"].eq(pair_id)
                & frame["shiptype_group"].eq(shiptype_group)
            ]
            median = (
                float(selected["total_voyage_time_days"].median())
                if len(selected)
                else np.nan
            )
            prior_median = prior_review_medians[(label, year)]
            key_rows.append(
                {
                    "city_pair": label,
                    "pair_id": pair_id,
                    "shiptype_group": shiptype_group,
                    "year": year,
                    "voyage_count": int(len(selected)),
                    "vessel_count": int(selected["imo"].nunique()),
                    "median_voyage_days": median,
                    "prior_review_median_days": prior_median,
                    "change_from_prior_review_days": median - prior_median,
                    "p25_voyage_days": (
                        float(selected["total_voyage_time_days"].quantile(0.25))
                        if len(selected)
                        else np.nan
                    ),
                    "p75_voyage_days": (
                        float(selected["total_voyage_time_days"].quantile(0.75))
                        if len(selected)
                        else np.nan
                    ),
                    "manual_finding": (
                        "plausible_stable"
                        if len(selected) and abs(median - prior_median) <= 5.0
                        else "requires_follow_up"
                    ),
                }
            )
    key_times = pd.DataFrame(key_rows)
    key_times.to_csv(report_dir / "key_city_pair_voyage_times.csv", index=False)

    panel_frames = []
    for year, frame in samples.items():
        supported = frame.loc[frame["BaselineSuezDependence_g"].notna()]
        panel = (
            supported.groupby(
                ["pair_id", "shiptype_group", "departure_month"],
                observed=True,
            )
            .agg(
                voyage_count=("candidate_id", "size"),
                vessel_count=("imo", "nunique"),
                median_voyage_days=("total_voyage_time_days", "median"),
                BaselineSuezDependence_g=("BaselineSuezDependence_g", "first"),
                suez_dependence_bin=("suez_dependence_bin", "first"),
            )
            .reset_index()
        )
        panel["year"] = year
        panel_frames.append(panel)
    panel = pd.concat(panel_frames, ignore_index=True)
    if panel.duplicated(["pair_id", "shiptype_group", "departure_month", "year"]).any():
        raise ValueError("Supported monthly panel contains duplicate keys")
    expected_supported = sum(
        c4_manifests[year]["summary"]["exposure_supported"] for year in (2021, 2025)
    )
    if int(panel["voyage_count"].sum()) != expected_supported:
        raise ValueError("Supported monthly panel voyage counts do not balance")
    panel.to_feather(artifact_dir / "supported_monthly_panel.feather")
    groups_by_year = panel.groupby(["pair_id", "shiptype_group"])["year"].nunique()
    common_index = groups_by_year.loc[groups_by_year.eq(2)].index
    panel_index = pd.MultiIndex.from_frame(panel[["pair_id", "shiptype_group"]])
    common_panel = panel.loc[panel_index.isin(common_index)].copy()
    month_counts = (
        common_panel.groupby(["pair_id", "shiptype_group", "year"])
        .size()
        .unstack("year", fill_value=0)
    )
    at_least_six = month_counts.loc[
        month_counts.get(2021, 0).ge(6) & month_counts.get(2025, 0).ge(6)
    ].index
    balanced = month_counts.loc[
        month_counts.get(2021, 0).eq(12) & month_counts.get(2025, 0).eq(12)
    ].index
    high_level = []
    for label, index in (
        ("common_groups", common_index),
        ("at_least_6_months_each_year", at_least_six),
        ("balanced_12_months_each_year", balanced),
    ):
        mask = panel_index.isin(index)
        selected = panel.loc[mask]
        high_level.append(
            {
                "sample": label,
                "pair_shiptype_groups": int(len(index)),
                "monthly_observations": int(len(selected)),
                "monthly_observations_ge_2_voyages": int(
                    selected["voyage_count"].ge(2).sum()
                ),
                "monthly_observations_ge_2_vessels": int(
                    selected["vessel_count"].ge(2).sum()
                ),
            }
        )
    group_exposure = (
        common_panel.groupby(["pair_id", "shiptype_group"], observed=True)
        .agg(
            BaselineSuezDependence_g=("BaselineSuezDependence_g", "first"),
            suez_dependence_bin=("suez_dependence_bin", "first"),
        )
        .reset_index()
    )
    regression_summary = pd.DataFrame(high_level)
    regression_summary.to_csv(report_dir / "regression_panel_summary.csv", index=False)
    final_summary_rows = []
    for year in (2021, 2025):
        final_summary_rows.append({"year": year, **c4_manifests[year]["summary"]})
    pd.DataFrame(final_summary_rows).to_csv(
        report_dir / "final_sample_summary.csv", index=False
    )
    summary = {
        "status": "pass",
        "rule_status": "sensitivity_full_rebuild",
        "final_sample": {
            str(year): c4_manifests[year]["summary"] for year in (2021, 2025)
        },
        "transition_extension": {
            str(year): transition_summaries[year] for year in (2021, 2025)
        },
        "common_supported_pair_shiptype_groups": int(len(common_index)),
        "common_positive_exposure_groups": int(
            group_exposure["BaselineSuezDependence_g"].gt(0).sum()
        ),
        "common_high_very_exposure_groups": int(
            group_exposure["suez_dependence_bin"].isin(["high", "very_high"]).sum()
        ),
        "regression_panel": high_level,
        "key_city_pair_cells_with_both_years": int(
            key_times.groupby("city_pair")["voyage_count"]
            .apply(lambda values: bool((values > 0).all()))
            .sum()
        ),
        "key_city_pair_rows_plausible_stable": int(
            key_times["manual_finding"].eq("plausible_stable").sum()
        ),
        "maximum_absolute_key_median_change_days": float(
            key_times["change_from_prior_review_days"].abs().max()
        ),
    }
    if summary["key_city_pair_cells_with_both_years"] != len(config["key_city_pairs"]):
        raise ValueError("At least one key city-pair cell lost endpoint-year support")
    if summary["key_city_pair_rows_plausible_stable"] != len(key_times):
        raise ValueError("At least one key city-pair median requires follow-up")
    write_json(report_dir / "summary.json", summary)
    lines = [
        "# C2 30-day full sample rebuild",
        "",
        "This sensitivity changes only C2. All downstream C3 rules remain fixed,",
        "and the 2021 q01/q99 definition is re-estimated on the rebuilt 2021",
        "population before being applied unchanged to 2025.",
        "",
        "## Final samples",
        "",
        "| Year | Final voyages | Exposure-supported | Positive Suez | High/very high |",
        "|---:|---:|---:|---:|---:|",
    ]
    for year in (2021, 2025):
        item = c4_manifests[year]["summary"]
        lines.append(
            f"| {year} | {item['final_retained']:,} | "
            f"{item['exposure_supported']:,} | {item['positive_exposure']:,} | "
            f"{item['high_very_exposure']:,} |"
        )
    lines.extend(
        [
            "",
            "## Regression panel",
            "",
            f"- Common supported pair-by-ship-type groups: {len(common_index):,}.",
            f"- Common positive-exposure groups: "
            f"{summary['common_positive_exposure_groups']:,}.",
            f"- Common high/very-high groups: "
            f"{summary['common_high_very_exposure_groups']:,}.",
            f"- Common-group monthly observations: "
            f"{high_level[0]['monthly_observations']:,}.",
            f"- Common-group observations with at least two vessels: "
            f"{high_level[0]['monthly_observations_ge_2_vessels']:,}.",
            "",
            "## Key city pairs",
            "",
            f"- Cells with observations in both endpoint years: "
            f"{summary['key_city_pair_cells_with_both_years']}/"
            f"{len(config['key_city_pairs'])}.",
            f"- All {summary['key_city_pair_rows_plausible_stable']}/"
            f"{len(key_times)} year-specific medians remain plausible and "
            "within five days of the prior reviewed values.",
            f"- Maximum absolute median change: "
            f"{summary['maximum_absolute_key_median_change_days']:.1f} days.",
            "- Detailed medians and interquartile ranges are in",
            "  `key_city_pair_voyage_times.csv`.",
            "",
            "The q01 gate remains methodologically red from the AIS review. The",
            "with-qtrim final count is reported for comparability, not as an",
            "automatic approval of that gate.",
            "",
        ]
    )
    (report_dir / "full_rebuild_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_dir = args.artifact_dir.resolve()
    report_dir = args.report_dir.resolve()
    config = load_json(config_path)
    verify_contract(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(config_path)
    c2_manifests = {
        year: materialize_c2(year, config, artifact_dir, config_hash)
        for year in config["analysis_years"]
    }
    transition_paths = {}
    transition_summaries = {}
    for year in config["analysis_years"]:
        transition_paths[year], transition_summaries[year] = build_transition_support(
            year, config, c2_manifests[year], artifact_dir
        )
    c3_manifests = {
        year: run_c3(
            year,
            config,
            c2_manifests[year],
            transition_paths[year],
            artifact_dir,
        )
        for year in config["analysis_years"]
    }
    pretrim = {
        year: prepare_c4_parts(year, config, artifact_dir)
        for year in config["analysis_years"]
    }
    thresholds = fit_2021_thresholds(pretrim[2021], config, artifact_dir)
    exposure = load_exposure(config)
    c4_manifests = {
        year: apply_c4(
            year,
            pretrim[year],
            thresholds,
            exposure,
            config,
            artifact_dir,
        )
        for year in config["analysis_years"]
    }
    summary = build_analysis_reports(
        config,
        c2_manifests,
        c3_manifests,
        c4_manifests,
        transition_summaries,
        artifact_dir,
        report_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
