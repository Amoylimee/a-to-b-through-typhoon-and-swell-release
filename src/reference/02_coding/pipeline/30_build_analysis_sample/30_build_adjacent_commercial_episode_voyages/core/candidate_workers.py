"""Shared parallel workers for adjacent-commercial-episode candidates.

The numbered stage entry points own execution and output selection. Historical
2021 fan-out inputs named below are frozen upstream support only; they do not
select the active enumeration rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_core import DataContractError, enumerate_vessel_candidates
from .candidate_pairs import (
    ADJACENT_COMMERCIAL_EPISODES,
    validate_candidate_enumeration_mode,
)


SCRIPT_PATH = Path(__file__).resolve()
C1_ROOT = SCRIPT_PATH.parents[1]
CODING_ROOT = SCRIPT_PATH.parents[4]
DATA_ROOT = CODING_ROOT / "data/processed/build_adjacent_commercial_episode_voyages"

DEFAULT_CONFIG = C1_ROOT / "config/frozen/upstream_2021_enumeration_inputs.json"
DEFAULT_ARTIFACT_DIR = DATA_ROOT / "main/provenance/adjacent_episode_2021"
DEFAULT_REPORT_DIR = DATA_ROOT / "main/validation/adjacent_episode_2021"
DEFAULT_RUN_DIR = (
    CODING_ROOT
    / ".runs/build_adjacent_commercial_episode_voyages/"
    "adjacent_episode_2021"
)

EPISODE_COLUMNS = [
    "mmsi",
    "episode_sequence",
    "imo",
    "vessel_uid_clean",
    "vessel_name",
    "detailed_vessel_type_normalized",
    "ship_type_sample",
    "topology_city_id",
    "passage_arrival_time",
    "passage_departure_time",
    "episode_id",
    "first_call_id",
    "last_call_id",
]

EVENT_COLUMNS = ["mmsi", "trip_id", "event_type", "event_time", "call_id"]
CALL_COLUMNS = ["call_id", "topology_city_id", "episode_id"]

_EPISODES: pd.DataFrame | None = None
_LEGS: pd.DataFrame | None = None
_EPISODE_SLICES: dict[str, tuple[int, int]] = {}
_LEG_SLICES: dict[str, tuple[int, int]] = {}
_CALL_META: dict[str, dict[str, str]] = {}
_WINDOWS: list[dict[str, Any]] = []
_LINER_TYPES: set[str] = set()
_TRAMP_TYPES: set[str] = set()
_CONTRACT_HASH = ""
_CANDIDATE_ENUMERATION_MODE = ADJACENT_COMMERCIAL_EPISODES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_inputs(config: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for relative, expected_hash in config["input_sha256"].items():
        path = CODING_ROOT / relative
        if not path.is_file():
            raise DataContractError(f"Missing frozen input: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise DataContractError(
                f"Frozen input hash mismatch for {relative}: {actual_hash} != {expected_hash}"
            )
        manifest[relative] = {
            "sha256": actual_hash,
            "size_bytes": path.stat().st_size,
        }
    return manifest


def input_path(config: dict[str, Any], suffix: str) -> Path:
    matches = [
        CODING_ROOT / relative
        for relative in config["input_sha256"]
        if relative.endswith(suffix)
    ]
    if len(matches) != 1:
        raise DataContractError(f"Expected one config input ending with {suffix}: {matches}")
    return matches[0]


def build_slices(frame: pd.DataFrame, key: str) -> dict[str, tuple[int, int]]:
    values = frame[key].astype(str).to_numpy()
    if len(values) == 0:
        return {}
    change = np.concatenate(([True], values[1:] != values[:-1], [True]))
    bounds = np.flatnonzero(change)
    return {
        str(values[start]): (int(start), int(stop))
        for start, stop in zip(bounds[:-1], bounds[1:])
    }


def load_runtime_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    episodes = pd.read_feather(
        input_path(config, "ordered_city_episodes_2021.feather"),
        columns=EPISODE_COLUMNS,
    ).sort_values(["mmsi", "episode_sequence"], kind="mergesort").reset_index(drop=True)
    if len(episodes) != int(config["expected_city_episodes"]):
        raise DataContractError(
            f"Expected {config['expected_city_episodes']} episodes, found {len(episodes)}"
        )
    if episodes["imo"].nunique() != int(config["expected_analysis_vessels"]):
        raise DataContractError("C1 input does not retain all 20,319 analysis vessels")

    events = pd.read_feather(
        input_path(config, "endpoint_events_2021.feather"),
        columns=EVENT_COLUMNS,
    )
    departures = events.loc[
        events["event_type"].eq("departure"),
        ["mmsi", "trip_id", "event_time", "call_id"],
    ].rename(columns={"event_time": "departure_time", "call_id": "departure_call_id"})
    arrivals = events.loc[
        events["event_type"].eq("arrival"),
        ["mmsi", "trip_id", "event_time", "call_id"],
    ].rename(columns={"event_time": "arrival_time", "call_id": "arrival_call_id"})
    if departures["trip_id"].duplicated().any() or arrivals["trip_id"].duplicated().any():
        raise DataContractError("Endpoint event input has a duplicated trip boundary")
    legs = departures.merge(
        arrivals,
        on=["mmsi", "trip_id"],
        how="inner",
        validate="one_to_one",
    ).sort_values(
        ["mmsi", "departure_time", "arrival_time", "trip_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(legs) != len(departures) or len(legs) != len(arrivals):
        raise DataContractError("A retained OD leg is missing an arrival or departure boundary")
    if (legs["arrival_time"] <= legs["departure_time"]).any():
        raise DataContractError("Retained OD legs contain non-positive durations")
    legs["leg_sequence"] = legs.groupby("mmsi", sort=False).cumcount().astype(np.int64)

    calls = pd.read_feather(
        input_path(config, "ordered_port_calls_2021.feather"),
        columns=CALL_COLUMNS,
    )
    if calls["call_id"].duplicated().any():
        raise DataContractError("Ordered call input has duplicate call IDs")
    call_meta = {
        str(call_id): {
            "topology_city_id": str(city or ""),
            "episode_id": str(episode or ""),
        }
        for call_id, city, episode in calls.itertuples(index=False, name=None)
    }
    return episodes, legs, call_meta


def parse_windows(config: dict[str, Any]) -> list[dict[str, Any]]:
    windows = []
    for item in config["period_windows"]:
        windows.append(
            {
                "period_id": str(item["period_id"]),
                "start": pd.Timestamp(item["start"]),
                "end": pd.Timestamp(item["end"]),
            }
        )
    for left, right in zip(windows, windows[1:]):
        if left["end"] > right["start"]:
            raise DataContractError("Configured recurrence windows overlap")
    return windows


def partition_vessels(episodes: pd.DataFrame, task_count: int) -> list[list[str]]:
    sizes = episodes.groupby("mmsi", sort=True).size()
    weighted = sorted(
        ((str(mmsi), int(size) ** 2) for mmsi, size in sizes.items()),
        key=lambda item: (-item[1], item[0]),
    )
    tasks: list[list[str]] = [[] for _ in range(task_count)]
    loads = [0] * task_count
    for mmsi, weight in weighted:
        target = min(range(task_count), key=lambda index: (loads[index], index))
        tasks[target].append(mmsi)
        loads[target] += weight
    for task in tasks:
        task.sort()
    return tasks


def initialize_globals(
    episodes: pd.DataFrame,
    legs: pd.DataFrame,
    call_meta: dict[str, dict[str, str]],
    config: dict[str, Any],
    contract_hash: str,
) -> None:
    global _EPISODES, _LEGS, _EPISODE_SLICES, _LEG_SLICES
    global _CALL_META, _WINDOWS, _LINER_TYPES, _TRAMP_TYPES, _CONTRACT_HASH
    global _CANDIDATE_ENUMERATION_MODE
    _EPISODES = episodes
    _LEGS = legs
    _EPISODE_SLICES = build_slices(episodes, "mmsi")
    _LEG_SLICES = build_slices(legs, "mmsi")
    _CALL_META = call_meta
    _WINDOWS = parse_windows(config)
    _LINER_TYPES = set(config["liner_proxy_types"])
    _TRAMP_TYPES = set(config["tramp_proxy_types"])
    _CANDIDATE_ENUMERATION_MODE = validate_candidate_enumeration_mode(
        config.get("candidate_enumeration_mode", ADJACENT_COMMERCIAL_EPISODES)
    )
    _CONTRACT_HASH = contract_hash


def part_is_complete(candidate_path: Path, stop_path: Path, sidecar_path: Path, contract_hash: str) -> bool:
    if not candidate_path.is_file() or not stop_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("contract_hash") == contract_hash
        and payload.get("candidate_sha256") == sha256_file(candidate_path)
        and payload.get("raw_stop_sha256") == sha256_file(stop_path)
    )


def process_task(
    task_id: int,
    mmsi_values: list[str],
    candidate_path_text: str,
    stop_path_text: str,
    sidecar_path_text: str,
) -> dict[str, Any]:
    if _EPISODES is None or _LEGS is None:
        raise RuntimeError("C1 worker globals were not initialized")
    candidate_frames: list[pd.DataFrame] = []
    stop_frames: list[pd.DataFrame] = []
    for mmsi in mmsi_values:
        episode_bounds = _EPISODE_SLICES[mmsi]
        leg_bounds = _LEG_SLICES.get(mmsi)
        if leg_bounds is None:
            continue
        vessel_candidates, vessel_stops = enumerate_vessel_candidates(
            _EPISODES.iloc[episode_bounds[0] : episode_bounds[1]],
            _LEGS.iloc[leg_bounds[0] : leg_bounds[1]],
            _CALL_META,
            _WINDOWS,
            _LINER_TYPES,
            _TRAMP_TYPES,
            _CANDIDATE_ENUMERATION_MODE,
        )
        if not vessel_candidates.empty:
            candidate_frames.append(vessel_candidates)
        if not vessel_stops.empty:
            stop_frames.append(vessel_stops)
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    stops = pd.concat(stop_frames, ignore_index=True) if stop_frames else pd.DataFrame()
    if not candidates.empty and candidates["candidate_id"].duplicated().any():
        raise DataContractError(f"Task {task_id} generated duplicate candidate IDs")
    if not stops.empty and stops["stop_id"].duplicated().any():
        raise DataContractError(f"Task {task_id} generated duplicate stop IDs")

    candidate_path = Path(candidate_path_text)
    stop_path = Path(stop_path_text)
    sidecar_path = Path(sidecar_path_text)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_feather(candidate_path, compression="zstd", compression_level=6)
    stops.to_feather(stop_path, compression="zstd", compression_level=6)
    payload = {
        "task_id": task_id,
        "contract_hash": _CONTRACT_HASH,
        "mmsi_count": len(mmsi_values),
        "candidate_rows": int(len(candidates)),
        "raw_stop_rows": int(len(stops)),
        "candidate_sha256": sha256_file(candidate_path),
        "raw_stop_sha256": sha256_file(stop_path),
    }
    write_json(sidecar_path, payload)
    return payload


def run_tasks(
    config: dict[str, Any],
    artifact_dir: Path,
    run_dir: Path,
    tasks: list[list[str]],
    contract_hash: str,
) -> tuple[list[dict[str, Any]], int]:
    candidate_dir = artifact_dir / "candidate_passages_2021"
    stop_dir = artifact_dir / "raw_stationary_episodes_2021"
    sidecar_dir = run_dir / "parts"
    for directory in (candidate_dir, stop_dir, sidecar_dir):
        directory.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, Any]] = []
    pending: list[tuple[int, list[str], Path, Path, Path]] = []
    for task_id, mmsi_values in enumerate(tasks):
        candidate_path = candidate_dir / f"part-{task_id:03d}.feather"
        stop_path = stop_dir / f"part-{task_id:03d}.feather"
        sidecar_path = sidecar_dir / f"part-{task_id:03d}.json"
        if part_is_complete(candidate_path, stop_path, sidecar_path, contract_hash):
            completed.append(json.loads(sidecar_path.read_text(encoding="utf-8")))
        else:
            pending.append((task_id, mmsi_values, candidate_path, stop_path, sidecar_path))

    if pending:
        workers = min(int(config["worker_count"]), len(pending), 16)
        print(f"Running {len(pending)} C1 tasks with {workers} forked workers", flush=True)
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            future_map = {
                executor.submit(
                    process_task,
                    task_id,
                    mmsi_values,
                    str(candidate_path),
                    str(stop_path),
                    str(sidecar_path),
                ): task_id
                for task_id, mmsi_values, candidate_path, stop_path, sidecar_path in pending
            }
            for future in as_completed(future_map):
                payload = future.result()
                completed.append(payload)
                print(
                    f"Task {payload['task_id']:03d}: {payload['candidate_rows']:,} candidates, "
                    f"{payload['raw_stop_rows']:,} raw stops",
                    flush=True,
                )
    return sorted(completed, key=lambda item: item["task_id"]), len(pending)


def aggregate_group_counts(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    result = combined.groupby(keys, sort=True, dropna=False)[["candidate_rows", "r2_flagged"]].sum().reset_index()
    result["r2_rate"] = result["r2_flagged"] / result["candidate_rows"]
    return result


def detect_histogram_valley(histogram: pd.DataFrame, minimum_days: float, maximum_days: float) -> dict[str, Any]:
    counts = histogram["unique_stop_count"].to_numpy(dtype=float)
    if len(counts) < 7 or counts.max(initial=0.0) <= 0:
        return {"stable_valley_found": False, "reason": "insufficient_histogram"}
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    centers = histogram["bin_center_days"].to_numpy(dtype=float)
    peaks = [
        index
        for index in range(1, len(smooth) - 1)
        if smooth[index] > smooth[index - 1]
        and smooth[index] >= smooth[index + 1]
        and smooth[index] >= 0.05 * smooth.max()
    ]
    candidates: list[dict[str, Any]] = []
    for left in peaks:
        for right in peaks:
            if right <= left + 2:
                continue
            valley_slice = smooth[left + 1 : right]
            if len(valley_slice) == 0:
                continue
            valley = left + 1 + int(np.argmin(valley_slice))
            day = float(centers[valley])
            if not minimum_days <= day <= maximum_days:
                continue
            depth_ratio = float(smooth[valley] / min(smooth[left], smooth[right]))
            candidates.append(
                {
                    "left_peak_days": float(centers[left]),
                    "right_peak_days": float(centers[right]),
                    "valley_days": day,
                    "valley_depth_ratio": depth_ratio,
                }
            )
    if not candidates:
        return {"stable_valley_found": False, "reason": "no_two_peak_valley_in_search_range"}
    best = min(candidates, key=lambda item: (item["valley_depth_ratio"], item["valley_days"]))
    best["stable_valley_found"] = bool(best["valley_depth_ratio"] <= 0.8)
    best["reason"] = "two_peak_valley" if best["stable_valley_found"] else "valley_not_deep_enough"
    return best


def write_dwell_plot(
    histogram: pd.DataFrame,
    valley: dict[str, Any],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.plot(
        histogram["bin_center_days"],
        histogram["unique_stop_count"],
        color="#174a7e",
        linewidth=1.7,
        label="Unique candidate-used tramp stops",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Raw stationary duration (days, log scale)")
    axis.set_ylabel("Stop count (log scale)")
    for value, label, color in [
        (1.0, "bunkering anchor: 1 d", "#2ca02c"),
        (2.05, "dry-bulk anchor: 2.05 d", "#d62728"),
        (2.7, "dry-bulk anchor: 2.7 d", "#9467bd"),
    ]:
        axis.axvline(value, color=color, linestyle="--", linewidth=1.1, label=label)
    if valley.get("stable_valley_found"):
        axis.axvline(
            float(valley["valley_days"]),
            color="#ff7f0e",
            linestyle="-.",
            linewidth=1.4,
            label="empirical valley candidate",
        )
    axis.grid(True, which="both", alpha=0.18)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_reports(
    config: dict[str, Any],
    artifact_dir: Path,
    report_dir: Path,
    part_payloads: list[dict[str, Any]],
    contract_hash: str,
    pending_task_count: int,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    candidate_paths = [
        artifact_dir / "candidate_passages_2021" / f"part-{item['task_id']:03d}.feather"
        for item in part_payloads
    ]
    stop_paths = [
        artifact_dir / "raw_stationary_episodes_2021" / f"part-{item['task_id']:03d}.feather"
        for item in part_payloads
    ]

    pair_frames: list[pd.DataFrame] = []
    r2_type_frames: list[pd.DataFrame] = []
    r2_regime_frames: list[pd.DataFrame] = []
    r2_top_frames: list[pd.DataFrame] = []
    r2_distinct_vessel_frames: list[pd.DataFrame] = []
    r2_short_frames: list[pd.DataFrame] = []
    r2_middle_bin_frames: list[pd.DataFrame] = []
    r2_duration_bin_frames: list[pd.DataFrame] = []
    r3_frames: list[pd.DataFrame] = []
    time_failures: list[pd.DataFrame] = []
    time_part_summaries: list[dict[str, Any]] = []
    tolerance = float(config["time_identity_tolerance_hours"])

    for path in candidate_paths:
        pair = pd.read_feather(path, columns=["origin_city", "dest_city", "imo"])
        pair_frames.append(
            pair.groupby(["origin_city", "dest_city"], sort=True)
            .agg(candidate_count_c1=("imo", "size"), vessel_count_c1=("imo", "nunique"))
            .reset_index()
        )
        r2 = pd.read_feather(
            path,
            columns=[
                "candidate_id",
                "imo",
                "vessel_name",
                "detailed_vessel_type_normalized",
                "service_regime_proxy",
                "vessel_family_proxy",
                "origin_city",
                "dest_city",
                "departure_time",
                "arrival_time",
                "total_hours",
                "n_intermediate_city_episodes",
                "intermediate_city_sequence",
                "r2_repeated_intermediate_city",
                "r2_repeated_city_count",
                "r2_repeated_cities",
                "r3_recurrence_count",
                "r3_window_eligible",
                "r3_singleton",
            ],
        )
        for key, destination in [
            ("detailed_vessel_type_normalized", r2_type_frames),
            ("service_regime_proxy", r2_regime_frames),
        ]:
            grouped = (
                r2.groupby(key, sort=True)["r2_repeated_intermediate_city"]
                .agg(candidate_rows="size", r2_flagged="sum")
                .reset_index()
            )
            destination.append(grouped)
        flagged = r2.loc[r2["r2_repeated_intermediate_city"]].copy()
        if not flagged.empty:
            r2_top_frames.append(
                flagged.sort_values(
                    ["r2_repeated_city_count", "n_intermediate_city_episodes", "total_hours", "candidate_id"],
                    ascending=[False, False, False, True],
                ).head(100)
            )
            r2_distinct_vessel_frames.append(
                flagged.sort_values(
                    ["r2_repeated_city_count", "n_intermediate_city_episodes", "total_hours", "candidate_id"],
                    ascending=[False, False, False, True],
                ).drop_duplicates("imo").head(100)
            )
            r2_short_frames.append(
                flagged.sort_values(
                    ["total_hours", "r2_repeated_city_count", "candidate_id"],
                    ascending=[True, False, True],
                ).head(100)
            )
        r2["middle_count_bin"] = pd.cut(
            r2["n_intermediate_city_episodes"],
            bins=[-1, 0, 1, 2, 4, 9, 19, 49, np.inf],
            labels=["0", "1", "2", "3-4", "5-9", "10-19", "20-49", "50+"],
        )
        r2["duration_days_bin"] = pd.cut(
            r2["total_hours"] / 24.0,
            bins=[0, 7, 14, 30, 60, 120, np.inf],
            labels=["0-7", "7-14", "14-30", "30-60", "60-120", "120+"],
            include_lowest=True,
        )
        for key, destination in [
            ("middle_count_bin", r2_middle_bin_frames),
            ("duration_days_bin", r2_duration_bin_frames),
        ]:
            destination.append(
                r2.groupby(key, sort=True, observed=True)["r2_repeated_intermediate_city"]
                .agg(candidate_rows="size", r2_flagged="sum")
                .reset_index()
            )
        r3_frames.append(
            r2.groupby(
                ["service_regime_proxy", "r3_window_eligible", "r3_recurrence_count", "r3_singleton"],
                sort=True,
            )
            .size()
            .rename("candidate_rows")
            .reset_index()
        )

        timing = pd.read_feather(
            path,
            columns=[
                "candidate_id",
                "imo",
                "origin_city",
                "dest_city",
                "total_hours",
                "observed_sea_hours",
                "intermediate_gap_hours",
                "time_identity_residual_hours",
                "n_intermediate_raw_stops",
                "n_strict_unresolved_raw_stops",
                "n_blank_topology_raw_stops",
                "n_conflicting_city_raw_stops",
            ],
        )
        abs_residual = timing["time_identity_residual_hours"].abs()
        failures = timing.loc[abs_residual > tolerance]
        if not failures.empty:
            time_failures.append(failures)
        time_part_summaries.append(
            {
                "candidate_rows": int(len(timing)),
                "failure_rows": int((abs_residual > tolerance).sum()),
                "max_abs_residual_hours": float(abs_residual.max() if len(timing) else 0.0),
                "candidates_with_strict_unresolved_stop": int((timing["n_strict_unresolved_raw_stops"] > 0).sum()),
                "candidates_with_blank_topology_stop": int((timing["n_blank_topology_raw_stops"] > 0).sum()),
                "candidates_with_conflicting_city_stop": int((timing["n_conflicting_city_raw_stops"] > 0).sum()),
                "total_hours": float(timing["total_hours"].sum()),
                "observed_sea_hours": float(timing["observed_sea_hours"].sum()),
                "intermediate_gap_hours": float(timing["intermediate_gap_hours"].sum()),
            }
        )

    pair_counts = (
        pd.concat(pair_frames, ignore_index=True)
        .groupby(["origin_city", "dest_city"], sort=True)[["candidate_count_c1", "vessel_count_c1"]]
        .sum()
        .reset_index()
    )
    frozen = pd.read_csv(
        input_path(config, "adjacent_episode_pair_manifest_2021.csv"),
        usecols=["origin_city", "dest_city", "candidate_count_2021", "vessel_count_2021"],
    )
    pair_audit = frozen.merge(
        pair_counts,
        on=["origin_city", "dest_city"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    pair_audit["candidate_count_match"] = pair_audit["candidate_count_2021"].eq(pair_audit["candidate_count_c1"])
    pair_audit["vessel_count_match"] = pair_audit["vessel_count_2021"].eq(pair_audit["vessel_count_c1"])
    pair_differences = pair_audit.loc[
        pair_audit["_merge"].ne("both")
        | ~pair_audit["candidate_count_match"]
        | ~pair_audit["vessel_count_match"]
    ].copy()
    pair_differences.to_csv(report_dir / "c1_pair_reproduction_differences.csv", index=False)
    pair_summary = {
        "c0_pair_rows": int(len(frozen)),
        "c1_pair_rows": int(len(pair_counts)),
        "c0_candidate_rows": int(frozen["candidate_count_2021"].sum()),
        "c1_candidate_rows": int(pair_counts["candidate_count_c1"].sum()),
        "difference_rows": int(len(pair_differences)),
        "exact_pair_and_count_reproduction": bool(pair_differences.empty),
    }
    write_json(report_dir / "c1_pair_reproduction_summary.json", pair_summary)
    if not pair_differences.empty:
        raise DataContractError(f"C1 failed the frozen C0 pair gate: {len(pair_differences)} differences")

    r2_by_type = aggregate_group_counts(r2_type_frames, ["detailed_vessel_type_normalized"])
    r2_by_regime = aggregate_group_counts(r2_regime_frames, ["service_regime_proxy"])
    r2_by_type.to_csv(report_dir / "c1_r2_by_vessel_type.csv", index=False)
    r2_by_regime.to_csv(report_dir / "c1_r2_by_service_proxy.csv", index=False)
    r2_top = (
        pd.concat(r2_top_frames, ignore_index=True)
        .sort_values(
            ["r2_repeated_city_count", "n_intermediate_city_episodes", "total_hours", "candidate_id"],
            ascending=[False, False, False, True],
        )
        .head(20)
    ) if r2_top_frames else pd.DataFrame()
    r2_top.to_csv(report_dir / "c1_r2_top20_cases.csv", index=False)
    r2_top_distinct_vessels = (
        pd.concat(r2_distinct_vessel_frames, ignore_index=True)
        .sort_values(
            ["r2_repeated_city_count", "n_intermediate_city_episodes", "total_hours", "candidate_id"],
            ascending=[False, False, False, True],
        )
        .drop_duplicates("imo")
        .head(20)
    )
    r2_top_distinct_vessels.to_csv(
        report_dir / "c1_r2_top20_distinct_vessels.csv", index=False
    )
    r2_shortest = (
        pd.concat(r2_short_frames, ignore_index=True)
        .sort_values(
            ["total_hours", "r2_repeated_city_count", "candidate_id"],
            ascending=[True, False, True],
        )
        .head(20)
    )
    r2_shortest.to_csv(report_dir / "c1_r2_shortest20_cases.csv", index=False)
    r2_by_middle_count = aggregate_group_counts(r2_middle_bin_frames, ["middle_count_bin"])
    r2_by_duration = aggregate_group_counts(r2_duration_bin_frames, ["duration_days_bin"])
    r2_by_middle_count.to_csv(report_dir / "c1_r2_by_middle_count.csv", index=False)
    r2_by_duration.to_csv(report_dir / "c1_r2_by_duration.csv", index=False)
    total_candidates = int(pair_summary["c1_candidate_rows"])
    total_r2 = int(r2_by_regime["r2_flagged"].sum())
    r2_summary = {
        "candidate_rows": total_candidates,
        "r2_flagged_rows": total_r2,
        "r2_rate": total_r2 / total_candidates,
        "disposition": "human_decision_required",
    }
    write_json(report_dir / "c1_r2_summary.json", r2_summary)

    r3_summary_table = (
        pd.concat(r3_frames, ignore_index=True)
        .groupby(
            ["service_regime_proxy", "r3_window_eligible", "r3_recurrence_count", "r3_singleton"],
            sort=True,
        )["candidate_rows"]
        .sum()
        .reset_index()
    )
    r3_summary_table.to_csv(report_dir / "c1_r3_recurrence_distribution.csv", index=False)
    eligible_rows = int(r3_summary_table.loc[r3_summary_table["r3_window_eligible"], "candidate_rows"].sum())
    outside_window_rows = total_candidates - eligible_rows
    singleton_rows = int(r3_summary_table.loc[r3_summary_table["r3_singleton"], "candidate_rows"].sum())
    r3_summary = {
        "candidate_rows": total_candidates,
        "window_eligible_rows": eligible_rows,
        "outside_configured_period_rows": outside_window_rows,
        "singleton_rows": singleton_rows,
        "recurrent_rows": eligible_rows - singleton_rows,
        "singleton_rate_among_window_eligible": singleton_rows / eligible_rows,
        "recurrence_key": ["imo", "origin_city", "dest_city", "period_id"],
        "intermediate_sequence_in_key": False,
    }
    write_json(report_dir / "c1_r3_summary.json", r3_summary)

    failure_frame = pd.concat(time_failures, ignore_index=True) if time_failures else pd.DataFrame(
        columns=[
            "candidate_id", "imo", "origin_city", "dest_city", "total_hours",
            "observed_sea_hours", "intermediate_gap_hours", "time_identity_residual_hours",
            "n_intermediate_raw_stops", "n_strict_unresolved_raw_stops",
            "n_blank_topology_raw_stops", "n_conflicting_city_raw_stops",
        ]
    )
    failure_frame.to_csv(report_dir / "c1_time_decomposition_failures.csv", index=False)
    time_summary = {
        "candidate_rows": int(sum(item["candidate_rows"] for item in time_part_summaries)),
        "failure_rows": int(sum(item["failure_rows"] for item in time_part_summaries)),
        "tolerance_hours": tolerance,
        "max_abs_residual_hours": float(max(item["max_abs_residual_hours"] for item in time_part_summaries)),
        "candidates_with_strict_unresolved_stop": int(sum(item["candidates_with_strict_unresolved_stop"] for item in time_part_summaries)),
        "candidates_with_blank_topology_stop": int(sum(item["candidates_with_blank_topology_stop"] for item in time_part_summaries)),
        "candidates_with_conflicting_city_stop": int(sum(item["candidates_with_conflicting_city_stop"] for item in time_part_summaries)),
        "total_hours": float(sum(item["total_hours"] for item in time_part_summaries)),
        "observed_sea_hours": float(sum(item["observed_sea_hours"] for item in time_part_summaries)),
        "intermediate_gap_hours": float(sum(item["intermediate_gap_hours"] for item in time_part_summaries)),
        "identity_passed": not time_failures,
    }
    write_json(report_dir / "c1_time_decomposition_summary.json", time_summary)

    stop_resolution_frames: list[pd.DataFrame] = []
    dwell_frames: list[pd.DataFrame] = []
    conflicting_stop_frames: list[pd.DataFrame] = []
    for path in stop_paths:
        stops = pd.read_feather(
            path,
            columns=[
                "stop_id",
                "detailed_vessel_type_normalized",
                "service_regime_proxy",
                "vessel_family_proxy",
                "dwell_days",
                "arrival_topology_city_id",
                "departure_topology_city_id",
                "location_resolution",
                "r5_fit_eligible_strict",
                "r5_fit_eligible_one_sided",
                "candidate_intermediate_usage_count",
                "used_by_any_r1_candidate",
            ],
        )
        used = stops.loc[stops["used_by_any_r1_candidate"]].copy()
        conflicting = used.loc[used["location_resolution"].eq("conflicting_cities")].copy()
        if not conflicting.empty:
            conflicting_stop_frames.append(conflicting)
        stop_resolution_frames.append(
            used.groupby(["service_regime_proxy", "location_resolution"], sort=True)
            .agg(unique_stop_rows=("stop_id", "size"), candidate_weighted_uses=("candidate_intermediate_usage_count", "sum"))
            .reset_index()
        )
        fit = used.loc[
            used["service_regime_proxy"].eq("tramp_proxy")
            & used["r5_fit_eligible_strict"]
            & used["dwell_days"].gt(0)
        ].copy()
        if not fit.empty:
            dwell_frames.append(fit)
    stop_resolution = (
        pd.concat(stop_resolution_frames, ignore_index=True)
        .groupby(["service_regime_proxy", "location_resolution"], sort=True)[["unique_stop_rows", "candidate_weighted_uses"]]
        .sum()
        .reset_index()
    )
    stop_resolution.to_csv(report_dir / "c1_5_raw_stop_location_resolution.csv", index=False)
    conflicting_stops = pd.concat(conflicting_stop_frames, ignore_index=True)
    yardstick = pd.read_feather(
        input_path(config, "yardstick_table_2021.feather"),
        columns=["origin_city", "dest_city", "d_yardstick", "same_network_node"],
    )
    conflicting_stops = conflicting_stops.merge(
        yardstick,
        left_on=["arrival_topology_city_id", "departure_topology_city_id"],
        right_on=["origin_city", "dest_city"],
        how="left",
        validate="many_to_one",
    )
    distance_values = conflicting_stops["d_yardstick"].dropna()
    total_conflicting_uses = int(conflicting_stops["candidate_intermediate_usage_count"].sum())
    conflicting_distance_summary: dict[str, Any] = {
        "unique_conflicting_stop_rows": int(len(conflicting_stops)),
        "candidate_weighted_uses": total_conflicting_uses,
        "missing_yardstick_rows": int(conflicting_stops["d_yardstick"].isna().sum()),
        "distance_quantiles_km": {
            str(quantile): float(distance_values.quantile(quantile))
            for quantile in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        },
        "threshold_counts": {},
    }
    for threshold_km in [0, 10, 25, 50, 100, 250, 500, 1000]:
        within = conflicting_stops["d_yardstick"].le(threshold_km)
        weighted_uses = int(
            conflicting_stops.loc[within, "candidate_intermediate_usage_count"].sum()
        )
        conflicting_distance_summary["threshold_counts"][str(threshold_km)] = {
            "unique_stop_rows": int(within.sum()),
            "unique_stop_share": float(within.mean()),
            "candidate_weighted_uses": weighted_uses,
            "candidate_weighted_share": weighted_uses / total_conflicting_uses,
        }
    write_json(
        report_dir / "c1_time_conflicting_city_stop_distance_summary.json",
        conflicting_distance_summary,
    )
    conflicting_stops.sort_values(
        ["d_yardstick", "candidate_intermediate_usage_count", "stop_id"],
        ascending=[False, False, True],
        na_position="last",
    ).head(100).to_csv(
        report_dir / "c1_time_conflicting_city_stop_long_distance_top100.csv",
        index=False,
    )
    dwell = pd.concat(dwell_frames, ignore_index=True)
    if dwell.empty:
        raise DataContractError("No strict candidate-used tramp dwell episodes remain for C1.5")
    quantile_rows: list[dict[str, Any]] = []
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.33, 0.50, 0.75, 0.90, 0.95, 0.99]
    for family, group in [("all_tramp_proxy", dwell), *list(dwell.groupby("vessel_family_proxy", sort=True))]:
        values = group["dwell_days"].to_numpy(dtype=float)
        row: dict[str, Any] = {"vessel_family_proxy": family, "unique_stop_rows": len(values)}
        for quantile in quantiles:
            row[f"p{int(quantile * 100):02d}_days"] = float(np.quantile(values, quantile))
        quantile_rows.append(row)
    quantile_table = pd.DataFrame(quantile_rows)
    quantile_table.to_csv(report_dir / "c1_5_tramp_dwell_quantiles.csv", index=False)

    histogram_config = config["dwell_histogram"]
    edges = np.logspace(
        np.log10(float(histogram_config["minimum_hours"]) / 24.0),
        np.log10(float(histogram_config["maximum_days"])),
        int(histogram_config["log_bin_count"]) + 1,
    )
    unique_counts, _ = np.histogram(dwell["dwell_days"], bins=edges)
    weighted_counts, _ = np.histogram(
        dwell["dwell_days"],
        bins=edges,
        weights=dwell["candidate_intermediate_usage_count"],
    )
    histogram = pd.DataFrame(
        {
            "bin_left_days": edges[:-1],
            "bin_right_days": edges[1:],
            "bin_center_days": np.sqrt(edges[:-1] * edges[1:]),
            "unique_stop_count": unique_counts.astype(np.int64),
            "candidate_weighted_usage_count": weighted_counts.astype(np.int64),
        }
    )
    histogram.to_csv(report_dir / "c1_5_tramp_dwell_histogram.csv", index=False)
    valley = detect_histogram_valley(
        histogram,
        float(histogram_config["candidate_search_minimum_days"]),
        float(histogram_config["candidate_search_maximum_days"]),
    )
    write_json(report_dir / "c1_5_bimodality_diagnostic.json", valley)
    write_dwell_plot(histogram, valley, report_dir / "c1_5_tramp_dwell_distribution.png")

    overall_quantiles = quantile_table.loc[quantile_table["vessel_family_proxy"].eq("all_tramp_proxy")].iloc[0]
    threshold_rows = [
        {
            "candidate_name": "institutional_bunkering_anchor",
            "threshold_days": 1.0,
            "basis": "bunkering and short technical calls are typically below one day",
            "status": "candidate_only",
        },
        {
            "candidate_name": "institutional_gap_geometric_midpoint_low",
            "threshold_days": float(np.sqrt(1.0 * 2.05)),
            "basis": "geometric midpoint between 1-day short-stop and 2.05-day dry-bulk anchors",
            "status": "candidate_only",
        },
        {
            "candidate_name": "institutional_gap_geometric_midpoint_high",
            "threshold_days": float(np.sqrt(1.0 * 2.7)),
            "basis": "geometric midpoint between 1-day short-stop and 2.7-day dry-bulk anchors",
            "status": "candidate_only",
        },
        {
            "candidate_name": "empirical_tramp_p25",
            "threshold_days": float(overall_quantiles["p25_days"]),
            "basis": "2021 strict candidate-used tramp dwell p25 fallback",
            "status": "candidate_only",
        },
        {
            "candidate_name": "empirical_tramp_p33",
            "threshold_days": float(overall_quantiles["p33_days"]),
            "basis": "2021 strict candidate-used tramp dwell p33 fallback",
            "status": "candidate_only",
        },
    ]
    if valley.get("stable_valley_found"):
        threshold_rows.append(
            {
                "candidate_name": "empirical_histogram_valley",
                "threshold_days": float(valley["valley_days"]),
                "basis": "deepest stable two-peak valley in the frozen 0.5-3.0 day search range",
                "status": "candidate_only",
            }
        )
    threshold_table = pd.DataFrame(threshold_rows).sort_values("threshold_days")
    threshold_table.to_csv(report_dir / "c1_5_threshold_candidates.csv", index=False)

    c1_summary = {
        "status": "c1_complete_human_decisions_required",
        "contract_hash": contract_hash,
        "candidate_rows": total_candidates,
        "pair_rows": int(len(pair_counts)),
        "pair_reproduction_passed": True,
        "r2_flagged_rows": total_r2,
        "r2_rate": r2_summary["r2_rate"],
        "r3_singleton_rows": singleton_rows,
        "r3_singleton_rate_among_window_eligible": r3_summary["singleton_rate_among_window_eligible"],
        "time_identity_passed": bool(time_summary["identity_passed"]),
        "time_identity_failure_rows": int(time_summary["failure_rows"]),
        "strict_candidate_used_tramp_stop_rows": int(len(dwell)),
        "stable_empirical_dwell_valley_found": bool(valley.get("stable_valley_found", False)),
        "r2_disposition": "human_decision_required",
        "r5_threshold": "human_decision_required",
        "next_phase_authorized": False,
    }
    write_json(report_dir / "c1_summary.json", c1_summary)

    output_manifest = {
        "contract_hash": contract_hash,
        "candidate_parts": [
            {
                "path": str(path.resolve().relative_to(CODING_ROOT)),
                "sha256": sha256_file(path),
                "rows": int(payload["candidate_rows"]),
            }
            for path, payload in zip(candidate_paths, part_payloads)
        ],
        "raw_stop_parts": [
            {
                "path": str(path.resolve().relative_to(CODING_ROOT)),
                "sha256": sha256_file(path),
                "rows": int(payload["raw_stop_rows"]),
            }
            for path, payload in zip(stop_paths, part_payloads)
        ],
        "tracked_reports": {
            path.name: sha256_file(path)
            for path in sorted(report_dir.iterdir())
            if path.is_file() and path.name != "c1_output_manifest.json"
        },
    }
    write_json(artifact_dir / "c1_output_manifest.json", output_manifest)
    write_json(
        report_dir / "c1_resume_audit.json",
        {
            "contract_hash": contract_hash,
            "task_count": len(part_payloads),
            "tasks_executed_this_invocation": pending_task_count,
            "tasks_skipped_existing": len(part_payloads) - pending_task_count,
            "all_part_sidecars_verified": True,
        },
    )

    (report_dir / "c1_r2_precheck.md").write_text(
        "# C1.2 R2 Precheck\n\n"
        f"- Candidate rows: {total_candidates:,}.\n"
        f"- Repeated-intermediate-city rows: {total_r2:,} ({r2_summary['r2_rate']:.3%}).\n"
        "- The top 20 cases are in `c1_r2_top20_cases.csv`.\n"
        "- No candidate was removed. R2 disposition requires a human decision.\n",
        encoding="utf-8",
    )
    (report_dir / "c1_time_decomposition_audit.md").write_text(
        "# C1.4 Time Decomposition Audit\n\n"
        f"- Candidate rows: {time_summary['candidate_rows']:,}.\n"
        f"- Identity failures above {tolerance:g} hours: {time_summary['failure_rows']:,}.\n"
        f"- Maximum absolute residual: {time_summary['max_abs_residual_hours']:.12g} hours.\n"
        f"- Candidates containing at least one strict-unresolved raw stop: "
        f"{time_summary['candidates_with_strict_unresolved_stop']:,}.\n"
        f"- Candidates containing conflicting-city raw stops: "
        f"{time_summary['candidates_with_conflicting_city_stop']:,}.\n\n"
        "The identity uses independently summed retained OD-leg durations and the gaps between adjacent legs. "
        "Location-quality flags remain explicit; an exact identity does not convert a conflicting-city gap into port dwell.\n",
        encoding="utf-8",
    )
    (report_dir / "c1_5_tramp_dwell_diagnostic.md").write_text(
        "# C1.5 Tramp Dwell Diagnostic\n\n"
        f"- Strict, candidate-used tramp raw stops: {len(dwell):,}.\n"
        f"- Stable 0.5-3.0 day histogram valley found: {bool(valley.get('stable_valley_found', False))}.\n"
        "- `c1_5_threshold_candidates.csv` contains candidates only; no R5 threshold is frozen.\n"
        "- Strict fitting requires both sides of a raw stop to resolve to the same nonblank city. "
        "One-sided, blank, and conflicting-city stops remain in the location-resolution audit.\n",
        encoding="utf-8",
    )
    output_manifest["tracked_reports"] = {
        path.name: sha256_file(path)
        for path in sorted(report_dir.iterdir())
        if path.is_file()
    }
    write_json(artifact_dir / "c1_output_manifest.json", output_manifest)
    return c1_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--task-count", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if int(config["analysis_year"]) != 2021:
        raise DataContractError("C1 development is authorized for 2021 only")
    if args.workers is not None:
        if not 1 <= args.workers <= 16:
            raise DataContractError("Worker override must be between 1 and 16")
        config["worker_count"] = args.workers
    if args.task_count is not None:
        if not 1 <= args.task_count <= 64:
            raise DataContractError("Task-count override must be between 1 and 64")
        config["task_count"] = args.task_count

    input_manifest = verify_inputs(config)
    candidate_contract_payload = {
        "config": config,
        "core_sha256": sha256_file(SCRIPT_PATH.with_name("candidate_core.py")),
        "input_sha256": config["input_sha256"],
    }
    runtime_provenance = {
        "candidate_contract": candidate_contract_payload,
        "report_script_sha256": sha256_file(SCRIPT_PATH),
        "python": sys.version,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }
    contract_hash = canonical_hash(candidate_contract_payload)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.run_dir / "execution_contract.json",
        {
            "contract_hash": contract_hash,
            "runtime_provenance": runtime_provenance,
            "inputs": input_manifest,
            "pid": os.getpid(),
        },
    )

    print("Loading frozen 2021 C0 artifacts", flush=True)
    episodes, legs, call_meta = load_runtime_inputs(config)
    known_types = set(episodes["detailed_vessel_type_normalized"].astype(str).unique())
    classified_types = set(config["liner_proxy_types"]) | set(config["tramp_proxy_types"])
    if known_types != classified_types:
        raise DataContractError(
            f"Diagnostic vessel-type proxy is incomplete; missing={sorted(known_types - classified_types)}, "
            f"extra={sorted(classified_types - known_types)}"
        )
    initialize_globals(episodes, legs, call_meta, config, contract_hash)
    tasks = partition_vessels(episodes, int(config["task_count"]))
    write_json(
        args.run_dir / "task_manifest.json",
        {
            "contract_hash": contract_hash,
            "tasks": [
                {"task_id": task_id, "mmsi_count": len(values), "mmsi_values": values}
                for task_id, values in enumerate(tasks)
            ],
        },
    )
    part_payloads, pending_task_count = run_tasks(
        config,
        args.artifact_dir,
        args.run_dir,
        tasks,
        contract_hash,
    )
    candidate_rows = int(sum(item["candidate_rows"] for item in part_payloads))
    if candidate_rows != int(config["expected_candidate_count"]):
        raise DataContractError(
            f"Expected {config['expected_candidate_count']} candidates, found {candidate_rows}"
        )
    summary = generate_reports(
        config,
        args.artifact_dir,
        args.report_dir,
        part_payloads,
        contract_hash,
        pending_task_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
