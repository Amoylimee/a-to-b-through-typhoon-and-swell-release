#!/usr/bin/env python3
"""Build the frozen, reproducible E002 holdout sample list from adopted products."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CODING_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = CODING_ROOT / "data/processed/products/companion_tracks/2025"
WEATHER_ROOT = CODING_ROOT / "data/processed/products/voyage_weather_facts/2025"
SEED = 20260923
DURATIONS_HOURS = (1, 2, 4, 6, 12, 24)
GAP_TOLERANCE_SECONDS = 10 * 60
DENSE_GAP_SECONDS = 30 * 60
CONTEXT_SECONDS = 60 * 60
PER_STRATUM = 500
STORM_NEAR_SAMPLE = 500
SHIP_TYPES = (
    "container", "general_cargo", "product_chemical_tanker", "bulk_carrier",
    "ro_ro_vehicle", "other_retained", "crude_oil_tanker", "gas_tanker", "reefer",
)


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def eligible_windows(
    timestamps_ns: np.ndarray,
    duration_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right observed-fix indexes meeting dense-context rules."""
    n = len(timestamps_ns)
    if n < 4:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    diffs = np.diff(timestamps_ns)
    starts = np.r_[0, np.flatnonzero(diffs > DENSE_GAP_SECONDS * 1_000_000_000) + 1]
    ends = np.r_[starts[1:] - 1, n - 1]
    run_start = np.empty(n, dtype=np.int64)
    run_end = np.empty(n, dtype=np.int64)
    for start, end in zip(starts, ends):
        run_start[start : end + 1] = start
        run_end[start : end + 1] = end

    nominal_ns = int(duration_hours * 3_600 * 1_000_000_000)
    target_ns = timestamps_ns + nominal_ns
    after = np.searchsorted(timestamps_ns, target_ns, side="left")
    before = after - 1
    valid_before = before > np.arange(n)
    valid_after = after < n
    before_error = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    after_error = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    before_error[valid_before] = np.abs(timestamps_ns[before[valid_before]] - target_ns[valid_before])
    after_error[valid_after] = np.abs(timestamps_ns[after[valid_after]] - target_ns[valid_after])
    use_after = after_error < before_error
    right = np.where(use_after, after, before)
    error = np.minimum(after_error, before_error)
    left = np.arange(n, dtype=np.int64)

    valid = (
        (right > left)
        & (error <= GAP_TOLERANCE_SECONDS * 1_000_000_000)
    )
    valid &= run_start[left] == run_start[np.clip(right, 0, n - 1)]
    valid &= timestamps_ns[left] - timestamps_ns[run_start[left]] >= CONTEXT_SECONDS * 1_000_000_000
    valid &= timestamps_ns[run_end[np.clip(right, 0, n - 1)]] - timestamps_ns[right] >= CONTEXT_SECONDS * 1_000_000_000
    actual_gap_ns = timestamps_ns[np.clip(right, 0, n - 1)] - timestamps_ns[left]
    if duration_hours <= 4:
        valid &= actual_gap_ns <= 4 * 3_600 * 1_000_000_000
    else:
        valid &= actual_gap_ns >= 4 * 3_600 * 1_000_000_000
        valid &= actual_gap_ns <= 24 * 3_600 * 1_000_000_000
    valid &= right - left >= 2
    return left[valid], right[valid].astype(np.int64)


def _candidate_summary(path: Path) -> pd.DataFrame:
    frame = pq.read_table(
        path,
        columns=[
            "candidate_id", "mmsi", "vessel_uid_clean", "shiptype_group",
            "origin_city", "dest_city", "departure_time", "arrival_time",
        ],
    ).to_pandas()
    if frame.candidate_id.duplicated().any():
        raise ValueError(f"duplicate candidate IDs in {path}")
    return frame.set_index("candidate_id", drop=False)


def _storm_times(shard_dir: Path) -> dict[str, np.ndarray]:
    pieces: list[pd.DataFrame] = []
    for path in sorted(shard_dir.glob("weather_2025-??.parquet")):
        table = pq.read_table(
            path,
            columns=["candidate_id", "position_time_utc"],
            filters=[
                ("tc_within_500km", "=", True),
                ("track_evidence", "=", "observed_ais"),
            ],
        )
        if table.num_rows:
            pieces.append(table.to_pandas())
    if not pieces:
        return {}
    frame = pd.concat(pieces, ignore_index=True)
    frame["position_time_utc"] = pd.to_datetime(frame.position_time_utc, utc=True)
    result: dict[str, np.ndarray] = {}
    for candidate_id, group in frame.groupby("candidate_id", sort=False, observed=True):
        result[str(candidate_id)] = np.unique(
            group.position_time_utc.astype("int64").to_numpy(dtype=np.int64)
        )
    return result


def _sample_row(
    *, candidate: pd.Series, candidate_id: str, source_shard: str,
    shiptype: str, duration: int, left_i: int, right_i: int,
    timestamps: np.ndarray, coords: np.ndarray, sample_set: str,
) -> dict[str, object]:
    hidden_count = int(right_i - left_i - 1)
    return {
        "sample_set": sample_set,
        "candidate_id": candidate_id,
        "source_shard": source_shard,
        "shiptype_group": shiptype,
        "nominal_gap_hours": int(duration),
        "actual_gap_hours": (timestamps[right_i] - timestamps[left_i]) / 3.6e12,
        "gap_start_utc": pd.Timestamp(timestamps[left_i], tz="UTC").isoformat(),
        "gap_end_utc": pd.Timestamp(timestamps[right_i], tz="UTC").isoformat(),
        "truth_observation_count": hidden_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    main_candidates: dict[tuple[int, str], list[tuple[int, dict[str, object]]]] = {}
    storm_candidates: list[tuple[int, dict[str, object]]] = []
    eligible_counts: dict[tuple[int, str], int] = {}
    storm_eligible_count = 0
    shard_paths = sorted((TRACK_ROOT / "run/shards").glob("shard_*/candidate_summary.parquet"))
    if len(shard_paths) != 253:
        raise SystemExit(f"expected 253 adopted track shards, found {len(shard_paths)}")

    for shard_no, summary_path in enumerate(shard_paths, start=1):
        shard_dir = summary_path.parent
        source_shard = shard_dir.name
        summary = _candidate_summary(summary_path)
        storm_by_candidate = _storm_times(WEATHER_ROOT / "run/shards" / source_shard)
        track_path = shard_dir / "ten_minute_complete_candidate.parquet"
        observed = pq.read_table(
            track_path,
            columns=["candidate_id", "observation_time_utc", "utc", "lat", "lon"],
            filters=[("track_evidence", "=", "observed_ais")],
        ).to_pandas()
        observed = observed.loc[observed.observation_time_utc.notna()].copy()
        observed["observation_time_utc"] = pd.to_datetime(
            observed.observation_time_utc, utc=True
        )
        observed.sort_values(["candidate_id", "observation_time_utc"], kind="stable", inplace=True)

        for candidate_id, group in observed.groupby("candidate_id", sort=False, observed=True):
            cid = str(candidate_id)
            if cid not in summary.index:
                raise ValueError(f"observed candidate absent from adopted summary: {cid}")
            candidate = summary.loc[cid]
            shiptype = str(candidate.shiptype_group)
            if shiptype == "nan":
                continue
            times = group.observation_time_utc.astype("int64").to_numpy(dtype=np.int64)
            coords = group[["lat", "lon"]].to_numpy(dtype=np.float64)
            if len(times) < 4 or np.any(np.diff(times) <= 0):
                continue
            storm_times = storm_by_candidate.get(cid, np.empty(0, dtype=np.int64))

            for duration in DURATIONS_HOURS:
                lefts, rights = eligible_windows(times, duration)
                if not len(lefts):
                    continue
                key = (duration, shiptype)
                eligible_counts[key] = eligible_counts.get(key, 0) + 1
                choice_rng = np.random.default_rng(stable_u64(SEED, cid, duration, "window"))
                chosen = int(choice_rng.integers(len(lefts)))
                left_i, right_i = int(lefts[chosen]), int(rights[chosen])
                row = _sample_row(
                    candidate=candidate, candidate_id=cid, source_shard=source_shard,
                    shiptype=shiptype, duration=duration, left_i=left_i,
                    right_i=right_i, timestamps=times, coords=coords,
                    sample_set="gap_length_x_shiptype",
                )
                sample_priority = stable_u64(SEED, "main", duration, shiptype, cid)
                existing = main_candidates.setdefault(key, [])
                existing.append((sample_priority, row))
                if len(existing) > PER_STRATUM * 2:
                    existing.sort(key=lambda x: x[0])
                    del existing[PER_STRATUM:]

                inside_storm = np.searchsorted(storm_times, times[left_i], side="right")
                storm_end = np.searchsorted(storm_times, times[right_i], side="left")
                if storm_end > inside_storm:
                    storm_eligible_count += 1
                    storm_row = dict(row)
                    storm_row["sample_set"] = "tc500km_additional"
                    storm_priority = stable_u64(SEED, "storm", duration, shiptype, cid)
                    storm_candidates.append((storm_priority, storm_row))
                    if len(storm_candidates) > STORM_NEAR_SAMPLE * 10:
                        storm_candidates.sort(key=lambda x: x[0])
                        del storm_candidates[STORM_NEAR_SAMPLE * 5:]

        print(
            f"shard {shard_no:03d}/253 {source_shard}: observed_rows={len(observed):,}; "
            f"eligible_strata={len(eligible_counts)}; storm_candidates={storm_eligible_count:,}",
            flush=True,
        )

    selected_rows: list[dict[str, object]] = []
    stratum_report: list[dict[str, object]] = []
    for duration in DURATIONS_HOURS:
        for shiptype in SHIP_TYPES:
            key = (duration, shiptype)
            ranked = sorted(main_candidates.get(key, []), key=lambda x: x[0])
            eligible_n = eligible_counts.get(key, 0)
            selected = ranked[:PER_STRATUM]
            stratum_report.append({
                "nominal_gap_hours": duration,
                "shiptype_group": shiptype,
                "eligible_voyages": eligible_n,
                "selected_segments": len(selected),
                "required_segments": PER_STRATUM,
                "status": "PASS" if len(selected) >= PER_STRATUM else "FAIL",
            })
            selected_rows.extend(row for _, row in selected)

    main_windows = {
        (str(row["candidate_id"]), str(row["gap_start_utc"]), str(row["gap_end_utc"]))
        for row in selected_rows
    }
    storm_ranked = sorted(storm_candidates, key=lambda x: x[0])
    storm_selected: list[tuple[int, dict[str, object]]] = []
    storm_seen_candidates: set[str] = set()
    for priority, row in storm_ranked:
        key = (str(row["candidate_id"]), str(row["gap_start_utc"]), str(row["gap_end_utc"]))
        candidate_id = key[0]
        if key in main_windows or candidate_id in storm_seen_candidates:
            continue
        storm_seen_candidates.add(candidate_id)
        storm_selected.append((priority, row))
        if len(storm_selected) >= STORM_NEAR_SAMPLE:
            break
    selected_rows.extend(row for _, row in storm_selected)
    storm_status = "PASS" if len(storm_selected) >= STORM_NEAR_SAMPLE else "FAIL"

    samples = pd.DataFrame(selected_rows)
    if samples.empty:
        raise SystemExit("no eligible holdout sample windows found")
    samples.sort_values(
        ["sample_set", "nominal_gap_hours", "shiptype_group", "candidate_id"],
        kind="stable",
        inplace=True,
    )
    samples.reset_index(drop=True, inplace=True)
    samples.insert(0, "sample_id", [f"H{i:06d}" for i in range(1, len(samples) + 1)])
    sample_path = args.out / "holdout_sample_list.csv"
    samples.to_csv(sample_path, index=False)
    report = {
        "seed": SEED,
        "gap_lengths_hours": DURATIONS_HOURS,
        "minimum_per_gap_length_x_shiptype": PER_STRATUM,
        "additional_tc500km_sample_size": STORM_NEAR_SAMPLE,
        "gap_duration_tolerance_seconds": GAP_TOLERANCE_SECONDS,
        "dense_context": {
            "window_plus_minus_hours": 1,
            "maximum_adjacent_observed_ais_gap_seconds": DENSE_GAP_SECONDS,
        },
        "track_adoption_manifest_sha256": __import__("hashlib").sha256(
            (TRACK_ROOT / "adoption_manifest.json").read_bytes()
        ).hexdigest(),
        "weather_adoption_manifest_sha256": __import__("hashlib").sha256(
            (WEATHER_ROOT / "adoption_manifest.json").read_bytes()
        ).hexdigest(),
        "all_strata_pass": all(x["status"] == "PASS" for x in stratum_report),
        "strata": stratum_report,
        "storm_near_eligible_voyage_windows": storm_eligible_count,
        "storm_near_selected": len(storm_selected),
        "storm_near_status": storm_status,
        "selected_total_rows": len(samples),
        "sample_list": str(sample_path),
        "sample_list_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        # The sample list intentionally contains identifiers and timestamps only;
        # coordinates and MMSI/API vessel identifiers remain in adopted products.
        "samples_have_coordinates": False,
        "sharing_classification": "internal review only; do not publish or send externally",
    }
    (args.out / "sample_inventory.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_strata_pass"] or storm_status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
