#!/usr/bin/env python3
"""Measure per-candidate AIS coverage under the weather-matching gap rule.

This is a read-only diagnostic.  It opens no product for writing and computes
no weather.  It answers one question: for every accepted one-leg candidate,
how much of its voyage window can a weather match actually see?

The rule applied here is the accepted weather rule. E020 interpolates a
position across a gap of at most four hours and starts a new segment when the
adjacent gap is strictly greater than four hours
(``maximum_interpolation_gap_hours = 4.0``, ``position_fallback = none``).
Time inside a longer gap has no defensible position, so it can carry no wave or
cyclone value.  Both endpoint gaps count: a candidate whose first fix arrives
nine hours after departure is dark for those nine hours regardless of how dense
the rest of the track is.

    measurable_hours = window_hours - sum(gap for gap in gaps if gap > 4h)
    coverage         = measurable_hours / window_hours

where ``gaps`` spans departure -> first fix, every adjacent pair, and last fix
-> arrival.

AIS validity and identity follow the accepted stage so that this diagnostic
counts exactly the observations a real weather match would use. Rows are
filtered by ``ais_identity.prepare_ais_batch`` and deduplicated by
``deduplicate_vessel_ais``; the authoritative identity is the exact pair
``(mmsi, vessel_uid_clean)`` recorded in CONTRACT section 5, and a row whose
MMSI matches while its source ``vessel_uid`` differs is an identity-join
failure that is excluded and counted, never relabelled.

Stage one shuffles cleaned AIS into vessel buckets, keeping only what the
candidate windows can reach.  Stage two evaluates candidates bucket by bucket.
Both stages checkpoint, so an interrupted run resumes without rereading.

    python3 10_measure_leg_ais_coverage.py --years 2025 --workers 12

Environment: sinbue_env (Python 3.10)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODING_ROOT = HERE.parents[3]
REPO_ROOT = CODING_ROOT.parent
WEATHER_STAGE_ROOT = HERE.parent
if str(WEATHER_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(WEATHER_STAGE_ROOT))
ENVIRONMENT_STAGE_ROOT = CODING_ROOT / "pipeline/00_environment_and_config"
if str(ENVIRONMENT_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_STAGE_ROOT))

from ais_identity import (  # noqa: E402
    deduplicate_vessel_ais,
    normalise_required_identifier,
    prepare_ais_batch,
    stable_vessel_buckets,
    validate_ais_physical_identity,
)
from accepted_candidate_root import load_accepted_2025_candidates  # noqa: E402

# The weather-matching interpolation limit.  Not a tunable of this diagnostic:
# it is read from the accepted E020 contract and must match it exactly.
MAXIMUM_INTERPOLATION_GAP_HOURS = 4.0
GAP_LIMIT_NS = np.int64(MAXIMUM_INTERPOLATION_GAP_HOURS * 3_600 * 1_000_000_000)
NS_PER_HOUR = 3_600 * 1_000_000_000

YEARS = (2025,)
PRODUCTS = CODING_ROOT / "data/processed/products"
CLEANED_AIS = PRODUCTS / "cleaned_ais"

CANDIDATE_COLUMNS = (
    "candidate_id",
    "mmsi",
    "vessel_uid_clean",
    "departure_time",
    "arrival_time",
    "total_hours",
    "observed_sea_hours",
    "intermediate_gap_hours",
    "n_observed_sea_legs",
    "shiptype_group",
    "origin_city",
    "dest_city",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a parquet file through a temporary name so readers never see half."""

    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False, compression="zstd")
    os.replace(partial, path)


# --------------------------------------------------------------------------
# stage one: shuffle cleaned AIS into vessel buckets
# --------------------------------------------------------------------------


def shuffle_one_shard(
    shard: Path,
    year: int,
    bucket_count: int,
    bucket_root: Path,
    wanted_mmsi: frozenset[str],
) -> dict[str, int]:
    """Read one cleaned-AIS shard and append its wanted rows to bucket files."""

    frame = feather.read_table(
        shard, columns=["vessel_uid", "mmsi", "UTC", "lat", "lon"], memory_map=True
    ).to_pandas()
    points, audit = prepare_ais_batch(
        frame,
        source_year=year,
        source_shard=shard.name,
        source_row_offset=0,
        bucket_count=bucket_count,
    )
    del frame
    if wanted_mmsi:
        points = points.loc[points["mmsi"].isin(wanted_mmsi)]
    audit["kept_rows"] = int(len(points))
    if points.empty:
        return audit

    for bucket, part in points.groupby("vessel_bucket", sort=False, observed=True):
        target = bucket_root / f"bucket_{int(bucket):04d}"
        target.mkdir(parents=True, exist_ok=True)
        out = target / f"{year}_{shard.stem}.feather"
        feather.write_feather(
            pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False),
            out,
            compression="zstd",
        )
    return audit


def run_shuffle(
    years: tuple[int, ...],
    bucket_count: int,
    bucket_root: Path,
    wanted_mmsi: frozenset[str],
    workers: int,
) -> dict[str, Any]:
    """Shuffle every needed cleaned-AIS year into vessel buckets."""

    done_marker = bucket_root / "_shuffle_complete.json"
    if done_marker.exists():
        print(f"  shuffle already complete at {bucket_root}")
        return json.loads(done_marker.read_text())

    if bucket_root.exists():
        shutil.rmtree(bucket_root)
    bucket_root.mkdir(parents=True)

    tasks: list[tuple[Path, int]] = []
    for year in years:
        shards = sorted((CLEANED_AIS / str(year)).glob("shard_*.feather"))
        if not shards:
            raise FileNotFoundError(f"no cleaned AIS shards for {year}")
        tasks.extend((shard, year) for shard in shards)

    totals = {
        "input_rows": 0,
        "valid_rows": 0,
        "kept_rows": 0,
        "rejected_identifier_rows": 0,
        "rejected_timestamp_rows": 0,
        "rejected_coordinate_rows": 0,
    }
    print(f"  shuffling {len(tasks)} shards over {len(years)} years -> {bucket_count} buckets")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                shuffle_one_shard, shard, year, bucket_count, bucket_root, wanted_mmsi
            ): (shard, year)
            for shard, year in tasks
        }
        for n, future in enumerate(as_completed(futures), start=1):
            audit = future.result()
            for key in totals:
                totals[key] += audit.get(key, 0)
            if n % 50 == 0 or n == len(tasks):
                print(f"    {n}/{len(tasks)} shards, {totals['kept_rows']:,} rows kept")

    done_marker.write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    return totals


# --------------------------------------------------------------------------
# stage two: evaluate candidate coverage bucket by bucket
# --------------------------------------------------------------------------


def coverage_for_vessel(
    utc_ns: np.ndarray,
    starts_ns: np.ndarray,
    ends_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-window dark hours, point counts, longest gap and endpoint dark.

    ``utc_ns`` must be sorted and unique.  Each window is closed on both ends;
    a fix exactly on a boundary counts as observed.
    """

    left = np.searchsorted(utc_ns, starts_ns, side="left")
    right = np.searchsorted(utc_ns, ends_ns, side="right")

    dark = np.zeros(len(starts_ns), dtype=np.float64)
    counts = (right - left).astype(np.int64)
    longest = np.zeros(len(starts_ns), dtype=np.float64)
    endpoint_dark = np.zeros(len(starts_ns), dtype=np.float64)

    for i in range(len(starts_ns)):
        lo, hi = left[i], right[i]
        start, end = starts_ns[i], ends_ns[i]
        if hi <= lo:
            # No fix at all inside the window: the whole window is dark when it
            # exceeds the interpolation limit, and otherwise still unobserved.
            span = end - start
            dark[i] = span / NS_PER_HOUR if span > GAP_LIMIT_NS else 0.0
            longest[i] = span / NS_PER_HOUR
            endpoint_dark[i] = dark[i]
            continue
        inner = utc_ns[lo:hi]
        edges = np.empty(len(inner) + 2, dtype=np.int64)
        edges[0] = start
        edges[1:-1] = inner
        edges[-1] = end
        gaps = np.diff(edges)
        longest[i] = gaps.max() / NS_PER_HOUR
        over = gaps > GAP_LIMIT_NS
        if over.any():
            dark[i] = gaps[over].sum() / NS_PER_HOUR
            # The first and last gap are the endpoint gaps.
            edge_over = over[[0, -1]]
            edge_gaps = gaps[[0, -1]]
            endpoint_dark[i] = edge_gaps[edge_over].sum() / NS_PER_HOUR
    return dark, counts, longest, endpoint_dark


def evaluate_bucket(
    bucket_dir: Path,
    candidates: pd.DataFrame,
    output_dir: Path,
) -> dict[str, int]:
    """Compute coverage for every candidate whose vessel lives in this bucket."""

    out_path = output_dir / f"{bucket_dir.name}.parquet"
    if out_path.exists():
        return {"resumed": 1, "candidates": int(len(candidates))}

    audit = {
        "identity_join_mismatch_rows": 0,
        "identity_join_empty": 0,
        "duplicate_vessel_utc_rows_skipped": 0,
        "irrelevant_vessel_rows_skipped": 0,
    }

    parts = sorted(bucket_dir.glob("*.feather"))
    if parts:
        points = pd.concat(
            [feather.read_table(p, memory_map=True).to_pandas() for p in parts],
            ignore_index=True,
        )
        # Order matters: the identity join runs FIRST, deduplication SECOND.
        # Reversing them lets a second vessel
        # sharing this MMSI reach the deduplicator, where it either trips the
        # disagreement guard or, when its position happens to agree, silently
        # wins the "keep first" tie and evicts the real vessel's fix.
        identity = candidates[["mmsi", "vessel_uid_clean"]].drop_duplicates()
        points, identity_audit = validate_ais_physical_identity(points, identity)
        points, duplicate_rows = deduplicate_vessel_ais(points)
        audit["identity_join_mismatch_rows"] = identity_audit[
            "api_identity_mismatch_rows"
        ]
        audit["irrelevant_vessel_rows_skipped"] = identity_audit[
            "irrelevant_vessel_rows_skipped"
        ]
        audit["duplicate_vessel_utc_rows_skipped"] = int(duplicate_rows)
    else:
        points = pd.DataFrame(columns=["mmsi", "vessel_uid", "UTC"])

    results: list[pd.DataFrame] = []

    by_vessel = {
        key: group for key, group in points.groupby("mmsi", sort=False, observed=True)
    }
    for mmsi, group in candidates.groupby("mmsi", sort=False, observed=True):
        vessel_points = by_vessel.get(mmsi)
        if vessel_points is None or vessel_points.empty:
            audit["identity_join_empty"] += int(len(group))
            usable = np.empty(0, dtype=np.int64)
        else:
            usable = np.sort(
                vessel_points["UTC"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            )

        starts = group["departure_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        ends = group["arrival_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        dark, counts, longest, endpoint_dark = coverage_for_vessel(usable, starts, ends)

        window_hours = (ends - starts) / NS_PER_HOUR
        measurable = np.maximum(window_hours - dark, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(window_hours > 0, measurable / window_hours, np.nan)

        results.append(
            pd.DataFrame(
                {
                    "candidate_id": group["candidate_id"].to_numpy(),
                    "departure_year": group["departure_year"].to_numpy(),
                    "departure_time": group["departure_time"].to_numpy(),
                    "arrival_time": group["arrival_time"].to_numpy(),
                    "shiptype_group": group["shiptype_group"].to_numpy(),
                    "origin_city": group["origin_city"].to_numpy(),
                    "dest_city": group["dest_city"].to_numpy(),
                    "window_hours": window_hours,
                    "observed_sea_hours": group["observed_sea_hours"].to_numpy(),
                    "ais_points_in_window": counts,
                    "dark_hours_over_4h": dark,
                    "endpoint_dark_hours": endpoint_dark,
                    "longest_gap_hours": longest,
                    "measurable_hours": measurable,
                    "coverage_ratio": ratio,
                }
            )
        )

    frame = (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame(columns=["candidate_id"])
    )
    write_atomic(frame, out_path)
    audit["candidates"] = int(len(frame))
    audit["resumed"] = 0
    return audit


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def summarise(frame: pd.DataFrame, tables: Path) -> dict[str, Any]:
    """Write the comparison tables and return the headline numbers."""

    tables.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame["departure_month"] = (
        pd.to_datetime(frame["departure_time"], utc=True).dt.strftime("%Y-%m")
    )
    def block(group: pd.DataFrame) -> pd.Series:
        window = group["window_hours"].sum()
        measurable = group["measurable_hours"].sum()
        return pd.Series(
            {
                "candidates": int(len(group)),
                "window_hours": window,
                "measurable_hours": measurable,
                # Time-weighted: the share of all voyage time a weather match sees.
                "coverage_time_weighted": 100 * measurable / window if window else np.nan,
                # Per-voyage: the typical voyage's own coverage.
                "coverage_median_per_voyage": 100 * group["coverage_ratio"].median(),
                "voyages_fully_covered_pct": 100 * (group["coverage_ratio"] >= 0.999).mean(),
                "voyages_under_50pct": 100 * (group["coverage_ratio"] < 0.50).mean(),
                "median_longest_gap_hours": group["longest_gap_hours"].median(),
            }
        )

    by_month = frame.groupby("departure_month").apply(block, include_groups=False)
    by_month.to_csv(tables / "01_coverage_by_month.csv")

    by_year = frame.groupby("departure_year").apply(block, include_groups=False)
    by_year.to_csv(tables / "02_coverage_by_year.csv")

    # Coverage is not spread evenly across voyage durations, so retain that
    # distribution within the accepted 2025 population.
    frame["duration_band"] = pd.cut(
        frame["window_hours"] / 24.0,
        bins=[0, 2, 5, 10, 20, 40, np.inf],
        labels=["<2d", "2-5d", "5-10d", "10-20d", "20-40d", ">40d"],
        right=False,
    )
    by_band = frame.groupby(["departure_year", "duration_band"], observed=True).apply(
        block, include_groups=False
    )
    by_band.to_csv(tables / "04_coverage_by_duration_band.csv")

    return {
        "by_year": by_year.to_dict(orient="index"),
    }


# --------------------------------------------------------------------------


def load_candidates(years: tuple[int, ...]) -> pd.DataFrame:
    if tuple(sorted(set(years))) != (2025,):
        raise ValueError("E001 leg coverage reads the accepted 2025 candidate root only")
    frame = load_accepted_2025_candidates(CANDIDATE_COLUMNS)
    # Canonicalise exactly as the AIS side does, so the identity join compares
    # like with like even if a future source reintroduces ".0" suffixes.
    frame["mmsi"] = normalise_required_identifier(frame["mmsi"], "candidate mmsi")
    frame["vessel_uid_clean"] = normalise_required_identifier(
        frame["vessel_uid_clean"], "candidate vessel_uid_clean"
    )
    return frame


def ais_years_needed(candidates: pd.DataFrame) -> tuple[int, ...]:
    """Every calendar year a candidate window can touch."""

    start = pd.to_datetime(candidates["departure_time"], utc=True).dt.year.min()
    end = pd.to_datetime(candidates["arrival_time"], utc=True).dt.year.max()
    available = [y for y in YEARS if start <= y <= end]
    if not available:
        raise ValueError("candidate windows touch no available AIS year")
    return tuple(available)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args()

    if args.workers > 12:
        raise SystemExit("CONTRACT caps this project at 12 workers")

    years = tuple(sorted(args.years))
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_root = args.run_root or (
        CODING_ROOT / ".runs" / f"{stamp}_E031_leg-ais-coverage"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    bucket_root = run_root / "vessel_buckets"
    per_candidate = run_root / "per_candidate"
    per_candidate.mkdir(exist_ok=True)

    print(f"E031 coverage diagnostic  ->  {run_root}")
    print(f"  departure years : {', '.join(str(y) for y in years)}")
    print(f"  gap rule        : segment breaks above {MAXIMUM_INTERPOLATION_GAP_HOURS} h")

    candidates = load_candidates(years)
    print(f"  candidates      : {len(candidates):,}")

    ais_years = ais_years_needed(candidates)
    print(f"  AIS years needed: {', '.join(str(y) for y in ais_years)}")

    wanted = frozenset(candidates["mmsi"].dropna().unique().tolist())
    print(f"  vessels         : {len(wanted):,}")

    shuffle_audit = run_shuffle(
        ais_years, args.buckets, bucket_root, wanted, args.workers
    )
    print(f"  AIS rows kept   : {shuffle_audit['kept_rows']:,}")

    candidates["vessel_bucket"] = stable_vessel_buckets(
        candidates["mmsi"], args.buckets
    )

    print("  evaluating candidate coverage")
    totals = {
        "identity_join_mismatch_rows": 0,
        "identity_join_empty": 0,
        "duplicate_vessel_utc_rows_skipped": 0,
        "irrelevant_vessel_rows_skipped": 0,
        "candidates": 0,
    }
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for bucket, group in candidates.groupby("vessel_bucket", sort=True):
            bucket_dir = bucket_root / f"bucket_{int(bucket):04d}"
            bucket_dir.mkdir(parents=True, exist_ok=True)
            futures[
                pool.submit(
                    evaluate_bucket, bucket_dir, group.reset_index(drop=True), per_candidate
                )
            ] = bucket
        for n, future in enumerate(as_completed(futures), start=1):
            audit = future.result()
            for key in totals:
                totals[key] += audit.get(key, 0)
            if n % 8 == 0 or n == len(futures):
                print(f"    {n}/{len(futures)} buckets")

    frame = pd.concat(
        [pd.read_parquet(p) for p in sorted(per_candidate.glob("bucket_*.parquet"))],
        ignore_index=True,
    )
    if len(frame) != len(candidates):
        raise SystemExit(
            f"row closure failed: {len(frame):,} results for {len(candidates):,} candidates"
        )

    headline = summarise(frame, run_root / "tables")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": sha256_file(Path(__file__)),
        "read_only": True,
        "products_modified": 0,
        "departure_years": list(years),
        "ais_years_read": list(ais_years),
        "maximum_interpolation_gap_hours": MAXIMUM_INTERPOLATION_GAP_HOURS,
        "candidates": int(len(frame)),
        "vessels": len(wanted),
        "shuffle_audit": shuffle_audit,
        "identity_join": {
            "policy": "exact (mmsi, vessel_uid_clean) pair; CONTRACT section 5",
            "order": "validate_ais_physical_identity then deduplicate_vessel_ais",
            "mismatch_rows_excluded": totals["identity_join_mismatch_rows"],
            "irrelevant_vessel_rows_skipped": totals["irrelevant_vessel_rows_skipped"],
            "duplicate_vessel_utc_rows_skipped": totals[
                "duplicate_vessel_utc_rows_skipped"
            ],
            "candidates_with_no_matching_ais": totals["identity_join_empty"],
        },
        "headline": headline,
    }
    (run_root / "coverage_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    print("\n--- coverage by departure year ---")
    for year, row in headline["by_year"].items():
        print(
            f"  {year:<8} candidates {int(row['candidates']):>9,}  "
            f"time-weighted {row['coverage_time_weighted']:.1f}%  "
            f"median per voyage {row['coverage_median_per_voyage']:.1f}%"
        )
    print(f"\nrun root: {run_root}")


if __name__ == "__main__":
    main()
