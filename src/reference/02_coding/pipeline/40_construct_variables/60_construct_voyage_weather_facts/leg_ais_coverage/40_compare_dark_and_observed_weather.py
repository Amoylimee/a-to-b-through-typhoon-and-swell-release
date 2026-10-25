#!/usr/bin/env python3
"""Ask whether the hours we cannot see carry different weather from the ones we can.

Read-only diagnostic.  It writes no product root and computes no accepted value.

The paper's third link is *rerouting exposes ships to worse seas*, so the wave
and cyclone variables are the mechanism rather than a control.  `30_bound_...`
showed the unseen hours can be bounded for waves and mostly cannot for cyclones.
That is about precision.  This script asks the prior question, which is about
bias: if a voyage's unseen hours are systematically calmer or rougher than its
seen ones, then a mean taken over seen hours alone is not an estimate of the
voyage's weather -- it is an estimate of the weather where the transponder
happened to be working.

This helper compares observed and unobserved hours within the same candidate
voyage.  It does not classify voyages by route or infer a behavioural response.

Observed hours are rebuilt with the accepted rule and the accepted code: an hour
counts when `hour_segments.interpolate_positions` can place the vessel inside one
continuity segment, which is to say when the surrounding AIS gap is at most four
hours.  Positions carry the accepted 117.94 km wet-cell snap, because here the
position is known and only the grid is coarse -- the opposite of the dark-hour
case, where a dry cell means somewhere the ship could not have been.

Rates rather than counts are reported throughout: a per-candidate hour cap keeps
one 40-day voyage from dominating, which biases counts but leaves rates intact.

    python3 40_compare_dark_and_observed_weather.py --bounds-run <dir> \\
        --coverage-run <dir> --out <dir>

Environment: sinbue_env (Python 3.10)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODING_ROOT = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

WEATHER_STAGE_ROOT = HERE.parent
if str(WEATHER_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(WEATHER_STAGE_ROOT))
ENVIRONMENT_STAGE_ROOT = CODING_ROOT / "pipeline/00_environment_and_config"
if str(ENVIRONMENT_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_STAGE_ROOT))

from weather_core.era5_match import Era5MonthlyMatcher  # noqa: E402
from weather_core.hour_segments import interpolate_positions  # noqa: E402
from weather_core.tropical_cyclone import (  # noqa: E402
    CycloneTrackIndex,
    load_ibtracs_snapshot,
)

from ais_identity import (  # noqa: E402
    deduplicate_vessel_ais,
    normalise_required_identifier,
    stable_vessel_buckets,
    validate_ais_physical_identity,
)
from accepted_candidate_root import load_accepted_2025_candidates  # noqa: E402
from weather_assets import (  # noqa: E402
    resolve_monthly_era5_assets,
    validate_month_time_grid,
)

MAXIMUM_INTERPOLATION_GAP_HOURS = 4.0
HIGH_WAVE_THRESHOLD_M = 4.0
TC_RADIUS_500_KM = 500.0
TC_RADIUS_300_KM = 300.0
TC_MAXIMUM_FIX_GAP_HOURS = 3.0
# A known position with a coarse grid: the accepted snap applies, unlike the
# dark-hour case where a dry cell is simply somewhere the ship could not be.
ACCEPTED_WET_CELL_SNAP_KM = 117.94
MAX_HOURS_PER_CANDIDATE = 240

WEATHER_CONFIG = json.loads(
    (WEATHER_STAGE_ROOT / "config/2025_ten_minute_weather.json").read_text(encoding="utf-8")
)
IBTRACS_PATH = Path(WEATHER_CONFIG["ibtracs_path"])
IBTRACS_SHA256 = WEATHER_CONFIG["ibtracs_sha256"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def observed_positions(
    bucket_dir: Path, candidates: pd.DataFrame, month_starts: pd.DatetimeIndex
) -> pd.DataFrame:
    """Hourly positions the accepted rule can actually place, for one bucket."""

    parts = sorted(bucket_dir.glob("*.feather"))
    if not parts:
        return pd.DataFrame()
    points = pd.concat(
        [feather.read_table(p, memory_map=True).to_pandas() for p in parts],
        ignore_index=True,
    )
    identity = candidates[["mmsi", "vessel_uid_clean"]].drop_duplicates()
    points, _ = validate_ais_physical_identity(points, identity)
    points, _ = deduplicate_vessel_ais(points)
    if points.empty:
        return pd.DataFrame()

    months = {p.strftime("%Y-%m") for p in month_starts}
    by_vessel = {k: g for k, g in points.groupby("mmsi", sort=False, observed=True)}
    frames: list[pd.DataFrame] = []
    for mmsi, group in candidates.groupby("mmsi", sort=False, observed=True):
        vessel = by_vessel.get(mmsi)
        if vessel is None or vessel.empty:
            continue
        track = vessel[["UTC", "lat", "lon"]].copy()
        track["vessel_key"] = str(mmsi)
        for row in group.itertuples(index=False):
            anchors = pd.date_range(
                pd.Timestamp(row.departure_time).ceil("h"),
                pd.Timestamp(row.arrival_time).floor("h"),
                freq="h",
            )
            anchors = anchors[anchors.strftime("%Y-%m").isin(months)]
            if anchors.empty:
                continue
            if len(anchors) > MAX_HOURS_PER_CANDIDATE:
                anchors = anchors[
                    np.linspace(0, len(anchors) - 1, MAX_HOURS_PER_CANDIDATE).astype(int)
                ]
            placed = interpolate_positions(
                track, pd.Series(anchors), MAXIMUM_INTERPOLATION_GAP_HOURS
            )
            placed = placed[placed["lat"].notna() & placed["lon"].notna()]
            if placed.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "candidate_id": row.candidate_id,
                        "hour_anchor_utc": placed["position_time_utc"].to_numpy(),
                        "lat": placed["lat"].to_numpy(),
                        "lon": placed["lon"].to_numpy(),
                    }
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds-run", type=Path, required=True)
    parser.add_argument("--coverage-run", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.workers > 12:
        raise SystemExit("CONTRACT caps this project at 12 workers")
    args.out.mkdir(parents=True, exist_ok=True)

    dark = pd.read_parquet(args.bounds_run / "20_dark_hour_exposure_bounds.parquet")
    dark = dark[~dark["implied_exceeds_cap"]]
    months = sorted(dark["month"].unique())
    print(f"months from the bounds run: {months}")

    bucket_root = args.coverage_run / "vessel_buckets"
    if not (bucket_root / "_shuffle_complete.json").exists():
        raise SystemExit(f"no completed shuffle at {bucket_root}")

    fixes, _ = load_ibtracs_snapshot(IBTRACS_PATH, expected_sha256=IBTRACS_SHA256)
    cyclones = CycloneTrackIndex(
        fixes, maximum_fix_gap_hours=TC_MAXIMUM_FIX_GAP_HOURS
    )

    years = sorted({int(m.split("-")[0]) for m in months})
    if years != [2025]:
        raise SystemExit("E001 weather comparison reads the accepted 2025 candidate root only")
    candidates = load_accepted_2025_candidates(
        ("candidate_id", "mmsi", "vessel_uid_clean", "departure_time", "arrival_time")
    )
    # Only the voyages the bounds run actually sampled, so the two sides are the
    # same ships over the same windows and the comparison is within-voyage.
    wanted = set(dark["candidate_id"].unique())
    candidates = candidates[candidates["candidate_id"].isin(wanted)].reset_index(
        drop=True
    )
    candidates["mmsi"] = normalise_required_identifier(candidates["mmsi"], "mmsi")
    candidates["vessel_uid_clean"] = normalise_required_identifier(
        candidates["vessel_uid_clean"], "vessel_uid_clean"
    )
    candidates["vessel_bucket"] = stable_vessel_buckets(candidates["mmsi"], args.buckets)
    print(f"voyages carried over from the bounds run: {len(candidates):,}")

    month_starts = pd.DatetimeIndex([pd.Timestamp(f"{m}-01", tz="UTC") for m in months])
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                observed_positions,
                bucket_root / f"bucket_{int(bucket):04d}",
                group.reset_index(drop=True),
                month_starts,
            ): bucket
            for bucket, group in candidates.groupby("vessel_bucket", sort=True)
        }
        for n, future in enumerate(as_completed(futures), start=1):
            got = future.result()
            if not got.empty:
                frames.append(got)
            if n % 16 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} buckets")
    positions = pd.concat(frames, ignore_index=True)
    positions["month"] = positions["hour_anchor_utc"].dt.strftime("%Y-%m")
    print(f"observed hours placed by the accepted rule: {len(positions):,}")

    matched = []
    for month in months:
        block = positions[positions["month"] == month]
        if block.empty:
            continue
        year, mon = (int(x) for x in month.split("-"))
        assets = resolve_monthly_era5_assets(year=year, month=mon)
        with Era5MonthlyMatcher(
            assets, maximum_wet_cell_snap_km=ACCEPTED_WET_CELL_SNAP_KM
        ) as matcher:
            validate_month_time_grid(matcher, year, mon)
            got = matcher.match(block.reset_index(drop=True))
        tc = cyclones.match_positions(
            got.rename(
                columns={
                    "hour_anchor_utc": "position_time_utc",
                    "lat": "latitude",
                    "lon": "longitude",
                }
            )[["position_time_utc", "latitude", "longitude"]],
            radius_500_km=TC_RADIUS_500_KM,
            radius_300_km=TC_RADIUS_300_KM,
        )
        got["within_500"] = tc["tc_within_500km"].to_numpy()
        got["within_300"] = tc["tc_within_300km"].to_numpy()
        matched.append(got)
        print(f"{month}: {len(got):,} observed hours matched")

    seen = pd.concat(matched, ignore_index=True)
    seen = seen[seen["swh_m"].notna()]

    seen["month"] = seen["hour_anchor_utc"].dt.strftime("%Y-%m")
    per_candidate = (
        seen.groupby(["candidate_id", "month"], sort=False)
        .agg(
            observed_hours=("swh_m", "size"),
            observed_swh_mean=("swh_m", "mean"),
            observed_high_wave_share=("swh_m", lambda s: float((s >= HIGH_WAVE_THRESHOLD_M).mean())),
            observed_tc500_share=("within_500", "mean"),
            observed_tc300_share=("within_300", "mean"),
        )
        .reset_index()
    )

    # One row per voyage, both sides of the same window side by side.
    dark_by_candidate = (
        dark.groupby(["candidate_id", "month"], as_index=False)
        .agg(
            dark_hours_bounded=("hours_bounded", "sum"),
            dark_swh_lo=("swh_lo_mean", "mean"),
            dark_swh_hi=("swh_hi_mean", "mean"),
            dark_high_wave_lo=("high_wave_hours_lo", "sum"),
            dark_high_wave_hi=("high_wave_hours_hi", "sum"),
            dark_tc500_lo=("tc500_hours_lo", "sum"),
            dark_tc500_hi=("tc500_hours_hi", "sum"),
            dark_tc_undetermined=("tc_undetermined_hours", "sum"),
        )
    )
    joined = dark_by_candidate.merge(per_candidate, on="candidate_id", how="inner")
    joined["dark_swh_mid"] = (joined["dark_swh_lo"] + joined["dark_swh_hi"]) / 2.0
    joined["swh_gap_mid"] = joined["dark_swh_mid"] - joined["observed_swh_mean"]
    joined["swh_gap_lo"] = joined["dark_swh_lo"] - joined["observed_swh_mean"]
    joined["swh_gap_hi"] = joined["dark_swh_hi"] - joined["observed_swh_mean"]
    joined.to_parquet(args.out / "30_dark_vs_observed_by_voyage.parquet", index=False)

    by_month = (
        joined.groupby("month", as_index=False)
        .agg(
            voyages=("candidate_id", "size"),
            observed_hours=("observed_hours", "sum"),
            dark_hours=("dark_hours_bounded", "sum"),
            observed_swh=("observed_swh_mean", "mean"),
            dark_swh_lo=("dark_swh_lo", "mean"),
            dark_swh_hi=("dark_swh_hi", "mean"),
            swh_gap_lo=("swh_gap_lo", "mean"),
            swh_gap_hi=("swh_gap_hi", "mean"),
        )
    )
    by_month.to_csv(args.out / "31_dark_vs_observed_by_month.csv", index=False)

    # Cyclone exposure is rare, so its denominator is the exposure itself: of
    # every hour a storm could have been within 500 km, how many are settled?
    tc_total_lo = float(joined["dark_tc500_lo"].sum())
    tc_total_hi = float(joined["dark_tc500_hi"].sum())
    tc_undetermined = float(joined["dark_tc_undetermined"].sum())
    observed_tc_hours = float((seen["within_500"]).sum())
    tc_summary = {
        "observed_tc500_hours": observed_tc_hours,
        "dark_tc500_hours_certain": tc_total_lo,
        "dark_tc500_hours_possible": tc_total_hi,
        "dark_tc500_hours_undetermined": tc_undetermined,
        "undetermined_share_of_all_possible_tc_hours": (
            tc_undetermined / (observed_tc_hours + tc_total_hi)
            if (observed_tc_hours + tc_total_hi) > 0
            else None
        ),
    }

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": sha256_file(Path(__file__)),
        "read_only": True,
        "products_modified": 0,
        "bounds_run": str(args.bounds_run),
        "months": months,
        "voyages": int(len(joined)),
        "observed_hours_matched": int(len(seen)),
        "maximum_interpolation_gap_hours": MAXIMUM_INTERPOLATION_GAP_HOURS,
        "wet_cell_snap_km": ACCEPTED_WET_CELL_SNAP_KM,
        "max_hours_per_candidate": MAX_HOURS_PER_CANDIDATE,
        "tropical_cyclone": tc_summary,
    }
    (args.out / "compare_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n--- seen versus unseen wave height, by month (metres) ---")
    print(by_month.round(3).to_string(index=False))
    print("\n--- cyclones ---")
    for k, v in tc_summary.items():
        print(f"  {k}: {v}")
    print(f"\noutputs: {args.out}")


if __name__ == "__main__":
    main()
