#!/usr/bin/env python3
"""Compare reconstructed points with held-out GFW cells in weather units."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[4]
TRACKS = CODING / "data/processed/products/companion_tracks/2025/run/shards"
WEATHER = CODING / "pipeline/40_construct_variables/60_construct_voyage_weather_facts"
sys.path.insert(0, str(WEATHER))
sys.path.insert(0, str(WEATHER / "leg_ais_coverage"))
from weather_assets import resolve_monthly_era5_assets, validate_month_time_grid  # noqa: E402
from weather_core.era5_match import Era5MonthlyMatcher  # noqa: E402
from weather_core.tropical_cyclone import CycloneTrackIndex, load_ibtracs_snapshot  # noqa: E402

CONFIG = json.loads((WEATHER / "config/2025_ten_minute_weather.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_for_shard(path: Path, compared_path: Path) -> pd.DataFrame:
    shard = path.stem
    cells = pd.read_parquet(path)
    compared = pd.read_parquet(compared_path, columns=["candidate_id", "hour_utc",
                                                    "comparison_class", "min_centre_distance_km"])
    if cells.duplicated(["candidate_id", "hour_utc"]).any() or compared.duplicated(["candidate_id", "hour_utc"]).any():
        raise ValueError(f"{shard}: GFW hour key repeated")
    cells = cells.merge(compared, on=["candidate_id", "hour_utc"], how="inner", validate="one_to_one")
    cells = cells.loc[cells.comparison_class.eq("gfw_supported_interpolation")].copy()
    if cells.empty:
        return cells
    cells["row_id"] = np.arange(len(cells), dtype=np.int64)
    tracks = pd.read_parquet(TRACKS / shard / "ten_minute_complete_candidate.parquet",
                             columns=["candidate_id", "utc", "lat", "lon", "track_evidence"])
    tracks = tracks.loc[tracks.candidate_id.isin(cells.candidate_id.unique())].copy()
    tracks["hour_utc"] = pd.to_datetime(tracks.utc, utc=True).dt.floor("h")
    joined = cells[["row_id", "candidate_id", "hour_utc", "lat", "lon"]].merge(
        tracks, on=["candidate_id", "hour_utc"], how="left", validate="one_to_many",
        suffixes=("_gfw", "_track"))
    valid = joined.lat_track.notna() & joined.lon_track.notna()
    lat_a = np.radians(joined.loc[valid, "lat_gfw"].to_numpy(dtype=float))
    lat_b = np.radians(joined.loc[valid, "lat_track"].to_numpy(dtype=float))
    dlat = lat_b - lat_a
    dlon = np.radians(joined.loc[valid, "lon_track"].to_numpy(dtype=float)
                      - joined.loc[valid, "lon_gfw"].to_numpy(dtype=float))
    hav = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    joined.loc[valid, "distance_km"] = 2 * 6371.0088 * np.arctan2(
        np.sqrt(hav), np.sqrt(np.maximum(0, 1 - hav)))
    nearest = joined.loc[valid].sort_values(["row_id", "distance_km", "utc"],
                                            kind="stable").drop_duplicates("row_id")
    if len(nearest) != len(cells):
        raise ValueError(f"{shard}: primary GFW hour lacks a positioned point")
    nearest = nearest.merge(cells[["row_id", "shiptype_group", "source_gap_hours",
                                   "min_centre_distance_km"]], on="row_id", validate="one_to_one")
    if not np.allclose(nearest.distance_km, nearest.min_centre_distance_km, atol=1e-5):
        raise ValueError(f"{shard}: nearest distance differs from frozen diagnostic")
    return nearest[["candidate_id", "hour_utc", "utc", "lat_gfw", "lon_gfw",
                    "lat_track", "lon_track", "distance_km", "shiptype_group", "source_gap_hours"]]


def compare_month(month: str, points: pd.DataFrame, cyclones: CycloneTrackIndex) -> pd.DataFrame:
    points = points.reset_index(drop=True).copy()
    points["pair_id"] = np.arange(len(points), dtype=np.int64)
    gfw = points[["pair_id", "hour_utc", "utc", "lat_gfw", "lon_gfw"]].rename(
        columns={"lat_gfw": "lat", "lon_gfw": "lon"}).assign(role="gfw")
    track = points[["pair_id", "hour_utc", "utc", "lat_track", "lon_track"]].rename(
        columns={"lat_track": "lat", "lon_track": "lon"}).assign(role="reconstructed")
    locations = pd.concat([gfw, track], ignore_index=True)
    locations = locations.rename(columns={"hour_utc": "hour_anchor_utc", "utc": "position_time_utc"})
    year, mon = (int(part) for part in month.split("-"))
    matcher = Era5MonthlyMatcher(resolve_monthly_era5_assets(year=year, month=mon),
                                 maximum_wet_cell_snap_km=float(CONFIG["maximum_wet_cell_snap_km"]),
                                 cached_hours=2)
    try:
        validate_month_time_grid(matcher, year, mon)
        matcher.preload_swh()
        waves = matcher.match(locations[["hour_anchor_utc", "lat", "lon"]])
        locations["swh_m"] = waves.swh_m.to_numpy()
    finally:
        matcher.close()
    tc_parts = []
    for start in range(0, len(locations), 50_000):
        tc_parts.append(cyclones.match_positions(
            locations.iloc[start:start + 50_000][["pair_id", "position_time_utc", "lat", "lon"]],
            time_col="position_time_utc", lat_col="lat", lon_col="lon"))
    tc = pd.concat(tc_parts, ignore_index=True)
    locations["tc300"] = tc.tc_within_300km.to_numpy()
    locations["tc500"] = tc.tc_within_500km.to_numpy()
    pivot = locations.pivot(index="pair_id", columns="role", values=["swh_m", "tc300", "tc500"])
    pivot.columns = [f"{metric}_{role}" for metric, role in pivot.columns]
    result = pd.concat([points, pivot.reset_index(drop=True)], axis=1)
    for role in ("gfw", "reconstructed"):
        result[f"swh_m_{role}"] = pd.to_numeric(result[f"swh_m_{role}"], errors="coerce").astype(float)
    for radius in (300, 500):
        g = result[f"tc{radius}_gfw"].astype("boolean")
        r = result[f"tc{radius}_reconstructed"].astype("boolean")
        result[f"tc{radius}_flip"] = g.ne(r).where(g.notna() & r.notna())
        result[f"tc{radius}_either_positive"] = (g | r).where(g.notna() & r.notna())
    result["swh_signed_difference_m"] = result.swh_m_reconstructed - result.swh_m_gfw
    result["swh_absolute_difference_m"] = result.swh_signed_difference_m.abs()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfw-run", type=Path, required=True)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if ".runs" not in args.internal.resolve().parts:
        parser.error("per-hour coordinates must stay in .runs")
    args.internal.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted((args.gfw_run / "heldout_by_shard").glob("shard_*.parquet"))
    if len(files) != 99:
        raise ValueError(f"GFW source shards changed: {len(files)}")
    parts = []
    for index, path in enumerate(files, 1):
        nearest = nearest_for_shard(path, args.gfw_run / "compared_by_shard" / path.name)
        parts.append(nearest)
        if index % 20 == 0:
            print(f"nearest GFW points {index}/{len(files)}", flush=True)
    points = pd.concat(parts, ignore_index=True)
    if len(points) != 2_293_663 or points.duplicated(["candidate_id", "hour_utc"]).any():
        raise ValueError("GFW primary comparison population changed")
    points.to_parquet(args.internal / "nearest_gfw_track_hours.parquet", index=False)
    fixes, ibtracs_audit = load_ibtracs_snapshot(
        CONFIG["ibtracs_path"], expected_sha256=CONFIG["ibtracs_sha256"],
        time_end_exclusive="2026-01-02T00:00:00Z")
    cyclones = CycloneTrackIndex(fixes, maximum_fix_gap_hours=CONFIG["maximum_tc_fix_gap_hours"])
    reports = []
    for month, block in points.groupby(points.hour_utc.dt.strftime("%Y-%m"), sort=True):
        compared = compare_month(month, block, cyclones)
        compared.to_parquet(args.internal / f"gfw_exposure_{month}.parquet", index=False)
        report = {"month": month, "hours": len(compared)}
        for metric in ("distance_km", "swh_absolute_difference_m"):
            data = pd.to_numeric(compared[metric], errors="coerce").dropna()
            report[f"{metric}_valid"] = len(data)
            for percentile in (50, 75, 90, 95, 99):
                report[f"{metric}_p{percentile}"] = float(np.quantile(data, percentile / 100)) if len(data) else None
        for radius in (300, 500):
            flip = compared[f"tc{radius}_flip"].dropna().astype(bool)
            near = compared[f"tc{radius}_either_positive"].fillna(False).astype(bool)
            report[f"tc{radius}_valid_hours"] = len(flip)
            report[f"tc{radius}_flip_hours"] = int(flip.sum())
            report[f"tc{radius}_either_positive_hours"] = int(near.sum())
            report[f"tc{radius}_flip_pct_all"] = 100 * float(flip.mean()) if len(flip) else None
            report[f"tc{radius}_flip_pct_either_positive"] = (
                100 * float(compared.loc[near, f"tc{radius}_flip"].astype(bool).mean()) if near.any() else None)
        reports.append(report)
        print(f"GFW weather exposure {month}: {len(compared):,} hours", flush=True)
    table = pd.DataFrame(reports)
    table.to_csv(args.out / "04_gfw_exposure_by_month.csv", index=False)
    all_results = pd.concat([pd.read_parquet(args.internal / f"gfw_exposure_{month}.parquet")
                             for month in table.month], ignore_index=True)
    summary = {"status": "formal_e002_gfw_diagnostic_pending_review",
               "hours": len(all_results), "source_gfw_run": str(args.gfw_run),
               "source_gfw_manifest_sha256": sha256(args.gfw_run / "climate_derived_source_manifest.json"),
               "local_gfw_raw_summary_sha256": sha256(CODING / "data/legacy_inputs/gfw_presence_responses_2025/summary.json"),
               "ibtracs_sha256": CONFIG["ibtracs_sha256"],
               "ibtracs_audit": ibtracs_audit,
               "meaning": "GFW 0.01-degree hourly cell centre and nearest reconstructed 10-minute point within the same UTC hour; both locations matched to frozen ERA5 and cyclone snapshot at the same ten-minute time. Same GFW source means this is not independent ground truth."}
    for metric in ("distance_km", "swh_absolute_difference_m"):
        data = pd.to_numeric(all_results[metric], errors="coerce").dropna()
        summary[metric] = {"valid_hours": len(data), **{
            f"p{p}": float(np.quantile(data, p / 100)) if len(data) else None
            for p in (50, 75, 90, 95, 99)}}
    for radius in (300, 500):
        flip = all_results[f"tc{radius}_flip"].dropna().astype(bool)
        near = all_results[f"tc{radius}_either_positive"].fillna(False).astype(bool)
        summary[f"tc{radius}"] = {"valid_hours": len(flip), "flip_hours": int(flip.sum()),
                                  "flip_pct_all": 100 * float(flip.mean()) if len(flip) else None,
                                  "either_positive_hours": int(near.sum()),
                                  "flip_pct_either_positive": 100 * float(all_results.loc[near, f"tc{radius}_flip"].astype(bool).mean()) if near.any() else None}
    (args.out / "04_gfw_exposure_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("hours", "distance_km", "swh_absolute_difference_m", "tc300", "tc500")}, indent=2))


if __name__ == "__main__":
    main()
