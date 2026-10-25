#!/usr/bin/env python3
"""E007 V1 weather: does the reconstructed GFW position give the same wave and cyclone result
as the true reference AIS position at the same hour?

From the 20_compare_real_gaps.py points, keep for each GFW voyage and UTC hour the reference AIS
observed point closest to the full hour, if within 5 minutes. Both the true (reference AIS) and the
GFW-version position at that instant are matched with the production code and frozen inputs:
ERA5 total significant wave height at the hour (nearest wet cell within 117.94 km) and
IBTrACS centres interpolated to the hour, 300/500 km. Production also places each full voyage
hour at its hh:00 position, so this is the comparison that the released hourly exposure uses.

    python 30_compare_real_gap_weather.py --points <20_ run> --out <new .runs dir> [--shards ...] [--workers N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
WEATHER = CODING / "pipeline/40_construct_variables/60_construct_voyage_weather_facts"
sys.path.insert(0, str(WEATHER))
from weather_core.era5_match import Era5MonthlyMatcher  # noqa: E402
from weather_core.tropical_cyclone import CycloneTrackIndex, load_ibtracs_snapshot  # noqa: E402

CONFIG = WEATHER / "config/2025_ten_minute_weather.json"
NS_HOUR = 3_600_000_000_000
NEAR_HOUR_NS = 5 * 60 * 1_000_000_000
SHARDS = [f"shard_{i:04d}" for i in range(64)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hourly_sample(points_dir: Path) -> pd.DataFrame:
    p = pd.read_parquet(points_dir / "points.parquet",
                        columns=["cand", "tau", "k_lat", "k_lon", "g_lat", "g_lon", "status", "gap_hours",
                                 "anchor_offset_h", "error_km"])
    p = p.loc[p.status.ne("unresolved")]
    hour = (p.tau + NS_HOUR // 2) // NS_HOUR * NS_HOUR
    lag = (p.tau - hour).abs()
    p = p.assign(hour=hour, lag=lag).loc[lag.le(NEAR_HOUR_NS)]
    p = p.sort_values(["cand", "hour", "lag"]).drop_duplicates(["cand", "hour"], keep="first")
    return p.drop(columns="lag").reset_index(drop=True)


def run_shard(shard: str, points_root: str, out: str, config: dict) -> dict:
    started = time.perf_counter()
    sample = hourly_sample(Path(points_root) / "shards" / shard)
    n = len(sample)
    both = pd.DataFrame({
        "hour_anchor_utc": pd.to_datetime(np.r_[sample.hour.to_numpy(), sample.hour.to_numpy()], utc=True),
        "lat": np.r_[sample.k_lat.to_numpy(float), sample.g_lat.to_numpy(float)],
        "lon": np.r_[sample.k_lon.to_numpy(float), sample.g_lon.to_numpy(float)],
    })
    both["position_time_utc"] = both.hour_anchor_utc
    swh = np.full(2 * n, np.nan)
    fixes, _ = load_ibtracs_snapshot(Path(config["ibtracs_path"]), expected_sha256=config["ibtracs_sha256"],
                                     time_end_exclusive="2026-01-02T00:00:00Z")
    cyclone = CycloneTrackIndex(fixes, maximum_fix_gap_hours=config["maximum_tc_fix_gap_hours"])
    dist = np.full(2 * n, np.nan)
    in300 = np.zeros(2 * n, dtype=bool)
    in500 = np.zeros(2 * n, dtype=bool)
    months = both.hour_anchor_utc.dt.month.to_numpy()
    for month in np.unique(months):
        rows = np.flatnonzero(months == month)
        wave_path = Path(config["wave_directory"]) / config["wave_filename_template"].format(month=int(month))
        with xr.open_dataset(wave_path) as dataset:
            if dataset.swh.attrs.get("units") != "m":
                raise ValueError("wave units are not metres")
        with Era5MonthlyMatcher([wave_path], maximum_wet_cell_snap_km=config["maximum_wet_cell_snap_km"],
                                cached_hours=2) as matcher:
            matcher.preload_swh()
            matched = matcher.match(both.iloc[rows][["hour_anchor_utc", "lat", "lon"]])
        swh[rows] = matched.swh_m.to_numpy(float)
        for offset in range(0, len(rows), 50_000):
            chunk = rows[offset:offset + 50_000]
            tc = cyclone.match_positions(both.iloc[chunk], time_col="position_time_utc", lat_col="lat", lon_col="lon")
            dist[chunk] = tc.tc_min_center_distance_km.to_numpy(float)
            in300[chunk] = tc.tc_within_300km.to_numpy(bool)
            in500[chunk] = tc.tc_within_500km.to_numpy(bool)
    if not config["ibtracs_full_year_coverage_verified"]:
        raise ValueError("cyclone snapshot coverage is not verified for 2025")
    result = sample.drop(columns=["tau", "k_lat", "k_lon", "g_lat", "g_lon"]).assign(
        swh_true=swh[:n].astype(np.float32), swh_gfw=swh[n:].astype(np.float32),
        tc_km_true=dist[:n].astype(np.float32), tc_km_gfw=dist[n:].astype(np.float32),
        tc300_true=in300[:n], tc300_gfw=in300[n:], tc500_true=in500[:n], tc500_gfw=in500[n:])
    target = Path(out) / "shards" / shard
    target.mkdir(parents=True, exist_ok=False)
    result.to_parquet(target / "hourly_weather_pairs.parquet", index=False)
    receipt = {"shard": shard, "hours": n, "wave_both_matched": int((np.isfinite(swh[:n]) & np.isfinite(swh[n:])).sum()),
               "tc300_either": int((in300[:n] | in300[n:]).sum()), "tc500_either": int((in500[:n] | in500[n:]).sum()),
               "seconds": round(time.perf_counter() - started, 1),
               "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)}
    (target / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", nargs="*", default=SHARDS)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    if json.loads((args.points / "execution.json").read_text()).get("state") != "pass":
        raise ValueError("real-gap comparison did not pass")
    config = json.loads(CONFIG.read_text())
    out.mkdir(parents=True, exist_ok=False)
    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()
    execution = {"step": "E007 V1 weather: true vs GFW-version position at the same hour", "state": "running",
                 "started_utc": pd.Timestamp.now(tz="UTC").isoformat(), "git_head": git("rev-parse", "HEAD"),
                 "git_dirty_experiment": bool(git("status", "--porcelain", "--", ".")),
                 "points_run": str(args.points.resolve().relative_to(CODING)), "weather_config_sha256": sha256(CONFIG),
                 "near_hour_minutes": 5, "shards": args.shards, "workers": args.workers}
    (out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    receipts, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_shard, s, str(args.points.resolve()), str(out), config): s for s in args.shards}
        for future in as_completed(futures):
            try:
                receipts.append(future.result())
            except Exception as error:  # recorded, no retry
                failures.append({"shard": futures[future], "error": repr(error)})
    execution.update(state="pass" if not failures else "fail", finished_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                     shards_done=len(receipts), failures=failures, hours=int(sum(r["hours"] for r in receipts)))
    (out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps({k: execution[k] for k in ["state", "shards_done", "hours", "failures"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
