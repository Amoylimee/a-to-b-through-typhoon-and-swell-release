#!/usr/bin/env python3
"""E007 V1: reachable-area containment with the GFW time-stamp tolerance.

GFW points carry only their UTC hour and are stamped at hh:30, so each gap end may be up to
30 minutes earlier or later than recorded. Step 4 already allows for this in its speed check
("record interval + 1 h"); the missing-position bounds do not. For every reference AIS point in an
unresolved gap, this recomputes containment with each end widened by 30 minutes of travel at
25 kn, and reports how far outside the points that remain outside lie.

    python 27_containment_tolerance.py --run <20_ run> [--workers N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from e007_core import NS_HOUR, SPEED_KM_H, haversine_km

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
GFW_TRACKS = CODING / ".runs/2026-10-13_2006_E007-one-cell-track-view/shards"
# Default to the adopted population; an explicit environment selects another frozen run.
GFW_TRACKS = Path(os.environ.get("E007_TRACK_SHARDS", GFW_TRACKS))
TOLERANCE_H = 0.5


def one(shard_dir: str) -> pd.DataFrame:
    shard = Path(shard_dir)
    p = pd.read_parquet(shard / "points.parquet", columns=["cand", "tau", "k_lat", "k_lon", "status", "gap_class", "gap_hours", "contained"],
                        filters=[("status", "=", "unresolved")])
    cand = pd.read_parquet(shard / "candidates.parquet", columns=["cand", "candidate_id"])
    seg = pd.read_parquet(GFW_TRACKS / shard.name / "unresolved_segments.parquet",
                          columns=["candidate_id", "gap_start_ns", "gap_end_ns", "start_lat", "start_lon", "end_lat", "end_lon"])
    seg = seg.merge(cand, on="candidate_id").drop(columns="candidate_id").astype({"cand": np.int32}).sort_values("gap_start_ns")
    p = pd.merge_asof(p.sort_values("tau"), seg, left_on="tau", right_on="gap_start_ns", by="cand", direction="backward")
    if (p.tau >= p.gap_end_ns).any() or p.gap_start_ns.isna().any():
        raise ValueError(f"{shard.name}: unresolved point outside its gap")
    to_h = 1.0 / NS_HOUR
    d_a = haversine_km(p.start_lat, p.start_lon, p.k_lat, p.k_lon)
    d_b = haversine_km(p.k_lat, p.k_lon, p.end_lat, p.end_lon)
    reach_a = SPEED_KM_H * (p.tau - p.gap_start_ns) * to_h
    reach_b = SPEED_KM_H * (p.gap_end_ns - p.tau) * to_h
    slack = SPEED_KM_H * TOLERANCE_H
    excess = np.maximum(d_a - reach_a, d_b - reach_b)
    return pd.DataFrame({"gap_class": p.gap_class.astype(str), "gap_hours": p.gap_hours, "contained": p.contained.astype(bool),
                         "contained_with_tolerance": excess <= slack + 1e-6, "excess_km": np.maximum(excess, 0)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    run = args.run.resolve()
    with ProcessPoolExecutor(args.workers) as pool:
        frame = pd.concat(pool.map(one, [str(s) for s in sorted((run / "shards").iterdir())]), ignore_index=True)
    frame["length"] = pd.cut(frame.gap_hours, [0, 1.01, 6.01, 24.01, 48.01, np.inf], labels=["1 h", "2-6 h", "7-24 h", "25-48 h", ">48 h"])
    outside = frame.loc[~frame.contained]
    table = frame.groupby(["gap_class", "length"], observed=True).agg(
        points=("contained", "size"), contained=("contained", "mean"), contained_with_tolerance=("contained_with_tolerance", "mean")).reset_index()
    excess = outside.groupby(["gap_class", "length"], observed=True).excess_km.quantile([0.5, 0.9]).unstack()
    excess.columns = [f"outside_excess_km_p{int(q * 100)}" for q in excess.columns]
    table = table.merge(excess.reset_index(), on=["gap_class", "length"], how="left")
    target = run / "summary"
    target.mkdir(exist_ok=True)
    table.to_csv(target / "reachable_area_containment_with_time_tolerance.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(json.dumps({"points": len(frame), "contained": float(frame.contained.mean()),
                      "contained_with_tolerance": float(frame.contained_with_tolerance.mean())}, indent=2))


if __name__ == "__main__":
    main()
