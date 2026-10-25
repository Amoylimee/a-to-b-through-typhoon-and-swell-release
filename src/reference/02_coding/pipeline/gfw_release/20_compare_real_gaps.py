#!/usr/bin/env python3
"""E007 V1: GFW-version positions in real gaps against reference AIS observed AIS at the same instant.

For every reference AIS observed AIS point (from 10_extract_reference_ais_observed.py) that falls inside a
GFW-version voyage of the same vessel, take the GFW-version position at that exact instant by
linear interpolation between the two surrounding ten-minute cells. Each point inherits the
GFW gap it sits in: the time between the two GFW hourly points around it, the gap class, and
how far it is from the nearer GFW point. Inside unresolved gaps the GFW version has no
position; there the point is tested against the 25-kn reachable area between the known
positions at the gap ends.

The reachable-area test uses the GFW points at the gap ends, which are never farther from the
gap than the hourly positions used by the production bounds, so the containment rate here is
conservative. reference AIS is an internal validation reference only.

    python 20_compare_real_gaps.py --reference-ais-root <10_ run> --out <new .runs dir> [--shards ...] [--workers N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from e007_core import NS_HOUR, NS_MIN10, haversine_km, interpolate_position, reachable_contains

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
GFW_TRACKS = CODING / ".runs/2026-10-13_2006_E007-one-cell-track-view"
# Default to the adopted population; an explicit environment selects another frozen run.
TRACK_SHARDS = Path(os.environ.get("E007_TRACK_SHARDS", GFW_TRACKS / "shards"))
SHARDS = [f"shard_{i:04d}" for i in range(64)]
TRACK_COLUMNS = ["candidate_id", "mmsi", "utc", "lat", "lon", "position_source", "gap_id", "gap_class"]


def ns(values) -> np.ndarray:
    return pd.DatetimeIndex(values).asi8


def load_gfw(shard: str):
    root = TRACK_SHARDS / shard
    cand = pd.read_parquet(root / "candidate_summary.parquet",
                           columns=["candidate_id", "shiptype_group", "origin_city", "dest_city",
                                    "departure_time", "arrival_time"])
    cand = cand.sort_values("candidate_id").reset_index(drop=True)
    g = pq.read_table(root / "ten_minute_complete_candidate.parquet", columns=TRACK_COLUMNS).to_pandas()
    g["cand"] = pd.Index(cand.candidate_id).get_indexer(g.candidate_id).astype(np.int32)
    if (g.cand < 0).any():
        raise ValueError(f"{shard}: track rows without a candidate summary")
    g["m"] = g.mmsi.astype(np.int64)
    g["t"] = ns(g.utc)
    g = g.drop(columns=["candidate_id", "mmsi", "utc"])
    for column in ("position_source", "gap_class"):
        g[column] = g[column].astype("category")

    # Gap extents: the cells of one gap lie strictly between two GFW points ten minutes outside them.
    gp = g.loc[g.gap_id.notna()].groupby("gap_id", sort=False).agg(
        cand=("cand", "first"), first=("t", "min"), last=("t", "max"), cells=("t", "size"),
        gap_class=("gap_class", "first"))
    gp["start"] = gp["first"] - NS_MIN10
    gp["end"] = gp["last"] + NS_MIN10
    if not ((gp["end"] - gp["start"]) // NS_MIN10 - 1).eq(gp.cells).all():
        raise ValueError(f"{shard}: a gap has missing interior cells")
    # The track omits the departure and arrival instants, so gap-end positions come from the
    # step-4 record of unresolved gaps, which stores the GFW points on both sides.
    seg = pd.read_parquet(root / "unresolved_segments.parquet",
                          columns=["candidate_id", "gap_start_ns", "gap_end_ns", "start_lat", "start_lon", "end_lat", "end_lon"])
    seg["cand"] = pd.Index(cand.candidate_id).get_indexer(seg.candidate_id)
    seg = seg.drop(columns="candidate_id").rename(columns={"gap_start_ns": "start", "gap_end_ns": "end"})
    unresolved = gp.gap_class.astype(str).str.startswith("unresolved")
    if seg.duplicated(["cand", "start", "end"]).any() or len(seg) != int(unresolved.sum()):
        raise ValueError(f"{shard}: unresolved gaps and their step-4 records do not correspond one to one")
    gp = gp.reset_index().merge(seg, on=["cand", "start", "end"], how="left", validate="one_to_one").set_index("gap_id")
    if gp.loc[gp.gap_class.astype(str).str.startswith("unresolved"), "start_lat"].isna().any():
        raise ValueError(f"{shard}: an unresolved gap has no step-4 record")
    gp["hours"] = (gp["end"] - gp["start"]) / NS_HOUR
    return cand, g, gp


def run_shard(shard: str, reference_ais_root: str, out: str) -> dict:
    started = time.perf_counter()
    cand, g, gp = load_gfw(shard)
    receipt = {"shard": shard, "gfw_candidates": len(cand), "gfw_cells": len(g), "gfw_gaps": len(gp)}

    # One vessel's voyages do not overlap, but an arrival cell can equal the next departure cell;
    # such cells are ambiguous for a time match and are left out.
    dup = g.duplicated(["m", "t"], keep=False)
    receipt["gfw_cells_shared_by_two_voyages"] = int(dup.sum())
    cells = g.loc[~dup, ["m", "t", "cand", "lat", "lon", "position_source", "gap_id"]]

    files = sorted((Path(reference_ais_root) / "parts" / shard).glob("*.parquet"))
    k = pd.concat([pd.read_parquet(f, columns=["mmsi", "observation_time_utc", "lat", "lon"]) for f in files],
                  ignore_index=True)
    receipt["reference_ais_points"] = len(k)
    k = pd.DataFrame({"m": k.mmsi.astype(np.int64).to_numpy(), "tau": ns(k.observation_time_utc),
                      "k_lat": k.lat.to_numpy(float), "k_lon": k.lon.to_numpy(float)})
    k["c0"] = k.tau - k.tau % NS_MIN10
    k["c1"] = k.c0 + NS_MIN10
    a = cells.add_prefix("a_").rename(columns={"a_m": "m", "a_t": "c0"})
    b = cells.add_prefix("b_").rename(columns={"b_m": "m", "b_t": "c1"})
    k = k.merge(a, on=["m", "c0"], how="inner").merge(b, on=["m", "c1"], how="left")
    receipt["reference_ais_points_in_gfw_voyage_cells"] = len(k)
    on_cell = k.tau.eq(k.c0)
    same = on_cell | k.b_cand.eq(k.a_cand)
    receipt["reference_ais_points_crossing_voyage_boundary"] = int((~same).sum())
    k = k.loc[same].reset_index(drop=True)
    on_cell = k.tau.eq(k.c0).to_numpy()

    a_obs = k.a_position_source.eq("observed_ais").to_numpy()
    a_has, b_has = k.a_lat.notna().to_numpy(), k.b_lat.notna().to_numpy()
    resolved = np.where(on_cell, a_has, a_has & b_has)
    observed = on_cell & a_obs
    a_gap, b_gap = k.a_gap_id.to_numpy(object), k.b_gap_id.to_numpy(object)
    # A resolved point belongs to the gap of whichever surrounding cell is interpolated; an
    # unresolved point to the gap of the unknown cell.
    gap_id = np.where(resolved, np.where(k.a_gap_id.notna(), a_gap, b_gap), np.where(a_has, b_gap, a_gap))
    gap_id = np.where(observed, None, gap_id)
    gaps = gp.reindex(gap_id)

    fraction = ((k.tau - k.c0) / NS_MIN10).to_numpy(float)
    g_lat, g_lon = interpolate_position(k.a_lat, k.a_lon, k.b_lat.fillna(k.a_lat), k.b_lon.fillna(k.a_lon),
                                        np.where(on_cell, 0.0, fraction))
    g_lat, g_lon = np.where(resolved, g_lat, np.nan), np.where(resolved, g_lon, np.nan)
    status = np.where(observed, "observed", np.where(resolved, "resolved", "unresolved"))
    if np.isnan(gaps["hours"].to_numpy()[status != "observed"]).any():
        raise ValueError(f"{shard}: point inside a voyage without a gap")

    tau = k.tau.to_numpy()
    start, end = gaps["start"].to_numpy(), gaps["end"].to_numpy()
    offset_h = np.where(observed, 0.0, np.minimum(tau - start, end - tau) / NS_HOUR)
    unresolved = status == "unresolved"
    contained = np.full(len(k), np.nan)
    feasible = np.full(len(k), np.nan)
    if unresolved.any():
        idx = np.flatnonzero(unresolved)
        both = gaps["start_lat"].notna().to_numpy()[idx] & gaps["end_lat"].notna().to_numpy()[idx]
        idx = idx[both]
        inside, ok = reachable_contains(k.k_lat.to_numpy()[idx], k.k_lon.to_numpy()[idx], tau[idx],
                                        gaps["start_lat"].to_numpy()[idx], gaps["start_lon"].to_numpy()[idx], start[idx],
                                        gaps["end_lat"].to_numpy()[idx], gaps["end_lon"].to_numpy()[idx], end[idx])
        contained[idx], feasible[idx] = inside, ok
    receipt["unresolved_points_without_gap_end_position"] = int(unresolved.sum() - np.isfinite(contained).sum())

    cell_source = np.where(observed, "observed_ais",
                           np.where(k.a_position_source.eq("observed_ais"), k.b_position_source.astype(str),
                                    k.a_position_source.astype(str)))
    result = pd.DataFrame({
        "cand": k.a_cand.to_numpy(np.int32), "tau": tau,
        "k_lat": k.k_lat.to_numpy(np.float32), "k_lon": k.k_lon.to_numpy(np.float32),
        "g_lat": g_lat.astype(np.float32), "g_lon": g_lon.astype(np.float32),
        "error_km": haversine_km(k.k_lat, k.k_lon, g_lat, g_lon).astype(np.float32),
        "status": pd.Categorical(status, categories=["observed", "resolved", "unresolved"]),
        "gap_class": pd.Categorical(gaps["gap_class"].astype(object).to_numpy()),
        "cell_source": pd.Categorical(cell_source),
        "gap_hours": gaps["hours"].to_numpy(np.float32),
        "anchor_offset_h": offset_h.astype(np.float32),
        "contained": pd.array(np.where(np.isfinite(contained), contained == 1, None), dtype="boolean"),
        "pair_feasible": pd.array(np.where(np.isfinite(feasible), feasible == 1, None), dtype="boolean"),
    })
    target = Path(out) / "shards" / shard
    target.mkdir(parents=True, exist_ok=False)
    result.to_parquet(target / "points.parquet", index=False, row_group_size=2_000_000)
    cand.drop(columns=["departure_time", "arrival_time"]).rename_axis("cand").reset_index().to_parquet(
        target / "candidates.parquet", index=False)
    receipt.update({
        "points_written": len(result),
        "status_counts": result.status.value_counts().to_dict(),
        "contained_true": int(result.contained.sum()),
        "contained_tested": int(result.contained.notna().sum()),
        "seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
    })
    (target / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=int) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-ais-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", nargs="*", default=SHARDS)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    upstream = json.loads((args.reference_ais_root / "execution.json").read_text())
    if upstream.get("state") != "pass":
        raise ValueError("reference AIS observed-point extraction did not pass")
    out.mkdir(parents=True, exist_ok=False)
    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()
    execution = {"step": "E007 V1: GFW positions in real gaps vs reference AIS observed AIS", "state": "running",
                 "started_utc": pd.Timestamp.now(tz="UTC").isoformat(), "git_head": git("rev-parse", "HEAD"),
                 "git_dirty_experiment": bool(git("status", "--porcelain", "--", ".")),
                 "reference_ais_root": str(args.reference_ais_root.resolve().relative_to(CODING)), "gfw_tracks": str(TRACK_SHARDS),
                 "shards": args.shards, "workers": args.workers, "speed_cap_kn": 25.0}
    (out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    receipts, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_shard, s, str(args.reference_ais_root.resolve()), str(out)): s for s in args.shards}
        for future in as_completed(futures):
            try:
                receipts.append(future.result())
            except Exception as error:  # recorded, no retry
                failures.append({"shard": futures[future], "error": repr(error)})
    execution.update(state="pass" if not failures else "fail", finished_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                     shards_done=len(receipts), failures=failures,
                     points_written=int(sum(r["points_written"] for r in receipts)))
    (out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps({k: execution[k] for k in ["state", "shards_done", "points_written", "failures"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
