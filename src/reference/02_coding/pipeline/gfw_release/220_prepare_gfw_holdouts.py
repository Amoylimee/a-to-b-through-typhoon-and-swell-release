#!/usr/bin/env python3
"""Frozen GFW-only holdout manifest; user-approved hourly density adapter."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from step6_core import *
from weather_core.tropical_cyclone import CycloneTrackIndex, load_ibtracs_snapshot
from weather_core.geo import great_circle_km

SHIP_TYPES = ("container", "general_cargo", "product_chemical_tanker", "bulk_carrier",
              "ro_ro_vehicle", "other_retained", "crude_oil_tanker", "gas_tanker", "reefer")
DURATIONS = (2, 4, 6, 12, 24)


def select_shard(shard: str, out: str) -> dict:
    started = time.perf_counter()
    decision = json.loads(DECISION.read_text())
    weather_run = ROOT / decision["weather_run"]
    state = json.loads((weather_run / "shards" / shard / "status.json").read_text())
    track = state["sources"]["track"]
    if record(Path(track["path"])) != track:
        raise ValueError(f"GFW track changed: {shard}")
    summary = pd.read_parquet(state["sources"]["summary"]["path"], columns=["candidate_id", "shiptype_group"]).set_index("candidate_id")
    observed = pd.read_parquet(track["path"], columns=["candidate_id", "observation_time_utc", "lat", "lon", "source_provider"],
                              filters=[("track_evidence", "=", "observed_ais")])
    if not observed.source_provider.fillna("").str.startswith("gfw").all() \
            or observed.observation_time_utc.isna().any() \
            or not observed.observation_time_utc.dt.minute.eq(30).all():
        raise ValueError("holdout truth is not observed GFW at hh:30")
    # Reuse accepted manifest ranking and dense-window checks, with the explicit
    # user-approved 60-minute observation limit. No reconstructed truth points.
    shared = module("e006_holdout_manifest_kernel", E002 / "20_quantify_uncertainty/20_interpolation_holdout/10_prepare_holdout_samples.py")
    shared.DENSE_GAP_SECONDS = 3600
    cfg = json.loads((WEATHER / "config/2025_ten_minute_weather.json").read_text())
    fixes, _ = load_ibtracs_snapshot(Path(cfg["ibtracs_path"]), expected_sha256=cfg["ibtracs_sha256"],
                                   time_end_exclusive="2026-01-02T00:00:00Z")
    cyclone = CycloneTrackIndex(fixes, maximum_fix_gap_hours=3)
    centres = cyclone.positions_at_times(observed.observation_time_utc.drop_duplicates())
    by_time = {int(pd.Timestamp(t).value): g[["storm_lat", "storm_lon"]].to_numpy(float)
               for t, g in centres.groupby("target_time_utc", sort=False)}
    near = np.zeros(len(observed), dtype=bool)
    for t, indexes in observed.groupby("observation_time_utc", sort=False).indices.items():
        storms = by_time.get(int(pd.Timestamp(t).value))
        if storms is None:
            continue
        coords = observed.iloc[indexes][["lat", "lon"]].to_numpy(float)
        distance = np.minimum.reduce([great_circle_km(coords[:, 1], coords[:, 0], float(s[1]), float(s[0])) for s in storms])
        near[indexes] = distance <= 500+1e-9
    observed["near_tc500"] = near
    observed.sort_values(["candidate_id", "observation_time_utc"], kind="stable", inplace=True)
    pools, counts, storm_pool = {}, {}, []
    for cid, block in observed.groupby("candidate_id", sort=False, observed=True):
        cid = str(cid)
        shiptype = str(summary.loc[cid, "shiptype_group"])
        times = block.observation_time_utc.astype("int64").to_numpy()
        if len(times) < 4 or np.any(np.diff(times) <= 0):
            continue
        flags = block.near_tc500.to_numpy(bool)
        for duration in DURATIONS:
            lefts, rights = shared.eligible_windows(times, duration)
            if not len(lefts):
                continue
            key = (duration, shiptype)
            counts[key] = counts.get(key, 0)+1
            rng = np.random.default_rng(shared.stable_u64(SEED, cid, duration, "window"))
            choice = int(rng.integers(len(lefts)))
            left, right = int(lefts[choice]), int(rights[choice])
            row = {"candidate_id": cid, "source_shard": shard, "shiptype_group": shiptype,
                   "nominal_gap_hours": duration, "actual_gap_hours": (times[right]-times[left])/NS_HOUR,
                   "gap_start_utc": pd.Timestamp(times[left], tz="UTC").isoformat(),
                   "gap_end_utc": pd.Timestamp(times[right], tz="UTC").isoformat(),
                   "truth_observation_count": right-left-1, "sample_set": "gap_length_x_shiptype"}
            ranking = shared.stable_u64(SEED, "main", duration, shiptype, cid)
            bucket = pools.setdefault(key, [])
            bucket.append((ranking, row))
            if len(bucket) > 1000:
                bucket.sort(key=lambda x: x[0]); del bucket[500:]
            # Additional storm windows may differ from the randomly chosen main
            # window: rank only actual hidden observed-GFW points within 500 km.
            cumulative = np.r_[0, np.cumsum(flags)]
            eligible_storm = np.flatnonzero((cumulative[rights]-cumulative[lefts+1]) > 0)
            if len(eligible_storm):
                selected = int(eligible_storm[int(rng.integers(len(eligible_storm)))])
                l, r = int(lefts[selected]), int(rights[selected])
                storm_row = dict(row, sample_set="tc500km_additional", actual_gap_hours=(times[r]-times[l])/NS_HOUR,
                                 gap_start_utc=pd.Timestamp(times[l], tz="UTC").isoformat(),
                                 gap_end_utc=pd.Timestamp(times[r], tz="UTC").isoformat(), truth_observation_count=r-l-1)
                storm_pool.append((shared.stable_u64(SEED, "storm", duration, shiptype, cid), storm_row))
                if len(storm_pool) > 5000:
                    storm_pool.sort(key=lambda x: x[0]); del storm_pool[2500:]
    selected = [dict(row, priority=f"{rank:016x}") for bucket in pools.values() for rank, row in sorted(bucket)[:500]]
    selected += [dict(row, priority=f"{rank:016x}") for rank, row in sorted(storm_pool)[:2500]]
    folder = Path(out) / "shards"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_parquet(folder / f"{shard}.parquet", index=False)
    report = {"shard": shard, "observed_rows": len(observed), "selected_pool_rows": len(selected),
              "counts": [{"duration": k[0], "shiptype": k[1], "eligible_voyages": n} for k, n in counts.items()],
              "source_track": track, "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    write_json(folder / f"{shard}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    source = binding()
    out = inside_runs(args.out)
    out.mkdir(parents=True, exist_ok=False)
    state = {"state": "running", "workers": args.workers, "completed": [], "failures": [],
             "authorization": source["authorization"], "density_seconds": 3600,
             "durations_hours": list(DURATIONS), "seed": SEED}
    state.update(provenance([HERE / "step6_core.py", Path(__file__), DECISION,
                            E002 / "20_quantify_uncertainty/20_interpolation_holdout/10_prepare_holdout_samples.py"]))
    write_json(out / "status.json", state)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(select_shard, s, str(out)): s for s in SHARDS}
        for future in as_completed(pending):
            shard = pending[future]
            try:
                report = future.result(); state["completed"].append(shard)
                print(json.dumps({k: report[k] for k in ("shard", "observed_rows", "elapsed_seconds")}), flush=True)
            except Exception as exc:
                state["failures"].append({"shard": shard, "error": repr(exc)})
            write_json(out / "status.json", state)
    if state["failures"]:
        state["state"] = "failed"; write_json(out / "status.json", state)
        raise RuntimeError(state["failures"])
    pool = pd.concat([pd.read_parquet(out / "shards" / f"{s}.parquet") for s in SHARDS], ignore_index=True)
    main_pool = pool.loc[pool.sample_set.eq("gap_length_x_shiptype")]
    selected = main_pool.sort_values("priority").groupby(["nominal_gap_hours", "shiptype_group"], sort=False).head(500).copy()
    strata = selected.groupby(["nominal_gap_hours", "shiptype_group"]).size()
    if any(strata.get((d, t), 0) != 500 for d in DURATIONS for t in SHIP_TYPES):
        raise ValueError(f"required 500 observed-GFW windows per stratum not met: {strata}")
    chosen = set(selected[["candidate_id", "gap_start_utc", "gap_end_utc"]].itertuples(index=False, name=None))
    storm_rows, seen = [], set()
    for row in pool.loc[pool.sample_set.eq("tc500km_additional")].sort_values("priority").to_dict("records"):
        key = (row["candidate_id"], row["gap_start_utc"], row["gap_end_utc"])
        if key in chosen or row["candidate_id"] in seen:
            continue
        storm_rows.append(row); seen.add(row["candidate_id"])
        if len(storm_rows) == 500: break
    if len(storm_rows) != 500:
        raise ValueError("additional TC500 sample cannot supply 500 unique voyages")
    samples = pd.concat([selected, pd.DataFrame(storm_rows)], ignore_index=True).drop(columns="priority")
    samples["sample_id"] = [hashlib.sha256(f"{SEED}|{r.candidate_id}|{r.gap_start_utc}|{r.gap_end_utc}".encode()).hexdigest()[:24]
                            for r in samples.itertuples(index=False)]
    if samples.sample_id.duplicated().any(): raise ValueError("duplicate holdout sample")
    samples.sort_values(["sample_set", "nominal_gap_hours", "shiptype_group", "sample_id"]).to_parquet(out / "samples.parquet", index=False)
    state.update(state="completed", samples=len(samples), additional_storm_samples=len(storm_rows),
                 one_hour_layer="unestimable_no_interior_hourly_observation", output=record(out / "samples.parquet"))
    write_json(out / "status.json", state)
    print(json.dumps({"state": state["state"], "samples": len(samples)}))


if __name__ == "__main__":
    main()
