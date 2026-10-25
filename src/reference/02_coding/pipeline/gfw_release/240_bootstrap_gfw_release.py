#!/usr/bin/env python3
"""Ship-cluster bootstrap of the GFW release estimands.

Wave mode resamples the mean over wave-matched hours, the primary wave field since the
E007 rule revision (`config/step6_rule_revision.json`); it no longer clips to the bounds.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from step6_core import *

BASE_COLUMNS = ["wave_ge4m_matched_hours", "wave_matched_hours",
                "tc300_positive_snapshot_hours", "tc500_positive_snapshot_hours", "voyage_hours"]
WAVE_COLUMNS = ["wave_integral_m_hours", "wave_matched_hours"]


def bootstrap_group(key, values, mode, seed_kernel):
    """Same accepted per-key RNG, draws, 100-row batches and linear percentiles."""
    seed = seed_kernel.stable_seed(tuple(key))
    row = dict(zip(KEY, key))
    row.update(ships=len(values), bootstrap_seed=seed, bootstrap_replicates=1000)
    names = (["share_wave_hours_ge4m", "tc300_at_least_pct", "tc500_at_least_pct"]
             if mode == "base" else ["mean_swh_matched_m"])
    def estimates(sums):
        sums = np.asarray(sums)
        if mode == "wave":
            return np.divide(sums[..., 0], sums[..., 1], out=np.full(sums.shape[:-1], np.nan),
                             where=sums[..., 1] > 0)[..., None]
        return np.stack([np.divide(sums[..., 0], sums[..., 1], out=np.full(sums.shape[:-1], np.nan), where=sums[..., 1] > 0),
                         100*sums[..., 2]/sums[..., 4], 100*sums[..., 3]/sums[..., 4]], axis=-1)
    point = estimates(values.sum(axis=0))
    boot = np.empty((1000, len(names)))
    if len(values) >= 3:
        rng = np.random.default_rng(seed)
        for start in range(0, 1000, 100):
            indexes = rng.integers(0, len(values), size=(100, len(values)))
            boot[start:start+100] = estimates(values[indexes].sum(axis=1))
    for i, name in enumerate(names):
        finite = boot[:, i][np.isfinite(boot[:, i])] if len(values) >= 3 else np.array([])
        row[name] = float(point[i])
        limits = np.quantile(finite, [.025, .975], method="linear") if len(finite) else [np.nan, np.nan]
        row[f"{name}_ci95_low"], row[f"{name}_ci95_high"] = map(float, limits)
        row[f"{name}_valid_bootstrap_draws"] = len(finite)
    return row


def run_partition(part: str, out: str, mode: str) -> dict:
    started = time.perf_counter()
    seed_kernel = module("e006_bootstrap_seed", E002 / "30_release/70_clip_full_voyage_wave.py")
    frame = pd.read_parquet(part)
    columns = BASE_COLUMNS if mode == "base" else WAVE_COLUMNS
    rows = []
    for key, block in frame.groupby(KEY, sort=True, observed=True):
        block = block.sort_values(SHIP)
        rows.append(bootstrap_group(key, block[columns].to_numpy(float), mode, seed_kernel))
    target = Path(out) / (Path(part).stem + "_bootstrap.parquet")
    pd.DataFrame(rows).to_parquet(target, index=False)
    return {"part": Path(part).name, "combinations": len(rows), "output": record(target),
            "elapsed_seconds": time.perf_counter()-started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--mode", choices=["base", "wave"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--probe-combinations", type=int, default=0)
    args = parser.parse_args(); source = binding(); out = inside_runs(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    panel = pd.read_parquet(args.prepared / "voyage_panel.parquet")
    combinations = pd.read_parquet(args.prepared / "all_combinations.parquet")
    selected = combinations.loc[combinations.voyages.ge(5), KEY]
    if args.probe_combinations:
        selected = selected.iloc[np.unique(np.linspace(0, len(selected)-1, min(args.probe_combinations, len(selected)), dtype=int))]
    panel = panel.merge(selected, on=KEY, how="inner", validate="many_to_one")
    if args.mode == "wave":
        revision()  # the matched-hour wave estimand is the user-approved revision
    columns = BASE_COLUMNS if args.mode == "base" else WAVE_COLUMNS
    if panel[SHIP].isna().any(): raise ValueError("missing ship cluster identity")
    clusters = panel.groupby([*KEY, SHIP], sort=True, observed=True)[columns].sum().reset_index()
    seed_kernel = module("e006_partition_seed", E002 / "30_release/70_clip_full_voyage_wave.py")
    codes = [seed_kernel.stable_seed(key) % args.workers for key in clusters[KEY].itertuples(index=False, name=None)]
    clusters["part"] = codes
    parts = out / "cluster_parts"; parts.mkdir()
    paths = []
    for part, block in clusters.groupby("part", sort=True):
        p = parts / f"part_{part:04d}.parquet"; block.drop(columns="part").to_parquet(p, index=False); paths.append(p)
    state = {"state": "running", "component": f"{args.mode}_bootstrap", "workers": args.workers,
             "combinations": len(selected), "selected_voyages": len(panel), "seed": SEED, "replicates": 1000,
             "minimum_ships": 3, "completed": [], "failures": [], "authorization": source["authorization"],
             "preparation": record(args.prepared / "preparation.json"),
             "wave_estimand": "matched_mean" if args.mode == "wave" else None,
             "network_requests": 0, "products_modified": 0, "adopted": False}
    state.update(provenance([HERE / "step6_core.py", Path(__file__), DECISION, REVISION,
                         E002 / "30_release/70_clip_full_voyage_wave.py"]))
    write_json(out / "status.json", state)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(run_partition, str(p), str(out), args.mode): p for p in paths}
        for future in as_completed(tasks):
            try:
                report = future.result(); state["completed"].append(report)
                print(json.dumps({k: report[k] for k in ("part", "combinations", "elapsed_seconds")}), flush=True)
            except Exception as exc:
                state["failures"].append({"part": str(tasks[future]), "error": repr(exc)})
            write_json(out / "status.json", state)
    if state["failures"]:
        state["state"] = "failed"; write_json(out / "status.json", state); raise RuntimeError(state["failures"])
    result = pd.concat([pd.read_parquet(Path(x["output"]["path"])) for x in state["completed"]], ignore_index=True).sort_values(KEY)
    if len(result) != len(selected) or result.duplicated(KEY).any(): raise ValueError("bootstrap combination grain changed")
    result.to_parquet(out / "bootstrap.parquet", index=False)
    state.update(state="completed", output=record(out / "bootstrap.parquet"), elapsed_seconds=time.perf_counter()-started)
    write_json(out / "status.json", state)


if __name__ == "__main__": main()
