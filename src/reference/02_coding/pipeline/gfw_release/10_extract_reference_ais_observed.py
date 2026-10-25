#!/usr/bin/env python3
"""E007 V1 input: reference AIS observed AIS points regrouped by GFW-version vessel shard.

Reads every shard of the adopted reference AIS 2025 companion-track product and keeps only rows whose
position_source is observed_ais, i.e. a real AIS report with its own observation time. GFW
anchors and interpolated rows are dropped, so the reference never contains GFW positions.
Rows are written per GFW-version shard, so each GFW shard can later be compared with the same
vessels' reference AIS observations. reference AIS points are an internal validation reference only.

    python 10_extract_reference_ais_observed.py --out <new .runs dir> [--reference-ais-shards shard_0000 ...] [--workers N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
REFERENCE_AIS = CODING / "data/processed/products/companion_tracks/2025"
GFW_TRACKS = CODING / ".runs/2026-10-06_2341_E006-gfw-ten-minute-tracks"
VESSEL = ["mmsi", "vessel_uid_clean"]
KEEP = ["candidate_id", *VESSEL, "observation_time_utc", "lat", "lon"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()


def vessel_map() -> pd.DataFrame:
    files = sorted((GFW_TRACKS / "shards").glob("shard_*/candidate_summary.parquet"))
    if len(files) != 64:
        raise ValueError(f"expected 64 GFW shards, found {len(files)}")
    frame = pd.concat([pd.read_parquet(f, columns=[*VESSEL, "source_shard"]) for f in files]).drop_duplicates()
    if frame.duplicated(VESSEL).any():
        raise ValueError("a GFW vessel identity appears in more than one shard")
    return frame.rename(columns={"source_shard": "gfw_shard"}).reset_index(drop=True)


def run_shard(reference_ais_shard: str, vessels: pd.DataFrame, out: str) -> dict:
    started = time.perf_counter()
    path = REFERENCE_AIS / "run/shards" / reference_ais_shard / "ten_minute_complete_candidate.parquet"
    receipt = {"reference_ais_shard": reference_ais_shard, "path": str(path.relative_to(CODING)), "sha256": sha256(path),
               "bytes": path.stat().st_size, "rows_total": pq.ParquetFile(path).metadata.num_rows}
    obs = pq.read_table(path, columns=KEEP, filters=[("position_source", "=", "observed_ais")]).to_pandas()
    receipt["observed_rows"] = len(obs)
    if obs.observation_time_utc.isna().any() or obs[["lat", "lon"]].isna().any().any():
        raise ValueError(f"{reference_ais_shard}: observed_ais row without time or position")
    obs = obs.merge(vessels, on=VESSEL, how="left")
    absent = obs.gfw_shard.isna()
    receipt["rows_vessel_not_in_gfw"] = int(absent.sum())
    receipt["vessels_not_in_gfw"] = int(obs.loc[absent, "mmsi"].nunique())
    obs = obs.loc[~absent]
    # One vessel cannot report two positions at one instant; identical repeats are kept once,
    # conflicting repeats are dropped and counted rather than resolved by choice.
    obs = obs.drop_duplicates([*VESSEL, "observation_time_utc", "lat", "lon"])
    conflict = obs.duplicated([*VESSEL, "observation_time_utc"], keep=False)
    receipt["conflicting_instant_rows_dropped"] = int(conflict.sum())
    obs = obs.loc[~conflict].rename(columns={"candidate_id": "reference_ais_candidate_id"})
    receipt["rows_kept"] = len(obs)
    receipt["vessels_kept"] = int(obs.mmsi.nunique())
    for gfw_shard, block in obs.groupby("gfw_shard", sort=True):
        target = Path(out) / "parts" / gfw_shard
        target.mkdir(parents=True, exist_ok=True)
        block.drop(columns="gfw_shard").sort_values([*VESSEL, "observation_time_utc"]).to_parquet(
            target / f"{reference_ais_shard}.parquet", index=False)
    receipt["seconds"] = round(time.perf_counter() - started, 1)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-ais-shards", nargs="*")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    out.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((REFERENCE_AIS / "adoption_manifest.json").read_text())
    if manifest.get("state") != "adopted" or manifest.get("shards") != 253:
        raise ValueError("reference AIS companion-track product is not the adopted 253-shard release")
    shards = args.reference_ais_shards or [f"shard_{i:04d}" for i in range(253)]
    vessels = vessel_map()
    vessels.to_parquet(out / "gfw_vessel_shards.parquet", index=False)
    execution = {
        "step": "E007 V1 input: reference AIS observed AIS regrouped by GFW shard",
        "state": "running", "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_head": git("rev-parse", "HEAD"), "git_dirty_experiment": bool(git("status", "--porcelain", "--", ".")),
        "script_sha256": sha256(Path(__file__)), "reference_ais_manifest_sha256": sha256(REFERENCE_AIS / "adoption_manifest.json"),
        "gfw_tracks_execution_sha256": sha256(GFW_TRACKS / "execution.json"),
        "gfw_vessels": len(vessels), "reference_ais_shards_requested": len(shards), "workers": args.workers,
        "rule": "keep position_source == observed_ais; drop vessels absent from the GFW version; "
                "drop conflicting positions at one instant",
    }
    write_json(out / "execution.json", execution)

    receipts, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_shard, s, vessels, str(out)): s for s in shards}
        for future in as_completed(futures):
            try:
                receipts.append(future.result())
            except Exception as error:  # recorded and surfaced; no retry
                failures.append({"reference_ais_shard": futures[future], "error": repr(error)})
    receipts.sort(key=lambda r: r["reference_ais_shard"])
    pd.DataFrame(receipts).to_csv(out / "reference_ais_shard_receipts.csv", index=False)
    totals = {k: int(sum(r[k] for r in receipts)) for k in
              ["rows_total", "observed_rows", "rows_vessel_not_in_gfw", "conflicting_instant_rows_dropped", "rows_kept"]}
    execution.update(state="pass" if not failures and len(receipts) == len(shards) else "fail",
                     finished_utc=pd.Timestamp.now(tz="UTC").isoformat(), totals=totals,
                     reference_ais_shards_done=len(receipts), failures=failures)
    write_json(out / "execution.json", execution)
    print(json.dumps({k: execution[k] for k in ["state", "totals", "reference_ais_shards_done"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
