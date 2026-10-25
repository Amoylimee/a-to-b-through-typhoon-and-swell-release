#!/usr/bin/env python3
"""E006 GFW version, step 1: GFW positions -> unified AIS shards (T030).

Turns `fleet_presence_unique/` (one GFW position per vessel per UTC hour) into the
unified shard schema that the cleaning stage reads (`10_ingest_data/30_build_unified_
ais_shards/schema.py`), so everything downstream runs unchanged.

* identity: the frozen 2025 roster maps each canonical MMSI to `vessel_uid` and IMO
  (vessel pool unchanged, T024); the mapping must be one-to-one;
* time: GFW gives only the UTC hour; each point is stamped at hh:30 (T030 a), and
  `UTC_10min` follows the frozen floor rule;
* fields GFW does not carry (speed, course, heading, navigational status) are null /
  "unknown"; provenance says `gfw_presence_v4.0`, `hourly_0.01deg`;
* shards: the frozen `stable_vessel_bucket` SHA-256 rule on canonical MMSI, 64 buckets,
  so each vessel sits in exactly one shard.

No network use; inputs are read only.

    python 100_build_gfw_unified_shards.py --input-root <prepared root> --out <new .runs root>
    python 100_build_gfw_unified_shards.py --verify <built root>

`--verify` re-reads every shard and checks the structural invariants of the unified
shard verifier that apply here (schema, one row per vessel and ten-minute bin, bins on
600 s, all rows in 2025), plus the hh:30 stamp and the roster identity mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
SCHEMA_PATH = CODING / "pipeline/10_ingest_data/30_build_unified_ais_shards/schema.py"
IDENTITY_PATH = CODING / "pipeline/40_construct_variables/60_construct_voyage_weather_facts/ais_identity.py"
ROSTER = CODING / "data/frozen_inputs/roster_2025/p2/extended_sample_mmsi_2025.csv"
BUCKETS = 64
PROVIDER = "gfw_presence_v4.0"
RESOLUTION = "hourly_0.01deg"
STAMP = pd.Timedelta(minutes=30)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = load("unified_shard_schema", SCHEMA_PATH)
IDENT = load("ais_identity", IDENTITY_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def roster() -> pd.DataFrame:
    frame = pd.read_csv(ROSTER, dtype=str, usecols=["mmsi", "vessel_uid", "registry_imo"])
    for column in frame.columns:
        frame[column] = frame[column].str.strip().str.replace(r"\.0$", "", regex=True)
    if frame.mmsi.duplicated().any() or frame.vessel_uid.duplicated().any():
        raise ValueError("roster is not one-to-one between MMSI and vessel_uid")
    frame["mmsi"] = frame.mmsi.astype("int64")
    frame["vessel_uid"] = frame.vessel_uid.astype("int64")
    frame["imo"] = pd.to_numeric(frame.registry_imo, errors="coerce").astype("Int64")
    frame["bucket"] = IDENT.stable_vessel_buckets(frame.mmsi.astype(str), BUCKETS).to_numpy()
    return frame[["mmsi", "vessel_uid", "imo", "bucket"]]


def write_shard(args) -> dict:
    bucket, path = args
    frame = pd.read_parquet(path)
    utc = frame.hour_utc.dt.tz_convert("UTC").dt.tz_localize(None) + STAMP
    out = pd.DataFrame({
        "vessel_uid": frame.vessel_uid.astype("int64"), "mmsi": frame.mmsi.astype("int64"),
        "imo": frame.imo.astype("Int64"), "UTC": utc, "UTC_10min": utc.dt.floor("10min"),
        "year": np.int16(2025), "lat": frame.lat.astype("float64"), "lon": frame.lon.astype("float64"),
        "speed": np.nan, "course": np.nan, "heading": np.nan,
        "nav_status": S.UNKNOWN, "nav_status_code": pd.array([pd.NA] * len(frame), dtype="Int16"),
        "source_provider": PROVIDER, "source_resolution_original": RESOLUTION,
    })
    out = out.sort_values(["vessel_uid", "UTC_10min", "UTC", "mmsi"], kind="stable").reset_index(drop=True)
    if out.duplicated(["vessel_uid", "UTC_10min"]).any():
        raise ValueError(f"bucket {bucket}: more than one row per vessel and ten-minute bin")
    table = pa.Table.from_pandas(out[S.SHARD_COLUMNS], schema=S.SHARD_SCHEMA, preserve_index=False)
    target = Path(path).parent.parent / "unified" / "2025" / f"shard_{bucket:04d}.feather"
    target.parent.mkdir(parents=True, exist_ok=True)
    pf.write_feather(table, target)
    Path(path).unlink()
    return {"shard": target.name, "bucket": int(bucket), "rows": int(len(out)),
            "vessels": int(out.vessel_uid.nunique()), "sha256": sha256_file(target),
            "utc_min": str(out.UTC.min()), "utc_max": str(out.UTC.max())}


def verify_shard(path: str) -> dict:
    table = pf.read_table(path)
    frame = table.to_pandas()
    ids = roster().set_index("vessel_uid")
    problems = []
    if not table.schema.equals(S.SHARD_SCHEMA, check_metadata=False):
        problems.append("schema differs")
    if frame.duplicated(["vessel_uid", "UTC_10min"]).any():
        problems.append("more than one row per vessel and bin")
    if not (frame.UTC_10min.astype("int64") % 600_000_000_000 == 0).all():
        problems.append("bins not on 600 s")
    if not frame.UTC_10min.eq(frame.UTC.dt.floor("10min")).all():
        problems.append("UTC_10min is not the floor of UTC")
    if not (frame.UTC.ge("2025-01-01") & frame.UTC.lt("2026-01-01")).all():
        problems.append("row outside 2025")
    if not (frame.UTC.dt.minute.eq(30) & frame.UTC.dt.second.eq(0)).all():
        problems.append("stamp not hh:30")
    mapped = ids.reindex(frame.vessel_uid)
    if not np.array_equal(mapped.mmsi.to_numpy(), frame.mmsi.to_numpy()):
        problems.append("vessel_uid to MMSI differs from roster")
    if not (frame.lat.between(-90, 90).all() and frame.lon.between(-180, 180).all()):
        problems.append("coordinate out of range")
    if not frame.source_provider.eq(PROVIDER).all():
        problems.append("provider label")
    bucket = int(Path(path).stem.split("_")[-1])
    if not IDENT.stable_vessel_buckets(frame.mmsi.drop_duplicates().astype(str), BUCKETS).eq(bucket).all():
        problems.append("vessel in the wrong bucket")
    return {"shard": Path(path).name, "rows": int(len(frame)), "problems": problems, "sha256": sha256_file(Path(path))}


def verify(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    paths = sorted(str(p) for p in (root / "unified/2025").glob("shard_*.feather"))
    with ProcessPoolExecutor(16) as pool:
        results = list(pool.map(verify_shard, paths))
    expected = {r["shard"]: r["sha256"] for r in manifest["shards"]}
    report = {"shards": len(results), "rows": sum(r["rows"] for r in results),
              "manifest_rows": manifest["rows"],
              "hashes_match_manifest": all(expected.get(r["shard"]) == r["sha256"] for r in results),
              "shards_with_problems": [r for r in results if r["problems"]],
              "verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    report["pass"] = (not report["shards_with_problems"] and report["hashes_match_manifest"]
                      and report["rows"] == report["manifest_rows"] and report["shards"] == len(expected))
    (root / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", "--acquisition-root", dest="input_root", type=Path,
                        help="prepared local monthly presence tables and their summary")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify", type=Path, help="check an already built root")
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
        return
    acq, out = args.input_root.resolve(), args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    out.mkdir(parents=True, exist_ok=False)
    source_summary = json.loads((acq / "fleet_presence_unique_summary.json").read_text())
    sources = sorted((acq / "fleet_presence_unique").glob("2025-*.parquet"))
    for path in sources:
        if sha256_file(path) != source_summary["outputs"][path.name]["sha256"]:
            raise ValueError(f"input changed since it was built: {path.name}")
    ids = roster()
    staging = out / "staging"
    staging.mkdir()
    # split the twelve monthly files into per-bucket staging files
    parts: dict[int, list[pd.DataFrame]] = {}
    total = 0
    for path in sources:
        month = pd.read_parquet(path)
        total += len(month)
        month = month.merge(ids, on="mmsi", how="left", validate="many_to_one")
        if month.vessel_uid.isna().any():
            raise ValueError(f"{path.name}: MMSI outside the frozen roster")
        for bucket, part in month.groupby("bucket", sort=True):
            parts.setdefault(int(bucket), []).append(part.drop(columns="bucket"))
    if total != source_summary["output_rows"]:
        raise ValueError("input rows differ from the unique-table summary")
    staged = []
    for bucket, frames in sorted(parts.items()):
        path = staging / f"bucket_{bucket:04d}.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
        staged.append((bucket, str(path)))
    del parts
    with ProcessPoolExecutor(args.workers) as pool:
        shards = list(pool.map(write_shard, staged))
    staging.rmdir()
    shard_frame = pd.DataFrame(shards)
    present = set()
    for record in shards:
        frame = pf.read_table(out / "unified/2025" / record["shard"], columns=["vessel_uid"]).to_pandas()
        vessels = set(frame.vessel_uid.unique())
        if present & vessels:
            raise ValueError("a vessel appears in more than one shard")
        present |= vessels
    absent = ids.loc[~ids.vessel_uid.isin(present)]
    absent.to_csv(out / "roster_vessels_without_gfw_positions.csv", index=False)
    manifest = {
        "step": "E006 GFW version step 1: unified shards", "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": "T030 a: points stamped at hh:30; fields without a GFW source are null; vessel pool unchanged",
        "source": {"root": str(acq), "unique_summary_sha256": sha256_file(acq / "fleet_presence_unique_summary.json"),
                   "files": {p.name: source_summary["outputs"][p.name] for p in sources}},
        "roster": {"path": str(ROSTER), "sha256": sha256_file(ROSTER), "vessels": int(len(ids))},
        "bucket_rule": {"function": "ais_identity.stable_vessel_bucket", "buckets": BUCKETS,
                        "code_sha256": sha256_file(IDENTITY_PATH)},
        "schema_sha256": sha256_file(SCHEMA_PATH), "script_sha256": sha256_file(Path(__file__)),
        "rows": int(shard_frame.rows.sum()), "vessels": int(len(present)),
        "roster_vessels_without_positions": int(len(absent)),
        "closure": {"rows_equal_input": int(shard_frame.rows.sum()) == total,
                    "each_vessel_in_one_shard": True, "one_row_per_vessel_and_10min_bin": True},
        "shards": shards, "network_requests": 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("shards", "source")}, indent=2, default=str))


if __name__ == "__main__":
    main()
