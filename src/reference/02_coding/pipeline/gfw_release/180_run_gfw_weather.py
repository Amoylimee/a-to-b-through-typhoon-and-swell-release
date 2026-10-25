#!/usr/bin/env python3
"""E006 step 5: the frozen weather engine on the authorized 64-shard GFW baseline.

The E001 command-line gate remains unchanged. This entry checks the user's
separate E006 authorization, exact baseline receipts and meteorological inputs,
then reuses the pipeline matcher and validator. Outputs stay in .runs/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODING = HERE.parents[1]
STAGE = CODING / "pipeline/40_construct_variables/60_construct_voyage_weather_facts"
CONFIG = STAGE / "config/2025_ten_minute_weather.json"
DECISION = HERE / "config/step5_weather_authorization.json"
ASSETS = HERE / "config/step5_frozen_weather_assets.json"
VALIDATOR = HERE / "190_validate_gfw_weather.py"
SHARDS = [f"shard_{i:04d}" for i in range(64)]
THREAD_CAPS = {key: "1" for key in
               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_authorized_inputs(decision_path: Path = DECISION, config_path: Path = CONFIG) -> dict:
    decision = json.loads(decision_path.read_text())
    expected = {
        "decision": "approved_E006_2025_weather_candidate",
        "year": 2025, "operations": ["weather_matching"], "required_shards": 64,
        "required_candidates": 1230622, "required_grid_rows": 982073126,
        "new_network_requests_approved": False, "products_modified": False,
        "adoption_authorized": False, "step6_authorized": False,
        "push_authorized": False, "merge_authorized": False,
    }
    for key, value in expected.items():
        if decision.get(key) != value:
            raise ValueError(f"E006 weather authorization differs: {key}")
    baseline = (ROOT / decision["baseline"]).resolve()
    validation_path = (ROOT / decision["validation"]).resolve()
    for path, key in ((baseline / "execution.json", "baseline_execution_sha256"),
                      (validation_path, "baseline_validation_sha256"),
                      (config_path, "weather_config_sha256")):
        if sha(path) != decision.get(key):
            raise ValueError(f"authorized input changed: {path}")
    state = json.loads((baseline / "execution.json").read_text())
    validation = json.loads(validation_path.read_text())
    if (state.get("status") != "pass" or state.get("complete_year") is not True
            or state.get("failures") or state.get("adopted") is not False
            or state.get("shards") != SHARDS or len(state.get("results", [])) != 64):
        raise ValueError("E006 step 4 is not a passed complete 64-shard baseline")
    results = {item["shard"]: item for item in state["results"]}
    if set(results) != set(SHARDS) or any(item["state"] != "completed" for item in results.values()):
        raise ValueError("step-4 shard results are incomplete or duplicated")
    rules = state["rules"]
    for key, value in {"gap_max_hours": 48.0, "speed_check_extra_hours": 1.0,
                       "max_speed_knots": 25.0, "route_timeout_s": 5.0,
                       "snap_to_ocean": False, "grid_seconds": 600,
                       "point_stamp": "hh:30", "anchors": "none"}.items():
        if rules.get(key) != value:
            raise ValueError(f"step-4 rule differs: {key}")
    if (validation.get("status") != "passed" or len(validation.get("checks", {})) != 14
            or not all(validation["checks"].values())
            or any(validation["problems"].values())
            or validation.get("candidates") != decision["required_candidates"]
            or validation.get("rows") != decision["required_grid_rows"]):
        raise ValueError("step-4 independent validation does not pass the authorized scope")
    if (sum(item["candidates"] for item in results.values()) != validation["candidates"]
            or sum(item["totals"]["grid_rows"] for item in results.values()) != validation["rows"]):
        raise ValueError("step-4 counts do not close")
    prepared = Path(state["prepared"]["path"])
    if sha(prepared) != state["prepared"]["sha256"]:
        raise ValueError("step-4 prepared candidate manifest changed")
    return {"baseline": str(baseline), "validation": str(validation_path),
            "decision": record(decision_path), "config": record(config_path),
            "results": results, "rules": rules}


def verify_weather_assets() -> dict:
    """Check every frozen file's bytes, SHA-256, units and calendar support."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    manifest = json.loads(ASSETS.read_text())
    expected_keys = {"ibtracs", *[f"era5_2025-{i:02d}" for i in range(1, 13)]}
    if set(manifest["assets"]) != expected_keys:
        raise ValueError("frozen weather asset list is not the complete 2025 snapshot")

    def check(item):
        key, expected = item
        path = ROOT / expected["path"]
        actual = record(path)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"frozen weather asset changed: {path}")
        return key, actual

    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = dict(pool.map(check, sorted(manifest["assets"].items())))
    # NetCDF metadata reads are serial because the installed backend is not
    # thread safe; the independent bulk file hashes above are parallel.
    for key, actual in checked.items():
        if key.startswith("era5_"):
            month = int(key[-2:])
            dates = pd.date_range(f"2025-{month:02d}-01",
                                  periods=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1]*24,
                                  freq="h")
            with xr.open_dataset(actual["path"]) as dataset:
                if dataset.swh.attrs.get("units") != "m":
                    raise ValueError("ERA5 swh units differ")
                if not np.array_equal(dataset.valid_time.values, dates.to_numpy()):
                    raise ValueError(f"ERA5 calendar support differs: {month}")
                actual["hourly_frames"] = len(dates)
    return checked


def execute_shard(args) -> dict:
    binding = require_authorized_inputs(args.decision, args.config)
    item = binding["results"][args.one_shard]
    sources = {
        "baseline_state": record(Path(binding["baseline"]) / "execution.json"),
        "baseline_validation": record(Path(binding["validation"])),
        "weather_authorization": record(args.decision),
    }
    for name in ("ten_minute_complete_candidate.parquet", "candidate_summary.parquet"):
        expected = next(value for value in item["outputs"] if Path(value["path"]).name == name)
        actual = record(Path(expected["path"]))
        if actual != expected:
            raise ValueError(f"step-4 output changed: {name}")
        sources["track" if name.startswith("ten_minute") else "summary"] = actual
    matcher = load_module(STAGE / "40_match_ten_minute_weather_shard.py", "e006_weather_matcher")
    namespace = SimpleNamespace(output=args.out, config=args.config, shard=args.one_shard,
                                max_candidates=args.probe_candidates)
    clearance = {key: binding[key] for key in ("baseline", "validation", "decision", "config", "rules")}
    try:
        result = matcher.match_validated_weather_inputs(namespace, clearance, sources)
        result.update(population="E006_gfw_presence_2025", step=5,
                      interpolation_gap_max_hours=48.0, adopted=False)
        write_json(args.out / "status.json", result)
    except BaseException as exc:
        if args.out.exists():
            path = args.out / "status.json"
            prior = json.loads(path.read_text()) if path.is_file() else {}
            prior.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(path, prior)
        raise
    return {key: result[key] for key in
            ("state", "shard", "candidates", "rows", "elapsed_seconds", "peak_rss_gib")}


def run(args) -> dict:
    binding = require_authorized_inputs(args.decision, args.config)
    selected = args.shards or SHARDS
    if len(set(selected)) != len(selected) or not set(selected).issubset(SHARDS):
        raise ValueError("unknown or duplicate weather shard")
    if not 1 <= args.workers <= 40:
        raise ValueError("workers must be between 1 and 40")
    out = args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("E006 weather output must stay in .runs")
    if out.exists():
        raise FileExistsError(out)
    if args.probe_candidates is not None and args.probe_candidates < 1:
        raise ValueError("probe candidate count must be positive")
    assets = verify_weather_assets()
    code = [Path(__file__), VALIDATOR, STAGE / "40_match_ten_minute_weather_shard.py",
            STAGE / "50_validate_ten_minute_weather_shard.py", *sorted((STAGE / "weather_core").glob("*.py"))]
    identity = {
        "authorization": record(args.decision), "config": record(args.config),
        "frozen_asset_manifest": record(ASSETS), "weather_assets": assets,
        "baseline_execution": record(Path(binding["baseline"]) / "execution.json"),
        "baseline_validation": record(Path(binding["validation"])),
        "code": [record(p) for p in code], "python": sys.executable,
        "source_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "selected_shards": selected, "workers": args.workers,
        "probe_candidates": args.probe_candidates, "network_requests": 0,
        "products_modified": 0, "adopted": False,
    }
    out.mkdir()
    (out / "logs").mkdir()
    (out / "shards").mkdir()
    write_json(out / "fingerprint.json", identity)
    started = time.perf_counter()
    state = {**identity, "step": 5, "population": "E006_gfw_presence_2025",
             "state": "running", "started_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
             "total_shards": len(selected), "completed_shards": [], "failures": [],
             "complete_year": selected == SHARDS and args.probe_candidates is None}
    stop = threading.Event()

    def one(shard):
        if stop.is_set():
            return None
        command = [sys.executable, str(Path(__file__).resolve()), "--one-shard", shard,
                   "--out", str(out / "shards" / shard), "--decision", str(args.decision),
                   "--config", str(args.config)]
        if args.probe_candidates is not None:
            command.extend(["--probe-candidates", str(args.probe_candidates)])
        with (out / "logs" / f"{shard}.log").open("x") as log:
            environment = {**os.environ, **THREAD_CAPS}
            for step in (command, [sys.executable, str(VALIDATOR), "--run", str(out / "shards" / shard)]):
                result = subprocess.run(step, stdout=log, stderr=subprocess.STDOUT, env=environment)
                if result.returncode:
                    raise RuntimeError(f"weather matching or validation failed ({result.returncode}): {shard}")
        return shard

    write_json(out / "status.json", state)
    with ThreadPoolExecutor(max_workers=min(args.workers, len(selected))) as pool:
        futures = {pool.submit(one, shard): shard for shard in selected}
        for future in as_completed(futures):
            shard = futures[future]
            try:
                completed = future.result()
                if completed:
                    state["completed_shards"].append(completed)
                    state["completed_shards"].sort()
            except BaseException as exc:
                stop.set()
                state["failures"].append({"shard": shard, "error": f"{type(exc).__name__}: {exc}"})
            state["elapsed_seconds"] = time.perf_counter() - started
            write_json(out / "status.json", state)
            print(json.dumps({"completed": len(state["completed_shards"]),
                              "total": len(selected), "failures": state["failures"]}), flush=True)
    if state["failures"] or len(state["completed_shards"]) != len(selected):
        state["state"] = "failed"
        write_json(out / "status.json", state)
        raise RuntimeError("E006 weather incomplete; inspect preserved shard logs")
    import pandas as pd
    frames = [pd.read_parquet(out / "shards" / shard / "voyage_weather_facts.parquet") for shard in selected]
    annual = pd.concat(frames, ignore_index=True).sort_values("candidate_id", kind="stable")
    if annual.candidate_id.duplicated().any():
        raise ValueError("duplicate candidate across weather shards")
    annual_path = out / "voyage_weather_facts_2025.parquet"
    annual.to_parquet(annual_path, index=False)
    receipts = [json.loads((out / "shards" / shard / "validation.json").read_text()) for shard in selected]
    if (not all(r["status"] == "pass" for r in receipts)
            or len(annual) != sum(r["candidates"] for r in receipts)
            or (state["complete_year"] and len(annual) != 1230622)):
        raise ValueError("weather annual candidate closure differs")
    state.update(state="completed", ended_at=datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
                 elapsed_seconds=time.perf_counter() - started, candidates=len(annual),
                 rows=sum(r["rows"] for r in receipts), annual_output=record(annual_path),
                 peak_shard_rss_gib=max(json.loads((out / "shards" / shard / "status.json").read_text())["peak_rss_gib"]
                                       for shard in selected),
                 totals={column: float(annual[column].sum()) for column in
                         ("voyage_hours", "track_hours", "unresolved_track_hours", "wave_matched_hours",
                          "wave_unmatched_hours", "wave_integral_m_hours", "wave_ge4m_matched_hours",
                          "tc300_positive_snapshot_hours", "tc500_positive_snapshot_hours")})
    write_json(out / "status.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--decision", type=Path, default=DECISION)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shards", nargs="+")
    parser.add_argument("--probe-candidates", type=int)
    parser.add_argument("--one-shard", choices=SHARDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output_existed = args.out.exists()
    try:
        result = execute_shard(args) if args.one_shard else run(args)
    except BaseException as exc:
        if not output_existed and args.out.exists():
            path = args.out / "status.json"
            prior = json.loads(path.read_text()) if path.is_file() else {}
            prior.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(path, prior)
        raise
    print(json.dumps({key: result[key] for key in
                      ("state", "candidates", "rows", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
