#!/usr/bin/env python3
"""E006 GFW version, step 2: clean -> audit -> mask on the GFW unified shards (T031).

Runs the accepted cleaning stages unchanged (`20_clean_and_harmonize/10_clean_ais_tracks`),
with the frozen 2025 cleaning config, specs table and extended roster, exactly as
E001 did for its L3 shards (`experiments/E001_2025_fleet_expansion/103_process_reference_ais_
l3_shard.py`). GFW rows carry no reported speed, course or heading, so the checks
that need them flag nothing (T030 b); inter-point speed checks, drift removal and the
4 h trajectory split apply as before. Per-vessel maximum speed comes from the frozen
specs, the same vessel metadata as the unchanged ship types (T024).

Speed allowance (T031): GFW gives only the UTC hour of each point, and points are
stamped at hh:30, so the real time between two consecutive hourly points can be up to
twice the labelled hour and inter-point speeds can read up to twice the true speed.
With `--speed-allowance 2`, each vessel's effective maximum speed -- computed by the
unchanged `enrich_with_specs` rule (service speed, else ship-type default, else 30 kn)
-- is multiplied by the allowance and passed in as its service speed, so the drift,
outlier and audit checks all scale together. Original and used limits are written to
`speed_limits_used.csv`. Pipeline code and the frozen specs file are not changed.

Each shard writes `<run>/shards/shard_NNNN/{clean,audit,mask}/` with an
`execution.json` that binds input, code and output hashes. No network use.

    python 110_run_gfw_cleaning.py --shard-root <step-1 root> --out <new .runs root> [--shards shard_0000 ...]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
CLEAN = CODING / "pipeline/20_clean_and_harmonize/10_clean_ais_tracks"
FROZEN = CODING / "data/frozen_inputs/roster_2025"
SPECS = FROZEN / "p1_full/vessel_specs_by_mmsi_2025.csv"
EXTENDED = FROZEN / "p2/extended_sample_mmsi_2025.csv"
CONFIG = CLEAN / "config/clean_ais_2025.yaml"
EXPECTED = {
    SPECS: "c5f4725a22d84778695cbbf5b9efbd9ff55ba9058b392c1b4ee75cf9c8466a2a",
    EXTENDED: "32526296cde85a63489de621f29880a48b78686391f0b3eef44f3e64c8d9c078",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def apply_speed_allowance(common, specs: pd.DataFrame, allowance: float, shard_dir: Path) -> pd.DataFrame:
    probe = pd.DataFrame({"vessel_uid": specs["vessel_uid"]})
    effective = common.enrich_with_specs(probe, specs)[["vessel_uid", "max_speed_knots", "max_speed_source"]]
    if effective["max_speed_knots"].isna().any() or effective["vessel_uid"].duplicated().any():
        raise ValueError("effective speed limits are not one finite value per vessel")
    effective = effective.rename(columns={"max_speed_knots": "original_max_speed_knots",
                                          "max_speed_source": "original_max_speed_source"})
    effective["speed_allowance"] = allowance
    effective["used_max_speed_knots"] = effective["original_max_speed_knots"] * allowance
    effective.to_csv(shard_dir / "speed_limits_used.csv", index=False)
    adjusted = specs.merge(effective[["vessel_uid", "used_max_speed_knots"]], on="vessel_uid", how="left",
                           validate="one_to_one")
    adjusted["specs_speed_service_knots"] = adjusted.pop("used_max_speed_knots")
    return adjusted


def run_shard(args) -> dict:
    source, shard_dir, allowance = Path(args[0]), Path(args[1]), float(args[2])
    started = time.monotonic()
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ValueError(f"frozen input changed: {path}")
    config = yaml.safe_load(CONFIG.read_text())
    if config["gap"]["threshold_hours"] != 4.0 or config["drift"]["speed_multiplier"] != 1.5:
        raise ValueError("protected cleaning parameters changed")
    sys.path.insert(0, str(CLEAN))
    import cleaning_common as common

    shard_dir.mkdir(parents=True, exist_ok=False)
    specs_raw = pd.read_csv(SPECS, dtype=str)
    specs_raw["vessel_uid"] = specs_raw["vessel_uid"].str.replace(r"\.0$", "", regex=True)
    normalized_specs = shard_dir / "cleaning_specs_2025.csv"
    specs_raw.to_csv(normalized_specs, index=False)
    specs = common.load_specs_table(normalized_specs)
    if allowance != 1.0:
        specs = apply_speed_allowance(common, specs, allowance, shard_dir)
    ids = set(pd.read_feather(source, columns=["vessel_uid"]).vessel_uid.astype(str))
    if not ids <= set(specs.vessel_uid):
        raise ValueError("frozen specs do not cover every input vessel")
    code = [*CLEAN.glob("*.py"), Path(__file__)]
    report = {"shard": source.stem, "stages": {}}
    inputs = [source, SPECS, EXTENDED, CONFIG]
    for stage_name in ("clean", "audit", "mask"):
        out = shard_dir / stage_name
        out.mkdir()
        log = out / "logs"
        log.mkdir()
        cfg = dict(config, input_dir=str(source.parent), output_dir=str(out), specs_table_path=str(normalized_specs),
                   gfw_speed_allowance=allowance)
        cfg["parallel"] = dict(config["parallel"], max_workers=1)
        execution = {"stage": stage_name, "year": 2025, "layer": "GFW", "status": "running", "started_utc":
                     datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "inputs": {str(p): sha(p) for p in inputs}, "code": {str(p): sha(p) for p in code},
                     "config": cfg, "adopted": False}
        record = out / "execution.json"
        write_json(record, execution)
        t0 = time.monotonic()
        try:
            if stage_name == "clean":
                module = load_module("gfw_clean", CLEAN / "10_clean_dynamic_ais.py")
                module.clean_single_shard(source, out, log, specs, cfg)
            elif stage_name == "audit":
                module = load_module("gfw_audit", CLEAN / "20_audit_dynamic_ais.py")
                task = module.build_task(shard_dir / "clean" / source.name, out, out / "tasks")
                result = module.audit_single_shard(task, log, specs, module.get_audit_config(cfg))
                if result["status"] != "success":
                    raise RuntimeError(result)
                if json.loads(Path(task["stats_path"]).read_text())["flagged_rows_truncated"]:
                    raise ValueError("audit flags truncated: masks require complete evidence")
            else:
                module = load_module("gfw_mask", CLEAN / "30_apply_masks_dynamic_ais.py")
                task = module.build_task(shard_dir / "clean" / source.name, out, shard_dir / "audit",
                                         out / "tasks", EXTENDED)
                result = module.process_task(task, log)
                if result["status"] != "success":
                    raise RuntimeError(result)
                stats = json.loads((out / f"{source.stem}_stats.json").read_text())
                if stats["rows_dropped_by_analysis_vessel_filter"]:
                    raise ValueError("roster vessels unexpectedly lost to the analysis-vessel filter")
            execution.update(status="success", seconds=time.monotonic() - t0,
                             outputs={str(p): sha(p) for p in sorted(out.rglob("*"))
                                      if p.is_file() and p != record})
        except Exception as exc:
            execution.update(status="failed", error=repr(exc)[:2000])
            write_json(record, execution)
            raise
        write_json(record, execution)
        inputs = [*inputs, record, *[Path(p) for p in execution["outputs"]]]
        report["stages"][stage_name] = {"seconds": execution["seconds"]}
    report["seconds"] = time.monotonic() - started
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", nargs="*")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--speed-allowance", type=float, default=1.0,
                        help="multiply each vessel's effective maximum speed (T031: 2 for hourly GFW points)")
    args = parser.parse_args()
    shard_root, out = args.shard_root.resolve(), args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    manifest = json.loads((shard_root / "manifest.json").read_text())
    if not json.loads((shard_root / "verify.json").read_text())["pass"]:
        raise ValueError("step-1 shards have not passed verification")
    wanted = args.shards or [r["shard"].removesuffix(".feather") for r in manifest["shards"]]
    sources = []
    for record in manifest["shards"]:
        name = record["shard"].removesuffix(".feather")
        if name in wanted:
            path = shard_root / "unified/2025" / record["shard"]
            if sha(path) != record["sha256"]:
                raise ValueError(f"step-1 shard changed: {path}")
            sources.append((str(path), str(out / "shards" / name), args.speed_allowance))
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results, failures = [], []
    with ProcessPoolExecutor(min(args.workers, len(sources))) as pool:
        futures = {pool.submit(run_shard, task): task for task in sources}
        for future in futures:
            try:
                results.append(future.result())
                print(f"done {results[-1]['shard']} {results[-1]['seconds']:.0f}s", flush=True)
            except Exception as error:
                failures.append({"shard": Path(futures[future][0]).stem, "error": repr(error)[:1000]})
                print(f"FAILED {failures[-1]}", flush=True)
    summary = {"step": "E006 GFW version step 2: cleaning", "speed_allowance": args.speed_allowance,
               "shards_requested": len(sources),
               "succeeded": len(results), "failed": failures, "wall_seconds": time.monotonic() - started,
               "results": results, "network_requests": 0}
    name = "run_summary.json" if not args.shards else f"run_summary_{'_'.join(sorted(wanted))[:60]}.json"
    write_json(out / name, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
