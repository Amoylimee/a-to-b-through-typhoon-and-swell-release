#!/usr/bin/env python3
"""E006 GFW version, step 3a: port calls and raw legs on the cleaned GFW shards (10a).

Runs the accepted port-call stage (`30_build_analysis_sample/10_identify_port_calls_and_legs/
10_build_port_calls_and_legs.py`) unchanged on each step-2 mask output, exactly as E001 did
for its L3 shards (`experiments/E001_2025_fleet_expansion/103_process_reference_ais_l3_shard.py`,
`ports` stage): frozen inclusive anchorage catalog, protected 3/4 km, 0.2/0.5 kn and 4 h
thresholds, and the frozen extended roster as the analysis vessel specs (T024). Stop
detection reads the cleaned inter-point speed of the hourly GFW points; the thresholds stay
as they are by user decision (T022/T030).

Each shard writes `<out>/shards/shard_NNNN/` with the stage's visits and legs, a per-vessel
input disposition and an `execution.json` binding input, code and output hashes. The run
writes `execution.json` (progress) and, for the complete 64-shard set, closes the row and
vessel denominators against step 2. No network use.

    python 120_run_gfw_port_calls.py --clean-root <step-2 root> --out <new .runs root> [--shards shard_0000 ...]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
PORT_STAGE = CODING / "pipeline/30_build_analysis_sample/10_identify_port_calls_and_legs"
ROSTER = CODING / "data/frozen_inputs/roster_2025/p2/extended_sample_mmsi_2025.csv"
CATALOG = CODING / "data/frozen_inputs/anchorage_catalog/anchorages_analytical_city_v3_dock_adjusted_inclusive.csv"
EXPECTED = {
    ROSTER: "32526296cde85a63489de621f29880a48b78686391f0b3eef44f3e64c8d9c078",
    CATALOG: "b7cc3ac333c0f58e2ff7b3ee341c21b1c41e627f893d829e955ca9da00338f77",
}
PORT_PARAMETERS = {"ENTRY_RADIUS_KM": 3.0, "EXIT_RADIUS_KM": 4.0,
                   "STOP_SPEED_ENTER_KN": 0.2, "STOP_SPEED_EXIT_KN": 0.5, "GAP_HOURS": 4.0}
SHARDS = [f"shard_{i:04d}" for i in range(64)]
STEP2_KEPT_ROWS = 255_216_589
STEP2_VESSELS = 34_884


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise ValueError(f"bound input changed: {path}")
    return path


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalized(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def code_records() -> list[dict]:
    return [record(p) for p in sorted({*PORT_STAGE.glob("*.py"), Path(__file__)})]


def write_analysis_specs(path: Path) -> None:
    """The frozen extended roster with the clean identity columns 10a, 30b and 30c read."""
    roster = pd.read_csv(ROSTER, dtype=str)
    for original, clean in (("vessel_uid", "vessel_uid_clean"), ("registry_imo", "registry_imo_clean"), ("mmsi", "mmsi_clean")):
        roster[clean] = normalized(roster[original])
        if roster[clean].isna().any() or roster[clean].eq("").any() or roster[clean].duplicated().any():
            raise ValueError(f"ambiguous roster identity: {clean}")
    roster.to_csv(path, index=False)


def step2_sources(clean_root: Path) -> dict[str, dict]:
    """Bind each shard's mask output to the hash its step-2 mask receipt recorded."""
    summary = json.loads((clean_root / "run_summary.json").read_text())
    if summary["succeeded"] != 64 or summary["failed"] or summary["speed_allowance"] != 2.0:
        raise ValueError("step 2 is not the complete 64-shard run with the T031 speed allowance")
    sources = {}
    for shard in SHARDS:
        receipt = json.loads((clean_root / "shards" / shard / "mask/execution.json").read_text())
        path = clean_root / "shards" / shard / "mask" / f"{shard}.feather"
        if receipt["status"] != "success" or receipt["outputs"].get(str(path)) is None:
            raise ValueError(f"no successful mask output for {shard}")
        sources[shard] = {"path": str(path), "bytes": path.stat().st_size, "sha256": receipt["outputs"][str(path)]}
    return sources


def process_one(task: dict) -> dict:
    started = time.monotonic()
    source = verify(task["source"])
    stage = load_module("e006_ports", PORT_STAGE / "10_build_port_calls_and_legs.py")
    stage.VESSEL_SPECS_PATH_TEMPLATE = Path(task["specs"])
    if any(getattr(stage, key) != value for key, value in PORT_PARAMETERS.items()):
        raise ValueError("protected port-call parameters changed")
    out = Path(task["output"])
    out.mkdir(parents=True, exist_ok=False)
    pipeline_task = stage.build_task(source, 2025, out, out / "tasks", Path(task["catalog"]))
    result = stage.process_task(pipeline_task, out)
    if result["status"] != "success" or result["n_input_rows"] != result["n_filtered_rows"]:
        raise ValueError(f"port-call failure or unexpected vessel loss: {result}")
    visits = pd.read_feather(result["visits_output_path"])
    legs = pd.read_feather(result["voyages_output_path"])
    points = pd.read_feather(source, columns=["vessel_uid", "UTC"])
    points["vessel_uid"] = points.vessel_uid.astype(str)
    if not points.UTC.dt.year.eq(2025).all() or not points.UTC.dt.minute.eq(30).all():
        raise ValueError("GFW point time outside 2025 or not stamped hh:30")
    vessels = points.groupby("vessel_uid").agg(
        cleaned_rows=("UTC", "size"), first_cleaned_utc=("UTC", "min"), last_cleaned_utc=("UTC", "max"))
    vessels["port_visits"] = visits.groupby("vessel_uid").size().reindex(vessels.index).fillna(0).astype(int)
    vessels["high_confidence_visits"] = (visits.loc[visits.visit_confidence.eq("high")].groupby("vessel_uid").size()
                                         .reindex(vessels.index).fillna(0).astype(int))
    vessels["raw_legs"] = legs.groupby("vessel_uid").size().reindex(vessels.index).fillna(0).astype(int)
    vessel_path = out / "vessel_input_disposition.csv"
    vessels.reset_index().to_csv(vessel_path, index=False)
    if visits.visit_id_internal.duplicated().any() or legs.internal_voyage_id.duplicated().any():
        raise ValueError("duplicate port or leg IDs")
    if len(legs):
        high = set(visits.loc[visits.visit_confidence.eq("high"), "visit_id_internal"])
        if not (set(legs.origin_visit_id_internal) | set(legs.dest_visit_id_internal)) <= high:
            raise ValueError("a raw leg has a low-confidence endpoint")
        starts, ends = pd.to_datetime(legs.trip_start, utc=True), pd.to_datetime(legs.trip_end, utc=True)
        if not (starts.dt.year.eq(2025) & ends.dt.year.eq(2025) & ends.gt(starts)).all():
            raise ValueError("raw-leg times fall outside 2025 or are not positive")
    result.update(shard=task["shard"], status="success", wall_seconds=time.monotonic() - started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source=task["source"], port_parameters=PORT_PARAMETERS,
                  catalog=task["catalog_record"], specs=task["specs_record"],
                  vessel_input_disposition=record(vessel_path),
                  outputs=[record(Path(result[key])) for key in ("visits_output_path", "voyages_output_path")])
    write_json(out / "execution.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", nargs="*")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    clean_root, root = args.clean_root.resolve(), args.out.resolve()
    if not root.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    shards = sorted(args.shards or SHARDS)
    if not set(shards) <= set(SHARDS) or len(shards) != len(set(shards)):
        raise ValueError("invalid shard selection")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ValueError(f"frozen input changed: {path}")
    sources = step2_sources(clean_root)
    root.mkdir(parents=True, exist_ok=False)
    specs = root / "analysis_vessel_specs_2025.csv"
    write_analysis_specs(specs)
    catalog_record, specs_record = record(CATALOG), record(specs)
    tasks = [{"shard": s, "source": sources[s], "specs": str(specs), "specs_record": specs_record,
              "catalog": str(CATALOG), "catalog_record": catalog_record,
              "output": str(root / "shards" / s)} for s in shards]
    complete = shards == SHARDS
    state = {"step": "E006 GFW version step 3a: port calls and raw legs (10a)", "status": "running", "year": 2025,
             "shards": shards, "complete_year": complete, "workers": min(args.workers, len(tasks)),
             "clean_root": str(clean_root), "step2_summary": record(clean_root / "run_summary.json"),
             "roster": record(ROSTER), "catalog": catalog_record, "specs": specs_record,
             "port_parameters": PORT_PARAMETERS, "code": code_records(), "adopted": False, "network_requests": 0,
             "results": [], "failures": []}
    write_json(root / "execution.json", state)
    started = time.monotonic()
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=state["workers"]) as pool:
        futures = {pool.submit(process_one, task): task["shard"] for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(json.dumps({"done": len(results), "shard": result["shard"], "rows": result["n_input_rows"],
                                  "visits": result["n_visits"], "raw_legs": result["n_voyages"],
                                  "seconds": round(result["wall_seconds"])}), flush=True)
            except Exception as error:
                failures.append({"shard": futures[future], "error": repr(error)[:2000]})
                print(f"FAILED {failures[-1]}", flush=True)
            state.update(results=sorted(results, key=lambda r: r["shard"]), failures=failures,
                         completed=len(results), elapsed_seconds=time.monotonic() - started)
            write_json(root / "execution.json", state)
    for item in state["code"]:
        verify(item)
    totals = {key: sum(r[key] for r in results) for key in
              ("n_input_rows", "n_unique_vessels", "n_visits", "n_high_confidence_visits", "n_voyages")}
    state["totals"] = totals
    if complete and not failures:
        disposition = pd.concat([pd.read_csv(verify(r["vessel_input_disposition"]), dtype={"vessel_uid": str})
                                 for r in results], ignore_index=True)
        closure = {"all_64_shards": len(results) == 64,
                   "cleaned_rows_close_to_step2": totals["n_input_rows"] == STEP2_KEPT_ROWS,
                   "vessels_close_to_step2": len(disposition) == STEP2_VESSELS == totals["n_unique_vessels"],
                   "each_vessel_in_one_shard": not disposition.vessel_uid.duplicated().any()}
        state["closure"] = closure
        if not all(closure.values()):
            failures.append({"shard": "all", "error": f"denominator closure failed: {closure}"})
    state.update(status="failed" if failures else "pass", failures=failures, wall_seconds=time.monotonic() - started)
    write_json(root / "execution.json", state)
    print(json.dumps({k: v for k, v in state.items() if k not in ("results", "code")}, indent=2, default=str))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
