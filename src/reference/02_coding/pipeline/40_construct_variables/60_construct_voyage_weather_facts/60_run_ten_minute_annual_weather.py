#!/usr/bin/env python3
"""Run and independently validate all 253 E001 2025 weather shards."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
CODING = HERE.parents[2]
MATCHER = HERE / "40_match_ten_minute_weather_shard.py"
VALIDATOR = HERE / "50_validate_ten_minute_weather_shard.py"
WEATHER_STAGE = HERE
EXPECTED_SHARDS = [f"shard_{number:04d}" for number in range(253)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fingerprint(baseline: Path, config: Path, decision: Path) -> dict[str, Any]:
    validation_path = baseline / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "pass" or validation.get("endpoint_policy") != "unresolved":
        raise ValueError("annual weather requires a validated contract-compliant baseline")
    if validation.get("selected_shards") != EXPECTED_SHARDS:
        raise ValueError("annual weather baseline does not contain exactly 253 shards")
    decision_value = json.loads(decision.read_text(encoding="utf-8"))
    scope = decision_value.get("approved_scope", {})
    if decision_value.get("decision") != "approved_2025_weather_candidate":
        raise ValueError("weather decision is not approved")
    if scope.get("operations") != ["weather_matching"] or scope.get("year") != 2025:
        raise ValueError("weather decision has the wrong operation or year")
    if scope.get("required_baseline_status") != "pass":
        raise ValueError("weather decision does not require a passing baseline")
    if scope.get("required_baseline_shards") != 253:
        raise ValueError("weather decision does not require the complete baseline")
    if scope.get("required_endpoint_policy") != "unresolved":
        raise ValueError("weather decision does not preserve the >24 h unknown rule")
    if scope.get("new_network_requests_approved") is not False:
        raise ValueError("weather decision must forbid new network requests")
    if scope.get("products_modified") is not False:
        raise ValueError("weather decision must keep accepted products unchanged")
    weather_config = json.loads(config.read_text(encoding="utf-8"))
    expected_config = {
        "year": 2025,
        "maximum_wet_cell_snap_km": 117.94,
        "maximum_tc_fix_gap_hours": 3.0,
        "tc_radius_km": [300.0, 500.0],
        "ibtracs_sha256": "2f5d6f7c0eeab83ec23e306c9fcece5ddd85734e6ac36ae90cbc7624d527d771",
        "ibtracs_full_year_coverage_verified": True,
        "ibtracs_best_track_final": False,
    }
    for key, expected in expected_config.items():
        if weather_config.get(key) != expected:
            raise ValueError(f"protected weather configuration differs: {key}")
    code_paths = [MATCHER, VALIDATOR, WEATHER_STAGE / "weather_core/ten_minute_track_adapter.py"]
    code_paths.extend(CODING / "pipeline/00_environment_and_config" / name
                      for name in ("ten_minute_review_gates.py", "review_gates.py"))
    code_paths.extend(sorted((WEATHER_STAGE / "weather_core").glob("*.py")))
    return {
        "baseline": str(baseline.resolve()),
        "baseline_validation_sha256": sha256(validation_path),
        "config": str(config.resolve()),
        "config_sha256": sha256(config),
        "decision": str(decision.resolve()),
        "decision_sha256": sha256(decision),
        "code_sha256": {str(path.relative_to(CODING)): sha256(path) for path in sorted(set(code_paths))},
        "python": sys.executable,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.workers <= 12:
        raise ValueError("weather workers must be between 1 and 12")
    baseline = args.baseline_run.resolve()
    config = args.config.resolve()
    decision = args.decision.resolve()
    output = args.output.resolve()
    if not output.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("annual weather output must stay under 02_coding/.runs")
    identity = _fingerprint(baseline, config, decision)
    plan = {
        "operation": "2025_weather_matching",
        "shards": len(EXPECTED_SHARDS),
        "workers": args.workers,
        "baseline": str(baseline),
        "output": str(output),
        "network_requests": 0,
        "products_modified": 0,
        "execute": args.execute,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    if output.exists() and args.resume:
        previous = json.loads((output / "fingerprint.json").read_text(encoding="utf-8"))
        previous.pop("review_clearance", None)
        if previous != identity:
            raise ValueError("resume fingerprint differs")
    sys.path.insert(0, str(CODING / "pipeline/00_environment_and_config"))
    from ten_minute_review_gates import require_weather_clearance
    clearance = require_weather_clearance(CODING, baseline, 2025, decision,
                                          getattr(args, "prepared_input", None))
    identity["review_clearance"] = clearance
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        if json.loads((output / "fingerprint.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("resume fingerprint differs")
    else:
        if args.resume:
            raise FileNotFoundError(output)
        output.mkdir(parents=True)
        (output / "shards").mkdir()
        (output / "logs").mkdir()
        write_json(output / "fingerprint.json", identity)

    stop = threading.Event()
    completed: list[str] = []
    skipped: list[str] = []
    not_started: list[str] = []
    errors: list[dict[str, str]] = []
    lock = threading.Lock()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    def status(state: str) -> dict[str, Any]:
        with lock:
            value = {
                "state": state,
                "started_utc": started_utc,
                "workers": args.workers,
                "completed_shards": sorted(completed),
                "skipped_validated_shards": sorted(skipped),
                "not_started_shards": sorted(not_started),
                "errors": list(errors),
                "total_shards": len(EXPECTED_SHARDS),
                "elapsed_seconds": time.perf_counter() - started,
                "network_requests": 0,
                "products_modified": 0,
            }
            write_json(output / "status.json", value)
            return value

    def one(shard: str) -> tuple[str, str]:
        target = output / "shards" / shard
        validation_path = target / "validation.json"
        if target.exists():
            if not validation_path.is_file():
                raise ValueError(f"existing weather shard lacks validation: {shard}")
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            shard_status = json.loads((target / "status.json").read_text(encoding="utf-8"))
            if validation.get("status") != "pass" or shard_status.get("state") != "completed":
                raise ValueError(f"existing weather shard is not valid: {shard}")
            if shard_status["sources"]["baseline_validation"]["sha256"] != identity["baseline_validation_sha256"]:
                raise ValueError(f"existing weather shard uses another baseline: {shard}")
            return shard, "skipped"
        if stop.is_set():
            return shard, "not_started"
        environment = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        log_path = output / "logs" / f"{shard}.log"
        with log_path.open("x", encoding="utf-8") as log:
            command = [
                sys.executable,
                str(MATCHER),
                "--baseline-run", str(baseline),
                "--config", str(config),
                "--shard", shard,
                "--output", str(target),
                "--decision", str(decision),
            ]
            if getattr(args, "prepared_input", None):
                command.extend(["--prepared-input", str(args.prepared_input)])
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"weather matcher exit {result.returncode}: {shard}")
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), "--run", str(target)],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if result.returncode:
                raise RuntimeError(f"weather validator exit {result.returncode}: {shard}")
        return shard, "completed"

    status("running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, shard): shard for shard in EXPECTED_SHARDS}
        for future in concurrent.futures.as_completed(futures):
            shard = futures[future]
            try:
                name, outcome = future.result()
                if outcome == "completed":
                    completed.append(name)
                elif outcome == "skipped":
                    skipped.append(name)
                else:
                    not_started.append(name)
            except BaseException as exc:
                stop.set()
                errors.append({"shard": shard, "error": f"{type(exc).__name__}: {exc}"})
            status("running")
    if errors or len(completed) + len(skipped) != len(EXPECTED_SHARDS):
        final = status("failed")
        raise RuntimeError(f"annual weather incomplete: {len(errors)} errors")

    frames = [
        pd.read_parquet(output / "shards" / shard / "voyage_weather_facts.parquet")
        for shard in EXPECTED_SHARDS
    ]
    annual = pd.concat(frames, ignore_index=True)
    if annual.candidate_id.astype(str).duplicated().any():
        raise ValueError("annual weather candidate IDs are not unique across shards")
    annual_path = output / "voyage_weather_facts_2025.parquet"
    annual.to_parquet(annual_path, index=False)
    totals: Counter[str] = Counter()
    for shard in EXPECTED_SHARDS:
        validation = json.loads((output / "shards" / shard / "validation.json").read_text(encoding="utf-8"))
        totals.update(
            {
                "candidates": int(validation["candidates"]),
                "rows": int(validation["rows"]),
            }
        )
    final = status("completed")
    final.update(
        finished_utc=datetime.now(timezone.utc).isoformat(),
        candidates=int(totals["candidates"]),
        rows=int(totals["rows"]),
        annual_output={
            "path": str(annual_path),
            "bytes": annual_path.stat().st_size,
            "sha256": sha256(annual_path),
        },
    )
    write_json(output / "status.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=WEATHER_STAGE / "config/2025_ten_minute_weather.json")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepared-input", type=Path,
                        help="Actual prepared input_manifest.json for a historical baseline")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
