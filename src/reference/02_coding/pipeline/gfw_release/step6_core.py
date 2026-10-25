"""E006 input binding and adapters; scientific kernels come from the pipeline/E002."""
from __future__ import annotations
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODING = HERE.parents[1]
E002 = CODING / "experiments/E002_2025_od_exposure_release"
WEATHER = CODING / "pipeline/40_construct_variables/60_construct_voyage_weather_facts"
# Default to the adopted one-cell voyage population. Historical reruns must
# explicitly select their frozen decision through E006_STEP6_DECISION.
DECISION = Path(os.environ.get("E006_STEP6_DECISION") or HERE / "config/accepted_step6_decision.json").resolve()
REVISION = HERE / "config/step6_rule_revision.json"
# Census counts of the E006 voyage set; a decision for another voyage set states its own.
E006_COUNTS = {"candidates": 1230622, "weather_rows": 163883958, "combinations": 255053,
               "release_combinations": 44216, "release_voyages": 910174}
KEY = ["origin_city", "dest_city", "shiptype_group"]
SHIP = "vessel_uid_clean"
NS_HOUR = 3_600_000_000_000
SEED = 20260923
SHARDS = [f"shard_{i:04d}" for i in range(64)]
for entry in (HERE, WEATHER):
    sys.path.insert(0, str(entry))


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


weather_runner = module("e006_weather_runner_step6", HERE / "180_run_gfw_weather.py")
record = weather_runner.record
write_json = weather_runner.write_json
aggregate = module("e006_u1_shared", E002 / "10_aggregate_od_shiptype_exposure/10_build_aggregation.py")


def expected_counts(decision: dict) -> dict:
    counts = {**E006_COUNTS, **decision.get("expected_counts", {})}
    if decision["required_candidates"] != counts["candidates"]:
        raise ValueError("step-6 decision states two candidate counts")
    return counts


def binding() -> dict:
    decision = json.loads(DECISION.read_text())
    counts = expected_counts(decision)
    fixed = {"year": 2025, "population": "E006_gfw_presence_2025",
             "format": "annual_directed_city_pair_x_shiptype", "minimum_voyages": 5,
             "bootstrap_replicates": 1000, "seed": SEED,
             "minimum_ships_for_sampling_interval": 3, "speed_cap_knots": 25,
             "port_space_tolerance_km": 0, "port_time_tolerance_hours": 1,
             "holdout_max_observation_gap_seconds": 3600,
             "holdout_durations_hours": [2, 4, 6, 12, 24],
             "holdout_segments_per_stratum": 500, "holdout_additional_tc500_segments": 500}
    if any(decision.get(k) != v for k, v in fixed.items()):
        raise ValueError("step-6 decision differs from the authorized frozen scope")
    if any(decision[k] for k in decision if k.endswith("_authorized")):
        raise ValueError("step-6 decision expands scope")
    run = ROOT / decision["weather_run"]
    validation = record(run / "validation.json")
    annual = record(run / "voyage_weather_facts_2025.parquet")
    cities = record(ROOT / decision["gfw_city_table"])
    for actual, expected in ((validation, decision["weather_validation_sha256"]),
                             (annual, decision["annual_weather_sha256"]),
                             (cities, decision["gfw_city_table_sha256"])):
        if actual["sha256"] != expected:
            raise ValueError(f"step-6 source changed: {actual['path']}")
    receipt = json.loads(Path(validation["path"]).read_text())
    state = json.loads((run / "status.json").read_text())
    if receipt["status"] != "pass" or not all(receipt["checks"].values()) \
            or state["state"] != "completed" or state["failures"] \
            or receipt["candidates"] != counts["candidates"] or receipt["shards"] != 64:
        raise ValueError("GFW step-5 annual source is not complete and validated")
    return {"decision": decision, "counts": counts, "weather_run": run, "weather_state": state,
            "validation": validation, "annual": annual, "cities": cities,
            "authorization": record(DECISION)}


def revision() -> dict:
    """User-approved E007 rule revision (2026-10-09); refuse silent drift."""
    rule = json.loads(REVISION.read_text())
    fixed = {"fix_time_tolerance_hours": 0.5, "port_time_tolerance_hours": 1,
             "wave_point_estimand": "matched_mean", "disconnected_water": "unchanged_unresolved"}
    if any(rule.get(k) != v for k, v in fixed.items()):
        raise ValueError("step-6 rule revision differs from the user decision")
    return rule


def widen_fix_ends(left, right, tolerance_ns: int):
    """GFW fixes are stamped hh:30, so the real time of each known end can be up to 30 min off.

    The fix before an unknown hour may be earlier and the fix after it later; positions are
    unchanged. Port endpoints are not passed here: they already carry their own 1 h tolerance.
    """
    return ((left[0] - tolerance_ns, *left[1:]) if left else None,
            (right[0] + tolerance_ns, *right[1:]) if right else None)


def inside_runs(out: Path) -> Path:
    out = out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("private stage-6 outputs must stay in .runs")
    return out


def read_weather(shard: str, *, candidates=None) -> pd.DataFrame:
    decision = json.loads(DECISION.read_text())
    run = ROOT / decision["weather_run"]
    columns = ["candidate_id", "hour_anchor_utc", "position_time_utc", "segment_weight_hours",
               "lat", "lon", "track_evidence", "swh_m"]
    filters = [("candidate_id", "in", list(candidates))] if candidates is not None else None
    return pd.concat([pd.read_parquet(p, columns=columns, filters=filters)
                      for p in sorted((run / "shards" / shard).glob("weather_2025-??.parquet"))],
                     ignore_index=True)


def waveform_estimates(values):
    """Accepted paired full-voyage wave estimand, with no zero imputation."""
    clip = module("e006_wave_estimand", E002 / "30_release/70_clip_full_voyage_wave.py")
    return clip.estimates(values)


def safe_ratio(numerator, denominator):
    return numerator / denominator.where(denominator > 0)


def numeric_close(actual, expected, label, atol=1e-7):
    if not np.allclose(actual, expected, rtol=1e-12, atol=atol, equal_nan=True):
        raise ValueError(f"closure failed: {label}")


def provenance(paths):
    return {"source_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "code": [record(p) for p in paths]}
