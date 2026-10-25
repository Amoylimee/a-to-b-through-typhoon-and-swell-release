#!/usr/bin/env python3
"""Independently check E006 step-5 shard and annual weather closure.

Reuse the pipeline's source-grid, hash, time and direct-ERA5 checks, then
recalculate high-wave/cyclone hours, all voyage denominators, and sampled
cyclone distances with a separate haversine formula.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from importlib import import_module
runner = import_module("180_run_gfw_weather")
sys.path.insert(0, str(runner.STAGE))
from weather_core.tropical_cyclone import CycloneTrackIndex, load_ibtracs_snapshot

EXTRA_SUMS = ["wave_ge4m_matched_hours", "tc300_positive_snapshot_hours",
              "tc500_positive_snapshot_hours"]


def independent_cyclone_distances(frame: pd.DataFrame, index: CycloneTrackIndex) -> int:
    valid = frame.loc[frame.lat.notna()]
    if valid.empty:
        return 0
    sample = valid.sample(min(24, len(valid)), random_state=20260921).reset_index(drop=True)
    centres = index.positions_at_times(sample.position_time_utc.drop_duplicates())
    actual = np.full(len(sample), np.nan)
    if not centres.empty:
        ship = sample.reset_index().rename(columns={"index": "sample_row",
                                                    "position_time_utc": "target_time_utc"})
        pairs = ship.merge(centres, on="target_time_utc", how="inner")
        if len(pairs):
            lat1 = np.deg2rad(pairs.lat.to_numpy(float))
            lat2 = np.deg2rad(pairs.storm_lat.to_numpy(float))
            dlat = lat2 - lat1
            dlon = np.deg2rad(pairs.storm_lon.to_numpy(float) - pairs.lon.to_numpy(float))
            haversine = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            pairs["independent_km"] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
            minimum = pairs.groupby("sample_row").independent_km.min()
            actual[minimum.index.to_numpy(int)] = minimum.to_numpy()
    if not np.allclose(actual, sample.tc_min_center_distance_km.to_numpy(float),
                       rtol=1e-10, atol=1e-5, equal_nan=True):
        raise ValueError("independent haversine cyclone distance differs")
    return len(sample)


def validate_shard(run: Path) -> dict:
    common = runner.load_module(runner.STAGE / "50_validate_ten_minute_weather_shard.py",
                                "e006_common_weather_validator")
    receipt = common.validate(run)
    status = json.loads((run / "status.json").read_text())
    config = json.loads(Path(status["sources"]["config"]["path"]).read_text())
    fixes, _ = load_ibtracs_snapshot(Path(config["ibtracs_path"]),
                                    expected_sha256=config["ibtracs_sha256"],
                                    time_end_exclusive="2026-01-02T00:00:00Z")
    index = CycloneTrackIndex(fixes, maximum_fix_gap_hours=config["maximum_tc_fix_gap_hours"])
    pieces = []
    sample_count = 0
    for path in sorted(run.glob("weather_*.parquet")):
        frame = pd.read_parquet(path, columns=[
            "candidate_id", "position_time_utc", "lat", "lon", "segment_weight_hours", "swh_m",
            "wave_snap_distance_km", "tc_min_center_distance_km", "tc_within_300km_snapshot",
            "tc_within_500km_snapshot", "tc_snapshot_coverage_verified",
        ])
        valid = frame.lat.notna()
        if not frame.tc_snapshot_coverage_verified.all():
            raise ValueError("TC snapshot calendar verification differs")
        for radius in (300, 500):
            flags = frame[f"tc_within_{radius}km_snapshot"]
            if flags.loc[~valid].notna().any():
                raise ValueError("unknown position has a cyclone flag")
            expected = frame.tc_min_center_distance_km.loc[valid].le(radius + 1e-9)
            if not np.array_equal(flags.loc[valid].to_numpy(bool), expected.to_numpy(bool)):
                raise ValueError(f"TC {radius} km flag differs from distance")
        if (frame.wave_snap_distance_km.dropna() > 117.94 + 1e-8).any():
            raise ValueError("wave wet-cell distance exceeds the frozen cap")
        sample_count += independent_cyclone_distances(frame, index)
        weights = frame.segment_weight_hours
        pieces.append(pd.DataFrame({
            "candidate_id": frame.candidate_id,
            "wave_ge4m_matched_hours": weights.where(frame.swh_m.ge(4), 0),
            "tc300_positive_snapshot_hours": weights.where(frame.tc_within_300km_snapshot.fillna(False), 0),
            "tc500_positive_snapshot_hours": weights.where(frame.tc_within_500km_snapshot.fillna(False), 0),
        }).groupby("candidate_id", as_index=False).sum())
    expected = pd.concat(pieces).groupby("candidate_id").sum()
    facts = pd.read_parquet(run / "voyage_weather_facts.parquet").set_index("candidate_id")
    expected.index = expected.index.astype(str)
    facts.index = facts.index.astype(str)
    for column in EXTRA_SUMS:
        if not np.allclose(expected[column], facts[column].reindex(expected.index), rtol=1e-12, atol=1e-8):
            raise ValueError(f"independent exposure sum differs: {column}")
    source = pd.read_parquet(Path(status["sources"]["summary"]["path"]),
                             columns=["candidate_id", "departure_time", "arrival_time"])
    source = source.sort_values("candidate_id", kind="stable")
    if status["max_candidates"] is not None:
        source = source.head(status["max_candidates"])
    source = source.set_index("candidate_id")
    source.index = source.index.astype(str)
    if set(facts.index) != set(source.index) or facts.index.duplicated().any():
        raise ValueError("weather candidate set differs from the original voyage summary")
    duration = (source.arrival_time - source.departure_time).dt.total_seconds()/3600
    if not np.allclose(facts.voyage_hours.reindex(duration.index), duration, rtol=0, atol=1e-8):
        raise ValueError("per-voyage full-time denominator differs")
    for total, known, unknown in (("voyage_hours", "track_hours", "unresolved_track_hours"),
                                 ("voyage_hours", "wave_matched_hours", "wave_unmatched_hours")):
        if not np.allclose(facts[total], facts[known] + facts[unknown], rtol=0, atol=1e-8):
            raise ValueError(f"time closure differs: {known}")
    mean = facts.wave_integral_m_hours / facts.wave_matched_hours.replace(0, np.nan)
    if not np.allclose(mean, facts.wave_mean_on_matched_hours_m, equal_nan=True, rtol=1e-12, atol=1e-10):
        raise ValueError("mean SWH denominator differs")
    for column in ["track_hours", "wave_matched_hours", *EXTRA_SUMS]:
        if (facts[column] < -1e-8).any() or (facts[column] > facts.voyage_hours + 1e-8).any():
            raise ValueError(f"exposure hours outside voyage duration: {column}")
    if (facts.tc300_positive_snapshot_hours > facts.tc500_positive_snapshot_hours + 1e-8).any():
        raise ValueError("TC 300 km exposure is not contained in 500 km exposure")
    receipt["checks"].update({
        "original_candidate_identity_set": True, "full_voyage_denominators": True,
        "known_plus_unknown_time_closure": True, "matched_hour_wave_mean": True,
        "independent_high_wave_and_tc_hour_sums": True, "all_tc_radius_flags": True,
        "all_wave_snap_distances": True, "independent_tc_haversine_distances": True,
    })
    receipt.update(independent_tc_distance_samples=sample_count, population="E006_gfw_presence_2025",
                   adopted=False, network_requests=0)
    runner.write_json(run / "validation.json", receipt)
    return receipt


def validate_annual(run: Path) -> dict:
    state = json.loads((run / "status.json").read_text())
    if state["state"] != "completed" or state["failures"]:
        raise ValueError("E006 weather run is incomplete")
    if state["network_requests"] != 0 or state["products_modified"] != 0 or state["adopted"] is not False:
        raise ValueError("E006 weather scope was expanded")
    fingerprint = json.loads((run / "fingerprint.json").read_text())
    authorization = Path(fingerprint["authorization"]["path"])
    binding = runner.require_authorized_inputs(authorization, Path(fingerprint["config"]["path"]))
    expected_shards = fingerprint["selected_shards"]
    if state["completed_shards"] != sorted(expected_shards):
        raise ValueError("completed weather shards differ from the selected scope")
    delivered = pd.read_parquet(run / "voyage_weather_facts_2025.parquet").set_index("candidate_id")
    if delivered.index.duplicated().any():
        raise ValueError("duplicate annual weather candidate")
    source_parts = []
    receipts = []
    for shard in expected_shards:
        folder = run / "shards" / shard
        receipt = json.loads((folder / "validation.json").read_text())
        shard_state = json.loads((folder / "status.json").read_text())
        if receipt["status"] != "pass" or not all(receipt["checks"].values()):
            raise ValueError(f"shard validation does not pass: {shard}")
        if shard_state["sources"]["baseline_validation"] != fingerprint["baseline_validation"]:
            raise ValueError(f"shard uses a different source baseline: {shard}")
        for key, asset in fingerprint["weather_assets"].items():
            expected = {field: asset[field] for field in ("path", "bytes", "sha256")}
            if shard_state["sources"].get(key) != expected:
                raise ValueError(f"shard uses a different weather snapshot: {shard} {key}")
        for item in shard_state["outputs"].values():
            if runner.record(Path(item["path"])) != item:
                raise ValueError(f"weather output changed: {item['path']}")
        receipts.append(receipt)
        source_parts.append(pd.read_parquet(folder / "voyage_weather_facts.parquet"))
    recomputed = pd.concat(source_parts, ignore_index=True).set_index("candidate_id")
    pd.testing.assert_frame_equal(delivered.sort_index(), recomputed.sort_index(), check_exact=True)
    if len(delivered) != sum(r["candidates"] for r in receipts):
        raise ValueError("annual candidate count does not close")
    if state["complete_year"] and (len(delivered) != 1230622 or expected_shards != runner.SHARDS):
        raise ValueError("annual scope is not the authorized complete GFW population")
    if runner.record(Path(state["annual_output"]["path"])) != state["annual_output"]:
        raise ValueError("annual weather output changed")
    expected_totals = {c: float(delivered[c].sum()) for c in state["totals"]}
    for column, value in expected_totals.items():
        if not np.isclose(value, state["totals"][column], rtol=1e-12, atol=1e-8):
            raise ValueError(f"annual total differs: {column}")
    receipt = {
        "status": "pass", "population": "E006_gfw_presence_2025", "complete_year": state["complete_year"],
        "shards": len(receipts), "candidates": len(delivered), "rows": sum(r["rows"] for r in receipts),
        "checks": {**{key: all(r["checks"].get(key) for r in receipts) for key in receipts[0]["checks"]},
                   "annual_candidate_uniqueness": True, "annual_exact_shard_assembly": True,
                   "authorized_scope_binding": True, "annual_output_hash": True,
                   "annual_totals_close": True, "all_shards_pass": True},
        "totals": expected_totals, "network_requests": 0, "products_modified": 0, "adopted": False,
        "source_branch": state["source_branch"], "source_commit": state["source_commit"],
        "weather_assets": fingerprint["weather_assets"],
        "baseline_validation": runner.record(Path(binding["validation"])),
        "output": state["annual_output"],
    }
    runner.write_json(run / "validation.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((runner.CODING / ".runs").resolve()):
        raise ValueError("validation output must stay inside .runs")
    result = validate_annual(run) if (run / "fingerprint.json").is_file() else validate_shard(run)
    print(json.dumps({key: result[key] for key in ("status", "candidates", "rows", "checks")}, indent=2))


if __name__ == "__main__":
    main()
