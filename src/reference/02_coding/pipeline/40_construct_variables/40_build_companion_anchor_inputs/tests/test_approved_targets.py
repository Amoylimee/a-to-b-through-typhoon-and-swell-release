from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "10_build_approved_target_hours.py"
SPEC = importlib.util.spec_from_file_location("approved_targets", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def gap(**updates):
    row = {
        "gap_id": "g1",
        "candidate_id": "c1",
        "gap_class": "gfw_candidate_4_24h",
        "gap_start": pd.Timestamp("2025-01-01T00:20:00Z"),
        "gap_end": pd.Timestamp("2025-01-01T05:40:00Z"),
        "gap_hours": 16 / 3,
        "gfw_eligible": True,
        "left_lat": 0.0,
        "left_lon": 0.0,
        "right_lat": 0.2,
        "right_lon": 0.2,
        "mmsi": "123456789",
        "vessel_uid_clean": "ship1",
        "layer": "L1",
        "source_shard": "shard_0000",
    }
    row.update(updates)
    return row


def test_full_utc_hours_only():
    gaps, targets, audit = MODULE.build_targets(pd.DataFrame([gap()]))
    assert gaps.loc[0, "target_eligible"]
    assert targets["hour_utc"].tolist() == [
        pd.Timestamp("2025-01-01T01:00:00Z"),
        pd.Timestamp("2025-01-01T02:00:00Z"),
        pd.Timestamp("2025-01-01T03:00:00Z"),
        pd.Timestamp("2025-01-01T04:00:00Z"),
    ]
    assert audit["full_utc_target_hours"] == 4


def test_endpoint_speed_above_cap_stays_in_denominator_without_target():
    fast = gap(right_lon=20.0)
    gaps, targets, audit = MODULE.build_targets(pd.DataFrame([fast]))
    assert gaps.loc[0, "target_eligibility"] == "source_endpoint_speed_above_25kn"
    assert targets.empty
    assert audit["source_endpoint_speed_above_25kn_gap_rows"] == 1
