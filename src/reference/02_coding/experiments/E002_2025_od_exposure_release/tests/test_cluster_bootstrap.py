from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "20_quantify_uncertainty/30_sample_size/10_cluster_bootstrap.py"
)
SPEC = importlib.util.spec_from_file_location("e002_cluster_bootstrap", SCRIPT)
assert SPEC and SPEC.loader
BOOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOT)


def test_multiple_voyages_remain_together_in_ship_cluster_bootstrap() -> None:
    group = pd.DataFrame({
        "candidate_id": ["v1", "v2"],
        "vessel_uid_clean": ["ship-a", "ship-a"],
        "origin_city": ["A", "A"],
        "dest_city": ["B", "B"],
        "shiptype_group": ["container", "container"],
        "wave_integral_m_hours": [1.0, 9.0],
        "wave_matched_hours": [1.0, 3.0],
        "wave_ge4m_matched_hours": [0.0, 1.0],
        "tc300_positive_snapshot_hours": [0.0, 0.0],
        "tc500_positive_snapshot_hours": [0.0, 0.0],
        "track_hours": [1.0, 3.0],
        "voyage_hours": [1.0, 3.0],
    })
    key = ("A", "B", "container")
    first = BOOT.bootstrap_one(group, key)
    second = BOOT.bootstrap_one(group, key)
    assert first == second
    assert first["ships"] == 1
    assert first["mean_swh_m_estimate"] == 2.5
    assert first["mean_swh_m_ci95_width"] == 0.0
