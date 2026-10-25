from __future__ import annotations

import numpy as np
import pandas as pd
from importlib import import_module
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "10_aggregate_od_shiptype_exposure"))
aggregate_panel = import_module("10_build_aggregation").aggregate_panel


def row(candidate_id, origin, destination, ship, *, total, located, matched, integral, high, tc300, tc500):
    return {
        "candidate_id": candidate_id,
        "origin_city": origin,
        "dest_city": destination,
        "shiptype_group": "container",
        "vessel_uid_clean": ship,
        "voyage_hours": total,
        "track_hours": located,
        "wave_matched_hours": matched,
        "wave_integral_m_hours": integral,
        "wave_ge4m_matched_hours": high,
        "tc300_positive_snapshot_hours": tc300,
        "tc500_positive_snapshot_hours": tc500,
        "unresolved_track_hours": total - located,
        "wave_unmatched_hours": total - matched,
    }


def test_aggregation_is_directional_and_uses_hour_weighted_denominators():
    panel = pd.DataFrame([
        row("1", "A", "B", "ship-1", total=10, located=8, matched=6, integral=12, high=2, tc300=1, tc500=2),
        row("2", "A", "B", "ship-2", total=2, located=1, matched=2, integral=6, high=1, tc300=0, tc500=0),
        row("3", "B", "A", "ship-1", total=4, located=4, matched=4, integral=8, high=0, tc300=1, tc500=1),
    ])

    result = aggregate_panel(panel)

    assert len(result) == 2
    ab = result.set_index(["origin_city", "dest_city"]).loc[("A", "B")]
    ba = result.set_index(["origin_city", "dest_city"]).loc[("B", "A")]
    assert ab.voyages == 2
    assert ab.ships == 2
    assert ab.voyage_hours == 12
    assert ab.track_hours == 9
    assert ab.position_coverage == 0.75
    assert ab.wave_matched_hours == 8
    assert ab.wave_coverage == 8 / 12
    assert ab.mean_swh_matched_m == 18 / 8
    assert ab.share_wave_hours_ge4m == 3 / 8
    assert ab.tc500_rate_per_position_hour == 2 / 9
    assert ba.voyages == 1
    assert ba.mean_swh_matched_m == 2


def test_zero_wave_matches_remain_unknown_not_zero_mean():
    panel = pd.DataFrame([
        row("1", "A", "B", "ship-1", total=5, located=0, matched=0, integral=0, high=0, tc300=0, tc500=0),
    ])
    result = aggregate_panel(panel).iloc[0]
    assert result.wave_coverage == 0
    assert np.isnan(result.mean_swh_matched_m)
    assert np.isnan(result.share_wave_hours_ge4m)
    assert np.isnan(result.tc500_rate_per_position_hour)
