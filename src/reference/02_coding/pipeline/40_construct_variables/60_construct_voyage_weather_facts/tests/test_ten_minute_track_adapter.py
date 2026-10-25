from __future__ import annotations

import numpy as np
import pandas as pd

from weather_core.ten_minute_track_adapter import build_sampling_positions


def test_exact_grid_positions_and_partial_edges_remain_unknown() -> None:
    voyages = pd.DataFrame(
        {
            "candidate_id": ["a"],
            "mmsi": ["1"],
            "vessel_uid_clean": ["x"],
            "departure_time": [pd.Timestamp("2025-01-01 00:21", tz="UTC")],
            "arrival_time": [pd.Timestamp("2025-01-01 02:41", tz="UTC")],
        }
    )
    times = pd.date_range("2025-01-01 00:30", "2025-01-01 02:40", freq="10min", tz="UTC")
    track = pd.DataFrame(
        {
            "candidate_id": "a",
            "mmsi": "1",
            "vessel_uid_clean": "x",
            "utc": times,
            "lat": np.arange(len(times), dtype=float),
            "lon": np.arange(len(times), dtype=float),
            "track_evidence": "observed_ais",
            "position_source": "observed_ais",
            "gap_id": pd.NA,
            "gap_class": pd.NA,
            "anchor_source": pd.NA,
        }
    )
    result = build_sampling_positions(voyages, track)
    assert np.isclose(result.segment_weight_hours.sum(), 2 + 20 / 60)
    assert result.hour_anchor_utc.tolist() == list(
        pd.to_datetime(["2025-01-01 00:00Z", "2025-01-01 01:00Z", "2025-01-01 02:00Z", "2025-01-01 03:00Z"])
    )
    assert result.track_evidence.tolist() == ["unresolved", "observed_ais", "observed_ais", "unresolved"]
    assert result.weather_position_method.tolist() == [
        "unresolved_partial_hour_midpoint",
        "exact_ten_minute_grid",
        "exact_ten_minute_grid",
        "unresolved_partial_hour_midpoint",
    ]


def test_unresolved_grid_point_stays_unknown() -> None:
    voyages = pd.DataFrame(
        {
            "candidate_id": ["a"],
            "mmsi": ["1"],
            "vessel_uid_clean": ["x"],
            "departure_time": [pd.Timestamp("2025-01-01 00:30", tz="UTC")],
            "arrival_time": [pd.Timestamp("2025-01-01 01:30", tz="UTC")],
        }
    )
    track = pd.DataFrame(
        {
            "candidate_id": ["a"],
            "mmsi": ["1"],
            "vessel_uid_clean": ["x"],
            "utc": [pd.Timestamp("2025-01-01 01:00", tz="UTC")],
            "lat": [np.nan],
            "lon": [np.nan],
            "track_evidence": ["unresolved"],
            "position_source": [pd.NA],
            "gap_id": ["g"],
            "gap_class": ["unresolved_internal_gt24h"],
            "anchor_source": [pd.NA],
        }
    )
    result = build_sampling_positions(voyages, track)
    assert result.track_evidence.tolist() == ["unresolved"]
    assert result.weather_position_method.tolist() == ["exact_ten_minute_grid"]
    assert result.lat.isna().all()
