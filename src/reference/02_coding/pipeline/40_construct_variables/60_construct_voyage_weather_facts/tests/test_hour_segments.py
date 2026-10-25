from __future__ import annotations

import numpy as np
import pandas as pd

from weather_core.hour_segments import (
    centered_hour_segments,
    interpolate_positions,
    required_hour_anchor_bounds,
)


def test_departure_at_0020_uses_intersection_midpoint_not_hour_anchor() -> None:
    legs = pd.DataFrame(
        {
            "physical_leg_id": ["leg-a"],
            "mmsi": ["1"],
            "leg_sequence": [0],
            "departure_time": [pd.Timestamp("2024-01-01 00:20", tz="UTC")],
            "arrival_time": [pd.Timestamp("2024-01-01 01:30", tz="UTC")],
        }
    )
    segments = centered_hour_segments(legs)
    first = segments.iloc[0]
    assert first["hour_anchor_utc"] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert first["segment_start_utc"] == pd.Timestamp("2024-01-01 00:20", tz="UTC")
    assert first["segment_end_utc"] == pd.Timestamp("2024-01-01 00:30", tz="UTC")
    assert first["position_time_utc"] == pd.Timestamp("2024-01-01 00:25", tz="UTC")
    assert np.isclose(first["segment_weight_hours"], 1.0 / 6.0)
    assert first["segment_weight_ns"] == 600_000_000_000


def test_interpolation_uses_shortest_dateline_path_and_rejects_gap() -> None:
    points = pd.DataFrame(
        {
            "vessel_key": ["1", "1", "1"],
            "UTC": ["2024-01-01 00:00", "2024-01-01 02:00", "2024-01-01 08:00"],
            "lat": [0.0, 2.0, 8.0],
            "lon": [179.0, -179.0, -170.0],
        }
    )
    target = pd.Series(pd.to_datetime(["2024-01-01 01:00", "2024-01-01 05:00"], utc=True))
    result = interpolate_positions(points, target, maximum_gap_hours=4.0)
    assert np.isclose(result.loc[0, "lat"], 1.0)
    assert np.isclose(abs(result.loc[0, "lon"]), 180.0)
    assert np.isnan(result.loc[1, "lat"])


def test_outer_time_grid_closes_inside_2021_2025() -> None:
    legs = pd.DataFrame(
        {
            "physical_leg_id": ["first", "last"],
            "mmsi": ["1", "2"],
            "leg_sequence": [0, 0],
            "departure_time": pd.to_datetime(
                ["2021-01-01 00:50:00", "2025-12-31 20:00:00"], utc=True
            ),
            "arrival_time": pd.to_datetime(
                ["2021-01-01 01:10:00", "2025-12-31 20:31:18"], utc=True
            ),
        }
    )
    first, last = required_hour_anchor_bounds(legs)
    assert first == pd.Timestamp("2021-01-01 01:00:00", tz="UTC")
    assert last == pd.Timestamp("2025-12-31 21:00:00", tz="UTC")


def test_exact_half_hour_boundaries_do_not_emit_empty_anchors() -> None:
    legs = pd.DataFrame(
        {
            "physical_leg_id": ["leg-a"],
            "mmsi": ["1"],
            "leg_sequence": [0],
            "departure_time": [pd.Timestamp("2024-01-01 00:30", tz="UTC")],
            "arrival_time": [pd.Timestamp("2024-01-01 01:30", tz="UTC")],
        }
    )
    segments = centered_hour_segments(legs)
    assert segments["hour_anchor_utc"].tolist() == [
        pd.Timestamp("2024-01-01 01:00", tz="UTC")
    ]
    first, last = required_hour_anchor_bounds(legs)
    assert first == last == pd.Timestamp("2024-01-01 01:00", tz="UTC")


def test_interpolation_rejects_mixed_vessels_before_timestamp_deduplication() -> None:
    points = pd.DataFrame(
        {
            "vessel_key": ["1", "2"],
            "UTC": ["2024-01-01 00:00", "2024-01-01 00:00"],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
        }
    )
    target = pd.Series(pd.to_datetime(["2024-01-01 00:00"], utc=True))
    import pytest

    with pytest.raises(ValueError, match="exactly one nonblank vessel_key"):
        interpolate_positions(points, target)


def test_interpolation_requires_explicit_vessel_key() -> None:
    points = pd.DataFrame(
        {"UTC": ["2024-01-01 00:00"], "lat": [0.0], "lon": [0.0]}
    )
    target = pd.Series(pd.to_datetime(["2024-01-01 00:00"], utc=True))
    import pytest

    with pytest.raises(ValueError, match="vessel_key"):
        interpolate_positions(points, target)


def test_interpolation_ignores_target_index_and_normalizes_exact_180() -> None:
    points = pd.DataFrame(
        {
            "vessel_key": ["1", "1"],
            "UTC": ["2024-01-01 00:00", "2024-01-01 02:00"],
            "lat": [0.0, 2.0],
            "lon": [180.0, -178.0],
        }
    )
    targets = pd.Series(
        pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"], utc=True),
        index=[99, 123],
    )
    result = interpolate_positions(points, targets)
    assert result.index.tolist() == [0, 1]
    assert result.loc[0, "lon"] == -180.0
    assert np.isclose(result.loc[1, "lat"], 1.0)
    assert np.isclose(result.loc[1, "lon"], -179.0)
    assert result["lon"].between(-180.0, 180.0, inclusive="left").all()
