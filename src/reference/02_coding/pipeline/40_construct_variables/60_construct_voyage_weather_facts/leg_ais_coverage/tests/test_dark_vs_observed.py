"""Guard the seen-versus-unseen comparison.

The comparison is only meaningful if "seen" means exactly what the accepted
pipeline means by it.  An hour counts when the surrounding AIS gap is at most
four hours, so an hour sitting inside a longer gap must produce no position at
all rather than a plausible-looking interpolation across it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "e031_compare", HERE.parent / "40_compare_dark_and_observed_weather.py"
)
compare = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = compare
_spec.loader.exec_module(compare)

interpolate = compare.interpolate_positions


def _track(hours_and_positions):
    return pd.DataFrame(
        [
            {
                "vessel_key": "v0",
                "UTC": pd.Timestamp(f"2024-03-01 {h:02d}:00", tz="UTC"),
                "lat": lat,
                "lon": lon,
            }
            for h, lat, lon in hours_and_positions
        ]
    )


def _targets(hours):
    return pd.Series(
        [pd.Timestamp(f"2024-03-01 {h:02d}:00", tz="UTC") for h in hours]
    )


def test_an_hour_inside_a_short_gap_is_seen():
    track = _track([(0, 10.0, 20.0), (3, 10.3, 20.3)])
    out = interpolate(track, _targets([1, 2]), compare.MAXIMUM_INTERPOLATION_GAP_HOURS)
    assert out["lat"].notna().all()
    assert out["lat"].iloc[0] == pytest.approx(10.1)


def test_an_hour_inside_a_long_gap_is_never_placed():
    """Five hours apart exceeds the limit, so nothing between them is seen."""

    track = _track([(0, 10.0, 20.0), (5, 10.5, 20.5)])
    out = interpolate(track, _targets([1, 2, 3, 4]), compare.MAXIMUM_INTERPOLATION_GAP_HOURS)
    assert out["lat"].isna().all(), "interpolating across a dark gap would invent a track"


def test_a_gap_exactly_on_the_limit_still_counts_as_seen():
    track = _track([(0, 10.0, 20.0), (4, 10.4, 20.4)])
    out = interpolate(track, _targets([2]), compare.MAXIMUM_INTERPOLATION_GAP_HOURS)
    assert out["lat"].notna().all()


def test_the_two_sides_use_opposite_snap_rules_on_purpose():
    """A known position snaps to the nearest wet cell; a possible one does not."""

    assert compare.ACCEPTED_WET_CELL_SNAP_KM == 117.94
    assert compare.MAXIMUM_INTERPOLATION_GAP_HOURS == 4.0
    assert compare.HIGH_WAVE_THRESHOLD_M == 4.0
