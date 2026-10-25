"""Guard the geometry and the bound arithmetic behind the exposure probe.

Three things here would be wrong in a way that still looks plausible in a table.
The reachable region must pinch to a point at each fix and be widest in between.
The enumeration must include a cell the region merely clips, because the ship
really could be in the clipped part.  And the two exposure bounds invert in
opposite ways: an hour counts towards the lower bound on rough-water hours only
when even the calmest reachable water clears four metres, and towards the lower
bound on cyclone hours only when every reachable cell is inside the radius.
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
    "e031_bounds", HERE.parent / "30_bound_dark_hour_wave_exposure.py"
)
bounds = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bounds
_spec.loader.exec_module(bounds)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_the_great_circle_distance_matches_a_known_separation():
    """One degree of latitude is 111.19 km on this sphere."""

    km = bounds.great_circle_km(
        np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0])
    )
    assert km[0] == pytest.approx(111.19, abs=0.05)


def test_the_great_circle_point_ends_where_it_should():
    lat, lon = bounds.great_circle_point(0.0, 0.0, 0.0, 10.0, np.array([0.0, 0.5, 1.0]))
    assert lat == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert lon == pytest.approx([0.0, 5.0, 10.0], abs=1e-6)


def test_a_degenerate_gap_returns_its_own_endpoint():
    """Coincident fixes leave the path undefined; the endpoint is the only answer.

    Which branch produces it is not the guarantee -- floating point can leave
    the separation angle just above the degeneracy threshold, in which case the
    ordinary slerp divides by a near-zero sine and renormalises back to the same
    point.  What must hold either way is that no position is invented.
    """

    lat, lon = bounds.great_circle_point(12.0, 43.0, 12.0, 43.0, np.array([0.25, 0.75]))
    assert lat == pytest.approx([12.0, 12.0], abs=1e-9)
    assert lon == pytest.approx([43.0, 43.0], abs=1e-9)


def test_every_enumerated_cell_is_actually_reachable():
    """Both leashes constrain; a cell violating either is a bug.

    The tolerance is the cell half-diagonal, because a cell qualifies when the
    region reaches any part of it, and its centre can sit that much beyond.
    """

    lat_a, lon_a, lat_b, lon_b = 12.0, 43.0, 20.0, 38.0
    r_a, r_b = 900.0, 1100.0
    lat, lon, capped = bounds.reachable_cells(lat_a, lon_a, lat_b, lon_b, r_a, r_b)
    slack = bounds.CELL_HALF_DIAGONAL_KM + 1e-6
    assert not capped
    assert lat.size > 2
    assert np.all(bounds.great_circle_km(lat_a, lon_a, lat, lon) <= r_a + slack)
    assert np.all(bounds.great_circle_km(lat, lon, lat_b, lon_b) <= r_b + slack)


def test_a_short_gap_inside_one_cell_enumerates_exactly_that_cell():
    """The case the whole bounds proposal turns on.

    Both fixes sit at a cell centre and the ship has time to move only a few
    kilometres.  There is then exactly one possible wave value, so the interval
    must be a point.  Testing cell centres against a fixed half-diagonal
    expansion would return nine cells here and manufacture a spread.
    """

    lat, lon, capped = bounds.reachable_cells(10.0, 40.0, 10.0, 40.0, 5.0, 5.0)
    assert not capped
    assert lat.size == 1
    assert (lat[0], lon[0]) == (10.0, 40.0)


def test_a_cap_that_crosses_a_cell_edge_admits_the_neighbour():
    """Reaching into a neighbouring cell is enough, even without its centre.

    A 40 km radius from a cell centre crosses the 27.8 km edge but stops well
    short of the 55.6 km neighbouring centre.  The ship can be in that cell, so
    the cell counts.
    """

    lat, lon, _ = bounds.reachable_cells(10.0, 40.0, 10.0, 40.0, 40.0, 40.0)
    assert lat.size > 1
    assert bounds.great_circle_km(10.0, 40.0, lat, lon).max() > 40.0


def test_enumerated_cells_sit_on_the_era5_grid():
    """Each enumerated position must be a cell centre, or it is not a cell."""

    lat, lon, _ = bounds.reachable_cells(12.0, 43.0, 20.0, 38.0, 900.0, 1100.0)
    step = bounds.ERA5_CELL_DEGREES
    assert np.allclose(np.remainder(lat, step), 0.0, atol=1e-9)
    assert np.allclose(np.remainder(lon, step), 0.0, atol=1e-9)


def test_the_enumeration_has_no_duplicates():
    """A duplicated cell would weight one possible value twice."""

    lat, lon, _ = bounds.reachable_cells(0.0, 0.0, 0.0, 6.0, 800.0, 800.0)
    pairs = {(round(a, 6), round(b, 6)) for a, b in zip(lat, lon)}
    assert len(pairs) == lat.size


def test_the_region_pinches_at_the_ends_and_widens_in_the_middle():
    """Both leashes are short at the ends and slack in the middle."""

    a, b = (0.0, 0.0), (0.0, 10.0)
    total_km = 2000.0
    sizes = []
    for elapsed_share in (0.05, 0.5, 0.95):
        lat, _, _ = bounds.reachable_cells(
            *a, *b, total_km * elapsed_share, total_km * (1.0 - elapsed_share)
        )
        sizes.append(lat.size)
    assert sizes[1] > sizes[0], "the middle must be wider than the start"
    assert sizes[1] > sizes[2], "the middle must be wider than the end"


def test_a_very_large_region_is_capped_and_says_so():
    lat, _, capped = bounds.reachable_cells(0.0, 0.0, 0.0, 5.0, 9000.0, 9000.0)
    assert capped is True
    assert lat.size <= bounds.MAX_CELLS_PER_HOUR


def test_a_region_inside_the_cap_is_never_flagged():
    _, _, capped = bounds.reachable_cells(0.0, 0.0, 0.0, 2.0, 300.0, 300.0)
    assert capped is False


def test_a_gap_across_the_date_line_stays_contiguous():
    """Unwrapping around A keeps the region from splitting at +/-180 degrees."""

    lat, lon, _ = bounds.reachable_cells(10.0, 179.0, 10.0, -179.0, 800.0, 800.0)
    assert lat.size > 2
    separation = np.abs(((lon - 179.0 + 180.0) % 360.0) - 180.0)
    assert separation.max() < 20.0


# ---------------------------------------------------------------------------
# bound arithmetic
# ---------------------------------------------------------------------------


def _matched(rows):
    """rows: (hour, swh, is_direct, within500, within300, tc_km, wind)."""

    return pd.DataFrame(
        [
            {
                "candidate_id": "c0",
                "gap_ordinal": 0,
                "hour_anchor_utc": pd.Timestamp(h, tz="UTC"),
                "swh_m": swh,
                "is_direct_path": direct,
                "cells_capped": False,
                "within_500": w500,
                "within_300": w300,
                "tc_min_distance_km": km,
                "tc_max_wind_kt": wind,
            }
            for h, swh, direct, w500, w300, km, wind in rows
        ]
    )


def test_the_interval_spans_the_enumerated_cells():
    out = bounds.summarise(
        _matched(
            [
                ("2024-03-01 01:00", 2.0, False, False, False, np.nan, 0.0),
                ("2024-03-01 01:00", 5.0, False, False, False, np.nan, 0.0),
                ("2024-03-01 01:00", 3.0, False, False, False, np.nan, 0.0),
                ("2024-03-01 01:00", 2.5, True, False, False, np.nan, 0.0),
            ]
        )
    )
    row = out.iloc[0]
    assert row["swh_lo_mean"] == pytest.approx(2.0)
    assert row["swh_hi_mean"] == pytest.approx(5.0)
    assert row["swh_mid_mean"] == pytest.approx(3.0)
    assert row["width_mean_m"] == pytest.approx(3.0)
    assert row["cells_median"] == 3
    assert row["direct_inside_share"] == pytest.approx(1.0)


def test_a_single_reachable_cell_gives_a_point_interval():
    """One cell means one possible value, so the width is a true zero."""

    out = bounds.summarise(
        _matched(
            [
                ("2024-03-01 01:00", 1.7, False, False, False, np.nan, 0.0),
                ("2024-03-01 01:00", 1.7, True, False, False, np.nan, 0.0),
            ]
        )
    )
    row = out.iloc[0]
    assert row["cells_median"] == 1
    assert row["single_cell_share"] == pytest.approx(1.0)
    assert row["width_mean_m"] == pytest.approx(0.0)


def test_high_wave_hours_bound_inverts():
    """Hour 1 must exceed 4 m; hour 2 only could; hour 3 cannot."""

    out = bounds.summarise(
        _matched(
            [
                ("2024-03-01 01:00", 4.5, False, False, False, np.nan, 0.0),
                ("2024-03-01 01:00", 5.5, False, False, False, np.nan, 0.0),
                ("2024-03-01 02:00", 1.0, False, False, False, np.nan, 0.0),
                ("2024-03-01 02:00", 4.2, False, False, False, np.nan, 0.0),
                ("2024-03-01 03:00", 1.0, False, False, False, np.nan, 0.0),
                ("2024-03-01 03:00", 2.0, False, False, False, np.nan, 0.0),
            ]
        )
    )
    row = out.iloc[0]
    assert row["hours_bounded"] == 3
    assert row["high_wave_hours_lo"] == 1, "only hour 1 clears 4 m at its calmest"
    assert row["high_wave_hours_hi"] == 2, "hours 1 and 2 clear 4 m at their roughest"


def test_cyclone_hours_split_three_ways():
    """Certain, impossible, and undetermined are three different answers.

    Hour 1: every reachable cell is inside 500 km, so the ship was exposed
    whatever it did.  Hour 2: some cells in, some out -- the evidence does not
    settle it.  Hour 3: no cell is inside, so it was not exposed.
    """

    out = bounds.summarise(
        _matched(
            [
                ("2024-09-01 01:00", 3.0, False, True, False, 400.0, 60.0),
                ("2024-09-01 01:00", 3.1, False, True, True, 250.0, 70.0),
                ("2024-09-01 02:00", 3.0, False, True, False, 480.0, 55.0),
                ("2024-09-01 02:00", 3.1, False, False, False, np.nan, 0.0),
                ("2024-09-01 03:00", 3.0, False, False, False, np.nan, 0.0),
                ("2024-09-01 03:00", 3.1, False, False, False, np.nan, 0.0),
            ]
        )
    )
    row = out.iloc[0]
    assert row["tc500_hours_lo"] == 1, "only hour 1 is exposed under every position"
    assert row["tc500_hours_hi"] == 2, "hours 1 and 2 are exposed under some position"
    assert row["tc_undetermined_hours"] == 1, "hour 2 alone is undetermined"
    assert row["tc300_hours_lo"] == 0, "hour 1 is inside 300 km only on one cell"
    assert row["tc300_hours_hi"] == 1


def test_closest_approach_takes_the_minimum_across_hours_on_both_bounds():
    """The variable is a closest approach, so both bounds minimise over hours.

    The upper bound is not the largest distance seen; it is the closest the ship
    was forced to come, which is the minimum of each hour's least favourable
    cell.
    """

    out = bounds.summarise(
        _matched(
            [
                ("2024-09-01 01:00", 3.0, False, True, False, 100.0, 60.0),
                ("2024-09-01 01:00", 3.0, False, True, False, 460.0, 40.0),
                ("2024-09-01 02:00", 3.0, False, True, False, 300.0, 50.0),
                ("2024-09-01 02:00", 3.0, False, True, False, 320.0, 45.0),
            ]
        )
    )
    row = out.iloc[0]
    assert row["tc_min_distance_lo"] == pytest.approx(100.0)
    assert row["tc_min_distance_hi"] == pytest.approx(320.0)


def test_storm_wind_bounds_span_the_reachable_cells():
    out = bounds.summarise(
        _matched(
            [
                ("2024-09-01 01:00", 3.0, False, True, False, 400.0, 65.0),
                ("2024-09-01 01:00", 3.0, False, False, False, np.nan, 0.0),
            ]
        )
    )
    row = out.iloc[0]
    assert row["tc_max_wind_lo"] == pytest.approx(0.0), "a cell out of range sees no wind"
    assert row["tc_max_wind_hi"] == pytest.approx(65.0)


def test_an_hour_whose_cells_are_all_dry_never_reaches_the_summary():
    """Dry cells are dropped upstream, so such an hour simply has no evidence."""

    out = bounds.summarise(
        _matched([("2024-03-01 01:00", 2.0, True, False, False, np.nan, 0.0)])
    )
    assert out.empty


def test_the_probe_pins_the_contract_constants():
    assert bounds.MAXIMUM_INTERPOLATION_GAP_HOURS == 4.0
    assert bounds.HIGH_WAVE_THRESHOLD_M == 4.0
    assert bounds.TC_RADIUS_500_KM == 500.0
    assert bounds.TC_RADIUS_300_KM == 300.0
    assert bounds.TC_MAXIMUM_FIX_GAP_HOURS == 3.0
    assert bounds.ERA5_CELL_DEGREES == 0.5
    assert bounds.NO_WET_CELL_SNAP_KM == 0.0, "a dry cell must drop, never snap"


def test_a_region_never_excludes_a_path_the_ship_provably_took():
    """A leg faster than its own cap must still contain its own straight line.

    Roughly one leg in eighty needs more speed than the cap estimated from its
    other intervals -- a bad fix, a shared MMSI, occasionally a genuinely quick
    passage.  Sizing the region from the cap alone would put the observed
    endpoint outside the set of places the ship could have been, which is not a
    bound at all.  The straight line wins, and the leg is flagged.
    """

    # Two fixes 500 km apart four hours apart demand about 67 knots.
    direct_km = bounds.great_circle_km(0.0, 0.0, 0.0, 4.5)
    implied_kn = float(direct_km / bounds.KM_PER_NAUTICAL_MILE / 4.0)
    assert implied_kn > 25.0, "the fixture must exceed the accepted ceiling"

    capped_speed_km_h = 25.0 * bounds.KM_PER_NAUTICAL_MILE
    lat, lon, _ = bounds.reachable_cells(
        0.0, 0.0, 0.0, 4.5, capped_speed_km_h * 2.0, capped_speed_km_h * 2.0
    )
    midpoint_missing = bounds.great_circle_km(0.0, 2.25, lat, lon).min() > 55.0
    assert midpoint_missing, "the cap alone cannot reach the halfway point"

    honest_speed_km_h = implied_kn * bounds.KM_PER_NAUTICAL_MILE
    lat, lon, _ = bounds.reachable_cells(
        0.0, 0.0, 0.0, 4.5, honest_speed_km_h * 2.0, honest_speed_km_h * 2.0
    )
    assert bounds.great_circle_km(0.0, 2.25, lat, lon).min() <= 55.0


def test_a_zero_slack_region_pinches_onto_the_great_circle_without_vanishing():
    """No slack means the path is determined, not that the region is empty.

    Every cell then sits exactly on the boundary, where haversine rounding can
    drop the lot.  The cells along the direct line must survive.
    """

    direct_km = float(bounds.great_circle_km(0.0, 0.0, 0.0, 4.5))
    half = direct_km / 2.0
    lat, lon, _ = bounds.reachable_cells(0.0, 0.0, 0.0, 4.5, half, half)
    assert lat.size >= 1
    assert bounds.great_circle_km(0.0, 2.25, lat, lon).min() <= 55.6
