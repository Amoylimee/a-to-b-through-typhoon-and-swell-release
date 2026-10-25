import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e007_core import NS_HOUR, SPEED_KM_H, haversine_km, interpolate_position, reachable_contains  # noqa: E402


def test_haversine_one_degree_of_latitude():
    assert abs(haversine_km(0, 0, 1, 0) - 111.195) < 0.01


def test_interpolation_takes_the_short_way_across_the_date_line():
    lat, lon = interpolate_position(10.0, 179.0, 12.0, -179.0, 0.5)
    assert np.isclose(lat, 11.0)
    assert np.isclose(abs(lon), 180.0)
    lat, lon = interpolate_position(0.0, 179.5, 0.0, -179.5, 0.25)
    assert np.isclose(lon, 179.75)


def test_reachable_area_contains_the_straight_path_and_excludes_far_points():
    t0, t1 = 0, 10 * NS_HOUR
    a = (0.0, 0.0)
    b = (0.0, 2.0)  # about 222 km, well inside 10 h at 25 kn
    mid = 5 * NS_HOUR
    inside, ok = reachable_contains(0.0, 1.0, mid, *a, t0, *b, t1)
    assert ok and inside
    far_km = SPEED_KM_H * 5 + 50
    inside, ok = reachable_contains(far_km / 111.195, 1.0, mid, *a, t0, *b, t1)
    assert ok and not inside


def test_infeasible_gap_ends_contain_nothing():
    inside, ok = reachable_contains(0.0, 5.0, NS_HOUR // 2, 0.0, 0.0, 0, 0.0, 10.0, NS_HOUR)
    assert not ok and not inside
