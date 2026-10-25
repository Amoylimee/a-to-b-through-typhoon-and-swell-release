from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weather_core.era5_match import Era5MonthlyMatcher, match_era5_hour_arrays
from weather_core.geo import EARTH_RADIUS_KM


def _longitude_delta_for_km(distance_km: float) -> float:
    return np.degrees(distance_km / EARTH_RADIUS_KM)


def _coastal_block():
    return (
        np.array([0.0]),
        np.array([0.0, 0.5, 1.0, 1.5]),
        np.array([[4.5, np.nan, np.nan, np.nan]]),
        "coast",
    )


def test_nearest_finite_nominal_cell_wins() -> None:
    block = (np.array([0.0]), np.array([0.0, 0.5]), np.array([[2.0, 5.0]]), "a")
    result = match_era5_hour_arrays(0.0, 0.1, [block])
    assert result.swh_m == 2.0
    assert result.match_status == "nearest_grid"


def test_noncovering_north_polar_tile_cannot_supply_equatorial_nominal() -> None:
    north = (np.array([80.0, 80.5]), np.array([0.0, 0.5]), np.ones((2, 2)), "north")
    result = match_era5_hour_arrays(0.0, 0.0, [north])
    assert np.isnan(result.swh_m)
    assert result.match_status == "outside_assets"


@pytest.mark.parametrize(
    ("distance_km", "matched"),
    [(117.93, True), (117.94, True), (117.95, False)],
)
def test_wet_cell_hard_cap_below_equal_above(distance_km: float, matched: bool) -> None:
    result = match_era5_hour_arrays(
        0.0,
        _longitude_delta_for_km(distance_km),
        [_coastal_block()],
        maximum_wet_cell_snap_km=117.94,
    )
    if matched:
        assert result.swh_m == 4.5
        assert result.match_status == "nearest_finite_wet_cell"
        assert np.isclose(result.snap_distance_km, distance_km, atol=1e-7)
    else:
        assert np.isnan(result.swh_m)
        assert result.match_status == "wet_cell_beyond_cap"


def test_multiple_blocks_compete_only_when_their_domain_covers_position() -> None:
    south = (np.array([-85.0, -84.5]), np.array([0.0, 0.5]), np.full((2, 2), 7.0), "south")
    main = (np.array([-40.0, -39.5]), np.array([0.0, 0.5]), np.full((2, 2), 2.0), "main")
    result = match_era5_hour_arrays(-84.9, 0.0, [main, south])
    assert result.swh_m == 7.0
    assert result.source_path == "south"


def test_raw_invalid_longitude_is_not_silently_wrapped() -> None:
    block = (np.array([0.0]), np.array([-81.0, -80.5]), np.ones((1, 2)), "a")
    result = match_era5_hour_arrays(0.0, 999.0, [block])
    assert np.isnan(result.swh_m)
    assert result.match_status == "invalid_position"


def test_global_seam_and_source_tie_are_path_order_invariant() -> None:
    first = (
        np.array([0.0]), np.array([-180.0, -179.5]), np.array([[3.0, 2.0]]), "a"
    )
    second = (
        np.array([0.0]), np.array([179.5, 180.0]), np.array([[2.0, 3.0]]), "b"
    )
    left = match_era5_hour_arrays(0.0, 180.0, [first, second])
    right = match_era5_hour_arrays(0.0, 180.0, [second, first])
    assert left == right
    assert left.matched_lon == -180.0
    assert left.source_path == "a"


@pytest.mark.parametrize(
    "second_value",
    [np.nan, 3.1, 4.000001],
)
def test_duplicate_seam_mask_value_and_threshold_conflicts_fail(second_value: float) -> None:
    first = (np.array([0.0]), np.array([-180.0]), np.array([[3.999999]]), "a")
    second = (np.array([0.0]), np.array([180.0]), np.array([[second_value]]), "b")
    with pytest.raises(ValueError, match="overlapping ERA5 tiles"):
        match_era5_hour_arrays(0.0, 180.0, [first, second])


def _write_month(path: Path, *, members: int = 1) -> None:
    times = pd.date_range("2024-01-01", periods=2, freq="h")
    value = np.arange(2 * members * 2 * 2, dtype=np.float32).reshape(2, members, 2, 2)
    xr.Dataset(
        {
            "swh": (("valid_time", "member", "latitude", "longitude"), value),
        },
        coords={
            "valid_time": times,
            "member": np.arange(members),
            "latitude": [0.5, 0.0],
            "longitude": [0.0, 0.5],
        },
    ).to_netcdf(path)


def test_monthly_matcher_drops_singleton_extra_dim_caches_hour_and_handles_empty(
    tmp_path: Path,
) -> None:
    path = tmp_path / "month.nc"
    _write_month(path)
    with Era5MonthlyMatcher([path]) as matcher:
        empty = matcher.match(
            pd.DataFrame(
                {
                    "hour_anchor_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "lat": pd.Series(dtype=float), "lon": pd.Series(dtype=float),
                }
            )
        )
        assert empty.empty and "match_status" in empty
        positions = pd.DataFrame(
            {
                "hour_anchor_utc": pd.to_datetime(["2024-01-01 00:00"] * 2, utc=True),
                "lat": [0.0, 0.5], "lon": [0.0, 0.5],
            }
        )
        first = matcher.match(positions)
        second = matcher.match(positions.iloc[:1])
        assert first["swh_m"].notna().all()
        assert second.loc[0, "swh_m"] == first.loc[0, "swh_m"]
        assert list(matcher._cache) == [0]


def test_monthly_matcher_rejects_fractional_second_anchor(tmp_path: Path) -> None:
    path = tmp_path / "month.nc"
    _write_month(path)
    with Era5MonthlyMatcher([path]) as matcher:
        positions = pd.DataFrame(
            {
                "hour_anchor_utc": [pd.Timestamp("2024-01-01 00:00:00.500", tz="UTC")],
                "lat": [0.0], "lon": [0.0],
            }
        )
        with pytest.raises(ValueError, match="nanoseconds"):
            matcher.match(positions)


def test_monthly_matcher_rejects_non_singleton_extra_dim(tmp_path: Path) -> None:
    path = tmp_path / "month.nc"
    _write_month(path, members=2)
    with pytest.raises(ValueError, match="not singleton"):
        Era5MonthlyMatcher([path])


def test_grid_validation_rejects_non_half_degree_step() -> None:
    bad = (np.array([0.0]), np.array([0.0, 0.6]), np.ones((1, 2)), "bad")
    with pytest.raises(ValueError, match="0.5 degree"):
        match_era5_hour_arrays(0.0, 0.0, [bad])


def test_grid_validation_rejects_half_degree_step_on_offset_lattice() -> None:
    bad = (np.array([0.0]), np.array([0.1, 0.6]), np.ones((1, 2)), "bad")
    with pytest.raises(ValueError, match="offset"):
        match_era5_hour_arrays(0.0, 0.1, [bad])


def test_exact_tile_boundary_and_epsilon_are_path_order_invariant() -> None:
    west = (np.array([0.0]), np.array([0.0, 0.5]), np.array([[1.0, 1.0]]), "z-west")
    east = (np.array([0.0]), np.array([1.0, 1.5]), np.array([[2.0, 2.0]]), "a-east")
    for blocks in ([west, east], [east, west]):
        below = match_era5_hour_arrays(0.0, 0.75 - 1e-8, blocks)
        exact = match_era5_hour_arrays(0.0, 0.75, blocks)
        above = match_era5_hour_arrays(0.0, 0.75 + 1e-8, blocks)
        assert below.swh_m == 1.0
        assert exact.swh_m == 1.0
        assert exact.matched_lon == 0.5
        assert above.swh_m == 2.0


def test_infinite_swh_is_invalid_not_missing() -> None:
    bad = (np.array([0.0]), np.array([0.0]), np.array([[np.inf]]), "bad")
    with pytest.raises(ValueError, match="infinite"):
        match_era5_hour_arrays(0.0, 0.0, [bad])


def test_preloading_month_is_exactly_equal_to_lazy_matching(tmp_path):
    path=tmp_path/'month.nc'
    _write_month(path)
    points=pd.DataFrame({'hour_anchor_utc':pd.to_datetime(['2024-01-01 00:00','2024-01-01 01:00'],utc=True),'lat':[0.0,0.5],'lon':[0.0,0.5]})
    with Era5MonthlyMatcher([path]) as lazy:
        expected=lazy.match(points)
    with Era5MonthlyMatcher([path]) as eager:
        assert eager.preload_swh()==32
        actual=eager.match(points)
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
