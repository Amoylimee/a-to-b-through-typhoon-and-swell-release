import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'pipeline/40_construct_variables/60_construct_voyage_weather_facts'))
from wave_support_bounds import physical_caps, eligible_grid_indices, matched_support_wave_bounds, NS_HOUR
from weather_core.geo import lon_lat_to_unit_xyz


def grid(lat, lon):
    xyz = lon_lat_to_unit_xyz(lon, lat)
    return SimpleNamespace(codes=np.arange(len(lat)), latitude=np.asarray(lat), longitude=np.asarray(lon), xyz=xyz, tree=cKDTree(xyz))


def row(left, right, hour):
    return SimpleNamespace(position_time_ns=hour,
        left_time_ns=left[0], left_lat=left[1], left_lon=left[2],
        right_time_ns=right[0], right_lat=right[1], right_lon=right[2])


def test_matching_margin_does_not_rescue_physically_impossible_voyage():
    r = row((0, 0, 0), (4 * NS_HOUR, 0, 3), 2 * NS_HOUR)
    assert physical_caps(r)[1] == 'speed_infeasible'
    assert matched_support_wave_bounds(r, pd.Timestamp(0, tz='UTC'), None, spatial_margin_km=117.94)[2] == 'speed_infeasible'


def test_coastal_matched_cell_is_included_but_distant_cell_is_not():
    h = pd.Timestamp('2025-07-02T04:00:00Z')
    r = row((h.value - int(3.5 * NS_HOUR), 53.875026, 9.118402),
            (h.value + int(1.5 * NS_HOUR), 53.550060, 9.820056), h.value)
    g = grid([53.5, 54, 52], [8.5, 11, 3])
    matcher = SimpleNamespace(grid=g, times=pd.DatetimeIndex([h]), _values_for_index=lambda _:np.array([0.18436,0.15995,9.]))
    lo, hi, status = matched_support_wave_bounds(r, h, matcher, spatial_margin_km=117.94)
    assert status == 'matched' and lo <= .18436 <= hi < 9


def test_date_line_cap_and_polar_cap_enumeration():
    g = grid([0,0,0,89.8,89.8], [179.5,-179.5,170,0,180])
    assert set(eligible_grid_indices(g,[(0,180,60)],0)) == {0,1}
    assert set(eligible_grid_indices(g,[(90,0,30)],0)) == {3,4}


def test_no_finite_wet_support_is_missing_instead_of_zero():
    h = pd.Timestamp('2025-01-01T01:00:00Z');r=row((h.value-NS_HOUR,0,0),(h.value+NS_HOUR,0,0),h.value)
    matcher = SimpleNamespace(grid=grid([0],[0]),times=pd.DatetimeIndex([h]),_values_for_index=lambda _:np.array([np.nan]))
    lo,hi,status=matched_support_wave_bounds(r,h,matcher,spatial_margin_km=117.94)
    assert status=='no_wet_era5_cells' and np.isnan(lo) and np.isnan(hi)


def test_full_sphere_margin_is_not_wrapped_to_small_chord():
    g=grid([0,0],[0,180]);assert set(eligible_grid_indices(g,[(0,0,50000)],117.94))=={0,1}


def test_eligible_cell_set_is_nested_as_margin_increases():
    g=grid([0,0,0,0],[0,.5,1,2]);caps=[(0,0,60),(0,1,60)]
    small=set(eligible_grid_indices(g,caps,0));large=set(eligible_grid_indices(g,caps,117.94))
    assert small and small.issubset(large) and 3 not in large
