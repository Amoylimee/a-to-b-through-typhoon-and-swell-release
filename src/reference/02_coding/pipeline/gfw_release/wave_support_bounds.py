"""Conservative wave bounds for the support of the existing ERA5 matcher.

Physical feasibility is evaluated first with the original 25-kn endpoints.
The 117.94-km spatial margin belongs to weather matching, never sailing speed
or elapsed time. Enumerate actual canonical grid centres exactly through its
spherical KD-tree; avoid approximate longitude boxes or grid thinning.
"""
from __future__ import annotations
from functools import partial
import json
from pathlib import Path
import numpy as np
from e007_core import haversine_km

HERE = Path(__file__).resolve().parent
CONFIG = HERE / 'config/wave_support_revision.json'
WEATHER = HERE.parents[1] / 'pipeline/40_construct_variables/60_construct_voyage_weather_facts'
NS_HOUR = 3_600_000_000_000
SPEED_KM_H = 25 * 1.852
EARTH_KM = 6371.0088
EPS_KM = 1e-6


def binding():
    rule = json.loads(CONFIG.read_text())
    weather = json.loads((WEATHER / 'config/2025_ten_minute_weather.json').read_text())
    required = {'speed_cap_knots': 25, 'maximum_wet_cell_snap_km': 117.94,
                'fix_time_tolerance_hours': 0.5, 'port_time_tolerance_hours': 1,
                'primary_wave_estimand': 'matched_mean'}
    if any(rule.get(k) != v for k, v in required.items()):
        raise ValueError('wave support revision differs from the authorized scope')
    if weather['maximum_wet_cell_snap_km'] != rule['maximum_wet_cell_snap_km']:
        raise ValueError('weather matcher and bound support margins differ')
    return rule


def physical_caps(row):
    if row is None:
        return [], 'unbounded'
    event = int(row.position_time_ns)
    ends = []
    for side in ['left', 'right']:
        stamp = getattr(row, f'{side}_time_ns')
        if stamp is not None and np.isfinite(stamp) and stamp >= 0:
            ends.append((side, int(stamp), float(getattr(row, f'{side}_lat')),
                         float(getattr(row, f'{side}_lon'))))
    if not ends:
        return [], 'unbounded'
    if len(ends) == 2:
        left, right = ends
        if not left[1] < event < right[1]:
            raise ValueError('unknown hour outside its two physical endpoints')
        allowed = SPEED_KM_H * (right[1] - left[1]) / NS_HOUR
        if float(haversine_km(left[2], left[3], right[2], right[3])) > allowed + EPS_KM:
            return [], 'speed_infeasible'
    return [(lat, lon, SPEED_KM_H * max(0, event - stamp if side == 'left' else stamp - event) / NS_HOUR)
            for side, stamp, lat, lon in ends], 'feasible'


def eligible_grid_indices(grid, caps, spatial_margin_km):
    """Exact grid-centre intersection, including date-line and polar regions."""
    expanded = sorted([(lat, lon, min(np.pi * EARTH_KM, radius + spatial_margin_km + EPS_KM))
                       for lat, lon, radius in caps], key=lambda x: x[2])
    if not expanded:
        return np.arange(len(grid.codes))
    lat, lon, radius = expanded[0]
    from weather_core.geo import lon_lat_to_unit_xyz
    focus = lon_lat_to_unit_xyz([lon], [lat])[0]
    if radius >= np.pi * EARTH_KM:
        idx = np.arange(len(grid.codes))
    else:
        chord = 2 * np.sin(radius / (2 * EARTH_KM))
        idx = np.asarray(grid.tree.query_ball_point(focus, chord), dtype=np.int64)
    for lat, lon, radius in expanded[1:]:
        if radius < np.pi * EARTH_KM and len(idx):
            other = lon_lat_to_unit_xyz([lon], [lat])[0]
            idx = idx[(grid.xyz[idx] @ other) >= np.cos(radius / EARTH_KM) - 1e-15]
    return np.sort(idx)


def matched_support_wave_bounds(row, hour, matcher, *, spatial_margin_km):
    caps, status = physical_caps(row)
    if status != 'feasible':
        return np.nan, np.nan, status
    index = int(matcher.times.get_indexer([hour])[0])
    if index < 0:
        raise ValueError(f'frozen ERA5 lacks hour {hour}')
    cells = eligible_grid_indices(matcher.grid, caps, spatial_margin_km)
    values = matcher._values_for_index(index)[cells]
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, 'no_wet_era5_cells'
    return float(values.min()), float(values.max()), 'matched'


def install(kernels):
    """Bind this experiment's adapter without modifying the accepted E002 file."""
    rule = binding()
    kernels._wave_bounds = partial(matched_support_wave_bounds,
        spatial_margin_km=rule['maximum_wet_cell_snap_km'])
    kernels.wave_support_revision = dict(rule)
    return kernels
