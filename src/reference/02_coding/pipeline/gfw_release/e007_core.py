"""Shared geometry for E007: distances, interpolation on the ten-minute grid, reachable-area tests."""
from __future__ import annotations

import numpy as np

EARTH_KM = 6371.0088
NS_MIN10 = 600_000_000_000
NS_HOUR = 3_600_000_000_000
# Production cap for reconstruction and missing-position bounds (CONTRACT §3 E002 缺失上下限).
SPEED_KM_H = 25.0 * 1.852


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def interpolate_position(lat0, lon0, lat1, lon1, fraction):
    """Linear in latitude and along the shorter longitude arc, as between adjacent ten-minute cells."""
    lat0, lon0, lat1, lon1, fraction = (np.asarray(x, dtype=float) for x in (lat0, lon0, lat1, lon1, fraction))
    dlon = lon1 - lon0
    dlon = np.where(dlon > 180.0, dlon - 360.0, np.where(dlon < -180.0, dlon + 360.0, dlon))
    lon = lon0 + fraction * dlon
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat0 + fraction * (lat1 - lat0), lon


def reachable_contains(lat, lon, t_ns, a_lat, a_lon, a_t_ns, b_lat, b_lon, b_t_ns, speed_km_h=SPEED_KM_H):
    """Whether (lat, lon) at t lies in {x : d(A,x) <= v(t-tA), d(x,B) <= v(tB-t)}.

    Returns (contained, feasible): feasible is False when A and B themselves are farther apart
    than the speed cap allows, in which case the production wave bound has no reachable cell.
    """
    to_h = 1.0 / NS_HOUR
    reach_a = speed_km_h * (np.asarray(t_ns, dtype=np.int64) - np.asarray(a_t_ns, dtype=np.int64)) * to_h
    reach_b = speed_km_h * (np.asarray(b_t_ns, dtype=np.int64) - np.asarray(t_ns, dtype=np.int64)) * to_h
    feasible = haversine_km(a_lat, a_lon, b_lat, b_lon) <= speed_km_h * (
        np.asarray(b_t_ns, dtype=np.int64) - np.asarray(a_t_ns, dtype=np.int64)) * to_h + 1e-6
    inside = (haversine_km(a_lat, a_lon, lat, lon) <= reach_a + 1e-6) & (haversine_km(lat, lon, b_lat, b_lon) <= reach_b + 1e-6)
    return inside & feasible, feasible
