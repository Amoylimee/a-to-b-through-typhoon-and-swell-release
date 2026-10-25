"""Small spherical-geometry primitives shared by weather matchers."""

from __future__ import annotations

import numpy as np


EARTH_RADIUS_KM = 6371.0088


def normalize_longitude(value):
    return ((np.asarray(value, dtype=np.float64) + 180.0) % 360.0) - 180.0


def lon_lat_to_unit_xyz(lon, lat) -> np.ndarray:
    lon_rad = np.radians(normalize_longitude(lon))
    lat_rad = np.radians(np.asarray(lat, dtype=np.float64))
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)]
    )


def chord_for_km(distance_km: float) -> float:
    return float(2.0 * np.sin(float(distance_km) / (2.0 * EARTH_RADIUS_KM)))


def great_circle_km(lon1, lat1, lon2, lat2):
    first = lon_lat_to_unit_xyz(np.atleast_1d(lon1), np.atleast_1d(lat1))
    second = lon_lat_to_unit_xyz(np.atleast_1d(lon2), np.atleast_1d(lat2))
    dot = np.sum(first * second, axis=1).clip(-1.0, 1.0)
    distance = EARTH_RADIUS_KM * np.arccos(dot)
    return distance[0] if np.ndim(lon1) == 0 and np.ndim(lon2) == 0 else distance
