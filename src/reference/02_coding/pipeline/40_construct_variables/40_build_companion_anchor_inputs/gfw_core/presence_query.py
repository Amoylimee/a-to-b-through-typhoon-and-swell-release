"""Conservative daily spatial queries for post-break GFW Vessel Presence.

The report endpoint cannot filter directly by MMSI.  For each bounded target
hour we therefore construct an outer spatial query.  Any position reachable at
some instant during that UTC hour must lie inside both of these spherical caps:

* the cap around the pre-gap fix at the radius reached by the end of the hour;
* the cap around the post-gap fix at the radius reachable at the start of the hour.

Intersecting the caps' latitude/longitude bounding boxes is deliberately loose,
then snapping outward to one-degree tiles is looser again.  It may download
irrelevant traffic, but it cannot discard a feasible GFW grid centre merely to
save cost.  Exact two-sided motion is checked only after the response arrives.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from .dark_gaps import EARTH_RADIUS_KM, KM_PER_NAUTICAL_MILE
from .presence_overlap import ONE_HOUR, utc_timestamp


LonInterval = tuple[float, float]
Tile = tuple[int, int]


def _normalise_lon(value: float) -> float:
    result = (float(value) + 180.0) % 360.0 - 180.0
    return -180.0 if math.isclose(result, 180.0) else result


def _split_lon_interval(center: float, half_width: float) -> list[LonInterval]:
    if half_width >= 180.0 - 1e-12:
        return [(-180.0, 180.0)]
    center = _normalise_lon(center)
    left = center - half_width
    right = center + half_width
    if left < -180.0:
        return [(left + 360.0, 180.0), (-180.0, right)]
    if right > 180.0:
        return [(left, 180.0), (-180.0, right - 360.0)]
    return [(left, right)]


def spherical_cap_bbox(
    lat: float,
    lon: float,
    radius_km: float,
) -> tuple[float, float, list[LonInterval]]:
    """Return a conservative lon/lat bounding box for a spherical cap."""

    if not -90.0 <= float(lat) <= 90.0:
        raise ValueError("cap latitude is outside [-90, 90]")
    if radius_km < 0:
        raise ValueError("cap radius cannot be negative")
    angular = min(float(radius_km) / EARTH_RADIUS_KM, math.pi)
    angular_deg = math.degrees(angular)
    south = max(-90.0, float(lat) - angular_deg)
    north = min(90.0, float(lat) + angular_deg)
    reaches_pole = south <= -90.0 + 1e-12 or north >= 90.0 - 1e-12
    if reaches_pole or angular >= math.pi - 1e-12:
        return south, north, [(-180.0, 180.0)]
    ratio = math.sin(angular) / max(math.cos(math.radians(float(lat))), 1e-15)
    if ratio >= 1.0:
        half_width = 180.0
    else:
        half_width = math.degrees(math.asin(max(-1.0, min(1.0, ratio))))
    return south, north, _split_lon_interval(float(lon), half_width)


def _intersect_lon_intervals(
    first: Iterable[LonInterval],
    second: Iterable[LonInterval],
) -> list[LonInterval]:
    result: list[LonInterval] = []
    for left_a, right_a in first:
        for left_b, right_b in second:
            left = max(left_a, left_b)
            right = min(right_a, right_b)
            if right >= left:
                result.append((left, right))
    return result


def _expand_lon_intervals(
    intervals: Iterable[LonInterval], padding: float
) -> list[LonInterval]:
    expanded: list[LonInterval] = []
    for left, right in intervals:
        center = (left + right) / 2.0
        half = (right - left) / 2.0 + float(padding)
        expanded.extend(_split_lon_interval(center, half))
    return expanded


def hour_outer_bounds(
    dark_hour: Mapping[str, Any],
    *,
    padding_degrees: float,
) -> tuple[float, float, list[LonInterval]]:
    """Outer bounds containing the full two-sided reachable hour region."""

    if not bool(dark_hour["bounded"]):
        raise ValueError("unbounded dark hour has no two-sided query geometry")
    hour = utc_timestamp(dark_hour["hour_utc"])
    source_start = utc_timestamp(dark_hour["source_gap_start_utc"])
    source_end = utc_timestamp(dark_hour["source_gap_end_utc"])
    if hour < source_start or hour + ONE_HOUR > source_end:
        raise ValueError("target UTC hour is not fully contained in its source gap")
    speed_km_per_hour = float(dark_hour["effective_speed_kn"]) * KM_PER_NAUTICAL_MILE
    radius_a = speed_km_per_hour * (hour + ONE_HOUR - source_start).total_seconds() / 3600.0
    radius_b = speed_km_per_hour * (source_end - hour).total_seconds() / 3600.0
    south_a, north_a, lon_a = spherical_cap_bbox(
        float(dark_hour["lat_a"]), float(dark_hour["lon_a"]), radius_a
    )
    south_b, north_b, lon_b = spherical_cap_bbox(
        float(dark_hour["lat_b"]), float(dark_hour["lon_b"]), radius_b
    )
    south = max(south_a, south_b) - float(padding_degrees)
    north = min(north_a, north_b) + float(padding_degrees)
    if north < south:
        raise ValueError("two cap bounding boxes do not intersect")
    intervals = _intersect_lon_intervals(lon_a, lon_b)
    if not intervals:
        raise ValueError("two cap longitude bounds do not intersect")
    return (
        max(-90.0, south),
        min(90.0, north),
        _expand_lon_intervals(intervals, float(padding_degrees)),
    )


def tiles_for_bounds(
    south: float,
    north: float,
    lon_intervals: Iterable[LonInterval],
    *,
    tile_degrees: float,
) -> set[Tile]:
    """Snap bounds outward to global lon/lat tiles."""

    step = float(tile_degrees)
    if step <= 0 or not math.isclose(180.0 / step, round(180.0 / step)):
        raise ValueError("tile size must divide 180 degrees")
    n_lat = int(round(180.0 / step))
    n_lon = int(round(360.0 / step))
    south = max(-90.0, float(south))
    north = min(90.0, float(north))
    lat_first = max(0, min(n_lat - 1, int(math.floor((south + 90.0) / step))))
    lat_last = max(
        lat_first,
        min(n_lat - 1, int(math.ceil((north + 90.0) / step) - 1)),
    )
    tiles: set[Tile] = set()
    for left, right in lon_intervals:
        left = max(-180.0, float(left))
        right = min(180.0, float(right))
        lon_first = max(0, min(n_lon - 1, int(math.floor((left + 180.0) / step))))
        lon_last = max(
            lon_first,
            min(n_lon - 1, int(math.ceil((right + 180.0) / step) - 1)),
        )
        for lat_index in range(lat_first, lat_last + 1):
            for lon_index in range(lon_first, lon_last + 1):
                tiles.add((lat_index, lon_index))
    return tiles


def tiles_for_hour(
    dark_hour: Mapping[str, Any],
    *,
    tile_degrees: float,
    padding_degrees: float,
) -> set[Tile]:
    south, north, intervals = hour_outer_bounds(
        dark_hour, padding_degrees=padding_degrees
    )
    return tiles_for_bounds(
        south, north, intervals, tile_degrees=tile_degrees
    )


def geometry_for_tiles(tiles: Iterable[Tile], *, tile_degrees: float):
    step = float(tile_degrees)
    polygons = []
    for lat_index, lon_index in sorted(set(tiles)):
        south = -90.0 + lat_index * step
        west = -180.0 + lon_index * step
        polygons.append(box(west, south, west + step, south + step))
    if not polygons:
        raise ValueError("query plan has no tiles")
    geometry = unary_union(polygons)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("query tile union is empty or invalid")
    return geometry


def plan_day_tiles(
    dark_hours: pd.DataFrame,
    *,
    tile_degrees: float,
    padding_degrees: float,
) -> tuple[set[Tile], dict[str, int | float]]:
    """Plan one serial daily request from bounded target hours only."""

    if dark_hours.empty:
        raise ValueError("representative day has no target dark hours")
    bounded = dark_hours.loc[dark_hours["bounded"].astype(bool)]
    if bounded.empty:
        raise ValueError("representative day has no bounded target dark hours")
    tiles: set[Tile] = set()
    hour_tile_memberships = 0
    maximum_tiles_per_hour = 0
    for row in bounded.to_dict(orient="records"):
        current = tiles_for_hour(
            row,
            tile_degrees=tile_degrees,
            padding_degrees=padding_degrees,
        )
        tiles.update(current)
        hour_tile_memberships += len(current)
        maximum_tiles_per_hour = max(maximum_tiles_per_hour, len(current))
    return tiles, {
        "target_full_dark_hours": int(len(dark_hours)),
        "bounded_target_hours": int(len(bounded)),
        "unbounded_target_hours": int(len(dark_hours) - len(bounded)),
        "distinct_target_mmsi": int(dark_hours["mmsi"].nunique()),
        "query_tiles": int(len(tiles)),
        "query_area_square_degrees": float(len(tiles) * float(tile_degrees) ** 2),
        "hour_tile_memberships_before_union": int(hour_tile_memberships),
        "maximum_tiles_for_one_hour": int(maximum_tiles_per_hour),
    }
