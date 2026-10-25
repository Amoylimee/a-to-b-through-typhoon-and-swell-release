"""Independent geometry helpers for the E032 sea-path freeze audit.

This module deliberately does not import :mod:`sea_path.routing`.  It derives
the documented 1-arc-minute index convention directly and checks emitted route
CSVs by densifying every segment at one quarter of a mask cell.  Keeping this
logic separate prevents the router from certifying itself with the same
geometry implementation that constructed the path.
"""

from __future__ import annotations

import csv
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Pixel-centre grid convention used by the frozen GSHHG mask."""

    n_lat: int = 10_800
    n_lon: int = 21_600
    resolution_deg: float = 1.0 / 60.0
    lat_min: float = -89.991667
    lon_min: float = -179.991667


@dataclass(frozen=True)
class Corridor:
    """Half-open 0.1-degree audit box: south <= lat < north."""

    corridor_id: str
    label: str
    south: float
    north: float
    west: float
    east: float


GRID = GridSpec()

CORRIDORS = (
    Corridor("suez", "Suez corridor (28-33N, 32-35E)", 28.0, 33.0, 32.0, 35.0),
    Corridor("panama", "Panama (8-10N, 81-79W)", 8.0, 10.0, -81.0, -79.0),
    Corridor("malacca", "Malacca (0-6N, 98-105E)", 0.0, 6.0, 98.0, 105.0),
    Corridor("red_sea", "Red Sea (12-28N, 33-44E)", 12.0, 28.0, 33.0, 44.0),
    Corridor(
        "bab_el_mandeb",
        "Bab el-Mandeb (11-14N, 42-45E)",
        11.0,
        14.0,
        42.0,
        45.0,
    ),
    Corridor("cape", "Cape route (40-30S, 15-30E)", -40.0, -30.0, 15.0, 30.0),
    Corridor(
        "open_indian",
        "Open Indian Ocean (20-5S, 60-80E)",
        -20.0,
        -5.0,
        60.0,
        80.0,
    ),
    Corridor(
        "open_north_pacific",
        "Open North Pacific (30-45N, 160E-160W)",
        30.0,
        45.0,
        160.0,
        200.0,
    ),
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap_longitude(longitude: float) -> float:
    """Normalize longitude to [-180, 180)."""
    return (float(longitude) + 180.0) % 360.0 - 180.0


def shortest_longitude_delta(start: float, end: float) -> float:
    """Return the shortest signed longitude increment."""
    return (float(end) - float(start) + 180.0) % 360.0 - 180.0


def grid_index(
    latitude: float,
    longitude: float,
    *,
    grid: GridSpec = GRID,
) -> tuple[int, int]:
    """Map a coordinate to the documented periodic pixel-centre grid."""
    row = int(round((float(latitude) - grid.lat_min) / grid.resolution_deg))
    column = int(round((float(longitude) - grid.lon_min) / grid.resolution_deg))
    row = max(0, min(grid.n_lat - 1, row))
    return row, column % grid.n_lon


def canonical_edge(
    left: tuple[int, int],
    right: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return an undirected edge in stable tuple order."""
    return (left, right) if left <= right else (right, left)


def measure_corridor(
    ocean_mask: np.ndarray,
    corridor: Corridor,
    *,
    grid: GridSpec = GRID,
) -> dict[str, object]:
    """Measure whether ocean-containing 0.1-degree cells have ocean centres.

    The loop intentionally preserves the original Supreme audit's half-open
    floating-point construction.  This makes the denominator and centre
    lookup byte-for-byte reviewable instead of replacing them with a subtly
    different raster aggregation.
    """
    denominator = 0
    centre_navigable = 0
    latitude = corridor.south
    while latitude < corridor.north - 1e-9:
        longitude = corridor.west
        while longitude < corridor.east - 1e-9:
            centre_lat = latitude + 0.05
            centre_lon = wrap_longitude(longitude + 0.05)
            row_0, col_0 = grid_index(
                latitude,
                wrap_longitude(longitude),
                grid=grid,
            )
            row_1, col_1 = grid_index(
                latitude + 0.1 - 1e-9,
                wrap_longitude(longitude + 0.1 - 1e-9),
                grid=grid,
            )
            if col_1 < col_0:
                col_1 = col_0
            patch = ocean_mask[row_0 : row_1 + 1, col_0 : col_1 + 1]
            if bool(patch.any()):
                denominator += 1
                if bool(ocean_mask[grid_index(centre_lat, centre_lon, grid=grid)]):
                    centre_navigable += 1
            longitude += 0.1
        latitude += 0.1

    centre_share = centre_navigable / denominator if denominator else float("nan")
    return {
        **asdict(corridor),
        "denominator_ocean_containing_0p1deg_cells": denominator,
        "centre_navigable_cells": centre_navigable,
        "centre_navigable_share": centre_share,
        "centre_not_navigable_cells": denominator - centre_navigable,
        "centre_not_navigable_share": 1.0 - centre_share,
    }


def measure_corridors(
    ocean_mask: np.ndarray,
    corridors: Iterable[Corridor] = CORRIDORS,
    *,
    workers: int = 8,
    grid: GridSpec = GRID,
) -> list[dict[str, object]]:
    """Measure independent corridor boxes concurrently on one read-only mask."""
    ordered = tuple(corridors)
    if not ordered:
        return []
    worker_count = max(1, min(int(workers), len(ordered)))
    by_id: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(measure_corridor, ocean_mask, item, grid=grid): item
            for item in ordered
        }
        for future in as_completed(futures):
            item = futures[future]
            by_id[item.corridor_id] = future.result()
    return [by_id[item.corridor_id] for item in ordered]


def _segment_audit(
    ocean_mask: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    grid: GridSpec = GRID,
) -> tuple[bool, float]:
    """Densify one emitted segment independently and report land contact."""
    lat_start, lon_start = start
    lat_end, lon_end = end
    delta_lat = lat_end - lat_start
    delta_lon = shortest_longitude_delta(lon_start, lon_end)
    cells = max(
        abs(delta_lat) / grid.resolution_deg,
        abs(delta_lon) / grid.resolution_deg,
    )
    samples = max(1, int(math.ceil(cells * 4.0)))
    touches_land = False
    for ordinal in range(samples + 1):
        fraction = ordinal / samples
        latitude = lat_start + delta_lat * fraction
        longitude = wrap_longitude(lon_start + delta_lon * fraction)
        if not bool(ocean_mask[grid_index(latitude, longitude, grid=grid)]):
            touches_land = True
            break
    midpoint_lat = 0.5 * (lat_start + lat_end)
    approximate_km = math.hypot(
        delta_lat,
        delta_lon * math.cos(math.radians(midpoint_lat)),
    ) * 111.195
    return touches_land, approximate_km


def audit_route_csv(
    ocean_mask: np.ndarray,
    route_csv: Path,
    declared_edges: frozenset[
        tuple[tuple[int, int], tuple[int, int]]
    ],
    *,
    grid: GridSpec = GRID,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Audit one route CSV without using the router's geometry helpers."""
    with Path(route_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Route CSV is empty: {route_csv}")
    fields = set(rows[0])
    latitude_field = "latitude" if "latitude" in fields else "lat"
    longitude_field = "longitude" if "longitude" in fields else "lon"
    if latitude_field not in fields or longitude_field not in fields:
        raise ValueError(f"Route CSV lacks latitude/longitude: {route_csv}")

    points = [
        (float(row[latitude_field]), float(row[longitude_field]))
        for row in rows
    ]
    land_waypoints = sum(
        not bool(ocean_mask[grid_index(latitude, longitude, grid=grid)])
        for latitude, longitude in points
    )
    flagged: list[dict[str, object]] = []
    max_gap_km = 0.0
    for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        touches_land, gap_km = _segment_audit(
            ocean_mask,
            start,
            end,
            grid=grid,
        )
        max_gap_km = max(max_gap_km, gap_km)
        if not touches_land:
            continue
        start_cell = grid_index(*start, grid=grid)
        end_cell = grid_index(*end, grid=grid)
        edge = canonical_edge(start_cell, end_cell)
        flagged.append(
            {
                "route_csv": Path(route_csv).name,
                "segment_index_zero_based": segment_index,
                "start_latitude": start[0],
                "start_longitude": start[1],
                "end_latitude": end[0],
                "end_longitude": end[1],
                "start_row": start_cell[0],
                "start_column": start_cell[1],
                "end_row": end_cell[0],
                "end_column": end_cell[1],
                "declared_channel_edge": edge in declared_edges,
            }
        )

    summary = {
        "route_csv": Path(route_csv).name,
        "waypoint_count": len(points),
        "land_waypoint_count": int(land_waypoints),
        "land_touching_segment_count": len(flagged),
        "declared_land_touching_segment_count": sum(
            bool(row["declared_channel_edge"]) for row in flagged
        ),
        "undeclared_land_touching_segment_count": sum(
            not bool(row["declared_channel_edge"]) for row in flagged
        ),
        "maximum_waypoint_gap_km_approx": max_gap_km,
    }
    return summary, flagged


def audit_route_directory(
    ocean_mask: np.ndarray,
    routes_dir: Path,
    declared_edges: frozenset[
        tuple[tuple[int, int], tuple[int, int]]
    ],
    *,
    workers: int = 8,
    grid: GridSpec = GRID,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Audit independent route CSVs concurrently with explicit failures."""
    route_files = sorted(Path(routes_dir).glob("*.csv"))
    if not route_files:
        raise FileNotFoundError(f"No route CSVs found in {routes_dir}")
    worker_count = max(1, min(int(workers), len(route_files)))
    results: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                audit_route_csv,
                ocean_mask,
                route_file,
                declared_edges,
                grid=grid,
            ): route_file
            for route_file in route_files
        }
        failures: list[str] = []
        for future in as_completed(futures):
            route_file = futures[future]
            try:
                results[route_file.name] = future.result()
            except Exception as exc:  # pragma: no cover - aggregated error path
                failures.append(f"{route_file.name}: {exc!r}")
        if failures:
            raise RuntimeError("Route audit failures:\n" + "\n".join(failures))

    summaries = [results[path.name][0] for path in route_files]
    segments = [
        segment
        for path in route_files
        for segment in results[path.name][1]
    ]
    return summaries, segments
