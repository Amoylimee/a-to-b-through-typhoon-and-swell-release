"""Great-circle-first sea-path routing on the 1-arc-minute ocean grid.

Given two ocean positions, first tests the shortest spherical great-circle
arc against every raster cell it touches.  An all-ocean arc is returned
directly.  Only blocked arc sections invoke A*; their raster detours are then
shortcut where a replacement great-circle segment is entirely ocean.

Ordinary diagonal grid movement cannot squeeze between land cells that only
touch at a corner.  Named Suez, Panama, and Bosporus exceptions preserve only
the hand-verified sub-grid channel edges.  Longitude is periodic throughout,
including the date line.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import NamedTuple

import numpy as np
from scipy.ndimage import label as ndimage_label

from .ocean_mask import (
    LAT_MIN,
    LON_MIN,
    N_LAT,
    N_LON,
    RESOLUTION_DEG,
    maritime_channel_edges,
    verify_routing_inputs,
)

EARTH_RADIUS_KM = 6371.0

_DEG2RAD = math.pi / 180.0

DEFAULT_GREAT_CIRCLE_STEP_KM = (
    EARTH_RADIUS_KM * RESOLUTION_DEG * _DEG2RAD * 0.5
)


class SeaPath(NamedTuple):
    waypoints: np.ndarray      # shape (N, 2), columns (lat_deg, lon_deg)
    cumulative_km: np.ndarray  # shape (N,), cumulative distance from start
    total_km: float


# Why a route did not resolve.  These are reported, never collapsed into a
# bare ``None``: an anchor that is not navigable and two genuinely unreachable
# basins are different measurements, and a coverage denominator that merges
# them cannot be interpreted.
ROUTE_RESOLVED = "resolved"
ROUTE_ANCHOR_NOT_NAVIGABLE = "anchor_not_navigable"
ROUTE_BASINS_DISCONNECTED = "basins_disconnected"
ROUTE_NO_PATH = "no_path"
ROUTE_SEARCH_TIMEOUT = "search_timeout"
ROUTE_SPEED_INFEASIBLE = "speed_infeasible"

ROUTE_STATUSES = (
    ROUTE_RESOLVED,
    ROUTE_ANCHOR_NOT_NAVIGABLE,
    ROUTE_BASINS_DISCONNECTED,
    ROUTE_NO_PATH,
    ROUTE_SEARCH_TIMEOUT,
    ROUTE_SPEED_INFEASIBLE,
)

#: Per-route wall-clock budget, in seconds.  The margin expansion in
#: ``_find_detour_grid_path`` ends in an unbounded whole-grid search, and the
#: 2026-09-01 coverage bound measured what that costs in production: a handful of
#: pathological gaps held one worker for about ten minutes while the other
#: 220,000 averaged 0.089 s.  Exceeding the budget is an explicit unresolved
#: outcome, never a partial or approximate route.
DEFAULT_ROUTE_TIMEOUT_S = 5.0

#: Fleet speed ceiling in knots.  The accepted cleaned AIS carries
#: ``max_speed_knots`` per vessel, but every vessel in the 2024 fleet holds
#: exactly 25.0 from ``imo_type_default``, so the field is a constant wearing a
#: per-vessel name.  Fixing it here states that plainly.
DEFAULT_MAX_SPEED_KNOTS = 25.0
KM_PER_NAUTICAL_MILE = 1.852

_HEAP_POPS_PER_DEADLINE_CHECK = 1024


class _SearchTimeout(Exception):
    """Internal signal that a route exceeded its wall-clock budget."""

#: Bounded radius, in grid cells, for the diagnostic nearest-ocean search
#: reported when an anchor is not navigable.  About 111 km at the equator.
NEAREST_OCEAN_SEARCH_CELLS = 60


class RouteResolution(NamedTuple):
    """One routing attempt: the path when it resolved, the reason when not."""

    path: SeaPath | None
    status: str
    detail: dict


def wrap_longitude(lon: float) -> float:
    """Wrap a longitude to the half-open interval [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def shortest_longitude_delta(lon_start: float, lon_end: float) -> float:
    """Return the signed shortest angular step from start to end."""
    return wrap_longitude(lon_end - lon_start)


def _validate_position(lat: float, lon: float) -> tuple[float, float]:
    """Validate a WGS84 position and normalize +180 degrees to -180."""
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError(f"Latitude/longitude must be finite: ({lat}, {lon})")
    if lat < -90.0 or lat > 90.0:
        raise ValueError(f"Latitude outside [-90, 90]: {lat}")
    if lon < -180.0 or lon > 180.0:
        raise ValueError(f"Longitude outside [-180, 180]: {lon}")
    return float(lat), wrap_longitude(float(lon))


def _unit_vector(lat: float, lon: float) -> np.ndarray:
    lat_rad = lat * _DEG2RAD
    lon_rad = lon * _DEG2RAD
    cos_lat = math.cos(lat_rad)
    return np.array(
        [cos_lat * math.cos(lon_rad), cos_lat * math.sin(lon_rad), math.sin(lat_rad)],
        dtype=float,
    )


def great_circle_waypoints(
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    *,
    max_step_km: float = DEFAULT_GREAT_CIRCLE_STEP_KM,
) -> np.ndarray:
    """Densify the shortest spherical arc between two validated positions."""
    lat_a, lon_a = _validate_position(lat_a, lon_a)
    lat_b, lon_b = _validate_position(lat_b, lon_b)
    if not math.isfinite(max_step_km) or max_step_km <= 0.0:
        raise ValueError(f"max_step_km must be positive and finite: {max_step_km}")

    start = _unit_vector(lat_a, lon_a)
    end = _unit_vector(lat_b, lon_b)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    angle = math.acos(dot)
    distance_km = EARTH_RADIUS_KM * angle
    n_segments = max(1, int(math.ceil(distance_km / max_step_km)))
    fractions = np.linspace(0.0, 1.0, n_segments + 1)

    if angle < 1e-14:
        vectors = np.repeat(start[None, :], n_segments + 1, axis=0)
    elif math.pi - angle < 1e-12:
        # Antipodal points have no unique shortest arc.  Choose a stable,
        # deterministic perpendicular so the result is reproducible.
        axes = np.eye(3)
        reference = axes[int(np.argmin(np.abs(axes @ start)))]
        tangent = np.cross(reference, start)
        tangent /= np.linalg.norm(tangent)
        theta = fractions * math.pi
        vectors = (
            np.cos(theta)[:, None] * start[None, :]
            + np.sin(theta)[:, None] * tangent[None, :]
        )
    else:
        sin_angle = math.sin(angle)
        left_weights = np.sin((1.0 - fractions) * angle) / sin_angle
        right_weights = np.sin(fractions * angle) / sin_angle
        vectors = left_weights[:, None] * start + right_weights[:, None] * end
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]

    waypoints = np.empty((n_segments + 1, 2), dtype=float)
    waypoints[:, 0] = np.degrees(
        np.arctan2(vectors[:, 2], np.hypot(vectors[:, 0], vectors[:, 1]))
    )
    waypoints[:, 1] = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    waypoints[0] = (lat_a, lon_a)
    waypoints[-1] = (lat_b, lon_b)
    return waypoints


def _wrapped_index_delta(start: int, end: int, width: int) -> int:
    """Return the shortest signed column-index delta on a periodic grid."""
    raw = (end - start) % width
    if raw > width // 2:
        raw -= width
    return int(raw)


def _supercover_grid_line(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    n_lat: int,
    n_lon: int,
) -> list[tuple[int, int]]:
    """Return every raster cell touched by a centre-to-centre grid segment.

    Exact corner crossings include both side cells, making the test
    conservative: a line cannot pass through a zero-width diagonal gap.
    Longitude follows the shortest periodic direction.
    """
    row, col = start
    end_row, end_col = end
    if not (0 <= row < n_lat and 0 <= end_row < n_lat):
        raise ValueError(f"Row outside grid: {start} -> {end}")

    delta_row = end_row - row
    delta_col = _wrapped_index_delta(col, end_col, n_lon)
    target_col = col + delta_col
    step_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
    step_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)
    n_row = abs(delta_row)
    n_col = abs(delta_col)
    moved_row = 0
    moved_col = 0

    cells: list[tuple[int, int]] = []

    def append_cell(cell_row: int, cell_col: int) -> None:
        cell = (cell_row, cell_col % n_lon)
        if not cells or cells[-1] != cell:
            cells.append(cell)

    append_cell(row, col)
    while moved_row < n_row or moved_col < n_col:
        decision = (1 + 2 * moved_col) * n_row - (1 + 2 * moved_row) * n_col
        if decision == 0:
            append_cell(row, col + step_col)
            append_cell(row + step_row, col)
            col += step_col
            row += step_row
            moved_col += 1
            moved_row += 1
            append_cell(row, col)
        elif decision < 0:
            col += step_col
            moved_col += 1
            append_cell(row, col)
        else:
            row += step_row
            moved_row += 1
            append_cell(row, col)

    assert row == end_row and col == target_col
    return cells


def split_path_at_antimeridian(path: SeaPath) -> list[np.ndarray]:
    """Split normalized waypoints into continuous map-plotting segments.

    A path crossing the antimeridian contains adjacent longitudes near +180
    and -180 degrees.  Plotting those values as one ordinary Cartesian line
    draws a false segment across the map.  This helper inserts the exact seam
    crossing and returns separate segments on either side of the seam.
    """
    waypoints = np.asarray(path.waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError(f"Expected waypoints with shape (N, 2), got {waypoints.shape}")
    if len(waypoints) == 0:
        return []

    first = np.array([waypoints[0, 0], wrap_longitude(waypoints[0, 1])])
    current: list[np.ndarray] = [first]
    segments: list[np.ndarray] = []

    for previous, following in zip(waypoints[:-1], waypoints[1:]):
        lat_a = float(previous[0])
        lat_b = float(following[0])
        lon_a = wrap_longitude(float(previous[1]))
        lon_b = wrap_longitude(float(following[1]))
        delta = shortest_longitude_delta(lon_a, lon_b)
        lon_b_unwrapped = lon_a + delta

        if lon_b_unwrapped > 180.0:
            fraction = (180.0 - lon_a) / delta
            crossing_lat = lat_a + fraction * (lat_b - lat_a)
            current.append(np.array([crossing_lat, 180.0]))
            segments.append(np.asarray(current))
            current = [
                np.array([crossing_lat, -180.0]),
                np.array([lat_b, lon_b]),
            ]
        elif lon_b_unwrapped < -180.0:
            fraction = (-180.0 - lon_a) / delta
            crossing_lat = lat_a + fraction * (lat_b - lat_a)
            current.append(np.array([crossing_lat, -180.0]))
            segments.append(np.asarray(current))
            current = [
                np.array([crossing_lat, 180.0]),
                np.array([lat_b, lon_b]),
            ]
        else:
            current.append(np.array([lat_b, lon_b]))

    segments.append(np.asarray(current))
    return segments


def _label_ocean_components(
    ocean_mask: np.ndarray,
    explicit_edges: frozenset[
        tuple[tuple[int, int], tuple[int, int]]
    ] = frozenset(),
) -> tuple[np.ndarray, np.ndarray, int]:
    """Label conservative ocean components with periodic longitude.

    ``scipy.ndimage.label`` does not know that the first and last longitude
    columns touch.  Start with 4-connectivity, union same-row seam cells and
    then union named sub-grid canal edges.  A permitted ordinary diagonal has
    two ocean side cells and is therefore already 4-connected; corner-only
    diagonals remain disconnected.
    """
    components, raw_count = ndimage_label(
        ocean_mask,
        structure=np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8,
        ),
    )
    parents = np.arange(raw_count + 1, dtype=np.int32)

    def find_root(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = int(parents[value])
        return value

    def union(left: int, right: int) -> None:
        root_left = find_root(left)
        root_right = find_root(right)
        if root_left != root_right:
            parents[root_right] = root_left

    pairs = np.unique(
        np.column_stack((components[:, 0], components[:, -1])), axis=0,
    )
    for left, right in pairs:
        left_int = int(left)
        right_int = int(right)
        if left_int and right_int and left_int != right_int:
            union(left_int, right_int)

    for left_cell, right_cell in explicit_edges:
        left_label = int(components[left_cell])
        right_label = int(components[right_cell])
        if left_label and right_label and left_label != right_label:
            union(left_label, right_label)

    roots = np.arange(raw_count + 1, dtype=np.int32)
    for component in range(1, raw_count + 1):
        roots[component] = find_root(component)
    periodic_count = len(np.unique(roots[1:])) if raw_count else 0
    return components, roots, periodic_count


class SeaPathRouter:
    """Great-circle-first router on the boolean ocean mask."""

    __slots__ = (
        "mask",
        "n_lat",
        "n_lon",
        "components",
        "n_components",
        "_component_roots",
        "_channel_edges",
        "_lat_min",
        "_lon_min",
        "_resolution_deg",
        "_lat_cos",
    )

    def __init__(
        self, ocean_mask: np.ndarray, *, verify_inputs: bool = True,
    ) -> None:
        if ocean_mask.shape != (N_LAT, N_LON):
            raise ValueError(f"Expected shape ({N_LAT}, {N_LON}), got {ocean_mask.shape}")
        if ocean_mask.dtype != np.bool_:
            raise ValueError(f"Expected bool dtype, got {ocean_mask.dtype}")

        self.mask = ocean_mask
        self.n_lat = N_LAT
        self.n_lon = N_LON
        self._lat_min = LAT_MIN
        self._lon_min = LON_MIN
        self._resolution_deg = RESOLUTION_DEG
        self._channel_edges = maritime_channel_edges()
        if verify_inputs:
            verify_routing_inputs(ocean=ocean_mask)

        (
            self.components,
            self._component_roots,
            self.n_components,
        ) = _label_ocean_components(ocean_mask, self._channel_edges)

        lats_rad = np.arange(N_LAT) * RESOLUTION_DEG * _DEG2RAD + LAT_MIN * _DEG2RAD
        self._lat_cos = np.cos(lats_rad)

    def _position_to_index(self, lat: float, lon: float) -> tuple[int, int]:
        """Map a position to this router's periodic cell grid."""
        row = int(round((lat - self._lat_min) / self._resolution_deg))
        col = int(round((lon - self._lon_min) / self._resolution_deg))
        row = max(0, min(self.n_lat - 1, row))
        col %= self.n_lon
        return row, col

    def _index_to_position(self, row: int, col: int) -> tuple[float, float]:
        """Return the centre position for a cell in this router's grid."""
        lat = self._lat_min + row * self._resolution_deg
        lon = self._lon_min + (col % self.n_lon) * self._resolution_deg
        return float(lat), wrap_longitude(float(lon))

    # ------------------------------------------------------------------
    # Haversine
    # ------------------------------------------------------------------

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        la1 = lat1 * _DEG2RAD
        la2 = lat2 * _DEG2RAD
        dla = (lat2 - lat1) * _DEG2RAD
        dlo = shortest_longitude_delta(lon1, lon2) * _DEG2RAD
        a = math.sin(dla * 0.5) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo * 0.5) ** 2
        return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))

    def _edge_km(self, i1: int, j1: int, i2: int, j2: int) -> float:
        """Distance between two adjacent cell centres."""
        lat1 = self._lat_min + i1 * self._resolution_deg
        lon1 = self._lon_min + j1 * self._resolution_deg
        lat2 = self._lat_min + i2 * self._resolution_deg
        lon2 = self._lon_min + j2 * self._resolution_deg
        return self.haversine_km(lat1, lon1, lat2, lon2)

    def _heuristic_km(self, i: int, j: int, gi: int, gj: int) -> float:
        """Admissible A* heuristic: haversine to goal."""
        lat1 = self._lat_min + i * self._resolution_deg
        lon1 = self._lon_min + j * self._resolution_deg
        lat2 = self._lat_min + gi * self._resolution_deg
        lon2 = self._lon_min + gj * self._resolution_deg
        return self.haversine_km(lat1, lon1, lat2, lon2)

    # ------------------------------------------------------------------
    # Nearest ocean cell
    # ------------------------------------------------------------------

    def nearest_ocean(
        self, lat: float, lon: float, *, max_cells: int = NEAREST_OCEAN_SEARCH_CELLS,
    ) -> tuple[float | None, tuple[int, int] | None]:
        """Return (distance_km, cell) of the nearest navigable cell.

        Diagnostic only -- it never moves an anchor.  The distance is what makes
        an ``anchor_not_navigable`` outcome interpretable: a position 2 km
        inland is a resolution artefact, one 40 km inland is a bad fix.  Cost is
        paid only when a route has already failed.
        """
        ci, cj = self._position_to_index(lat, lon)
        if self.mask[ci, cj]:
            return 0.0, (ci, cj)

        low = max(0, ci - max_cells)
        high = min(self.n_lat - 1, ci + max_cells)
        rows = np.arange(low, high + 1)
        cols = np.arange(cj - max_cells, cj + max_cells + 1) % self.n_lon
        patch = self.mask[low : high + 1][:, cols]
        if not patch.any():
            return None, None

        row_hit, col_hit = np.nonzero(patch)
        cand_rows = rows[row_hit]
        cand_cols = cols[col_hit]
        cand_lat = self._lat_min + cand_rows * self._resolution_deg
        cand_lon = self._lon_min + cand_cols * self._resolution_deg

        lat_rad = lat * _DEG2RAD
        cand_lat_rad = cand_lat * _DEG2RAD
        d_lat = cand_lat_rad - lat_rad
        d_lon = (((cand_lon - lon + 180.0) % 360.0) - 180.0) * _DEG2RAD
        hav = (
            np.sin(d_lat * 0.5) ** 2
            + math.cos(lat_rad) * np.cos(cand_lat_rad) * np.sin(d_lon * 0.5) ** 2
        )
        distances = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(hav)))
        best = int(np.argmin(distances))
        return float(distances[best]), (int(cand_rows[best]), int(cand_cols[best]))

    def snap_to_ocean(
        self, lat: float, lon: float, max_cells: int = 5,
    ) -> tuple[int, int] | None:
        """Find nearest ocean cell within a search radius.

        Returns (i_lat, i_lon) or None if no ocean within radius.
        """
        ci, cj = self._position_to_index(lat, lon)
        if self.mask[ci, cj]:
            return ci, cj

        best = None
        best_dist = float("inf")
        for di in range(-max_cells, max_cells + 1):
            ni = ci + di
            if ni < 0 or ni >= self.n_lat:
                continue
            for dj in range(-max_cells, max_cells + 1):
                nj = (cj + dj) % self.n_lon
                if self.mask[ni, nj]:
                    d = self._edge_km(ci, cj, ni, nj)
                    if d < best_dist:
                        best_dist = d
                        best = (ni, nj)
        return best

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_edge(
        left: tuple[int, int], right: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (left, right) if left <= right else (right, left)

    def _edge_is_explicit_channel(
        self, left: tuple[int, int], right: tuple[int, int],
    ) -> bool:
        return self._canonical_edge(left, right) in self._channel_edges

    def _neighbors(self, i: int, j: int):
        """Yield navigable neighbours without ordinary corner cutting."""
        mask = self.mask
        n_lat = self.n_lat
        n_lon = self.n_lon
        for di in (-1, 0, 1):
            ni = i + di
            if ni < 0 or ni >= n_lat:
                continue
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nj = (j + dj) % n_lon
                if not mask[ni, nj]:
                    continue

                if di != 0 and dj != 0:
                    left = (i, j)
                    right = (ni, nj)
                    ordinary_diagonal = bool(mask[i, nj] and mask[ni, j])
                    if not ordinary_diagonal and not self._edge_is_explicit_channel(
                        left, right,
                    ):
                        continue
                yield ni, nj

    def _find_grid_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        margin_cells: int | None = None,
        deadline: float | None = None,
    ) -> list[tuple[int, int]] | None:
        """Find a conservative raster path, optionally inside a local window."""
        si, sj = start
        gi, gj = goal
        if not self.mask[si, sj] or not self.mask[gi, gj]:
            return None

        start_component = self._component_roots[self.components[si, sj]]
        goal_component = self._component_roots[self.components[gi, gj]]
        if start_component != goal_component:
            return None
        if start == goal:
            return [start]

        if margin_cells is not None and margin_cells < 0:
            raise ValueError(f"margin_cells must be non-negative: {margin_cells}")
        goal_relative_col = _wrapped_index_delta(sj, gj, self.n_lon)
        min_row = max(0, min(si, gi) - (margin_cells or 0))
        max_row = min(self.n_lat - 1, max(si, gi) + (margin_cells or 0))
        min_relative_col = min(0, goal_relative_col) - (margin_cells or 0)
        max_relative_col = max(0, goal_relative_col) + (margin_cells or 0)

        def inside_window(row: int, col: int) -> bool:
            if margin_cells is None:
                return True
            relative_col = _wrapped_index_delta(sj, col, self.n_lon)
            return (
                min_row <= row <= max_row
                and min_relative_col <= relative_col <= max_relative_col
            )

        start_h = self._heuristic_km(si, sj, gi, gj)
        open_heap: list[tuple[float, float, int, int]] = [
            (start_h, 0.0, si, sj),
        ]
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        pops = 0
        while open_heap:
            # Checked on a stride so the clock read cannot dominate the search.
            pops += 1
            if (
                deadline is not None
                and pops % _HEAP_POPS_PER_DEADLINE_CHECK == 0
                and time.monotonic() > deadline
            ):
                raise _SearchTimeout
            _, queued_g, ci, cj = heapq.heappop(open_heap)
            current = (ci, cj)
            current_g = g_score.get(current, float("inf"))
            if queued_g > current_g + 1e-12:
                continue
            if current == goal:
                break

            for ni, nj in self._neighbors(ci, cj):
                if not inside_window(ni, nj):
                    continue
                neighbor = (ni, nj)
                tentative_g = current_g + self._edge_km(ci, cj, ni, nj)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    estimate = tentative_g + self._heuristic_km(ni, nj, gi, gj)
                    heapq.heappush(
                        open_heap, (estimate, tentative_g, ni, nj),
                    )
        else:
            return None

        indices = [goal]
        node = goal
        while node != start:
            node = came_from[node]
            indices.append(node)
        indices.reverse()
        return indices

    def _find_detour_grid_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        deadline: float | None = None,
    ) -> list[tuple[int, int]] | None:
        """Search locally first, expanding only when the obstacle requires it."""
        direct_span = max(
            abs(goal[0] - start[0]),
            abs(_wrapped_index_delta(start[1], goal[1], self.n_lon)),
        )
        initial_margin = max(16, min(128, direct_span // 4))
        margins: list[int] = []
        margin = initial_margin
        while margin <= 1024:
            margins.append(margin)
            margin *= 2

        for local_margin in margins:
            path = self._find_grid_path(
                start, goal, margin_cells=local_margin, deadline=deadline,
            )
            if path is not None:
                return path

        # The local expansions are an optimization, not a scientific cap.
        # Retain a complete final search for unusual long detours.
        return self._find_grid_path(start, goal, deadline=deadline)

    def _segment_is_ocean(
        self,
        start: np.ndarray | tuple[float, float],
        end: np.ndarray | tuple[float, float],
        *,
        required_component: int | None = None,
    ) -> bool:
        """Check every supercover cell touched by a short spherical segment."""
        points = great_circle_waypoints(
            float(start[0]), float(start[1]), float(end[0]), float(end[1]),
        )
        previous = self._position_to_index(
            float(points[0, 0]), float(points[0, 1]),
        )
        for point in points[1:]:
            following = self._position_to_index(
                float(point[0]), float(point[1]),
            )
            for cell in _supercover_grid_line(
                previous, following, n_lat=self.n_lat, n_lon=self.n_lon,
            ):
                if not self.mask[cell]:
                    return False
                if (
                    required_component is not None
                    and int(self._component_roots[self.components[cell]])
                    != required_component
                ):
                    return False
            previous = following
        return True

    def great_circle_is_ocean(
        self, lat_a: float, lon_a: float, lat_b: float, lon_b: float,
    ) -> bool:
        """Return whether the direct great-circle arc touches only ocean cells."""
        start = _validate_position(lat_a, lon_a)
        goal = _validate_position(lat_b, lon_b)
        return self._segment_is_ocean(start, goal)

    def _indices_are_visible(
        self, start: tuple[int, int], goal: tuple[int, int],
    ) -> bool:
        """Return whether a shortcut between two path cells is navigable."""
        if start == goal:
            return True
        if self._edge_is_explicit_channel(start, goal):
            return True
        return self._segment_is_ocean(
            self._index_to_position(*start), self._index_to_position(*goal),
        )

    def _smooth_grid_path(
        self, path: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Remove A* turns only when the replacement spherical arc is ocean."""
        if len(path) <= 2:
            return path.copy()

        smoothed = [path[0]]
        anchor = 0
        last = len(path) - 1
        while anchor < last:
            furthest = anchor + 1
            probe = furthest + 1
            while probe <= last and self._indices_are_visible(path[anchor], path[probe]):
                furthest = probe
                probe += 1
            smoothed.append(path[furthest])
            anchor = furthest
        return smoothed

    def _grid_path_waypoints(
        self, path: list[tuple[int, int]],
    ) -> np.ndarray:
        """Densify smoothed grid shortcuts while preserving explicit canals."""
        if len(path) == 1:
            return np.asarray([self._index_to_position(*path[0])], dtype=float)

        chunks: list[np.ndarray] = []
        for left, right in zip(path[:-1], path[1:]):
            left_position = self._index_to_position(*left)
            right_position = self._index_to_position(*right)
            if self._segment_is_ocean(left_position, right_position):
                chunk = great_circle_waypoints(*left_position, *right_position)
            elif self._edge_is_explicit_channel(left, right):
                chunk = np.asarray([left_position, right_position], dtype=float)
            else:
                raise RuntimeError(f"Non-navigable smoothed edge: {left} -> {right}")
            chunks.append(chunk if not chunks else chunk[1:])
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _blocked_runs(blocked: list[bool]) -> list[tuple[int, int]]:
        """Return inclusive runs of blocked great-circle segment indices."""
        runs: list[tuple[int, int]] = []
        index = 0
        while index < len(blocked):
            if not blocked[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(blocked) and blocked[index + 1]:
                index += 1
            runs.append((start, index))
            index += 1
        return runs

    @staticmethod
    def _combine_waypoint_chunks(chunks: list[np.ndarray]) -> np.ndarray:
        """Concatenate route chunks while removing identical join points."""
        combined: list[np.ndarray] = []
        for chunk in chunks:
            for point in np.asarray(chunk, dtype=float):
                if combined:
                    same_lat = abs(float(point[0]) - float(combined[-1][0])) < 1e-10
                    same_lon = abs(
                        shortest_longitude_delta(
                            float(combined[-1][1]), float(point[1]),
                        )
                    ) < 1e-10
                    if same_lat and same_lon:
                        continue
                combined.append(np.asarray(point, dtype=float))
        return np.asarray(combined, dtype=float)

    @classmethod
    def _sea_path_from_waypoints(cls, waypoints: np.ndarray) -> SeaPath:
        cumulative = np.zeros(len(waypoints), dtype=float)
        for index in range(1, len(waypoints)):
            cumulative[index] = cumulative[index - 1] + cls.haversine_km(
                float(waypoints[index - 1, 0]),
                float(waypoints[index - 1, 1]),
                float(waypoints[index, 0]),
                float(waypoints[index, 1]),
            )
        return SeaPath(
            waypoints=waypoints,
            cumulative_km=cumulative,
            total_km=float(cumulative[-1]),
        )

    @staticmethod
    def _speed_gated(
        path: SeaPath,
        detail: dict,
        duration_hours: float | None,
        max_speed_knots: float,
    ) -> RouteResolution:
        """Reject a route the vessel could not have sailed in the time available."""
        if duration_hours is None:
            return RouteResolution(path, ROUTE_RESOLVED, detail)
        if not math.isfinite(duration_hours) or duration_hours <= 0.0:
            raise ValueError(f"duration_hours must be positive: {duration_hours}")
        implied = path.total_km / duration_hours / KM_PER_NAUTICAL_MILE
        detail["implied_speed_knots"] = implied
        if implied > max_speed_knots:
            return RouteResolution(None, ROUTE_SPEED_INFEASIBLE, detail)
        return RouteResolution(path, ROUTE_RESOLVED, detail)

    def find_path(
        self,
        lat_a: float,
        lon_a: float,
        lat_b: float,
        lon_b: float,
        *,
        snap: bool = False,
        max_snap_cells: int = 5,
    ) -> SeaPath | None:
        """Find a shortest-spherical, land-avoiding path, or ``None``.

        Convenience wrapper over :meth:`resolve_path`.  It discards the reason a
        route did not resolve, so any caller that reports an unresolved rate
        must use :meth:`resolve_path` instead.
        """
        return self.resolve_path(
            lat_a, lon_a, lat_b, lon_b,
            snap=snap, max_snap_cells=max_snap_cells,
        ).path

    def resolve_path(
        self,
        lat_a: float,
        lon_a: float,
        lat_b: float,
        lon_b: float,
        *,
        snap: bool = False,
        max_snap_cells: int = 5,
        diagnose_anchors: bool = True,
        timeout_s: float | None = DEFAULT_ROUTE_TIMEOUT_S,
        duration_hours: float | None = None,
        max_speed_knots: float = DEFAULT_MAX_SPEED_KNOTS,
    ) -> RouteResolution:
        """Route between two positions and report why if it does not resolve.

        Observed anchors are not moved by default.  ``snap=True`` is an explicit
        opt-in for diagnostic/coastal-mask handling; a caller using it must
        preserve and report the requested and snapped coordinates.

        ``timeout_s`` bounds the whole route, including every margin expansion of
        the detour search; exceeding it yields ``search_timeout`` rather than a
        partial path.  Supplying ``duration_hours`` additionally rejects a route
        the vessel could not have sailed in the time available, as
        ``speed_infeasible`` -- the mask cannot represent every sub-grid strait,
        and where it forces a detour around one the route can run several times
        the true distance.

        The unresolved statuses are deliberately kept apart.  An anchor
        that is not navigable is a position-resolution problem and is often
        recoverable; two disconnected basins mean the pair is unroutable in
        principle; a failed detour means the search did not close.  Collapsing
        them into one ``None`` makes an unresolved rate uninterpretable.
        """
        requested_start = _validate_position(lat_a, lon_a)
        requested_goal = _validate_position(lat_b, lon_b)
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0.0):
            raise ValueError(f"timeout_s must be positive and finite: {timeout_s}")
        if not math.isfinite(max_speed_knots) or max_speed_knots <= 0.0:
            raise ValueError(f"max_speed_knots must be positive: {max_speed_knots}")
        started = time.monotonic()
        deadline = None if timeout_s is None else started + timeout_s
        detail: dict = {
            "requested_start": requested_start,
            "requested_goal": requested_goal,
            "snap_requested": bool(snap),
            "timeout_s": timeout_s,
            "max_speed_knots": max_speed_knots,
        }

        if snap:
            start = self.snap_to_ocean(*requested_start, max_cells=max_snap_cells)
            goal = self.snap_to_ocean(*requested_goal, max_cells=max_snap_cells)
        else:
            si, sj = self._position_to_index(*requested_start)
            gi, gj = self._position_to_index(*requested_goal)
            start = (si, sj) if self.mask[si, sj] else None
            goal = (gi, gj) if self.mask[gi, gj] else None

        if start is None or goal is None:
            blocking = []
            if start is None:
                blocking.append("start")
            if goal is None:
                blocking.append("goal")
            detail["blocking_anchors"] = blocking
            if diagnose_anchors:
                for name, position in (
                    ("start", requested_start), ("goal", requested_goal),
                ):
                    distance, cell = self.nearest_ocean(*position)
                    detail[f"{name}_nearest_ocean_km"] = distance
                    detail[f"{name}_nearest_ocean_cell"] = cell
            return RouteResolution(None, ROUTE_ANCHOR_NOT_NAVIGABLE, detail)

        start_position = (
            requested_start if self._position_to_index(*requested_start) == start
            else self._index_to_position(*start)
        )
        goal_position = (
            requested_goal if self._position_to_index(*requested_goal) == goal
            else self._index_to_position(*goal)
        )
        detail["resolved_start"] = start_position
        detail["resolved_goal"] = goal_position
        detail["start_offset_km"] = self.haversine_km(
            *requested_start, *start_position,
        )
        detail["goal_offset_km"] = self.haversine_km(
            *requested_goal, *goal_position,
        )

        # A zero-length path is only correct when the two positions are the
        # same position.  Sharing a mask cell is not the same claim: the grid
        # is 1 arc-minute, so two distinct observed coordinates up to a cell
        # diagonal apart land in one cell, and collapsing them discarded the
        # goal and reported total_km = 0 for a real displacement.  The mask
        # resolution bounds where the router may claim to know about land; it
        # is not licence to forget a coordinate the caller supplied.  Distinct
        # positions fall through to the ordinary great-circle path below, so
        # they keep the same distance, navigability and speed treatment as
        # every other route.
        if start_position == goal_position:
            detail["strategy"] = "coincident_positions"
            detail["elapsed_s"] = time.monotonic() - started
            return RouteResolution(
                SeaPath(
                    waypoints=np.asarray([start_position], dtype=float),
                    cumulative_km=np.array([0.0]),
                    total_km=0.0,
                ),
                ROUTE_RESOLVED,
                detail,
            )

        route_component = int(
            self._component_roots[self.components[start]]
        )
        goal_component = int(
            self._component_roots[self.components[goal]]
        )
        if route_component != goal_component:
            detail["start_component"] = route_component
            detail["goal_component"] = goal_component
            return RouteResolution(None, ROUTE_BASINS_DISCONNECTED, detail)

        direct = great_circle_waypoints(*start_position, *goal_position)
        blocked = [
            not self._segment_is_ocean(
                left, right, required_component=route_component,
            )
            for left, right in zip(direct[:-1], direct[1:])
        ]
        detail["blocked_direct_segment_count"] = int(sum(blocked))
        if not any(blocked):
            detail["strategy"] = "direct_great_circle"
            detail["blocked_direct_run_count"] = 0
            detail["elapsed_s"] = time.monotonic() - started
            return self._speed_gated(
                self._sea_path_from_waypoints(direct), detail,
                duration_hours, max_speed_knots,
            )

        runs = self._blocked_runs(blocked)
        detail["strategy"] = "great_circle_with_local_astar_for_blocked_runs"
        detail["blocked_direct_run_count"] = len(runs)

        chunks: list[np.ndarray] = []
        direct_cursor = 0
        try:
            for blocked_start, blocked_end in runs:
                entry_index = blocked_start
                exit_index = blocked_end + 1
                chunks.append(direct[direct_cursor : entry_index + 1])

                entry_cell = self._position_to_index(*direct[entry_index])
                exit_cell = self._position_to_index(*direct[exit_index])
                detour = self._find_detour_grid_path(
                    entry_cell, exit_cell, deadline=deadline,
                )
                if detour is None:
                    detail["failed_detour_entry_cell"] = entry_cell
                    detail["failed_detour_exit_cell"] = exit_cell
                    return RouteResolution(None, ROUTE_NO_PATH, detail)
                smoothed = self._smooth_grid_path(detour)
                chunks.append(self._grid_path_waypoints(smoothed))
                direct_cursor = exit_index
        except _SearchTimeout:
            detail["elapsed_s"] = time.monotonic() - started
            return RouteResolution(None, ROUTE_SEARCH_TIMEOUT, detail)

        chunks.append(direct[direct_cursor:])
        waypoints = self._combine_waypoint_chunks(chunks)
        if any(
            not self._segment_is_ocean(left, right)
            and not self._edge_is_explicit_channel(
                self._position_to_index(*left), self._position_to_index(*right),
            )
            for left, right in zip(waypoints[:-1], waypoints[1:])
        ):
            raise RuntimeError("Final route failed the ocean-only segment audit")
        detail["elapsed_s"] = time.monotonic() - started
        return self._speed_gated(
            self._sea_path_from_waypoints(waypoints), detail,
            duration_hours, max_speed_knots,
        )


# ------------------------------------------------------------------
# Time-linear interpolation along a sea path
# ------------------------------------------------------------------

def implied_speed_kmh(path: SeaPath, t_start: float, t_end: float) -> float:
    """Return the constant speed required to traverse a path in the interval."""
    if not math.isfinite(t_start) or not math.isfinite(t_end):
        raise ValueError("t_start and t_end must be finite")
    duration_seconds = t_end - t_start
    if duration_seconds <= 0.0:
        raise ValueError(f"t_end must be after t_start: {t_end} <= {t_start}")
    return path.total_km / (duration_seconds / 3600.0)


def interpolate_along_path(
    path: SeaPath,
    t_start: float,
    t_end: float,
    target_times: np.ndarray,
    *,
    max_speed_kmh: float | None = None,
) -> np.ndarray:
    """Interpolate positions along a sea path at given times.

    Uses constant-speed assumption: position at time t is placed at
    the point along the path where cumulative distance equals
    (t - t_start) / (t_end - t_start) × total_distance.

    Parameters
    ----------
    path : SeaPath from find_path()
    t_start, t_end : float, epoch seconds of the known boundary fixes
    target_times : 1-D array of epoch seconds to interpolate

    Returns
    -------
    positions : array of shape (len(target_times), 2), columns (lat, lon)
    """
    if not math.isfinite(t_start) or not math.isfinite(t_end):
        raise ValueError("t_start and t_end must be finite")
    duration = t_end - t_start
    if duration <= 0:
        raise ValueError(f"t_end must be after t_start: {t_end} <= {t_start}")
    if max_speed_kmh is not None:
        if not math.isfinite(max_speed_kmh) or max_speed_kmh <= 0.0:
            raise ValueError(
                f"max_speed_kmh must be positive and finite: {max_speed_kmh}"
            )
        required_speed = implied_speed_kmh(path, t_start, t_end)
        if required_speed > max_speed_kmh:
            raise ValueError(
                f"route requires {required_speed:.6f} km/h, above "
                f"max_speed_kmh={max_speed_kmh:.6f}"
            )

    target_times = np.asarray(target_times, dtype=float)
    if target_times.ndim != 1:
        raise ValueError(f"target_times must be one-dimensional: {target_times.shape}")
    if not np.isfinite(target_times).all():
        raise ValueError("target_times must contain only finite values")
    outside = (target_times < t_start) | (target_times > t_end)
    if outside.any():
        first = float(target_times[np.flatnonzero(outside)[0]])
        raise ValueError(
            f"target time outside anchor interval [{t_start}, {t_end}]: {first}"
        )

    fractions = (target_times - t_start) / duration

    target_dists = fractions * path.total_km

    positions = np.empty((len(target_times), 2))

    if path.total_km == 0.0 or len(path.waypoints) == 1:
        positions[:, 0] = path.waypoints[0, 0]
        positions[:, 1] = path.waypoints[0, 1]
        return positions

    wp = path.waypoints
    cum = path.cumulative_km
    n_wp = len(wp)

    seg_idx = 0
    sort_order = np.argsort(target_dists)

    for rank in sort_order:
        d = target_dists[rank]

        while seg_idx < n_wp - 2 and cum[seg_idx + 1] < d:
            seg_idx += 1

        if seg_idx >= n_wp - 1:
            positions[rank] = wp[-1]
            continue

        seg_len = cum[seg_idx + 1] - cum[seg_idx]
        if seg_len < 1e-12:
            positions[rank] = wp[seg_idx]
            continue

        t = (d - cum[seg_idx]) / seg_len
        positions[rank, 0] = wp[seg_idx, 0] + t * (wp[seg_idx + 1, 0] - wp[seg_idx, 0])
        lon_delta = shortest_longitude_delta(
            float(wp[seg_idx, 1]), float(wp[seg_idx + 1, 1]),
        )
        positions[rank, 1] = wrap_longitude(wp[seg_idx, 1] + t * lon_delta)

    return positions
