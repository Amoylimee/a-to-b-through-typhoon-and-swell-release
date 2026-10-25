"""Tests for the sea-path routing algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from sea_path.ocean_mask import latlon_to_index
from sea_path.routing import (
    SeaPath,
    SeaPathRouter,
    _label_ocean_components,
    _supercover_grid_line,
    great_circle_waypoints,
    implied_speed_kmh,
    interpolate_along_path,
    split_path_at_antimeridian,
)


def synthetic_router(mask, explicit_edges=frozenset()):
    """Build a small-index router for artificial-mask path tests."""
    r = SeaPathRouter.__new__(SeaPathRouter)
    r.mask = mask
    r.n_lat, r.n_lon = mask.shape
    r._lat_min = 0.0
    r._lon_min = 0.0
    r._resolution_deg = 1.0
    r._channel_edges = frozenset(
        (left, right) if left <= right else (right, left)
        for left, right in explicit_edges
    )
    r.components, r._component_roots, r.n_components = _label_ocean_components(
        mask, r._channel_edges,
    )
    r._lat_cos = np.ones(r.n_lat)
    return r


# ------------------------------------------------------------------
# Synthetic mask tests (no real data needed)
# ------------------------------------------------------------------

class TestSyntheticRouting:
    """Tests on a small hand-crafted ocean mask."""

    @staticmethod
    def _small_router():
        """7×7 grid with a horizontal wall and one navigable gap."""
        mask = np.ones((7, 7), dtype=bool)
        mask[3, :] = False
        mask[3, 5] = True
        return synthetic_router(mask)

    def test_path_through_gap(self):
        r = self._small_router()
        path = r._find_grid_path((1, 2), (5, 2))
        assert path is not None
        assert (3, 5) in path

    def test_public_router_detours_only_blocked_arc_on_artificial_mask(self):
        r = self._small_router()

        path = r.find_path(1.0, 2.0, 5.0, 2.0)

        assert path is not None
        visited = {r._position_to_index(*point) for point in path.waypoints}
        assert (3, 5) in visited
        assert all(
            r._segment_is_ocean(left, right)
            for left, right in zip(path.waypoints[:-1], path.waypoints[1:])
        )

    def test_disconnected_returns_none(self):
        mask = np.ones((7, 7), dtype=bool)
        mask[3, :] = False
        r = synthetic_router(mask)
        assert r._find_grid_path((1, 2), (5, 2)) is None

    def test_periodic_component_labels_join_across_antimeridian(self):
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 0] = True
        mask[1, -1] = True

        components, roots, count = _label_ocean_components(mask)

        assert count == 1
        assert roots[components[1, 0]] == roots[components[1, -1]]

    def test_periodic_component_labels_reject_corner_only_diagonal(self):
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 0] = True
        mask[1, -1] = True

        components, roots, count = _label_ocean_components(mask)

        assert count == 2
        assert roots[components[0, 0]] != roots[components[1, -1]]

    def test_corner_only_diagonal_is_not_a_neighbor(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        r = synthetic_router(mask)

        assert (1, 1) not in set(r._neighbors(0, 0))
        assert r._find_grid_path((0, 0), (1, 1)) is None

    def test_named_channel_edge_is_an_explicit_corner_exception(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        edge = ((0, 0), (1, 1))
        r = synthetic_router(mask, frozenset({edge}))

        assert (1, 1) in set(r._neighbors(0, 0))
        assert r._find_grid_path((0, 0), (1, 1)) == [(0, 0), (1, 1)]

    def test_supercover_includes_both_cells_at_corner_crossing(self):
        cells = _supercover_grid_line((0, 0), (1, 1), n_lat=3, n_lon=3)
        assert set(cells) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_smoothing_removes_visible_astar_turns(self):
        mask = np.ones((5, 7), dtype=bool)
        r = synthetic_router(mask)
        raw = [(1, 1), (2, 2), (3, 3), (3, 4), (3, 5)]

        smoothed = r._smooth_grid_path(raw)

        assert smoothed == [raw[0], raw[-1]]


class TestGreatCircleGeometry:
    def test_transpacific_arc_curves_north_and_crosses_antimeridian(self):
        points = great_circle_waypoints(34.0, 140.0, 33.7, -118.3)

        assert points[:, 0].max() > 45.0
        assert np.any(np.abs(np.diff(points[:, 1])) > 180.0)

    def test_direct_arc_preserves_exact_endpoints(self):
        points = great_circle_waypoints(5.1234, 140.4321, 6.2345, 150.5432)
        np.testing.assert_allclose(points[0], [5.1234, 140.4321])
        np.testing.assert_allclose(points[-1], [6.2345, 150.5432])


# ------------------------------------------------------------------
# Real mask: path finding
# ------------------------------------------------------------------

class TestRealPaths:
    def test_same_point_returns_zero_distance(self, router):
        path = router.find_path(20.0, 38.0, 20.0, 38.0)
        assert path is not None
        assert path.total_km == 0.0
        assert len(path.waypoints) == 1

    def test_nearby_ocean_points(self, router):
        """Two points in the Red Sea, ~100 km apart."""
        path = router.find_path(20.0, 38.0, 21.0, 38.0)
        assert path is not None
        assert 100 < path.total_km < 150
        assert len(path.waypoints) >= 2

    def test_path_does_not_cross_land(self, router, ocean_mask):
        """Gulf of Aden to Red Sea — path must stay in the ocean."""
        path = router.find_path(12.5, 48.0, 20.0, 38.0)
        assert path is not None
        for lat, lon in path.waypoints:
            i, j = latlon_to_index(lat, lon)
            assert ocean_mask[i, j], f"Land crossing at ({lat}, {lon})"
        assert all(
            router._segment_is_ocean(left, right)
            or router._edge_is_explicit_channel(
                latlon_to_index(*left), latlon_to_index(*right),
            )
            for left, right in zip(path.waypoints[:-1], path.waypoints[1:])
        )

    def test_bab_el_mandeb_passage(self, router):
        """Gulf of Aden → Red Sea through Bab el-Mandeb."""
        path = router.find_path(12.0, 45.0, 15.0, 42.0)
        assert path is not None
        assert path.total_km > 0

    def test_cape_route_segment(self, router):
        """Indian Ocean segment south of Africa."""
        path = router.find_path(-30.0, 35.0, -35.0, 20.0)
        assert path is not None
        assert path.total_km > 0

    def test_suez_canal_routes_through(self, router):
        """Med → Red Sea routes through the opened Suez Canal (~500 km)."""
        path = router.find_path(31.5, 32.3, 29.5, 32.6)
        assert path is not None
        assert path.total_km < 500

    def test_panama_canal_connects_nearshore_endpoints_locally(self, router):
        """Both Panama canal entrances connect to their adjacent seas."""
        entry = (5961, 6006)
        exit = (5935, 6026)

        grid_path = router._find_grid_path(
            entry, exit, margin_cells=16,
        )

        assert grid_path is not None
        assert any(
            router._edge_is_explicit_channel(left, right)
            for left, right in zip(grid_path[:-1], grid_path[1:])
        )

    def test_panama_nearshore_public_route_is_ocean_safe(self, router):
        path = router.find_path(9.45, -79.95, 8.90, -79.55, snap=True)

        assert path is not None
        assert path.total_km < 150.0
        assert all(
            router._segment_is_ocean(left, right)
            or router._edge_is_explicit_channel(
                router._position_to_index(*left),
                router._position_to_index(*right),
            )
            for left, right in zip(path.waypoints[:-1], path.waypoints[1:])
        )

    def test_black_sea_routes_through_bosporus(self, router):
        """Black Sea → Med routes through the opened Bosporus."""
        path = router.find_path(43.0, 35.0, 38.0, 24.0, snap=True)
        assert path is not None

    def test_short_antimeridian_crossing(self, router):
        """Nearby Pacific points on opposite longitude edges stay nearby."""
        path = router.find_path(0.0, 179.9, 0.0, -179.9)
        assert path is not None
        assert path.total_km < 30.0
        assert np.any(np.abs(np.diff(path.waypoints[:, 1])) > 180.0)

        midpoint = interpolate_along_path(path, 0.0, 1.0, np.array([0.5]))[0]
        assert abs(midpoint[0]) < 0.05
        assert abs(abs(midpoint[1]) - 180.0) < 0.05

    def test_open_transpacific_path_has_great_circle_geometry(self, router):
        path = router.find_path(34.0, 140.0, 33.7, -118.3, snap=True)
        assert path is not None
        direct = router.haversine_km(34.0, 140.0, 33.7, -118.3)
        assert path.total_km < direct * 1.01
        assert path.waypoints[:, 0].max() > 45.0
        np.testing.assert_allclose(path.waypoints[0], [34.0, 140.0])
        assert router.mask[latlon_to_index(*path.waypoints[-1])]

    def test_snap_near_coast(self, router):
        """Point slightly on land near a coast should snap to ocean."""
        snapped = router.snap_to_ocean(31.0, 32.0, max_cells=5)
        assert snapped is not None
        i, j = snapped
        assert router.mask[i, j]

    def test_snap_deep_inland_returns_none(self, router):
        """Point deep inland should fail to snap."""
        snapped = router.snap_to_ocean(25.0, 30.0, max_cells=3)
        assert snapped is None

    def test_find_path_does_not_snap_by_default(self, router):
        assert router.find_path(25.0, 30.0, 20.0, 38.0) is None

    def test_find_path_rejects_out_of_range_longitude(self, router):
        with pytest.raises(ValueError, match="Longitude outside"):
            router.find_path(0.0, 181.0, 0.0, 170.0)


# ------------------------------------------------------------------
# Path quality
# ------------------------------------------------------------------

class TestPathQuality:
    def test_straight_ocean_path_is_roughly_haversine(self, router):
        """Two points in open ocean — path should be close to great circle."""
        lat_a, lon_a = 0.0, 60.0
        lat_b, lon_b = 0.0, 65.0
        path = router.find_path(lat_a, lon_a, lat_b, lon_b)
        assert path is not None
        direct = router.haversine_km(lat_a, lon_a, lat_b, lon_b)
        assert path.total_km < direct * 1.15
        np.testing.assert_allclose(path.waypoints[0], [lat_a, lon_a])
        np.testing.assert_allclose(path.waypoints[-1], [lat_b, lon_b])

    def test_path_waypoints_are_monotonically_increasing_distance(self, router):
        path = router.find_path(12.0, 45.0, 20.0, 38.0)
        assert path is not None
        diffs = np.diff(path.cumulative_km)
        assert (diffs >= 0).all()


# ------------------------------------------------------------------
# Time interpolation
# ------------------------------------------------------------------

class TestInterpolation:
    def test_midpoint_is_halfway(self):
        wp = np.array([[0.0, 0.0], [10.0, 0.0]])
        cum = np.array([0.0, 100.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=100.0)

        t_start = 0.0
        t_end = 100.0
        target = np.array([50.0])
        result = interpolate_along_path(path, t_start, t_end, target)
        assert abs(result[0, 0] - 5.0) < 0.01
        assert abs(result[0, 1] - 0.0) < 0.01

    def test_endpoints_match(self):
        wp = np.array([[10.0, 20.0], [30.0, 40.0]])
        cum = np.array([0.0, 200.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=200.0)

        result = interpolate_along_path(path, 0.0, 100.0, np.array([0.0, 100.0]))
        np.testing.assert_allclose(result[0], [10.0, 20.0], atol=0.01)
        np.testing.assert_allclose(result[1], [30.0, 40.0], atol=0.01)

    def test_multiple_segments(self):
        wp = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        cum = np.array([0.0, 100.0, 200.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=200.0)

        targets = np.array([25.0, 50.0, 75.0, 100.0])
        result = interpolate_along_path(path, 0.0, 100.0, targets)
        assert abs(result[0, 0] - 5.0) < 0.01   # 25% along first segment
        assert abs(result[1, 0] - 10.0) < 0.01   # end of first segment
        assert abs(result[2, 1] - 5.0) < 0.01    # 25% along second segment
        assert abs(result[3, 1] - 10.0) < 0.01   # end of path

    def test_zero_length_path(self):
        wp = np.array([[5.0, 10.0]])
        cum = np.array([0.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=0.0)

        result = interpolate_along_path(path, 0.0, 100.0, np.array([50.0]))
        np.testing.assert_allclose(result[0], [5.0, 10.0])

    def test_rejects_times_outside_anchor_range(self):
        wp = np.array([[0.0, 0.0], [10.0, 0.0]])
        cum = np.array([0.0, 100.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=100.0)

        with pytest.raises(ValueError, match="outside anchor interval"):
            interpolate_along_path(path, 10.0, 20.0, np.array([5.0, 25.0]))

    def test_rejects_nonfinite_target_times(self):
        wp = np.array([[0.0, 0.0], [10.0, 0.0]])
        cum = np.array([0.0, 100.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=100.0)

        with pytest.raises(ValueError, match="finite"):
            interpolate_along_path(path, 0.0, 100.0, np.array([np.nan]))

    @pytest.mark.parametrize("bad_anchor", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_anchor_times(self, bad_anchor):
        wp = np.array([[0.0, 0.0], [10.0, 0.0]])
        cum = np.array([0.0, 100.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=100.0)

        with pytest.raises(ValueError, match="finite"):
            interpolate_along_path(path, bad_anchor, 100.0, np.array([50.0]))
        with pytest.raises(ValueError, match="finite"):
            implied_speed_kmh(path, 0.0, bad_anchor)

    def test_rejects_route_above_supplied_speed_cap(self):
        wp = np.array([[0.0, 0.0], [0.0, 1.0]])
        cum = np.array([0.0, 100.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=100.0)

        assert implied_speed_kmh(path, 0.0, 7200.0) == 50.0
        with pytest.raises(ValueError, match="above max_speed_kmh"):
            interpolate_along_path(
                path,
                0.0,
                7200.0,
                np.array([3600.0]),
                max_speed_kmh=40.0,
            )

    def test_antimeridian_midpoint_stays_at_antimeridian(self):
        wp = np.array([[0.0, 179.991667], [0.0, -179.991667]])
        cum = np.array([0.0, 1.853])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=1.853)

        result = interpolate_along_path(path, 0.0, 3600.0, np.array([1800.0]))

        assert abs(result[0, 0]) < 1e-12
        assert abs(abs(result[0, 1]) - 180.0) < 1e-6

    @pytest.mark.parametrize(
        ("start_lon", "end_lon", "expected_lons"),
        [
            (179.0, -179.0, [179.5, -179.5]),
            (-179.0, 179.0, [-179.5, 179.5]),
        ],
    )
    def test_antimeridian_interpolation_uses_shortest_direction(
        self, start_lon, end_lon, expected_lons,
    ):
        wp = np.array([[5.0, start_lon], [5.0, end_lon]])
        cum = np.array([0.0, 200.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=200.0)

        result = interpolate_along_path(
            path, 0.0, 100.0, np.array([25.0, 75.0]),
        )

        np.testing.assert_allclose(result[:, 0], [5.0, 5.0])
        np.testing.assert_allclose(result[:, 1], expected_lons)

    def test_antimeridian_plot_segments_do_not_cross_map(self):
        wp = np.array([[10.0, 179.0], [11.0, -179.0]])
        cum = np.array([0.0, 200.0])
        path = SeaPath(waypoints=wp, cumulative_km=cum, total_km=200.0)

        segments = split_path_at_antimeridian(path)

        assert len(segments) == 2
        np.testing.assert_allclose(segments[0][-1], [10.5, 180.0])
        np.testing.assert_allclose(segments[1][0], [10.5, -180.0])
        assert all(
            (np.abs(np.diff(segment[:, 1])) <= 180.0).all()
            for segment in segments
        )


# ------------------------------------------------------------------
# Unresolved reasons must stay distinguishable
# ------------------------------------------------------------------

class TestRouteResolutionReasons:
    """A bare ``None`` cannot support a reported unresolved rate.

    An anchor that is not navigable, two unreachable basins, and a failed
    detour are different measurements; a denominator that merges them cannot
    be interpreted.  These tests hold them apart.
    """

    def test_resolved_route_reports_its_strategy(self, router):
        from sea_path.routing import ROUTE_RESOLVED
        result = router.resolve_path(20.0, 38.0, 21.0, 38.0)
        assert result.status == ROUTE_RESOLVED
        assert result.path is not None
        assert result.detail["strategy"] in {
            "direct_great_circle",
            "great_circle_with_local_astar_for_blocked_runs",
            "coincident_positions",
        }

    def test_land_anchor_is_reported_with_its_distance_to_water(self, router):
        """A Suez-corridor position 4 km inland is a resolution artefact."""
        from sea_path.routing import ROUTE_ANCHOR_NOT_NAVIGABLE
        result = router.resolve_path(29.20, 32.60, 31.60, 32.35)
        assert result.path is None
        assert result.status == ROUTE_ANCHOR_NOT_NAVIGABLE
        assert result.detail["blocking_anchors"] == ["start"]
        assert 0.0 < result.detail["start_nearest_ocean_km"] < 20.0
        assert result.detail["goal_nearest_ocean_km"] == 0.0

    def test_deep_inland_anchor_reports_no_water_in_range(self, router):
        from sea_path.routing import ROUTE_ANCHOR_NOT_NAVIGABLE
        result = router.resolve_path(25.0, 30.0, 20.0, 38.0)
        assert result.status == ROUTE_ANCHOR_NOT_NAVIGABLE
        assert result.detail["start_nearest_ocean_km"] is None

    def test_isolated_basin_is_reported_as_disconnected_not_as_a_bad_anchor(
        self, router,
    ):
        """The Vistula Lagoon is navigable water with no route to the Red Sea."""
        from sea_path.routing import ROUTE_BASINS_DISCONNECTED
        result = router.resolve_path(54.225, 19.375, 20.0, 38.0)
        assert result.path is None
        assert result.status == ROUTE_BASINS_DISCONNECTED
        assert result.detail["start_component"] != result.detail["goal_component"]

    def test_find_path_is_resolve_path_without_the_reason(self, router):
        for args in [
            (20.0, 38.0, 21.0, 38.0),
            (29.20, 32.60, 31.60, 32.35),
            (54.225, 19.375, 20.0, 38.0),
        ]:
            plain = router.find_path(*args)
            rich = router.resolve_path(*args)
            if plain is None:
                assert rich.path is None
            else:
                np.testing.assert_allclose(plain.waypoints, rich.path.waypoints)
                assert plain.total_km == rich.path.total_km

    def test_every_status_is_a_declared_one(self, router):
        from sea_path.routing import ROUTE_STATUSES
        for args in [
            (20.0, 38.0, 21.0, 38.0),
            (29.20, 32.60, 31.60, 32.35),
            (54.225, 19.375, 20.0, 38.0),
            (25.0, 30.0, 20.0, 38.0),
        ]:
            assert router.resolve_path(*args).status in ROUTE_STATUSES

    def test_nearest_ocean_does_not_move_a_navigable_anchor(self, router):
        distance, cell = router.nearest_ocean(20.0, 38.0)
        assert distance == 0.0
        assert router.mask[cell]


# ------------------------------------------------------------------
# Production guards: wall-clock budget and fleet speed ceiling
# ------------------------------------------------------------------

class TestRouteBudgetAndSpeedGate:
    """Both guards turn a bad route into an explicit record, never a bad answer."""

    def test_frozen_defaults_match_the_contract(self):
        from sea_path.routing import DEFAULT_MAX_SPEED_KNOTS, DEFAULT_ROUTE_TIMEOUT_S
        assert DEFAULT_ROUTE_TIMEOUT_S == 5.0
        assert DEFAULT_MAX_SPEED_KNOTS == 25.0

    def test_search_timeout_is_reported_not_approximated(self, router):
        """A budget that expires must yield no path at all, not a partial one."""
        from sea_path.routing import ROUTE_SEARCH_TIMEOUT
        result = router.resolve_path(12.0, 45.0, 15.0, 42.0, timeout_s=0.002)
        assert result.status == ROUTE_SEARCH_TIMEOUT
        assert result.path is None

    def test_same_route_resolves_without_a_budget(self, router):
        from sea_path.routing import ROUTE_RESOLVED
        result = router.resolve_path(12.0, 45.0, 15.0, 42.0, timeout_s=None)
        assert result.status == ROUTE_RESOLVED
        assert result.path is not None

    def test_default_budget_leaves_ordinary_routes_untouched(self, router):
        from sea_path.routing import ROUTE_RESOLVED
        for args in [(12.0, 45.0, 15.0, 42.0), (29.5, 32.6, 31.5, 32.3),
                     (-35.5, 18.0, -34.0, 25.0), (20.0, 38.0, 21.0, 38.0)]:
            assert router.resolve_path(*args).status == ROUTE_RESOLVED

    def test_speed_gate_rejects_a_route_the_vessel_could_not_sail(self, router):
        from sea_path.routing import ROUTE_SPEED_INFEASIBLE
        result = router.resolve_path(20.0, 38.0, 21.2, 37.9, duration_hours=0.5)
        assert result.status == ROUTE_SPEED_INFEASIBLE
        assert result.path is None
        assert result.detail["implied_speed_knots"] > 25.0

    def test_speed_gate_admits_a_feasible_route(self, router):
        from sea_path.routing import ROUTE_RESOLVED
        result = router.resolve_path(20.0, 38.0, 21.2, 37.9, duration_hours=12.0)
        assert result.status == ROUTE_RESOLVED
        assert result.detail["implied_speed_knots"] < 25.0

    def test_speed_gate_is_skipped_without_a_duration(self, router):
        from sea_path.routing import ROUTE_RESOLVED
        result = router.resolve_path(20.0, 38.0, 21.2, 37.9)
        assert result.status == ROUTE_RESOLVED
        assert "implied_speed_knots" not in result.detail

    def test_invalid_guard_values_are_rejected(self, router):
        for kwargs in ({"timeout_s": 0.0}, {"timeout_s": -1.0},
                       {"max_speed_knots": 0.0}, {"max_speed_knots": -5.0}):
            with pytest.raises(ValueError):
                router.resolve_path(20.0, 38.0, 21.0, 38.0, **kwargs)
        with pytest.raises(ValueError):
            router.resolve_path(20.0, 38.0, 21.0, 38.0, duration_hours=0.0)

    def test_every_status_stays_in_the_declared_vocabulary(self, router):
        from sea_path.routing import ROUTE_STATUSES
        for kwargs in ({}, {"timeout_s": 0.002}, {"duration_hours": 0.5}):
            assert router.resolve_path(12.0, 45.0, 15.0, 42.0, **kwargs).status in ROUTE_STATUSES


class TestPositionsInsideOneMaskCell:
    """A shared mask cell is not a shared position.

    The mask is 1 arc-minute, about 1.85 km of latitude, so two distinct
    observed coordinates routinely land in one cell.  The router used to treat
    that as arrival: it returned the start alone, discarded the goal and
    reported ``total_km = 0`` for a real displacement, which then froze the
    consumer's interpolated hours at the leg start.  Every coordinate pair
    below is real production input taken from the 2025 reconstruction, so these
    tests fail against the collapsing router rather than describing it.

    The mask resolution bounds what the router may claim to know about land.
    It is not licence to forget a coordinate the caller supplied.
    """

    # One cell of gap_cfc37172756403143507, the 24 h case whose subsegments
    # were all degenerate: two GFW hour-box centres 0.01 degrees apart.
    ONE_CELL = (35.280000, 119.820000, 35.280000, 119.830002)
    # Widest separation observed across the 54,055 affected legs, in open water.
    WIDEST = (-0.385462, 87.015862, -0.400000, 87.000000)
    # An affected pair whose cell has seven land neighbours out of eight.
    BESIDE_LAND = (35.470000, 135.3300018310547, 35.472252, 135.330600)
    # Two positions inside the westernmost column, hard against the seam.
    AT_SEAM = (0.0, -179.999, 0.0, -179.985)
    # Endpoint tolerance in degrees; see the note in the endpoint test below.
    ENDPOINT_DEG = 1e-12

    def _same_cell(self, router, lat_a, lon_a, lat_b, lon_b):
        return (router._position_to_index(lat_a, lon_a)
                == router._position_to_index(lat_b, lon_b))

    def test_identical_positions_remain_a_legal_zero_length_route(self, router):
        from sea_path.routing import ROUTE_RESOLVED
        result = router.resolve_path(35.28, 119.82, 35.28, 119.82, snap=False)
        assert result.status == ROUTE_RESOLVED
        assert result.detail["strategy"] == "coincident_positions"
        assert len(result.path.waypoints) == 1
        assert result.path.total_km == 0.0

    def test_distinct_positions_in_one_cell_keep_both_endpoints(self, router):
        lat_a, lon_a, lat_b, lon_b = self.ONE_CELL
        assert self._same_cell(router, lat_a, lon_a, lat_b, lon_b)
        path = router.resolve_path(lat_a, lon_a, lat_b, lon_b, snap=False).path
        # resolve_path normalizes longitude through wrap_longitude, so the
        # round trip can differ in the last bit -- 1e-12 degrees is about a
        # tenth of a micrometre, ten orders below the 0.01-degree collapse
        # this guards.  Do not tighten this to exact equality.
        assert path.waypoints[0] == pytest.approx((lat_a, lon_a), abs=self.ENDPOINT_DEG)
        assert path.waypoints[-1] == pytest.approx((lat_b, lon_b), abs=self.ENDPOINT_DEG)

    def test_distinct_positions_in_one_cell_report_the_real_distance(self, router):
        lat_a, lon_a, lat_b, lon_b = self.ONE_CELL
        path = router.resolve_path(lat_a, lon_a, lat_b, lon_b, snap=False).path
        straight = router.haversine_km(lat_a, lon_a, lat_b, lon_b)
        assert straight > 0.9  # 0.01 degrees of longitude at 35 degrees north
        assert path.total_km == pytest.approx(straight, abs=1e-9)
        assert path.cumulative_km[-1] == pytest.approx(path.total_km, abs=0.0)

    def test_a_one_cell_route_is_an_ordinary_route(self, router):
        """No special case: it earns the same land and speed treatment."""
        lat_a, lon_a, lat_b, lon_b = self.ONE_CELL
        result = router.resolve_path(lat_a, lon_a, lat_b, lon_b, snap=False)
        assert result.detail["strategy"] == "direct_great_circle"

    def test_the_widest_observed_separation_still_resolves(self, router):
        from sea_path.routing import ROUTE_RESOLVED
        lat_a, lon_a, lat_b, lon_b = self.WIDEST
        assert self._same_cell(router, lat_a, lon_a, lat_b, lon_b)
        result = router.resolve_path(
            lat_a, lon_a, lat_b, lon_b, snap=False, duration_hours=0.5491666666666667)
        assert result.status == ROUTE_RESOLVED
        assert result.path.total_km == pytest.approx(2.392486604, abs=1e-6)
        assert result.detail["implied_speed_knots"] < 25.0

    def test_a_one_cell_route_beside_land_still_resolves(self, router):
        """Seven of the eight neighbouring cells are land; the segment is not."""
        from sea_path.routing import ROUTE_RESOLVED
        lat_a, lon_a, lat_b, lon_b = self.BESIDE_LAND
        assert self._same_cell(router, lat_a, lon_a, lat_b, lon_b)
        result = router.resolve_path(
            lat_a, lon_a, lat_b, lon_b, snap=False, duration_hours=1.3233333333333333)
        assert result.status == ROUTE_RESOLVED
        assert result.path.total_km == pytest.approx(
            router.haversine_km(lat_a, lon_a, lat_b, lon_b), abs=1e-9)

    def test_a_one_cell_route_obeys_the_speed_ceiling(self, router):
        """A zero-length path passed any duration; a real one must not."""
        from sea_path.routing import ROUTE_SPEED_INFEASIBLE
        lat_a, lon_a, lat_b, lon_b = self.WIDEST
        result = router.resolve_path(
            lat_a, lon_a, lat_b, lon_b, snap=False,
            duration_hours=0.01, max_speed_knots=25.0)
        assert result.status == ROUTE_SPEED_INFEASIBLE
        assert result.path is None
        assert result.detail["implied_speed_knots"] > 25.0

    def test_a_one_cell_route_at_the_seam_is_not_a_wraparound(self, router):
        lat_a, lon_a, lat_b, lon_b = self.AT_SEAM
        assert self._same_cell(router, lat_a, lon_a, lat_b, lon_b)
        path = router.resolve_path(lat_a, lon_a, lat_b, lon_b, snap=False).path
        assert path.total_km == pytest.approx(1.556728973, abs=1e-6)
        assert path.waypoints[-1][1] == pytest.approx(lon_b, abs=self.ENDPOINT_DEG)
        assert (np.abs(path.waypoints[:, 1] + 180.0) < 0.02).all()

    def test_hourly_interpolation_moves_along_a_one_cell_route(self, router):
        """The consequence a consumer sees: the vessel is no longer frozen."""
        lat_a, lon_a, lat_b, lon_b = self.ONE_CELL
        path = router.resolve_path(lat_a, lon_a, lat_b, lon_b, snap=False).path
        t_start, t_end = 1_755_092_000.0, 1_755_095_600.0
        targets = t_start + np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * (t_end - t_start)
        positions = interpolate_along_path(path, t_start, t_end, targets)

        assert positions[0] == pytest.approx((lat_a, lon_a), abs=self.ENDPOINT_DEG)
        assert positions[-1] == pytest.approx((lat_b, lon_b), abs=self.ENDPOINT_DEG)
        travelled = [router.haversine_km(lat_a, lon_a, lat, lon) for lat, lon in positions]
        assert all(b > a for a, b in zip(travelled, travelled[1:]))
        assert travelled[2] == pytest.approx(path.total_km / 2.0, rel=1e-6)
        # The defect it replaces: every hour sat on the start position.
        assert travelled[2] > 0.4

    def test_consecutive_one_cell_legs_stay_continuous(self, router):
        """End of one leg is the start of the next, so the track has no jump."""
        nodes = [(35.280807, 119.825010), (35.280000, 119.820000),
                 (35.280000, 119.830002), (35.280437, 119.824760)]
        for left, right in zip(nodes[:-1], nodes[1:]):
            path = router.resolve_path(*left, *right, snap=False).path
            assert path.waypoints[0] == pytest.approx(left, abs=self.ENDPOINT_DEG)
            assert path.waypoints[-1] == pytest.approx(right, abs=self.ENDPOINT_DEG)
            assert path.total_km > 0.0

    def test_interpolating_a_zero_length_route_stays_on_the_point(self, router):
        """Unchanged behaviour for genuinely coincident anchors."""
        path = router.resolve_path(35.28, 119.82, 35.28, 119.82, snap=False).path
        targets = np.array([0.0, 1800.0, 3600.0]) + 1_755_092_000.0
        positions = interpolate_along_path(path, 1_755_092_000.0, 1_755_095_600.0, targets)
        assert positions == pytest.approx(
            np.repeat([[35.28, 119.82]], 3, axis=0), abs=self.ENDPOINT_DEG)
