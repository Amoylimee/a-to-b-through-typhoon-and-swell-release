"""Tests for the ocean mask preprocessing and index conventions."""

from __future__ import annotations

import numpy as np
import pytest

from sea_path.ocean_mask import (
    MARITIME_CHANNEL_EDGE_OVERRIDES,
    MARITIME_CHANNEL_OVERRIDES,
    N_LAT,
    N_LON,
    build_ocean_mask,
    index_to_latlon,
    latlon_to_index,
    maritime_channel_edges,
)


# ---------------------------------------------------------------------------
# Index conversion round-trip
# ---------------------------------------------------------------------------

class TestIndexConversion:
    def test_origin_south_west(self):
        i, j = latlon_to_index(-89.991667, -179.991667)
        assert (i, j) == (0, 0)

    def test_origin_north_east(self):
        i, j = latlon_to_index(89.991667, 179.991667)
        assert (i, j) == (N_LAT - 1, N_LON - 1)

    def test_equator_prime_meridian(self):
        i, j = latlon_to_index(0.05, 0.05)
        lat, lon = index_to_latlon(i, j)
        assert abs(lat - 0.05) < 0.01
        assert abs(lon - 0.05) < 0.01

    def test_round_trip(self):
        for lat_val, lon_val in [
            (31.25, 32.35),
            (-33.85, 18.45),
            (1.25, 103.85),
            (51.05, 1.35),
        ]:
            i, j = latlon_to_index(lat_val, lon_val)
            lat_back, lon_back = index_to_latlon(i, j)
            assert abs(lat_back - lat_val) < 0.051
            assert abs(lon_back - lon_val) < 0.051

    def test_clamp_out_of_range(self):
        i, j = latlon_to_index(-95.0, 200.0)
        assert i == 0
        assert j == N_LON - 1


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

class TestBuildOceanMask:
    def test_only_zero_is_ocean(self):
        raw = np.array([[0, 1, 2, 3, 4]], dtype=np.int8)
        ocean = build_ocean_mask(raw)
        np.testing.assert_array_equal(ocean, [[True, False, False, False, False]])

    def test_dtype(self):
        raw = np.zeros((10, 10), dtype=np.int8)
        ocean = build_ocean_mask(raw)
        assert ocean.dtype == np.bool_


class TestMaritimeChannelEdges:
    def test_every_ordered_override_step_has_an_explicit_edge(self):
        edges = maritime_channel_edges()

        for points in MARITIME_CHANNEL_OVERRIDES.values():
            cells = [latlon_to_index(*point) for point in points]
            for left, right in zip(cells, cells[1:]):
                edge = (left, right) if left <= right else (right, left)
                assert edge in edges

    def test_nonadjacent_channel_definition_fails(self):
        with pytest.raises(ValueError, match="Non-adjacent"):
            maritime_channel_edges({"bad": [(0.0, 0.0), (0.0, 1.0)]})

    def test_named_subgrid_channel_edges_are_frozen(self):
        edges = maritime_channel_edges()
        for named_edges in MARITIME_CHANNEL_EDGE_OVERRIDES.values():
            for left, right in named_edges:
                edge = (left, right) if left <= right else (right, left)
                assert edge in edges

    def test_panama_entrance_edges_are_frozen(self):
        assert MARITIME_CHANNEL_EDGE_OVERRIDES["panama_canal"] == [
            ((5963, 6009), (5962, 6008)),
            ((5938, 6024), (5937, 6025)),
            ((5937, 6025), (5936, 6026)),
        ]


# ---------------------------------------------------------------------------
# Sanity checks on the real mask (require generated .npy)
# ---------------------------------------------------------------------------

class TestSanityLocations:
    """Check obvious locations against the real mask."""

    def _is_ocean(self, mask, lat, lon):
        i, j = latlon_to_index(lat, lon)
        return bool(mask[i, j])

    def test_central_pacific(self, ocean_mask):
        assert self._is_ocean(ocean_mask, 0.05, -160.05)

    def test_north_atlantic(self, ocean_mask):
        assert self._is_ocean(ocean_mask, 40.05, -30.05)

    def test_indian_ocean(self, ocean_mask):
        assert self._is_ocean(ocean_mask, -10.05, 70.05)

    def test_south_china_sea(self, ocean_mask):
        assert self._is_ocean(ocean_mask, 12.05, 113.05)

    def test_mediterranean(self, ocean_mask):
        assert self._is_ocean(ocean_mask, 35.05, 18.05)

    def test_sahara_is_land(self, ocean_mask):
        assert not self._is_ocean(ocean_mask, 23.05, 10.05)

    def test_central_africa_is_land(self, ocean_mask):
        assert not self._is_ocean(ocean_mask, 0.05, 25.05)

    def test_amazon_is_land(self, ocean_mask):
        assert not self._is_ocean(ocean_mask, -3.05, -60.05)

    def test_great_lakes_not_navigable(self, ocean_mask):
        assert not self._is_ocean(ocean_mask, 43.05, -82.05)

    def test_caspian_sea_not_navigable(self, ocean_mask):
        assert not self._is_ocean(ocean_mask, 41.05, 50.05)

    def test_lake_victoria_not_navigable(self, ocean_mask):
        assert not self._is_ocean(ocean_mask, -1.05, 32.95)

    def test_red_sea_is_ocean(self, ocean_mask):
        assert self._is_ocean(ocean_mask, 20.05, 38.05)

    def test_gulf_of_aden_is_ocean(self, ocean_mask):
        assert self._is_ocean(ocean_mask, 12.05, 48.05)


class TestChokepointConnectivity:
    """Verify chokepoint connectivity at 1-arc-minute resolution.

    These are documentation tests — they record the known state of
    the mask, not requirements.  Disconnected passages will need
    explicit maritime-channel overrides added later.
    """

    def _connected_ns(self, mask, lat_s, lat_n, lon_w, lon_e):
        """Check north-south ocean connectivity in a bounding box."""
        from scipy.ndimage import label
        i_s, j_w = latlon_to_index(lat_s, lon_w)
        i_n, j_e = latlon_to_index(lat_n, lon_e)
        patch = mask[i_s:i_n + 1, j_w:j_e + 1]
        labeled, _ = label(patch, structure=np.ones((3, 3)))
        top_labels = set(labeled[-3:, :].flatten()) - {0}
        bot_labels = set(labeled[:3, :].flatten()) - {0}
        return bool(top_labels & bot_labels)

    # Connected passages
    def test_gibraltar_connected(self, ocean_mask):
        assert self._connected_ns(ocean_mask, 35.0, 37.0, -7.0, -4.0)

    def test_bab_el_mandeb_connected(self, ocean_mask):
        assert self._connected_ns(ocean_mask, 11.0, 14.0, 42.0, 45.0)

    def test_danish_straits_connected(self, ocean_mask):
        assert self._connected_ns(ocean_mask, 54.0, 58.0, 9.0, 14.0)

    # Canals opened by maritime channel overrides
    def test_suez_canal_connected(self, ocean_mask):
        assert self._connected_ns(ocean_mask, 29.0, 31.7, 32.0, 33.0)

    def test_panama_canal_connected(self, ocean_mask):
        assert self._connected_ns(ocean_mask, 8.0, 10.0, -80.5, -79.0)

    def test_bosporus_connected(self, ocean_mask):
        assert self._connected_ns(ocean_mask, 40.7, 41.7, 28.5, 29.5)


class TestMaskShape:
    def test_dimensions(self, ocean_mask):
        assert ocean_mask.shape == (10800, 21600)

    def test_dtype(self, ocean_mask):
        assert ocean_mask.dtype == np.bool_

    def test_ocean_fraction_reasonable(self, ocean_mask):
        frac = ocean_mask.sum() / ocean_mask.size
        assert 0.60 < frac < 0.75


# ---------------------------------------------------------------------------
# Frozen identity of the routing inputs
# ---------------------------------------------------------------------------

class TestFrozenRoutingInputs:
    """The mask is not committed, so its identity is guarded instead.

    These tests are the reason a silent mask rebuild cannot reach a result.
    """

    def test_declared_channel_edges_match_frozen_identity(self):
        from sea_path.ocean_mask import (
            FROZEN_CHANNEL_EDGE_COUNT,
            FROZEN_CHANNEL_EDGE_SHA256,
            channel_edge_digest,
            maritime_channel_edges,
        )
        edges = maritime_channel_edges()
        assert len(edges) == FROZEN_CHANNEL_EDGE_COUNT
        assert channel_edge_digest(edges) == FROZEN_CHANNEL_EDGE_SHA256

    def test_real_mask_matches_frozen_identity(self, ocean_mask, ocean_mask_path):
        from pathlib import Path
        from sea_path.ocean_mask import (
            FROZEN_OCEAN_CELL_COUNT,
            FROZEN_OCEAN_MASK_SHA256,
            verify_routing_inputs,
        )
        report = verify_routing_inputs(mask_path=ocean_mask_path, ocean=ocean_mask)
        assert report["verified"] is True
        assert report["mask_sha256"] == FROZEN_OCEAN_MASK_SHA256
        assert report["ocean_cell_count"] == FROZEN_OCEAN_CELL_COUNT

    def test_guard_rejects_a_mask_with_a_different_cell_count(self):
        from sea_path.ocean_mask import (
            OceanMaskIntegrityError,
            verify_routing_inputs,
        )
        wrong = np.ones((4, 4), dtype=np.bool_)
        with pytest.raises(OceanMaskIntegrityError, match="navigable cell count"):
            verify_routing_inputs(ocean=wrong)

    def test_guard_rejects_an_added_channel_edge(self, monkeypatch):
        from sea_path import ocean_mask as om
        extra = dict(om.MARITIME_CHANNEL_EDGE_OVERRIDES)
        extra["unauthorized"] = [((100, 100), (101, 101))]
        monkeypatch.setattr(om, "MARITIME_CHANNEL_EDGE_OVERRIDES", extra)
        with pytest.raises(om.OceanMaskIntegrityError, match="channel edge"):
            om.verify_routing_inputs()

    def test_guard_rejects_a_mask_file_with_a_different_digest(self, tmp_path):
        from sea_path.ocean_mask import (
            OceanMaskIntegrityError,
            verify_routing_inputs,
        )
        decoy = tmp_path / "not_the_frozen_mask.npy"
        np.save(decoy, np.zeros((4, 4), dtype=np.bool_))
        with pytest.raises(OceanMaskIntegrityError, match="SHA-256"):
            verify_routing_inputs(mask_path=decoy)
