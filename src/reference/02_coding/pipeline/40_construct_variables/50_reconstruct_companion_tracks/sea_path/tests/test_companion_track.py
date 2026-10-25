"""Tests for companion-track construction over 4-24 hour source gaps."""

from __future__ import annotations

import numpy as np
import pytest

from sea_path.companion_track import (
    ANCHOR_SOURCE_EMPTY,
    GAP_LONG_UNRESOLVED,
    GAP_RECONSTRUCTED,
    GAP_SHORT,
    Anchor,
    classify_gap,
    hourly_utc_grid,
    reconstruct_gap,
)
from sea_path.routing import (
    ROUTE_RESOLVED,
    ROUTE_SEARCH_TIMEOUT,
    ROUTE_SPEED_INFEASIBLE,
)

HOUR = 3600.0


class TestGapClassification:
    """Duration decides treatment before any routing happens."""

    @pytest.mark.parametrize("hours,expected", [
        (0.5, GAP_SHORT), (4.0, GAP_SHORT),
        (4.01, GAP_RECONSTRUCTED), (12.0, GAP_RECONSTRUCTED), (24.0, GAP_RECONSTRUCTED),
        (24.01, GAP_LONG_UNRESOLVED), (100.0, GAP_LONG_UNRESOLVED),
    ])
    def test_boundaries_follow_the_contract(self, hours, expected):
        assert classify_gap(hours) == expected

    def test_rejects_a_nonpositive_gap(self):
        with pytest.raises(ValueError):
            classify_gap(0.0)


class TestHourlyGrid:
    def test_excludes_the_observed_bounding_fixes(self):
        """A reconstructed point must never overwrite an observed one."""
        grid = hourly_utc_grid(10 * HOUR, 14 * HOUR)
        assert list(grid / HOUR) == [11.0, 12.0, 13.0]

    def test_offset_gap_lands_on_whole_hours(self):
        grid = hourly_utc_grid(10 * HOUR + 900, 13 * HOUR + 900)
        assert list(grid / HOUR) == [11.0, 12.0, 13.0]

    def test_gap_inside_one_hour_yields_nothing(self):
        assert len(hourly_utc_grid(10 * HOUR + 60, 10 * HOUR + 600)) == 0


class TestReconstruction:
    def test_short_gap_is_left_to_the_accepted_treatment(self, router):
        out = reconstruct_gap(router, 0.0, 20.0, 38.0, 3 * HOUR, 20.3, 38.1)
        assert out.gap_class == GAP_SHORT
        assert len(out.positions) == 0
        assert out.unresolved_hours == 0.0

    def test_long_gap_is_never_reconstructed_and_keeps_its_hours(self, router):
        out = reconstruct_gap(router, 0.0, 20.0, 38.0, 30 * HOUR, 22.0, 38.5)
        assert out.gap_class == GAP_LONG_UNRESOLVED
        assert len(out.positions) == 0
        assert out.unresolved_hours == pytest.approx(30.0)

    def test_reconstructs_an_ordinary_gap_on_whole_hours(self, router, ocean_mask):
        from sea_path.ocean_mask import latlon_to_index
        out = reconstruct_gap(router, 0.0, 20.0, 38.0, 12 * HOUR, 21.2, 37.9)
        assert out.gap_class == GAP_RECONSTRUCTED
        assert out.leg_statuses == (ROUTE_RESOLVED,)
        assert len(out.positions) == 11
        assert np.allclose(out.times / HOUR, np.arange(1, 12))
        for lat, lon in out.positions:
            i, j = latlon_to_index(float(lat), float(lon))
            assert ocean_mask[i, j], f"reconstructed onto land at ({lat}, {lon})"
        assert out.reconstructed_hours == pytest.approx(12.0)
        assert out.unresolved_hours == 0.0

    def test_speed_infeasible_gap_emits_nothing(self, router):
        """A route the vessel could not sail must leave no positions behind."""
        out = reconstruct_gap(router, 0.0, 20.0, 38.0, 5 * HOUR, 27.0, 34.5)
        assert out.leg_statuses == (ROUTE_SPEED_INFEASIBLE,)
        assert len(out.positions) == 0
        assert out.reconstructed_hours == 0.0
        assert out.unresolved_hours == pytest.approx(5.0)

    def test_search_timeout_gap_emits_nothing(self, router):
        out = reconstruct_gap(
            router, 0.0, 12.0, 45.0, 12 * HOUR, 15.0, 42.0, timeout_s=0.002,
        )
        assert out.leg_statuses == (ROUTE_SEARCH_TIMEOUT,)
        assert len(out.positions) == 0
        assert out.unresolved_hours == pytest.approx(12.0)

    def test_anchors_split_the_gap_into_legs(self, router):
        """The anchored path is the same construction with more nodes."""
        anchors = [Anchor(epoch_s=6 * HOUR, lat=20.6, lon=37.95, source="test")]
        out = reconstruct_gap(
            router, 0.0, 20.0, 38.0, 12 * HOUR, 21.2, 37.9, anchors=anchors,
        )
        assert len(out.leg_statuses) == 2
        assert set(out.leg_statuses) == {ROUTE_RESOLVED}
        assert set(np.unique(out.leg_index)) == {0, 1}
        assert len(out.positions) == 11

    def test_a_position_lands_on_the_anchor_it_passes_through(self, router):
        anchors = [Anchor(epoch_s=6 * HOUR, lat=20.6, lon=37.95, source="test")]
        out = reconstruct_gap(
            router, 0.0, 20.0, 38.0, 12 * HOUR, 21.2, 37.9, anchors=anchors,
        )
        # the leg boundary is a whole hour, so no emitted point sits on it;
        # the points either side must bracket the anchor latitude
        before = out.positions[out.times < 6 * HOUR][-1]
        after = out.positions[out.times > 6 * HOUR][0]
        assert before[0] <= 20.6 <= after[0]

    def test_anchor_outside_the_gap_is_rejected(self, router):
        with pytest.raises(ValueError, match="outside the gap"):
            reconstruct_gap(
                router, 0.0, 20.0, 38.0, 12 * HOUR, 21.2, 37.9,
                anchors=[Anchor(epoch_s=20 * HOUR, lat=21.0, lon=38.0, source="x")],
            )

    def test_empty_anchor_source_is_named(self):
        """The unanchored build must be self-identifying in its output."""
        assert ANCHOR_SOURCE_EMPTY == "none_gfw_phase_one_incomplete"


class TestAnchorHourProvenance:
    """An anchor lands on the hourly grid by design, and must survive it.

    GFW Presence matches on the same UTC hour, so an anchor's timestamp normally
    coincides with a grid point.  Legs are open at both ends, so without explicit
    handling the shared hour falls out of both neighbours -- discarding exactly
    the hours that carry provider evidence.
    """

    def test_the_shared_hour_is_emitted_as_the_anchor_itself(self, router):
        from sea_path.companion_track import POSITION_ANCHOR, POSITION_INTERPOLATED
        anchors = [Anchor(epoch_s=6 * HOUR, lat=20.6, lon=37.95, source="test")]
        out = reconstruct_gap(
            router, 0.0, 20.0, 38.0, 12 * HOUR, 21.2, 37.9, anchors=anchors,
        )
        assert list((out.times / HOUR).astype(int)) == list(range(1, 12))
        hit = int(np.flatnonzero(out.times == 6 * HOUR)[0])
        assert out.position_source[hit] == POSITION_ANCHOR
        np.testing.assert_allclose(out.positions[hit], [20.6, 37.95])
        assert set(out.position_source) == {POSITION_ANCHOR, POSITION_INTERPOLATED}

    def test_emitted_times_are_strictly_increasing(self, router):
        anchors = [Anchor(epoch_s=6 * HOUR, lat=20.6, lon=37.95, source="test")]
        out = reconstruct_gap(
            router, 0.0, 20.0, 38.0, 12 * HOUR, 21.2, 37.9, anchors=anchors,
        )
        assert (np.diff(out.times) > 0).all()

    def test_an_unanchored_gap_is_entirely_interpolated(self, router):
        from sea_path.companion_track import POSITION_INTERPOLATED
        out = reconstruct_gap(router, 0.0, 20.0, 38.0, 12 * HOUR, 21.2, 37.9)
        assert set(out.position_source) == {POSITION_INTERPOLATED}

    def test_an_anchor_off_the_hour_adds_no_extra_position(self, router):
        """Only a coinciding hour is special; an off-grid anchor changes nothing."""
        from sea_path.companion_track import POSITION_INTERPOLATED
        anchors = [Anchor(epoch_s=6 * HOUR + 900, lat=20.6, lon=37.95, source="t")]
        out = reconstruct_gap(
            router, 0.0, 20.0, 38.0, 12 * HOUR, 21.2, 37.9, anchors=anchors,
        )
        assert list((out.times / HOUR).astype(int)) == list(range(1, 12))
        assert set(out.position_source) == {POSITION_INTERPOLATED}
