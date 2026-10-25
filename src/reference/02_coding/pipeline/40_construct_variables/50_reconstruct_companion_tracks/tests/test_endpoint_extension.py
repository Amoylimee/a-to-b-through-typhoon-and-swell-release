from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sea_path.routing import RouteResolution, SeaPath
from candidate_endpoint_extension import (
    ANCHOR_SOURCE,
    TRACK_EVIDENCE,
    overlay_endpoint_extension,
    reconstruct_gap,
    select_endpoint_extension_gaps,
)


def gap(gap_id="g1", hours=30.0, gap_class="unresolved_internal_gt24h"):
    start = pd.Timestamp("2025-01-01T00:03:00Z")
    return {
        "gap_id": gap_id,
        "candidate_id": "voyage-1",
        "mmsi": "123",
        "gap_class": gap_class,
        "gap_start": start,
        "gap_end": start + pd.Timedelta(hours=hours),
        "gap_hours": hours,
        "left_lat": 20.0,
        "left_lon": 38.0,
        "right_lat": 20.3,
        "right_lon": 38.2,
        "layer": "other",
    }


@pytest.mark.parametrize("hours,keep", [(24.0, False), (24.0001, True), (48.0, True), (48.0001, False)])
def test_endpoint_band_has_exact_open_closed_boundaries(hours, keep):
    frame = pd.DataFrame([gap(hours=hours)])
    selected = select_endpoint_extension_gaps(frame)
    assert len(selected) == int(keep)


def test_endpoint_or_no_ais_gap_is_not_selected():
    selected = select_endpoint_extension_gaps(
        pd.DataFrame([gap(gap_class="endpoint_or_no_ais_unknown")])
    )
    assert selected.empty


class FakeRouter:
    def __init__(self, status="resolved"):
        self.status = status

    def resolve_path(self, *args, **kwargs):
        if self.status != "resolved":
            return RouteResolution(None, self.status, {})
        path = SeaPath(
            waypoints=np.array([[20.0, 38.0], [20.3, 38.2]], dtype=float),
            cumulative_km=np.array([0.0, 10.0]),
            total_km=10.0,
        )
        return RouteResolution(
            path,
            "resolved",
            {"implied_speed_knots": 0.2, "strategy": "test", "elapsed_s": 0.0},
        )


def test_reconstruction_uses_only_endpoint_policy_and_ten_minute_grid():
    track, status, legs = reconstruct_gap(FakeRouter(), gap())
    assert len(track) > 0
    assert track.utc.eq(track.utc.dt.floor("10min")).all()
    assert track.track_evidence.eq(TRACK_EVIDENCE).all()
    assert track.anchor_source.eq(ANCHOR_SOURCE).all()
    assert track.lat.notna().all() and track.lon.notna().all()
    assert status["gap_outcome"] == "resolved"
    assert status["unresolved_hours"] == 0.0
    assert legs.iloc[0].leg_status == "resolved"
    assert legs.iloc[0].schedule_status == "resolved"


def test_failed_route_keeps_the_complete_grid_unresolved():
    track, status, legs = reconstruct_gap(FakeRouter("no_path"), gap())
    assert len(track) > 0
    assert track.track_evidence.eq("unresolved").all()
    assert track.lat.isna().all() and track.lon.isna().all()
    assert status["resolved_grid_rows"] == 0
    assert status["unresolved_hours"] == pytest.approx(30.0)
    assert legs.iloc[0].schedule_status == "no_path"


def test_overlay_fills_unknown_only_and_preserves_existing_valid_rows():
    utc = pd.to_datetime(
        ["2025-01-01T00:10:00Z", "2025-01-01T00:20:00Z"], utc=True
    )
    base = pd.DataFrame(
        {
            "candidate_id": "voyage-1",
            "utc": utc,
            "mmsi": "123",
            "lat": [np.nan, np.nan],
            "lon": [np.nan, np.nan],
            "track_evidence": "unresolved",
        }
    )
    patch = base.iloc[[1]].copy()
    patch.loc[:, "lat"] = 20.1
    patch.loc[:, "lon"] = 38.1
    patch.loc[:, "track_evidence"] = TRACK_EVIDENCE
    out = overlay_endpoint_extension(base, patch)
    assert pd.isna(out.loc[0, "lat"])
    assert out.loc[1, "lat"] == pytest.approx(20.1)
    assert out.loc[1, "track_evidence"] == TRACK_EVIDENCE

    valid = base.copy()
    valid.loc[1, "track_evidence"] = "observed_ais"
    valid.loc[1, "lat"] = 20.0
    with pytest.raises(ValueError, match="overwrite"):
        overlay_endpoint_extension(valid, patch)
