from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "20_quantify_uncertainty/10_missing_positions/10_probe_reachability.py"
)
SPEC = importlib.util.spec_from_file_location("e002_missing_position_probe", SCRIPT)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_primary_and_per_voyage_sensitivity_caps_are_separate(monkeypatch) -> None:
    captured = []

    def fake_reachable_cells(lat_a, lon_a, lat_b, lon_b, radius_a, radius_b):
        captured.append((radius_a, radius_b))
        return np.array([lat_a]), np.array([lon_a]), False

    monkeypatch.setattr(PROBE.legacy, "reachable_cells", fake_reachable_cells)
    row = pd.Series({
        "position_time_utc": pd.Timestamp("2025-01-01T06:00:00Z"),
        "left_time_ns": pd.Timestamp("2025-01-01T00:00:00Z").value,
        "right_time_ns": pd.Timestamp("2025-01-01T12:00:00Z").value,
        "left_lat": 0.0,
        "left_lon": 0.0,
        "right_lat": 0.0,
        "right_lon": 1.0,
        "endpoint_case": "two_sided",
    })

    PROBE._reachable(row, 25.0)
    PROBE._reachable(row, 12.0)

    assert len(captured) == 2
    np.testing.assert_allclose(captured[0], [25 * 1.852 * 6, 25 * 1.852 * 6])
    np.testing.assert_allclose(captured[1], [12 * 1.852 * 6, 12 * 1.852 * 6])


def test_endpoint_pair_infeasible_at_speed_cap_is_flagged_not_widened(monkeypatch) -> None:
    monkeypatch.setattr(
        PROBE.legacy,
        "great_circle_km",
        lambda lat_a, lon_a, lat_b, lon_b: 10_000.0,
    )
    row = pd.Series({
        "position_time_utc": pd.Timestamp("2025-01-01T06:00:00Z"),
        "left_time_ns": pd.Timestamp("2025-01-01T00:00:00Z").value,
        "right_time_ns": pd.Timestamp("2025-01-01T12:00:00Z").value,
        "left_lat": 0.0,
        "left_lon": 0.0,
        "right_lat": 0.0,
        "right_lon": 1.0,
        "endpoint_case": "two_sided",
    })

    lat, lon, capped, status = PROBE._reachable(row, 25.0)

    assert len(lat) == len(lon) == 0
    assert not capped
    assert status == "speed_infeasible"


def test_bounded_segments_use_nearest_positioned_rows_and_keep_no_position_unbounded() -> None:
    times = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T01:00:00Z",
            "2025-01-01T02:00:00Z",
            "2025-01-01T01:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame({
        "candidate_id": ["a", "a", "a", "b"],
        "hour_anchor_utc": times.floor("h"),
        "position_time_utc": times,
        "segment_weight_hours": [0.0, 0.5, 0.0, 0.75],
        "track_evidence": ["observed_ais", "unresolved", "observed_ais", "unresolved"],
        "lat": [0.0, np.nan, 0.0, np.nan],
        "lon": [0.0, np.nan, 1.0, np.nan],
    })
    summary = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "origin_city": ["AAA:A", "BBB:B"],
        "dest_city": ["CCC:C", "DDD:D"],
        "shiptype_group": ["container", "bulk_carrier"],
    }).set_index("candidate_id", drop=False)

    events, counts, hours, unbounded_n, unbounded_hours = PROBE._enumerate_bounded_segments(
        frame, summary, "shard_0000"
    )

    assert events.candidate_id.tolist() == ["a"]
    assert events.endpoint_case.tolist() == ["two_sided"]
    assert events.loc[0, "left_time_ns"] == times[0].value
    assert events.loc[0, "right_time_ns"] == times[2].value
    assert counts == {"2025-01": 1}
    assert hours == {"2025-01": 0.5}
    assert unbounded_n == 1
    assert unbounded_hours == 0.75


def test_only_retained_sample_rows_receive_endpoint_resolution() -> None:
    times = pd.to_datetime(
        ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z"],
        utc=True,
    )
    frame = pd.DataFrame({
        "candidate_id": ["a", "a", "a"],
        "position_time_utc": times,
        "track_evidence": ["observed_ais", "unresolved", "observed_ais"],
        "lat": [0.0, np.nan, 0.0],
        "lon": [0.0, np.nan, 1.0],
    })
    selected = pd.DataFrame({
        "candidate_id": ["a"],
        "position_time_utc": [times[1]],
    })

    resolved = PROBE._resolve_sample_endpoints(selected, frame, "shard_0000")

    assert resolved.loc[0, "left_time_ns"] == times[0].value
    assert resolved.loc[0, "right_time_ns"] == times[2].value
    assert resolved.loc[0, "endpoint_case"] == "two_sided"
    assert resolved.loc[0, "voyage_speed_cap_sensitivity_kn"] == 25.0
    assert bool(resolved.loc[0, "endpoints_resolved"])
    assert resolved.loc[0, "source_shard"] == "shard_0000"
