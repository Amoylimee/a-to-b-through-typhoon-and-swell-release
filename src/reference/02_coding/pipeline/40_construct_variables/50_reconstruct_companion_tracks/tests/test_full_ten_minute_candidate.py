from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from candidate_full_ten_minute import (  # noqa: E402
    ENDPOINT_LABEL,
    build_full_candidate_grid,
    validate_candidate_frame,
)


def candidate(*, start: str = "2025-01-01T00:00:00Z", end: str = "2025-01-01T05:00:00Z") -> dict[str, str]:
    return {
        "candidate_id": "candidate-1",
        "mmsi": "123456789",
        "vessel_uid_clean": "api-1",
        "layer": "L1",
        "source_shard": "shard_0000",
        "departure_time": start,
        "arrival_time": end,
    }


def raw_points(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "vessel_uid": "api-1",
                "mmsi": "123456789",
                "UTC": utc,
                "UTC_10min": pd.Timestamp(utc).floor("10min"),
                "lat": lat,
                "lon": lon,
                "source_provider": "provider",
                "source_resolution_original": "default_2min",
            }
            for utc, _, lat, lon in rows
        ]
    )


def source_gap(*, start: str, end: str, gap_class: str, gap_id: str = "gap-1") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gap_id": gap_id,
                "candidate_id": "candidate-1",
                "gap_class": gap_class,
                "gap_start": start,
                "gap_end": end,
            }
        ]
    )


def patch_rows(
    times: list[str],
    *,
    gap_id: str,
    label: str,
    anchor_source: str = "source",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gap_id": [gap_id] * len(times),
            "candidate_id": ["candidate-1"] * len(times),
            "mmsi": ["123456789"] * len(times),
            "utc": times,
            "lat": [10.0 + index for index in range(len(times))],
            "lon": [20.0 + index for index in range(len(times))],
            "track_evidence": [label] * len(times),
            "anchor_source": [anchor_source] * len(times),
        }
    )


def test_short_gap_fills_only_unobserved_grid_bins_and_uses_shortest_longitude() -> None:
    points = raw_points(
        [
            ("2025-01-01T00:01:00Z", "", 10.0, 179.0),
            ("2025-01-01T01:01:00Z", "", 20.0, -179.0),
        ]
    )
    frame, summary = build_full_candidate_grid(
        candidate(end="2025-01-01T02:00:00Z"),
        points,
        source_gap(start="2025-01-01T00:01:00Z", end="2025-01-01T01:01:00Z", gap_class="gfw_candidate_4_24h"),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert len(frame) == 11
    assert summary["observed_ais_rows"] == 1
    assert summary["ais_interpolated_le4h_rows"] == 5
    assert summary["unresolved_rows"] == 5
    assert abs(abs(float(frame.loc[frame.utc.eq(pd.Timestamp("2025-01-01T00:30:00Z")), "lon"].iloc[0])) - 180.0) < 0.05
    validate_candidate_frame(frame)


def test_gfw_patch_fills_long_gap_but_never_overwrites_observed_bin() -> None:
    points = raw_points(
        [
            ("2025-01-01T00:01:00Z", "", 1.0, 1.0),
            ("2025-01-01T04:51:00Z", "", 5.0, 5.0),
        ]
    )
    gap = source_gap(
        start="2025-01-01T00:01:00Z",
        end="2025-01-01T04:51:00Z",
        gap_class="gfw_candidate_4_24h",
    )
    patch = patch_rows(
        [
            "2025-01-01T00:10:00Z",
            "2025-01-01T00:20:00Z",
            "2025-01-01T04:50:00Z",
        ],
        gap_id="gap-1",
        label="gfw_supported_interpolation",
    )
    frame, summary = build_full_candidate_grid(candidate(), points, gap, patch, pd.DataFrame())
    assert summary["gfw_supported_interpolation_rows"] == 2
    assert summary["base_skipped_existing_rows"] == 1
    observed = frame.loc[frame.utc.eq(pd.Timestamp("2025-01-01T04:50:00Z"))].iloc[0]
    assert observed.track_evidence == "observed_ais"
    assert observed.observation_time_utc == pd.Timestamp("2025-01-01T04:51:00Z")
    assert frame.loc[frame.utc.eq(pd.Timestamp("2025-01-01T00:10:00Z")), "track_evidence"].iloc[0] == "gfw_supported_interpolation"
    validate_candidate_frame(frame)


def test_endpoint_patch_keeps_candidate_only_label_and_unresolved_rows_are_explicit() -> None:
    points = raw_points(
        [
            ("2025-01-01T00:01:00Z", "", 1.0, 1.0),
            ("2025-01-02T01:01:00Z", "", 2.0, 2.0),
        ]
    )
    gap = source_gap(
        start="2025-01-01T00:01:00Z",
        end="2025-01-02T01:01:00Z",
        gap_class="unresolved_internal_gt24h",
        gap_id="gap-gt24",
    )
    endpoint = patch_rows(
        ["2025-01-01T12:00:00Z"],
        gap_id="gap-gt24",
        label=ENDPOINT_LABEL,
    )
    frame, summary = build_full_candidate_grid(
        candidate(end="2025-01-02T02:00:00Z"),
        points,
        gap,
        pd.DataFrame(),
        endpoint,
    )
    assert summary["endpoint_24_48h_rows"] == 1
    assert (frame.track_evidence == ENDPOINT_LABEL).sum() == 1
    assert (frame.track_evidence == "unresolved").sum() > 0
    validate_candidate_frame(frame)
