from __future__ import annotations

import pandas as pd
import pytest

from sea_path.anchor_proximity import (
    annotate_anchor_proximity,
    assert_position_identity_exact,
    classify_leg_bounds,
    position_identity_sha256,
)


def _track() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gap_id": ["g1", "g1", "g1", "g1", "g2"],
            "utc": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T01:30:00Z",
                    "2024-01-01T03:00:00Z",
                    "2024-01-02T01:00:00Z",
                ],
                utc=True,
            ),
            "lat": [1.0, 1.1, 1.2, 1.3, 2.0],
            "lon": [3.0, 3.1, 3.2, 3.3, 4.0],
            "leg_index": [0, 1, 1, 2, 0],
            "kept_field": [10, 11, 12, 13, 20],
        }
    )


def _anchors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gap_id": ["g1", "g1", "g2"],
            "target_hour_id": ["a0", "a2", "not_scheduled"],
            "scheduled_utc": pd.to_datetime(
                ["2024-01-01T00:30:00Z", "2024-01-01T02:30:00Z", None],
                utc=True,
            ),
            "schedule_status": ["resolved", "resolved", "speed_infeasible"],
        }
    )


def _legs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gap_id": ["g1", "g1", "g1", "g2"],
            "leg_index": [0, 1, 2, 0],
            "start_node_type": [
                "observed_ais_start",
                "gfw_interval_anchor",
                "gfw_interval_anchor",
                "observed_ais_start",
            ],
            "end_node_type": [
                "gfw_interval_anchor",
                "gfw_interval_anchor",
                "observed_ais_end",
                "observed_ais_end",
            ],
        }
    )


def test_annotation_uses_same_gap_nearest_time_and_null_for_no_anchor() -> None:
    original = _track()
    annotated = annotate_anchor_proximity(original, _anchors(), _legs())

    assert annotated["hours_to_nearest_anchor"].tolist()[:4] == [0.5, 0.5, 1.0, 0.5]
    assert annotated["nearest_anchor_target_hour_id"].tolist()[:4] == [
        "a0",
        "a0",
        "a0",  # exact tie is resolved to the earlier scheduled anchor
        "a2",
    ]
    assert pd.isna(annotated.loc[4, "hours_to_nearest_anchor"])
    assert pd.isna(annotated.loc[4, "nearest_anchor_target_hour_id"])
    assert annotated["leg_bounds_kind"].tolist() == [
        "anchor_fix",
        "anchor_anchor",
        "anchor_anchor",
        "anchor_fix",
        "fix_fix",
    ]
    pd.testing.assert_frame_equal(original, annotated[original.columns], check_exact=True)
    assert_position_identity_exact(original, annotated)
    assert position_identity_sha256(original) == position_identity_sha256(annotated)


def test_classify_leg_bounds_rejects_unknown_endpoint_type() -> None:
    legs = _legs()
    legs.loc[0, "start_node_type"] = "unknown"
    with pytest.raises(ValueError, match="unexpected endpoint types"):
        classify_leg_bounds(legs)


def test_annotation_rejects_missing_leg_mapping() -> None:
    with pytest.raises(ValueError, match="positions do not map"):
        annotate_anchor_proximity(_track(), _anchors(), _legs().iloc[:-1])


def test_annotation_rejects_duplicate_scheduled_time_within_gap() -> None:
    anchors = pd.concat([_anchors(), _anchors().iloc[[0]]], ignore_index=True)
    anchors.loc[len(anchors) - 1, "target_hour_id"] = "a0_duplicate"
    with pytest.raises(ValueError, match="not unique within gap"):
        annotate_anchor_proximity(_track(), anchors, _legs())
