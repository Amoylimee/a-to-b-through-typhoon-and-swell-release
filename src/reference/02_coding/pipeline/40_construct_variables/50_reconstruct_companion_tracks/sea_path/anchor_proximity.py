"""Pure annotation helpers for the accepted E032 interval-anchor track.

The functions in this module do not route, schedule anchors, or alter position
coordinates.  They only join already-emitted positions to already-scheduled
anchors and route-leg endpoint metadata.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc


POSITION_IDENTITY_COLUMNS = ("gap_id", "utc", "lat", "lon")
ANNOTATION_COLUMNS = (
    "hours_to_nearest_anchor",
    "nearest_anchor_target_hour_id",
    "leg_bounds_kind",
)
LEG_BOUNDS_KINDS = ("anchor_anchor", "anchor_fix", "fix_fix")
ANCHOR_NODE_TYPE = "gfw_interval_anchor"
FIX_NODE_TYPES = frozenset({"observed_ais_start", "observed_ais_end"})
RESOLVED = "resolved"


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def assert_position_identity_exact(
    original: pd.DataFrame,
    candidate: pd.DataFrame,
) -> None:
    """Assert exact row-order, dtype, timestamp, and floating-point identity."""

    _require_columns(original, set(POSITION_IDENTITY_COLUMNS), "original track")
    _require_columns(candidate, set(POSITION_IDENTITY_COLUMNS), "candidate track")
    pd.testing.assert_frame_equal(
        original.loc[:, POSITION_IDENTITY_COLUMNS],
        candidate.loc[:, POSITION_IDENTITY_COLUMNS],
        check_exact=True,
        check_dtype=True,
        check_index_type=True,
        check_names=True,
        check_like=False,
    )


def position_identity_sha256(frame: pd.DataFrame) -> str:
    """Hash the four protected position fields as a metadata-free Arrow stream."""

    _require_columns(frame, set(POSITION_IDENTITY_COLUMNS), "track")
    table = pa.Table.from_pandas(
        frame.loc[:, POSITION_IDENTITY_COLUMNS], preserve_index=False
    ).replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table.combine_chunks())
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def classify_leg_bounds(route_legs: pd.DataFrame) -> pd.DataFrame:
    """Return one allowed endpoint-kind label for every unique route leg."""

    required = {"gap_id", "leg_index", "start_node_type", "end_node_type"}
    _require_columns(route_legs, required, "route legs")
    legs = route_legs.loc[:, sorted(required)].copy()
    if legs[["gap_id", "leg_index"]].isna().any().any():
        raise ValueError("route legs contain null gap_id or leg_index")
    duplicated = legs.duplicated(["gap_id", "leg_index"], keep=False)
    if duplicated.any():
        raise ValueError(
            "route legs are not unique on (gap_id, leg_index): "
            f"{int(duplicated.sum())} rows"
        )

    allowed_node_types = FIX_NODE_TYPES | {ANCHOR_NODE_TYPE}
    observed_node_types = set(legs["start_node_type"].dropna().astype(str)) | set(
        legs["end_node_type"].dropna().astype(str)
    )
    unexpected = sorted(observed_node_types.difference(allowed_node_types))
    if unexpected:
        raise ValueError(f"route legs contain unexpected endpoint types: {unexpected}")
    if legs[["start_node_type", "end_node_type"]].isna().any().any():
        raise ValueError("route legs contain null endpoint types")

    start_anchor = legs["start_node_type"].eq(ANCHOR_NODE_TYPE)
    end_anchor = legs["end_node_type"].eq(ANCHOR_NODE_TYPE)
    legs["leg_bounds_kind"] = np.select(
        [start_anchor & end_anchor, start_anchor | end_anchor],
        ["anchor_anchor", "anchor_fix"],
        default="fix_fix",
    )
    if not set(legs["leg_bounds_kind"]).issubset(LEG_BOUNDS_KINDS):
        raise AssertionError("derived leg_bounds_kind escaped the frozen vocabulary")
    return legs[["gap_id", "leg_index", "leg_bounds_kind"]]


def _nearest_anchor_annotations(
    track: pd.DataFrame,
    scheduled_anchors: pd.DataFrame,
) -> pd.DataFrame:
    required_anchors = {
        "gap_id",
        "target_hour_id",
        "scheduled_utc",
        "schedule_status",
    }
    _require_columns(track, {"gap_id", "utc"}, "track")
    _require_columns(scheduled_anchors, required_anchors, "scheduled anchors")
    if track[["gap_id", "utc"]].isna().any().any():
        raise ValueError("track contains null gap_id or utc")
    if not isinstance(track["utc"].dtype, pd.DatetimeTZDtype):
        raise TypeError("track utc must be timezone-aware")

    anchors = scheduled_anchors.loc[
        scheduled_anchors["schedule_status"].eq(RESOLVED)
        & scheduled_anchors["scheduled_utc"].notna(),
        ["gap_id", "target_hour_id", "scheduled_utc"],
    ].copy()
    if anchors[["gap_id", "target_hour_id", "scheduled_utc"]].isna().any().any():
        raise ValueError("resolved scheduled anchors contain null required values")
    if not isinstance(anchors["scheduled_utc"].dtype, pd.DatetimeTZDtype):
        raise TypeError("scheduled anchor utc must be timezone-aware")
    duplicate_ids = anchors.duplicated(["gap_id", "target_hour_id"], keep=False)
    duplicate_times = anchors.duplicated(["gap_id", "scheduled_utc"], keep=False)
    if duplicate_ids.any() or duplicate_times.any():
        raise ValueError(
            "resolved scheduled anchors are not unique within gap: "
            f"duplicate ids={int(duplicate_ids.sum())}, "
            f"duplicate times={int(duplicate_times.sum())}"
        )

    left = track.loc[:, ["gap_id", "utc"]].copy()
    left["_position_row_ordinal"] = np.arange(len(left), dtype=np.int64)
    if anchors.empty:
        return pd.DataFrame(
            {
                "_position_row_ordinal": left["_position_row_ordinal"],
                "hours_to_nearest_anchor": pd.Series(
                    pd.array([pd.NA] * len(left), dtype="Float64")
                ),
                "nearest_anchor_target_hour_id": pd.Series(
                    pd.array([pd.NA] * len(left), dtype="string")
                ),
            }
        )

    left = left.sort_values(["utc", "gap_id"], kind="stable")
    right = anchors.sort_values(["scheduled_utc", "gap_id"], kind="stable")
    previous = pd.merge_asof(
        left,
        right.rename(
            columns={
                "scheduled_utc": "_previous_anchor_utc",
                "target_hour_id": "_previous_anchor_id",
            }
        ),
        left_on="utc",
        right_on="_previous_anchor_utc",
        by="gap_id",
        direction="backward",
        allow_exact_matches=True,
    )
    following = pd.merge_asof(
        left,
        right.rename(
            columns={
                "scheduled_utc": "_following_anchor_utc",
                "target_hour_id": "_following_anchor_id",
            }
        ),
        left_on="utc",
        right_on="_following_anchor_utc",
        by="gap_id",
        direction="forward",
        allow_exact_matches=True,
    )

    previous_hours = (
        previous["utc"] - previous["_previous_anchor_utc"]
    ).dt.total_seconds() / 3600.0
    following_hours = (
        following["_following_anchor_utc"] - following["utc"]
    ).dt.total_seconds() / 3600.0
    choose_following = previous_hours.isna() | (
        following_hours.notna() & following_hours.lt(previous_hours)
    )
    # Exact ties select the earlier scheduled anchor.  This makes the traceable
    # target-hour ID deterministic without affecting the distance value.
    nearest_hours = previous_hours.where(~choose_following, following_hours)
    nearest_ids = previous["_previous_anchor_id"].astype("string").where(
        ~choose_following,
        following["_following_anchor_id"].astype("string"),
    )
    annotations = pd.DataFrame(
        {
            "_position_row_ordinal": left["_position_row_ordinal"].to_numpy(),
            "hours_to_nearest_anchor": pd.array(nearest_hours, dtype="Float64"),
            "nearest_anchor_target_hour_id": pd.array(nearest_ids, dtype="string"),
        }
    ).sort_values("_position_row_ordinal", kind="stable")
    annotations.reset_index(drop=True, inplace=True)
    return annotations


def annotate_anchor_proximity(
    track: pd.DataFrame,
    scheduled_anchors: pd.DataFrame,
    route_legs: pd.DataFrame,
) -> pd.DataFrame:
    """Append proximity and leg-bound fields while preserving every input row."""

    required_track = set(POSITION_IDENTITY_COLUMNS) | {"leg_index"}
    _require_columns(track, required_track, "track")
    if any(column in track.columns for column in ANNOTATION_COLUMNS):
        raise ValueError("track already contains one or more annotation columns")
    if track.duplicated(["gap_id", "utc"]).any():
        raise ValueError("track is not unique on (gap_id, utc)")

    nearest = _nearest_anchor_annotations(track, scheduled_anchors)
    leg_kinds = classify_leg_bounds(route_legs)
    position_legs = track.loc[:, ["gap_id", "leg_index"]].copy()
    position_legs["_position_row_ordinal"] = np.arange(len(track), dtype=np.int64)
    position_legs = position_legs.merge(
        leg_kinds,
        on=["gap_id", "leg_index"],
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    ).sort_values("_position_row_ordinal", kind="stable")
    missing_legs = position_legs["_merge"].ne("both")
    if missing_legs.any():
        raise ValueError(
            f"{int(missing_legs.sum())} positions do not map to one route leg"
        )

    annotated = track.copy()
    annotated["hours_to_nearest_anchor"] = pd.array(
        nearest["hours_to_nearest_anchor"], dtype="Float64"
    )
    annotated["nearest_anchor_target_hour_id"] = pd.array(
        nearest["nearest_anchor_target_hour_id"], dtype="string"
    )
    annotated["leg_bounds_kind"] = pd.array(
        position_legs["leg_bounds_kind"], dtype="string"
    )
    if annotated["hours_to_nearest_anchor"].lt(0).fillna(False).any():
        raise AssertionError("nearest-anchor distance cannot be negative")
    if not annotated["hours_to_nearest_anchor"].isna().equals(
        annotated["nearest_anchor_target_hour_id"].isna()
    ):
        raise AssertionError("nearest-anchor hours and target ID nullity disagree")
    if annotated["leg_bounds_kind"].isna().any():
        raise AssertionError("leg_bounds_kind cannot be null")
    if not set(annotated["leg_bounds_kind"]).issubset(LEG_BOUNDS_KINDS):
        raise AssertionError("leg_bounds_kind escaped the frozen vocabulary")
    assert_position_identity_exact(track, annotated)
    return annotated
