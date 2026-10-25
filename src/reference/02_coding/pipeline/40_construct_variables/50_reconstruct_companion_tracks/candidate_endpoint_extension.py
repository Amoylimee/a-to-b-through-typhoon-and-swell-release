"""The separate 24--48 hour AIS-endpoint extension layer.

The accepted E001 reconstruction currently stops at 24 hours and uses GFW
anchors inside that band.  This module adds the later climate-project-style
patch as a candidate layer only:

* only internal gaps whose two AIS boundary fixes are present are eligible;
* the duration rule is strictly ``> 24`` and ``<= 48`` hours;
* the two boundary fixes are the only endpoints used, with no GFW anchor;
* the existing frozen sea-path, 25-knot ceiling, five-second timeout and
  ten-minute grid are reused;
* a patch may fill an unresolved base row, but may never replace a valid row.

It deliberately does not change the protected 4--24 hour runtime contract or
the accepted product.  The experiment runner writes this layer separately so
that it can be audited and overlaid only after review.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from sea_path.routing import (
    DEFAULT_MAX_SPEED_KNOTS,
    DEFAULT_ROUTE_TIMEOUT_S,
    ROUTE_RESOLVED,
    SeaPathRouter,
    interpolate_along_path,
)
from track_core.e001_cached_ten_minute import strict_grid_seconds


MIN_ENDPOINT_EXTENSION_HOURS = 24.0
MAX_ENDPOINT_EXTENSION_HOURS = 48.0
GRID_SECONDS = 600
MAX_SPEED_KNOTS = DEFAULT_MAX_SPEED_KNOTS
ROUTE_TIMEOUT_SECONDS = DEFAULT_ROUTE_TIMEOUT_S

INTERNAL_LONG_GAP_CLASS = "unresolved_internal_gt24h"
TRACK_EVIDENCE = "companion_endpoint_24_48h"
ANCHOR_SOURCE = "identity_validated_ais_endpoints"
KEY_COLUMNS = ("candidate_id", "utc")


def _row_value(row: Any, name: str) -> Any:
    """Read a field from either a named tuple, a Series, or a mapping."""
    try:
        return getattr(row, name)
    except AttributeError:
        try:
            return row[name]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"gap is missing required field: {name}") from exc


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _epoch_seconds(value: Any) -> float:
    timestamp = _utc_timestamp(value)
    result = float(timestamp.timestamp())
    if not math.isfinite(result):
        raise ValueError(f"gap timestamp is not finite: {value!r}")
    return result


def gap_duration_hours(gap: Any) -> float:
    """Return the duration from the two source timestamps, not a rounded label."""
    start = _epoch_seconds(_row_value(gap, "gap_start"))
    end = _epoch_seconds(_row_value(gap, "gap_end"))
    duration = (end - start) / 3600.0
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"gap boundaries must be finite and ordered: {start}, {end}")
    return duration


def select_endpoint_extension_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the internal ``(24, 48]`` source-gap band."""
    required = {
        "gap_id", "gap_class", "gap_start", "gap_end", "left_lat", "left_lon",
        "right_lat", "right_lon",
    }
    missing = sorted(required.difference(gaps.columns))
    if missing:
        raise ValueError(f"gap table is missing required columns: {missing}")
    if gaps.gap_id.duplicated().any():
        raise ValueError("source gap identity is not unique")

    start = pd.to_datetime(gaps.gap_start, utc=True, errors="coerce")
    end = pd.to_datetime(gaps.gap_end, utc=True, errors="coerce")
    duration = (end - start).dt.total_seconds() / 3600.0
    finite_endpoints = np.isfinite(
        gaps[["left_lat", "left_lon", "right_lat", "right_lon"]]
        .to_numpy(dtype=float)
    ).all(axis=1)
    keep = (
        gaps.gap_class.eq(INTERNAL_LONG_GAP_CLASS)
        & start.notna()
        & end.notna()
        & duration.gt(MIN_ENDPOINT_EXTENSION_HOURS)
        & duration.le(MAX_ENDPOINT_EXTENSION_HOURS)
        & finite_endpoints
    )
    result = gaps.loc[keep].copy()
    result["endpoint_gap_hours"] = duration.loc[keep].astype(float)
    return result


def _validate_gap(gap: Any) -> tuple[float, float, float]:
    start = _epoch_seconds(_row_value(gap, "gap_start"))
    end = _epoch_seconds(_row_value(gap, "gap_end"))
    duration = (end - start) / 3600.0
    if not MIN_ENDPOINT_EXTENSION_HOURS < duration <= MAX_ENDPOINT_EXTENSION_HOURS:
        raise ValueError(
            "endpoint extension only accepts gaps strictly over 24 and up to "
            f"48 hours: {duration:.9f}"
        )

    declared = getattr(gap, "gap_hours", None)
    if declared is None and isinstance(gap, dict):
        declared = gap.get("gap_hours")
    if declared is not None and math.isfinite(float(declared)):
        if not math.isclose(float(declared), duration, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"declared gap_hours differs from source timestamps: {declared} vs {duration}"
            )

    coordinates = [
        float(_row_value(gap, "left_lat")),
        float(_row_value(gap, "left_lon")),
        float(_row_value(gap, "right_lat")),
        float(_row_value(gap, "right_lon")),
    ]
    if not np.isfinite(coordinates).all():
        raise ValueError("endpoint extension requires two finite AIS boundary coordinates")
    return start, end, duration


def _empty_track(gap: Any, targets: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gap_id": str(_row_value(gap, "gap_id")),
            "candidate_id": str(_row_value(gap, "candidate_id")),
            "mmsi": str(_row_value(gap, "mmsi")),
            "utc": pd.to_datetime(targets, unit="s", utc=True),
            "lat": np.full(len(targets), np.nan),
            "lon": np.full(len(targets), np.nan),
            "track_evidence": "unresolved",
            "leg_index": pd.array([None] * len(targets), dtype="Int16"),
            "anchor_source": pd.Series([None] * len(targets), dtype="string"),
        }
    )


def _route_leg(gap: Any, start: float, end: float, status: str, detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "gap_id": str(_row_value(gap, "gap_id")),
        "candidate_id": str(_row_value(gap, "candidate_id")),
        "mmsi": str(_row_value(gap, "mmsi")),
        "leg_index": 0,
        "leg_status": status,
        "schedule_status": "resolved" if status == ROUTE_RESOLVED else status,
        "scheduled_start_epoch_s": start,
        "scheduled_end_epoch_s": end,
        "implied_speed_knots": detail.get("implied_speed_knots"),
        "route_strategy": detail.get("strategy"),
        "route_elapsed_s": detail.get("elapsed_s"),
    }


def reconstruct_gap(
    router: SeaPathRouter,
    gap: Any,
    *,
    timeout_s: float = ROUTE_TIMEOUT_SECONDS,
    max_speed_knots: float = MAX_SPEED_KNOTS,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Reconstruct one eligible gap and return track, status, and one route leg."""
    start, end, duration = _validate_gap(gap)
    targets = strict_grid_seconds(
        _utc_timestamp(_row_value(gap, "gap_start")),
        _utc_timestamp(_row_value(gap, "gap_end")),
    )
    track = _empty_track(gap, targets)

    resolution = router.resolve_path(
        float(_row_value(gap, "left_lat")),
        float(_row_value(gap, "left_lon")),
        float(_row_value(gap, "right_lat")),
        float(_row_value(gap, "right_lon")),
        snap=False,
        timeout_s=timeout_s,
        duration_hours=duration,
        max_speed_knots=max_speed_knots,
    )
    if resolution.status == ROUTE_RESOLVED:
        if resolution.path is None:
            raise AssertionError("resolved route has no path")
        positions = interpolate_along_path(
            resolution.path,
            start,
            end,
            targets,
            max_speed_kmh=max_speed_knots * 1.852,
        )
        track.loc[:, ["lat", "lon"]] = positions
        track["track_evidence"] = TRACK_EVIDENCE
        track["leg_index"] = 0
        track["anchor_source"] = ANCHOR_SOURCE

    resolved = int(track.track_evidence.ne("unresolved").sum())
    status = {
        "gap_id": str(_row_value(gap, "gap_id")),
        "candidate_id": str(_row_value(gap, "candidate_id")),
        "mmsi": str(_row_value(gap, "mmsi")),
        "layer": str(getattr(gap, "layer", "endpoint_24_48h")),
        "gap_hours": duration,
        "gap_outcome": resolution.status,
        "ten_minute_grid_rows": len(track),
        "resolved_grid_rows": resolved,
        "unresolved_grid_rows": len(track) - resolved,
        "reconstructed_hours": duration if resolved else 0.0,
        "unresolved_hours": 0.0 if resolved else duration,
        "interpolation_policy": "identity_validated_ais_endpoints_only",
    }
    leg = pd.DataFrame([_route_leg(gap, start, end, resolution.status, resolution.detail)])
    return track, status, leg


def resolved_patch_rows(track: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that can actually fill an unresolved base row."""
    required = {"track_evidence", "lat", "lon"}
    missing = sorted(required.difference(track.columns))
    if missing:
        raise ValueError(f"track is missing patch columns: {missing}")
    result = track.loc[track.track_evidence.ne("unresolved")].copy()
    if result[["lat", "lon"]].isna().any().any():
        raise ValueError("resolved patch row has a missing position")
    return result


def overlay_endpoint_extension(frame: pd.DataFrame, patch: pd.DataFrame) -> pd.DataFrame:
    """Fill unresolved base rows without changing keys, weights, or valid rows."""
    for name, value in (("base", frame), ("patch", patch)):
        missing = sorted(set(KEY_COLUMNS).difference(value.columns))
        if missing:
            raise ValueError(f"{name} is missing patch key columns: {missing}")
        if value.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"duplicate {name} candidate/time key")
    if patch.empty:
        return frame.copy()
    if "track_evidence" not in frame or "track_evidence" not in patch:
        raise ValueError("base and patch must carry track_evidence")
    if patch.track_evidence.eq("unresolved").any():
        raise ValueError("patch must contain resolved rows only")
    if patch[["lat", "lon"]].isna().any().any():
        raise ValueError("patch contains missing positions")

    base_index = pd.MultiIndex.from_frame(frame[list(KEY_COLUMNS)])
    patch_index = pd.MultiIndex.from_frame(patch[list(KEY_COLUMNS)])
    positions = base_index.get_indexer(patch_index)
    if (positions < 0).any():
        raise ValueError("patch key absent from base")

    old = frame.iloc[positions]
    if old.track_evidence.ne("unresolved").any():
        raise ValueError("refusing to overwrite a valid base position")
    if old[["lat", "lon"]].notna().any().any():
        raise ValueError("unresolved base row already has coordinates")

    result = frame.copy()
    for column in (
        "mmsi", "utc", "segment_start_utc", "segment_end_utc",
        "segment_weight_ns", "segment_weight_hours",
    ):
        if column in frame.columns and column in patch.columns:
            left = old[column].reset_index(drop=True)
            right = patch.iloc[:, patch.columns.get_loc(column)].reset_index(drop=True)
            if not left.equals(right):
                raise ValueError(f"patch changes protected column: {column}")
    for column in patch.columns:
        if column in result.columns:
            result.iloc[positions, result.columns.get_loc(column)] = patch[column].to_numpy()
    return result
