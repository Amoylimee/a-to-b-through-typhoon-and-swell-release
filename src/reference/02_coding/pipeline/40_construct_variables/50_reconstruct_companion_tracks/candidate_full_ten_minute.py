"""Assemble a complete candidate ten-minute voyage grid.

This module is deliberately separate from the existing reconstruction runtime.
It combines the frozen cleaned AIS view with the two already completed
candidate supplement layers, while keeping observed AIS rows authoritative.
The result is still a candidate artifact: it does not alter an accepted
product or the protected runtime contract.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


GRID_SECONDS = 600
GRID_NS = GRID_SECONDS * 1_000_000_000
SHORT_GAP_MAX_NS = 4 * 3_600_000_000_000

OBSERVED_LABEL = "observed_ais"
SHORT_LABEL = "ais_interpolated_le4h"
GFW_ANCHOR_LABEL = "gfw_anchor"
GFW_INTERPOLATION_LABEL = "gfw_supported_interpolation"
UNRESOLVED_LABEL = "unresolved"
ENDPOINT_LABEL = "companion_endpoint_24_48h"

ALLOWED_EVIDENCE = frozenset(
    {
        OBSERVED_LABEL,
        SHORT_LABEL,
        GFW_ANCHOR_LABEL,
        GFW_INTERPOLATION_LABEL,
        ENDPOINT_LABEL,
        UNRESOLVED_LABEL,
    }
)


def _value(row: Any, name: str) -> Any:
    try:
        return getattr(row, name)
    except AttributeError:
        if isinstance(row, Mapping):
            return row[name]
        return row[name]


def _utc_timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        return result.tz_localize("UTC")
    return result.tz_convert("UTC")


def _timestamp_ns(value: Any) -> int:
    result = _utc_timestamp(value).value
    if not np.isfinite(result):
        raise ValueError(f"non-finite timestamp: {value!r}")
    return int(result)


def strict_grid_ns(start: Any, end: Any) -> np.ndarray:
    """Return exact UTC ten-minute instants strictly inside ``[start, end]``."""
    left = _timestamp_ns(start)
    right = _timestamp_ns(end)
    if right <= left:
        raise ValueError("voyage window must be positive")
    first = left // GRID_NS + 1
    last = (right + GRID_NS - 1) // GRID_NS - 1
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64) * GRID_NS


def source_gap_id(candidate_id: str, start_ns: int, end_ns: int, gap_class: str) -> str:
    """Create a stable ID for a short AIS-bracketed interval."""
    payload = f"{candidate_id}|{int(start_ns)}|{int(end_ns)}|{gap_class}".encode()
    return hashlib.sha256(payload).hexdigest()


def _short_gap_ids(candidate_id: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cache: dict[tuple[int, int], str] = {}
    values: list[str] = []
    for start, end in zip(left.tolist(), right.tolist()):
        key = (int(start), int(end))
        if key not in cache:
            cache[key] = source_gap_id(
                candidate_id, key[0], key[1], "ais_supported_le4h"
            )
        values.append(cache[key])
    return np.asarray(values, dtype=object)


def _optional_raw_column(raw: pd.DataFrame, name: str, length: int) -> np.ndarray:
    if name not in raw.columns:
        return np.full(length, None, dtype=object)
    return raw[name].to_numpy(dtype=object)


def _patch_position_source(label: str) -> str:
    if label == GFW_ANCHOR_LABEL:
        return "gfw_anchor"
    if label == GFW_INTERPOLATION_LABEL:
        return "gfw_sea_path_interpolation"
    if label == ENDPOINT_LABEL:
        return "ais_endpoint_sea_path_24_48h"
    raise ValueError(f"unexpected supplement evidence label: {label}")


def _validate_patch_columns(patch: pd.DataFrame, name: str) -> None:
    required = {
        "candidate_id",
        "gap_id",
        "utc",
        "lat",
        "lon",
        "track_evidence",
    }
    missing = sorted(required.difference(patch.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if patch.duplicated(["candidate_id", "utc"]).any():
        raise ValueError(f"{name} contains duplicate candidate/time keys")


def _apply_patch(
    *,
    target_ns: np.ndarray,
    evidence: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    gap_ids: np.ndarray,
    gap_classes: np.ndarray,
    anchor_sources: np.ndarray,
    position_sources: np.ndarray,
    patch: pd.DataFrame,
    candidate_id: str,
    expected_gap_class: str,
) -> dict[str, int]:
    """Apply valid supplement positions without overwriting an existing row."""
    if patch.empty:
        return {"valid_rows": 0, "applied_rows": 0, "skipped_existing_rows": 0}
    _validate_patch_columns(patch, expected_gap_class)
    patch = patch.loc[patch.candidate_id.astype(str).eq(candidate_id)].copy()
    if patch.empty:
        return {"valid_rows": 0, "applied_rows": 0, "skipped_existing_rows": 0}

    patch["utc"] = pd.to_datetime(patch["utc"], utc=True, errors="raise")
    patch_ns = patch["utc"].astype("int64").to_numpy(dtype=np.int64)
    valid = (
        patch["track_evidence"].astype(str).ne(UNRESOLVED_LABEL).to_numpy()
        & np.isfinite(patch["lat"].to_numpy(dtype=float))
        & np.isfinite(patch["lon"].to_numpy(dtype=float))
    )
    patch = patch.loc[valid].copy()
    patch_ns = patch_ns[valid]
    if patch.empty:
        return {"valid_rows": 0, "applied_rows": 0, "skipped_existing_rows": 0}
    if len(target_ns) == 0:
        raise ValueError(f"{expected_gap_class} patch has no matching voyage grid")

    indices = np.searchsorted(target_ns, patch_ns, side="left")
    exact = (indices < len(target_ns)) & (target_ns[np.minimum(indices, len(target_ns) - 1)] == patch_ns)
    if not exact.all():
        raise ValueError(f"{expected_gap_class} patch contains a time outside the full grid")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{expected_gap_class} patch contains duplicate grid positions")

    old_evidence = evidence[indices]
    can_apply = old_evidence == UNRESOLVED_LABEL
    skipped = int((~can_apply).sum())
    if can_apply.any():
        apply_indices = indices[can_apply]
        apply_patch = patch.iloc[np.flatnonzero(can_apply)]
        old_gap = gap_ids[apply_indices]
        mismatched = np.asarray(
            [value is not None and str(value) != str(expected) for value, expected in zip(old_gap, apply_patch.gap_id)]
        )
        missing_gap = np.asarray([value is None for value in old_gap])
        if mismatched.any() or missing_gap.any():
            raise ValueError(f"{expected_gap_class} patch does not align with source gap IDs")
        labels = apply_patch.track_evidence.astype(str).to_numpy(dtype=object)
        if not set(labels).issubset(ALLOWED_EVIDENCE):
            raise ValueError(f"{expected_gap_class} patch has an unknown evidence label")
        if (apply_patch.gap_id.astype(str).to_numpy() != np.asarray(old_gap, dtype=str)).any():
            raise ValueError(f"{expected_gap_class} patch changes a source gap ID")
        lat[apply_indices] = apply_patch.lat.to_numpy(dtype=float)
        lon[apply_indices] = apply_patch.lon.to_numpy(dtype=float)
        evidence[apply_indices] = labels
        anchor_sources[apply_indices] = _optional_raw_column(
            apply_patch, "anchor_source", len(apply_patch)
        )
        position_sources[apply_indices] = np.asarray(
            [_patch_position_source(label) for label in labels], dtype=object
        )
        gap_classes[apply_indices] = expected_gap_class

    return {
        "valid_rows": int(len(patch)),
        "applied_rows": int(can_apply.sum()),
        "skipped_existing_rows": skipped,
    }


def _assign_long_gap_ids(
    *,
    target_ns: np.ndarray,
    evidence: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    gap_ids: np.ndarray,
    gap_classes: np.ndarray,
    gaps: pd.DataFrame,
) -> int:
    if gaps.empty:
        return 0
    required = {"gap_id", "candidate_id", "gap_class", "gap_start", "gap_end"}
    missing = sorted(required.difference(gaps.columns))
    if missing:
        raise ValueError(f"gap table is missing required columns: {missing}")
    if gaps.duplicated("gap_id").any():
        raise ValueError("source gap IDs are not unique within a candidate shard")

    assigned = 0
    ordered = gaps.sort_values(["gap_start", "gap_end", "gap_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        start = _timestamp_ns(_value(row, "gap_start"))
        end = _timestamp_ns(_value(row, "gap_end"))
        if end <= start:
            raise ValueError(f"non-positive source gap: {_value(row, 'gap_id')}")
        left = int(np.searchsorted(target_ns, start, side="right"))
        right = int(np.searchsorted(target_ns, end, side="left"))
        if right <= left:
            continue
        indices = np.arange(left, right, dtype=np.int64)
        unresolved = evidence[indices] == UNRESOLVED_LABEL
        if unresolved.any():
            selected = indices[unresolved]
            existing = gap_ids[selected]
            if any(value is not None and str(value) != str(row.gap_id) for value in existing):
                raise ValueError("overlapping source gaps claim the same unresolved grid row")
            if np.isfinite(lat[selected]).any() or np.isfinite(lon[selected]).any():
                raise ValueError("unresolved grid row already carries a position")
            gap_ids[selected] = str(row.gap_id)
            gap_classes[selected] = str(row.gap_class)
            assigned += int(unresolved.sum())
    return assigned


def build_full_candidate_grid(
    candidate: Any,
    points: pd.DataFrame,
    gaps: pd.DataFrame,
    base_patch: pd.DataFrame,
    endpoint_patch: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one candidate's full grid and return rows plus auditable counts."""
    candidate_id = str(_value(candidate, "candidate_id"))
    vessel_uid = str(_value(candidate, "vessel_uid_clean"))
    mmsi = str(_value(candidate, "mmsi"))
    start = _utc_timestamp(_value(candidate, "departure_time"))
    end = _utc_timestamp(_value(candidate, "arrival_time"))
    target_ns = strict_grid_ns(start, end)
    n_rows = len(target_ns)

    evidence = np.full(n_rows, UNRESOLVED_LABEL, dtype=object)
    position_sources = np.full(n_rows, None, dtype=object)
    gap_ids = np.full(n_rows, None, dtype=object)
    gap_classes = np.full(n_rows, None, dtype=object)
    anchor_sources = np.full(n_rows, None, dtype=object)
    lat = np.full(n_rows, np.nan, dtype=float)
    lon = np.full(n_rows, np.nan, dtype=float)
    observation_time = np.full(n_rows, np.datetime64("NaT"), dtype="datetime64[ns]")
    source_provider = np.full(n_rows, None, dtype=object)
    source_resolution = np.full(n_rows, None, dtype=object)

    raw = points.copy()
    required_raw = {"UTC", "UTC_10min", "lat", "lon"}
    missing_raw = sorted(required_raw.difference(raw.columns))
    if missing_raw:
        raise ValueError(f"raw AIS points are missing columns: {missing_raw}")
    if not raw.empty:
        raw["UTC"] = pd.to_datetime(raw["UTC"], utc=True, errors="raise")
        raw["UTC_10min"] = pd.to_datetime(raw["UTC_10min"], utc=True, errors="raise")
        raw = raw.sort_values(["UTC", "UTC_10min"], kind="stable").reset_index(drop=True)
        if "vessel_uid" in raw and not raw.vessel_uid.astype(str).eq(vessel_uid).all():
            raise ValueError(f"raw AIS API identity mismatch for {candidate_id}")
        if "mmsi" in raw and not raw.mmsi.astype(str).eq(mmsi).all():
            raise ValueError(f"raw AIS MMSI identity mismatch for {candidate_id}")
        expected_bins = raw["UTC"].dt.floor("10min")
        if not raw["UTC_10min"].eq(expected_bins).all():
            raise ValueError(f"raw AIS UTC_10min is not the frozen floor rule for {candidate_id}")
        if raw.duplicated("UTC").any() or raw.duplicated("UTC_10min").any():
            raise ValueError(f"raw AIS has duplicate timestamps for {candidate_id}")
        if not np.isfinite(raw[["lat", "lon"]].to_numpy(dtype=float)).all():
            raise ValueError(f"raw AIS has non-finite coordinates for {candidate_id}")

    if raw.empty:
        inside = raw.copy()
    else:
        raw_ns = raw["UTC"].astype("int64").to_numpy(dtype=np.int64)
        inside = raw.loc[(raw_ns >= start.value) & (raw_ns <= end.value)].copy()

    if not inside.empty and n_rows:
        inside_ns = inside["UTC"].astype("int64").to_numpy(dtype=np.int64)
        inside_bins = inside["UTC_10min"].astype("int64").to_numpy(dtype=np.int64)
        target_index = np.searchsorted(target_ns, inside_bins, side="left")
        exact = (target_index < n_rows) & (
            target_ns[np.minimum(target_index, max(n_rows - 1, 0))] == inside_bins
        )
        if exact.any():
            selected = np.flatnonzero(exact)
            destination = target_index[selected]
            lat[destination] = inside.iloc[selected].lat.to_numpy(dtype=float)
            lon[destination] = inside.iloc[selected].lon.to_numpy(dtype=float)
            evidence[destination] = OBSERVED_LABEL
            position_sources[destination] = OBSERVED_LABEL
            observation_time[destination] = inside_ns[selected].astype("datetime64[ns]")
            source_provider[destination] = _optional_raw_column(
                inside.iloc[selected], "source_provider", len(selected)
            )
            source_resolution[destination] = _optional_raw_column(
                inside.iloc[selected], "source_resolution_original", len(selected)
            )

        right_index = np.searchsorted(inside_ns, target_ns, side="right")
        valid = (right_index > 0) & (right_index < len(inside_ns))
        if valid.any():
            candidate_indices = np.flatnonzero(valid)
            left_index = right_index[candidate_indices] - 1
            right_values = right_index[candidate_indices]
            duration = inside_ns[right_values] - inside_ns[left_index]
            short = duration <= SHORT_GAP_MAX_NS
            candidate_indices = candidate_indices[short]
            left_index = left_index[short]
            right_values = right_values[short]
            still_unresolved = evidence[candidate_indices] == UNRESOLVED_LABEL
            candidate_indices = candidate_indices[still_unresolved]
            left_index = left_index[still_unresolved]
            right_values = right_values[still_unresolved]
            if len(candidate_indices):
                left_ns = inside_ns[left_index]
                right_ns = inside_ns[right_values]
                fraction = (target_ns[candidate_indices] - left_ns) / (right_ns - left_ns)
                left_lat = inside.lat.to_numpy(dtype=float)[left_index]
                right_lat = inside.lat.to_numpy(dtype=float)[right_values]
                left_lon = inside.lon.to_numpy(dtype=float)[left_index]
                right_lon = inside.lon.to_numpy(dtype=float)[right_values]
                delta_lon = (right_lon - left_lon + 180.0) % 360.0 - 180.0
                lat[candidate_indices] = left_lat + fraction * (right_lat - left_lat)
                lon[candidate_indices] = (left_lon + fraction * delta_lon + 180.0) % 360.0 - 180.0
                evidence[candidate_indices] = SHORT_LABEL
                position_sources[candidate_indices] = "ais_linear_between_fixes_le4h"
                gap_ids[candidate_indices] = _short_gap_ids(
                    candidate_id, left_ns, right_ns
                )
                gap_classes[candidate_indices] = "ais_supported_le4h"

    candidate_gaps = gaps.loc[gaps.candidate_id.astype(str).eq(candidate_id)].copy()
    assigned_long_rows = _assign_long_gap_ids(
        target_ns=target_ns,
        evidence=evidence,
        lat=lat,
        lon=lon,
        gap_ids=gap_ids,
        gap_classes=gap_classes,
        gaps=candidate_gaps,
    )

    base_counts = _apply_patch(
        target_ns=target_ns,
        evidence=evidence,
        lat=lat,
        lon=lon,
        gap_ids=gap_ids,
        gap_classes=gap_classes,
        anchor_sources=anchor_sources,
        position_sources=position_sources,
        patch=base_patch,
        candidate_id=candidate_id,
        expected_gap_class="gfw_candidate_4_24h",
    )
    endpoint_counts = _apply_patch(
        target_ns=target_ns,
        evidence=evidence,
        lat=lat,
        lon=lon,
        gap_ids=gap_ids,
        gap_classes=gap_classes,
        anchor_sources=anchor_sources,
        position_sources=position_sources,
        patch=endpoint_patch,
        candidate_id=candidate_id,
        expected_gap_class="unresolved_internal_gt24h",
    )

    if not set(evidence).issubset(ALLOWED_EVIDENCE):
        raise ValueError(f"unknown final evidence label for {candidate_id}")
    resolved = evidence != UNRESOLVED_LABEL
    if np.any(resolved & (~np.isfinite(lat) | ~np.isfinite(lon))):
        raise ValueError(f"resolved grid row lacks a position for {candidate_id}")
    if np.any((~resolved) & (np.isfinite(lat) | np.isfinite(lon))):
        raise ValueError(f"unresolved grid row has a position for {candidate_id}")
    if np.any((evidence == OBSERVED_LABEL) & pd.isna(observation_time)):
        raise ValueError(f"observed grid row lacks actual AIS time for {candidate_id}")

    candidate_columns = {
        "candidate_id": pd.Series([candidate_id] * n_rows, dtype="string"),
        "mmsi": pd.Series([mmsi] * n_rows, dtype="string"),
        "vessel_uid_clean": pd.Series([vessel_uid] * n_rows, dtype="string"),
        "layer": pd.Series([str(_value(candidate, "layer"))] * n_rows, dtype="string"),
        "utc": pd.to_datetime(target_ns, unit="ns", utc=True),
        "lat": lat,
        "lon": lon,
        "track_evidence": pd.Series(evidence, dtype="string"),
        "position_source": pd.Series(position_sources, dtype="string"),
        "gap_id": pd.Series(gap_ids, dtype="string"),
        "gap_class": pd.Series(gap_classes, dtype="string"),
        "anchor_source": pd.Series(anchor_sources, dtype="string"),
        "observation_time_utc": pd.to_datetime(observation_time, utc=True),
        "source_provider": pd.Series(source_provider, dtype="string"),
        "source_resolution_original": pd.Series(source_resolution, dtype="string"),
        "source_shard": pd.Series(
            [str(_value(candidate, "source_shard"))] * n_rows, dtype="string"
        ),
    }
    result = pd.DataFrame(candidate_columns)
    summary = {
        "candidate_id": candidate_id,
        "mmsi": mmsi,
        "vessel_uid_clean": vessel_uid,
        "layer": str(_value(candidate, "layer")),
        "source_shard": str(_value(candidate, "source_shard")),
        "departure_time": start,
        "arrival_time": end,
        "grid_rows": int(n_rows),
        "observed_ais_rows": int((evidence == OBSERVED_LABEL).sum()),
        "ais_interpolated_le4h_rows": int((evidence == SHORT_LABEL).sum()),
        "gfw_anchor_rows": int((evidence == GFW_ANCHOR_LABEL).sum()),
        "gfw_supported_interpolation_rows": int(
            (evidence == GFW_INTERPOLATION_LABEL).sum()
        ),
        "endpoint_24_48h_rows": int((evidence == ENDPOINT_LABEL).sum()),
        "unresolved_rows": int((evidence == UNRESOLVED_LABEL).sum()),
        "source_gap_grid_rows_assigned": int(assigned_long_rows),
        "base_valid_patch_rows": base_counts["valid_rows"],
        "base_applied_rows": base_counts["applied_rows"],
        "base_skipped_existing_rows": base_counts["skipped_existing_rows"],
        "endpoint_valid_patch_rows": endpoint_counts["valid_rows"],
        "endpoint_applied_rows": endpoint_counts["applied_rows"],
        "endpoint_skipped_existing_rows": endpoint_counts["skipped_existing_rows"],
    }
    return result, summary


def validate_candidate_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Run cheap, independent shape checks on a completed candidate frame."""
    required = {
        "candidate_id",
        "mmsi",
        "vessel_uid_clean",
        "utc",
        "lat",
        "lon",
        "track_evidence",
        "gap_id",
        "gap_class",
        "observation_time_utc",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"complete candidate frame is missing columns: {missing}")
    if frame.duplicated(["candidate_id", "utc"]).any():
        raise ValueError("complete candidate frame contains duplicate candidate/time keys")
    labels = set(frame.track_evidence.dropna().astype(str))
    if not labels.issubset(ALLOWED_EVIDENCE):
        raise ValueError(f"complete candidate frame has unexpected labels: {labels - ALLOWED_EVIDENCE}")
    resolved = frame.track_evidence.astype(str).ne(UNRESOLVED_LABEL)
    if frame.loc[resolved, ["lat", "lon"]].isna().any().any():
        raise ValueError("resolved complete candidate row lacks coordinates")
    if frame.loc[~resolved, ["lat", "lon"]].notna().any().any():
        raise ValueError("unresolved complete candidate row has coordinates")
    if not pd.to_datetime(frame.utc, utc=True).eq(pd.to_datetime(frame.utc, utc=True).dt.floor("10min")).all():
        raise ValueError("complete candidate frame contains off-grid timestamps")
    return {
        "rows": int(len(frame)),
        "candidates": int(frame.candidate_id.nunique()),
        "label_counts": frame.track_evidence.astype(str).value_counts().to_dict(),
    }
