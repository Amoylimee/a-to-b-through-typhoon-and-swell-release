"""E001 cached-GFW reconstruction on a strict UTC ten-minute gap grid."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sea_path.real_anchors import AnchorWindow, GapTask, reconstruct_real_anchor_task

GRID_SECONDS = 600
ANCHOR_SOURCE = "gfw_presence_v4_2025_cached_admissible_grid_centre"


def strict_grid_seconds(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """UTC ten-minute instants strictly between the two observed AIS fixes."""
    left = float(pd.Timestamp(start).timestamp())
    right = float(pd.Timestamp(end).timestamp())
    first = np.floor(left / GRID_SECONDS) + 1
    last = np.ceil(right / GRID_SECONDS) - 1
    if last < first:
        return np.empty(0, dtype=float)
    return np.arange(first, last + 1, dtype=float) * GRID_SECONDS


def stable_rank(value: str, namespace: str = "E001-cached-gfw-10min-v1") -> str:
    return hashlib.sha256(f"{namespace}|{value}".encode()).hexdigest()


def make_task(gap: Any, anchors: pd.DataFrame) -> GapTask:
    ordered = anchors.sort_values(["hour_utc", "target_hour_id"], kind="stable")
    windows = tuple(
        AnchorWindow(
            target_hour_id=str(r.target_hour_id),
            hour_start_s=float(pd.Timestamp(r.hour_utc).timestamp()),
            hour_end_s=float(pd.Timestamp(r.hour_utc).timestamp() + 3600),
            lat=float(r.anchor_lat),
            lon=float(r.anchor_lon),
            source=ANCHOR_SOURCE,
        )
        for r in ordered.itertuples(index=False)
    )
    targets = strict_grid_seconds(gap.gap_start, gap.gap_end)
    return GapTask(
        gap_id=str(gap.gap_id), candidate_id=str(gap.candidate_id),
        mmsi=str(gap.mmsi), route=str(gap.layer),
        t_start=float(pd.Timestamp(gap.gap_start).timestamp()),
        lat_start=float(gap.left_lat), lon_start=float(gap.left_lon),
        t_end=float(pd.Timestamp(gap.gap_end).timestamp()),
        lat_end=float(gap.right_lat), lon_end=float(gap.right_lon),
        target_times=tuple(targets.tolist()), anchors=windows,
        consumer_admissible_anchor_hours=len(windows),
        upstream_status_counts=(("admissible_unique_anchor", len(windows)),),
    )


def unresolved_track(gap: Any, reason: str) -> tuple[pd.DataFrame, dict]:
    targets = strict_grid_seconds(gap.gap_start, gap.gap_end)
    track = pd.DataFrame({
        "gap_id": str(gap.gap_id), "candidate_id": str(gap.candidate_id),
        "mmsi": str(gap.mmsi), "utc": pd.to_datetime(targets, unit="s", utc=True),
        "lat": np.full(len(targets), np.nan), "lon": np.full(len(targets), np.nan),
        "track_evidence": "unresolved", "leg_index": pd.array([None] * len(targets), dtype="Int16"),
        "anchor_source": pd.Series([None] * len(targets), dtype="string"),
    })
    status = {
        "gap_id": str(gap.gap_id), "candidate_id": str(gap.candidate_id),
        "mmsi": str(gap.mmsi), "layer": str(gap.layer), "gap_hours": float(gap.gap_hours),
        "gap_outcome": reason, "admissible_anchor_hours": 0,
        "ten_minute_grid_rows": len(targets), "resolved_grid_rows": 0,
        "unresolved_grid_rows": len(targets), "reconstructed_hours": 0.0,
        "unresolved_hours": float(gap.gap_hours),
    }
    return track, status


def reconstruct_gap(router: Any, gap: Any, anchors: pd.DataFrame):
    """Return complete grid, status, scheduled anchors and route legs for one gap."""
    task = make_task(gap, anchors)
    result = reconstruct_real_anchor_task(router, task)
    targets = np.asarray(task.target_times, dtype=float)
    track = pd.DataFrame({
        "gap_id": str(gap.gap_id), "candidate_id": str(gap.candidate_id),
        "mmsi": str(gap.mmsi), "utc": pd.to_datetime(targets, unit="s", utc=True),
        "lat": np.full(len(targets), np.nan), "lon": np.full(len(targets), np.nan),
        "track_evidence": "unresolved", "leg_index": pd.array([None] * len(targets), dtype="Int16"),
        "anchor_source": pd.Series([None] * len(targets), dtype="string"),
    })
    if len(result["times"]):
        loc = pd.Index(targets).get_indexer(result["times"])
        if (loc < 0).any():
            raise AssertionError("reconstruction emitted a point outside the ten-minute grid")
        track.loc[loc, ["lat", "lon"]] = result["positions"]
        labels = np.where(result["position_source"] == "anchor", "gfw_anchor",
                          "gfw_supported_interpolation")
        track.loc[loc, "track_evidence"] = labels
        track.loc[loc, "leg_index"] = result["leg_index"].astype("int16")
        track.loc[loc, "anchor_source"] = ANCHOR_SOURCE
    resolved = int(track.track_evidence.ne("unresolved").sum())
    status = {
        "gap_id": str(gap.gap_id), "candidate_id": str(gap.candidate_id),
        "mmsi": str(gap.mmsi), "layer": str(gap.layer), "gap_hours": float(gap.gap_hours),
        "gap_outcome": result["gap_outcome"],
        "admissible_anchor_hours": int(result["admissible_anchor_hours"]),
        "scheduled_anchor_windows": int(result["scheduled_anchor_windows"]),
        "ten_minute_grid_rows": len(targets), "resolved_grid_rows": resolved,
        "unresolved_grid_rows": len(targets) - resolved,
        "reconstructed_hours": float(result["reconstructed_hours"]),
        "unresolved_hours": float(result["unresolved_hours"]),
    }
    schedule = pd.DataFrame(result["anchor_records"])
    if len(schedule):
        schedule["scheduled_utc"] = pd.to_datetime(schedule.scheduled_epoch_s, unit="s", utc=True)
        schedule["evidence_hour_start_utc"] = pd.to_datetime(schedule.hour_start_epoch_s, unit="s", utc=True)
        schedule["evidence_hour_end_utc"] = pd.to_datetime(schedule.hour_end_epoch_s, unit="s", utc=True)
        schedule["track_evidence"] = "gfw_anchor"
    legs = pd.DataFrame(result["leg_records"])
    return track, status, schedule, legs
