"""Extract the project's strictly-over-four-hour AIS gaps from voyage windows."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


NS_PER_HOUR = 3_600_000_000_000
KM_PER_NAUTICAL_MILE = 1.852
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    first = np.radians(lat_a)
    second = np.radians(lat_b)
    dlat = second - first
    dlon = np.radians(lon_b - lon_a)
    h = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(first) * np.cos(second) * np.sin(dlon / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def effective_speed_kn(
    speed_cap_kn: float,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    gap_hours: float,
) -> tuple[float, bool]:
    if gap_hours <= 0:
        raise ValueError("gap duration must be positive")
    implied = (
        haversine_km(lat_a, lon_a, lat_b, lon_b) / KM_PER_NAUTICAL_MILE / gap_hours
    )
    return (
        (float(implied), True)
        if implied > speed_cap_kn
        else (float(speed_cap_kn), False)
    )


def stable_gap_id(candidate_id: str, sequence: int, start_ns: int, end_ns: int) -> str:
    payload = f"{candidate_id}|{sequence}|{start_ns}|{end_ns}".encode("utf-8")
    return "gap_" + hashlib.sha256(payload).hexdigest()[:20]


def _ns(value: pd.Timestamp) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return int(stamp.value)


def extract_candidate_gaps(
    candidate: dict[str, Any],
    vessel_points: pd.DataFrame,
    *,
    probe_start: pd.Timestamp,
    probe_end: pd.Timestamp,
    gap_limit_hours: float,
    speed_cap_kn: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return every dark interval and one candidate-level closure record.

    The probe interval is closed at both ends. A source gap remains dark when
    its full duration is over four hours even if clipping to the probe window
    leaves less than four hours inside the probe.
    """

    departure_ns = _ns(candidate["departure_time"])
    arrival_ns = _ns(candidate["arrival_time"])
    start_ns = max(departure_ns, _ns(probe_start))
    end_ns = min(arrival_ns, _ns(probe_end))
    if end_ns <= start_ns:
        raise ValueError("candidate does not overlap the probe window")

    if vessel_points.empty:
        point_ns = np.empty(0, np.int64)
        point_lat = np.empty(0, float)
        point_lon = np.empty(0, float)
    else:
        # The bucket caller has already applied accepted identity validation,
        # coordinate validation, deterministic UTC deduplication and sorting.
        # Slice its immutable arrays instead of copying a whole vessel track
        # once per candidate window.
        if "_utc_ns" in vessel_points.columns:
            all_ns = vessel_points["_utc_ns"].to_numpy(np.int64)
        else:
            all_ns = (
                pd.to_datetime(vessel_points["UTC"], utc=True)
                .astype("int64")
                .to_numpy(np.int64)
            )
        left = int(np.searchsorted(all_ns, departure_ns, side="left"))
        right = int(np.searchsorted(all_ns, arrival_ns, side="right"))
        point_ns = all_ns[left:right]
        point_lat = vessel_points["lat"].to_numpy(float)[left:right]
        point_lon = vessel_points["lon"].to_numpy(float)[left:right]
    limit_ns = int(float(gap_limit_hours) * NS_PER_HOUR)

    source_gaps: list[dict[str, Any]] = []
    if point_ns.size == 0:
        source_gaps.append(
            {
                "gap_type": "no_track",
                "source_start_ns": departure_ns,
                "source_end_ns": arrival_ns,
                "bounded": False,
            }
        )
    else:
        if point_ns[0] - departure_ns > limit_ns:
            source_gaps.append(
                {
                    "gap_type": "leading_unbracketed",
                    "source_start_ns": departure_ns,
                    "source_end_ns": int(point_ns[0]),
                    "bounded": False,
                }
            )
        for index, duration_ns in enumerate(np.diff(point_ns)):
            if int(duration_ns) <= limit_ns:
                continue
            source_gaps.append(
                {
                    "gap_type": "internal_bounded",
                    "source_start_ns": int(point_ns[index]),
                    "source_end_ns": int(point_ns[index + 1]),
                    "bounded": True,
                    "lat_a": float(point_lat[index]),
                    "lon_a": float(point_lon[index]),
                    "lat_b": float(point_lat[index + 1]),
                    "lon_b": float(point_lon[index + 1]),
                }
            )
        if arrival_ns - point_ns[-1] > limit_ns:
            source_gaps.append(
                {
                    "gap_type": "trailing_unbracketed",
                    "source_start_ns": int(point_ns[-1]),
                    "source_end_ns": arrival_ns,
                    "bounded": False,
                }
            )

    rows: list[dict[str, Any]] = []
    for sequence, gap in enumerate(source_gaps):
        clipped_start = max(int(gap["source_start_ns"]), start_ns)
        clipped_end = min(int(gap["source_end_ns"]), end_ns)
        if clipped_end <= clipped_start:
            continue
        full_hours = (
            int(gap["source_end_ns"]) - int(gap["source_start_ns"])
        ) / NS_PER_HOUR
        if full_hours <= gap_limit_hours:
            continue
        row = {
            "gap_id": stable_gap_id(
                str(candidate["candidate_id"]), sequence, clipped_start, clipped_end
            ),
            "candidate_id": str(candidate["candidate_id"]),
            "mmsi": str(candidate["mmsi"]),
            "imo": candidate.get("imo"),
            "vessel_uid_clean": str(candidate["vessel_uid_clean"]),
            "route": candidate["route"],
            "gap_type": gap["gap_type"],
            "bounded": bool(gap["bounded"]),
            "source_gap_start_utc": pd.Timestamp(int(gap["source_start_ns"]), tz="UTC"),
            "source_gap_end_utc": pd.Timestamp(int(gap["source_end_ns"]), tz="UTC"),
            "gap_start_utc": pd.Timestamp(clipped_start, tz="UTC"),
            "gap_end_utc": pd.Timestamp(clipped_end, tz="UTC"),
            "dark_hours_in_probe": (clipped_end - clipped_start) / NS_PER_HOUR,
            "full_source_gap_hours": full_hours,
            "speed_cap_kn": float(speed_cap_kn),
            "lat_a": np.nan,
            "lon_a": np.nan,
            "lat_b": np.nan,
            "lon_b": np.nan,
            "effective_speed_kn": np.nan,
            "implied_exceeds_cap": False,
        }
        if gap["bounded"]:
            effective, exceeds = effective_speed_kn(
                speed_cap_kn,
                gap["lat_a"],
                gap["lon_a"],
                gap["lat_b"],
                gap["lon_b"],
                full_hours,
            )
            row.update(
                {
                    "lat_a": gap["lat_a"],
                    "lon_a": gap["lon_a"],
                    "lat_b": gap["lat_b"],
                    "lon_b": gap["lon_b"],
                    "effective_speed_kn": effective,
                    "implied_exceeds_cap": exceeds,
                }
            )
        rows.append(row)

    probe_hours = (end_ns - start_ns) / NS_PER_HOUR
    dark_hours = float(sum(row["dark_hours_in_probe"] for row in rows))
    closure = {
        "candidate_id": str(candidate["candidate_id"]),
        "mmsi": str(candidate["mmsi"]),
        "route": candidate["route"],
        "probe_start_utc": pd.Timestamp(start_ns, tz="UTC"),
        "probe_end_utc": pd.Timestamp(end_ns, tz="UTC"),
        "probe_voyage_hours": probe_hours,
        "dark_hours_over_4h": dark_hours,
        "measurable_hours": probe_hours - dark_hours,
        "dark_gap_count": len(rows),
        "bounded_gap_count": sum(int(row["bounded"]) for row in rows),
        "unbracketed_gap_count": sum(int(not row["bounded"]) for row in rows),
    }
    if not -1e-9 <= closure["measurable_hours"] <= probe_hours + 1e-9:
        raise AssertionError("candidate dark-hour closure is outside the voyage window")
    return rows, closure
