"""Conservative GFW acquisition opportunity and candidate association helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from shapely import get_coordinates
from shapely.geometry import Point

from .dark_gaps import EARTH_RADIUS_KM, KM_PER_NAUTICAL_MILE


NS_PER_HOUR = 3_600_000_000_000


def haversine_km_vector(
    lat_a: float, lon_a: float, lat_b: np.ndarray, lon_b: np.ndarray
) -> np.ndarray:
    first = np.radians(float(lat_a))
    second = np.radians(np.asarray(lat_b, dtype=float))
    dlat = second - first
    dlon = np.radians(np.asarray(lon_b, dtype=float) - float(lon_a))
    h = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(first) * np.cos(second) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def great_circle_point(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float, fraction: float
) -> tuple[float, float]:
    """One point on the shortest spherical path, with longitude wrapping."""

    f = float(np.clip(fraction, 0.0, 1.0))
    lon_r = np.radians([lon_a, lon_b])
    lat_r = np.radians([lat_a, lat_b])
    vectors = np.column_stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    )
    dot = float(np.clip(np.dot(vectors[0], vectors[1]), -1.0, 1.0))
    omega = float(np.arccos(dot))
    if abs(np.sin(omega)) < 1e-12:
        return float(lat_a), float(lon_a)
    point = (
        np.sin((1.0 - f) * omega) / np.sin(omega) * vectors[0]
        + np.sin(f * omega) / np.sin(omega) * vectors[1]
    )
    point = point / np.linalg.norm(point)
    lat = float(np.degrees(np.arcsin(np.clip(point[2], -1.0, 1.0))))
    lon = float(np.degrees(np.arctan2(point[1], point[0])))
    return lat, lon


def great_circle_points(
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(fractions, dtype=float)
    points = [great_circle_point(lat_a, lon_a, lat_b, lon_b, value) for value in values]
    return (
        np.asarray([item[0] for item in points], dtype=float),
        np.asarray([item[1] for item in points], dtype=float),
    )


def gap_region_labels(
    gap: Mapping[str, Any],
    regions: Mapping[str, list[float]],
    *,
    maximum_samples: int,
) -> list[str]:
    """Apply E031's diagnostic great-circle attribution to one source gap."""

    if not bool(gap["bounded"]):
        return []
    if int(maximum_samples) <= 0:
        raise ValueError("maximum region-attribution samples must be positive")
    source_start = pd.Timestamp(gap["source_gap_start_utc"])
    source_end = pd.Timestamp(gap["source_gap_end_utc"])
    clipped_start = pd.Timestamp(gap["gap_start_utc"])
    clipped_end = pd.Timestamp(gap["gap_end_utc"])
    source_seconds = (source_end - source_start).total_seconds()
    if source_seconds <= 0:
        raise ValueError("source gap must have positive duration")
    first_fraction = (clipped_start - source_start).total_seconds() / source_seconds
    last_fraction = (clipped_end - source_start).total_seconds() / source_seconds
    if (
        first_fraction < -1e-12
        or last_fraction > 1.0 + 1e-12
        or last_fraction <= first_fraction
    ):
        raise ValueError("clipped gap interval is outside its source gap")
    probe_hours = float(gap["dark_hours_in_probe"])
    samples = int(min(max(np.ceil(probe_hours), 2), int(maximum_samples)))
    local = (np.arange(samples, dtype=float) + 0.5) / samples
    fractions = first_fraction + local * (last_fraction - first_fraction)
    lat, lon = great_circle_points(
        float(gap["lat_a"]),
        float(gap["lon_a"]),
        float(gap["lat_b"]),
        float(gap["lon_b"]),
        fractions,
    )
    labels: list[str] = []
    for name, bounds in regions.items():
        south, north, west, east = map(float, bounds)
        if ((lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)).any():
            labels.append(name)
    return labels


def leash_radii_at(
    gap: Mapping[str, Any], timestamp: pd.Timestamp
) -> tuple[float, float, float]:
    source_start = pd.Timestamp(gap["source_gap_start_utc"])
    source_end = pd.Timestamp(gap["source_gap_end_utc"])
    instant = pd.Timestamp(timestamp)
    for name, value in (
        ("source_start", source_start),
        ("source_end", source_end),
        ("timestamp", instant),
    ):
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware")
    source_start = source_start.tz_convert("UTC")
    source_end = source_end.tz_convert("UTC")
    instant = instant.tz_convert("UTC")
    duration = (source_end - source_start).total_seconds() / 3600.0
    elapsed = (instant - source_start).total_seconds() / 3600.0
    if duration <= 0 or elapsed < -1e-9 or elapsed > duration + 1e-9:
        raise ValueError("satellite timestamp is outside the source gap")
    speed = float(gap["effective_speed_kn"]) * KM_PER_NAUTICAL_MILE
    return (
        speed * max(elapsed, 0.0),
        speed * max(duration - elapsed, 0.0),
        elapsed / duration,
    )


def points_reachable(
    gap: Mapping[str, Any],
    timestamp: pd.Timestamp,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    tolerance_km: float = 0.001,
) -> np.ndarray:
    if not bool(gap["bounded"]):
        return np.zeros(len(lat), dtype=bool)
    radius_a, radius_b, _ = leash_radii_at(gap, timestamp)
    distance_a = haversine_km_vector(float(gap["lat_a"]), float(gap["lon_a"]), lat, lon)
    distance_b = haversine_km_vector(float(gap["lat_b"]), float(gap["lon_b"]), lat, lon)
    return (distance_a <= radius_a + tolerance_km) & (
        distance_b <= radius_b + tolerance_km
    )


def footprint_reachable(
    gap: Mapping[str, Any],
    footprint: Any,
    scene_time: pd.Timestamp,
    *,
    boundary_resolution_degrees: float,
    tolerance_km: float = 0.001,
) -> tuple[bool, str, int]:
    """Test whether an acquisition footprint intersects the reachable lens.

    The direct great-circle point is an exact positive witness. Otherwise the
    clipped footprint boundary is deterministically segmentized before exact
    geodesic leash checks. A negative is resolution-qualified; E032 will audit
    finer resolutions before treating it as a final no-footprint verdict.
    """

    if not bool(gap["bounded"]):
        return False, "unbracketed_gap", 0
    _, _, fraction = leash_radii_at(gap, scene_time)
    direct_lat, direct_lon = great_circle_point(
        float(gap["lat_a"]),
        float(gap["lon_a"]),
        float(gap["lat_b"]),
        float(gap["lon_b"]),
        fraction,
    )
    if footprint.covers(Point(direct_lon, direct_lat)):
        return True, "direct_path_witness", 1

    if boundary_resolution_degrees <= 0:
        raise ValueError("footprint boundary resolution must be positive")
    sampled = footprint.segmentize(float(boundary_resolution_degrees))
    coordinates = get_coordinates(sampled)
    representative = footprint.representative_point()
    coordinates = np.vstack([coordinates, [representative.x, representative.y]])
    reachable = points_reachable(
        gap,
        scene_time,
        coordinates[:, 1],
        coordinates[:, 0],
        tolerance_km=tolerance_km,
    )
    if reachable.any():
        return True, "sampled_footprint_witness", int(len(coordinates))
    return False, "no_witness_at_resolution", int(len(coordinates))


def canonical_gfw_mmsi(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text if text.isdigit() else None


def classify_reachable_detections(
    target_mmsi: str,
    detections: pd.DataFrame,
    *,
    unmatched_category: str = "unmatched",
) -> tuple[str, pd.DataFrame, dict[str, int]]:
    """Classify without a score threshold and without choosing a top candidate."""

    work = detections.copy()
    if work.empty:
        return (
            "footprint_without_reachable_detection",
            work,
            {
                "reachable_detections": 0,
                "target_mmsi_matches_pending_crosswalk": 0,
                "gfw_other_matches": 0,
                "unmatched_candidates": 0,
            },
        )
    work["canonical_mmsi"] = work["mmsi"].map(canonical_gfw_mmsi)
    work["gfw_is_unmatched"] = (
        work["matched_category"].astype("string").eq(unmatched_category)
    )
    work["target_mmsi_match_pending_crosswalk"] = (~work["gfw_is_unmatched"]) & work[
        "canonical_mmsi"
    ].eq(str(target_mmsi))
    work["gfw_other_match"] = (
        (~work["gfw_is_unmatched"])
        & work["canonical_mmsi"].notna()
        & ~work["canonical_mmsi"].eq(str(target_mmsi))
    )
    target = int(work["target_mmsi_match_pending_crosswalk"].sum())
    other = int(work["gfw_other_match"].sum())
    unmatched = int(work["gfw_is_unmatched"].sum())
    if target:
        category = "target_mmsi_match_pending_crosswalk"
    elif unmatched > 1:
        category = "unmatched_ambiguous_candidates"
    elif unmatched == 1:
        category = "unmatched_unique_candidate"
    elif other:
        category = "other_vessel_match_only"
    else:
        category = "reachable_detection_identity_unresolved"
    return (
        category,
        work,
        {
            "reachable_detections": int(len(work)),
            "target_mmsi_matches_pending_crosswalk": target,
            "gfw_other_matches": other,
            "unmatched_candidates": unmatched,
        },
    )
