#!/usr/bin/env python3
"""Census TC upper bounds for every directed OD × ship-type combination."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
CODING_ROOT = HERE.parents[3]
TRACK_ROOT = CODING_ROOT / "data/processed/products/companion_tracks/2025"
U1_GROUPS = CODING_ROOT / ".runs/2026-09-24_0135_E002-2025-od-exposure-release/10_u1/20_od_shiptype_all_combinations.parquet"
EXPECTED_WEATHER_ROWS = 174_615_386
EXPECTED_UNKNOWN_ROWS = 18_074_055
EXPECTED_BOUNDED_ROWS = 17_637_842
EXPECTED_UNBOUNDED_ROWS = 436_213
EXPECTED_UNKNOWN_HOURS = 16_744_981.166388888
SPEED_KN = 25.0
KM_PER_NAUTICAL_MILE = 1.852
NS_PER_HOUR = 3.6e12
GROUP_COLUMNS = ["origin_city", "dest_city", "shiptype_group"]

PROBE_SCRIPT = HERE / "10_probe_reachability.py"
PROBE_SPEC = importlib.util.spec_from_file_location("e002_u2_probe_for_tc_upper", PROBE_SCRIPT)
assert PROBE_SPEC and PROBE_SPEC.loader
PROBE = importlib.util.module_from_spec(PROBE_SPEC)
sys.modules[PROBE_SPEC.name] = PROBE
PROBE_SPEC.loader.exec_module(PROBE)
LEGACY = PROBE.legacy


def _unit(lat: float, lon: float) -> np.ndarray:
    lat_r = np.radians(float(lat))
    lon_r = np.radians(float(lon))
    return np.array(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)],
        dtype=np.float64,
    )


def _cap(center: np.ndarray, radius_km: float) -> tuple[np.ndarray, float] | None:
    """Represent a closed spherical cap as ``center · x >= cos(radius)``."""
    if radius_km < 0:
        raise ValueError("spherical-cap radius must be non-negative")
    radius_rad = float(radius_km) / float(LEGACY.EARTH_RADIUS_KM)
    if radius_rad >= np.pi:
        return None  # the whole sphere
    return center, float(np.cos(radius_rad))


def _inside_all(point: np.ndarray, caps: list[tuple[np.ndarray, float]]) -> bool:
    return all(float(np.dot(center, point)) >= cos_radius - 2e-12 for center, cos_radius in caps)


def _boundary_representative(center: np.ndarray, cos_radius: float) -> np.ndarray:
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(center)))]
    tangent = axis - float(np.dot(axis, center)) * center
    tangent /= np.linalg.norm(tangent)
    return cos_radius * center + np.sqrt(max(0.0, 1.0 - cos_radius * cos_radius)) * tangent


def spherical_caps_intersect(caps: list[tuple[np.ndarray, float] | None]) -> bool:
    """Exact closed-cap intersection on a sphere, using centres and circle vertices."""
    active = [item for item in caps if item is not None]
    if not active:
        return True

    # If a cap centre lies in every cap, the common region is non-empty.
    for center, _ in active:
        if _inside_all(center, active):
            return True

    # An intersection with no circle crossings can only be nested; its smallest
    # cap centre is covered above. Boundary representatives also cover coincident
    # circles whose common arc has no isolated vertices.
    for center, cos_radius in active:
        point = _boundary_representative(center, cos_radius)
        if _inside_all(point, active):
            return True

    # Intersections of two small-circle boundaries are the only remaining
    # vertices of a non-empty common region.
    for first in range(len(active)):
        u, cos_a = active[first]
        for second in range(first + 1, len(active)):
            v, cos_b = active[second]
            dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
            tangent = v - dot * u
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-12:
                # Parallel/coincident boundary circles are handled by their
                # representatives and by intersections with the other circles.
                continue
            tangent /= tangent_norm
            along_tangent = (cos_b - dot * cos_a) / tangent_norm
            remaining = 1.0 - cos_a * cos_a - along_tangent * along_tangent
            if remaining < -2e-12:
                continue
            normal = np.cross(u, tangent)
            normal /= np.linalg.norm(normal)
            offset = np.sqrt(max(0.0, remaining))
            for sign in (-1.0, 1.0):
                point = cos_a * u + along_tangent * tangent + sign * offset * normal
                point /= np.linalg.norm(point)
                if _inside_all(point, active):
                    return True
    return False


def necessary_center_filter(
    *,
    storm_lat: np.ndarray,
    storm_lon: np.ndarray,
    event_ns: int,
    left_time_ns: int,
    left_lat: float,
    left_lon: float,
    right_time_ns: int,
    right_lat: float,
    right_lon: float,
    radius_km: float,
    speed_kn: float = SPEED_KN,
) -> np.ndarray:
    """Necessary endpoint-distance conditions, evaluated per storm centre."""
    km_per_hour = speed_kn * KM_PER_NAUTICAL_MILE
    possible = np.ones(len(storm_lat), dtype=bool)
    if left_time_ns >= 0:
        elapsed_h = (event_ns - left_time_ns) / NS_PER_HOUR
        if elapsed_h < -1e-12:
            raise ValueError("event precedes its left known endpoint")
        distance = LEGACY.great_circle_km(
            np.full(len(storm_lat), left_lat),
            np.full(len(storm_lon), left_lon),
            storm_lat,
            storm_lon,
        )
        possible &= distance <= radius_km + km_per_hour * max(0.0, elapsed_h) + 1e-9
    if right_time_ns >= 0:
        elapsed_h = (right_time_ns - event_ns) / NS_PER_HOUR
        if elapsed_h < -1e-12:
            raise ValueError("event follows its right known endpoint")
        distance = LEGACY.great_circle_km(
            np.full(len(storm_lat), right_lat),
            np.full(len(storm_lon), right_lon),
            storm_lat,
            storm_lon,
        )
        possible &= distance <= radius_km + km_per_hour * max(0.0, elapsed_h) + 1e-9
    return possible


def _segment_caps(row: Any, storm_lat: float, storm_lon: float, radius_km: float):
    speed_km_h = SPEED_KN * KM_PER_NAUTICAL_MILE
    event_ns = int(row.position_time_ns)
    caps: list[tuple[np.ndarray, float] | None] = []
    has_left = int(row.left_time_ns) >= 0
    has_right = int(row.right_time_ns) >= 0
    if has_left:
        dt_h = (event_ns - int(row.left_time_ns)) / NS_PER_HOUR
        if dt_h < -1e-12:
            raise ValueError("event precedes its left known endpoint")
        caps.append(_cap(_unit(row.left_lat, row.left_lon), speed_km_h * max(0.0, dt_h)))
    if has_right:
        dt_h = (int(row.right_time_ns) - event_ns) / NS_PER_HOUR
        if dt_h < -1e-12:
            raise ValueError("event follows its right known endpoint")
        caps.append(_cap(_unit(row.right_lat, row.right_lon), speed_km_h * max(0.0, dt_h)))
    if not caps:
        raise ValueError("bounded unknown segment lacks a known endpoint")
    caps.append(_cap(_unit(storm_lat, storm_lon), radius_km))
    return caps


def _endpoint_speed_feasible(row: Any) -> bool:
    if int(row.left_time_ns) < 0 or int(row.right_time_ns) < 0:
        return True
    elapsed_h = (int(row.right_time_ns) - int(row.left_time_ns)) / NS_PER_HOUR
    if elapsed_h <= 0:
        raise ValueError("known endpoint times are not strictly increasing")
    distance = float(
        LEGACY.great_circle_km(
            float(row.left_lat), float(row.left_lon),
            float(row.right_lat), float(row.right_lon),
        )
    )
    return distance <= SPEED_KN * KM_PER_NAUTICAL_MILE * elapsed_h + 1e-6


def _prepare_active_segments(
    active_unknown: pd.DataFrame,
    frame: pd.DataFrame,
    known_mask: np.ndarray,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve nearest positioned endpoints only for hours with a TC centre."""
    if active_unknown.empty:
        return pd.DataFrame()
    active_unknown = active_unknown.reset_index(drop=True).copy()
    active_unknown["_segment_id"] = np.arange(len(active_unknown), dtype=np.int64)
    active_candidate_ids = set(active_unknown.candidate_id.astype(str).unique())
    known = frame.loc[
        known_mask & frame.candidate_id.isin(active_candidate_ids),
        ["candidate_id", "position_time_utc", "lat", "lon"],
    ].copy()
    known.sort_values(["candidate_id", "position_time_utc"], kind="stable", inplace=True)
    known_by_candidate = {
        str(candidate_id): block
        for candidate_id, block in known.groupby("candidate_id", sort=False, observed=True)
    }
    summary = summary.set_index("candidate_id", drop=False)
    pieces: list[pd.DataFrame] = []
    for candidate_id, block in active_unknown.groupby("candidate_id", sort=False, observed=True):
        candidate_id = str(candidate_id)
        positions = known_by_candidate.get(candidate_id)
        if positions is None or positions.empty:
            raise ValueError(f"bounded segment has no known endpoint: {candidate_id}")
        times = positions.position_time_utc.astype("int64").to_numpy(dtype=np.int64)
        coords = positions[["lat", "lon"]].to_numpy(dtype=np.float64)
        event_ns = block.position_time_utc.astype("int64").to_numpy(dtype=np.int64)
        right = np.searchsorted(times, event_ns, side="left")
        left = right - 1
        has_left = left >= 0
        has_right = right < len(times)
        if not (has_left | has_right).all():
            raise ValueError(f"bounded segment has no temporally adjacent endpoint: {candidate_id}")
        left_safe = np.clip(left, 0, len(times) - 1)
        right_safe = np.clip(right, 0, len(times) - 1)
        resolved = block[["_segment_id", "candidate_id", "position_time_utc", "segment_weight_hours"]].copy()
        resolved["position_time_ns"] = event_ns
        resolved["left_time_ns"] = np.where(has_left, times[left_safe], -1).astype(np.int64)
        resolved["right_time_ns"] = np.where(has_right, times[right_safe], -1).astype(np.int64)
        resolved["left_lat"] = np.where(has_left, coords[left_safe, 0], np.nan)
        resolved["left_lon"] = np.where(has_left, coords[left_safe, 1], np.nan)
        resolved["right_lat"] = np.where(has_right, coords[right_safe, 0], np.nan)
        resolved["right_lon"] = np.where(has_right, coords[right_safe, 1], np.nan)
        resolved["endpoint_case"] = np.where(has_left & has_right, "two_sided", "one_sided")
        meta = summary.loc[candidate_id]
        for column in GROUP_COLUMNS:
            resolved[column] = meta[column]
        pieces.append(resolved)
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if result._segment_id.duplicated().any():
        raise ValueError("active segment endpoint resolution produced duplicate segment IDs")
    return result


def _accumulate_combo(target: dict[tuple[object, ...], dict[str, Any]], row: Any) -> None:
    key = tuple(getattr(row, column) for column in GROUP_COLUMNS)
    item = target.setdefault(key, {"segments": 0, "unknown_weight_hours": 0.0})
    item["segments"] += 1
    item["unknown_weight_hours"] += float(row.segment_weight_hours)


def process_shard(
    shard: str,
    cyclones,
) -> dict[str, Any]:
    """Scan one formal shard; exact geometry is called only after TC prefilter."""
    summary_path = TRACK_ROOT / "run/shards" / shard / "candidate_summary.parquet"
    summary = pq.read_table(
        summary_path,
        columns=["candidate_id", *GROUP_COLUMNS],
    ).to_pandas()
    summary["candidate_id"] = summary.candidate_id.astype(str)
    frame = PROBE.read_shard(shard)
    frame["candidate_id"] = frame.candidate_id.astype(str)
    unknown_mask = frame.track_evidence.eq("unresolved").to_numpy()
    known_mask = (
        frame.track_evidence.ne("unresolved").to_numpy()
        & np.isfinite(frame.lat.to_numpy(dtype=np.float64))
        & np.isfinite(frame.lon.to_numpy(dtype=np.float64))
    )
    unknown = frame.loc[
        unknown_mask,
        ["candidate_id", "position_time_utc", "segment_weight_hours"],
    ].copy()
    unknown_rows = len(unknown)
    unknown_hours = float(unknown.segment_weight_hours.sum())
    unknown_ids = set(unknown.candidate_id.unique())
    known_ids = set(frame.loc[known_mask, "candidate_id"].unique())
    unbounded_ids = unknown_ids.difference(known_ids)
    bounded_mask = ~unknown.candidate_id.isin(unbounded_ids)
    bounded = unknown.loc[bounded_mask].copy()
    unbounded = unknown.loc[~bounded_mask].copy()

    unbounded_group: list[dict[str, Any]] = []
    if not unbounded.empty:
        by_candidate = unbounded.groupby("candidate_id", sort=False, observed=True).segment_weight_hours.sum()
        by_candidate = by_candidate.rename("unbounded_unknown_hours").reset_index()
        by_candidate = by_candidate.merge(summary, on="candidate_id", how="left", validate="one_to_one")
        grouped = by_candidate.groupby(
            GROUP_COLUMNS, dropna=False, sort=False, observed=True
        ).agg(
            unbounded_voyages=("candidate_id", "nunique"),
            unbounded_unknown_hours=("unbounded_unknown_hours", "sum"),
        ).reset_index()
        unbounded_group = grouped.to_dict("records")

    geometry_stats: dict[str, Any] = {
        "weather_rows": int(len(frame)),
        "unknown_segments": int(unknown_rows),
        "unknown_weight_hours": unknown_hours,
        "bounded_segments": int(len(bounded)),
        "bounded_weight_hours": float(bounded.segment_weight_hours.sum()),
        "unbounded_segments": int(len(unbounded)),
        "unbounded_voyages": int(len(unbounded_ids)),
        "unbounded_weight_hours": float(unbounded.segment_weight_hours.sum()),
        "unbounded_by_combo": unbounded_group,
        "active_tc_segments": 0,
        "necessary_tc300_segments": 0,
        "necessary_tc500_segments": 0,
        "exact_geometry_segments": 0,
        "endpoint_speed_infeasible_segments": 0,
        "exact_tc300_segments": 0,
        "exact_tc500_segments": 0,
        "exact_geometry_calls": 0,
        "tc300_possible_by_combo": {},
        "tc500_possible_by_combo": {},
    }
    if bounded.empty:
        return geometry_stats

    bounded_times = bounded.position_time_utc.drop_duplicates().tolist()
    storms = cyclones.positions_at_times(bounded_times)
    if storms.empty:
        return geometry_stats
    storms = storms[["target_time_utc", "storm_lat", "storm_lon"]].rename(
        columns={"target_time_utc": "position_time_utc"}
    )
    active_times = storms.position_time_utc.drop_duplicates()
    active_unknown = bounded.loc[bounded.position_time_utc.isin(active_times)].copy()
    geometry_stats["active_tc_segments"] = int(len(active_unknown))
    if active_unknown.empty:
        return geometry_stats

    resolved = _prepare_active_segments(active_unknown, frame, known_mask, summary)
    joined = resolved.merge(storms, on="position_time_utc", how="inner", sort=False)
    if joined.empty:
        return geometry_stats

    left_ns = joined.left_time_ns.to_numpy(dtype=np.int64)
    right_ns = joined.right_time_ns.to_numpy(dtype=np.int64)
    event_ns = joined.position_time_ns.to_numpy(dtype=np.int64)
    storm_lat = joined.storm_lat.to_numpy(dtype=np.float64)
    storm_lon = joined.storm_lon.to_numpy(dtype=np.float64)
    has_left = left_ns >= 0
    has_right = right_ns >= 0
    left_ok_300 = np.ones(len(joined), dtype=bool)
    left_ok_500 = np.ones(len(joined), dtype=bool)
    right_ok_300 = np.ones(len(joined), dtype=bool)
    right_ok_500 = np.ones(len(joined), dtype=bool)
    speed_km_h = SPEED_KN * KM_PER_NAUTICAL_MILE
    if has_left.any():
        dt_left = (event_ns[has_left] - left_ns[has_left]) / NS_PER_HOUR
        if (dt_left < -1e-12).any():
            raise ValueError("active event precedes its left endpoint")
        d_left = LEGACY.great_circle_km(
            joined.left_lat.to_numpy(dtype=np.float64)[has_left],
            joined.left_lon.to_numpy(dtype=np.float64)[has_left],
            storm_lat[has_left],
            storm_lon[has_left],
        )
        left_ok_300[has_left] = d_left <= 300.0 + speed_km_h * np.maximum(dt_left, 0.0) + 1e-9
        left_ok_500[has_left] = d_left <= 500.0 + speed_km_h * np.maximum(dt_left, 0.0) + 1e-9
    if has_right.any():
        dt_right = (right_ns[has_right] - event_ns[has_right]) / NS_PER_HOUR
        if (dt_right < -1e-12).any():
            raise ValueError("active event follows its right endpoint")
        d_right = LEGACY.great_circle_km(
            joined.right_lat.to_numpy(dtype=np.float64)[has_right],
            joined.right_lon.to_numpy(dtype=np.float64)[has_right],
            storm_lat[has_right],
            storm_lon[has_right],
        )
        right_ok_300[has_right] = d_right <= 300.0 + speed_km_h * np.maximum(dt_right, 0.0) + 1e-9
        right_ok_500[has_right] = d_right <= 500.0 + speed_km_h * np.maximum(dt_right, 0.0) + 1e-9
    necessary_300 = left_ok_300 & right_ok_300
    necessary_500 = left_ok_500 & right_ok_500

    segment_ids = joined._segment_id.to_numpy(dtype=np.int64)
    order = np.argsort(segment_ids, kind="stable")
    sorted_ids = segment_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    geometry_stats["necessary_tc300_segments"] = int(
        np.logical_or.reduceat(necessary_300[order], starts).sum()
    )
    group_necessary_500 = np.logical_or.reduceat(necessary_500[order], starts)
    geometry_stats["necessary_tc500_segments"] = int(group_necessary_500.sum())

    possible300: dict[tuple[object, ...], dict[str, Any]] = {}
    possible500: dict[tuple[object, ...], dict[str, Any]] = {}
    exact_count_300 = 0
    exact_count_500 = 0
    infeasible_count = 0
    exact_segments = 0
    exact_calls = 0
    for group_number, start in enumerate(starts):
        if not group_necessary_500[group_number]:
            continue
        end = starts[group_number + 1] if group_number + 1 < len(starts) else len(order)
        rows_ix = order[start:end]
        rows_ix = rows_ix[necessary_500[rows_ix]]
        first = joined.iloc[int(rows_ix[0])]
        row = first
        exact_segments += 1
        if not _endpoint_speed_feasible(row):
            infeasible_count += 1
            continue
        event_row = row
        found_300 = False
        found_500 = False
        for row_ix in rows_ix:
            storm_row = joined.iloc[int(row_ix)]
            if not found_500:
                exact_calls += 1
                if spherical_caps_intersect(
                    _segment_caps(event_row, float(storm_row.storm_lat), float(storm_row.storm_lon), 500.0)
                ):
                    found_500 = True
            if necessary_300[int(row_ix)] and not found_300:
                exact_calls += 1
                if spherical_caps_intersect(
                    _segment_caps(event_row, float(storm_row.storm_lat), float(storm_row.storm_lon), 300.0)
                ):
                    found_300 = True
            if found_300 and found_500:
                break
        if found_300 and not found_500:
            raise AssertionError("300 km exact upper cannot exceed the 500 km exact upper")
        if found_300 or found_500:
            if found_300:
                exact_count_300 += 1
                _accumulate_combo(possible300, event_row)
            if found_500:
                exact_count_500 += 1
                _accumulate_combo(possible500, event_row)

    geometry_stats.update({
        "exact_geometry_segments": int(exact_segments),
        "endpoint_speed_infeasible_segments": int(infeasible_count),
        "exact_tc300_segments": int(exact_count_300),
        "exact_tc500_segments": int(exact_count_500),
        "exact_geometry_calls": int(exact_calls),
        "tc300_possible_by_combo": possible300,
        "tc500_possible_by_combo": possible500,
    })
    return geometry_stats


def _merge_combo_stats(u1: pd.DataFrame, run_stats: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows300 = [
        {**dict(zip(GROUP_COLUMNS, key)), "tc300_bounded_possible_segments": v["segments"],
         "tc300_bounded_possible_unknown_hours": v["unknown_weight_hours"]}
        for key, v in run_stats["tc300_possible_by_combo"].items()
    ]
    rows500 = [
        {**dict(zip(GROUP_COLUMNS, key)), "tc500_bounded_possible_segments": v["segments"],
         "tc500_bounded_possible_unknown_hours": v["unknown_weight_hours"]}
        for key, v in run_stats["tc500_possible_by_combo"].items()
    ]
    unbounded_rows = run_stats["unbounded_by_combo"]
    additions = u1.copy()
    for rows in (rows300, rows500, unbounded_rows):
        if not rows:
            continue
        additions = additions.merge(
            pd.DataFrame(rows), on=GROUP_COLUMNS, how="left", validate="one_to_one"
        )
    zero_columns = [
        "tc300_bounded_possible_segments", "tc300_bounded_possible_unknown_hours",
        "tc500_bounded_possible_segments", "tc500_bounded_possible_unknown_hours",
        "unbounded_voyages", "unbounded_unknown_hours",
    ]
    for column in zero_columns:
        if column not in additions:
            additions[column] = 0
        additions[column] = additions[column].fillna(0)
    additions["unbounded_voyages"] = additions.unbounded_voyages.astype("int64")
    additions["tc300_hours_lower"] = additions.tc300_hours_observed.astype("float64")
    additions["tc500_hours_lower"] = additions.tc500_hours_observed.astype("float64")
    # Spatially unbounded voyages contribute their full unknown duration as a
    # conservative exposure upper; their count and hours remain separately shown.
    additions["tc300_hours_upper"] = (
        additions.tc300_hours_lower
        + additions.tc300_bounded_possible_unknown_hours
        + additions.unbounded_unknown_hours
    )
    additions["tc500_hours_upper"] = (
        additions.tc500_hours_lower
        + additions.tc500_bounded_possible_unknown_hours
        + additions.unbounded_unknown_hours
    )
    additions["tc300_bound_width_hours"] = additions.tc300_hours_upper - additions.tc300_hours_lower
    additions["tc500_bound_width_hours"] = additions.tc500_hours_upper - additions.tc500_hours_lower
    failures = {
        "tc300_lower_exceeds_upper": int((additions.tc300_hours_lower > additions.tc300_hours_upper + 1e-9).sum()),
        "tc500_lower_exceeds_upper": int((additions.tc500_hours_lower > additions.tc500_hours_upper + 1e-9).sum()),
        "tc300_upper_below_lower_exact": int((additions.tc300_hours_upper + 1e-9 < additions.tc300_hours_lower).sum()),
        "tc500_upper_below_lower_exact": int((additions.tc500_hours_upper + 1e-9 < additions.tc500_hours_lower).sum()),
        "tc300_upper_exceeds_tc500_upper": int((additions.tc300_hours_upper > additions.tc500_hours_upper + 1e-9).sum()),
    }
    if any(failures.values()):
        raise ValueError(f"U2 combination bound checks failed: {failures}")
    if additions["tc300_hours_upper"].isna().any() or additions["tc500_hours_upper"].isna().any():
        raise ValueError("U2 TC upper columns must be populated for every combination")
    widths = {}
    for radius in (300, 500):
        width = additions[f"tc{radius}_bound_width_hours"].to_numpy(dtype=np.float64)
        widths[str(radius)] = {
            "combination_count": int(len(width)),
            "zero_width_combinations": int(np.isclose(width, 0.0, atol=1e-9).sum()),
            "mean_hours": float(np.mean(width)),
            "median_hours": float(np.median(width)),
            "p90_hours": float(np.quantile(width, 0.90)),
            "p95_hours": float(np.quantile(width, 0.95)),
            "maximum_hours": float(np.max(width)),
            "total_lower_hours": float(additions[f"tc{radius}_hours_lower"].sum()),
            "total_upper_hours": float(additions[f"tc{radius}_hours_upper"].sum()),
        }
    return additions, {
        "checks": failures,
        "width_distribution_by_radius": widths,
        "unbounded_voyage_combinations": int(additions.unbounded_voyages.gt(0).sum()),
        "unbounded_voyages": int(additions.unbounded_voyages.sum()),
        "unbounded_unknown_hours": float(additions.unbounded_unknown_hours.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-shards", default="", help="comma-separated shards for a resource probe only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    track_shards = sorted((TRACK_ROOT / "run/shards").glob("shard_*/candidate_summary.parquet"))
    if len(track_shards) != 253:
        raise SystemExit(f"expected 253 adopted track shards, found {len(track_shards)}")
    fixes, ibtracs_audit = PROBE.load_ibtracs_snapshot(
        PROBE.CONFIG["ibtracs_path"], expected_sha256=PROBE.CONFIG["ibtracs_sha256"]
    )
    cyclones = PROBE.CycloneTrackIndex(fixes, maximum_fix_gap_hours=3.0)
    start = time.perf_counter()

    if args.probe_shards:
        selected = [x.strip() for x in args.probe_shards.split(",") if x.strip()]
        by_name = {p.parent.name: p for p in track_shards}
        missing = sorted(set(selected) - set(by_name))
        if missing:
            raise SystemExit(f"unknown probe shards: {missing}")
        probe_rows = []
        for shard in selected:
            before = time.perf_counter()
            result = process_shard(shard, cyclones)
            elapsed = time.perf_counter() - before
            probe_rows.append({
                "shard": shard,
                "elapsed_wall_seconds": elapsed,
                "weather_rows": result["weather_rows"],
                "bounded_segments": result["bounded_segments"],
                "active_tc_segments": result["active_tc_segments"],
                "necessary_tc500_segments": result["necessary_tc500_segments"],
                "exact_geometry_segments": result["exact_geometry_segments"],
                "exact_geometry_calls": result["exact_geometry_calls"],
            })
            print(json.dumps(probe_rows[-1]), flush=True)
        rows = sum(x["weather_rows"] for x in probe_rows)
        seconds = sum(x["elapsed_wall_seconds"] for x in probe_rows)
        projected_hours = (seconds / max(rows, 1) * EXPECTED_WEATHER_ROWS) / 3600.0 * 1.5
        report = {
            "probe_only": True,
            "shards": probe_rows,
            "probe_weather_rows": rows,
            "probe_wall_seconds": seconds,
            "linear_projection_hours": projected_hours / 1.5,
            "with_50pct_headroom_hours": projected_hours,
            "required_limit_hours": 24,
            "passes_runtime_gate": projected_hours <= 24,
            "ibtracs_sha256": ibtracs_audit["sha256"],
        }
        (args.out / "tc_upper_resource_probe.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, default=str))
        return

    combo300: dict[tuple[object, ...], dict[str, Any]] = {}
    combo500: dict[tuple[object, ...], dict[str, Any]] = {}
    unbounded_by_combo: dict[tuple[object, ...], dict[str, Any]] = {}
    totals: dict[str, Any] = defaultdict(float)
    for number, summary_path in enumerate(track_shards, start=1):
        shard = summary_path.parent.name
        before = time.perf_counter()
        result = process_shard(shard, cyclones)
        totals["weather_rows"] += result["weather_rows"]
        totals["unknown_segments"] += result["unknown_segments"]
        totals["unknown_weight_hours"] += result["unknown_weight_hours"]
        totals["bounded_segments"] += result["bounded_segments"]
        totals["bounded_weight_hours"] += result["bounded_weight_hours"]
        totals["unbounded_segments"] += result["unbounded_segments"]
        totals["unbounded_voyages"] += result["unbounded_voyages"]
        totals["unbounded_weight_hours"] += result["unbounded_weight_hours"]
        for name in (
            "active_tc_segments", "necessary_tc300_segments", "necessary_tc500_segments",
            "exact_geometry_segments", "endpoint_speed_infeasible_segments",
            "exact_tc300_segments", "exact_tc500_segments", "exact_geometry_calls",
        ):
            totals[name] += result[name]
        for key, value in result["tc300_possible_by_combo"].items():
            item = combo300.setdefault(key, {"segments": 0, "unknown_weight_hours": 0.0})
            item["segments"] += value["segments"]
            item["unknown_weight_hours"] += value["unknown_weight_hours"]
        for key, value in result["tc500_possible_by_combo"].items():
            item = combo500.setdefault(key, {"segments": 0, "unknown_weight_hours": 0.0})
            item["segments"] += value["segments"]
            item["unknown_weight_hours"] += value["unknown_weight_hours"]
        for row in result["unbounded_by_combo"]:
            key = tuple(row[column] for column in GROUP_COLUMNS)
            item = unbounded_by_combo.setdefault(
                key, {"unbounded_voyages": 0, "unbounded_unknown_hours": 0.0}
            )
            item["unbounded_voyages"] += int(row["unbounded_voyages"])
            item["unbounded_unknown_hours"] += float(row["unbounded_unknown_hours"])
        if number % 10 == 0 or number == len(track_shards):
            print(
                f"U2 TC upper {number:03d}/253 {shard}: rows={result['weather_rows']:,}; "
                f"active={result['active_tc_segments']:,}; "
                f"necessary500={result['necessary_tc500_segments']:,}; "
                f"exact500={result['exact_tc500_segments']:,}; "
                f"elapsed={time.perf_counter() - before:.1f}s",
                flush=True,
            )

    expected = {
        "weather_rows": EXPECTED_WEATHER_ROWS,
        "unknown_segments": EXPECTED_UNKNOWN_ROWS,
        "bounded_segments": EXPECTED_BOUNDED_ROWS,
        "unbounded_segments": EXPECTED_UNBOUNDED_ROWS,
    }
    for key, value in expected.items():
        if int(totals[key]) != value:
            raise ValueError(f"U2 census {key} mismatch: {totals[key]} != {value}")
    if abs(float(totals["unknown_weight_hours"]) - EXPECTED_UNKNOWN_HOURS) > 1e-4:
        raise ValueError("U2 census unknown-position hours do not close to the U1 product")
    if int(totals["bounded_segments"]) + int(totals["unbounded_segments"]) != int(totals["unknown_segments"]):
        raise ValueError("bounded plus unbounded unknown segment counts do not close")
    if abs(float(totals["bounded_weight_hours"]) + float(totals["unbounded_weight_hours"]) - float(totals["unknown_weight_hours"])) > 1e-4:
        raise ValueError("bounded plus unbounded unknown-position hours do not close")

    u1 = pd.read_parquet(U1_GROUPS)
    combos, combo_report = _merge_combo_stats(u1, {
        "tc300_possible_by_combo": combo300,
        "tc500_possible_by_combo": combo500,
        "unbounded_by_combo": [
            {**dict(zip(GROUP_COLUMNS, key)), **value}
            for key, value in unbounded_by_combo.items()
        ],
    })
    # Preserve an explicit lower <= upper proof for every directed OD × ship-type key.
    lower_checks = {
        "combination_count": int(len(combos)),
        "tc300_lower_le_upper_rows": int(combos.tc300_hours_lower.le(combos.tc300_hours_upper + 1e-9).sum()),
        "tc500_lower_le_upper_rows": int(combos.tc500_hours_lower.le(combos.tc500_hours_upper + 1e-9).sum()),
        "tc300_upper_populated_rows": int(combos.tc300_hours_upper.notna().sum()),
        "tc500_upper_populated_rows": int(combos.tc500_hours_upper.notna().sum()),
    }
    if lower_checks["tc300_lower_le_upper_rows"] != len(combos) or lower_checks["tc500_lower_le_upper_rows"] != len(combos):
        raise ValueError(f"not all combinations satisfy lower <= upper: {lower_checks}")
    combo_path = args.out / "tc_upper_by_od_shiptype.parquet"
    combo_csv = args.out / "tc_upper_by_od_shiptype.csv"
    combos.to_parquet(combo_path, index=False)
    combos.to_csv(combo_csv, index=False)
    unbounded_csv = args.out / "unbounded_voyages_by_od_shiptype.csv"
    combos.loc[
        combos.unbounded_voyages.gt(0),
        [*GROUP_COLUMNS, "unbounded_voyages", "unbounded_unknown_hours"],
    ].to_csv(unbounded_csv, index=False)
    report = {
        "ok": True,
        "year": 2025,
        "speed_cap_kn": SPEED_KN,
        "necessary_filter": "For each storm centre and TC radius, discard only if distance from a known endpoint exceeds R + 25 kn * elapsed time; a two-sided segment must pass both endpoint conditions.",
        "exact_geometry": "Closed spherical-cap intersection: both endpoint travel caps (when present) and the TC radius cap must have a common point on the sphere.",
        "storm_track_inputs": {
            "ibtracs_sha256": ibtracs_audit["sha256"],
            "fixes": cyclones.n_fixes,
            "interpolation_intervals": cyclones.n_intervals,
            "maximum_fix_gap_hours": 3.0,
        },
        "population_closure": {**expected, **{k: int(totals[k]) for k in expected},
            "unknown_weight_hours_expected": EXPECTED_UNKNOWN_HOURS,
            "unknown_weight_hours_actual": float(totals["unknown_weight_hours"]),
            "bounded_weight_hours": float(totals["bounded_weight_hours"]),
            "unbounded_weight_hours": float(totals["unbounded_weight_hours"]),
        },
        "filter_and_geometry_counts": {
            name: int(totals[name]) for name in (
                "active_tc_segments", "necessary_tc300_segments", "necessary_tc500_segments",
                "exact_geometry_segments", "endpoint_speed_infeasible_segments",
                "exact_tc300_segments", "exact_tc500_segments", "exact_geometry_calls",
            )
        },
        "combination_bounds": combo_report,
        "lower_upper_checks": lower_checks,
        "unbounded_policy": "For spatially unbounded no-position voyages, add the complete unknown weighted duration as a conservative TC upper contribution; voyage counts and hours are separately listed.",
        "outputs": {
            "combination_parquet": str(combo_path),
            "combination_csv": str(combo_csv),
            "unbounded_by_od_shiptype_csv": str(unbounded_csv),
        },
        "elapsed_wall_seconds": time.perf_counter() - start,
    }
    report_path = args.out / "tc_upper_census_closure.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
