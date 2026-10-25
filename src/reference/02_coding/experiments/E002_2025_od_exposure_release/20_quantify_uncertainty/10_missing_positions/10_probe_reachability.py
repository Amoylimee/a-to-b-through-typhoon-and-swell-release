#!/usr/bin/env python3
"""Probe the migrated reachable-area method on formal 2025 weather segments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CODING_ROOT = Path(__file__).resolve().parents[4]
PRODUCT_ROOT = CODING_ROOT / "data/processed/products"
TRACK_ROOT = PRODUCT_ROOT / "companion_tracks/2025"
WEATHER_ROOT = PRODUCT_ROOT / "voyage_weather_facts/2025"
WEATHER_STAGE = CODING_ROOT / "pipeline/40_construct_variables/60_construct_voyage_weather_facts"
LEGACY = WEATHER_STAGE / "leg_ais_coverage/30_bound_dark_hour_wave_exposure.py"
LEGACY_DIR = LEGACY.parent
CONFIG = json.loads((WEATHER_STAGE / "config/2025_ten_minute_weather.json").read_text())
ENV_STAGE = CODING_ROOT / "pipeline/00_environment_and_config"
ROUTE_STAGE = CODING_ROOT / "pipeline/40_construct_variables/50_reconstruct_companion_tracks"
SEED = 20260923
SPEED_KN = 25.0
KM_PER_NAUTICAL_MILE = 1.852
EXPECTED_WEATHER_ROWS = 174_615_386
EXPECTED_UNRESOLVED_WEIGHT_HOURS = 16_744_981.166388888
SPEED_BOUND_CONFIG_PATH = LEGACY_DIR / "config/legacy_speed_bound.json"
SPEED_BOUND_CONFIG = json.loads(SPEED_BOUND_CONFIG_PATH.read_text())

for entry in (WEATHER_STAGE, LEGACY_DIR, ENV_STAGE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

spec = importlib.util.spec_from_file_location("legacy_bounds", LEGACY)
legacy = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
legacy.MAX_CELLS_PER_HOUR = 300_000  # complete 0.5-degree global grid, no thinning

from weather_core.era5_match import Era5MonthlyMatcher  # noqa: E402
from weather_core.tropical_cyclone import (  # noqa: E402
    CycloneTrackIndex,
    load_ibtracs_snapshot,
)
from weather_assets import resolve_monthly_era5_assets, validate_month_time_grid  # noqa: E402


def read_shard(shard: str) -> pd.DataFrame:
    shard_dir = WEATHER_ROOT / "run/shards" / shard
    pieces = []
    cols = [
        "candidate_id", "hour_anchor_utc", "segment_start_utc", "segment_end_utc",
        "position_time_utc", "segment_weight_hours", "lat", "lon", "track_evidence",
    ]
    for path in sorted(shard_dir.glob("weather_2025-??.parquet")):
        pieces.append(pq.read_table(path, columns=cols).to_pandas())
    frame = pd.concat(pieces, ignore_index=True)
    for column in ("hour_anchor_utc", "position_time_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.reset_index(drop=True)


def collect_probe_rows(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    reservoir: list[dict[str, object]] = []
    seen = 0
    for candidate_id, group in frame.groupby("candidate_id", sort=False, observed=True):
        group = group.reset_index(drop=True)
        known = (
            group.track_evidence.ne("unresolved")
            & np.isfinite(group.lat.to_numpy(dtype=float))
            & np.isfinite(group.lon.to_numpy(dtype=float))
        )
        positions = group.loc[known, ["position_time_utc", "lat", "lon"]].copy()
        if positions.empty:
            continue
        positions.sort_values("position_time_utc", kind="stable", inplace=True)
        times = positions.position_time_utc.astype("int64").to_numpy(dtype=np.int64)
        coords = positions[["lat", "lon"]].to_numpy(dtype=np.float64)
        cap_kn = legacy.voyage_speed_cap(
            coords[:, 0], coords[:, 1], times,
            legacy.SpeedBoundDefinition.from_config(SPEED_BOUND_CONFIG),
        )
        missing = group.loc[group.track_evidence.eq("unresolved")]
        for row in missing.itertuples(index=False):
            event_ns = int(row.position_time_utc.value)
            at = int(np.searchsorted(times, event_ns, side="left"))
            left = at - 1 if at else None
            right = at if at < len(times) else None
            if left is None and right is None:
                continue
            # An interior segment gets both endpoints; voyage-end segments are one-sided.
            record: dict[str, object] = {
                "candidate_id": str(candidate_id),
                "hour_anchor_utc": row.hour_anchor_utc,
                "position_time_utc": row.position_time_utc,
                "segment_weight_hours": float(row.segment_weight_hours),
                "left_time_ns": int(times[left]) if left is not None else None,
                "right_time_ns": int(times[right]) if right is not None else None,
                "left_lat": float(coords[left, 0]) if left is not None else None,
                "left_lon": float(coords[left, 1]) if left is not None else None,
                "right_lat": float(coords[right, 0]) if right is not None else None,
                "right_lon": float(coords[right, 1]) if right is not None else None,
                "endpoint_case": "two_sided" if left is not None and right is not None else "one_sided",
                "voyage_speed_cap_sensitivity_kn": cap_kn,
            }
            seen += 1
            if len(reservoir) < n:
                reservoir.append(record)
            else:
                replacement = int(rng.integers(seen))
                if replacement < n:
                    reservoir[replacement] = record
    return pd.DataFrame(reservoir)


def _enumerate_bounded_segments(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    source_shard: str,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, float], int, float]:
    """Build per-hour unknown-segment geometry inputs from one formal shard."""
    event_frames: list[pd.DataFrame] = []
    bounded_count: dict[str, int] = defaultdict(int)
    bounded_hours: dict[str, float] = defaultdict(float)
    unbounded_candidates = 0
    unbounded_hours = 0.0
    for candidate_id, group in frame.groupby("candidate_id", sort=False, observed=True):
        group = group.reset_index(drop=True)
        unknown = group.loc[group.track_evidence.eq("unresolved")].copy()
        if unknown.empty:
            continue
        known = group.loc[
            group.track_evidence.ne("unresolved")
            & np.isfinite(group.lat.to_numpy(dtype=float))
            & np.isfinite(group.lon.to_numpy(dtype=float)),
            ["position_time_utc", "lat", "lon"],
        ].sort_values("position_time_utc", kind="stable")
        if known.empty:
            unbounded_candidates += 1
            unbounded_hours += float(unknown.segment_weight_hours.sum())
            continue
        times = known.position_time_utc.astype("int64").to_numpy(dtype=np.int64)
        coords = known[["lat", "lon"]].to_numpy(dtype=np.float64)
        cap_kn = legacy.voyage_speed_cap(
            coords[:, 0], coords[:, 1], times,
            legacy.SpeedBoundDefinition.from_config(SPEED_BOUND_CONFIG),
        )
        event_ns = unknown.position_time_utc.astype("int64").to_numpy(dtype=np.int64)
        right = np.searchsorted(times, event_ns, side="left")
        left = right - 1
        has_left = left >= 0
        has_right = right < len(times)
        if not (has_left | has_right).any():
            unbounded_candidates += 1
            unbounded_hours += float(unknown.segment_weight_hours.sum())
            continue
        valid = has_left | has_right
        selected = unknown.iloc[np.flatnonzero(valid)].copy()
        left_selected = left[valid]
        right_selected = right[valid]
        left_valid = has_left[valid]
        right_valid = has_right[valid]
        event_ns = event_ns[valid]
        selected["candidate_id"] = str(candidate_id)
        selected["sample_month"] = selected.hour_anchor_utc.dt.strftime("%Y-%m")
        selected["source_shard"] = source_shard
        selected["left_time_ns"] = np.where(left_valid, times[np.maximum(left_selected, 0)], np.nan)
        selected["right_time_ns"] = np.where(right_valid, times[np.minimum(right_selected, len(times) - 1)], np.nan)
        selected["left_lat"] = np.where(left_valid, coords[np.maximum(left_selected, 0), 0], np.nan)
        selected["left_lon"] = np.where(left_valid, coords[np.maximum(left_selected, 0), 1], np.nan)
        selected["right_lat"] = np.where(right_valid, coords[np.minimum(right_selected, len(times) - 1), 0], np.nan)
        selected["right_lon"] = np.where(right_valid, coords[np.minimum(right_selected, len(times) - 1), 1], np.nan)
        selected["endpoint_case"] = np.where(left_valid & right_valid, "two_sided", "one_sided")
        selected["voyage_speed_cap_sensitivity_kn"] = cap_kn
        selected["position_time_utc"] = pd.to_datetime(event_ns, utc=True)
        meta = summary.loc[str(candidate_id)]
        for column in ("origin_city", "dest_city", "shiptype_group"):
            selected[column] = meta[column]
        event_frames.append(selected[[
            "candidate_id", "source_shard", "sample_month", "hour_anchor_utc",
            "position_time_utc", "segment_weight_hours", "left_time_ns", "right_time_ns",
            "left_lat", "left_lon", "right_lat", "right_lon", "endpoint_case",
            "voyage_speed_cap_sensitivity_kn", "origin_city", "dest_city", "shiptype_group",
        ]])
        for month, rows in selected.groupby("sample_month", sort=False, observed=True):
            bounded_count[str(month)] += int(len(rows))
            bounded_hours[str(month)] += float(rows.segment_weight_hours.sum())
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return events, dict(bounded_count), dict(bounded_hours), unbounded_candidates, unbounded_hours


def _resolve_sample_endpoints(
    sample_rows: pd.DataFrame,
    frame: pd.DataFrame,
    source_shard: str,
) -> pd.DataFrame:
    """Resolve endpoints and legacy sensitivity caps only for retained sample rows."""
    if sample_rows.empty:
        return sample_rows.copy()
    known_mask = (
        frame.track_evidence.ne("unresolved")
        & np.isfinite(frame.lat.to_numpy(dtype=float))
        & np.isfinite(frame.lon.to_numpy(dtype=float))
    )
    known = frame.loc[
        known_mask & frame.candidate_id.astype(str).isin(sample_rows.candidate_id.astype(str).unique()),
        ["candidate_id", "position_time_utc", "lat", "lon"],
    ].copy()
    known["candidate_id"] = known.candidate_id.astype(str)
    known.sort_values(["candidate_id", "position_time_utc"], kind="stable", inplace=True)
    by_candidate = {
        str(candidate_id): group
        for candidate_id, group in known.groupby("candidate_id", sort=False, observed=True)
    }
    result = sample_rows.copy()
    for column in (
        "left_time_ns", "right_time_ns", "left_lat", "left_lon",
        "right_lat", "right_lon", "voyage_speed_cap_sensitivity_kn",
    ):
        if column not in result:
            result[column] = np.nan
    if "endpoint_case" not in result:
        result["endpoint_case"] = pd.Series(index=result.index, dtype="object")
    if "endpoints_resolved" not in result:
        result["endpoints_resolved"] = pd.Series(False, index=result.index, dtype=bool)
    for candidate_id, rows in result.groupby("candidate_id", sort=False, observed=True):
        positions = by_candidate.get(str(candidate_id))
        if positions is None or positions.empty:
            raise ValueError(f"sampled bounded candidate has no positioned endpoints: {candidate_id}")
        times = positions.position_time_utc.astype("int64").to_numpy(dtype=np.int64)
        coords = positions[["lat", "lon"]].to_numpy(dtype=np.float64)
        cap_kn = legacy.voyage_speed_cap(
            coords[:, 0], coords[:, 1], times,
            legacy.SpeedBoundDefinition.from_config(SPEED_BOUND_CONFIG),
        )
        event_ns = rows.position_time_utc.astype("int64").to_numpy(dtype=np.int64)
        right = np.searchsorted(times, event_ns, side="left")
        left = right - 1
        has_left = left >= 0
        has_right = right < len(times)
        if not (has_left | has_right).all():
            raise ValueError(f"sampled segment has no positioned endpoint: {candidate_id}")
        left_safe = np.clip(left, 0, len(times) - 1)
        right_safe = np.clip(right, 0, len(times) - 1)
        result.loc[rows.index, "left_time_ns"] = np.where(has_left, times[left_safe], np.nan)
        result.loc[rows.index, "right_time_ns"] = np.where(has_right, times[right_safe], np.nan)
        result.loc[rows.index, "left_lat"] = np.where(has_left, coords[left_safe, 0], np.nan)
        result.loc[rows.index, "left_lon"] = np.where(has_left, coords[left_safe, 1], np.nan)
        result.loc[rows.index, "right_lat"] = np.where(has_right, coords[right_safe, 0], np.nan)
        result.loc[rows.index, "right_lon"] = np.where(has_right, coords[right_safe, 1], np.nan)
        result.loc[rows.index, "endpoint_case"] = np.where(
            has_left & has_right, "two_sided", "one_sided"
        )
        result.loc[rows.index, "voyage_speed_cap_sensitivity_kn"] = cap_kn
        result.loc[rows.index, "endpoints_resolved"] = True
        result.loc[rows.index, "source_shard"] = source_shard
    return result


def scan_all_formal_shards(per_month: int) -> tuple[pd.DataFrame, dict[str, object]]:
    track_shards = sorted((TRACK_ROOT / "run/shards").glob("shard_*/candidate_summary.parquet"))
    if len(track_shards) != 253:
        raise ValueError(f"expected 253 adopted track shards, found {len(track_shards)}")
    selected_by_month: dict[str, pd.DataFrame] = {}
    all_segment_rows: dict[str, int] = defaultdict(int)
    all_segment_hours: dict[str, float] = defaultdict(float)
    bounded_segment_rows: dict[str, int] = defaultdict(int)
    bounded_segment_hours: dict[str, float] = defaultdict(float)
    unbounded_candidate_ids: set[str] = set()
    unbounded_segment_rows = 0
    unbounded_by_group: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"voyages": 0, "unknown_weight_hours": 0.0}
    )
    unbounded_hours = 0.0
    total_weather_rows = 0
    for shard_number, summary_path in enumerate(track_shards, start=1):
        source_shard = summary_path.parent.name
        summary = pq.read_table(
            summary_path,
            columns=["candidate_id", "origin_city", "dest_city", "shiptype_group"],
        ).to_pandas()
        summary["candidate_id"] = summary.candidate_id.astype(str)
        summary.set_index("candidate_id", drop=False, inplace=True)
        frame = read_shard(source_shard)
        total_weather_rows += int(len(frame))
        unknown = frame.loc[
            frame.track_evidence.eq("unresolved"),
            ["candidate_id", "hour_anchor_utc", "position_time_utc", "segment_weight_hours"],
        ].copy()
        unknown["candidate_id"] = unknown.candidate_id.astype(str)
        unknown["sample_month"] = unknown.hour_anchor_utc.dt.strftime("%Y-%m")
        for month, rows in unknown.groupby("sample_month", sort=False, observed=True):
            all_segment_rows[str(month)] += int(len(rows))
            all_segment_hours[str(month)] += float(rows.segment_weight_hours.sum())
        known_mask = (
            frame.track_evidence.ne("unresolved")
            & np.isfinite(frame.lat.to_numpy(dtype=float))
            & np.isfinite(frame.lon.to_numpy(dtype=float))
        )
        known_candidate_ids = set(frame.loc[known_mask, "candidate_id"].astype(str).unique())
        bounded = unknown.loc[unknown.candidate_id.isin(known_candidate_ids)].copy()
        unbounded = unknown.loc[~unknown.candidate_id.isin(known_candidate_ids)].copy()
        for month, rows in bounded.groupby("sample_month", sort=False, observed=True):
            bounded_segment_rows[str(month)] += int(len(rows))
            bounded_segment_hours[str(month)] += float(rows.segment_weight_hours.sum())
        if not unbounded.empty:
            candidates_without_positions = set(unbounded.candidate_id.unique())
            unbounded_candidate_ids.update(candidates_without_positions)
            unbounded_segment_rows += int(len(unbounded))
            unbounded_hours += float(unbounded.segment_weight_hours.sum())
            no_position_by_candidate = unbounded.groupby(
                "candidate_id", sort=False, observed=True
            ).segment_weight_hours.sum().rename("unknown_weight_hours").reset_index()
            group_meta = summary.loc[
                sorted(candidates_without_positions),
                ["origin_city", "dest_city", "shiptype_group"],
            ].reset_index()
            no_position_by_group = no_position_by_candidate.merge(
                group_meta, on="candidate_id", how="left", validate="one_to_one"
            ).groupby(
                ["origin_city", "dest_city", "shiptype_group"],
                dropna=False,
                sort=False,
                observed=True,
            ).agg(
                voyages=("candidate_id", "nunique"),
                unknown_weight_hours=("unknown_weight_hours", "sum"),
            ).reset_index()
            for row in no_position_by_group.itertuples(index=False):
                key = (str(row.origin_city), str(row.dest_city), str(row.shiptype_group))
                unbounded_by_group[key]["voyages"] += int(row.voyages)
                unbounded_by_group[key]["unknown_weight_hours"] += float(row.unknown_weight_hours)
        if not bounded.empty:
            bounded["sample_priority"] = (
                pd.util.hash_pandas_object(
                    bounded[["candidate_id", "position_time_utc"]].astype("string"),
                    index=False,
                    hash_key="E002-20260923-01",
                ).to_numpy(dtype=np.uint64)
                ^ np.uint64(SEED)
            )
            bounded["source_shard"] = source_shard
            for month, block in bounded.groupby("sample_month", sort=False, observed=True):
                local = block.nsmallest(per_month, "sample_priority").merge(
                    summary.reset_index(drop=True)[
                        ["candidate_id", "origin_city", "dest_city", "shiptype_group"]
                    ],
                    on="candidate_id",
                    how="left",
                    validate="many_to_one",
                )
                prior = selected_by_month.get(str(month))
                merged = local if prior is None else pd.concat([prior, local], ignore_index=True)
                selected = merged.nsmallest(per_month, "sample_priority").reset_index(drop=True)
                endpoint_columns = [
                    "left_time_ns", "right_time_ns", "left_lat", "left_lon",
                    "right_lat", "right_lon", "endpoint_case",
                    "voyage_speed_cap_sensitivity_kn", "endpoints_resolved",
                ]
                for column in endpoint_columns:
                    if column not in selected:
                        if column == "endpoint_case":
                            selected[column] = pd.Series(index=selected.index, dtype="object")
                        elif column == "endpoints_resolved":
                            selected[column] = pd.Series(False, index=selected.index, dtype=bool)
                        else:
                            selected[column] = np.nan
                selected["endpoints_resolved"] = (
                    selected.endpoints_resolved.astype("boolean").fillna(False).astype(bool)
                )
                pending = selected.loc[~selected.endpoints_resolved].copy()
                if not pending.empty:
                    resolved = _resolve_sample_endpoints(pending, frame, source_shard)
                    selected.loc[resolved.index, endpoint_columns] = resolved[endpoint_columns]
                selected_by_month[str(month)] = selected
        print(
            f"U2 scan {shard_number:03d}/253 {source_shard}: weather_rows={len(frame):,}; "
            f"unknown_segments={len(unknown):,}; bounded_sample_pool={len(bounded):,}",
            flush=True,
        )
    all_unknown_rows = sum(all_segment_rows.values())
    bounded_rows = sum(bounded_segment_rows.values())
    if total_weather_rows != EXPECTED_WEATHER_ROWS:
        raise ValueError(
            f"adopted weather row count mismatch: {total_weather_rows:,} "
            f"!= {EXPECTED_WEATHER_ROWS:,}"
        )
    if bounded_rows + unbounded_segment_rows != all_unknown_rows:
        raise ValueError(
            "unresolved segment classification does not close: "
            f"bounded={bounded_rows:,}, unbounded={unbounded_segment_rows:,}, "
            f"all={all_unknown_rows:,}"
        )
    if abs(sum(all_segment_hours.values()) - EXPECTED_UNRESOLVED_WEIGHT_HOURS) > 1e-4:
        raise ValueError("weighted unresolved hours do not close to the adopted U1 total")
    if abs(
        sum(bounded_segment_hours.values()) + unbounded_hours
        - sum(all_segment_hours.values())
    ) > 1e-4:
        raise ValueError("bounded plus unbounded unknown-position hours do not close")
    sample = pd.concat(selected_by_month.values(), ignore_index=True)
    if not sample.endpoints_resolved.fillna(False).all():
        raise ValueError("one or more selected bounded segments lacks resolved endpoints")
    sample["sample_weight"] = sample.sample_month.map({
        month: bounded_segment_rows[month] / max(int((sample.sample_month == month).sum()), 1)
        for month in bounded_segment_rows
    })
    population = {
        "total_weather_rows": total_weather_rows,
        "all_unresolved_segment_rows": all_unknown_rows,
        "unknown_segments_by_month": dict(all_segment_rows),
        "unknown_weight_hours_by_month": dict(all_segment_hours),
        "bounded_segments_by_month": dict(bounded_segment_rows),
        "bounded_weight_hours_by_month": dict(bounded_segment_hours),
        "unbounded_candidate_count": len(unbounded_candidate_ids),
        "unbounded_segment_rows": unbounded_segment_rows,
        "unbounded_weight_hours": unbounded_hours,
        "unbounded_by_od_shiptype": [
            {
                "origin_city": key[0],
                "dest_city": key[1],
                "shiptype_group": key[2],
                **value,
            }
            for key, value in sorted(unbounded_by_group.items())
        ],
        "per_month_sample_target": per_month,
        "sample_count_by_month": sample.sample_month.value_counts().sort_index().to_dict(),
    }
    return sample, population


def _reachable(
    row: pd.Series, speed_cap_kn: float = SPEED_KN
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    h_ns = int(pd.Timestamp(row.position_time_utc).value)
    speed_km_h = float(speed_cap_kn) * KM_PER_NAUTICAL_MILE
    if row.endpoint_case == "two_sided":
        t0, t1 = int(row.left_time_ns), int(row.right_time_ns)
        lat0, lon0 = float(row.left_lat), float(row.left_lon)
        lat1, lon1 = float(row.right_lat), float(row.right_lon)
        if not t0 < h_ns < t1:
            raise ValueError("two-sided unknown segment midpoint falls outside its endpoints")
        duration_h = (t1 - t0) / 3.6e12
        direct_km = float(legacy.great_circle_km(lat0, lon0, lat1, lon1))
        if direct_km > speed_km_h * duration_h + 1e-6:
            return np.empty(0), np.empty(0), False, "speed_infeasible"
        radius_a = speed_km_h * ((h_ns - t0) / 3.6e12)
        radius_b = speed_km_h * ((t1 - h_ns) / 3.6e12)
        lat, lon, capped = legacy.reachable_cells(
            lat0, lon0, lat1, lon1, radius_a, radius_b
        )
        return lat, lon, capped, "two_sided"

    if row.left_time_ns is not None and pd.notna(row.left_time_ns):
        t, lat0, lon0 = int(row.left_time_ns), float(row.left_lat), float(row.left_lon)
        radius = speed_km_h * max(0.0, (h_ns - t) / 3.6e12)
    else:
        t, lat0, lon0 = int(row.right_time_ns), float(row.right_lat), float(row.right_lon)
        radius = speed_km_h * max(0.0, (t - h_ns) / 3.6e12)
    lat, lon, capped = legacy.reachable_cells(lat0, lon0, lat0, lon0, radius, radius)
    return lat, lon, capped, "one_sided"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default="shard_0000")
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument(
        "--full-year-sample-per-month", type=int, default=0,
        help="scan all adopted 2025 shards and sample this many bounded unknown segments per month",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if args.full_year_sample_per_month > 0:
        sample, population = scan_all_formal_shards(args.full_year_sample_per_month)
        sample_mode = "seeded_monthly_reservoir_from_all_253_adopted_weather_shards"
        unbounded_table = pd.DataFrame(
            population.pop("unbounded_by_od_shiptype", [])
        )
        unbounded_table_path = args.out / "unbounded_voyages_by_od_shiptype.csv"
        unbounded_table.to_csv(unbounded_table_path, index=False)
        population["unbounded_by_od_shiptype_csv"] = str(unbounded_table_path)
    else:
        frame = read_shard(args.shard)
        sample = collect_probe_rows(frame, args.sample_size)
        sample_mode = "single_shard_reservoir_probe_only"
        summary_path = TRACK_ROOT / "run/shards" / args.shard / "candidate_summary.parquet"
        summary = pq.read_table(
            summary_path,
            columns=["candidate_id", "origin_city", "dest_city", "shiptype_group"],
        ).to_pandas()
        summary["candidate_id"] = summary.candidate_id.astype(str)
        sample["candidate_id"] = sample.candidate_id.astype(str)
        sample = sample.merge(summary, on="candidate_id", how="left", validate="many_to_one")
        sample["source_shard"] = args.shard
        sample["sample_month"] = sample.hour_anchor_utc.dt.strftime("%Y-%m")
        sample["sample_weight"] = 1.0
        population = {
            "sampling_mode": "single_shard_reservoir_probe_only",
            "source_shard": args.shard,
            "sample_weight_is_population_expansion": False,
        }
        del frame
    if sample.empty:
        raise SystemExit("no bounded unresolved weather segments in selected sample frame")
    sample = sample.loc[sample.position_time_utc.dt.year.eq(2025)].copy()
    sample.to_parquet(args.out / "unknown_segments_sample_internal.parquet", index=False)

    fixes, ibtracs_audit = load_ibtracs_snapshot(
        CONFIG["ibtracs_path"], expected_sha256=CONFIG["ibtracs_sha256"]
    )
    cyclones = CycloneTrackIndex(fixes, maximum_fix_gap_hours=3.0)

    values: list[dict[str, object]] = []
    wave_preload_bytes = 0
    monthly_audit: list[dict[str, object]] = []
    for month, month_sample in sample.groupby("sample_month", sort=True, observed=True):
        year, mon = (int(x) for x in str(month).split("-"))
        matcher = Era5MonthlyMatcher(
            resolve_monthly_era5_assets(year=year, month=mon),
            maximum_wet_cell_snap_km=0.0,
        )
        try:
            validate_month_time_grid(matcher, year, mon)
            preloaded = matcher.preload_swh()
            wave_preload_bytes = max(wave_preload_bytes, preloaded)
            monthly_audit.append({
                "month": str(month),
                "sample_rows": int(len(month_sample)),
                "swh_resident_bytes": int(preloaded),
            })
            print(
                f"{month}: sample_rows={len(month_sample):,}; "
                f"preloaded_swh={preloaded / 1e9:.2f} GB",
                flush=True,
            )
            for index, row in month_sample.iterrows():
                sensitivity_cap = float(row.get("voyage_speed_cap_sensitivity_kn", SPEED_KN))
                cases = (("fixed_25_kn_primary", SPEED_KN), ("per_voyage_cap_sensitivity", sensitivity_cap))
                for speed_case, speed_cap in cases:
                    lat, lon, capped, endpoint_case = _reachable(row, speed_cap)
                    record: dict[str, object] = {
                        "sample_index": int(index),
                        "candidate_id": str(row.candidate_id),
                        "source_shard": str(row.source_shard),
                        "origin_city": row.origin_city,
                        "dest_city": row.dest_city,
                        "shiptype_group": row.shiptype_group,
                        "sample_month": str(month),
                        "sample_weight": float(row.sample_weight),
                        "speed_case": speed_case,
                        "speed_cap_kn": float(speed_cap),
                        "endpoint_case": endpoint_case,
                        "hour_anchor_utc": pd.Timestamp(row.hour_anchor_utc).isoformat(),
                        "position_time_utc": pd.Timestamp(row.position_time_utc).isoformat(),
                        "weight_hours": float(row.segment_weight_hours),
                        "reachable_grid_cells": int(len(lat)),
                        "cells_capped": bool(capped),
                    }
                    if endpoint_case == "speed_infeasible" or len(lat) == 0:
                        record["geometry_status"] = "speed_infeasible"
                        values.append(record)
                        continue
                    if len(lat) > legacy.MAX_CELLS_PER_HOUR:
                        raise AssertionError(
                            "complete ERA5 grid exceeds the no-thinning cell limit"
                        )
                    positions = pd.DataFrame({
                        "hour_anchor_utc": [row.hour_anchor_utc] * len(lat),
                        "lat": lat,
                        "lon": lon,
                    })
                    match_start = time.perf_counter()
                    matched = matcher.match(positions)
                    wet = matched.loc[matched.swh_m.notna()].copy()
                    storm_positions = cyclones.positions_at_times([row.position_time_utc])
                    tc300_possible = tc500_possible = tc300_certain = tc500_certain = False
                    if not wet.empty and not storm_positions.empty:
                        within300 = np.zeros(len(wet), dtype=bool)
                        within500 = np.zeros(len(wet), dtype=bool)
                        for storm in storm_positions.itertuples(index=False):
                            distance = legacy.great_circle_km(
                                wet.lat.to_numpy(dtype=np.float64),
                                wet.lon.to_numpy(dtype=np.float64),
                                float(storm.storm_lat),
                                float(storm.storm_lon),
                            )
                            within300 |= distance <= 300.0 + 1e-9
                            within500 |= distance <= 500.0 + 1e-9
                        tc300_possible, tc500_possible = bool(within300.any()), bool(within500.any())
                        tc300_certain, tc500_certain = bool(within300.all()), bool(within500.all())
                    elapsed = time.perf_counter() - match_start
                    record.update({
                        "geometry_status": "matched" if not wet.empty else "no_wet_era5_cells",
                        "wet_cells_with_swh": int(len(wet)),
                        "swh_min_m": float(wet.swh_m.min()) if not wet.empty else np.nan,
                        "swh_max_m": float(wet.swh_m.max()) if not wet.empty else np.nan,
                        "swh_width_m": float(wet.swh_m.max() - wet.swh_m.min()) if not wet.empty else np.nan,
                        "tc300_possible": tc300_possible,
                        "tc300_certain": tc300_certain,
                        "tc500_possible": tc500_possible,
                        "tc500_certain": tc500_certain,
                        "match_seconds": elapsed,
                    })
                    values.append(record)
        finally:
            matcher.close()
    result = pd.DataFrame(values)
    if result.empty:
        raise SystemExit("sampled unknown segments produced no geometry records")
    result.to_csv(args.out / "reachable_bounds_probe.csv", index=False)

    matched_rows = result.loc[result.geometry_status.eq("matched")].copy()
    if not matched_rows.empty:
        matched_rows["weighted_hours"] = (
            matched_rows.sample_weight * matched_rows.weight_hours
        )
        matched_rows["weighted_swh_lo"] = matched_rows.weighted_hours * matched_rows.swh_min_m
        matched_rows["weighted_swh_hi"] = matched_rows.weighted_hours * matched_rows.swh_max_m
        for radius in (300, 500):
            matched_rows[f"weighted_tc{radius}_possible_hours"] = (
                matched_rows.weighted_hours * matched_rows[f"tc{radius}_possible"].astype(float)
            )
            matched_rows[f"weighted_tc{radius}_certain_hours"] = (
                matched_rows.weighted_hours * matched_rows[f"tc{radius}_certain"].astype(float)
            )
        sampled_group = matched_rows.groupby(
            ["speed_case", "origin_city", "dest_city", "shiptype_group"],
            dropna=False,
            sort=True,
            observed=True,
        ).agg(
            sampled_segments=("candidate_id", "size"),
            effective_unknown_weight_hours=("weighted_hours", "sum"),
            sampled_weighted_swh_lo_integral=("weighted_swh_lo", "sum"),
            sampled_weighted_swh_hi_integral=("weighted_swh_hi", "sum"),
            tc300_possible_unknown_hours=("weighted_tc300_possible_hours", "sum"),
            tc300_certain_unknown_hours=("weighted_tc300_certain_hours", "sum"),
            tc500_possible_unknown_hours=("weighted_tc500_possible_hours", "sum"),
            tc500_certain_unknown_hours=("weighted_tc500_certain_hours", "sum"),
        ).reset_index()
        denominator = sampled_group.effective_unknown_weight_hours.replace(0, np.nan)
        sampled_group["sampled_swh_lower_mean_m"] = sampled_group.sampled_weighted_swh_lo_integral / denominator
        sampled_group["sampled_swh_upper_mean_m"] = sampled_group.sampled_weighted_swh_hi_integral / denominator
        sampled_group.to_csv(args.out / "sampled_bounds_by_od_shiptype.csv", index=False)
    else:
        sampled_group = pd.DataFrame()

    summary_by_speed_case: dict[str, dict[str, object]] = {}
    for speed_case, block in matched_rows.groupby("speed_case", sort=True, observed=True):
        summary_by_speed_case[str(speed_case)] = {
            "matched_segments": int(len(block)),
            "endpoint_case_counts": block.endpoint_case.value_counts().to_dict(),
            "reachable_cell_count_median": float(block.reachable_grid_cells.median()),
            "reachable_cell_count_p90": float(block.reachable_grid_cells.quantile(.9)),
            "swh_width_median": float(block.swh_width_m.median()),
            "swh_width_p90": float(block.swh_width_m.quantile(.9)),
            "tc500_possible_share": float(block.tc500_possible.mean()),
            "tc500_certain_share": float(block.tc500_certain.mean()),
            "capped_count": int(block.cells_capped.sum()),
        }
    wall = time.perf_counter() - start
    report = {
        "ok": True,
        "formal_products_only": True,
        "sampling_mode": sample_mode,
        "source_shard": args.shard if args.full_year_sample_per_month <= 0 else None,
        "sample_size_requested": args.sample_size if args.full_year_sample_per_month <= 0 else None,
        "sample_per_month_requested": args.full_year_sample_per_month if args.full_year_sample_per_month > 0 else None,
        "sample_size_selected": int(len(sample)),
        "sample_size_matched": int(len(matched_rows)),
        "geometry_status_counts": result.geometry_status.value_counts().to_dict(),
        "summary_by_speed_case": summary_by_speed_case,
        "population_scan": population,
        "seed": SEED,
        "speed_cap_kn_primary": SPEED_KN,
        "speed_cap_sensitivity": "legacy p95 x margin clipped to configured floor/ceiling, based on positioned product rows; never used for the primary case",
        "speed_cap_config_sha256": hashlib.sha256(SPEED_BOUND_CONFIG_PATH.read_bytes()).hexdigest(),
        "geometry_source": str(LEGACY),
        "geometry_source_sha256": hashlib.sha256(LEGACY.read_bytes()).hexdigest(),
        "cell_thinning_limit_override": 300_000,
        "meaning_of_cell_thinning_limit_override": "above complete 361x720 0.5-degree global grid; no reachable region is thinned",
        "ibtracs_sha256": CONFIG["ibtracs_sha256"],
        "ibtracs_audit": ibtracs_audit,
        "elapsed_wall_seconds": wall,
        "matched_rows": len(matched_rows),
        "preloaded_swh_resident_bytes": int(wave_preload_bytes),
        "monthly_asset_audit": monthly_audit,
        "output_csv": str(args.out / "reachable_bounds_probe.csv"),
        "sampled_od_shiptype_csv": str(args.out / "sampled_bounds_by_od_shiptype.csv") if not sampled_group.empty else None,
    }
    (args.out / "reachable_bounds_probe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
