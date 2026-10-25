#!/usr/bin/env python3
"""Run E002 holdouts with unchanged E001 production interpolation functions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CODING_ROOT = Path(__file__).resolve().parents[4]
PRODUCT_ROOT = CODING_ROOT / "data/processed/products"
TRACK_ROOT = PRODUCT_ROOT / "companion_tracks/2025"
WEATHER_STAGE = CODING_ROOT / "pipeline/40_construct_variables/60_construct_voyage_weather_facts"
WEATHER_LEGACY_DIR = WEATHER_STAGE / "leg_ais_coverage"
TRACK_STAGE = CODING_ROOT / "pipeline/40_construct_variables/50_reconstruct_companion_tracks"
ROUTE_STAGE = TRACK_STAGE / "sea_path"
ENV_STAGE = CODING_ROOT / "pipeline/00_environment_and_config"
CONFIG = json.loads((WEATHER_STAGE / "config/2025_ten_minute_weather.json").read_text())
MASK_PATH = CODING_ROOT / "data/frozen_inputs/external/gshhg/ocean_mask_01m.npy"

for entry in (TRACK_STAGE, WEATHER_STAGE, WEATHER_LEGACY_DIR, ENV_STAGE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from candidate_full_ten_minute import build_full_candidate_grid  # noqa: E402
from sea_path.ocean_mask import load_ocean_mask  # noqa: E402
from sea_path.routing import DEFAULT_MAX_SPEED_KNOTS, DEFAULT_ROUTE_TIMEOUT_S, SeaPathRouter  # noqa: E402
from track_core.e001_cached_ten_minute import reconstruct_gap  # noqa: E402
from weather_core.era5_match import Era5MonthlyMatcher  # noqa: E402
from weather_core.tropical_cyclone import CycloneTrackIndex, load_ibtracs_snapshot  # noqa: E402
from weather_assets import resolve_monthly_era5_assets, validate_month_time_grid  # noqa: E402


EMPTY_GAPS = pd.DataFrame(columns=["candidate_id", "gap_id", "gap_class", "gap_start", "gap_end"])
EMPTY_PATCH = pd.DataFrame(columns=["candidate_id", "gap_id", "utc", "lat", "lon", "track_evidence", "anchor_source"])
EMPTY_ANCHORS = pd.DataFrame(columns=["hour_utc", "target_hour_id", "anchor_lat", "anchor_lon"])
METRICS = {
    "position_error_km": ("median", "p90", "p95"),
    "wave_signed_difference_m": ("mean", "median", "p90_abs"),
    "tc300_flip_rate": ("mean",),
    "tc500_flip_rate": ("mean",),
}


def load_candidates_for_shard(shard: str, candidate_ids: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_path = TRACK_ROOT / "run/shards" / shard / "candidate_summary.parquet"
    summary = pq.read_table(
        summary_path,
        columns=["candidate_id", "mmsi", "vessel_uid_clean", "layer", "departure_time", "arrival_time"],
        filters=[("candidate_id", "in", candidate_ids)],
    ).to_pandas()
    if summary.candidate_id.astype(str).duplicated().any():
        raise ValueError(f"duplicate candidate summary keys in {shard}")
    track_path = TRACK_ROOT / "run/shards" / shard / "ten_minute_complete_candidate.parquet"
    points = pq.read_table(
        track_path,
        columns=["candidate_id", "mmsi", "vessel_uid_clean", "utc", "lat", "lon", "observation_time_utc", "track_evidence"],
        filters=[("candidate_id", "in", candidate_ids), ("track_evidence", "=", "observed_ais")],
    ).to_pandas()
    points["candidate_id"] = points.candidate_id.astype(str)
    points["utc"] = pd.to_datetime(points.utc, utc=True)
    points["observation_time_utc"] = pd.to_datetime(points.observation_time_utc, utc=True)
    result = {
        cid: group.sort_values("observation_time_utc", kind="stable").reset_index(drop=True)
        for cid, group in points.groupby("candidate_id", sort=False, observed=True)
    }
    return summary.set_index(summary.candidate_id.astype(str), drop=False), result


def _select_pilot(samples: pd.DataFrame, per_stratum: int) -> pd.DataFrame:
    def priority(row: pd.Series, label: str) -> str:
        key = "|".join(
            [
                "20260923",
                label,
                str(row.candidate_id),
                str(row.gap_start_utc),
                str(row.gap_end_utc),
            ]
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    main = samples.loc[samples.sample_set.eq("gap_length_x_shiptype")]
    selected_parts = []
    for _, group in main.groupby(
        ["nominal_gap_hours", "shiptype_group"], sort=True, observed=True
    ):
        ranked = group.copy()
        ranked["_pilot_priority"] = ranked.apply(
            lambda row: priority(row, "main"), axis=1
        )
        selected_parts.append(
            ranked.sort_values("_pilot_priority", kind="stable").head(per_stratum).drop(
                columns="_pilot_priority"
            )
        )
    storm = samples.loc[samples.sample_set.eq("tc500km_additional")].copy()
    if not storm.empty:
        storm["_pilot_priority"] = storm.apply(
            lambda row: priority(row, "storm"), axis=1
        )
        storm = storm.sort_values("_pilot_priority", kind="stable").head(50).drop(
            columns="_pilot_priority"
        )
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else main.iloc[0:0]
    return pd.concat([selected, storm], ignore_index=True).drop_duplicates(
        ["candidate_id", "gap_start_utc", "gap_end_utc"]
    )


def _prediction_for_sample(sample: pd.Series, candidate: pd.Series, observed: pd.DataFrame, router: SeaPathRouter | None) -> tuple[pd.DataFrame, dict[str, object]]:
    start = pd.Timestamp(sample.gap_start_utc).tz_convert("UTC")
    end = pd.Timestamp(sample.gap_end_utc).tz_convert("UTC")
    raw = observed.copy()
    raw.rename(columns={"observation_time_utc": "UTC", "utc": "UTC_10min"}, inplace=True)
    raw["UTC_10min"] = pd.to_datetime(raw.UTC_10min, utc=True)
    raw["UTC"] = pd.to_datetime(raw.UTC, utc=True)
    hidden = raw.loc[(raw.UTC > start) & (raw.UTC < end)].copy()
    if hidden.empty:
        raise ValueError("selected interval contains no hidden observed AIS rows")
    input_points = raw.loc[~((raw.UTC > start) & (raw.UTC < end))].copy()
    leaked = input_points.loc[(input_points.UTC > start) & (input_points.UTC < end)]
    if not leaked.empty or hidden.UTC.isin(input_points.UTC).any():
        raise AssertionError("held-out observation timestamps leaked into interpolation input")
    if hidden.UTC_10min.isin(input_points.UTC_10min).any():
        raise AssertionError("held-out observation bins leaked through another input point")

    nominal = int(sample.nominal_gap_hours)
    actual_gap_h = float(sample.actual_gap_hours)
    if nominal <= 4:
        if actual_gap_h > 4.0 + 1e-12:
            raise ValueError("short interpolation holdout exceeds the 4-hour production limit")
        candidate_arg = {
            "candidate_id": str(sample.candidate_id),
            "vessel_uid_clean": str(candidate.vessel_uid_clean),
            "mmsi": str(candidate.mmsi),
            "layer": str(candidate.layer),
            "source_shard": str(sample.source_shard),
            "departure_time": start,
            "arrival_time": end,
        }
        grid, _ = build_full_candidate_grid(
            candidate_arg, input_points, EMPTY_GAPS, EMPTY_PATCH, EMPTY_PATCH
        )
        predicted = grid[["utc", "lat", "lon", "track_evidence"]].copy()
        route_status = "short_ais_interpolation"
    else:
        if router is None:
            raise RuntimeError("sea-route router is missing for a >4-hour holdout")
        gap = SimpleNamespace(
            gap_id=f"E002-{sample.sample_id}",
            candidate_id=str(sample.candidate_id),
            mmsi=str(candidate.mmsi),
            layer=str(candidate.layer),
            gap_start=start,
            gap_end=end,
            left_lat=float(input_points.loc[input_points.UTC.eq(start), "lat"].iloc[0]),
            left_lon=float(input_points.loc[input_points.UTC.eq(start), "lon"].iloc[0]),
            right_lat=float(input_points.loc[input_points.UTC.eq(end), "lat"].iloc[0]),
            right_lon=float(input_points.loc[input_points.UTC.eq(end), "lon"].iloc[0]),
            gap_hours=actual_gap_h,
        )
        track, route_report, _, _ = reconstruct_gap(router, gap, EMPTY_ANCHORS)
        predicted = track[["utc", "lat", "lon", "track_evidence"]].copy()
        route_status = str(route_report.get("gap_outcome", "unknown"))

    hidden["candidate_id"] = str(sample.candidate_id)
    hidden["time_utc"] = hidden.UTC
    hidden["utc_bin"] = hidden.UTC_10min
    predictions = predicted.rename(columns={"utc": "utc_bin", "track_evidence": "prediction_evidence"})
    paired = hidden[["candidate_id", "time_utc", "utc_bin", "lat", "lon"]].merge(
        predictions[["utc_bin", "lat", "lon", "prediction_evidence"]],
        on="utc_bin", how="left", suffixes=("_truth", "_pred"), validate="one_to_one",
    )
    paired["sample_id"] = str(sample.sample_id)
    paired["sample_set"] = str(sample.sample_set)
    paired["shiptype_group"] = str(sample.shiptype_group)
    paired["nominal_gap_hours"] = nominal
    paired["actual_gap_hours"] = actual_gap_h
    paired["route_status"] = route_status
    return paired, {
        "sample_id": str(sample.sample_id),
        "candidate_id": str(sample.candidate_id),
        "sample_set": str(sample.sample_set),
        "shiptype_group": str(sample.shiptype_group),
        "nominal_gap_hours": nominal,
        "actual_gap_hours": actual_gap_h,
        "heldout_truth_points": int(len(hidden)),
        "paired_prediction_points": int(paired.lat_pred.notna().sum()),
        "route_status": route_status,
        "input_points": int(len(input_points)),
        "heldout_points_leaked": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot-per-stratum", type=int, default=0, help="if positive, run this many sample rows per gap length x ship type stratum plus a small storm subset")
    parser.add_argument("--route-init-only", action="store_true", help="initialize the frozen route stack but do not execute sample windows")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(args.sample_list)
    if args.pilot_per_stratum:
        samples = _select_pilot(samples, args.pilot_per_stratum)
    if samples.empty:
        raise SystemExit("no holdout rows selected")

    long_needed = samples.nominal_gap_hours.gt(4).any()
    router = None
    route_init_seconds = 0.0
    if long_needed:
        init_start = time.perf_counter()
        ocean = load_ocean_mask(MASK_PATH, verify=True)
        router = SeaPathRouter(ocean, verify_inputs=False)
        route_init_seconds = time.perf_counter() - init_start
        del ocean
    if args.route_init_only:
        report = {
            "ok": True,
            "route_router_initialized": router is not None,
            "route_initializer_seconds": route_init_seconds,
            "route_timeout_seconds": DEFAULT_ROUTE_TIMEOUT_S,
            "route_max_speed_knots": DEFAULT_MAX_SPEED_KNOTS,
            "sample_rows_not_run": True,
        }
        (args.out / "route_init_probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    prediction_parts: list[pd.DataFrame] = []
    sample_reports: list[dict[str, object]] = []
    completed_windows = 0
    for shard, shard_samples in samples.groupby("source_shard", sort=True, observed=True):
        candidate_ids = sorted(shard_samples.candidate_id.astype(str).unique())
        summary, points_by_candidate = load_candidates_for_shard(str(shard), candidate_ids)
        for _, sample in shard_samples.iterrows():
            cid = str(sample.candidate_id)
            if cid not in summary.index or cid not in points_by_candidate:
                sample_reports.append({"sample_id": sample.sample_id, "candidate_id": cid, "status": "input_missing"})
                continue
            candidate = summary.loc[cid]
            observed = points_by_candidate[cid].copy()
            task_start = time.perf_counter()
            try:
                paired, report = _prediction_for_sample(sample, candidate, observed, router)
                prediction_parts.append(paired)
                report["wall_seconds"] = time.perf_counter() - task_start
                report["status"] = "predicted" if report["paired_prediction_points"] == report["heldout_truth_points"] else "partial_prediction"
                sample_reports.append(report)
            except Exception as exc:  # keep every selected sample and its failure reason in the denominator
                sample_reports.append({
                    "sample_id": str(sample.sample_id),
                    "candidate_id": cid,
                    "sample_set": str(sample.sample_set),
                    "shiptype_group": str(sample.shiptype_group),
                    "nominal_gap_hours": int(sample.nominal_gap_hours),
                    "actual_gap_hours": float(sample.actual_gap_hours),
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc),
                    "heldout_points_leaked": 0,
                })
            completed_windows += 1
            if completed_windows % 250 == 0:
                print(f"completed {completed_windows:,}/{len(samples):,} selected windows", flush=True)
        print(f"completed source shard {shard}: {len(shard_samples):,} holdout windows", flush=True)

    pairs = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    if not pairs.empty:
        pairs["hour_anchor_utc"] = pd.to_datetime(pairs.time_utc, utc=True).dt.floor("h")
        pairs["point_id"] = np.arange(len(pairs), dtype=np.int64)
        location_rows = pd.concat([
            pairs[["point_id", "sample_id", "candidate_id", "time_utc", "hour_anchor_utc", "lat_truth", "lon_truth"]]
            .rename(columns={"lat_truth": "lat", "lon_truth": "lon"}).assign(location_role="truth"),
            pairs[["point_id", "sample_id", "candidate_id", "time_utc", "hour_anchor_utc", "lat_pred", "lon_pred"]]
            .rename(columns={"lat_pred": "lat", "lon_pred": "lon"}).assign(location_role="prediction"),
        ], ignore_index=True)
        location_rows = location_rows.loc[location_rows.lat.notna() & location_rows.lon.notna()].copy()
        location_rows["hour_anchor_utc"] = pd.to_datetime(location_rows.hour_anchor_utc, utc=True)
        location_rows["wave_m"] = np.nan
        for month, block in location_rows.groupby(location_rows.hour_anchor_utc.dt.strftime("%Y-%m"), sort=True):
            year, mon = (int(value) for value in month.split("-"))
            matcher = Era5MonthlyMatcher(
                resolve_monthly_era5_assets(year=year, month=mon),
                maximum_wet_cell_snap_km=float(CONFIG["maximum_wet_cell_snap_km"]),
            )
            try:
                validate_month_time_grid(matcher, year, mon)
                preload_bytes = matcher.preload_swh()
                print(f"{month}: preloaded {preload_bytes / 1e9:.2f} GB ERA5 SWH", flush=True)
                matched = matcher.match(block[["hour_anchor_utc", "lat", "lon"]])
            finally:
                matcher.close()
            location_rows.loc[block.index, "wave_m"] = matched.swh_m.to_numpy()
        wave_values = location_rows.pivot(index="point_id", columns="location_role", values="wave_m")
        for role in ("truth", "prediction"):
            if role not in wave_values.columns:
                wave_values[role] = np.nan
        pairs = pairs.merge(wave_values.rename(columns={"truth": "swh_truth_m", "prediction": "swh_pred_m"}), on="point_id", how="left", validate="one_to_one")
        loc_tc = location_rows.rename(columns={"lat": "latitude", "lon": "longitude", "time_utc": "position_time_utc"})
        fixes, ibtracs_audit = load_ibtracs_snapshot(
            CONFIG["ibtracs_path"], expected_sha256=CONFIG["ibtracs_sha256"]
        )
        cyclones = CycloneTrackIndex(fixes, maximum_fix_gap_hours=3.0)
        tc = cyclones.match_positions(loc_tc[["point_id", "sample_id", "candidate_id", "position_time_utc", "latitude", "longitude", "location_role"]])
        tc = tc[["point_id", "location_role", "tc_within_300km", "tc_within_500km"]]
        tc = tc.pivot(index="point_id", columns="location_role", values=["tc_within_300km", "tc_within_500km"])
        tc = tc.reindex(columns=pd.MultiIndex.from_product([
            ["tc_within_300km", "tc_within_500km"], ["truth", "prediction"]
        ]))
        tc.columns = [f"{metric}_{role}" for metric, role in tc.columns]
        pairs = pairs.merge(tc.reset_index(), on="point_id", how="left", validate="one_to_one")
        pairs["wave_difference_m"] = pairs.swh_pred_m - pairs.swh_truth_m
        pairs["tc300_flip"] = np.where(
            pairs.tc_within_300km_prediction.notna() & pairs.tc_within_300km_truth.notna(),
            pairs.tc_within_300km_prediction.ne(pairs.tc_within_300km_truth), np.nan,
        )
        pairs["tc500_flip"] = np.where(
            pairs.tc_within_500km_prediction.notna() & pairs.tc_within_500km_truth.notna(),
            pairs.tc_within_500km_prediction.ne(pairs.tc_within_500km_truth), np.nan,
        )
        pairs.to_parquet(args.out / "holdout_point_results.parquet", index=False)
    else:
        ibtracs_audit = {}

    sample_frame = pd.DataFrame(sample_reports)
    sample_frame.to_csv(args.out / "holdout_sample_outcomes.csv", index=False)
    if not pairs.empty:
        valid_position = pairs.dropna(subset=["lat_pred", "lon_pred"]).copy()
        from weather_core.geo import great_circle_km  # import the frozen production distance helper
        valid_position["position_error_km"] = great_circle_km(
            valid_position.lon_truth.to_numpy(), valid_position.lat_truth.to_numpy(),
            valid_position.lon_pred.to_numpy(), valid_position.lat_pred.to_numpy(),
        )
        metrics = (
            valid_position.groupby(["sample_set", "nominal_gap_hours", "shiptype_group"], dropna=False, observed=True)
            .agg(
                truth_points=("candidate_id", "size"),
                position_error_median_km=("position_error_km", "median"),
                position_error_p90_km=("position_error_km", lambda x: x.quantile(.9)),
                position_error_p95_km=("position_error_km", lambda x: x.quantile(.95)),
                matched_wave_points=("wave_difference_m", "count"),
                wave_difference_mean_m=("wave_difference_m", "mean"),
                wave_difference_median_m=("wave_difference_m", "median"),
                wave_difference_p90_abs_m=("wave_difference_m", lambda x: x.abs().quantile(.9)),
                tc300_flip_rate=("tc300_flip", "mean"),
                tc500_flip_rate=("tc500_flip", "mean"),
            ).reset_index()
        )
        metrics.to_csv(args.out / "holdout_metrics_by_stratum.csv", index=False)
    else:
        metrics = pd.DataFrame()

    sample_counts = sample_frame.groupby(["sample_set", "nominal_gap_hours", "shiptype_group"], dropna=False).size().reset_index(name="selected_windows")
    outcome_counts = sample_frame.groupby(["sample_set", "nominal_gap_hours", "shiptype_group", "status"], dropna=False).size().reset_index(name="windows")
    sample_counts.to_csv(args.out / "holdout_selected_counts.csv", index=False)
    outcome_counts.to_csv(args.out / "holdout_outcome_counts.csv", index=False)
    leak_count = int(sample_frame.get("heldout_points_leaked", pd.Series(0, index=sample_frame.index)).fillna(0).astype(int).sum())
    status_counts = sample_frame.status.value_counts(dropna=False).to_dict()
    complete_statuses = {"predicted"}
    paired_counts = sample_frame.get(
        "paired_prediction_points", pd.Series(np.nan, index=sample_frame.index)
    )
    truth_counts = sample_frame.get(
        "heldout_truth_points", pd.Series(np.nan, index=sample_frame.index)
    )
    all_predictions_complete = bool(
        len(sample_frame) == len(samples)
        and sample_frame.status.isin(complete_statuses).all()
        and paired_counts.eq(truth_counts).all()
    )
    report = {
        "ok": bool(len(sample_frame) == len(samples) and leak_count == 0),
        "all_selected_windows_accounted_for": bool(len(sample_frame) == len(samples)),
        "all_predictions_complete": all_predictions_complete,
        "sample_windows_selected": int(len(samples)),
        "sample_windows_with_outcome": int(len(sample_frame)),
        "window_status_counts": status_counts,
        "prediction_point_rows": int(len(pairs)),
        "route_initializer_seconds": route_init_seconds,
        "route_timeout_seconds": DEFAULT_ROUTE_TIMEOUT_S,
        "route_max_speed_knots": DEFAULT_MAX_SPEED_KNOTS,
        "route_interpolation_code": "imported unchanged: track_core.e001_cached_ten_minute.reconstruct_gap",
        "short_interpolation_code": "imported unchanged: candidate_full_ten_minute.build_full_candidate_grid",
        "heldout_observation_leaks": leak_count,
        "ibtracs_sha256": CONFIG["ibtracs_sha256"] if not pairs.empty else None,
        "ibtracs_audit": ibtracs_audit,
        "sample_outcomes_csv": str(args.out / "holdout_sample_outcomes.csv"),
        "metrics_by_stratum_csv": str(args.out / "holdout_metrics_by_stratum.csv") if not metrics.empty else None,
    }
    (args.out / "holdout_closure.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
