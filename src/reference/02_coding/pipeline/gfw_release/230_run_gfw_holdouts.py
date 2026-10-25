#!/usr/bin/env python3
"""Hide real GFW points, reuse unchanged step-4 production routing, match weather."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import resource
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from step6_core import *
sys.path.insert(0, str(WEATHER / "leg_ais_coverage"))
from weather_core.era5_match import Era5MonthlyMatcher
from weather_core.tropical_cyclone import CycloneTrackIndex, load_ibtracs_snapshot
from weather_core.geo import great_circle_km


def predict_window(sample, observations: pd.DataFrame, candidate, production, route):
    start, end = pd.Timestamp(sample.gap_start_utc), pd.Timestamp(sample.gap_end_utc)
    hidden = observations.loc[(observations.UTC > start) & (observations.UTC < end)].copy()
    visible = observations.loc[~((observations.UTC > start) & (observations.UTC < end))].copy()
    if len(hidden) != sample.truth_observation_count or hidden.empty:
        raise ValueError("heldout observations disagree with the frozen sample list")
    if hidden.UTC.isin(visible.UTC).any() or hidden.UTC_10min.isin(visible.UTC_10min).any():
        raise ValueError("heldout observation leaks into the reconstruction input")
    if np.diff(observations.UTC.astype("int64").to_numpy()).min(initial=NS_HOUR) <= 0:
        raise ValueError("observations are not ordered and unique")
    scoped = SimpleNamespace(candidate_id=sample.candidate_id, vessel_uid_clean=getattr(candidate, SHIP),
                             mmsi=candidate.mmsi, departure_time=start, arrival_time=end,
                             origin_city=candidate.origin_city, dest_city=candidate.dest_city,
                             shiptype_group=candidate.shiptype_group)
    prediction, _, failures = production.build_track(scoped, visible, route, sample.source_shard)
    predicted = prediction[["utc", "lat", "lon", "track_evidence"]].rename(columns={
        "utc": "UTC_10min", "lat": "lat_pred", "lon": "lon_pred", "track_evidence": "prediction_evidence"})
    paired = hidden[["UTC", "UTC_10min", "lat", "lon"]].rename(columns={"lat": "lat_truth", "lon": "lon_truth"})
    paired = paired.merge(predicted, on="UTC_10min", how="left", validate="one_to_one")
    for field in ("sample_id", "candidate_id", "sample_set", "nominal_gap_hours", "shiptype_group"):
        paired[field] = getattr(sample, field)
    outcome = {"sample_id": sample.sample_id, "source_shard": sample.source_shard,
               "sample_set": sample.sample_set, "nominal_gap_hours": sample.nominal_gap_hours,
               "shiptype_group": sample.shiptype_group, "truth_points": len(hidden),
               "predicted_points": int(paired.lat_pred.notna().sum()), "heldout_points_leaked": 0,
               "route_outcomes": [f["status"] for f in failures],
               "status": "predicted" if paired.lat_pred.notna().all() else "prediction_unresolved"}
    return paired, outcome


def run_shard(shard: str, samples_path: str, prepared: str, out: str, pilot_per_stratum: int) -> dict:
    started = time.perf_counter()
    folder = Path(out) / "shards" / shard
    folder.mkdir(parents=True, exist_ok=False)
    samples = pd.read_parquet(samples_path, filters=[("source_shard", "=", shard)])
    if pilot_per_stratum:
        samples = samples.groupby(["sample_set", "nominal_gap_hours", "shiptype_group"], sort=True).head(pilot_per_stratum)
    decision = json.loads(DECISION.read_text())
    weather_root = ROOT / decision["weather_run"]
    source = json.loads((weather_root / "shards" / shard / "status.json").read_text())["sources"]["track"]
    if record(Path(source["path"])) != source:
        raise ValueError("GFW source track changed")
    points = pd.read_parquet(source["path"], columns=["candidate_id", "utc", "observation_time_utc", "lat", "lon", "source_provider", "source_resolution_original"],
        filters=[("candidate_id", "in", list(samples.candidate_id.unique())), ("track_evidence", "=", "observed_ais")])
    if not points.source_provider.fillna("").str.startswith("gfw").all():
        raise ValueError("holdout input includes non-GFW observations")
    points = points.rename(columns={"utc": "UTC_10min", "observation_time_utc": "UTC"})
    by_id = {str(cid): g.sort_values("UTC").reset_index(drop=True) for cid, g in points.groupby("candidate_id", sort=False)}
    panel = pd.read_parquet(Path(prepared) / "voyage_panel.parquet",
                columns=["candidate_id", SHIP, "mmsi", *KEY], filters=[("candidate_id", "in", list(by_id))]).set_index("candidate_id")
    production = module("e006_holdout_production", HERE / "160_build_gfw_ten_minute_tracks.py")
    route = production.router()
    pairs, outcomes = [], []
    for sample in samples.itertuples(index=False):
        observed = by_id[str(sample.candidate_id)]
        start, end = pd.Timestamp(sample.gap_start_utc), pd.Timestamp(sample.gap_end_utc)
        context = observed.loc[(observed.UTC >= start-pd.Timedelta(hours=1)) & (observed.UTC <= end+pd.Timedelta(hours=1))]
        if not context.UTC.iloc[0] == start-pd.Timedelta(hours=1) \
                or not context.UTC.iloc[-1] == end+pd.Timedelta(hours=1) \
                or not np.all(np.diff(context.UTC.astype("int64")) == NS_HOUR):
            raise ValueError("sample lacks uninterrupted hourly observed-GFW context")
        paired, report = predict_window(sample, observed, panel.loc[str(sample.candidate_id)], production, route)
        pairs.append(paired); outcomes.append(report)
    pairs = pd.concat(pairs, ignore_index=True)
    pairs["point_id"] = np.arange(len(pairs))
    pairs["hour_anchor_utc"] = pairs.UTC.dt.floor("h")  # accepted E002 holdout time rule
    locations = []
    for role in ("truth", "pred"):
        block = pairs[["point_id", "UTC", "hour_anchor_utc", f"lat_{role}", f"lon_{role}"]].rename(columns={f"lat_{role}": "lat", f"lon_{role}": "lon"})
        block["role"] = role; locations.append(block)
    positions = pd.concat(locations, ignore_index=True)
    positions["swh_m"] = np.nan
    config = json.loads((WEATHER / "config/2025_ten_minute_weather.json").read_text())
    from weather_assets import resolve_monthly_era5_assets, validate_month_time_grid
    for month, block in positions.groupby(positions.hour_anchor_utc.dt.strftime("%Y-%m"), sort=True):
        year, mon = map(int, month.split("-"))
        with Era5MonthlyMatcher(resolve_monthly_era5_assets(year=year, month=mon),
                               maximum_wet_cell_snap_km=config["maximum_wet_cell_snap_km"]) as matcher:
            validate_month_time_grid(matcher, year, mon); matcher.preload_swh()
            positions.loc[block.index, "swh_m"] = matcher.match(block[["hour_anchor_utc", "lat", "lon"]]).swh_m.to_numpy()
    fixes, _ = load_ibtracs_snapshot(Path(config["ibtracs_path"]), expected_sha256=config["ibtracs_sha256"], time_end_exclusive="2026-01-02T00:00:00Z")
    index = CycloneTrackIndex(fixes, maximum_fix_gap_hours=3)
    positions = positions.rename(columns={"UTC": "position_time_utc"})
    matched = index.match_positions(positions, lat_col="lat", lon_col="lon")
    for role in ("truth", "pred"):
        block = matched.loc[matched.role.eq(role)].sort_values("point_id")
        pairs[f"swh_{role}_m"] = block.swh_m.to_numpy()
        for radius in (300, 500):
            pairs[f"tc{radius}_{role}"] = np.where(block.position_valid.to_numpy(), block[f"tc_within_{radius}km"].to_numpy(), np.nan)
    valid = pairs.lat_pred.notna()
    pairs["position_error_km"] = np.nan
    pairs.loc[valid, "position_error_km"] = great_circle_km(pairs.loc[valid, "lon_truth"].to_numpy(), pairs.loc[valid, "lat_truth"].to_numpy(), pairs.loc[valid, "lon_pred"].to_numpy(), pairs.loc[valid, "lat_pred"].to_numpy())
    pairs["wave_difference_m"] = pairs.swh_pred_m-pairs.swh_truth_m
    for radius in (300, 500):
        pairs[f"tc{radius}_flip"] = np.where(valid, pairs[f"tc{radius}_pred"].ne(pairs[f"tc{radius}_truth"]), np.nan)
    pairs.to_parquet(folder / "point_results.parquet", index=False)
    pd.DataFrame(outcomes).to_parquet(folder / "outcomes.parquet", index=False)
    report = {"state": "completed", "shard": shard, "samples": len(samples), "truth_points": len(pairs),
              "predicted_points": int(valid.sum()), "heldout_points_leaked": 0,
              "source_track": source, "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
              "outputs": {p.name: record(p) for p in folder.glob("*.parquet")}}
    write_json(folder / "status.json", report)
    return report


def reduce_results(out: Path):
    pairs = pd.concat([pd.read_parquet(p) for p in sorted((out / "shards").glob("*/point_results.parquet"))], ignore_index=True)
    outcomes = pd.concat([pd.read_parquet(p) for p in sorted((out / "shards").glob("*/outcomes.parquet"))], ignore_index=True)
    outcomes.to_parquet(out / "all_outcomes.parquet", index=False)
    # Window-level weather errors use exactly the paired, matched hourly truth
    # points. Missing predictions remain explicitly counted in outcomes.
    window_rows = []
    for sid, group in pairs.groupby("sample_id", sort=False):
        info = group.iloc[0]
        row = {"sample_id": sid, "sample_set": info.sample_set, "nominal_gap_hours": info.nominal_gap_hours,
               "shiptype_group": info.shiptype_group, "truth_points": len(group),
               "predicted_points": int(group.lat_pred.notna().sum()),
               "wave_paired_points": int(group.wave_difference_m.notna().sum()),
               "window_wave_mean_difference_m": float(group.wave_difference_m.mean())}
        for radius in (300, 500):
            valid = group[f"tc{radius}_pred"].notna()
            row[f"window_tc{radius}_share_difference_pp"] = (100*float((group.loc[valid, f"tc{radius}_pred"]-group.loc[valid, f"tc{radius}_truth"]).mean()) if valid.any() else np.nan)
        window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    windows.to_parquet(out / "window_results.parquet", index=False)
    metrics = []
    for key, block in pairs.groupby(["sample_set", "nominal_gap_hours", "shiptype_group"], sort=True):
        row = dict(zip(["sample_set", "nominal_gap_hours", "shiptype_group"], key))
        row.update(truth_points=len(block), predicted_points=int(block.lat_pred.notna().sum()), wave_paired_points=int(block.wave_difference_m.notna().sum()))
        for label, values in (("position_error_km", block.position_error_km), ("wave_absolute_difference_m", block.wave_difference_m.abs())):
            for q in (.5, .9, .95): row[f"{label}_p{int(q*100)}"] = float(values.quantile(q))
        for radius in (300, 500): row[f"tc{radius}_flip_rate"] = float(block[f"tc{radius}_flip"].mean())
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(out / "metrics_by_stratum.csv", index=False, lineterminator="\n")
    summaries = []
    for key, block in windows.groupby(["sample_set", "nominal_gap_hours"], sort=True):
        row = dict(zip(["sample_set", "nominal_gap_hours"], key)); row["windows"] = len(block)
        row["fully_predicted_windows"] = int(block.predicted_points.eq(block.truth_points).sum())
        for metric in ("window_wave_mean_difference_m", "window_tc300_share_difference_pp", "window_tc500_share_difference_pp"):
            row[f"{metric}_p90_abs"] = float(block[metric].abs().quantile(.9))
            row[f"{metric}_available_windows"] = int(block[metric].notna().sum())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out / "window_error_summary.csv", index=False, lineterminator="\n")
    return {"samples": len(outcomes), "truth_points": len(pairs), "predicted_points": int(pairs.lat_pred.notna().sum()),
            "heldout_points_leaked": int(outcomes.heldout_points_leaked.sum()), "unresolved_windows": int(outcomes.status.ne("predicted").sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shards", nargs="+", default=SHARDS)
    parser.add_argument("--pilot-per-stratum", type=int, default=0)
    args = parser.parse_args(); source = binding()
    out = inside_runs(args.out); out.mkdir(parents=True, exist_ok=False)
    state = {"state": "running", "completed": [], "failures": [], "workers": args.workers,
             "sample_manifest": record(args.samples), "authorization": source["authorization"], "pilot_per_stratum": args.pilot_per_stratum}
    state.update(provenance([HERE / "step6_core.py", Path(__file__), HERE / "220_prepare_gfw_holdouts.py",
                            HERE / "160_build_gfw_ten_minute_tracks.py", DECISION]))
    write_json(out / "status.json", state); started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(run_shard, s, str(args.samples), str(args.prepared), str(out), args.pilot_per_stratum): s for s in args.shards}
        for future in as_completed(tasks):
            shard = tasks[future]
            try:
                result = future.result(); state["completed"].append(shard)
                print(json.dumps(result), flush=True)
            except Exception:
                state["failures"].append({"shard": shard, "traceback": traceback.format_exc()})
            write_json(out / "status.json", state)
    if state["failures"]:
        state["state"] = "failed"; write_json(out / "status.json", state); raise RuntimeError(state["failures"])
    state.update(reduce_results(out), state="completed", elapsed_seconds=time.perf_counter()-started)
    write_json(out / "status.json", state)


if __name__ == "__main__":
    main()
