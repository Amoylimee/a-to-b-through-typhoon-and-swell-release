#!/usr/bin/env python3
"""GFW census adapter for accepted E002 reachable-area TC/SWH kernels.

Known GFW ends are widened by their hh:30 time-stamp tolerance (E007 rule revision,
`config/step6_rule_revision.json`); port endpoints keep their own 0/1 h rules.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from step6_core import *
from port_constraints import port_ends_usable


def run_shard(shard: str, prepared: str, out: str, probe_candidates: int | None,
              fix_tolerance_hours: float, hour_detail: bool = False,
              selected_candidate_ids: list[str] | None = None) -> dict:
    started = time.perf_counter()
    fix_tolerance_ns = int(round(fix_tolerance_hours * NS_HOUR))
    detail = []
    out = Path(out) / shard
    out.mkdir(parents=True, exist_ok=False)
    kernels = module("e006_bound_kernels", E002 / "20_quantify_uncertainty/10_missing_positions/60_compute_tolerant_bounds.py")
    config = json.loads((WEATHER / "config/2025_ten_minute_weather.json").read_text())
    fixes, _ = kernels.PROBE.load_ibtracs_snapshot(Path(config["ibtracs_path"]),
                    expected_sha256=config["ibtracs_sha256"], time_end_exclusive="2026-01-02T00:00:00Z")
    cyclones = kernels.PROBE.CycloneTrackIndex(fixes, maximum_fix_gap_hours=3)
    ports = pd.read_parquet(Path(prepared) / "port_endpoints.parquet",
                            filters=[("source_shard", "=", shard)]).set_index("candidate_id")
    candidate_ids = sorted(ports.index.astype(str))
    if probe_candidates:
        # Deliberately include voyages with unresolved positions in resource probes.
        panel = pd.read_parquet(Path(prepared) / "voyage_panel.parquet",
             columns=["candidate_id", "source_shard", "unresolved_track_hours"],
             filters=[("source_shard", "=", shard)])
        candidate_ids = sorted(panel.loc[panel.unresolved_track_hours.gt(0), "candidate_id"].astype(str))[:probe_candidates]
    if selected_candidate_ids is not None:
        if not set(selected_candidate_ids).issubset(set(candidate_ids)):
            raise ValueError('candidate replay does not belong to the selected shard')
        candidate_ids = sorted(selected_candidate_ids)
    frame = read_weather(shard, candidates=candidate_ids if probe_candidates or selected_candidate_ids is not None else None)
    unknown = frame.loc[frame.track_evidence.eq("unresolved")].copy()
    # Keep known positions only for voyages whose unknown times need boundaries.
    known = frame.loc[frame.candidate_id.isin(unknown.candidate_id.unique()) & frame.lat.notna(),
                      ["candidate_id", "position_time_utc", "lat", "lon"]]
    positions = {}
    for cid, group in known.groupby("candidate_id", sort=False, observed=True):
        block = group.sort_values("position_time_utc")
        positions[str(cid)] = (block.position_time_utc.astype("int64").to_numpy(), block[["lat", "lon"]].to_numpy(float))
    result = defaultdict(lambda: defaultdict(float))
    progress = {"state": "running", "shard": shard, "probe_candidates": probe_candidates,
                "selected_candidate_ids": selected_candidate_ids,
                "fix_time_tolerance_hours": fix_tolerance_hours,
                "weather_rows": len(frame), "unknown_segments": len(unknown), "months": []}
    write_json(out / "status.json", progress)
    del frame, known
    for month, block in unknown.groupby(unknown.hour_anchor_utc.dt.strftime("%Y-%m"), sort=True):
        t0 = time.perf_counter()
        year, mon = map(int, month.split("-"))
        matcher = kernels.PROBE.Era5MonthlyMatcher(kernels.PROBE.resolve_monthly_era5_assets(year=year, month=mon),
                                                   maximum_wet_cell_snap_km=0, cached_hours=2)
        try:
            kernels.PROBE.validate_month_time_grid(matcher, year, mon)
            matcher.preload_swh()
            storms = cyclones.positions_at_times(block.position_time_utc.drop_duplicates())
            storm_map = {int(pd.Timestamp(t).value): (g.storm_lat.to_numpy(float), g.storm_lon.to_numpy(float))
                         for t, g in storms.groupby("target_time_utc", sort=False)}
            for cid, rows in block.groupby("candidate_id", sort=False, observed=True):
                cid = str(cid)
                port = ports.loc[cid]
                observed = positions.get(cid)
                times, coords = observed if observed is not None else (np.array([], dtype=np.int64), np.empty((0, 2)))
                indexes = np.searchsorted(times, rows.position_time_utc.astype("int64").to_numpy())
                for offset, item in enumerate(rows.itertuples(index=False)):
                    event = int(item.position_time_utc.value)
                    weight = float(item.segment_weight_hours)
                    entry = result[cid]
                    entry["unknown_segments"] += 1
                    entry["unknown_hours"] += weight
                    at = int(indexes[offset])
                    left = (int(times[at-1]), *map(float, coords[at-1])) if at else None
                    right = (int(times[at]), *map(float, coords[at])) if at < len(times) else None
                    left, right = widen_fix_ends(left, right, fix_tolerance_ns)
                    prior = None
                    for rule, shift_h in (("zero", 0), ("main", 1)):
                        use_ports = port_ends_usable(port, observed is not None)
                        port_left = ((int(port.departure_time.value)-shift_h*NS_HOUR,
                                      float(port.departure_port_lat), float(port.departure_port_lon))
                                     if use_ports and pd.notna(port.departure_port_lat) else None)
                        port_right = ((int(port.arrival_time.value)+shift_h*NS_HOUR,
                                       float(port.arrival_port_lat), float(port.arrival_port_lon))
                                      if use_ports and pd.notna(port.arrival_port_lat) else None)
                        ends = (left or port_left, right or port_right)
                        geometry = kernels._geometry(event, *ends) if any(x is not None for x in ends) else None
                        if geometry is None:
                            entry[f"{rule}_unbounded_hours"] += weight
                        if prior is not None and ends == prior[0]:
                            wave, possible = prior[1:]
                        else:
                            wave = kernels._wave_bounds(geometry, item.hour_anchor_utc, matcher)
                            possible = [kernels._tc_possible(geometry, storm_map.get(event), radius,
                                        old_rule=(ends == (left, right))) for radius in (300, 500)]
                        prior = (ends, wave, possible)
                        lo, hi, status = wave
                        if status == "matched":
                            entry[f"{rule}_wave_counted_hours"] += weight
                            entry[f"{rule}_wave_min_integral"] += lo*weight
                            entry[f"{rule}_wave_max_integral"] += hi*weight
                        else:
                            entry[f"{rule}_wave_{status}_hours"] += weight
                        for radius, hit in zip((300, 500), possible):
                            if hit:
                                entry[f"{rule}_tc{radius}_possible_hours"] += weight
                        if hour_detail:
                            detail.append((cid, event, rule, weight, lo, hi, status, *possible,
                                           left is not None, right is not None))
                        if possible[0] and not possible[1]:
                            raise ValueError("300-km upper not contained in 500-km upper")
        finally:
            matcher.close()
        progress["months"].append({"month": month, "unknown_segments": len(block), "seconds": time.perf_counter()-t0})
        write_json(out / "status.json", progress)
    facts = pd.DataFrame.from_dict(result, orient="index").rename_axis("candidate_id").reset_index()
    if facts.empty:
        facts = pd.DataFrame({"candidate_id": pd.Series([], dtype=str)})
    for rule in ("zero", "main"):
        for suffix in ("unbounded_hours", "wave_counted_hours", "wave_min_integral", "wave_max_integral",
                       "tc300_possible_hours", "tc500_possible_hours"):
            column = f"{rule}_{suffix}"
            if column not in facts:
                facts[column] = 0.0
    facts = facts.fillna(0)
    numeric_close(facts.get("unknown_hours", pd.Series([], dtype=float)).sum(), unknown.segment_weight_hours.sum(), "unknown census hours")
    facts.to_parquet(out / "voyage_bounds.parquet", index=False)
    if hour_detail:
        pd.DataFrame(detail, columns=["candidate_id", "position_time_ns", "rule", "weight_hours", "wave_lo_m",
                                     "wave_hi_m", "wave_status", "tc300_possible", "tc500_possible",
                                     "left_fix", "right_fix"]).to_parquet(out / "hour_bounds.parquet", index=False)
    progress.update(state="completed", voyages_with_unknown=len(facts),
                    totals={c: float(facts[c].sum()) for c in facts if c != "candidate_id"},
                    elapsed_seconds=time.perf_counter()-started,
                    peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                    output=record(out / "voyage_bounds.parquet"))
    write_json(out / "status.json", progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", default=SHARDS)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--probe-candidates", type=int)
    parser.add_argument("--fix-time-tolerance-hours", type=float,
                        help="probe override; the census always uses the rule revision value")
    parser.add_argument("--hour-detail", action="store_true", help="probe only: also save per-hour bounds")
    parser.add_argument('--candidate-ids', nargs='+', help='replay only these candidates in one shard; never a census')
    args = parser.parse_args()
    source = binding()
    if not 1 <= args.workers <= 40 or not set(args.shards).issubset(SHARDS):
        raise ValueError("invalid worker count or GFW shard scope")
    if args.candidate_ids and (len(args.shards) != 1 or args.probe_candidates):
        raise ValueError('candidate replay requires one shard and no count-based probe')
    rule = revision()
    tolerance = rule["fix_time_tolerance_hours"] if args.fix_time_tolerance_hours is None else args.fix_time_tolerance_hours
    if (args.fix_time_tolerance_hours is not None or args.hour_detail) and not (args.probe_candidates or args.candidate_ids):
        raise ValueError("tolerance overrides and per-hour detail are for probes only")
    assets = weather_runner.verify_weather_assets()
    out = inside_runs(args.out)
    out.mkdir(parents=True, exist_ok=False)
    state = {"state": "running", "step": 6, "component": "missing_weather_bounds",
             "shards": sorted(args.shards), "workers": args.workers, "probe_candidates": args.probe_candidates,
             "fix_time_tolerance_hours": tolerance, "hour_detail": args.hour_detail,
             "selected_candidate_ids": args.candidate_ids,
             "completed": [], "failures": [], "authorization": source["authorization"],
             "preparation": record(args.prepared / "preparation.json"), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "network_requests": 0, "products_modified": 0, "adopted": False}
    state["weather_assets"] = assets
    state.update(provenance([HERE / "step6_core.py", Path(__file__), DECISION, REVISION,
        HERE / 'port_constraints.py',
        E002 / "20_quantify_uncertainty/10_missing_positions/60_compute_tolerant_bounds.py",
        E002 / "20_quantify_uncertainty/10_missing_positions/10_probe_reachability.py",
        E002 / "20_quantify_uncertainty/10_missing_positions/20_compute_tc_upper.py",
        WEATHER / "leg_ais_coverage/30_bound_dark_hour_wave_exposure.py"]))
    write_json(out / "status.json", state)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_shard, s, str(args.prepared), str(out), args.probe_candidates,
                               tolerance, args.hour_detail, args.candidate_ids): s for s in args.shards}
        for future in as_completed(futures):
            shard = futures[future]
            try:
                result = future.result()
                state["completed"].append(shard)
                print(json.dumps({k: result[k] for k in ("shard", "unknown_segments", "elapsed_seconds", "peak_rss_gib")}), flush=True)
            except Exception:
                state["failures"].append({"shard": shard, "traceback": traceback.format_exc()})
            write_json(out / "status.json", state)
    state.update(state="failed" if state["failures"] else "completed", elapsed_seconds=time.perf_counter()-started)
    write_json(out / "status.json", state)
    if state["failures"]:
        raise RuntimeError(state["failures"])


if __name__ == "__main__":
    main()
