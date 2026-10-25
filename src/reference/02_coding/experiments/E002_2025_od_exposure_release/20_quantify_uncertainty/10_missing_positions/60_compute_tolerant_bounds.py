#!/usr/bin/env python3
"""T014 census TC/SWH bounds with calibrated port endpoint tolerance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CODING = HERE.parents[3]
PORTS = CODING / ".runs/2026-09-24_1500_E002-T012-exposure-units/10_c1/port_endpoints_by_candidate.parquet"
NS_HOUR = 3_600_000_000_000
SPEED_KM_H = 25.0 * 1.852
CACHED_HOURS = int(os.environ.get("E002_T012_CACHED_HOURS", "2"))
SPACE_KM = 0.0
TIME_H = 0.0
PORT_TIME_SHIFT_NS = 0
UNBOUNDED_CONFLICT_CANDIDATES: set[str] = set()


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROBE = _module("e002_t012_probe", HERE / "10_probe_reachability.py")
TC = _module("e002_t012_tc", HERE / "20_compute_tc_upper.py")
from weather_core.era5_match import _cell_code  # noqa: E402

PROBE.legacy.MAX_CELLS_PER_HOUR = 300_000


def _geometry(event_ns: int, left: tuple | None, right: tuple | None) -> SimpleNamespace:
    return SimpleNamespace(
        position_time_utc=pd.Timestamp(event_ns, tz="UTC"),
        position_time_ns=event_ns,
        left_time_ns=left[0] if left else -1,
        left_lat=left[1] if left else np.nan,
        left_lon=left[2] if left else np.nan,
        right_time_ns=right[0] if right else -1,
        right_lat=right[1] if right else np.nan,
        right_lon=right[2] if right else np.nan,
        endpoint_case="two_sided" if left and right else "one_sided",
    )


def _wave_bounds(row: SimpleNamespace | None, hour: pd.Timestamp, matcher) -> tuple:
    if row is None:
        return (np.nan, np.nan, "unbounded")
    wave_row = SimpleNamespace(**vars(row))
    if wave_row.left_time_ns < 0:
        wave_row.left_time_ns = None
    if wave_row.right_time_ns < 0:
        wave_row.right_time_ns = None
    lat, lon, capped, status = PROBE._reachable(wave_row)
    if capped:
        raise ValueError("reachable ERA5 grid was thinned")
    if status == "speed_infeasible" or len(lat) == 0:
        return (np.nan, np.nan, "speed_infeasible")
    time_index = matcher.times.get_indexer([hour])[0]
    if time_index < 0:
        raise ValueError(f"frozen ERA5 lacks hour {hour}")
    # reachable_cells returns exact half-degree centers, apart from its
    # documented empty-grid fallback. The latter uses the original matcher.
    on_grid = bool(
        np.all(np.abs(lat * 2 - np.rint(lat * 2)) < 1e-9)
        and np.all(np.abs(lon * 2 - np.rint(lon * 2)) < 1e-9)
    )
    if on_grid:
        code = _cell_code(lat, lon)
        pos = np.searchsorted(matcher.grid.codes, code)
        valid = pos < len(matcher.grid.codes)
        valid[valid] &= matcher.grid.codes[pos[valid]] == code[valid]
        values = matcher._values_for_index(int(time_index))[pos[valid]]
        values = values[np.isfinite(values)]
    else:
        frame = pd.DataFrame({"hour_anchor_utc": [hour] * len(lat), "lat": lat, "lon": lon})
        values = matcher.match(frame).swh_m.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan, "no_wet_era5_cells")
    return (float(values.min()), float(values.max()), "matched")


def _tc_possible(row: SimpleNamespace | None, storms: tuple[np.ndarray, np.ndarray] | None,
                 radius_km: float, *, old_rule: bool) -> bool:
    if row is None:
        return True  # established U2 conservative unbounded rule
    if storms is None:
        return False
    if row.left_time_ns >= 0 and row.right_time_ns >= 0 and not TC._endpoint_speed_feasible(row):
        return False if old_rule else True  # conflicting added port cannot narrow an upper bound
    lat, lon = storms
    necessary = TC.necessary_center_filter(
        storm_lat=lat, storm_lon=lon, event_ns=row.position_time_ns,
        left_time_ns=row.left_time_ns, left_lat=row.left_lat, left_lon=row.left_lon,
        right_time_ns=row.right_time_ns, right_lat=row.right_lat, right_lon=row.right_lon,
        radius_km=radius_km,
    )
    for storm_lat, storm_lon in zip(lat[necessary], lon[necessary]):
        if TC.spherical_caps_intersect(
            TC._segment_caps(row, float(storm_lat), float(storm_lon), radius_km)
        ):
            return True
    return False


def _initial() -> dict:
    return {
        "unknown_segments": 0, "unknown_hours": 0.0,
        "old_unbounded_segments": 0, "old_unbounded_hours": 0.0,
        "new_unbounded_segments": 0, "new_unbounded_hours": 0.0,
        "port_added_segments": 0, "port_added_hours": 0.0,
        "old_wave_counted_segments": 0, "old_wave_counted_hours": 0.0,
        "new_wave_counted_segments": 0, "new_wave_counted_hours": 0.0,
        "old_wave_no_wet_segments": 0, "new_wave_no_wet_segments": 0,
        "old_wave_speed_infeasible_segments": 0, "new_wave_speed_infeasible_segments": 0,
        "old_wave_min_integral": 0.0, "old_wave_max_integral": 0.0,
        "new_wave_min_integral": 0.0, "new_wave_max_integral": 0.0,
        "old_tc300_possible_hours": 0.0, "old_tc500_possible_hours": 0.0,
        "new_tc300_possible_hours": 0.0, "new_tc500_possible_hours": 0.0,
    }


def process_shard(shard: str, out: Path, ports: pd.DataFrame, cyclones) -> dict:
    started = time.perf_counter()
    frame = PROBE.read_shard(shard)
    frame["candidate_id"] = frame.candidate_id.astype(str)
    unknown = frame.loc[frame.track_evidence.eq("unresolved")].copy()
    known = frame.loc[
        frame.track_evidence.ne("unresolved")
        & np.isfinite(frame.lat.to_numpy(dtype=float))
        & np.isfinite(frame.lon.to_numpy(dtype=float)),
        ["candidate_id", "position_time_utc", "lat", "lon"],
    ]
    known_by_id = {}
    for candidate_id, block in known.groupby("candidate_id", sort=False, observed=True):
        ordered = block.sort_values("position_time_utc", kind="stable")
        known_by_id[str(candidate_id)] = (
            ordered.position_time_utc.astype("int64").to_numpy(dtype=np.int64),
            ordered[["lat", "lon"]].to_numpy(dtype=float),
        )
    port_by_id = ports.loc[ports.index.intersection(unknown.candidate_id.unique())]
    if len(port_by_id) != unknown.candidate_id.nunique():
        raise ValueError(f"{shard}: candidate_id missing from C1 endpoint table")
    results = defaultdict(_initial)
    month_stats = []
    for month, block in unknown.groupby(unknown.hour_anchor_utc.dt.to_period("M"), sort=True):
        year, mon = int(month.year), int(month.month)
        matcher = PROBE.Era5MonthlyMatcher(
            PROBE.resolve_monthly_era5_assets(year=year, month=mon),
            maximum_wet_cell_snap_km=0.0,
            cached_hours=CACHED_HOURS,
        )
        month_started = time.perf_counter()
        try:
            PROBE.validate_month_time_grid(matcher, year, mon)
            matcher.preload_swh()
            storm_frame = cyclones.positions_at_times(block.position_time_utc.drop_duplicates().tolist())
            storm_map = {}
            for stamp, group in storm_frame.groupby("target_time_utc", sort=False):
                storm_map[int(pd.Timestamp(stamp).value)] = (
                    group.storm_lat.to_numpy(dtype=float), group.storm_lon.to_numpy(dtype=float),
                )
            for candidate_id, rows in block.groupby("candidate_id", sort=False, observed=True):
                candidate_id = str(candidate_id)
                positions = known_by_id.get(candidate_id)
                port = port_by_id.loc[candidate_id]
                old_unbounded = positions is None
                # With no intermediate positions, require both frozen port ends.
                use_port = candidate_id not in UNBOUNDED_CONFLICT_CANDIDATES
                full_ports = bool(port.port_endpoints_connected) and use_port
                departure = (
                    (int(port.departure_time.value) - PORT_TIME_SHIFT_NS,
                     float(port.departure_port_lat), float(port.departure_port_lon))
                    if use_port and pd.notna(port.departure_port_lat) and (not old_unbounded or full_ports)
                    else None
                )
                arrival = (
                    (int(port.arrival_time.value) + PORT_TIME_SHIFT_NS,
                     float(port.arrival_port_lat), float(port.arrival_port_lon))
                    if use_port and pd.notna(port.arrival_port_lat) and (not old_unbounded or full_ports)
                    else None
                )
                if positions is not None:
                    times, coords = positions
                    event_ns = rows.position_time_utc.astype("int64").to_numpy(dtype=np.int64)
                    right_idx = np.searchsorted(times, event_ns, side="left")
                for number, item in enumerate(rows.itertuples(index=False)):
                    event = int(item.position_time_utc.value)
                    weight = float(item.segment_weight_hours)
                    record = results[candidate_id]
                    record["unknown_segments"] += 1
                    record["unknown_hours"] += weight
                    left = right = None
                    if positions is not None:
                        at = int(right_idx[number])
                        if at > 0:
                            left = (int(times[at - 1]), float(coords[at - 1, 0]), float(coords[at - 1, 1]))
                        if at < len(times):
                            right = (int(times[at]), float(coords[at, 0]), float(coords[at, 1]))
                    old_row = _geometry(event, left, right) if left or right else None
                    new_left, new_right = left or departure, right or arrival
                    new_row = _geometry(event, new_left, new_right) if new_left or new_right else None
                    changed = new_left != left or new_right != right
                    if old_row is None:
                        record["old_unbounded_segments"] += 1
                        record["old_unbounded_hours"] += weight
                    if new_row is None:
                        record["new_unbounded_segments"] += 1
                        record["new_unbounded_hours"] += weight
                    if changed:
                        record["port_added_segments"] += 1
                        record["port_added_hours"] += weight
                    old_wave = _wave_bounds(old_row, item.hour_anchor_utc, matcher)
                    new_wave = _wave_bounds(new_row, item.hour_anchor_utc, matcher) if changed else old_wave
                    for rule, wave in (("old", old_wave), ("new", new_wave)):
                        lo, hi, status = wave
                        if status == "matched":
                            record[f"{rule}_wave_counted_segments"] += 1
                            record[f"{rule}_wave_counted_hours"] += weight
                            record[f"{rule}_wave_min_integral"] += lo * weight
                            record[f"{rule}_wave_max_integral"] += hi * weight
                        elif status in ("no_wet_era5_cells", "speed_infeasible"):
                            record[f"{rule}_wave_{'no_wet' if status == 'no_wet_era5_cells' else 'speed_infeasible'}_segments"] += 1
                    storms = storm_map.get(event)
                    old300 = _tc_possible(old_row, storms, 300.0, old_rule=True)
                    old500 = _tc_possible(old_row, storms, 500.0, old_rule=True)
                    if changed:
                        new300 = _tc_possible(new_row, storms, 300.0, old_rule=False)
                        new500 = _tc_possible(new_row, storms, 500.0, old_rule=False)
                    else:
                        new300, new500 = old300, old500
                    if old300 and not old500 or new300 and not new500:
                        raise AssertionError("TC300 upper exceeds TC500 upper")
                    for name, possible in (("old_tc300", old300), ("old_tc500", old500),
                                           ("new_tc300", new300), ("new_tc500", new500)):
                        if possible:
                            record[f"{name}_possible_hours"] += weight
        finally:
            matcher.close()
        month_stats.append({"month": str(month), "unknown_segments": int(len(block)),
                            "seconds": time.perf_counter() - month_started})
    output = pd.DataFrame.from_dict(results, orient="index").rename_axis("candidate_id").reset_index()
    out.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out / f"{shard}_voyage_bounds.parquet", index=False)
    totals = output.drop(columns="candidate_id").sum(numeric_only=True).to_dict()
    report = {"shard": shard, "weather_rows": int(len(frame)), "unknown_segments": int(len(unknown)),
              "voyages_with_unknown": int(len(output)), "totals": totals,
              "months": month_stats, "cached_hours": CACHED_HOURS,
              "port_space_tolerance_km": SPACE_KM, "port_time_tolerance_h": TIME_H,
              "unbounded_conflict_voyages": len(UNBOUNDED_CONFLICT_CANDIDATES),
              "elapsed_seconds": time.perf_counter() - started}
    (out / f"{shard}_closure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


_PORTS = None
_CYCLONES = None


def _worker(shard: str, out: Path) -> dict:
    return process_shard(shard, out, _PORTS, _CYCLONES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--space-km", type=float, required=True)
    parser.add_argument("--time-h", type=float, required=True)
    parser.add_argument("--unbounded-conflict-audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.shard) == bool(args.all):
        parser.error("select exactly one of --shard or --all")
    global _PORTS, _CYCLONES, SPACE_KM, TIME_H, PORT_TIME_SHIFT_NS, UNBOUNDED_CONFLICT_CANDIDATES
    SPACE_KM = args.space_km
    TIME_H = args.time_h
    if SPACE_KM not in (0, 10, 25, 50) or TIME_H not in (0, 1, 2):
        parser.error("tolerance must be in the frozen Contract grid")
    # A port coordinate ball of S km enlarges every 25-kn reach cap by S km.
    # Moving only synthetic port times by S / speed + T hours is equivalent.
    PORT_TIME_SHIFT_NS = int(round((TIME_H + SPACE_KM / SPEED_KM_H) * NS_HOUR))
    if args.unbounded_conflict_audit:
        audit = pd.read_parquet(args.unbounded_conflict_audit)
        if len(audit) != 969 or "reachable_50km_2h" not in audit:
            raise ValueError("port conflict audit differs from frozen 969-hour grid")
        UNBOUNDED_CONFLICT_CANDIDATES = set(
            audit.loc[~audit.reachable_50km_2h, "candidate_id"].astype(str))
        if len(UNBOUNDED_CONFLICT_CANDIDATES) != 5:
            raise ValueError("expected five structurally conflicting GFW voyages")
    _PORTS = pd.read_parquet(PORTS).set_index("candidate_id", verify_integrity=True)
    fixes, _ = PROBE.load_ibtracs_snapshot(
        PROBE.CONFIG["ibtracs_path"], expected_sha256=PROBE.CONFIG["ibtracs_sha256"]
    )
    _CYCLONES = PROBE.CycloneTrackIndex(fixes, maximum_fix_gap_hours=3.0)
    if args.shard:
        print(json.dumps(_worker(args.shard, args.out), indent=2), flush=True)
        return
    shards = sorted(path.parent.name for path in PROBE.TRACK_ROOT.glob("run/shards/shard_*/candidate_summary.parquet"))
    if len(shards) != 253:
        raise ValueError(f"expected 253 adopted shards, found {len(shards)}")
    args.out.mkdir(parents=True, exist_ok=True)
    already = [shard for shard in shards if args.resume
               and (args.out / f"{shard}_closure.json").is_file()
               and (args.out / f"{shard}_voyage_bounds.parquet").is_file()]
    todo = [shard for shard in shards if shard not in set(already)]
    started = time.perf_counter()
    failures = []
    print(f"C2/C3 resume retained {len(already)} closed shards; processing {len(todo)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, shard, args.out): shard for shard in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            shard = futures[future]
            try:
                result = future.result()
                print(f"C2/C3 {len(already)+done:03d}/253 {shard}: unknown={result['unknown_segments']:,}; "
                      f"shard_s={result['elapsed_seconds']:.1f}; wall_s={time.perf_counter()-started:.1f}", flush=True)
            except Exception as error:
                failures.append({"shard": shard, "error": repr(error)})
                print(f"C2/C3 FAILED {shard}: {error!r}", flush=True)
    report = {"shards_expected": 253, "reused_closed_shards": len(already),
              "processed_shards": len(todo), "workers": args.workers,
              "port_space_tolerance_km": SPACE_KM, "port_time_tolerance_h": TIME_H,
              "unbounded_conflict_voyages": len(UNBOUNDED_CONFLICT_CANDIDATES),
              "cached_hours": CACHED_HOURS, "failures": failures,
              "elapsed_wall_seconds": time.perf_counter() - started}
    (args.out / "full_run_status.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} C2/C3 shards failed")


if __name__ == "__main__":
    main()
