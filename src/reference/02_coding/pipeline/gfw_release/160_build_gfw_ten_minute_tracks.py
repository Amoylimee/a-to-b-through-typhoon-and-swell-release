#!/usr/bin/env python3
"""E006 GFW version, step 4: ten-minute tracks of the candidate voyages (T036).

Each retained voyage of the step-3 candidate root (T034) gets a UTC ten-minute grid strictly
inside (departure, arrival), the same grid as the reference AIS product
(`candidate_full_ten_minute.strict_grid_ns`). Rules (CONTRACT §3 "E006 GFW 版构建规则" e):

- A cleaned GFW point (stamped hh:30) fills its own grid cell as `observed_ais`.
- Two consecutive GFW points of the voyage less than 48 h apart are joined by the frozen
  great-circle-first sea path (`sea_path/routing.py`: the direct great circle when it stays on
  the 1-arc-minute ocean mask, A* only around blocked sections), and the cells between them are
  placed along that path at constant speed by the recorded times (`interpolate_along_path`),
  labelled `gfw_supported_interpolation`. This replaces both the <=4 h straight line and the
  4-24 h reconstruction of the reference AIS version.
- The 25 kn feasibility check divides the path length by the recorded gap plus one hour: each
  point's real time lies anywhere in its UTC hour, so the real gap can be up to one hour longer.
  The route timeout stays 5 s and positions are never snapped to the ocean.
- An end point on a land cell of the 1-arc-minute mask lies in water the mask cannot draw (harbour
  basin, river, canal): the great-circle stretch from it to the arc's first sea cell is kept and the
  rest is routed as above; an arc that never reaches a sea cell is the great circle (T037). These
  cells carry `position_source = gfw_sea_path_unmapped_water`.
- Gaps of 48 h or more, routes that do not resolve (status kept in `gap_class`), and any cell
  before the first or after the last point of the voyage stay `unresolved` without a position.

Output per GFW vessel shard, in the reference AIS product layout: `ten_minute_complete_candidate.parquet`
(one row per voyage cell), `candidate_summary.parquet` (one row per voyage), and
`unresolved_segments.parquet` (every point pair that could not be joined, with the router's
reason). No network use; outputs stay in `.runs/`.

    python 160_build_gfw_ten_minute_tracks.py --out <new .runs root> [--shards shard_0000 ...]
        [--probe-candidates N] [--workers 32]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
TRACK_STAGE = CODING / "pipeline/40_construct_variables/50_reconstruct_companion_tracks"
if str(TRACK_STAGE) not in sys.path:
    sys.path.insert(0, str(TRACK_STAGE))

from candidate_full_ten_minute import (  # noqa: E402
    GFW_INTERPOLATION_LABEL,
    OBSERVED_LABEL,
    UNRESOLVED_LABEL,
    source_gap_id,
    strict_grid_ns,
    validate_candidate_frame,
)
from sea_path.ocean_mask import load_ocean_mask  # noqa: E402
from sea_path.routing import (  # noqa: E402
    ROUTE_RESOLVED,
    ROUTE_SPEED_INFEASIBLE,
    ROUTE_STATUSES,
    SeaPathRouter,
    great_circle_waypoints,
    interpolate_along_path,
)

CANDIDATE_ROOT = CODING / ".runs/2026-10-06_1728_E006-gfw-voyage-base-t034/candidate_root"
PORT_ROOT = CODING / ".runs/2026-10-06_1650_E006-gfw-port-calls"
CLEAN_ROOT = CODING / ".runs/2026-10-05_2305_E006-gfw-cleaning"
MASK_PATH = CODING / "data/frozen_inputs/external/gshhg/ocean_mask_01m.npy"
SHARDS = [f"shard_{i:04d}" for i in range(64)]
LAYER = "GFW2025"
GAP_MAX_H = 48.0
TIME_UNCERTAINTY_H = 1.0
MAX_SPEED_KNOTS = 25.0
ROUTE_TIMEOUT_S = 5.0
NS_PER_HOUR = 3_600_000_000_000
POINT_COLUMNS = ["vessel_uid", "mmsi", "UTC", "UTC_10min", "lat", "lon", "source_provider", "source_resolution_original"]
CANDIDATE_COLUMNS = ["candidate_id", "vessel_uid_clean", "mmsi", "departure_time", "arrival_time",
                     "origin_city", "dest_city", "shiptype_group"]
RULES = {"gap_max_hours": GAP_MAX_H, "speed_check_extra_hours": TIME_UNCERTAINTY_H,
         "max_speed_knots": MAX_SPEED_KNOTS, "route_timeout_s": ROUTE_TIMEOUT_S, "snap_to_ocean": False,
         "grid_seconds": 600, "point_stamp": "hh:30", "anchors": "none",
         "unmapped_water": ("an end point on a mask land cell (harbour, river, canal the 1-arc-minute mask cannot "
                            "draw) follows the great circle until the arc first reaches a sea cell; the rest is routed "
                            "great-circle-first with A*; an arc with no sea cell is the great circle (T037)")}

CELL_CLASSES = ["gfw_sea_path_le4h", "gfw_sea_path_4_24h", "gfw_sea_path_24_48h", "unresolved_gap_ge48h",
                *[f"unresolved_route_{s}" for s in ROUTE_STATUSES if s != ROUTE_RESOLVED], "unresolved_voyage_end"]
STRATEGIES = ["direct_great_circle", "great_circle_with_local_astar_for_blocked_runs", "coincident_positions",
              "unmapped_water_great_circle", "unmapped_water_then_sea_route", "none"]
COUNT_KEYS = ["points_in_window", "starts_on_point", "ends_on_point", "pairs", "cells_unmapped_water_rule",
              *[f"route_{s}" for s in ROUTE_STATUSES], *[f"strategy_{s}" for s in STRATEGIES],
              *[f"cells_{c}" for c in CELL_CLASSES]]

_ROUTER: SeaPathRouter | None = None


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise ValueError(f"bound input changed: {path}")
    return path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def router() -> SeaPathRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = SeaPathRouter(load_ocean_mask(MASK_PATH))
    return _ROUTER


def gap_class(hours: float) -> str:
    if hours <= 4.0:
        return "gfw_sea_path_le4h"
    if hours <= 24.0:
        return "gfw_sea_path_4_24h"
    return "gfw_sea_path_24_48h"


# --- One point pair ---------------------------------------------------------------------


def is_sea(route: SeaPathRouter, lat: float, lon: float) -> bool:
    return bool(route.mask[route._position_to_index(lat, lon)])


def join_pair(route: SeaPathRouter, lat_a: float, lon_a: float, lat_b: float, lon_b: float, hours: float):
    """Great circle first, detouring only around land the ocean mask can resolve (T036, T037).

    Returns (status, path, detail). Both ends on sea cells: the frozen router, unchanged. An end on
    a mask land cell sits in water the mask cannot draw, so the great-circle stretch from it to the
    first sea cell on the arc is kept, and the sea stretch between the first and last sea cells is
    routed by the same router. The 25 kn check divides the whole path by the gap plus one hour.
    """
    budget = hours + TIME_UNCERTAINTY_H
    if is_sea(route, lat_a, lon_a) and is_sea(route, lat_b, lon_b):
        result = route.resolve_path(lat_a, lon_a, lat_b, lon_b, timeout_s=ROUTE_TIMEOUT_S,
                                    duration_hours=budget, max_speed_knots=MAX_SPEED_KNOTS)
        return result.status, result.path, dict(result.detail)
    direct = great_circle_waypoints(lat_a, lon_a, lat_b, lon_b)
    sea = np.fromiter((is_sea(route, la, lo) for la, lo in direct), dtype=bool, count=len(direct))
    detail: dict = {"unmapped_water_rule": True}
    if not sea.any():
        waypoints, detail["strategy"] = direct, "unmapped_water_great_circle"
    else:
        first, last = int(np.argmax(sea)), int(len(sea) - 1 - np.argmax(sea[::-1]))
        result = route.resolve_path(*direct[first], *direct[last], timeout_s=ROUTE_TIMEOUT_S)
        if result.status != ROUTE_RESOLVED or result.path is None:
            return result.status, None, {**result.detail, **detail, "strategy": "unmapped_water_then_sea_route"}
        waypoints = np.vstack([direct[:first], result.path.waypoints, direct[last + 1:]])
        detail["strategy"] = "unmapped_water_then_sea_route"
    path = SeaPathRouter._sea_path_from_waypoints(np.asarray(waypoints, dtype=float))
    detail["implied_speed_knots"] = path.total_km / 1.852 / budget
    if detail["implied_speed_knots"] > MAX_SPEED_KNOTS:
        return ROUTE_SPEED_INFEASIBLE, None, detail
    return ROUTE_RESOLVED, path, detail


# --- One voyage -------------------------------------------------------------------------


def build_track(candidate, points: pd.DataFrame, route: SeaPathRouter, shard: str):
    """Return the voyage's grid rows, its summary, and the point pairs left unresolved."""
    candidate_id = str(candidate.candidate_id)
    start = pd.Timestamp(candidate.departure_time).tz_convert("UTC")
    end = pd.Timestamp(candidate.arrival_time).tz_convert("UTC")
    grid = strict_grid_ns(start, end)
    n = len(grid)
    evidence = np.full(n, UNRESOLVED_LABEL, dtype=object)
    position_source = np.full(n, None, dtype=object)
    gap_ids = np.full(n, None, dtype=object)
    gap_classes = np.full(n, None, dtype=object)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    observation = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    provider = np.full(n, None, dtype=object)
    resolution = np.full(n, None, dtype=object)
    counts: Counter[str] = Counter()
    failed: list[dict] = []

    ns = points.UTC.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    inside = (ns >= start.value) & (ns <= end.value)
    p_ns, p_lat, p_lon = ns[inside], points.lat.to_numpy(float)[inside], points.lon.to_numpy(float)[inside]
    p_bin = points.UTC_10min.to_numpy(dtype="datetime64[ns]").astype(np.int64)[inside]
    counts["points_in_window"] = len(p_ns)
    counts["starts_on_point"] = int(len(p_ns) > 0 and p_ns[0] == start.value)
    counts["ends_on_point"] = int(len(p_ns) > 0 and p_ns[-1] == end.value)

    # Observed points fill their own ten-minute cell.
    at = np.searchsorted(grid, p_bin)
    hit = (at < n) & (grid[np.minimum(at, max(n - 1, 0))] == p_bin) if n else np.zeros(len(p_bin), bool)
    lat[at[hit]], lon[at[hit]] = p_lat[hit], p_lon[hit]
    evidence[at[hit]] = OBSERVED_LABEL
    position_source[at[hit]] = OBSERVED_LABEL
    observation[at[hit]] = p_ns[hit].astype("datetime64[ns]")
    provider[at[hit]] = points.source_provider.to_numpy(object)[inside][hit]
    resolution[at[hit]] = points.source_resolution_original.to_numpy(object)[inside][hit]

    # Consecutive point pairs: sea path below 48 h, otherwise unresolved.
    for a in range(len(p_ns) - 1):
        ta, tb = int(p_ns[a]), int(p_ns[a + 1])
        left, right = np.searchsorted(grid, ta, side="right"), np.searchsorted(grid, tb, side="left")
        hours = (tb - ta) / NS_PER_HOUR
        counts["pairs"] += 1
        if right <= left:
            continue
        cells = np.arange(left, right)
        if hours >= GAP_MAX_H:
            label, status, detail = "unresolved_gap_ge48h", "gap_ge48h", {}
        else:
            status, path, detail = join_pair(route, p_lat[a], p_lon[a], p_lat[a + 1], p_lon[a + 1], hours)
            counts[f"route_{status}"] += 1
            counts[f"strategy_{detail.get('strategy', 'none')}"] += 1
            if status == ROUTE_RESOLVED and path is not None:
                label = gap_class(hours)
                positions = interpolate_along_path(path, ta / 1e9, tb / 1e9, grid[cells] / 1e9)
                lat[cells], lon[cells] = positions[:, 0], positions[:, 1]
                evidence[cells] = GFW_INTERPOLATION_LABEL
                unmapped = bool(detail.get("unmapped_water_rule"))
                position_source[cells] = "gfw_sea_path_unmapped_water" if unmapped else "gfw_sea_path_interpolation"
                counts["cells_unmapped_water_rule"] += len(cells) if unmapped else 0
            else:
                label = f"unresolved_route_{status}"
        gap_classes[cells] = label
        gap_ids[cells] = source_gap_id(candidate_id, ta, tb, label)
        counts[f"cells_{label}"] += len(cells)
        if label.startswith("unresolved"):
            failed.append({"candidate_id": candidate_id, "gap_start_ns": ta, "gap_end_ns": tb, "gap_hours": hours,
                           "gap_class": label, "status": status, "cells": len(cells),
                           "start_lat": p_lat[a], "start_lon": p_lon[a], "end_lat": p_lat[a + 1], "end_lon": p_lon[a + 1],
                           "implied_speed_knots": float(detail.get("implied_speed_knots") or np.nan),
                           "start_nearest_ocean_km": float(detail.get("start_nearest_ocean_km") or np.nan),
                           "goal_nearest_ocean_km": float(detail.get("goal_nearest_ocean_km") or np.nan)})

    # Cells outside the first/last point of the voyage keep no position.
    ends = (evidence == UNRESOLVED_LABEL) & pd.isna(gap_classes)
    gap_classes[ends] = "unresolved_voyage_end"
    counts["cells_unresolved_voyage_end"] = int(ends.sum())

    frame = pd.DataFrame({
        "candidate_id": pd.Series([candidate_id] * n, dtype="string"),
        "mmsi": pd.Series([str(candidate.mmsi)] * n, dtype="string"),
        "vessel_uid_clean": pd.Series([str(candidate.vessel_uid_clean)] * n, dtype="string"),
        "layer": pd.Series([LAYER] * n, dtype="string"),
        "utc": pd.to_datetime(grid, unit="ns", utc=True),
        "lat": lat, "lon": lon,
        "track_evidence": pd.Series(evidence, dtype="string"),
        "position_source": pd.Series(position_source, dtype="string"),
        "gap_id": pd.Series(gap_ids, dtype="string"),
        "gap_class": pd.Series(gap_classes, dtype="string"),
        "anchor_source": pd.Series([None] * n, dtype="string"),
        "observation_time_utc": pd.to_datetime(observation, utc=True),
        "source_provider": pd.Series(provider, dtype="string"),
        "source_resolution_original": pd.Series(resolution, dtype="string"),
        "source_shard": pd.Series([shard] * n, dtype="string"),
    })
    observed = int((evidence == OBSERVED_LABEL).sum())
    interpolated = int((evidence == GFW_INTERPOLATION_LABEL).sum())
    summary = {
        "candidate_id": candidate_id, "mmsi": str(candidate.mmsi), "vessel_uid_clean": str(candidate.vessel_uid_clean),
        "layer": LAYER, "source_shard": shard, "departure_time": start, "arrival_time": end,
        "origin_city": candidate.origin_city, "dest_city": candidate.dest_city, "shiptype_group": candidate.shiptype_group,
        "window_hours": (end - start).total_seconds() / 3600, "expected_grid_rows": n, "grid_rows": n,
        "observed_ais_rows": observed, "ais_interpolated_le4h_rows": 0, "gfw_anchor_rows": 0,
        "gfw_supported_interpolation_rows": interpolated, "endpoint_24_48h_rows": 0,
        "unresolved_rows": int(n - observed - interpolated),
        **{key: int(counts.get(key, 0)) for key in COUNT_KEYS},
    }
    if set(counts) - set(COUNT_KEYS):
        raise ValueError(f"unexpected count keys: {set(counts) - set(COUNT_KEYS)}")
    return frame, summary, failed


# --- One shard --------------------------------------------------------------------------


def work(task: dict) -> dict:
    shard, out = task["shard"], Path(task["output"])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        source = verify(task["points"])
        candidates = pq.read_table(verify(task["candidates"])).to_pandas()
        points = feather.read_table(source, columns=POINT_COLUMNS).to_pandas()
        points["vessel_uid"] = points.vessel_uid.astype(str)
        points["UTC"] = pd.to_datetime(points.UTC, utc=True)
        points["UTC_10min"] = pd.to_datetime(points.UTC_10min, utc=True)
        if not points.UTC_10min.eq(points.UTC.dt.floor("10min")).all() or not points.UTC.dt.minute.eq(30).all():
            raise ValueError("GFW points are not stamped hh:30 on the frozen floor rule")
        groups = {key: group.sort_values("UTC").reset_index(drop=True)
                  for key, group in points.groupby("vessel_uid", sort=False)}
        if any(group.UTC.duplicated().any() for group in groups.values()):
            raise ValueError("duplicate GFW point times within a vessel")
        route = router()
        empty = points.iloc[0:0]
        writers: dict[str, pq.ParquetWriter] = {}
        pending: dict[str, list[pd.DataFrame]] = {"track": [], "summary": [], "failed": []}
        totals: Counter[str] = Counter()

        def flush() -> None:
            for name, file in (("track", "ten_minute_complete_candidate.parquet"), ("summary", "candidate_summary.parquet"),
                               ("failed", "unresolved_segments.parquet")):
                if not pending[name]:
                    continue
                table = pa.Table.from_pandas(pd.concat(pending[name], ignore_index=True), preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(out / file, table.schema, compression="zstd")
                writers[name].write_table(table.cast(writers[name].schema))
                pending[name].clear()

        for i, candidate in enumerate(candidates.itertuples(index=False), 1):
            frame, summary, failed = build_track(candidate, groups.get(str(candidate.vessel_uid_clean), empty), route, shard)
            validate_candidate_frame(frame)
            pending["track"].append(frame)
            pending["summary"].append(pd.DataFrame([summary]))
            if failed:
                pending["failed"].append(pd.DataFrame(failed))
            for key, value in summary.items():
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                    totals[key] += int(value)
            if i % 256 == 0:
                flush()
            if i % 2000 == 0:
                write_json(out / "status.json", {"state": "running", "shard": shard, "completed": i,
                                                 "candidates": len(candidates), "seconds": time.monotonic() - started})
        flush()
        if "failed" not in writers:
            pa_schema = pa.schema([("candidate_id", pa.string())])
            writers["failed"] = pq.ParquetWriter(out / "unresolved_segments.parquet", pa_schema, compression="zstd")
        for writer in writers.values():
            writer.close()
        if totals["grid_rows"] != int(candidates.expected_grid_rows.sum()):
            raise ValueError("shard grid rows do not close")
        result = {"state": "completed", "shard": shard, "candidates": len(candidates), "totals": dict(totals),
                  "seconds": time.monotonic() - started,
                  "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                  "outputs": [record(out / f) for f in ("ten_minute_complete_candidate.parquet", "candidate_summary.parquet",
                                                        "unresolved_segments.parquet")]}
        write_json(out / "status.json", result)
        return result
    except BaseException as error:
        write_json(out / "status.json", {"state": "failed", "shard": shard, "error": repr(error)[:4000]})
        raise


# --- Preparation and driver -------------------------------------------------------------


def prepare(root: Path, shards: list[str], probe: int | None) -> list[dict]:
    """Split the candidate root by GFW vessel shard and bind every input by hash."""
    assembly = json.loads((CANDIDATE_ROOT / "assembly_manifest.json").read_text())
    validation = json.loads((CANDIDATE_ROOT.parent / "completion_validation.json").read_text())
    if assembly["status"] != "pass" or validation["status"] != "passed" or not validation["checks"]["corrected_cities_compact"]:
        raise ValueError("step-3 candidate root is not the passed T034 build")
    parts = assembly["years"]["2025"]["parts"]
    candidates = pd.concat([pd.read_feather(verify(p), columns=CANDIDATE_COLUMNS) for p in parts], ignore_index=True)
    if len(candidates) != assembly["years"]["2025"]["row_count"] or candidates.candidate_id.duplicated().any():
        raise ValueError("candidate root rows do not close")
    for column in ("candidate_id", "vessel_uid_clean", "mmsi"):
        candidates[column] = candidates[column].astype(str)
    ports = json.loads((PORT_ROOT / "execution.json").read_text())
    shard_of = {}
    for item in ports["results"]:
        for vessel in pd.read_csv(verify(item["vessel_input_disposition"]), dtype={"vessel_uid": str}).vessel_uid:
            shard_of[vessel] = item["shard"]
    candidates["shard"] = candidates.vessel_uid_clean.map(shard_of)
    if candidates.shard.isna().any():
        raise ValueError("a candidate vessel has no GFW shard")
    candidates["expected_grid_rows"] = [len(strict_grid_ns(a, b)) for a, b in
                                        zip(candidates.departure_time, candidates.arrival_time)]
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    tasks = []
    for shard in shards:
        subset = candidates.loc[candidates.shard.eq(shard)].sort_values(["vessel_uid_clean", "departure_time"])
        if probe:
            subset = subset.head(probe)
        path = prepared / f"{shard}.parquet"
        subset.drop(columns="shard").reset_index(drop=True).to_parquet(path, index=False)
        receipt = json.loads((CLEAN_ROOT / "shards" / shard / "mask/execution.json").read_text())
        mask = CLEAN_ROOT / "shards" / shard / "mask" / f"{shard}.feather"
        points = {"path": str(mask), "bytes": mask.stat().st_size, "sha256": receipt["outputs"][str(mask)]}
        tasks.append({"shard": shard, "candidates": record(path), "points": points,
                      "output": str(root / "shards" / shard), "candidate_rows": len(subset),
                      "expected_grid_rows": int(subset.expected_grid_rows.sum())})
    write_json(root / "prepared/manifest.json", {"candidate_root": record(CANDIDATE_ROOT / "assembly_manifest.json"),
                                                  "candidate_rows_total": len(candidates), "probe_per_shard": probe,
                                                  "tasks": tasks})
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", nargs="*")
    parser.add_argument("--probe-candidates", type=int)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    root = args.out.resolve()
    if not root.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"nested numerical threads must be capped: {name}")
    shards = sorted(args.shards or SHARDS)
    if not set(shards) <= set(SHARDS):
        raise ValueError("unknown shard")
    root.mkdir(parents=True, exist_ok=False)
    complete = shards == SHARDS and not args.probe_candidates
    state = {"step": "E006 GFW version step 4: ten-minute tracks", "status": "running", "rules": RULES,
             "complete_year": complete, "shards": shards, "probe_candidates_per_shard": args.probe_candidates,
             "workers": args.workers, "host": socket.gethostname(), "started_at": now(),
             "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             "code": [record(p) for p in (Path(__file__), TRACK_STAGE / "candidate_full_ten_minute.py",
                                          TRACK_STAGE / "sea_path/routing.py", TRACK_STAGE / "sea_path/ocean_mask.py")],
             "mask": record(MASK_PATH), "network_requests": 0, "adopted": False, "results": [], "failures": []}
    write_json(root / "execution.json", state)
    started = time.monotonic()
    tasks = prepare(root, shards, args.probe_candidates)
    state["prepared"] = record(root / "prepared/manifest.json")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(work, task): task["shard"] for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                state["results"].append(result)
                print(json.dumps({"done": len(state["results"]), "shard": result["shard"], "candidates": result["candidates"],
                                  "seconds": round(result["seconds"])}), flush=True)
            except Exception as error:
                state["failures"].append({"shard": futures[future], "error": repr(error)[:2000]})
                print(f"FAILED {state['failures'][-1]}", flush=True)
            state["elapsed_seconds"] = time.monotonic() - started
            write_json(root / "execution.json", state)
    totals: Counter[str] = Counter()
    for result in state["results"]:
        totals.update(result["totals"])
    state.update(status="failed" if state["failures"] else "pass", totals=dict(totals),
                 wall_seconds=time.monotonic() - started, ended_at=now())
    state["results"].sort(key=lambda r: r["shard"])
    write_json(root / "execution.json", state)
    keys = ("grid_rows", "observed_ais_rows", "gfw_supported_interpolation_rows", "unresolved_rows", "pairs",
            "route_resolved", "cells_unresolved_voyage_end")
    print(json.dumps({"status": state["status"], **{k: totals.get(k, 0) for k in keys},
                      "wall_seconds": round(state["wall_seconds"])}, indent=2))
    if state["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
