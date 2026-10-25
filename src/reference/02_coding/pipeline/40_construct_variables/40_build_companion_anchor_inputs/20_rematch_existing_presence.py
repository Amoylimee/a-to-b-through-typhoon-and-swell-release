#!/usr/bin/env python3
"""Audit old GFW query coverage and rematch its saved Presence rows offline."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.prepared import prep

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gate_scope import require_approved_scope, sha256
from gfw_core.anchor_admissibility import classify_admissible_anchors
from gfw_core.presence_query import hour_outer_bounds


PADDING_DEGREES = 0.011
CELL_DEGREES = 0.01
TOLERANCE_KM = 0.001
EMPTY_AUDIT_COLUMNS = [
    "exact_grid_centres", "feasible_grid_centres", "multiplicity_grid_centres",
    "exact_identity_time_hit", "motion_feasible_hit", "anchor_status",
    "admissible_unique_anchor", "anchor_lat", "anchor_lon",
]


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", mode="w", delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def block_parts(ordinal: int) -> tuple[str, str]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=6 * ordinal)
    hour = start.hour
    return start.date().isoformat(), f"{hour:02d}_{hour + 6:02d}"


def cached_query_geometry(plan_dir: Path):
    body_paths = sorted((plan_dir / "shards").glob("*/query_body.json"))
    if not body_paths:
        return None, []
    geometries = []
    records = []
    for path in body_paths:
        payload = json.loads(path.read_text())
        features = payload.get("geojson", {}).get("features", [])
        if not features:
            raise ValueError(f"query body has no feature: {path}")
        geometries.extend(shape(feature["geometry"]) for feature in features)
        records.append({"path": str(path), "sha256": sha256(path)})
    merged = unary_union(geometries)
    if merged.is_empty or not merged.is_valid:
        raise ValueError(f"cached query geometry is invalid: {plan_dir}")
    return merged, records


def classify_cache_coverage(targets: pd.DataFrame, geometry) -> pd.DataFrame:
    result = targets.copy()
    if result.empty:
        result["cache_coverage"] = pd.Series(dtype="string")
        return result
    if geometry is None:
        result["cache_coverage"] = "none"
        return result
    prepared = prep(geometry)
    verdicts = []
    for row in result.to_dict(orient="records"):
        south, north, intervals = hour_outer_bounds(row, padding_degrees=PADDING_DEGREES)
        regions = [box(left, south, right, north) for left, right in intervals]
        if all(prepared.covers(region) for region in regions):
            verdicts.append("full")
        elif any(prepared.intersects(region) for region in regions):
            verdicts.append("partial")
        else:
            verdicts.append("none")
    result["cache_coverage"] = pd.array(verdicts, dtype="string")
    return result


def read_presence(block_dir: Path, targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    completed_path = block_dir / "completed_manifest.json"
    if not completed_path.is_file():
        return pd.DataFrame(columns=["hour_utc", "mmsi", "hours", "lat", "lon"]), {
            "completed_manifest": None,
            "cache_block_complete": False,
        }
    completed = json.loads(completed_path.read_text())
    if completed.get("status") != "completed" or completed.get("failure") is not None:
        raise ValueError(f"cached block is not complete: {completed_path}")
    presence_path = block_dir / "merged_presence_rows.parquet"
    if targets.empty:
        return pd.DataFrame(columns=["hour_utc", "mmsi", "hours", "lat", "lon"]), {
            "completed_manifest": str(completed_path),
            "completed_manifest_sha256": sha256(completed_path),
            "cache_block_complete": True,
            "presence_path": str(presence_path),
            "presence_rows_all_mmsi": int(completed["response"]["canonical_rows_after_exact_deduplication"]),
            "presence_rows_target_mmsi": 0,
        }
    table = pq.read_table(presence_path, columns=["hour_utc", "mmsi", "hours", "lat", "lon"])
    if len(targets):
        values = pa.array(targets["mmsi"].drop_duplicates().astype(str), type=pa.string())
        table = table.filter(pc.is_in(table["mmsi"], value_set=values))
    else:
        table = table.slice(0, 0)
    return table.to_pandas(), {
        "completed_manifest": str(completed_path),
        "completed_manifest_sha256": sha256(completed_path),
        "cache_block_complete": True,
        "presence_path": str(presence_path),
        "presence_rows_all_mmsi": int(completed["response"]["canonical_rows_after_exact_deduplication"]),
        "presence_rows_target_mmsi": int(table.num_rows),
    }


def process_block(ordinal: int, target_root: Path, cache_root: Path, out: Path) -> dict:
    status_path = out / "status" / f"block_{ordinal:04d}.json"
    if status_path.is_file():
        stored = json.loads(status_path.read_text())
        hour_path = Path(stored["hour_audit"]["path"])
        cell_path = Path(stored["exact_cells"]["path"])
        if (
            stored.get("status") == "pass"
            and hour_path.is_file()
            and cell_path.is_file()
            and sha256(hour_path) == stored["hour_audit"]["sha256"]
            and sha256(cell_path) == stored["exact_cells"]["sha256"]
        ):
            return {**stored, "resumed": True}

    started = time.monotonic()
    target_path = target_root / "target_hours" / f"block_{ordinal:04d}.parquet"
    targets = pq.read_table(target_path).to_pandas()
    day, label = block_parts(ordinal)
    plan_dir = cache_root / "plans" / day / label
    block_dir = cache_root / "blocks" / day / label
    geometry, query_bodies = cached_query_geometry(plan_dir)
    covered = classify_cache_coverage(targets, geometry)
    full = covered.loc[covered["cache_coverage"].eq("full")].copy()
    presence, cache_evidence = read_presence(block_dir, full)

    if len(full):
        audited, cells, audit = classify_admissible_anchors(
            full,
            presence,
            cell_degrees=CELL_DEGREES,
            tolerance_km=TOLERANCE_KM,
        )
    else:
        audited = full.copy()
        for column in EMPTY_AUDIT_COLUMNS:
            audited[column] = pd.Series(dtype="string" if column == "anchor_status" else "float64")
        cells = pd.DataFrame(
            columns=["hour_utc", "mmsi", "hours", "lat", "lon", "target_hour_id"]
        )
        audit = {
            "target_hours": 0,
            "admissible_unique_anchor_hours": 0,
            "no_exact_identity_time_presence_hours": 0,
            "identity_multiplicity_hours": 0,
            "multiple_feasible_grid_centre_hours": 0,
            "motion_conflict_hours": 0,
        }

    not_full = covered.loc[~covered["cache_coverage"].eq("full")].copy()
    if len(not_full):
        not_full["exact_grid_centres"] = 0
        not_full["feasible_grid_centres"] = 0
        not_full["multiplicity_grid_centres"] = 0
        not_full["exact_identity_time_hit"] = False
        not_full["motion_feasible_hit"] = False
        not_full["anchor_status"] = "cache_" + not_full["cache_coverage"].astype(str) + "_coverage"
        not_full["admissible_unique_anchor"] = False
        not_full["anchor_lat"] = float("nan")
        not_full["anchor_lon"] = float("nan")
    hours = pd.concat([audited, not_full], ignore_index=True).sort_values(
        ["hour_utc", "mmsi"], kind="stable"
    )
    if len(hours) != len(targets) or hours["target_hour_id"].duplicated().any():
        raise AssertionError("cache audit does not close to unique target hours")

    hour_path = out / "hour_audit" / f"block_{ordinal:04d}.parquet"
    cell_path = out / "exact_cells" / f"block_{ordinal:04d}.parquet"
    write_parquet_atomic(hours, hour_path)
    write_parquet_atomic(cells, cell_path)
    counts = hours["cache_coverage"].value_counts().to_dict()
    result = {
        "status": "pass",
        "block_ordinal": ordinal,
        "day": day,
        "label": label,
        "target_hours": int(len(hours)),
        "cache_full_coverage_hours": int(counts.get("full", 0)),
        "cache_partial_coverage_hours": int(counts.get("partial", 0)),
        "cache_no_coverage_hours": int(counts.get("none", 0)),
        "admissible_unique_anchor_hours": int(audit["admissible_unique_anchor_hours"]),
        "no_exact_identity_time_presence_hours": int(audit["no_exact_identity_time_presence_hours"]),
        "identity_multiplicity_hours": int(audit["identity_multiplicity_hours"]),
        "multiple_feasible_grid_centre_hours": int(audit["multiple_feasible_grid_centre_hours"]),
        "motion_conflict_hours": int(audit["motion_conflict_hours"]),
        "query_bodies": query_bodies,
        "cache_evidence": cache_evidence,
        "hour_audit": {"path": str(hour_path), "rows": len(hours), "sha256": sha256(hour_path)},
        "exact_cells": {"path": str(cell_path), "rows": len(cells), "sha256": sha256(cell_path)},
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
        "gfw_requests": 0,
        "products_modified": 0,
        "resumed": False,
    }
    write_json_atomic(result, status_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--blocks", type=int, nargs="*")
    args = parser.parse_args()
    project = args.project.resolve()
    decision, _ = require_approved_scope(project, args.review_package, "anchor_response_rematch")
    target_root = args.target_root.resolve()
    cache_root = args.cache_root.resolve()
    out = args.out.resolve()
    target_manifest_path = target_root / "target_manifest.json"
    target_manifest = json.loads(target_manifest_path.read_text())
    if target_manifest["candidate_scope_sha256"] != decision["approved_scope"]["candidate_scope_sha256"]:
        raise SystemExit("target manifest candidate scope differs from CP1")
    allowed = [item["block_ordinal"] for item in target_manifest["blocks"]]
    blocks = sorted(args.blocks if args.blocks is not None else allowed)
    if not set(blocks).issubset(allowed):
        raise SystemExit("requested block is absent from target manifest")
    if args.workers < 1 or args.workers > 16:
        raise SystemExit("workers must be within [1, 16]")

    out.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=min(args.workers, max(1, len(blocks)))) as pool:
        futures = {
            pool.submit(process_block, block, target_root, cache_root, out): block
            for block in blocks
        }
        for future in as_completed(futures):
            block = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"block {block:04d}: {result['target_hours']:,} targets; "
                    f"cache full {result['cache_full_coverage_hours']:,}; "
                    f"anchors {result['admissible_unique_anchor_hours']:,}; "
                    f"{result['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as error:
                failures.append({"block_ordinal": block, "error": f"{type(error).__name__}: {error}"})
                print(f"block {block:04d}: FAILED {type(error).__name__}: {error}", flush=True)

    totals = {}
    for key in [
        "target_hours", "cache_full_coverage_hours", "cache_partial_coverage_hours",
        "cache_no_coverage_hours", "admissible_unique_anchor_hours",
        "no_exact_identity_time_presence_hours", "identity_multiplicity_hours",
        "multiple_feasible_grid_centre_hours", "motion_conflict_hours",
    ]:
        totals[key] = int(sum(item[key] for item in results))
    manifest = {
        "status": "failed" if failures else "pass",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operation": "anchor_response_rematch",
        "target_manifest": {"path": str(target_manifest_path), "sha256": sha256(target_manifest_path)},
        "cache_root": str(cache_root),
        "cache_summary_sha256": sha256(cache_root / "summary.json"),
        "parameters": {
            "query_tile_degrees": 1.0,
            "query_padding_degrees": PADDING_DEGREES,
            "presence_cell_degrees": CELL_DEGREES,
            "motion_tolerance_km": TOLERANCE_KM,
            "full_query_geometry_required_for_anchor_admission": True,
        },
        "blocks_requested": blocks,
        "blocks_completed": len(results),
        "totals": totals,
        "failures": sorted(failures, key=lambda item: item["block_ordinal"]),
        "gfw_requests": 0,
        "products_modified": 0,
    }
    write_json_atomic(manifest, out / "rematch_manifest.json")
    print(json.dumps(totals, indent=2), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} blocks failed")


if __name__ == "__main__":
    main()
