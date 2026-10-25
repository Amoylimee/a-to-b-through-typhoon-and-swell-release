#!/usr/bin/env python3
"""Expand the CP1-approved 4--24 hour source gaps to full UTC-hour targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gate_scope import require_approved_scope, sha256


HOUR_NS = 3_600_000_000_000
EARTH_RADIUS_KM = 6371.0
KM_PER_NAUTICAL_MILE = 1.852
SPEED_CAP_KN = 25.0


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


def haversine_speed(gaps: pd.DataFrame) -> np.ndarray:
    lat_a = np.radians(gaps["left_lat"].to_numpy(float))
    lat_b = np.radians(gaps["right_lat"].to_numpy(float))
    dlat = lat_b - lat_a
    dlon = np.radians(gaps["right_lon"].to_numpy(float) - gaps["left_lon"].to_numpy(float))
    value = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    distance_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(value, 0, 1)))
    return distance_km / KM_PER_NAUTICAL_MILE / gaps["gap_hours"].to_numpy(float)


def ceil_hour_ns(values: np.ndarray) -> np.ndarray:
    return ((values + HOUR_NS - 1) // HOUR_NS) * HOUR_NS


def build_targets(gaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {
        "gap_id", "candidate_id", "gap_class", "gap_start", "gap_end", "gap_hours",
        "gfw_eligible", "left_lat", "left_lon", "right_lat", "right_lon", "mmsi",
        "vessel_uid_clean", "layer", "source_shard",
    }
    missing = sorted(required.difference(gaps.columns))
    if missing:
        raise ValueError(f"approved gap scope missing columns: {missing}")
    if gaps["gap_id"].duplicated().any():
        raise ValueError("approved gap_id is not unique")
    if not gaps["gap_class"].eq("gfw_candidate_4_24h").all():
        raise ValueError("approved scope contains a non-4--24h gap")
    if not gaps["gfw_eligible"].all():
        raise ValueError("approved scope contains a gap not marked GFW eligible")
    if not (gaps["gap_hours"].gt(4) & gaps["gap_hours"].le(24)).all():
        raise ValueError("approved gap duration is outside (4, 24]")

    work = gaps.copy()
    work["implied_endpoint_speed_kn"] = haversine_speed(work)
    finite_coordinates = np.isfinite(
        work[["left_lat", "left_lon", "right_lat", "right_lon"]].to_numpy(float)
    ).all(axis=1)
    speed_ok = work["implied_endpoint_speed_kn"].le(SPEED_CAP_KN)
    work["target_eligibility"] = np.select(
        [~finite_coordinates, ~speed_ok],
        ["invalid_boundary_coordinates", "source_endpoint_speed_above_25kn"],
        default="eligible_approved_gap",
    )
    work["target_eligible"] = work["target_eligibility"].eq("eligible_approved_gap")

    eligible = work.loc[work["target_eligible"]].reset_index(drop=True)
    start_ns = pd.to_datetime(eligible["gap_start"], utc=True).astype("int64").to_numpy()
    end_ns = pd.to_datetime(eligible["gap_end"], utc=True).astype("int64").to_numpy()
    first_ns = ceil_hour_ns(start_ns)
    last_ns = ((end_ns - HOUR_NS) // HOUR_NS) * HOUR_NS
    counts = np.maximum(0, ((last_ns - first_ns) // HOUR_NS) + 1).astype(np.int64)
    total = int(counts.sum())
    repeated = np.repeat(np.arange(len(eligible), dtype=np.int64), counts)
    starts = np.repeat(first_ns, counts)
    group_starts = np.repeat(np.cumsum(counts) - counts, counts)
    offsets = np.arange(total, dtype=np.int64) - group_starts
    hour_ns = starts + offsets * HOUR_NS

    target = pd.DataFrame(
        {
            "gap_id": eligible["gap_id"].to_numpy()[repeated],
            "candidate_id": eligible["candidate_id"].to_numpy()[repeated],
            "mmsi": eligible["mmsi"].astype("string").to_numpy()[repeated],
            "vessel_uid_clean": eligible["vessel_uid_clean"].astype("string").to_numpy()[repeated],
            "layer": eligible["layer"].to_numpy()[repeated],
            "source_shard": eligible["source_shard"].to_numpy()[repeated],
            "hour_utc": pd.to_datetime(hour_ns, utc=True),
            "source_gap_start_utc": pd.to_datetime(start_ns[repeated], utc=True),
            "source_gap_end_utc": pd.to_datetime(end_ns[repeated], utc=True),
            "lat_a": eligible["left_lat"].to_numpy(float)[repeated],
            "lon_a": eligible["left_lon"].to_numpy(float)[repeated],
            "lat_b": eligible["right_lat"].to_numpy(float)[repeated],
            "lon_b": eligible["right_lon"].to_numpy(float)[repeated],
            "effective_speed_kn": SPEED_CAP_KN,
            "bounded": True,
            "reconstruction_scope": "gfw_anchor_candidate",
        }
    )
    target["target_hour_id"] = pd.Series(dtype="string", index=target.index)
    if len(target):
        target["target_hour_id"] = (
            target["mmsi"].astype("string")
            + "|"
            + target["hour_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        )
    target["block_ordinal"] = (
        (target["hour_utc"].dt.dayofyear - 1) * 4 + target["hour_utc"].dt.hour // 6
    ).astype("int16")

    duplicate = target.duplicated(["mmsi", "hour_utc"], keep=False)
    conflicts = target.loc[duplicate].copy()
    if len(conflicts):
        disagreement = conflicts.groupby(["mmsi", "hour_utc"], observed=True).agg(
            gaps=("gap_id", "nunique"), candidates=("candidate_id", "nunique")
        )
        if (disagreement[["gaps", "candidates"]] > 1).any(axis=None):
            raise ValueError(
                f"{len(disagreement):,} MMSI/hour keys belong to conflicting approved gaps"
            )
        target.drop_duplicates(["mmsi", "hour_utc"], inplace=True)

    audit = {
        "approved_gap_rows": int(len(work)),
        "eligible_gap_rows": int(work["target_eligible"].sum()),
        "source_endpoint_speed_above_25kn_gap_rows": int(
            work["target_eligibility"].eq("source_endpoint_speed_above_25kn").sum()
        ),
        "invalid_boundary_coordinate_gap_rows": int(
            work["target_eligibility"].eq("invalid_boundary_coordinates").sum()
        ),
        "full_utc_target_hours": int(len(target)),
        "approved_gap_continuous_hours": float(work["gap_hours"].sum()),
        "eligible_gap_continuous_hours": float(eligible["gap_hours"].sum()),
        "eligible_partial_boundary_hours": float(eligible["gap_hours"].sum() - len(target)),
        "duplicate_target_rows_removed": int(duplicate.sum()),
        "maximum_implied_endpoint_speed_kn": float(work["implied_endpoint_speed_kn"].max()),
    }
    return work, target, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--block-stop", type=int, default=1460)
    args = parser.parse_args()
    project = args.project.resolve()
    decision, review = require_approved_scope(
        project, args.review_package, "anchor_target_build"
    )
    if not (0 <= args.block_start < args.block_stop <= 1460):
        raise SystemExit("block range must lie within [0, 1460]")

    gap_path = Path(review["scope_binding"]["proposed_gfw_gap_scope"]["path"])
    gaps = pq.read_table(gap_path).to_pandas()
    gap_audit, target, audit = build_targets(gaps)
    target = target.loc[
        target["block_ordinal"].between(args.block_start, args.block_stop - 1)
    ].copy()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(gap_audit, out / "gap_eligibility.parquet")
    block_records = []
    for block in range(args.block_start, args.block_stop):
        frame = target.loc[target["block_ordinal"].eq(block)].drop(columns="block_ordinal")
        path = out / "target_hours" / f"block_{block:04d}.parquet"
        write_parquet_atomic(frame, path)
        block_records.append(
            {"block_ordinal": block, "target_hours": len(frame), "path": str(path), "sha256": sha256(path)}
        )
    manifest = {
        "status": "pass",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operation": "anchor_target_build",
        "year": 2025,
        "block_range": [args.block_start, args.block_stop],
        "complete_year": args.block_start == 0 and args.block_stop == 1460,
        "speed_cap_kn": SPEED_CAP_KN,
        "cp1_decision_sha256": sha256(project / "02_coding/data/review_gates/CP1_user_decision.json"),
        "input_snapshot_sha256": decision["input_snapshot_sha256"],
        "candidate_scope_sha256": decision["approved_scope"]["candidate_scope_sha256"],
        "gfw_gap_scope": {"path": str(gap_path), "sha256": sha256(gap_path)},
        "audit": {**audit, "selected_block_target_hours": int(len(target))},
        "gap_eligibility": {
            "path": str(out / "gap_eligibility.parquet"),
            "rows": len(gap_audit),
            "sha256": sha256(out / "gap_eligibility.parquet"),
        },
        "blocks": block_records,
        "source_implementation": {
            "repository": "<upstream-code-repository>",
            "commit": "ca7ee5d8ed42db0389c55c21eea6d6e5ea5f9139",
            "adaptation": "generic CP1 candidate gaps; no Suez/Bab or route facts",
        },
        "gfw_requests": 0,
        "products_modified": 0,
    }
    (out / "target_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["audit"], indent=2), flush=True)


if __name__ == "__main__":
    main()
