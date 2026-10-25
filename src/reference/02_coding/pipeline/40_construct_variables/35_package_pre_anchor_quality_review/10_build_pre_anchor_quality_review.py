#!/usr/bin/env python3
"""Build the E001 2025 CP1 pre-anchor trajectory-quality review package.

The unit of work is one cleaned-AIS shard. Each accepted candidate is assigned
to the single shard containing its ``vessel_uid_clean``. The stage reads the
generic 30e candidate root, never a Suez/Cape route product.

The diagnostic partitions every voyage window into four mutually exclusive
time classes before any GFW data are used:

* ``ais_supported_le4h``: between two AIS fixes no more than four hours apart;
* ``gfw_candidate_4_24h``: an internal gap over four and at most 24 hours;
* ``unresolved_internal_gt24h``: an internal gap over 24 hours;
* ``endpoint_or_no_ais_unknown``: time outside the first/last fix, including
  windows with no AIS fix. This remains unknown because extrapolation is banned.

This stage identifies the proposed GFW scope and then stops at CP1. It does not
query GFW, interpolate positions, match weather, or modify a product root.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import resource
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
CODING_ROOT = HERE.parents[2]
REPO_ROOT = CODING_ROOT.parent
DEFAULT_CANDIDATE_ROOT = (
    CODING_ROOT / ".runs/2026-09-10_1030_E001-2025-candidate-root"
)
DEFAULT_CLEANED_VIEW = (
    CODING_ROOT / ".runs/2026-09-10_1040_E001-2025-ordered-measurement-input"
)
DEFAULT_QTRIM_INPUT = (
    CODING_ROOT
    / ".runs/2026-09-10_1020_E001-2025-formal-qtrim-input/complete_pretrim_pool_manifest_2025.json"
)
DEFAULT_QTRIM_MANIFEST = (
    CODING_ROOT
    / ".runs/2026-09-10_1021_E001-2025-formal-qtrim/manifests/final_sample_manifest_2025.json"
)
DEFAULT_SHARD_MAP = (
    CODING_ROOT
    / ".runs/2026-09-10_1100_E001-2025-ordered-measurement-full/shard_maps/api_ship_to_shard_2025.json"
)

NS_PER_HOUR = 3_600_000_000_000
SHORT_GAP_HOURS = 4.0
GFW_MAX_GAP_HOURS = 24.0
SHORT_GAP_NS = np.int64(SHORT_GAP_HOURS * NS_PER_HOUR)
GFW_MAX_GAP_NS = np.int64(GFW_MAX_GAP_HOURS * NS_PER_HOUR)

CANDIDATE_COLUMNS = [
    "candidate_id",
    "mmsi",
    "vessel_uid_clean",
    "departure_time",
    "arrival_time",
    "total_hours",
    "shiptype_group",
    "origin_city",
    "dest_city",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False, compression="zstd")
    os.replace(partial, path)


def normalize_identifier(values: pd.Series, field: str) -> pd.Series:
    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{field} contains missing/blank identifiers")
    return result


def gap_id(candidate_id: str, start_ns: int, end_ns: int, gap_class: str) -> str:
    raw = f"{candidate_id}|{start_ns}|{end_ns}|{gap_class}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def wrapped_midpoint_lon(left: float, right: float) -> float:
    delta = ((right - left + 180.0) % 360.0) - 180.0
    return ((left + delta / 2.0 + 180.0) % 360.0) - 180.0


def region_10deg(lat: float, lon: float) -> str:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return "unknown"
    lat_bin = int(math.floor(max(-90.0, min(89.999999, lat)) / 10.0) * 10)
    lon_bin = int(math.floor(max(-180.0, min(179.999999, lon)) / 10.0) * 10)
    return f"lat[{lat_bin},{lat_bin + 10})_lon[{lon_bin},{lon_bin + 10})"


def classify_window(
    candidate_id: str,
    start_ns: int,
    end_ns: int,
    utc_ns: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Classify one voyage window and return metrics plus explicit gaps."""

    left = int(np.searchsorted(utc_ns, start_ns, side="left"))
    right = int(np.searchsorted(utc_ns, end_ns, side="right"))
    window_ns = int(end_ns - start_ns)
    if window_ns <= 0:
        raise ValueError(f"non-positive voyage window: {candidate_id}")

    gaps: list[dict[str, Any]] = []
    supported_ns = 0
    gfw_ns = 0
    unresolved_gt24_ns = 0
    endpoint_ns = 0
    longest_internal_ns = 0

    def add_gap(
        gap_start: int,
        gap_end: int,
        gap_class: str,
        left_lat: float = np.nan,
        left_lon: float = np.nan,
        right_lat: float = np.nan,
        right_lon: float = np.nan,
    ) -> None:
        duration_ns = int(gap_end - gap_start)
        midpoint_lat = (
            float((left_lat + right_lat) / 2.0)
            if np.isfinite(left_lat) and np.isfinite(right_lat)
            else np.nan
        )
        midpoint_lon = (
            wrapped_midpoint_lon(float(left_lon), float(right_lon))
            if np.isfinite(left_lon) and np.isfinite(right_lon)
            else np.nan
        )
        gaps.append(
            {
                "gap_id": gap_id(candidate_id, gap_start, gap_end, gap_class),
                "candidate_id": candidate_id,
                "gap_class": gap_class,
                "gap_start": pd.Timestamp(gap_start, unit="ns", tz="UTC"),
                "gap_end": pd.Timestamp(gap_end, unit="ns", tz="UTC"),
                "gap_hours": duration_ns / NS_PER_HOUR,
                "gfw_eligible": gap_class == "gfw_candidate_4_24h",
                "left_lat": left_lat,
                "left_lon": left_lon,
                "right_lat": right_lat,
                "right_lon": right_lon,
                "midpoint_lat": midpoint_lat,
                "midpoint_lon": midpoint_lon,
                "region_10deg": region_10deg(midpoint_lat, midpoint_lon),
            }
        )

    if right <= left:
        endpoint_ns = window_ns
        add_gap(start_ns, end_ns, "endpoint_or_no_ais_unknown")
        return (
            {
                "ais_points_in_window": 0,
                "ais_supported_le4h_hours": 0.0,
                "gfw_candidate_4_24h_hours": 0.0,
                "unresolved_internal_gt24h_hours": 0.0,
                "endpoint_or_no_ais_unknown_hours": endpoint_ns / NS_PER_HOUR,
                "gfw_candidate_gap_count": 0,
                "unresolved_gt24_gap_count": 0,
                "endpoint_gap_count": 1,
                "longest_internal_gap_hours": np.nan,
            },
            gaps,
        )

    times = utc_ns[left:right]
    lats = lat[left:right]
    lons = lon[left:right]

    if times[0] > start_ns:
        endpoint_ns += int(times[0] - start_ns)
        add_gap(start_ns, int(times[0]), "endpoint_or_no_ais_unknown")
    if times[-1] < end_ns:
        endpoint_ns += int(end_ns - times[-1])
        add_gap(int(times[-1]), end_ns, "endpoint_or_no_ais_unknown")

    if len(times) > 1:
        intervals = np.diff(times)
        longest_internal_ns = int(intervals.max())
        for i, duration in enumerate(intervals):
            duration = int(duration)
            if duration <= SHORT_GAP_NS:
                supported_ns += duration
            elif duration <= GFW_MAX_GAP_NS:
                gfw_ns += duration
                add_gap(
                    int(times[i]),
                    int(times[i + 1]),
                    "gfw_candidate_4_24h",
                    float(lats[i]),
                    float(lons[i]),
                    float(lats[i + 1]),
                    float(lons[i + 1]),
                )
            else:
                unresolved_gt24_ns += duration
                add_gap(
                    int(times[i]),
                    int(times[i + 1]),
                    "unresolved_internal_gt24h",
                    float(lats[i]),
                    float(lons[i]),
                    float(lats[i + 1]),
                    float(lons[i + 1]),
                )

    closure = supported_ns + gfw_ns + unresolved_gt24_ns + endpoint_ns
    if closure != window_ns:
        raise AssertionError(
            f"time closure failed for {candidate_id}: {closure} != {window_ns}"
        )

    return (
        {
            "ais_points_in_window": int(len(times)),
            "ais_supported_le4h_hours": supported_ns / NS_PER_HOUR,
            "gfw_candidate_4_24h_hours": gfw_ns / NS_PER_HOUR,
            "unresolved_internal_gt24h_hours": unresolved_gt24_ns / NS_PER_HOUR,
            "endpoint_or_no_ais_unknown_hours": endpoint_ns / NS_PER_HOUR,
            "gfw_candidate_gap_count": sum(
                item["gap_class"] == "gfw_candidate_4_24h" for item in gaps
            ),
            "unresolved_gt24_gap_count": sum(
                item["gap_class"] == "unresolved_internal_gt24h" for item in gaps
            ),
            "endpoint_gap_count": sum(
                item["gap_class"] == "endpoint_or_no_ais_unknown" for item in gaps
            ),
            "longest_internal_gap_hours": (
                longest_internal_ns / NS_PER_HOUR if longest_internal_ns else 0.0
            ),
        },
        gaps,
    )


def validate_existing_status(status_path: Path) -> dict[str, Any] | None:
    if not status_path.exists():
        return None
    try:
        status = json.loads(status_path.read_text())
        if status.get("status") != "pass":
            return None
        for key in ("candidate_output", "gap_output"):
            record = status[key]
            path = Path(record["path"])
            if not path.exists() or sha256_file(path) != record["sha256"]:
                return None
        return status
    except Exception:
        return None


def process_shard(
    shard_path: str,
    candidates: pd.DataFrame,
    output_root: str,
) -> dict[str, Any]:
    shard = Path(shard_path)
    output = Path(output_root)
    stem = shard.stem
    status_path = output / "status" / f"{stem}.json"
    existing = validate_existing_status(status_path)
    if existing is not None:
        existing["resumed"] = True
        return existing

    started = time.monotonic()
    candidate_path = output / "per_shard_candidates" / f"{stem}.parquet"
    gap_path = output / "per_shard_gaps" / f"{stem}.parquet"
    for path in (candidate_path, gap_path, status_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    table = feather.read_table(
        shard, columns=["vessel_uid", "mmsi", "UTC", "lat", "lon"], memory_map=True
    )
    points = table.to_pandas()
    del table
    points["vessel_uid"] = normalize_identifier(points["vessel_uid"], "AIS vessel_uid")
    points["mmsi"] = normalize_identifier(points["mmsi"], "AIS mmsi")
    points["UTC"] = pd.to_datetime(points["UTC"], utc=True)
    if points["UTC"].isna().any():
        raise ValueError(f"invalid UTC in {shard}")

    wanted = candidates[["vessel_uid_clean", "mmsi"]].drop_duplicates()
    wanted_pairs = set(map(tuple, wanted.itertuples(index=False, name=None)))
    wanted_api_ids = set(wanted["vessel_uid_clean"])
    point_pairs = list(zip(points["vessel_uid"], points["mmsi"]))
    keep = np.fromiter((pair in wanted_pairs for pair in point_pairs), dtype=bool)
    relevant_api = points["vessel_uid"].isin(wanted_api_ids).to_numpy()
    identity_mismatch_rows = int((relevant_api & ~keep).sum())
    irrelevant_vessel_rows = int((~relevant_api).sum())
    points = points.loc[keep].copy()
    points.sort_values(["vessel_uid", "mmsi", "UTC"], kind="stable", inplace=True)

    dup_mask = points.duplicated(["vessel_uid", "mmsi", "UTC"], keep=False)
    conflicting_duplicate_groups = 0
    if dup_mask.any():
        duplicated = points.loc[dup_mask]
        variability = duplicated.groupby(
            ["vessel_uid", "mmsi", "UTC"], sort=False
        ).agg(lat_n=("lat", "nunique"), lon_n=("lon", "nunique"))
        conflicting_duplicate_groups = int(
            ((variability["lat_n"] > 1) | (variability["lon_n"] > 1)).sum()
        )
        if conflicting_duplicate_groups:
            raise ValueError(
                f"{shard.name}: {conflicting_duplicate_groups} conflicting duplicate UTC groups"
            )
    duplicate_rows_removed = int(
        points.duplicated(["vessel_uid", "mmsi", "UTC"], keep="first").sum()
    )
    points.drop_duplicates(["vessel_uid", "mmsi", "UTC"], keep="first", inplace=True)

    point_groups: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, group in points.groupby(["vessel_uid", "mmsi"], sort=False):
        point_groups[key] = (
            group["UTC"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
            group["lat"].to_numpy(dtype=np.float64),
            group["lon"].to_numpy(dtype=np.float64),
        )

    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        key = (row.vessel_uid_clean, row.mmsi)
        arrays = point_groups.get(
            key,
            (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            ),
        )
        start_ns = int(pd.Timestamp(row.departure_time).value)
        end_ns = int(pd.Timestamp(row.arrival_time).value)
        metrics, gaps = classify_window(
            row.candidate_id, start_ns, end_ns, arrays[0], arrays[1], arrays[2]
        )
        window_hours = (end_ns - start_ns) / NS_PER_HOUR
        accounted = (
            metrics["ais_supported_le4h_hours"]
            + metrics["gfw_candidate_4_24h_hours"]
            + metrics["unresolved_internal_gt24h_hours"]
            + metrics["endpoint_or_no_ais_unknown_hours"]
        )
        metric_rows.append(
            {
                "candidate_id": row.candidate_id,
                "mmsi": row.mmsi,
                "vessel_uid_clean": row.vessel_uid_clean,
                "layer": row.layer,
                "source_shard": stem,
                "departure_time": row.departure_time,
                "arrival_time": row.arrival_time,
                "departure_month": pd.Timestamp(row.departure_time).strftime("%Y-%m"),
                "shiptype_group": row.shiptype_group,
                "origin_city": row.origin_city,
                "dest_city": row.dest_city,
                "window_hours": window_hours,
                **metrics,
                "time_closure_error_hours": accounted - window_hours,
                "ais_supported_share": metrics["ais_supported_le4h_hours"] / window_hours,
                "gfw_candidate_share": metrics["gfw_candidate_4_24h_hours"] / window_hours,
                "unresolved_share_before_gfw": (
                    metrics["gfw_candidate_4_24h_hours"]
                    + metrics["unresolved_internal_gt24h_hours"]
                    + metrics["endpoint_or_no_ais_unknown_hours"]
                )
                / window_hours,
            }
        )
        for item in gaps:
            item.update(
                {
                    "mmsi": row.mmsi,
                    "vessel_uid_clean": row.vessel_uid_clean,
                    "layer": row.layer,
                    "source_shard": stem,
                    "departure_month": pd.Timestamp(row.departure_time).strftime("%Y-%m"),
                    "shiptype_group": row.shiptype_group,
                    "origin_city": row.origin_city,
                    "dest_city": row.dest_city,
                }
            )
            gap_rows.append(item)

    candidate_frame = pd.DataFrame(metric_rows)
    gap_frame = pd.DataFrame(gap_rows)
    if gap_frame.empty:
        gap_frame = pd.DataFrame(
            columns=[
                "gap_id",
                "candidate_id",
                "gap_class",
                "gap_start",
                "gap_end",
                "gap_hours",
                "gfw_eligible",
            ]
        )
    write_parquet(candidate_frame, candidate_path)
    write_parquet(gap_frame, gap_path)

    status = {
        "status": "pass",
        "shard": stem,
        "input_path": str(shard.resolve()),
        "input_rows": int(
            len(points)
            + identity_mismatch_rows
            + irrelevant_vessel_rows
            + duplicate_rows_removed
        ),
        "candidate_rows": int(len(candidate_frame)),
        "gap_rows": int(len(gap_frame)),
        "gfw_eligible_gaps": int(gap_frame.get("gfw_eligible", pd.Series(dtype=bool)).sum()),
        "identity_mismatch_rows_excluded": identity_mismatch_rows,
        "irrelevant_vessel_rows_skipped": irrelevant_vessel_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "conflicting_duplicate_groups": conflicting_duplicate_groups,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "candidate_output": {
            "path": str(candidate_path.resolve()),
            "rows": int(len(candidate_frame)),
            "sha256": sha256_file(candidate_path),
        },
        "gap_output": {
            "path": str(gap_path.resolve()),
            "rows": int(len(gap_frame)),
            "sha256": sha256_file(gap_path),
        },
        "resumed": False,
    }
    write_json(status_path, status)
    return status


def load_candidates(
    candidate_root: Path, qtrim_input_manifest: Path, shard_map_path: Path
) -> tuple[pd.DataFrame, dict[str, str]]:
    parts = sorted((candidate_root / "2025").glob("part-*.feather"))
    if len(parts) != 16:
        raise ValueError(f"expected 16 candidate parts, found {len(parts)}")
    candidates = pd.concat(
        [feather.read_table(path, columns=CANDIDATE_COLUMNS).to_pandas() for path in parts],
        ignore_index=True,
    )
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate_id in candidate root")
    candidates["mmsi"] = normalize_identifier(candidates["mmsi"], "candidate mmsi")
    candidates["vessel_uid_clean"] = normalize_identifier(
        candidates["vessel_uid_clean"], "candidate vessel_uid_clean"
    )
    candidates["departure_time"] = pd.to_datetime(candidates["departure_time"], utc=True)
    candidates["arrival_time"] = pd.to_datetime(candidates["arrival_time"], utc=True)

    qtrim_input = json.loads(qtrim_input_manifest.read_text())
    layer_frames = []
    for partition in qtrim_input["partitions"]:
        for source in partition["sources"]:
            frame = feather.read_table(source["path"], columns=["candidate_id"]).to_pandas()
            frame["layer"] = source["layer"]
            layer_frames.append(frame)
    layer_map = pd.concat(layer_frames, ignore_index=True)
    if layer_map["candidate_id"].duplicated().any():
        raise ValueError("candidate_id appears in multiple L1/L2/L3 inputs")
    candidates = candidates.merge(layer_map, on="candidate_id", how="left", validate="one_to_one")
    if candidates["layer"].isna().any():
        raise ValueError("final candidates missing L1/L2/L3 provenance")

    shard_payload = json.loads(shard_map_path.read_text())
    shard_index = {str(k): str(v) for k, v in shard_payload["index"].items()}
    candidates["source_shard"] = candidates["vessel_uid_clean"].map(shard_index)
    if candidates["source_shard"].isna().any():
        missing = candidates.loc[candidates["source_shard"].isna(), "vessel_uid_clean"].unique()
        raise ValueError(f"{len(missing)} candidate identities absent from cleaned-AIS shard map")
    return candidates, shard_index


def aggregate_block(group: pd.DataFrame) -> pd.Series:
    window = float(group["window_hours"].sum())
    ais = float(group["ais_supported_le4h_hours"].sum())
    gfw = float(group["gfw_candidate_4_24h_hours"].sum())
    unresolved_gt24 = float(group["unresolved_internal_gt24h_hours"].sum())
    endpoint = float(group["endpoint_or_no_ais_unknown_hours"].sum())
    return pd.Series(
        {
            "candidates": int(len(group)),
            "vessels": int(group["vessel_uid_clean"].nunique()),
            "window_hours": window,
            "ais_supported_le4h_hours": ais,
            "gfw_candidate_4_24h_hours": gfw,
            "unresolved_internal_gt24h_hours": unresolved_gt24,
            "endpoint_or_no_ais_unknown_hours": endpoint,
            "ais_supported_pct": 100 * ais / window if window else np.nan,
            "gfw_candidate_pct": 100 * gfw / window if window else np.nan,
            "unresolved_gt24_pct": 100 * unresolved_gt24 / window if window else np.nan,
            "endpoint_unknown_pct": 100 * endpoint / window if window else np.nan,
            "voyages_with_gfw_candidate_pct": 100
            * (group["gfw_candidate_gap_count"] > 0).mean(),
            "voyages_with_gt24_gap_pct": 100
            * (group["unresolved_gt24_gap_count"] > 0).mean(),
            "voyages_with_no_ais_pct": 100 * (group["ais_points_in_window"] == 0).mean(),
            "median_ais_supported_share_pct": 100 * group["ais_supported_share"].median(),
        }
    )


def build_tables(candidates: pd.DataFrame, gaps: pd.DataFrame, tables: Path) -> dict[str, Any]:
    tables.mkdir(parents=True, exist_ok=True)
    headline = aggregate_block(candidates).to_frame().T
    headline.to_csv(tables / "01_headline_time_support.csv", index=False)
    for number, field, name in [
        (2, "layer", "by_layer"),
        (3, "departure_month", "by_month"),
        (4, "shiptype_group", "by_shiptype"),
    ]:
        out = candidates.groupby(field, dropna=False).apply(
            aggregate_block, include_groups=False
        )
        out.to_csv(tables / f"{number:02d}_{name}.csv")

    gap_summary = gaps.groupby(["gap_class"], dropna=False).agg(
        gaps=("gap_id", "size"),
        candidates=("candidate_id", "nunique"),
        vessels=("vessel_uid_clean", "nunique"),
        hours=("gap_hours", "sum"),
        median_hours=("gap_hours", "median"),
        max_hours=("gap_hours", "max"),
    )
    gap_summary.to_csv(tables / "05_gap_classes.csv")

    eligible = gaps.loc[gaps["gfw_eligible"]].copy()
    for number, field, name in [
        (6, "layer", "gfw_scope_by_layer"),
        (7, "departure_month", "gfw_scope_by_month"),
        (8, "shiptype_group", "gfw_scope_by_shiptype"),
        (9, "region_10deg", "gfw_scope_by_region_10deg"),
    ]:
        out = eligible.groupby(field, dropna=False).agg(
            gaps=("gap_id", "size"),
            candidates=("candidate_id", "nunique"),
            vessels=("vessel_uid_clean", "nunique"),
            hours=("gap_hours", "sum"),
        )
        out.sort_values("hours", ascending=False).to_csv(tables / f"{number:02d}_{name}.csv")

    examples = pd.concat(
        [
            candidates.nlargest(8, "gfw_candidate_4_24h_hours").assign(
                example_reason="largest_gfw_candidate_hours"
            ),
            candidates.nlargest(8, "unresolved_internal_gt24h_hours").assign(
                example_reason="largest_internal_gt24h_hours"
            ),
            candidates.nlargest(8, "endpoint_or_no_ais_unknown_hours").assign(
                example_reason="largest_endpoint_or_no_ais_unknown"
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("candidate_id")
    examples.to_csv(tables / "10_review_examples.csv", index=False)

    return {
        "headline": headline.iloc[0].to_dict(),
        "gfw_eligible_gaps": int(len(eligible)),
        "gfw_eligible_candidates": int(eligible["candidate_id"].nunique()),
        "gfw_eligible_vessels": int(eligible["vessel_uid_clean"].nunique()),
    }


def build_plot(examples: pd.DataFrame, cleaned_root: Path, figures: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    selected = examples.head(12).copy()
    if selected.empty:
        return ""
    ncols = 3
    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False)
    cache: dict[str, pd.DataFrame] = {}
    for ax, row in zip(axes.ravel(), selected.itertuples(index=False)):
        if row.source_shard not in cache:
            cache[row.source_shard] = feather.read_table(
                cleaned_root / f"{row.source_shard}.feather",
                columns=["vessel_uid", "mmsi", "UTC", "lat", "lon"],
            ).to_pandas()
        points = cache[row.source_shard]
        expected_api = str(row.vessel_uid_clean).removesuffix(".0")
        expected_mmsi = str(row.mmsi).removesuffix(".0")
        mask = (
            points["vessel_uid"].astype(str).str.replace(r"\.0$", "", regex=True)
            == expected_api
        ) & (
            points["mmsi"].astype(str).str.replace(r"\.0$", "", regex=True)
            == expected_mmsi
        )
        track = points.loc[mask].copy()
        track["UTC"] = pd.to_datetime(track["UTC"], utc=True)
        track = track.loc[
            (track["UTC"] >= row.departure_time) & (track["UTC"] <= row.arrival_time)
        ].sort_values("UTC")
        if len(track):
            lon = np.rad2deg(np.unwrap(np.deg2rad(track["lon"].to_numpy())))
            lat = track["lat"].to_numpy()
            times = track["UTC"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            breaks = np.flatnonzero(np.diff(times) > SHORT_GAP_NS)
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks + 1, len(track)]
            for start, end in zip(starts, ends):
                ax.plot(lon[start:end], lat[start:end], color="#2463a8", linewidth=0.8)
            ax.scatter(lon, lat, s=2, color="#2463a8", label="AIS fixes")
            if len(breaks):
                durations = np.diff(times)[breaks]
                eligible = durations <= GFW_MAX_GAP_NS
                ax.scatter(
                    lon[breaks[eligible]],
                    lat[breaks[eligible]],
                    s=18,
                    facecolors="none",
                    edgecolors="#d97706",
                    label=">4–24h gap edge",
                )
                ax.scatter(
                    lon[breaks[~eligible]],
                    lat[breaks[~eligible]],
                    s=18,
                    facecolors="none",
                    edgecolors="#b42318",
                    label=">24h gap edge",
                )
        else:
            ax.text(0.5, 0.5, "no AIS in voyage window", ha="center", va="center")
        ax.set_title(
            f"{row.example_reason}\n{row.candidate_id[:18]} | {row.layer}", fontsize=8
        )
        ax.set_xlabel("unwrapped longitude")
        ax.set_ylabel("latitude")
        ax.grid(alpha=0.2)
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=7, loc="best")
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    fig.suptitle("CP1 review examples: AIS observations before GFW", fontsize=14)
    fig.tight_layout()
    path = figures / "01_representative_and_anomalous_tracks.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def build_report(output: Path, manifest: dict[str, Any]) -> None:
    h = manifest["headline"]
    pct = lambda key: f"{float(h[key]):.2f}%"
    report = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>E001 2025 CP1</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3df;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eef2f7;padding:2px 5px}}.stop{{border-left:5px solid #b42318;background:#fff1f0;padding:12px}}img{{max-width:100%}}</style></head>
<body><h1>E001 2025 CP1：GFW 锚点前轨迹质量</h1>
<p class="stop"><strong>当前停点：</strong>本包只识别拟进入 GFW 的质量不足区间。尚未查询 GFW、插值或匹配天气；实际范围须由用户审核。</p>
<h2>时间支持概览</h2><table><tr><th>候选航次</th><th>船舶</th><th>AIS ≤4h 支撑</th><th>拟用 GFW 的 4–24h 内部缺口</th><th>&gt;24h 内部未知</th><th>端点/无 AIS 未知</th></tr>
<tr><td>{int(h['candidates']):,}</td><td>{int(h['vessels']):,}</td><td>{pct('ais_supported_pct')}</td><td>{pct('gfw_candidate_pct')}</td><td>{pct('unresolved_gt24_pct')}</td><td>{pct('endpoint_unknown_pct')}</td></tr></table>
<p>拟进入 GFW 范围：{manifest['gfw_eligible_gaps']:,} 个 gap，{manifest['gfw_eligible_candidates']:,} 条航次，{manifest['gfw_eligible_vessels']:,} 艘船。</p>
<h2>定义</h2><ul><li>只把两侧均有 AIS、间隔严格大于 4 小时且不超过 24 小时的内部 gap 列为 GFW candidate。</li><li>超过 24 小时、航次首尾端点缺口和窗口内没有 AIS 的时间保持未知。</li><li>10 分钟最终轨迹尚未生成；它位于 GFW、CP2 和海路约束插值之后。</li></ul>
<h2>检查材料</h2><p>先看 <code>tables/01_headline_time_support.csv</code>、<code>tables/05_gap_classes.csv</code>、<code>tables/06_gfw_scope_by_layer.csv</code> 和 <code>tables/10_review_examples.csv</code>。逐航次和逐 gap 明细分别在 <code>candidate_quality.parquet</code> 与 <code>gap_intervals.parquet</code>。</p>
<img src="figures/01_representative_and_anomalous_tracks.png" alt="AIS review examples">
<h2>输入快照</h2><p><code>{html.escape(manifest['input_snapshot_sha256'])}</code></p>
</body></html>"""
    (output / "report.html").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--cleaned-view", type=Path, default=DEFAULT_CLEANED_VIEW)
    parser.add_argument("--qtrim-input-manifest", type=Path, default=DEFAULT_QTRIM_INPUT)
    parser.add_argument("--qtrim-manifest", type=Path, default=DEFAULT_QTRIM_MANIFEST)
    parser.add_argument("--shard-map", type=Path, default=DEFAULT_SHARD_MAP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only-shards", nargs="*", default=None)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must be between 1 and 12")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cleaned_manifest = args.cleaned_view / "cleaned_ais_view_manifest.json"
    candidate_manifest = args.candidate_root / "assembly_manifest.json"
    required = [
        cleaned_manifest,
        candidate_manifest,
        args.qtrim_input_manifest,
        args.qtrim_manifest,
        args.shard_map,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    inputs = {
        "candidate_root_manifest": {
            "path": str(candidate_manifest.resolve()),
            "sha256": sha256_file(candidate_manifest),
        },
        "cleaned_ais_view_manifest": {
            "path": str(cleaned_manifest.resolve()),
            "sha256": sha256_file(cleaned_manifest),
        },
        "qtrim_input_manifest": {
            "path": str(args.qtrim_input_manifest.resolve()),
            "sha256": sha256_file(args.qtrim_input_manifest),
        },
        "qtrim_manifest": {
            "path": str(args.qtrim_manifest.resolve()),
            "sha256": sha256_file(args.qtrim_manifest),
        },
        "shard_map": {
            "path": str(args.shard_map.resolve()),
            "sha256": sha256_file(args.shard_map),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }
    input_snapshot_sha256 = canonical_hash(inputs)

    candidates, _ = load_candidates(
        args.candidate_root, args.qtrim_input_manifest, args.shard_map
    )
    cleaned_root = args.cleaned_view / "cleaned_ais/2025"
    grouped = {name: group.reset_index(drop=True) for name, group in candidates.groupby("source_shard")}
    if args.only_shards:
        wanted = {name if name.endswith(".feather") else f"{name}.feather" for name in args.only_shards}
        grouped = {name: group for name, group in grouped.items() if name in wanted}
        candidates = pd.concat(grouped.values(), ignore_index=True) if grouped else candidates.iloc[0:0]
    if not grouped:
        raise ValueError("no candidate shards selected")
    missing_files = sorted(name for name in grouped if not (cleaned_root / name).exists())
    if missing_files:
        raise FileNotFoundError(f"missing cleaned shards: {missing_files[:5]}")

    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = {
        "status": "running",
        "started_utc": started_utc,
        "pid": os.getpid(),
        "workers": args.workers,
        "total_shards": len(grouped),
        "completed_shards": 0,
        "failed_shards": 0,
        "candidate_rows": int(len(candidates)),
        "input_snapshot_sha256": input_snapshot_sha256,
    }
    write_json(output / "run_state.json", state)
    print(f"CP1 output: {output}", flush=True)
    print(
        f"Required / Actual / Status: 30e generic candidate root / {len(candidates):,} candidates / PASS",
        flush=True,
    )
    print(
        f"Required / Actual / Status: cleaned AIS / {len(grouped)} shards / PASS",
        flush=True,
    )
    print("Required / Actual / Status: GFW/weather writes / disabled / PASS", flush=True)

    statuses: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                process_shard,
                str(cleaned_root / shard_name),
                group,
                str(output),
            ): shard_name
            for shard_name, group in sorted(grouped.items())
        }
        for future in as_completed(future_map):
            shard_name = future_map[future]
            try:
                status = future.result()
                statuses.append(status)
            except Exception as exc:
                failures.append(
                    {
                        "shard": shard_name,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            state["completed_shards"] = len(statuses)
            state["failed_shards"] = len(failures)
            state["last_update_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            state["elapsed_seconds"] = (
                pd.Timestamp.now(tz="UTC") - pd.Timestamp(started_utc)
            ).total_seconds()
            if statuses:
                state["mean_completed_shard_seconds"] = float(
                    np.mean([item["elapsed_seconds"] for item in statuses])
                )
                state["max_peak_rss_gb"] = float(max(item["peak_rss_gb"] for item in statuses))
            write_json(output / "run_state.json", state)
            done = len(statuses) + len(failures)
            if done % 10 == 0 or done == len(future_map):
                print(
                    f"progress {done}/{len(future_map)}; pass={len(statuses)} fail={len(failures)}",
                    flush=True,
                )

    if failures:
        write_json(output / "failures.json", failures)
        state["status"] = "failed"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(output / "run_state.json", state)
        raise RuntimeError(f"{len(failures)} shards failed; see failures.json")

    candidate_files = sorted((output / "per_shard_candidates").glob("*.parquet"))
    gap_files = sorted((output / "per_shard_gaps").glob("*.parquet"))
    candidate_quality = pd.concat([pd.read_parquet(path) for path in candidate_files], ignore_index=True)
    gap_intervals = pd.concat([pd.read_parquet(path) for path in gap_files], ignore_index=True)
    if len(candidate_quality) != len(candidates):
        raise ValueError(
            f"candidate closure failed: {len(candidate_quality)} != {len(candidates)}"
        )
    if candidate_quality["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate quality rows")
    if not np.allclose(candidate_quality["time_closure_error_hours"], 0, atol=1e-9):
        raise ValueError("time decomposition does not close")

    write_parquet(candidate_quality, output / "candidate_quality.parquet")
    write_parquet(gap_intervals, output / "gap_intervals.parquet")
    summary = build_tables(candidate_quality, gap_intervals, output / "tables")
    examples = pd.read_csv(output / "tables/10_review_examples.csv")
    for field in ("departure_time", "arrival_time"):
        examples[field] = pd.to_datetime(examples[field], utc=True)
    plot_path = build_plot(examples, cleaned_root, output / "figures")

    manifest = {
        "status": "awaiting_user_cp1_review",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "E001 2025 fleet expansion",
        "scope": "pre-anchor trajectory quality on generic 30e candidate root",
        "read_only_inputs": True,
        "products_modified": 0,
        "gfw_requests": 0,
        "interpolation_rows_written": 0,
        "weather_rows_written": 0,
        "short_gap_hours": SHORT_GAP_HOURS,
        "gfw_max_gap_hours": GFW_MAX_GAP_HOURS,
        "endpoint_policy": "unknown; no extrapolation",
        "inputs": inputs,
        "input_snapshot_sha256": input_snapshot_sha256,
        "workers": args.workers,
        "shards": len(statuses),
        "candidate_rows": int(len(candidate_quality)),
        "gap_rows": int(len(gap_intervals)),
        "identity_mismatch_rows_excluded": int(
            sum(item["identity_mismatch_rows_excluded"] for item in statuses)
        ),
        "irrelevant_vessel_rows_skipped": int(
            sum(item["irrelevant_vessel_rows_skipped"] for item in statuses)
        ),
        "duplicate_rows_removed": int(sum(item["duplicate_rows_removed"] for item in statuses)),
        "peak_worker_rss_gb": float(max(item["peak_rss_gb"] for item in statuses)),
        "headline": summary["headline"],
        "gfw_eligible_gaps": summary["gfw_eligible_gaps"],
        "gfw_eligible_candidates": summary["gfw_eligible_candidates"],
        "gfw_eligible_vessels": summary["gfw_eligible_vessels"],
        "representative_figure": plot_path,
    }
    write_json(output / "cp1_review_manifest.json", manifest)
    build_report(output, manifest)
    (output / "00_README.md").write_text(
        "# E001 2025 CP1 pre-anchor quality review\n\n"
        "先看 `report.html`，再看 `tables/01_headline_time_support.csv`、"
        "`tables/05_gap_classes.csv` 和 `tables/10_review_examples.csv`。\n\n"
        "本包停在 CP1：没有请求 GFW、没有插值、没有天气匹配、没有产品采纳。"
        "用户审核并绑定 `cp1_review_manifest.json` 中的 input snapshot 后，"
        "才可继续实际 GFW 范围。\n",
        encoding="utf-8",
    )
    state.update(
        {
            "status": "awaiting_user_cp1_review",
            "finished_utc": manifest["generated_utc"],
            "completed_shards": len(statuses),
            "failed_shards": 0,
            "candidate_rows_completed": int(len(candidate_quality)),
            "gap_rows_completed": int(len(gap_intervals)),
        }
    )
    write_json(output / "run_state.json", state)
    print("CP1 package complete; awaiting user review", flush=True)


if __name__ == "__main__":
    main()
