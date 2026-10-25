#!/usr/bin/env python3
"""Formal E002 GFW holdout audit; source products are read only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[4]
OUT: Path
CLIMATE = Path("<upstream-code-repository>/jiachine-learning/02_coding/.runs/2026-09-05_1351_E032_annual-six-hour-acquisition-2025")
COPY = CODING / "data/legacy_inputs/gfw_presence_responses_2025"
ANCHOR = CODING / ".runs/2026-09-13_1930_E001-2025-merged-anchor-reconstruction/cache_rematch"
PANEL = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release/10_u1/10_voyage_exposure_panel.parquet"
TRACKS = CODING / "data/processed/products/companion_tracks/2025/run/shards"
START = pd.Timestamp("2025-01-01", tz="UTC")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ordinal_for(path: Path) -> int:
    day = pd.Timestamp(path.parents[1].name, tz="UTC")
    return int((day - START).days) * 4 + int(path.parent.name[:2]) // 6


def build_holdout() -> dict:
    files = sorted(CLIMATE.glob("blocks/*/*/exact_identity_time_cells.parquet"))
    if len(files) != 1459:
        raise ValueError(f"climate GFW block files: {len(files)} != 1459")
    if sha256(COPY / "summary.json") != sha256(CLIMATE / "summary.json"):
        raise ValueError("local raw GFW copy and upstream summary differ")
    manifest = [{"path": str(path), "sha256": sha256(path)} for path in files]
    (OUT / "climate_derived_source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    parts = []
    totals = {"gfw_exact_rows": 0, "gfw_admissible_rows": 0,
              "ambiguous_admissible_cells_excluded": 0,
              "excluded_e001_admissible_hour_audit_overlap": 0,
              "not_in_e001_admissible_hour_audit": 0}
    for index, path in enumerate(files, start=1):
        g = pd.read_parquet(path, columns=[
            "candidate_id", "mmsi", "hour_utc", "lat", "lon", "admissible_centre",
            "source_gap_start_utc", "source_gap_end_utc",
        ])
        totals["gfw_exact_rows"] += len(g)
        g = g.loc[g.admissible_centre.eq(True)].drop(columns="admissible_centre").copy()
        duplicate = g.duplicated(["candidate_id", "hour_utc"], keep=False)
        totals["ambiguous_admissible_cells_excluded"] += int(duplicate.sum())
        g = g.loc[~duplicate].copy()
        totals["gfw_admissible_rows"] += len(g)
        if g.empty:
            continue
        g["candidate_id"] = g.candidate_id.astype(str)
        ordinal = ordinal_for(path)
        used_parts = []
        for prefix in ("", "gp2_"):
            audit = ANCHOR / "hour_audit" / f"{prefix}block_{ordinal:04d}.parquet"
            if audit.is_file():
                a = pd.read_parquet(audit, columns=[
                    "candidate_id", "hour_utc", "admissible_unique_anchor",
                ])
                used_parts.append(a.loc[a.admissible_unique_anchor.eq(True), ["candidate_id", "hour_utc"]])
        if used_parts:
            used = pd.concat(used_parts, ignore_index=True)
            used["candidate_id"] = used.candidate_id.astype(str)
            if used.duplicated(["candidate_id", "hour_utc"]).any():
                raise ValueError(f"E001 final anchor key repeated in block {ordinal}")
            g = g.merge(used.assign(audit_overlap=True), on=["candidate_id", "hour_utc"],
                        how="left", validate="one_to_one")
            used_count = int(g.audit_overlap.eq(True).sum())
            g = g.loc[g.audit_overlap.isna()].drop(columns="audit_overlap")
        else:
            used_count = 0
        totals["excluded_e001_admissible_hour_audit_overlap"] += used_count
        totals["not_in_e001_admissible_hour_audit"] += len(g)
        if len(g):
            g["source_gap_hours"] = (pd.to_datetime(g.source_gap_end_utc, utc=True)
                                     - pd.to_datetime(g.source_gap_start_utc, utc=True)).dt.total_seconds() / 3600
            parts.append(g.drop(columns=["source_gap_start_utc", "source_gap_end_utc"]))
        if index % 200 == 0:
            print(f"source blocks {index}/{len(files)}; held-out={totals['not_in_e001_admissible_hour_audit']:,}", flush=True)
    h = pd.concat(parts, ignore_index=True)
    if h.duplicated(["candidate_id", "hour_utc"]).any():
        raise ValueError("held-out GFW candidate-hour repeated across blocks")
    u = pd.read_parquet(PANEL, columns=["candidate_id", "mmsi", "source_shard", "shiptype_group"])
    u["candidate_id"] = u.candidate_id.astype(str)
    h["mmsi"] = h.mmsi.astype(str)
    h = h.merge(u, on="candidate_id", how="left", validate="many_to_one", suffixes=("_gfw", "_e001"))
    if (h.loc[h.source_shard.notna(), "mmsi_gfw"] != h.loc[h.source_shard.notna(), "mmsi_e001"].astype(str)).any():
        raise ValueError("cross-project candidate ID matched a different MMSI")
    totals["not_in_e001_voyage_panel"] = int(h.source_shard.isna().sum())
    h = h.loc[h.source_shard.notna()].drop(columns=["mmsi_gfw", "mmsi_e001"])
    totals["heldout_in_e001_panel"] = len(h)
    totals["heldout_ships"] = int(u.loc[u.candidate_id.isin(set(h.candidate_id)), "mmsi"].nunique())
    totals["heldout_voyages"] = int(h.candidate_id.nunique())
    target_dir = OUT / "heldout_by_shard"
    target_dir.mkdir(exist_ok=True)
    for shard, block in h.groupby("source_shard", sort=True):
        block.to_parquet(target_dir / f"{shard}.parquet", index=False)
    totals["e001_shards_with_heldout"] = len(list(target_dir.glob("shard_*.parquet")))
    (OUT / "source_counts.json").write_text(json.dumps(totals, indent=2) + "\n")
    return totals


def compare_shard(path: Path) -> dict:
    shard = path.stem
    h = pd.read_parquet(path)
    h = h.reset_index(drop=True)
    h["row_id"] = np.arange(len(h), dtype=np.int64)
    ids = set(h.candidate_id)
    t = pd.read_parquet(TRACKS / shard / "ten_minute_complete_candidate.parquet", columns=[
        "candidate_id", "utc", "lat", "lon", "track_evidence",
    ])
    t = t.loc[t.candidate_id.isin(ids)].copy()
    t["hour_utc"] = pd.to_datetime(t.utc, utc=True).dt.floor("h")
    joined = h[["row_id", "candidate_id", "hour_utc", "lat", "lon"]].merge(
        t, on=["candidate_id", "hour_utc"], how="left", validate="one_to_many",
        suffixes=("_gfw", "_track"),
    )
    valid = joined.lat_track.notna() & joined.lon_track.notna()
    a = np.radians(joined.loc[valid, "lat_gfw"].to_numpy(dtype=float))
    b = np.radians(joined.loc[valid, "lat_track"].to_numpy(dtype=float))
    dlat = b - a
    dlon = np.radians(joined.loc[valid, "lon_track"].to_numpy(dtype=float)
                     - joined.loc[valid, "lon_gfw"].to_numpy(dtype=float))
    x = np.sin(dlat / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dlon / 2) ** 2
    joined.loc[valid, "centre_distance_km"] = 2 * 6371.0088 * np.arctan2(np.sqrt(x), np.sqrt(np.maximum(0, 1 - x)))
    labels = ("observed_ais", "ais_interpolated_le4h", "gfw_anchor",
              "gfw_supported_interpolation", "unresolved")
    for label in labels:
        joined[f"n_{label}"] = joined.track_evidence.eq(label).astype("int16")
    agg = joined.groupby("row_id", sort=False).agg(
        ten_minute_rows=("utc", "count"), positioned_rows=("centre_distance_km", "count"),
        min_centre_distance_km=("centre_distance_km", "min"),
        **{f"n_{label}": (f"n_{label}", "sum") for label in labels},
    ).reset_index()
    result = h[["row_id", "candidate_id", "hour_utc", "shiptype_group", "source_gap_hours"]].merge(
        agg, on="row_id", how="left", validate="one_to_one")
    if result.ten_minute_rows.isna().any():
        raise ValueError(f"{shard}: held-out hour has no aggregation")
    result["comparison_class"] = np.select([
        result.n_gfw_anchor.gt(0), result.n_observed_ais.gt(0),
        result.n_ais_interpolated_le4h.gt(0),
        result.n_gfw_supported_interpolation.gt(0) & result.n_unresolved.eq(0),
        result.n_gfw_supported_interpolation.gt(0),
    ], ["gfw_anchor_overlap", "observed_ais_overlap", "short_ais_interpolation",
        "gfw_supported_interpolation", "gfw_interpolation_partial"], default="unresolved")
    result.drop(columns="row_id").to_parquet(OUT / "compared_by_shard" / f"{shard}.parquet", index=False)
    return {"shard": shard, "heldout_hours": len(result),
            "with_position": int(result.positioned_rows.gt(0).sum()),
            "comparison_class": result.comparison_class.value_counts().to_dict()}


def summarize(source: dict, shard_reports: list[dict], elapsed: float) -> dict:
    files = sorted((OUT / "compared_by_shard").glob("shard_*.parquet"))
    result = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if len(result) != source["heldout_in_e001_panel"]:
        raise ValueError("held-out hours do not close across E001 shards")
    summary = {"status": "formal_e002_diagnostic_pending_review", "source_counts": source,
               "compared_hours": len(result), "comparison_class": result.comparison_class.value_counts().to_dict(),
               "elapsed_seconds": elapsed,
               "distance_definition": "minimum great-circle distance from GFW 0.01-degree hourly cell centre to positioned 10-minute product points in the same UTC hour; not exact position error",
               "sources": {
                   "climate_gfw_summary_sha256": sha256(CLIMATE / "summary.json"),
                   "e001_copied_gfw_summary_sha256": sha256(COPY / "summary.json"),
                   "climate_derived_source_manifest_sha256": sha256(OUT / "climate_derived_source_manifest.json"),
                   "e001_anchor_union_manifest_sha256": sha256(ANCHOR / "rematch_manifest.json"),
                   "e001_voyage_panel_sha256": sha256(PANEL),
               }, "distance_by_class": {}}
    for label, block in result.groupby("comparison_class", sort=True):
        values = block.min_centre_distance_km.dropna().to_numpy(dtype=float)
        summary["distance_by_class"][label] = {
            "candidate_hours": len(block), "hours_with_position": len(values),
            "complete_six_grid_rows": int(block.ten_minute_rows.eq(6).sum()),
            "p50_km": float(np.quantile(values, .5)) if len(values) else None,
            "p75_km": float(np.quantile(values, .75)) if len(values) else None,
            "p90_km": float(np.quantile(values, .9)) if len(values) else None,
            "p95_km": float(np.quantile(values, .95)) if len(values) else None,
            "p99_km": float(np.quantile(values, .99)) if len(values) else None,
            "within_5km": int((values <= 5).sum()), "within_10km": int((values <= 10).sum()),
        }
    primary = result.loc[result.comparison_class.eq("gfw_supported_interpolation")]
    summary["primary_by_gap_band"] = {}
    bands = pd.cut(primary.source_gap_hours, [0, 6, 12, 24, np.inf],
                   labels=["gt4_to6h", "gt6_to12h", "gt12_to24h", "gt24h"])
    for band, block in primary.groupby(bands, observed=True):
        values = block.min_centre_distance_km.dropna().to_numpy(dtype=float)
        summary["primary_by_gap_band"][str(band)] = {
            "candidate_hours": len(block),
            "p50_km": float(np.quantile(values, .5)) if len(values) else None,
            "p90_km": float(np.quantile(values, .9)) if len(values) else None,
            "p95_km": float(np.quantile(values, .95)) if len(values) else None,
        }
    summary["primary_by_month"] = {}
    for month, block in primary.groupby(primary.hour_utc.dt.month):
        values = block.min_centre_distance_km.dropna().to_numpy(dtype=float)
        summary["primary_by_month"][str(month)] = {
            "candidate_hours": len(block),
            "p50_km": float(np.quantile(values, .5)) if len(values) else None,
            "p90_km": float(np.quantile(values, .9)) if len(values) else None,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT = args.out.resolve()
    if ".runs" not in OUT.parts:
        parser.error("per-hour GFW results must stay in .runs")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = build_holdout()
    print("source counts", source, flush=True)
    out = OUT / "compared_by_shard"
    out.mkdir(exist_ok=True)
    tasks = sorted((OUT / "heldout_by_shard").glob("shard_*.parquet"))
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compare_shard, path): path for path in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            reports.append(future.result())
            if index % 25 == 0 or index == len(tasks):
                print(f"track shards {index}/{len(tasks)}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    (OUT / "shard_counts.json").write_text(json.dumps(reports, indent=2) + "\n")
    result = summarize(source, reports, time.perf_counter() - started)
    print(json.dumps({key: result[key] for key in ("status", "compared_hours", "comparison_class", "elapsed_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
