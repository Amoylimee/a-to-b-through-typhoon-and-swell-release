#!/usr/bin/env python3
"""Package the matched GFW anchors for the user's CP2 review, then stop.

This stage reports. It never requests, never interpolates, and never writes a
product. Its job is to let the user see where the anchors came from, how many
survived each identity/time/motion rule, which gaps they can and cannot
support, and what a few real cases look like -- including the failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq

GAP_BANDS = [(4, 6), (6, 8), (8, 12), (12, 18), (18, 24)]
STATUS_ORDER = [
    "admissible_unique_anchor",
    "no_exact_identity_time_presence",
    "exact_identity_time_motion_conflict",
    "identity_multiplicity",
    "multiple_feasible_grid_centres",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_table(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {"path": str(path), "rows": int(len(frame)), "sha256": sha256(path)}


def gap_band(hours: float) -> str:
    for low, high in GAP_BANDS:
        if low < hours <= high:
            return f"{low}-{high}h"
    return ">24h" if hours > 24 else "<=4h"


def load_hour_audit(rematch_root: Path) -> pd.DataFrame:
    paths = sorted((rematch_root / "hour_audit").glob("block_*.parquet"))
    if not paths:
        raise SystemExit("no hour audit blocks to package")
    columns = [
        "target_hour_id", "gap_id", "candidate_id", "mmsi", "vessel_uid_clean", "layer",
        "source_shard", "hour_utc", "source_gap_start_utc", "source_gap_end_utc",
        "lat_a", "lon_a", "lat_b", "lon_b", "effective_speed_kn",
        "cache_coverage", "exact_grid_centres", "feasible_grid_centres",
        "multiplicity_grid_centres", "anchor_status", "admissible_unique_anchor",
        "anchor_lat", "anchor_lon",
    ]
    frames = [pq.read_table(path, columns=columns).to_pandas() for path in paths]
    hours = pd.concat(frames, ignore_index=True)
    if hours["target_hour_id"].duplicated().any():
        raise SystemExit("packaged target hours are not unique")
    return hours


def status_counts(frame: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    keys = (by or []) + ["anchor_status"]
    counts = frame.groupby(keys, observed=True).size().rename("target_hours").reset_index()
    wide = counts.pivot_table(
        index=by or None, columns="anchor_status", values="target_hours",
        aggfunc="sum", fill_value=0,
    )
    for status in STATUS_ORDER:
        if status not in wide.columns:
            wide[status] = 0
    wide = wide[STATUS_ORDER]
    wide["target_hours"] = wide.sum(axis=1)
    wide["anchor_share"] = (
        wide["admissible_unique_anchor"] / wide["target_hours"].where(wide["target_hours"] > 0)
    ).round(6)
    return wide.reset_index()


def example_rows(hours: pd.DataFrame, gaps: pd.DataFrame, per_class: int) -> pd.DataFrame:
    per_gap = hours.groupby("gap_id", observed=True).agg(
        target_hours=("target_hour_id", "size"),
        anchor_hours=("admissible_unique_anchor", "sum"),
        motion_conflict_hours=("anchor_status", lambda s: int((s == "exact_identity_time_motion_conflict").sum())),
        multi_centre_hours=("anchor_status", lambda s: int((s == "multiple_feasible_grid_centres").sum())),
        vessel_uid_clean=("vessel_uid_clean", "first"),
        mmsi=("mmsi", "first"),
        layer=("layer", "first"),
        source_shard=("source_shard", "first"),
        candidate_id=("candidate_id", "first"),
        gap_start=("source_gap_start_utc", "first"),
        gap_end=("source_gap_end_utc", "first"),
    ).reset_index()
    per_gap["anchor_share"] = per_gap["anchor_hours"] / per_gap["target_hours"]
    per_gap = per_gap.merge(
        gaps[["gap_id", "gap_hours", "region_10deg", "shiptype_group", "departure_month"]],
        on="gap_id", how="left",
    )
    def stratified(frame: pd.DataFrame, rank: str) -> pd.DataFrame:
        # Ranking by size alone returns only L3, which is exactly the layer whose anchor
        # rate is unremarkable. Spread the picks so the reviewer also sees L1 and L2.
        if frame.empty:
            return frame
        ordered = frame.sort_values(rank, ascending=False, kind="stable")
        spread = ordered.groupby("layer", observed=True).head(2)
        return spread.head(per_class) if len(spread) >= per_class else ordered.head(per_class)

    picks = []
    fully = per_gap.loc[per_gap["anchor_share"].eq(1.0) & per_gap["target_hours"].ge(6)]
    picks.append(("representative_fully_anchored", stratified(fully, "target_hours")))
    partial = per_gap.loc[per_gap["anchor_share"].between(0.2, 0.8, inclusive="both")]
    picks.append(("representative_partially_anchored", stratified(partial, "target_hours")))
    none = per_gap.loc[per_gap["anchor_hours"].eq(0)]
    picks.append(("anomalous_no_anchor", stratified(none, "target_hours")))
    conflict = per_gap.loc[per_gap["motion_conflict_hours"] > 0]
    picks.append(("anomalous_motion_conflict", stratified(conflict, "motion_conflict_hours")))
    multi = per_gap.loc[per_gap["multi_centre_hours"] > 0]
    picks.append(("anomalous_multiple_centres", stratified(multi, "multi_centre_hours")))
    out = []
    for label, frame in picks:
        if len(frame):
            out.append(frame.assign(example_class=label))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def observed_ais(ais_files: dict[str, str], shard: str, vessel_uid: str, start, end) -> pd.DataFrame:
    path = ais_files.get(str(shard))
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=["UTC", "lat", "lon"])
    table = feather.read_table(path, columns=["UTC", "vessel_uid", "lat", "lon"])
    frame = table.to_pandas()
    frame = frame.loc[frame["vessel_uid"].astype(str).eq(str(vessel_uid))]
    frame["UTC"] = pd.to_datetime(frame["UTC"], utc=True)
    window = frame.loc[frame["UTC"].between(start - pd.Timedelta("6h"), end + pd.Timedelta("6h"))]
    return window.sort_values("UTC")


def draw_examples(
    examples: pd.DataFrame,
    hours: pd.DataFrame,
    cells: pd.DataFrame,
    ais_files: dict[str, str],
    path: Path,
) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One panel per layer where the class has more than one, so the drawn cases do not
    # all come from the same layer.
    spread = examples.drop_duplicates(["example_class", "layer"]).groupby(
        "example_class", observed=True
    ).head(2)
    topped = examples.groupby("example_class", observed=True).head(2)
    chosen = pd.concat([spread, topped]).drop_duplicates("gap_id")
    chosen = chosen.groupby("example_class", observed=True).head(2).reset_index(drop=True)
    if chosen.empty:
        raise SystemExit("no example gaps to draw")
    columns = 2
    rows = int(np.ceil(len(chosen) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6.4 * columns, 4.6 * rows), squeeze=False)
    drawn = []
    for index, row in chosen.iterrows():
        axis = axes[index // columns][index % columns]
        gap_hours = hours.loc[hours["gap_id"].eq(row["gap_id"])].sort_values("hour_utc")
        gap_cells = cells.loc[cells["gap_id"].eq(row["gap_id"])]
        start = pd.Timestamp(row["gap_start"])
        end = pd.Timestamp(row["gap_end"])
        track = observed_ais(ais_files, row["source_shard"], row["vessel_uid_clean"], start, end)
        if len(track):
            axis.plot(track["lon"], track["lat"], "-", color="#4a4a4a", linewidth=1.0, zorder=1,
                      label=f"observed AIS ({len(track)} pts)")
            axis.plot(track["lon"], track["lat"], ".", color="#4a4a4a", markersize=2.5, zorder=2)
        axis.plot(
            [gap_hours["lon_a"].iloc[0], gap_hours["lon_b"].iloc[0]],
            [gap_hours["lat_a"].iloc[0], gap_hours["lat_b"].iloc[0]],
            "s", color="#1f77b4", markersize=8, zorder=5, label="gap boundary AIS fixes",
        )
        rejected = gap_cells.loc[~gap_cells["admissible_centre"].astype(bool)]
        if len(rejected):
            axis.plot(rejected["lon"], rejected["lat"], "x", color="#d62728", markersize=6,
                      zorder=4, label=f"rejected GFW candidates ({len(rejected)})")
        kept = gap_hours.loc[gap_hours["admissible_unique_anchor"].astype(bool)]
        if len(kept):
            axis.plot(kept["anchor_lon"], kept["anchor_lat"], "o", color="#2ca02c", markersize=5,
                      markerfacecolor="none", zorder=6, label=f"retained anchors ({len(kept)})")
        axis.set_title(
            f"{row['example_class']}  ({row['layer']}, {float(row['gap_hours']):.1f}h, "
            f"anchors {int(row['anchor_hours'])}/{int(row['target_hours'])})\n"
            f"gap {str(row['gap_id'])[:12]}",
            fontsize=9,
        )
        # The gap is what the reviewer is judging, so frame it. Observed AIS from the
        # surrounding +/-6h is still drawn and may run outside these limits.
        focus_lon = [gap_hours["lon_a"].iloc[0], gap_hours["lon_b"].iloc[0]]
        focus_lat = [gap_hours["lat_a"].iloc[0], gap_hours["lat_b"].iloc[0]]
        if len(gap_cells):
            focus_lon += list(gap_cells["lon"]); focus_lat += list(gap_cells["lat"])
        if len(kept):
            focus_lon += list(kept["anchor_lon"]); focus_lat += list(kept["anchor_lat"])
        span = max(max(focus_lon) - min(focus_lon), max(focus_lat) - min(focus_lat), 0.05)
        margin = 0.15 * span
        axis.set_xlim(min(focus_lon) - margin, max(focus_lon) + margin)
        axis.set_ylim(min(focus_lat) - margin, max(focus_lat) + margin)
        axis.set_xlabel("longitude"); axis.set_ylabel("latitude")
        axis.legend(fontsize=7, loc="best")
        axis.grid(alpha=0.25)
        drawn.append({
            "example_class": row["example_class"], "gap_id": row["gap_id"],
            "observed_ais_points": int(len(track)), "rejected_cells": int(len(rejected)),
            "retained_anchors": int(len(kept)),
        })
    for spare in range(len(chosen), rows * columns):
        axes[spare // columns][spare % columns].axis("off")
    figure.suptitle(
        "GFW anchor candidates and retained anchors -- discrete points, not a reconstructed track", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.2, w_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {"path": str(path), "sha256": sha256(path), "panels": drawn}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--rematch-root", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-run", type=Path, required=True)
    parser.add_argument("--cp1-package", type=Path, required=True)
    parser.add_argument("--cleaned-ais-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples-per-class", type=int, default=5)
    args = parser.parse_args()

    rematch_root = args.rematch_root.resolve()
    out = args.out.resolve()
    tables = out / "tables"

    rematch_manifest = json.loads((rematch_root / "rematch_manifest.json").read_text(encoding="utf-8"))
    if rematch_manifest["status"] != "pass" or rematch_manifest["failures"]:
        raise SystemExit("the rematch run is not a clean pass")
    adapter = json.loads(args.adapter_manifest.read_text(encoding="utf-8"))
    acquisition = json.loads((args.acquisition_run / "run_state.json").read_text(encoding="utf-8"))
    if acquisition.get("state") != "completed" or acquisition.get("failed_shards"):
        raise SystemExit("the acquisition run is not a clean completion")

    hours = load_hour_audit(rematch_root)
    gaps = pq.read_table(
        args.cp1_package / "gfw_candidate_gap_scope.parquet",
        columns=["gap_id", "gap_hours", "region_10deg", "shiptype_group", "departure_month", "layer"],
    ).to_pandas()
    scoped = hours.merge(gaps.drop(columns=["layer"]), on="gap_id", how="left", validate="many_to_one")
    if scoped["gap_hours"].isna().any():
        raise SystemExit("some target hours do not join the CP1 gap scope")
    scoped["gap_band"] = scoped["gap_hours"].map(gap_band)
    scoped["hour_month"] = pd.to_datetime(scoped["hour_utc"], utc=True).dt.month

    prior_dir = args.adapter_manifest.parent / "prior_cached_pass"
    prior = pd.concat(
        [pq.read_table(path).to_pandas() for path in sorted(prior_dir.glob("block_*.parquet"))],
        ignore_index=True,
    ).rename(columns={
        "cache_coverage": "prior_cache_coverage",
        "anchor_status": "prior_anchor_status",
        "admissible_unique_anchor": "prior_admissible_unique_anchor",
    })
    scoped = scoped.merge(prior, on="target_hour_id", how="left", validate="one_to_one")

    cells = pd.concat(
        [
            pq.read_table(path, columns=[
                "gap_id", "target_hour_id", "hour_utc", "mmsi", "lat", "lon",
                "hours", "hour_multiplicity", "motion_feasible", "admissible_centre",
            ]).to_pandas()
            for path in sorted((rematch_root / "exact_cells").glob("block_*.parquet"))
        ],
        ignore_index=True,
    )

    out.mkdir(parents=True, exist_ok=True)
    table_records: dict[str, dict] = {}

    headline = status_counts(scoped)
    headline.insert(0, "scope", "GP2 priority 4-24h target hours")
    table_records["01_headline_anchor_status"] = write_table(
        headline, tables / "01_headline_anchor_status.csv"
    )

    per_gap = scoped.groupby("gap_id", observed=True).agg(
        target_hours=("target_hour_id", "size"),
        anchor_hours=("admissible_unique_anchor", "sum"),
        gap_hours=("gap_hours", "first"),
        layer=("layer", "first"),
        region_10deg=("region_10deg", "first"),
        shiptype_group=("shiptype_group", "first"),
        departure_month=("departure_month", "first"),
        gap_band=("gap_band", "first"),
        candidate_id=("candidate_id", "first"),
    ).reset_index()
    per_gap["anchor_share"] = (per_gap["anchor_hours"] / per_gap["target_hours"]).round(6)
    per_gap["support_class"] = np.select(
        [per_gap["anchor_hours"].eq(0), per_gap["anchor_hours"].eq(per_gap["target_hours"])],
        ["no_usable_anchor", "every_hour_anchored"],
        default="partially_anchored",
    )
    per_gap.to_parquet(out / "anchor_support_by_gap.parquet", index=False, compression="zstd")

    gap_summary = per_gap.groupby("support_class", observed=True).agg(
        gaps=("gap_id", "size"), target_hours=("target_hours", "sum"),
        anchor_hours=("anchor_hours", "sum"), gap_hours=("gap_hours", "sum"),
    ).reset_index()
    gap_summary["gap_share"] = (gap_summary["gaps"] / len(per_gap)).round(6)
    table_records["02_gap_support_classes"] = write_table(
        gap_summary, tables / "02_gap_support_classes.csv"
    )

    for name, key in [
        ("03_by_layer", "layer"),
        ("04_by_gap_band", "gap_band"),
        ("05_by_departure_month", "departure_month"),
        ("06_by_region_10deg", "region_10deg"),
        ("07_by_shiptype_group", "shiptype_group"),
    ]:
        table_records[name] = write_table(status_counts(scoped, [key]), tables / f"{name}.csv")

    match_quality = pd.DataFrame([
        {"check": "target_hours", "value": int(len(scoped))},
        {"check": "query_geometry_full_coverage_hours", "value": int(scoped["cache_coverage"].eq("full").sum())},
        {"check": "query_geometry_partial_coverage_hours", "value": int(scoped["cache_coverage"].eq("partial").sum())},
        {"check": "query_geometry_no_coverage_hours", "value": int(scoped["cache_coverage"].eq("none").sum())},
        {"check": "exact_identity_time_cells", "value": int(len(cells))},
        {"check": "cells_motion_feasible", "value": int(cells["motion_feasible"].astype(bool).sum())},
        {"check": "cells_rejected_by_motion", "value": int((~cells["motion_feasible"].astype(bool)).sum())},
        {"check": "cells_with_hour_multiplicity", "value": int(cells["hour_multiplicity"].astype(bool).sum())},
        {"check": "hours_with_more_than_one_feasible_centre", "value": int((scoped["feasible_grid_centres"] > 1).sum())},
        {"check": "hours_with_zero_exact_cells", "value": int(scoped["exact_grid_centres"].fillna(0).eq(0).sum())},
        {"check": "admissible_unique_anchor_hours", "value": int(scoped["admissible_unique_anchor"].astype(bool).sum())},
        {"check": "distinct_gaps_in_scope", "value": int(per_gap["gap_id"].nunique())},
        {"check": "distinct_candidates_in_scope", "value": int(scoped["candidate_id"].nunique())},
        {"check": "distinct_mmsi_in_scope", "value": int(scoped["mmsi"].nunique())},
    ])
    table_records["08_match_and_filter_quality"] = write_table(
        match_quality, tables / "08_match_and_filter_quality.csv"
    )

    before_after = scoped.groupby(
        ["prior_anchor_status", "anchor_status"], observed=True, dropna=False
    ).size().rename("target_hours").reset_index()
    before_after["share_of_scope"] = (before_after["target_hours"] / len(scoped)).round(6)
    table_records["09_before_after_vs_cached_pass"] = write_table(
        before_after.sort_values("target_hours", ascending=False),
        tables / "09_before_after_vs_cached_pass.csv",
    )

    source = pd.DataFrame([
        {"item": "spatial_reports_completed", "value": acquisition["completed_shards"]},
        {"item": "spatial_reports_failed", "value": acquisition["failed_shards"]},
        {"item": "formal_report_requests_sent", "value": acquisition["formal_report_requests_sent"]},
        {"item": "quota_preflight_requests_sent", "value": acquisition["auxiliary_quota_preflight_requests_sent"]},
        {"item": "authorized_retries_sent", "value": acquisition.get("authorized_retries_sent", 0)},
        {"item": "automatic_retries_sent", "value": acquisition["automatic_retries_sent"]},
        {"item": "response_bytes", "value": acquisition["response_bytes"]},
        {"item": "canonical_identity_presence_rows", "value": adapter["presence_rows"]},
        {"item": "six_hour_blocks", "value": adapter["blocks"]},
        {"item": "target_hours_in_scope", "value": adapter["target_hours"]},
        {"item": "gfw_requests_in_this_stage", "value": 0},
        {"item": "products_modified", "value": 0},
    ])
    table_records["10_source_and_completeness"] = write_table(
        source, tables / "10_source_and_completeness.csv"
    )

    examples = example_rows(scoped, gaps, args.examples_per_class)
    table_records["11_review_examples"] = write_table(examples, tables / "11_review_examples.csv")

    ais_manifest = json.loads(args.cleaned_ais_manifest.read_text(encoding="utf-8"))
    ais_files = {}
    for record in ais_manifest["files"]:
        shard = Path(record["path"]).stem
        ais_files[shard] = record["path"]
        ais_files[shard.replace("shard_", "")] = record["path"]
        ais_files[str(int(shard.split("_")[-1]))] = record["path"]
    figure = draw_examples(examples, scoped, cells, ais_files, out / "figures" / "01_anchor_candidates_and_retained.png")

    scoped.to_parquet(out / "anchor_hour_verdicts.parquet", index=False, compression="zstd")
    admissible = scoped.loc[scoped["admissible_unique_anchor"].astype(bool)].copy()
    admissible.to_parquet(out / "admissible_anchors.parquet", index=False, compression="zstd")

    snapshot = hashlib.sha256()
    for key in sorted(table_records):
        snapshot.update(table_records[key]["sha256"].encode("ascii"))
    snapshot.update(sha256(out / "anchor_hour_verdicts.parquet").encode("ascii"))

    manifest = {
        "status": "awaiting_user_cp2_review",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "E001 2025 fleet expansion",
        "scope": "GFW anchors matched from the GP2 priority acquisition over CP1-approved internal 4-24h gaps",
        "read_only_inputs": True,
        "gfw_requests": 0,
        "products_modified": 0,
        "interpolation_rows_written": 0,
        "weather_rows_written": 0,
        "inputs": {
            "rematch_manifest": {
                "path": str(rematch_root / "rematch_manifest.json"),
                "sha256": sha256(rematch_root / "rematch_manifest.json"),
            },
            "adapter_manifest": {
                "path": str(args.adapter_manifest),
                "sha256": sha256(args.adapter_manifest),
            },
            "acquisition_run_state": {
                "path": str(args.acquisition_run / "run_state.json"),
                "sha256": sha256(args.acquisition_run / "run_state.json"),
            },
            "cp1_package_manifest": {
                "path": str(args.cp1_package / "cp1_review_manifest.json"),
                "sha256": sha256(args.cp1_package / "cp1_review_manifest.json"),
            },
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "parameters": rematch_manifest["parameters"],
        "totals": {
            "target_hours": int(len(scoped)),
            "admissible_unique_anchor_hours": int(scoped["admissible_unique_anchor"].astype(bool).sum()),
            "distinct_gaps": int(per_gap["gap_id"].nunique()),
            "gaps_with_any_anchor": int(per_gap["anchor_hours"].gt(0).sum()),
            "gaps_fully_anchored": int(per_gap["support_class"].eq("every_hour_anchored").sum()),
            "gaps_without_anchor": int(per_gap["support_class"].eq("no_usable_anchor").sum()),
            "distinct_candidates": int(scoped["candidate_id"].nunique()),
            "distinct_mmsi": int(scoped["mmsi"].nunique()),
        },
        "tables": table_records,
        "figures": [figure],
        "products": {
            "anchor_hour_verdicts": str(out / "anchor_hour_verdicts.parquet"),
            "admissible_anchors": str(out / "admissible_anchors.parquet"),
            "anchor_support_by_gap": str(out / "anchor_support_by_gap.parquet"),
        },
        "input_snapshot_sha256": snapshot.hexdigest(),
        "user_decision_required": (
            "target coverage is not reconstruction success; CP2 decides which anchors may enter "
            "sea-route constrained interpolation"
        ),
    }
    write_json_atomic(manifest, out / "cp2_review_manifest.json")
    print(json.dumps({"status": manifest["status"], **manifest["totals"],
                      "input_snapshot_sha256": manifest["input_snapshot_sha256"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
