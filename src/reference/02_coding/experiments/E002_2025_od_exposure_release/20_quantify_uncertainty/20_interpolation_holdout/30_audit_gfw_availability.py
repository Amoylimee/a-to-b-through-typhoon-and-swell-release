#!/usr/bin/env python3
"""Audit local GFW anchor availability inside the frozen E002 holdout windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


CODING_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = CODING_ROOT / "data/processed/products/companion_tracks/2025"
GFW_EVIDENCE = ["gfw_anchor", "gfw_supported_interpolation"]


def summarize_window_gfw_records(
    sample: pd.Series,
    gfw_points: pd.DataFrame,
) -> dict[str, object]:
    """Count cached GFW anchor/support rows strictly inside one held-out window.

    Observed AIS rows are deliberately excluded by the caller's Parquet filter;
    only persisted GFW-labelled evidence can make a window GFW-available here.
    """
    start = pd.to_datetime(sample.gap_start_utc, utc=True)
    end = pd.to_datetime(sample.gap_end_utc, utc=True)
    in_window = gfw_points.loc[gfw_points.utc.gt(start) & gfw_points.utc.lt(end)]
    anchors = in_window.loc[in_window.track_evidence.eq("gfw_anchor")]
    supported = in_window.loc[
        in_window.track_evidence.eq("gfw_supported_interpolation")
    ]
    return {
        "sample_id": str(sample.sample_id),
        "candidate_id": str(sample.candidate_id),
        "sample_set": str(sample.sample_set),
        "shiptype_group": str(sample.shiptype_group),
        "nominal_gap_hours": int(sample.nominal_gap_hours),
        "gfw_anchor_rows_in_window": int(len(anchors)),
        "gfw_anchor_hours_in_window": int(
            pd.to_datetime(anchors.utc, utc=True).dt.floor("h").nunique()
        ),
        "gfw_supported_rows_in_window": int(len(supported)),
        "gfw_anchor_available": bool(not anchors.empty),
    }


def count_gp2_targets_in_window(
    sample: pd.Series,
    target_hours: pd.DataFrame,
) -> dict[str, int]:
    """Count persisted GP2-rematch hour windows overlapping this holdout."""
    start = pd.to_datetime(sample.gap_start_utc, utc=True)
    end = pd.to_datetime(sample.gap_end_utc, utc=True)
    overlaps = target_hours.loc[
        target_hours.hour_utc.lt(end)
        & (target_hours.hour_utc + pd.Timedelta(hours=1)).gt(start)
    ]
    admissible = overlaps.admissible_unique_anchor.fillna(False).astype(bool)
    return {
        "gp2_target_hour_windows": int(len(overlaps)),
        "gp2_admissible_anchor_hours": int(admissible.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gp2-hour-audit", type=Path, required=True)
    parser.add_argument("--gp2-rematch-manifest", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(args.sample_list)
    required = {
        "sample_id", "candidate_id", "source_shard", "sample_set",
        "shiptype_group", "nominal_gap_hours", "gap_start_utc", "gap_end_utc",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"sample list missing columns: {missing}")
    if samples.sample_id.astype(str).duplicated().any():
        raise ValueError("sample list contains duplicate sample IDs")

    results: list[dict[str, object]] = []
    shard_count = 0
    for shard, block in samples.groupby("source_shard", sort=True, observed=True):
        shard_count += 1
        candidate_ids = sorted(block.candidate_id.astype(str).unique())
        path = TRACK_ROOT / "run/shards" / str(shard) / "ten_minute_complete_candidate.parquet"
        points = pq.read_table(
            path,
            columns=["candidate_id", "utc", "track_evidence"],
            filters=[
                ("candidate_id", "in", candidate_ids),
                ("track_evidence", "in", GFW_EVIDENCE),
            ],
        ).to_pandas()
        if points.empty:
            by_candidate: dict[str, pd.DataFrame] = {}
        else:
            points["candidate_id"] = points.candidate_id.astype(str)
            points["utc"] = pd.to_datetime(points.utc, utc=True)
            if not points.track_evidence.isin(GFW_EVIDENCE).all():
                raise AssertionError("non-GFW evidence passed the selected Parquet filter")
            by_candidate = {
                str(candidate_id): group
                for candidate_id, group in points.groupby(
                    "candidate_id", sort=False, observed=True
                )
            }
        for row in block.itertuples(index=False):
            candidate_points = by_candidate.get(str(row.candidate_id))
            if candidate_points is None:
                candidate_points = pd.DataFrame(
                    columns=["candidate_id", "utc", "track_evidence"]
                )
            results.append(
                summarize_window_gfw_records(pd.Series(row._asdict()), candidate_points)
            )

    per_window = pd.DataFrame(results)
    if len(per_window) != len(samples):
        raise AssertionError("GFW audit did not return exactly one row per holdout window")
    if per_window.sample_id.astype(str).duplicated().any():
        raise AssertionError("GFW audit returned duplicate sample IDs")

    rematch_manifest = json.loads(args.gp2_rematch_manifest.read_text(encoding="utf-8"))
    if (
        rematch_manifest.get("status") != "pass"
        or rematch_manifest.get("gfw_requests") != 0
        or rematch_manifest.get("products_modified") != 0
    ):
        raise ValueError("GP2 rematch manifest is not a passed offline, no-product-write audit")
    gp2_dataset = ds.dataset(str(args.gp2_hour_audit), format="parquet")
    gp2_table = gp2_dataset.to_table(
        columns=[
            "candidate_id", "hour_utc", "anchor_status",
            "admissible_unique_anchor", "target_hour_id",
        ]
    )
    gp2_hours = gp2_table.to_pandas()
    gp2_hours["candidate_id"] = gp2_hours.candidate_id.astype(str)
    gp2_hours["hour_utc"] = pd.to_datetime(gp2_hours.hour_utc, utc=True)
    sample_ids = set(samples.candidate_id.astype(str))
    gp2_hours = gp2_hours.loc[gp2_hours.candidate_id.isin(sample_ids)].copy()
    gp2_by_candidate = {
        str(candidate_id): group
        for candidate_id, group in gp2_hours.groupby(
            "candidate_id", sort=False, observed=True
        )
    }
    gp2_counts = []
    for row in samples.itertuples(index=False):
        gp2_rows = gp2_by_candidate.get(str(row.candidate_id))
        if gp2_rows is None:
            gp2_rows = gp2_hours.iloc[0:0]
        gp2_counts.append(
            count_gp2_targets_in_window(pd.Series(row._asdict()), gp2_rows)
        )
    gp2_frame = pd.DataFrame(gp2_counts)
    gp2_frame.insert(0, "sample_id", samples.sample_id.astype(str).to_numpy())
    per_window = per_window.merge(
        gp2_frame, on="sample_id", how="left", validate="one_to_one"
    )
    per_window["gfw_anchor_available"] = (
        per_window.gfw_anchor_rows_in_window.gt(0)
        | per_window.gp2_admissible_anchor_hours.gt(0)
    )
    per_window.to_csv(args.out / "gfw_availability_by_window_internal.csv", index=False)
    by_stratum = per_window.groupby(
        ["sample_set", "nominal_gap_hours", "shiptype_group"],
        dropna=False,
        sort=True,
        observed=True,
    ).agg(
        selected_windows=("sample_id", "size"),
        windows_with_gfw_anchor=("gfw_anchor_available", "sum"),
        gfw_anchor_rows=("gfw_anchor_rows_in_window", "sum"),
        gfw_anchor_hours=("gfw_anchor_hours_in_window", "sum"),
        gfw_supported_rows=("gfw_supported_rows_in_window", "sum"),
        gp2_target_hour_windows=("gp2_target_hour_windows", "sum"),
        gp2_admissible_anchor_hours=("gp2_admissible_anchor_hours", "sum"),
    ).reset_index()
    by_stratum.to_csv(args.out / "gfw_availability_by_stratum.csv", index=False)
    report = {
        "ok": bool(len(per_window) == len(samples)),
        "sample_window_count": int(len(samples)),
        "sample_shard_count": int(shard_count),
        "window_count_with_gfw_anchor": int(per_window.gfw_anchor_available.sum()),
        "window_count_without_gfw_anchor": int((~per_window.gfw_anchor_available).sum()),
        "gfw_anchor_rows_in_windows": int(per_window.gfw_anchor_rows_in_window.sum()),
        "gfw_anchor_hours_in_windows": int(per_window.gfw_anchor_hours_in_window.sum()),
        "gfw_supported_interpolation_rows_in_windows": int(
            per_window.gfw_supported_rows_in_window.sum()
        ),
        "gp2_rematch_manifest_sha256": hashlib.sha256(
            args.gp2_rematch_manifest.read_bytes()
        ).hexdigest(),
        "gp2_hour_audit_root": str(args.gp2_hour_audit),
        "gp2_rematch_manifest_path": str(args.gp2_rematch_manifest),
        "gp2_rematch_status": rematch_manifest["status"],
        "gp2_gfw_requests": int(rematch_manifest["gfw_requests"]),
        "gp2_products_modified": int(rematch_manifest["products_modified"]),
        "gp2_hour_audit_rows_for_sample_candidates": int(len(gp2_hours)),
        "gp2_target_hour_windows_overlapping_holdouts": int(
            per_window.gp2_target_hour_windows.sum()
        ),
        "gp2_admissible_anchor_hours_overlapping_holdouts": int(
            per_window.gp2_admissible_anchor_hours.sum()
        ),
        "only_persisted_gfw_labelled_rows_counted": True,
        "source": "adopted 2025 companion_tracks ten_minute_complete_candidate.parquet",
        "per_window_internal_csv": str(args.out / "gfw_availability_by_window_internal.csv"),
        "per_stratum_csv": str(args.out / "gfw_availability_by_stratum.csv"),
    }
    (args.out / "gfw_availability_closure.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
