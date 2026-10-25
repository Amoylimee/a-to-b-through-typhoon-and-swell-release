#!/usr/bin/env python3
"""E006 GFW version, step 4 check: independent validation of the ten-minute tracks, and the
internal comparison with the accepted reference AIS companion-track product.

Validation re-reads every shard output of `160_build_gfw_ten_minute_tracks.py`:
- every voyage of the step-3 candidate root appears exactly once, and each voyage's grid rows equal
  the strict ten-minute grid between its departure and arrival;
- output files match the hashes the run recorded;
- labels are the allowed ones, resolved rows have coordinates and unresolved rows none, observed
  rows sit at hh:30 with their observation time, and `(candidate_id, utc)` is unique;
- shard totals add up to the run totals.

Comparison (internal only; the reference AIS product is never an input to the GFW version):
- overall shares of observed, interpolated and unresolved time in both versions;
- for voyages matched within 6 h by `140_compare_gfw_voyage_base.py`, the per-voyage resolved share
  in both versions, so coverage is compared on the same journeys.

    python 170_validate_gfw_ten_minute_tracks.py --run <step-4 root> --match <140 output root> --out <new .runs dir>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
REFERENCE_AIS_TRACKS = CODING / "data/processed/products/companion_tracks/2025/run/shards"
ALLOWED = {"observed_ais", "gfw_supported_interpolation", "unresolved"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_shard(result: dict) -> dict:
    for item in result["outputs"]:
        if sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"output changed: {item['path']}")
    out = Path(result["outputs"][0]["path"]).parent
    columns = ["candidate_id", "utc", "lat", "lon", "track_evidence", "position_source", "gap_class", "observation_time_utc"]
    counts: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    problems: Counter[str] = Counter()
    file = pq.ParquetFile(out / "ten_minute_complete_candidate.parquet")
    for batch in file.iter_batches(batch_size=2_000_000, columns=columns):
        frame = batch.to_pandas()
        labels = frame.track_evidence.astype(str)
        resolved = labels.ne("unresolved")
        coords = frame[["lat", "lon"]].notna().all(axis=1)
        observed = labels.eq("observed_ais")
        utc = pd.to_datetime(frame.utc, utc=True)
        problems["unknown_label"] += int((~labels.isin(ALLOWED)).sum())
        problems["resolved_without_position"] += int((resolved & ~coords).sum())
        problems["unresolved_with_position"] += int((~resolved & frame[["lat", "lon"]].notna().any(axis=1)).sum())
        problems["off_grid_time"] += int((utc != utc.dt.floor("10min")).sum())
        problems["observed_not_hh30"] += int((observed & utc.dt.minute.ne(30)).sum())
        problems["observed_time_mismatch"] += int((observed & pd.to_datetime(frame.observation_time_utc, utc=True).ne(utc)).sum())
        counts.update(labels.value_counts().to_dict())
        counts.update({f"source_{k}": v for k, v in frame.position_source.dropna().astype(str).value_counts().items()})
        counts.update({f"class_{k}": v for k, v in frame.gap_class.dropna().astype(str).value_counts().items()})
        seen.update(frame.candidate_id.astype(str).value_counts().to_dict())
    summary = pd.read_parquet(out / "candidate_summary.parquet", columns=["candidate_id", "grid_rows", "expected_grid_rows"])
    by_id = summary.set_index("candidate_id").grid_rows
    problems["summary_duplicate_candidates"] = int(summary.candidate_id.duplicated().sum())
    problems["grid_rows_not_expected"] = int(summary.grid_rows.ne(summary.expected_grid_rows).sum())
    observed_ids = pd.Series(seen)
    problems["rows_per_candidate_mismatch"] = int(observed_ids.reindex(by_id.index).fillna(0).ne(by_id).sum())
    duplicates = 0
    for batch in file.iter_batches(batch_size=4_000_000, columns=["candidate_id", "utc"]):
        duplicates += int(batch.to_pandas().duplicated().sum())  # batches hold whole voyages in write order
    problems["duplicate_candidate_time"] = duplicates
    return {"shard": result["shard"], "counts": dict(counts), "problems": dict(problems),
            "candidates": int(len(summary)), "rows": int(sum(seen.values()))}


def reference_ais_summaries() -> pd.DataFrame:
    columns = ["candidate_id", "grid_rows", "observed_ais_rows", "ais_interpolated_le4h_rows",
               "gfw_supported_interpolation_rows", "unresolved_rows"]
    return pd.concat([pd.read_parquet(p, columns=columns) for p in sorted(glob.glob(str(REFERENCE_AIS_TRACKS / "*/candidate_summary.parquet")))],
                     ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--match", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    state = json.loads((run / "execution.json").read_text())
    if state["status"] != "pass" or not state["complete_year"] or state["failures"] or len(state["results"]) != 64:
        raise ValueError("step 4 is not the complete passed 64-shard run")
    out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(min(args.workers, 64)) as pool:
        shards = list(pool.map(check_shard, state["results"]))
    counts: Counter[str] = Counter()
    problems: Counter[str] = Counter()
    for item in shards:
        counts.update(item["counts"])
        problems.update(item["problems"])
    prepared = json.loads((run / "prepared/manifest.json").read_text())
    rows = sum(item["rows"] for item in shards)
    checks = {
        "all_candidates_once": sum(item["candidates"] for item in shards) == prepared["candidate_rows_total"],
        "rows_equal_expected": rows == sum(t["expected_grid_rows"] for t in prepared["tasks"]) == state["totals"]["grid_rows"],
        "labels_close": counts["observed_ais"] + counts["gfw_supported_interpolation"] + counts["unresolved"] == rows,
        "run_totals_match": (counts["observed_ais"] == state["totals"]["observed_ais_rows"]
                             and counts["gfw_supported_interpolation"] == state["totals"]["gfw_supported_interpolation_rows"]),
        **{f"no_{key}": value == 0 for key, value in problems.items()},
    }
    share = {k: counts[k] / rows for k in ("observed_ais", "gfw_supported_interpolation", "unresolved")}

    # Internal comparison with the reference AIS product.
    reference_ais = reference_ais_summaries()
    k_rows = int(reference_ais.grid_rows.sum())
    k_share = {"observed_ais": reference_ais.observed_ais_rows.sum() / k_rows,
               "interpolated_or_reconstructed": (reference_ais.ais_interpolated_le4h_rows.sum() + reference_ais.gfw_supported_interpolation_rows.sum()) / k_rows,
               "unresolved": reference_ais.unresolved_rows.sum() / k_rows}
    gfw = pd.concat([pd.read_parquet(Path(r["outputs"][1]["path"]), columns=["candidate_id", "grid_rows", "unresolved_rows", "window_hours"])
                     for r in state["results"]], ignore_index=True)
    match = pd.read_parquet(args.match / "reference_ais_to_gfw_time_match.parquet", columns=["candidate_id", "gfw_pair_id", "time_score_h"])
    match = match.loc[match.time_score_h.le(6)]
    joined = (match.merge(reference_ais.rename(columns={"grid_rows": "k_rows", "unresolved_rows": "k_unresolved"}), on="candidate_id")
                   .merge(gfw.rename(columns={"candidate_id": "gfw_pair_id", "grid_rows": "g_rows", "unresolved_rows": "g_unresolved"}),
                          on="gfw_pair_id"))
    joined = joined.loc[joined.k_rows.gt(0) & joined.g_rows.gt(0)]
    k_resolved = 1 - joined.k_unresolved / joined.k_rows
    g_resolved = 1 - joined.g_unresolved / joined.g_rows
    summary = {
        "status": "passed" if all(checks.values()) else "failed", "checks": checks, "problems": dict(problems),
        "rows": rows, "candidates": prepared["candidate_rows_total"],
        "gfw_shares": share,
        "gfw_position_sources": {k.removeprefix("source_"): v for k, v in counts.items() if k.startswith("source_")},
        "gfw_gap_classes": {k.removeprefix("class_"): v for k, v in counts.items() if k.startswith("class_")},
        "reference_ais_product": {"candidates": int(len(reference_ais)), "rows": k_rows, "shares": k_share},
        "matched_voyages_within_6h": {
            "pairs": int(len(joined)),
            "reference_ais_resolved_share_rowweighted": float(1 - joined.k_unresolved.sum() / joined.k_rows.sum()),
            "gfw_resolved_share_rowweighted": float(1 - joined.g_unresolved.sum() / joined.g_rows.sum()),
            "per_voyage_difference_gfw_minus_reference_ais_quantiles": (g_resolved - k_resolved).quantile([.1, .25, .5, .75, .9]).round(4).to_dict(),
            "window_hours_ratio_median": float((joined.g_rows / joined.k_rows).median()),
        },
        "note": "GFW points are stamped hh:30 and voyages start and end on them, which raises the positioned share slightly (T030 a)",
        "network_requests": 0,
    }
    (out / "validation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n")
    pd.DataFrame(shards).to_json(out / "shard_checks.json", orient="records", indent=1)
    print(json.dumps({k: summary[k] for k in ("status", "rows", "gfw_shares", "reference_ais_product", "matched_voyages_within_6h")},
                     indent=2, default=float))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
