#!/usr/bin/env python3
"""E007 V1 summary: position error in real gaps, reachable-area containment, and the share of
GFW-version voyage time that each gap length represents.

Reads the per-shard points from 20_compare_real_gaps.py and writes CSV tables to
<run>/summary/. Voyages whose GFW points disagree with reference AIS even next to a GFW point
(median error above 25 km within 10 minutes of it) are flagged as likely identity mismatches
(for example a shared MMSI); tables report results with and without them.

    python 25_summarize_real_gaps.py --run <20_ run> [--workers N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
GFW_TRACKS = CODING / ".runs/2026-10-13_2006_E007-one-cell-track-view"
HOLDOUT = CODING / "outputs/2026-10-11_1907_E007-one-cell-short-stop-repair/release/04_holdout_digest.csv"
# Default to the adopted population; an explicit environment selects another frozen run.
TRACK_SHARDS = Path(os.environ.get("E007_TRACK_SHARDS", GFW_TRACKS / "shards"))
HOLDOUT = Path(os.environ.get("E007_HOLDOUT_DIGEST", HOLDOUT))
BINS = [0, 1.01, 2.01, 4.01, 6.01, 12.01, 24.01, 48.01, np.inf]
LABELS = ["1 h", "2 h", "3-4 h", "5-6 h", "7-12 h", "13-24 h", "25-48 h", ">48 h"]
FLAG_KM, FLAG_OFFSET_H, FLAG_MIN_POINTS = 25.0, 1 / 6, 3
QUANTILES = (0.5, 0.75, 0.9, 0.95)


def load(run: Path, workers: int) -> pd.DataFrame:
    shards = sorted((run / "shards").iterdir())

    def one(shard: Path) -> pd.DataFrame:
        frame = pd.read_parquet(shard / "points.parquet", columns=["cand", "status", "gap_class", "gap_hours",
                                                                   "anchor_offset_h", "error_km", "contained", "pair_feasible"])
        ships = pd.read_parquet(shard / "candidates.parquet", columns=["cand", "shiptype_group"])
        frame["shiptype_group"] = pd.Categorical(ships.set_index("cand").shiptype_group.reindex(frame.cand).to_numpy())
        # Voyage key: shard number and candidate index, cheaper than strings over 5e8 rows.
        frame["voyage"] = np.int64(int(shard.name.split("_")[1])) * 10_000_000 + frame.cand.to_numpy(np.int64)
        return frame

    with ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(one, shards))
    frame = pd.concat(parts, ignore_index=True)
    for column in ("status", "gap_class", "shiptype_group"):
        frame[column] = frame[column].astype("category")
    frame["length"] = pd.cut(frame.gap_hours, BINS, labels=LABELS)
    return frame


def describe(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    def stats(block: pd.DataFrame) -> pd.Series:
        e = block.error_km.to_numpy(float)
        values = np.quantile(e, QUANTILES) if len(e) else [np.nan] * len(QUANTILES)
        out = {"points": len(e), "voyages": block.voyage.nunique()}
        out.update({f"error_km_p{int(q * 100)}": v for q, v in zip(QUANTILES, values)})
        out.update({"share_over_50km": float((e > 50).mean()) if len(e) else np.nan,
                    "share_over_100km": float((e > 100).mean()) if len(e) else np.nan})
        return pd.Series(out)

    return frame.groupby(keys, observed=True).apply(stats, include_groups=False).reset_index()


def _time_by_length(path: str) -> pd.DataFrame:
    # The cells of one gap are contiguous in the track, so gaps are runs of one dictionary code.
    table = pq.read_table(path, columns=["gap_id", "gap_class", "position_source"],
                          read_dictionary=["gap_id", "gap_class", "position_source"])
    gap = table.column("gap_id").combine_chunks()
    codes = pc.fill_null(gap.indices, -1).to_numpy()
    source = table.column("position_source").combine_chunks()
    observed = int(pc.sum(pc.equal(source.dictionary.take(pc.fill_null(source.indices, 0)), "observed_ais")
                          .filter(pc.is_valid(source.indices))).as_py() or 0)
    klass = table.column("gap_class").combine_chunks()
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    lengths = np.diff(np.r_[starts, len(codes)])
    keep = codes[starts] >= 0
    starts, lengths = starts[keep], lengths[keep]
    if len(starts) != len(np.unique(codes[codes >= 0])):
        raise ValueError(f"{path}: a gap's cells are not contiguous")
    names = klass.dictionary.to_pylist()
    gaps = pd.DataFrame({"gap_class": [names[i] for i in klass.indices.take(pa.array(starts)).to_numpy()],
                         "cells": lengths})
    gaps["length"] = pd.cut((gaps.cells + 1) / 6, BINS, labels=LABELS)
    out = gaps.groupby(["gap_class", "length"], observed=True).agg(gaps=("cells", "size"), cells=("cells", "sum")).reset_index()
    return pd.concat([out, pd.DataFrame([{"gap_class": "observed_gfw_point", "length": "", "gaps": 0, "cells": observed}])])


def gfw_time_by_length(workers: int) -> pd.DataFrame:
    paths = [str(p / "ten_minute_complete_candidate.parquet") for p in sorted(TRACK_SHARDS.iterdir())]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_time_by_length, paths))
    parts = [p.astype({"length": str}) for p in parts]
    table = pd.concat(parts).groupby(["gap_class", "length"], dropna=False)[["gaps", "cells"]].sum().reset_index()
    table["share_of_voyage_cells"] = table.cells / table.cells.sum()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    run = args.run.resolve()
    if json.loads((run / "execution.json").read_text()).get("state") != "pass":
        raise ValueError("real-gap comparison did not pass")
    out = run / "summary"
    out.mkdir(exist_ok=False)
    frame = load(run, args.workers)

    near = frame.loc[frame.status.ne("unresolved") & frame.anchor_offset_h.le(FLAG_OFFSET_H)]
    per_voyage = near.groupby("voyage", observed=True).error_km.agg(["median", "size"])
    flagged = per_voyage.loc[(per_voyage["size"] >= FLAG_MIN_POINTS) & (per_voyage["median"] > FLAG_KM)]
    flagged.rename(columns={"median": "near_point_median_error_km", "size": "near_points"}).to_csv(out / "flagged_voyages.csv")
    frame["flagged"] = frame.voyage.isin(flagged.index)

    resolved = frame.loc[frame.status.eq("resolved")]
    tables = []
    for subset, block in (("all", resolved), ("excluding_flagged", resolved.loc[~resolved.flagged])):
        tables.append(describe(block, ["length"]).assign(subset=subset))
    by_length = pd.concat(tables)
    by_length.to_csv(out / "positions_by_gap_length.csv", index=False)
    clean = resolved.loc[~resolved.flagged]
    describe(clean, ["length", "shiptype_group"]).to_csv(out / "positions_by_gap_length_shiptype.csv", index=False)
    describe(clean.assign(offset=pd.cut(clean.anchor_offset_h, [-0.01, 1 / 12, 1 / 6, 0.5, 1, 2, 3, 6, 12, 24])),
             ["length", "offset"]).to_csv(out / "positions_by_gap_length_and_offset.csv", index=False)
    describe(frame.loc[~frame.flagged & frame.status.eq("observed")].assign(length="exact GFW point"), ["length"]).to_csv(
        out / "positions_at_gfw_points.csv", index=False)

    unresolved = frame.loc[frame.status.eq("unresolved")]
    contain = unresolved.groupby(["flagged", "gap_class", "length"], observed=True).agg(
        points=("contained", "size"), tested=("contained", "count"), contained_share=("contained", "mean"),
        ends_feasible_share=("pair_feasible", "mean"), voyages=("voyage", "nunique")).reset_index()
    contain.to_csv(out / "reachable_area_containment.csv", index=False)

    holdout = pd.read_csv(HOLDOUT)
    holdout = holdout.loc[holdout.sample_set.eq("gap_length_x_shiptype"), ["nominal_gap_hours", "position_error_p90_km", "position_predicted_points"]]
    exact = clean.loc[clean.gap_hours.round(3).isin(holdout.nominal_gap_hours.astype(float))]
    real = describe(exact.assign(nominal_gap_hours=exact.gap_hours.round().astype(int)), ["nominal_gap_hours"])
    holdout.merge(real[["nominal_gap_hours", "points", "voyages", "error_km_p50", "error_km_p90", "error_km_p95"]]
                  .rename(columns=lambda c: c if c == "nominal_gap_hours" else f"real_{c}"), on="nominal_gap_hours", how="left").rename(
        columns={"position_error_p90_km": "holdout_error_km_p90", "position_predicted_points": "holdout_points"}).to_csv(
        out / "holdout_vs_real_gaps.csv", index=False)

    gfw_time_by_length(args.workers).to_csv(out / "gfw_voyage_time_by_gap_length.csv", index=False)
    totals = {"points": len(frame), "status": frame.status.value_counts().to_dict(),
              "flagged_voyages": len(flagged), "points_in_flagged_voyages": int(frame.flagged.sum()),
              "flag_rule": f"median error > {FLAG_KM} km within {FLAG_OFFSET_H * 60:.0f} min of a GFW point, >= {FLAG_MIN_POINTS} such points"}
    (out / "totals.json").write_text(json.dumps(totals, indent=2, default=int) + "\n")
    print(json.dumps(totals, indent=2, default=int))


if __name__ == "__main__":
    main()
