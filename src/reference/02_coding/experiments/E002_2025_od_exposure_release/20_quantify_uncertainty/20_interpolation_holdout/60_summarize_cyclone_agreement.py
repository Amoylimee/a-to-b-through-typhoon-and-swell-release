#!/usr/bin/env python3
"""Summarise cyclone-proximity agreement in the existing U3 holdout results.

Reads the frozen U3 point results without rerunning the holdout. For each gap
length it reports (i) the share of complete windows whose share of positions
within 300 or 500 km of a cyclone centre is identical for reconstructed and
withheld positions, and (ii) among positions within the radius on either side,
the share whose inside/outside classification agrees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CODING = Path(__file__).resolve().parents[4]
U3 = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release/20_u3_full"
RADII = (300, 500)


def summarize(points: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Return one row per gap length from complete-window points."""
    rows = []
    for gap, ids in windows.groupby("nominal_gap_hours", sort=True).sample_id:
        block = points.loc[points.sample_id.isin(set(ids))]
        row = {"gap_hours": int(gap), "windows_complete": int(len(ids))}
        for radius in RADII:
            truth = block[f"tc_within_{radius}km_truth"].astype(bool)
            pred = block[f"tc_within_{radius}km_prediction"].astype(bool)
            shares = pd.DataFrame({"t": truth, "p": pred, "sample_id": block.sample_id})
            shares = shares.groupby("sample_id")[["t", "p"]].mean()
            near = truth | pred
            row[f"windows_identical_share_{radius}km_pct"] = float(
                100.0 * shares.t.eq(shares.p).mean()
            )
            row[f"near_positions_{radius}km"] = int(near.sum())
            row[f"near_positions_same_class_{radius}km_pct"] = (
                float(100.0 * truth[near].eq(pred[near]).mean()) if near.any() else float("nan")
            )
            row[f"all_positions_class_changed_{radius}km_pct"] = float(
                100.0 * truth.ne(pred).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    outcomes = pd.read_csv(U3 / "holdout_sample_outcomes.csv", usecols=[
        "sample_id", "nominal_gap_hours", "status", "paired_prediction_points",
    ])
    if len(outcomes) != 27_500 or outcomes.sample_id.duplicated().any():
        raise ValueError("U3 outcome inventory differs from the accepted run")
    complete = outcomes.loc[outcomes.status.eq("predicted")].copy()
    if len(complete) != 26_956:
        raise ValueError("U3 complete-window count changed")
    columns = ["sample_id"] + [
        f"tc_within_{radius}km_{side}" for radius in RADII for side in ("truth", "prediction")
    ]
    points = pd.read_parquet(U3 / "holdout_point_results.parquet", columns=columns)
    points = points.loc[points.sample_id.isin(set(complete.sample_id))].copy()
    counts = points.groupby("sample_id").size().rename("point_rows")
    inventory = complete.join(counts, on="sample_id")
    if inventory.point_rows.isna().any() or not inventory.point_rows.eq(
        inventory.paired_prediction_points
    ).all():
        raise ValueError("U3 complete-window point inventory does not close")
    if points[columns[1:]].isna().any().any():
        raise ValueError("cyclone flags are missing for complete-window points")
    summary = summarize(points, complete)
    summary.to_csv(out / "u3_cyclone_agreement_by_gap.csv", index=False)
    report = {
        "complete_windows": int(len(complete)),
        "complete_window_point_rows": int(len(points)),
        "complete_window_point_inventory_closed": True,
        "near_position_definition": "within the radius for the reconstructed or the withheld position",
        "source_point_results": str(U3 / "holdout_point_results.parquet"),
    }
    (out / "u3_cyclone_agreement_closure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().out), indent=2))
