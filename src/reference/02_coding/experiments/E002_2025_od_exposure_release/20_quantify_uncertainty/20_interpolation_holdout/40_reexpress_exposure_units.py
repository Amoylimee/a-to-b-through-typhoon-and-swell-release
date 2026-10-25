#!/usr/bin/env python3
"""Express the existing U3 point results as per-window exposure errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[4]
U3 = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release/20_u3_full"
PERCENTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


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
    points = pd.read_parquet(U3 / "holdout_point_results.parquet", columns=[
        "sample_id", "swh_pred_m", "swh_truth_m",
        "tc_within_300km_truth", "tc_within_300km_prediction",
        "tc_within_500km_truth", "tc_within_500km_prediction",
    ])
    points = points.loc[points.sample_id.isin(set(complete.sample_id))].copy()
    point_counts = points.groupby("sample_id").size().rename("point_rows")
    inventory = complete.join(point_counts, on="sample_id")
    if inventory.point_rows.isna().any() or not inventory.point_rows.eq(
        inventory.paired_prediction_points
    ).all():
        raise ValueError("U3 complete-window point inventory does not close")
    data = complete[["sample_id", "nominal_gap_hours"]].copy()
    for radius in (300, 500):
        truth = f"tc_within_{radius}km_truth"
        pred = f"tc_within_{radius}km_prediction"
        paired = points.loc[points[truth].notna() & points[pred].notna()]
        rates = paired.groupby("sample_id", sort=False)[[truth, pred]].mean()
        rates[f"tc{radius}_absolute_error_pp"] = (rates[pred] - rates[truth]).abs() * 100.0
        data = data.merge(rates[[f"tc{radius}_absolute_error_pp"]], on="sample_id",
                          how="left", validate="one_to_one")
    wave = points.loc[points.swh_pred_m.notna() & points.swh_truth_m.notna()]
    means = wave.groupby("sample_id", sort=False)[["swh_pred_m", "swh_truth_m"]].mean()
    means["wave_mean_absolute_error_m"] = (means.swh_pred_m - means.swh_truth_m).abs()
    data = data.merge(means[["wave_mean_absolute_error_m"]], on="sample_id",
                      how="left", validate="one_to_one")
    data.to_parquet(out / "u3_window_exposure_errors_internal.parquet", index=False)
    rows = []
    for gap, group in data.groupby("nominal_gap_hours", sort=True):
        for metric, unit in (("wave_mean_absolute_error_m", "m"),
                             ("tc300_absolute_error_pp", "percentage_points"),
                             ("tc500_absolute_error_pp", "percentage_points")):
            valid = group[metric].dropna().to_numpy(dtype=float)
            row = {"gap_hours": int(gap), "metric": metric, "unit": unit,
                   "windows_selected": int(outcomes.nominal_gap_hours.eq(gap).sum()),
                   "windows_complete": int(len(group)), "windows_with_metric": int(len(valid)),
                   "windows_without_metric": int(len(group) - len(valid))}
            row.update({f"p{int(q*100)}": float(np.quantile(valid, q)) if len(valid) else np.nan
                        for q in PERCENTILES})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "u3_exposure_error_quantiles.csv", index=False)
    report = {"selected_windows": int(len(outcomes)), "complete_windows": int(len(data)),
              "partial_windows_excluded": int(len(outcomes) - len(data)),
              "complete_window_point_rows": int(len(points)),
              "complete_window_point_inventory_closed": True,
              "comparison": "absolute difference of paired per-window means or shares",
              "tc_unit": "percentage points", "wave_unit": "m",
              "source_point_results": str(U3 / "holdout_point_results.parquet")}
    (out / "u3_exposure_conversion_closure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().out), indent=2))
