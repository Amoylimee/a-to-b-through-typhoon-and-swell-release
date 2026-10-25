#!/usr/bin/env python3
"""Independently recalculate C4 percentiles and check review-table boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[3]
RUN = CODING / ".runs/2026-09-24_1500_E002-T012-exposure-units"
U1 = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release/10_u1/20_od_shiptype_all_combinations.parquet"
KEY = ["origin_city", "dest_city", "shiptype_group"]
QUANTILES = (50, 75, 90, 95, 99)


def verify(out: Path) -> dict:
    panel = pd.read_parquet(RUN / "50_voyage_exposure_intervals_internal.parquet")
    groups = pd.read_parquet(U1, columns=[*KEY, "voyages"])
    groups["release"] = groups.voyages.ge(5)
    panel = panel.merge(groups[[*KEY, "release"]], on=KEY, how="left", validate="many_to_one")
    if panel.release.isna().any():
        raise ValueError("C4 independent check cannot assign publication set")
    expected = pd.read_csv(out / "01_voyage_error_quantiles.csv")
    comparisons = 0
    for row in expected.itertuples(index=False):
        mask = np.ones(len(panel), dtype=bool) if row.population == "all" else panel.release.to_numpy()
        observed = panel.loc[mask, f"{row.rule}_{row.metric}"].dropna().to_numpy(dtype=float)
        if len(observed) != row.voyages_with_bound or mask.sum() != row.voyages_total:
            raise ValueError(f"C4 denominator mismatch: {row.population}/{row.rule}/{row.metric}")
        for pct in QUANTILES:
            actual = float(np.quantile(observed, pct / 100)) if len(observed) else np.nan
            reported = getattr(row, f"p{pct}")
            if not np.isclose(actual, reported, rtol=0, atol=1e-10, equal_nan=True):
                raise ValueError(f"C4 p{pct} mismatch: {row.population}/{row.rule}/{row.metric}")
            comparisons += 1
    guarantees = pd.read_csv(out / "01a_wave_population_guarantee.csv")
    guarantee_comparisons = 0
    for row in guarantees.itertuples(index=False):
        mask = np.ones(len(panel), dtype=bool) if row.population == "all" else panel.release.to_numpy()
        widths = panel.loc[mask, f"{row.rule}_wave_width_m"].to_numpy(dtype=float)
        if len(widths) != row.voyages_total or np.isfinite(widths).sum() != row.voyages_with_bound:
            raise ValueError(f"wave population guarantee denominator mismatch: {row.population}/{row.rule}")
        finite = np.sort(widths[np.isfinite(widths)])
        for pct in QUANTILES:
            rank = (len(widths) - 1) * pct / 100
            low, high = int(np.floor(rank)), int(np.ceil(rank))
            actual = (float(finite[low] + (rank - low) * (finite[high] - finite[low]))
                      if high < len(finite) else np.nan)
            reported = getattr(row, f"p{pct}")
            if np.isnan(actual):
                if not np.isnan(reported):
                    raise ValueError(f"p{pct} should have no finite population guarantee")
            elif not np.isclose(actual, reported, rtol=0, atol=1e-10):
                raise ValueError(f"wave population guarantee p{pct} mismatch")
            guarantee_comparisons += 1

    release = pd.read_csv(out / "04_release_table_draft_voyages_ge5.csv")
    below = pd.read_csv(out / "05_below_threshold_summary.csv")
    all_groups = pd.read_parquet(U1)
    if len(release) != 46_981 or int(release.voyages.sum()) != 1_034_602:
        raise ValueError("UG1 publication set changed")
    if len(release) + int(below.combinations.iloc[0]) != len(all_groups):
        raise ValueError("publication plus omitted combination count does not close")
    if int(release.voyages.sum() + below.voyages.iloc[0]) != int(all_groups.voyages.sum()):
        raise ValueError("publication plus omitted voyage count does not close")
    for column in ("voyage_hours", "track_hours", "unresolved_track_hours",
                   "wave_matched_hours", "wave_unmatched_hours", "wave_ge4m_matched_hours",
                   "tc300_hours_observed", "tc500_hours_observed"):
        if column in below and column in release:
            actual = release[column].sum() + below[column].iloc[0]
            if not np.isclose(actual, all_groups[column].sum(), atol=1e-4, rtol=0):
                raise ValueError(f"publication plus omitted {column} does not close")
    restricted = {"candidate_id", "mmsi", "vessel_uid", "vessel_uid_clean",
                  "lat", "lon", "latitude", "longitude", "voyages_missing_ship_id"}
    for path in out.glob("*.csv"):
        columns = pd.read_csv(path, nrows=0).columns
        if any(column.lower() in restricted or column.lower().endswith(
            ("_lat", "_lon", "_latitude", "_longitude")) for column in columns):
            raise ValueError(f"restricted internal column in review output: {path.name}")
    report = {"status": "pass", "quantile_cells_independently_recalculated": comparisons,
              "population_guarantee_cells_independently_recalculated": guarantee_comparisons,
              "voyages_recalculated": int(len(panel)), "release_combinations": int(len(release)),
              "release_voyages": int(release.voyages.sum()),
              "review_csv_files_checked_for_restricted_columns": len(list(out.glob("*.csv")))}
    (out / "12_independent_recalculation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().out), indent=2))
