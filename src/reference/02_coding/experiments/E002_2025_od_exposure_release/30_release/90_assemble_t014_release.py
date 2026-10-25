#!/usr/bin/env python3
"""Combine the closed tolerant bounds with unchanged T013 full-duration rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[3]
T013 = CODING / "outputs/2026-09-24_2048_E002-T013-full-voyage-share-review"
KEY = ["origin_city", "dest_city", "shiptype_group"]


def build(tolerant_draft: Path, out: Path, below_out: Path | None = None) -> dict:
    if ".runs" not in out.resolve().parts:
        raise ValueError("pre-clipping combination draft must stay under .runs")
    out.parent.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(T013 / "01_release_table_voyages_ge5.csv")
    new = pd.read_csv(tolerant_draft)
    if len(base) != 46_981 or len(new) != len(base) or base.duplicated(KEY).any() or new.duplicated(KEY).any():
        raise ValueError("release combination grain changed")
    required = [*KEY, "voyages", "voyage_hours", "wave_matched_hours",
                "new_unbounded_hours", "new_wave_counted_hours", "new_wave_lower_m",
                "new_wave_upper_m", "new_wave_width_m"]
    for radius in (300, 500):
        required.extend([f"new_tc{radius}_hours_upper", f"new_tc{radius}_bounded_possible_hours"])
    merged = base.merge(new[required], on=KEY, how="left", validate="one_to_one", suffixes=("", "_check"))
    for field in ("voyages", "voyage_hours", "wave_matched_hours"):
        if not np.allclose(merged[field], merged[f"{field}_check"], rtol=0, atol=1e-5):
            raise ValueError(f"T013/tolerant census mismatch: {field}")
        merged.drop(columns=f"{field}_check", inplace=True)
    replacements = {"unbounded_hours": "new_unbounded_hours",
                    "wave_unknown_counted_hours": "new_wave_counted_hours",
                    "mean_swh_min_possible_m": "new_wave_lower_m",
                    "mean_swh_max_possible_m": "new_wave_upper_m",
                    "mean_swh_possible_range_m": "new_wave_width_m"}
    for target, source in replacements.items():
        merged[target] = merged[source]
        merged.drop(columns=source, inplace=True)
    no_observed = merged.wave_matched_hours.eq(0)
    no_wave = no_observed & merged.wave_unknown_counted_hours.eq(0)
    merged["wave_mean_missing_reason"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged.loc[no_observed & ~no_wave, "wave_mean_missing_reason"] = "no_matched_wave_hours_reachable_bounds_available"
    merged.loc[no_wave, "wave_mean_missing_reason"] = "no_matched_or_reachable_wave_hours"
    for radius in (300, 500):
        name = f"tc{radius}"
        merged[f"{name}_hours_at_most"] = merged[f"new_{name}_hours_upper"]
        merged[f"{name}_bounded_possible_hours"] = merged[f"new_{name}_bounded_possible_hours"]
        merged[f"{name}_at_most_pct"] = 100 * merged[f"{name}_hours_at_most"] / merged.voyage_hours
        merged.drop(columns=[f"new_{name}_hours_upper", f"new_{name}_bounded_possible_hours"], inplace=True)
        expected = merged[f"{name}_hours_at_least"] + merged[f"{name}_bounded_possible_hours"] + merged.unbounded_hours
        if not np.allclose(merged[f"{name}_hours_at_most"], expected, rtol=0, atol=1e-5):
            raise ValueError(f"{name} hour decomposition fails")
        if (merged[f"{name}_at_least_pct"] > merged[f"{name}_at_most_pct"] + 1e-8).any():
            raise ValueError(f"{name} least exceeds most")
    if any(column.startswith("new_") for column in merged):
        raise ValueError("internal rule prefix leaked")
    merged.to_csv(out, index=False)
    if below_out is not None:
        below = pd.read_csv(tolerant_draft.parent / "05_below_threshold_summary.csv")
        below.rename(columns={"tc300_hours_observed": "tc300_hours_at_least",
                              "tc500_hours_observed": "tc500_hours_at_least",
                              "new_unbounded_hours": "unbounded_hours",
                              "new_wave_counted_hours": "wave_unknown_counted_hours",
                              "new_tc300_bounded_possible_hours": "tc300_bounded_possible_hours",
                              "new_tc500_bounded_possible_hours": "tc500_bounded_possible_hours",
                              "new_tc300_hours_upper": "tc300_hours_at_most",
                              "new_tc500_hours_upper": "tc500_hours_at_most"}, inplace=True)
        if any(name.startswith("new_") for name in below):
            raise ValueError("internal prefix in omitted-combination summary")
        below_out.parent.mkdir(parents=True, exist_ok=True)
        below.to_csv(below_out, index=False)
        if int(merged.voyages.sum() + below.voyages.iloc[0]) != 1_354_168:
            raise ValueError("published plus omitted voyages do not close")
    return {"combinations": len(merged), "voyages": int(merged.voyages.sum()),
            "no_wave_anywhere": int(no_wave.sum()),
            "no_observed_wave_but_bounds": int((no_observed & ~no_wave).sum()),
            "source_t013": str(T013), "source_tolerant_draft": str(tolerant_draft)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerant-draft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--below-out", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.tolerant_draft, args.out, args.below_out), indent=2))
