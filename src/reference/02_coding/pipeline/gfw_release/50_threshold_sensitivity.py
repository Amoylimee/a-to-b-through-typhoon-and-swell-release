#!/usr/bin/env python3
"""E007 V4: what the minimum-voyage release threshold keeps and drops.

For thresholds 1, 3, 5, 10 and 20 voyages, reports how many records, voyages and voyage hours
are kept, how this differs by vessel type and origin country, and whether the voyage time that
is dropped has different wave and cyclone exposure from the time that is kept. Exposure here
is pooled over voyage time (sum of hours / sum of hours), the same pooling as the release.

    python 50_threshold_sensitivity.py --out <new .runs dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
COMBINATIONS = CODING / ".runs/2026-10-07_1410_E006-gfw-release-step6/assembly/all_combinations.parquet"
THRESHOLDS = (1, 3, 5, 10, 20)
SUMS = ["voyages", "voyage_hours", "wave_matched_hours", "wave_integral_m_hours", "wave_ge4m_matched_hours",
        "tc300_hours_at_least", "tc500_hours_at_least"]


def pooled(frame: pd.DataFrame) -> dict:
    s = frame[SUMS].sum()
    return {"records": len(frame), "voyages": int(s.voyages), "voyage_hours": float(s.voyage_hours),
            "wave_mean_m": s.wave_integral_m_hours / s.wave_matched_hours if s.wave_matched_hours else np.nan,
            "share_wave_hours_ge4m": s.wave_ge4m_matched_hours / s.wave_matched_hours if s.wave_matched_hours else np.nan,
            "tc300_at_least_pct": 100 * s.tc300_hours_at_least / s.voyage_hours if s.voyage_hours else np.nan,
            "tc500_at_least_pct": 100 * s.tc500_hours_at_least / s.voyage_hours if s.voyage_hours else np.nan}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    out.mkdir(parents=True, exist_ok=False)
    combos = pd.read_parquet(COMBINATIONS)
    combos["origin_country"] = combos.origin_city.str.split(":").str[0]
    total = pooled(combos)

    rows = []
    for n in THRESHOLDS:
        kept, dropped = combos.loc[combos.voyages.ge(n)], combos.loc[combos.voyages.lt(n)]
        k, d = pooled(kept), pooled(dropped)
        rows.append({"minimum_voyages": n, **{f"kept_{key}": v for key, v in k.items()},
                     "kept_share_of_voyages": k["voyages"] / total["voyages"],
                     "kept_share_of_voyage_hours": k["voyage_hours"] / total["voyage_hours"],
                     "kept_cities": int(pd.unique(pd.concat([kept.origin_city, kept.dest_city])).size),
                     "kept_origin_countries": int(kept.origin_country.nunique()),
                     **{f"dropped_{key}": v for key, v in d.items()}})
    pd.DataFrame(rows).to_csv(out / "threshold_summary.csv", index=False)

    by_type = []
    for n in THRESHOLDS:
        for ship, block in combos.groupby("shiptype_group"):
            kept = block.loc[block.voyages.ge(n)]
            by_type.append({"minimum_voyages": n, "shiptype_group": ship, "records": len(kept),
                            "kept_share_of_voyages": kept.voyages.sum() / block.voyages.sum(),
                            **{f"kept_{key}": v for key, v in pooled(kept).items() if key not in ("records", "voyages", "voyage_hours")}})
    pd.DataFrame(by_type).to_csv(out / "threshold_by_shiptype.csv", index=False)

    country = combos.groupby("origin_country").voyages.sum().rename("voyages").to_frame()
    for n in THRESHOLDS:
        country[f"kept_share_n{n}"] = combos.loc[combos.voyages.ge(n)].groupby("origin_country").voyages.sum().reindex(country.index).fillna(0) / country.voyages
    country.sort_values("voyages", ascending=False).to_csv(out / "threshold_by_origin_country.csv")

    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()
    execution = {"step": "E007 V4: release threshold sensitivity", "state": "pass", "git_head": git("rev-parse", "HEAD"),
                 "git_dirty_experiment": bool(git("status", "--porcelain", "--", ".")),
                 "input": str(COMBINATIONS.relative_to(CODING)), "combinations": len(combos), "thresholds": list(THRESHOLDS),
                 "countries_with_under_half_of_voyages_kept_at_n5": int((country.kept_share_n5 < 0.5).sum()),
                 "countries": len(country)}
    (out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(pd.DataFrame(rows).round(4).T.to_string())
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
