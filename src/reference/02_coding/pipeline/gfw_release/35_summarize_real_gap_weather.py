#!/usr/bin/env python3
"""E007 V1 weather summary: hourly agreement between true and GFW-version positions, and the
effect on released-record exposure computed over the same hours.

Hourly level, by real gap length: wave-height difference; for the 4 m threshold and the
300/500 km cyclone radii, how often the two positions disagree, split into false alarms
(GFW position says yes, true position says no) and misses (the reverse). Disagreement is
reported among hours where either position meets the condition, and over all hours.

Record level: for each released origin-destination-vessel-type record, the high-wave share
and cyclone shares over its paired hours, once with the true and once with the GFW-version
positions. Only hours with both wave matches enter the wave share.

    python 35_summarize_real_gap_weather.py --run <30_ run> --points <20_ run>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
RELEASE = CODING / ".runs/2026-10-11_1907_E007-one-cell-release/assembly/release_full_precision.parquet"
# Default to the adopted population; an explicit environment selects another frozen run.
RELEASE = Path(os.environ.get("E007_RELEASE_ASSEMBLY", RELEASE.parent)) / "release_full_precision.parquet"
BINS = [0, 1.01, 2.01, 4.01, 6.01, 12.01, 24.01, 48.01]
LABELS = ["1 h", "2 h", "3-4 h", "5-6 h", "7-12 h", "13-24 h", "25-48 h"]
KEY = ["origin_city", "dest_city", "shiptype_group"]


def load(run: Path, points: Path, workers: int) -> pd.DataFrame:
    def one(shard: Path) -> pd.DataFrame:
        frame = pd.read_parquet(shard / "hourly_weather_pairs.parquet")
        cand = pd.read_parquet(points / "shards" / shard.name / "candidates.parquet").set_index("cand")
        for column in KEY:
            frame[column] = cand[column].reindex(frame.cand).to_numpy()
        return frame

    with ThreadPoolExecutor(workers) as pool:
        frame = pd.concat(pool.map(one, sorted((run / "shards").iterdir())), ignore_index=True)
    frame["length"] = pd.cut(frame.gap_hours, BINS, labels=LABELS).astype(object).fillna("at GFW point")
    return frame


def flips(frame: pd.DataFrame, true: np.ndarray, gfw: np.ndarray, label: str) -> dict:
    either = true | gfw
    return {f"{label}_hours": len(frame), f"{label}_true_hours": int(true.sum()), f"{label}_gfw_hours": int(gfw.sum()),
            f"{label}_either_hours": int(either.sum()),
            f"{label}_agree_share_among_either": float((true == gfw)[either].mean()) if either.any() else np.nan,
            f"{label}_false_alarm_hours": int((gfw & ~true).sum()), f"{label}_miss_hours": int((true & ~gfw).sum()),
            f"{label}_disagree_share_all_hours": float((true != gfw).mean()) if len(frame) else np.nan}


def by_length(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for length, block in [("all", frame), *frame.groupby("length", sort=False)]:
        wave = block.loc[block.swh_true.notna() & block.swh_gfw.notna()]
        diff = (wave.swh_gfw - wave.swh_true).abs().to_numpy(float)
        row = {"length": length, "hours": len(block), "wave_paired_hours": len(wave)}
        row.update({f"wave_abs_diff_m_p{int(q * 100)}": float(np.quantile(diff, q)) if len(diff) else np.nan for q in (0.5, 0.9, 0.95, 0.99)})
        row["wave_exactly_equal_share"] = float((diff == 0).mean()) if len(diff) else np.nan
        row.update(flips(wave, (wave.swh_true >= 4).to_numpy(), (wave.swh_gfw >= 4).to_numpy(), "ge4m"))
        for r in (300, 500):
            row.update(flips(block, block[f"tc{r}_true"].to_numpy(bool), block[f"tc{r}_gfw"].to_numpy(bool), f"tc{r}"))
        rows.append(row)
    order = {k: i for i, k in enumerate(["all", "at GFW point", *LABELS])}
    return pd.DataFrame(rows).sort_values("length", key=lambda s: s.map(order))


def records(frame: pd.DataFrame) -> pd.DataFrame:
    released = pd.read_parquet(RELEASE, columns=[*KEY, "voyages", "share_wave_hours_ge4m", "tc300_at_least_pct", "tc500_at_least_pct"])
    frame = frame.merge(released[KEY], on=KEY, how="inner")
    paired = frame.swh_true.notna() & frame.swh_gfw.notna()
    frame = frame.assign(w=paired.astype(float), h4_true=((frame.swh_true >= 4) & paired).astype(float),
                         h4_gfw=((frame.swh_gfw >= 4) & paired).astype(float))
    s = frame.groupby(KEY, observed=True).agg(hours=("w", "size"), wave_hours=("w", "sum"), h4_true=("h4_true", "sum"),
                                               h4_gfw=("h4_gfw", "sum"), tc300_true=("tc300_true", "sum"), tc300_gfw=("tc300_gfw", "sum"),
                                               tc500_true=("tc500_true", "sum"), tc500_gfw=("tc500_gfw", "sum"))
    out = pd.DataFrame({"paired_hours": s.hours, "wave_paired_hours": s.wave_hours,
                        "ge4m_share_true": s.h4_true / s.wave_hours.where(s.wave_hours > 0),
                        "ge4m_share_gfw": s.h4_gfw / s.wave_hours.where(s.wave_hours > 0),
                        "tc300_pct_true": 100 * s.tc300_true / s.hours, "tc300_pct_gfw": 100 * s.tc300_gfw / s.hours,
                        "tc500_pct_true": 100 * s.tc500_true / s.hours, "tc500_pct_gfw": 100 * s.tc500_gfw / s.hours}).reset_index()
    out = out.merge(released, on=KEY)
    for m in ("ge4m_share", "tc300_pct", "tc500_pct"):
        out[f"{m}_diff"] = out[f"{m}_gfw"] - out[f"{m}_true"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    run = args.run.resolve()
    if json.loads((run / "execution.json").read_text()).get("state") != "pass":
        raise ValueError("weather comparison did not pass")
    out = run / "summary"
    out.mkdir(exist_ok=False)
    frame = load(run, args.points.resolve(), args.workers)
    hourly = by_length(frame)
    hourly.to_csv(out / "hourly_agreement_by_gap_length.csv", index=False)
    near = frame.loc[(frame.tc_km_true <= 800) | (frame.tc_km_gfw <= 800)]
    by_length(near).to_csv(out / "hourly_agreement_within_800km_of_a_cyclone.csv", index=False)
    rec = records(frame)
    rec.to_csv(out / "record_exposure_true_vs_gfw_positions.csv", index=False)
    q = lambda s: {p: float(s.abs().quantile(p)) for p in (0.5, 0.9, 0.99, 1.0)}
    enough = rec.loc[rec.paired_hours >= 24]
    summary = {"hours": len(frame), "records": len(rec), "records_with_ge24_paired_hours": len(enough),
               "record_abs_diff_quantiles": {m: q(enough[f"{m}_diff"].dropna()) for m in ("ge4m_share", "tc300_pct", "tc500_pct")},
               "records_any_difference": {m: int(enough[f"{m}_diff"].fillna(0).ne(0).sum()) for m in ("ge4m_share", "tc300_pct", "tc500_pct")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(hourly.round(4).T.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
