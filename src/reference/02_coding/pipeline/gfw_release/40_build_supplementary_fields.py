#!/usr/bin/env python3
"""E007 V3: supplementary per-record fields and reference lists for the GFW-version release.

For each released (origin_city, dest_city, shiptype_group) record:
- time coverage across the year: months with voyage time, months with departures, and the
  largest single-month share of voyage hours;
- how the ten-minute track time was obtained: at or between GFW hourly points no more than
  4 h apart, in 4-48 h gaps, or unknown;
- wave support: matched hours plus unknown hours with a reachable-area wave bound, and their
  share of voyage hours (the denominator of the released wave mean and its bounds);
- the observed wave mean before it is constrained to the possible range, and how far the
  released value moved.
Also lists the 2025 IBTrACS systems retained by the production cyclone filter.

    python 40_build_supplementary_fields.py --out <new .runs dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
STEP6 = CODING / ".runs/2026-10-07_1410_E006-gfw-release-step6/assembly"
TRACKS = CODING / ".runs/2026-10-06_2341_E006-gfw-ten-minute-tracks/shards"
WEATHER = CODING / "pipeline/40_construct_variables/60_construct_voyage_weather_facts"
sys.path.insert(0, str(WEATHER))
from weather_core.tropical_cyclone import load_ibtracs_snapshot  # noqa: E402

KEY = ["origin_city", "dest_city", "shiptype_group"]
NS_HOUR = 3_600_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def month_hours(voyages: pd.DataFrame) -> pd.DataFrame:
    dep = pd.DatetimeIndex(voyages.departure_time)
    arr = pd.DatetimeIndex(voyages.arrival_time)
    first = dep.year * 12 + dep.month - 1
    span = (arr.year * 12 + arr.month - 1) - first + 1
    rows = np.repeat(np.arange(len(voyages)), span)
    index = np.repeat(first.to_numpy(), span) + (np.arange(len(rows)) - np.repeat(np.cumsum(span) - span, span))
    start = pd.to_datetime({"year": index // 12, "month": index % 12 + 1, "day": 1}).to_numpy("datetime64[ns]")
    end = pd.to_datetime({"year": (index + 1) // 12, "month": (index + 1) % 12 + 1, "day": 1}).to_numpy("datetime64[ns]")
    lo = np.maximum(start, dep.tz_convert(None).to_numpy()[rows])
    hi = np.minimum(end, arr.tz_convert(None).to_numpy()[rows])
    hours = (hi - lo).astype("timedelta64[ns]").astype(np.int64) / NS_HOUR
    frame = voyages.iloc[rows][KEY].reset_index(drop=True)
    frame["month"] = index % 12 + 1
    frame["hours"] = np.clip(hours, 0, None)
    return frame.loc[frame.hours.gt(0)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    out.mkdir(parents=True, exist_ok=False)

    release = pd.read_parquet(STEP6 / "release_full_precision.parquet")
    voyages = pd.read_parquet(STEP6 / "voyage_intervals.parquet",
                              columns=["candidate_id", "source_shard", *KEY, "departure_time", "arrival_time", "voyage_hours",
                                       "wave_matched_hours", "wave_integral_m_hours"])
    voyages = voyages.merge(release[KEY], on=KEY, how="inner")
    if voyages.groupby(KEY).size().reindex(pd.MultiIndex.from_frame(release[KEY])).to_numpy().tolist() != release.voyages.tolist():
        raise ValueError("voyage counts do not reproduce the release table")

    # Year coverage.
    months = month_hours(voyages)
    by_month = months.groupby([*KEY, "month"], observed=True).hours.sum().reset_index()
    cover = by_month.groupby(KEY, observed=True).agg(active_months=("month", "nunique"), _hours=("hours", "sum"),
                                                     _max=("hours", "max")).reset_index()
    cover["max_month_share_of_voyage_hours"] = cover._max / cover._hours
    dep_months = voyages.assign(m=pd.DatetimeIndex(voyages.departure_time).month).groupby(KEY, observed=True).m.nunique()
    cover = cover.drop(columns=["_hours", "_max"]).merge(dep_months.rename("departure_months").reset_index(), on=KEY)

    # Track-time composition from the step-4 ten-minute cell counts.
    cells = pd.concat([pd.read_parquet(p, columns=["candidate_id", "grid_rows", "observed_ais_rows", "cells_gfw_sea_path_le4h",
                                                   "cells_gfw_sea_path_4_24h", "cells_gfw_sea_path_24_48h", "unresolved_rows"])
                       for p in sorted(TRACKS.glob("shard_*/candidate_summary.parquet"))])
    cells = voyages[["candidate_id", *KEY]].merge(cells, on="candidate_id", how="left", validate="one_to_one")
    if cells.grid_rows.isna().any():
        raise ValueError("released voyage without step-4 cell counts")
    parts = cells.observed_ais_rows + cells.cells_gfw_sea_path_le4h + cells.cells_gfw_sea_path_4_24h + cells.cells_gfw_sea_path_24_48h + cells.unresolved_rows
    if not parts.eq(cells.grid_rows).all():
        raise ValueError("step-4 cell classes do not add up to the grid")
    comp = cells.groupby(KEY, observed=True)[["grid_rows", "observed_ais_rows", "cells_gfw_sea_path_le4h", "cells_gfw_sea_path_4_24h",
                                              "cells_gfw_sea_path_24_48h", "unresolved_rows"]].sum()
    comp = pd.DataFrame({
        "share_time_at_or_within_4h_of_gfw_points": (comp.observed_ais_rows + comp.cells_gfw_sea_path_le4h) / comp.grid_rows,
        "share_time_in_4_48h_gaps": (comp.cells_gfw_sea_path_4_24h + comp.cells_gfw_sea_path_24_48h) / comp.grid_rows,
        "share_time_unknown": comp.unresolved_rows / comp.grid_rows,
    }).reset_index()

    # Wave support and the constraint of the released wave mean.
    wave = voyages.groupby(KEY, observed=True)[["wave_matched_hours", "wave_integral_m_hours"]].sum().reset_index()
    fields = release[[*KEY, "voyages", "voyage_hours", "wave_matched_hours", "wave_unknown_counted_hours", "mean_swh_full_voyage_m",
                      "mean_swh_min_possible_m", "mean_swh_max_possible_m"]].merge(
        wave.rename(columns={"wave_matched_hours": "_matched", "wave_integral_m_hours": "_integral"}), on=KEY)
    if not np.allclose(fields._matched, fields.wave_matched_hours, atol=1e-6):
        raise ValueError("matched wave hours do not reproduce the release table")
    fields["wave_support_hours"] = fields.wave_matched_hours + fields.wave_unknown_counted_hours
    fields["wave_support_coverage"] = fields.wave_support_hours / fields.voyage_hours
    fields["wave_mean_observed_m"] = np.where(fields._matched > 0, fields._integral / fields._matched.where(fields._matched > 0), np.nan)
    expected = np.clip(fields.wave_mean_observed_m, fields.mean_swh_min_possible_m, fields.mean_swh_max_possible_m)
    check = fields.wave_mean_observed_m.notna()
    if not np.allclose(expected[check], fields.mean_swh_full_voyage_m[check], atol=1e-9, equal_nan=True):
        raise ValueError("released wave mean is not the observed mean constrained to its possible range")
    fields["wave_constraint_adjustment_m"] = fields.mean_swh_full_voyage_m - fields.wave_mean_observed_m
    fields["wave_constrained"] = fields.wave_constraint_adjustment_m.abs().gt(1e-9)
    fields = fields.drop(columns=["_matched", "_integral"])

    table = release[KEY].merge(cover, on=KEY).merge(comp, on=KEY).merge(fields, on=KEY)
    if len(table) != len(release):
        raise ValueError("supplementary fields do not cover every released record")
    table.to_csv(out / "record_supplementary_fields.csv", index=False)

    adj = table.loc[table.wave_constrained, "wave_constraint_adjustment_m"]
    q = lambda s, p: float(s.quantile(p)) if len(s) else None
    census = {
        "records": len(table),
        "active_months": table.active_months.value_counts().sort_index().to_dict(),
        "active_months_quantiles": {p: q(table.active_months, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "records_with_over_half_of_hours_in_one_month": int((table.max_month_share_of_voyage_hours > 0.5).sum()),
        "share_time_unknown_quantiles": {p: q(table.share_time_unknown, p) for p in (0.5, 0.9, 0.99)},
        "share_time_in_4_48h_gaps_quantiles": {p: q(table.share_time_in_4_48h_gaps, p) for p in (0.5, 0.9, 0.99)},
        "wave_support_coverage_quantiles": {p: q(table.wave_support_coverage, p) for p in (0.01, 0.1, 0.5)},
        "records_wave_support_coverage_below_0_9": int((table.wave_support_coverage < 0.9).sum()),
        "records_with_observed_wave_mean": int(table.wave_mean_observed_m.notna().sum()),
        "records_wave_constrained": int(table.wave_constrained.sum()),
        "records_wave_constrained_up": int((adj > 0).sum()),
        "records_wave_constrained_down": int((adj < 0).sum()),
        "wave_constraint_abs_adjustment_m_quantiles": {p: q(adj.abs(), p) for p in (0.5, 0.9, 0.99, 1.0)},
        "records_wave_constrained_by_over_0_1m": int((adj.abs() > 0.1).sum()),
    }
    (out / "census.json").write_text(json.dumps(census, indent=2, default=float) + "\n")

    config = json.loads((WEATHER / "config/2025_ten_minute_weather.json").read_text())
    fixes, _ = load_ibtracs_snapshot(Path(config["ibtracs_path"]), expected_sha256=config["ibtracs_sha256"],
                                     time_end_exclusive="2026-01-02T00:00:00Z")
    in_year = fixes.loc[fixes.fix_time_utc.dt.year.eq(2025)]
    raw = pd.read_csv(config["ibtracs_path"], skiprows=[1], usecols=["SID", "SEASON", "NAME", "BASIN"], dtype=str,
                      low_memory=False).drop_duplicates("SID")
    systems = in_year.groupby("SID").agg(first_fix_utc=("fix_time_utc", "min"), last_fix_utc=("fix_time_utc", "max"),
                                         fixes_2025=("fix_time_utc", "size"),
                                         track_types=("track_type", lambda s: ";".join(sorted(set(s)))),
                                         natures=("nature", lambda s: ";".join(sorted(set(s))))).reset_index()
    systems = systems.merge(raw, on="SID", how="left")
    systems.to_csv(out / "ibtracs_2025_systems.csv", index=False)

    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()
    execution = {"step": "E007 V3: supplementary per-record fields and reference lists", "state": "pass",
                 "git_head": git("rev-parse", "HEAD"), "git_dirty_experiment": bool(git("status", "--porcelain", "--", ".")),
                 "inputs": {str(p.relative_to(CODING)): sha256(p) for p in
                            [STEP6 / "release_full_precision.parquet", STEP6 / "voyage_intervals.parquet"]},
                 "ibtracs_sha256": config["ibtracs_sha256"], "records": len(table),
                 "ibtracs_systems_2025": len(systems), "ibtracs_fixes_2025": len(in_year),
                 "season_not_2025_but_fixes_in_2025": int(systems.SEASON.ne("2025").sum())}
    (out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps(census, indent=2, default=float))
    print(json.dumps({k: execution[k] for k in ["records", "ibtracs_systems_2025", "ibtracs_fixes_2025",
                                                "season_not_2025_but_fixes_in_2025"]}, indent=2))


if __name__ == "__main__":
    main()
