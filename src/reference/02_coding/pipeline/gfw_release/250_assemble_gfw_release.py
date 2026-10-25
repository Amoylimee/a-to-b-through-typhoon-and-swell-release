#!/usr/bin/env python3
"""Assemble the 49-field annual GFW candidate and private audit tables.

Since the E007 rule revision the primary wave field is the mean over wave-matched hours; the
value clipped into the possible range is kept as a secondary field with its flag and shift.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from step6_core import *

RELEASE_COLUMNS = "origin_city,dest_city,shiptype_group,voyages,ships,voyage_hours,track_hours,position_coverage,unresolved_track_hours,wave_matched_hours,wave_unmatched_hours,wave_ge4m_matched_hours,wave_coverage,mean_swh_matched_m,mean_swh_matched_ci95_low,mean_swh_matched_ci95_high,mean_swh_min_possible_m,mean_swh_max_possible_m,mean_swh_possible_range_m,wave_bound_hours,wave_bound_coverage,wave_unboundable_hours,mean_swh_constrained_m,mean_swh_constrained_shift_m,wave_mean_was_constrained,wave_mean_missing_reason,share_wave_hours_ge4m,share_wave_hours_ge4m_ci95_low,share_wave_hours_ge4m_ci95_high,voyages_with_tc300,voyages_with_tc500,single_ship,sampling_interval_unavailable_fewer_than_3_ships,unbounded_hours,wave_unknown_counted_hours,tc300_hours_at_least,tc300_bounded_possible_hours,tc300_hours_at_most,tc300_at_least_pct,tc300_at_most_pct,tc300_at_least_pct_ci95_low,tc300_at_least_pct_ci95_high,tc500_hours_at_least,tc500_bounded_possible_hours,tc500_hours_at_most,tc500_at_least_pct,tc500_at_most_pct,tc500_at_least_pct_ci95_low,tc500_at_least_pct_ci95_high".split(",")


def release_bootstrap_fields(frame):
    # Internal bootstrap keys use the metric's full name; the inherited release
    # uses the unitless stem for its wave CI fields.
    return frame.rename(columns={f"mean_swh_matched_m_ci95_{side}":
                                 f"mean_swh_matched_ci95_{side}" for side in ("low", "high")})


def assemble(root: Path):
    source = binding()
    root = inside_runs(root)
    prep = json.loads((root / "preparation/preparation.json").read_text())
    state = json.loads((root / "bounds/status.json").read_text())
    if state["state"] != "completed" or sorted(state["completed"]) != SHARDS or state["probe_candidates"] is not None or state["failures"]:
        raise ValueError("annual release requires 64 complete census bound shards")
    rule = revision()
    if state.get("fix_time_tolerance_hours") != rule["fix_time_tolerance_hours"]:
        raise ValueError("bounds were not computed with the revised GFW time-stamp tolerance")
    parts, receipts = [], []
    for shard in SHARDS:
        report = json.loads((root / "bounds" / shard / "status.json").read_text())
        if record(Path(report["output"]["path"])) != report["output"]:
            raise ValueError("bound payload changed")
        parts.append(pd.read_parquet(report["output"]["path"]))
        receipts.append(report)
    if sum(r["weather_rows"] for r in receipts) != source["counts"]["weather_rows"]:
        raise ValueError("weather census row denominator differs")
    extras = pd.concat(parts, ignore_index=True)
    panel = pd.read_parquet(root / "preparation/voyage_panel.parquet").merge(extras, on="candidate_id", how="left", validate="one_to_one")
    numeric = [c for c in extras if c != "candidate_id"]
    panel[numeric] = panel[numeric].fillna(0)
    numeric_close(panel.unknown_hours, panel.unresolved_track_hours, "per-voyage unknown time")
    for rule in ("main", "zero"):
        denominator = panel.wave_matched_hours + panel[f"{rule}_wave_counted_hours"]
        panel[f"{rule}_wave_lower_m"] = safe_ratio(panel.wave_integral_m_hours+panel[f"{rule}_wave_min_integral"], denominator)
        panel[f"{rule}_wave_upper_m"] = safe_ratio(panel.wave_integral_m_hours+panel[f"{rule}_wave_max_integral"], denominator)
        panel[f"{rule}_wave_range_m"] = panel[f"{rule}_wave_upper_m"]-panel[f"{rule}_wave_lower_m"]
        for radius in (300, 500):
            panel[f"{rule}_tc{radius}_range_pp"] = 100*safe_ratio(panel[f"{rule}_tc{radius}_possible_hours"], panel.voyage_hours)
    grouped = aggregate.aggregate_panel(panel)
    bound_columns = [c for c in extras if c != "candidate_id"]
    group_bounds = panel.groupby(KEY, sort=True, observed=True)[bound_columns].sum().reset_index()
    grouped = grouped.merge(group_bounds, on=KEY, validate="one_to_one")
    grouped["unbounded_hours"] = grouped.main_unbounded_hours
    grouped["wave_unknown_counted_hours"] = grouped.main_wave_counted_hours
    denominator = grouped.wave_matched_hours+grouped.main_wave_counted_hours
    grouped["mean_swh_min_possible_m"] = safe_ratio(grouped.wave_integral_m_hours+grouped.main_wave_min_integral, denominator)
    grouped["mean_swh_max_possible_m"] = safe_ratio(grouped.wave_integral_m_hours+grouped.main_wave_max_integral, denominator)
    grouped["mean_swh_possible_range_m"] = grouped.mean_swh_max_possible_m-grouped.mean_swh_min_possible_m
    grouped["wave_bound_hours"] = denominator
    grouped["wave_bound_coverage"] = safe_ratio(denominator, grouped.voyage_hours)
    grouped["wave_unboundable_hours"] = grouped.voyage_hours-denominator
    # Secondary: the matched mean clipped into the possible range (the pre-revision point value).
    grouped["mean_swh_constrained_m"] = np.clip(grouped.mean_swh_matched_m, grouped.mean_swh_min_possible_m, grouped.mean_swh_max_possible_m)
    grouped["mean_swh_constrained_shift_m"] = grouped.mean_swh_constrained_m-grouped.mean_swh_matched_m
    grouped["wave_mean_was_constrained"] = grouped.mean_swh_constrained_shift_m.abs().gt(1e-9).astype("boolean")
    grouped.loc[grouped.mean_swh_matched_m.isna(), "wave_mean_was_constrained"] = pd.NA
    grouped["wave_mean_missing_reason"] = pd.Series(pd.NA, index=grouped.index, dtype="string")
    none = grouped.wave_matched_hours.eq(0)
    grouped.loc[none & denominator.gt(0), "wave_mean_missing_reason"] = "no_matched_wave_hours_reachable_bounds_available"
    grouped.loc[denominator.eq(0), "wave_mean_missing_reason"] = "no_matched_or_reachable_wave_hours"
    for radius in (300, 500):
        prefix = f"tc{radius}"
        grouped[f"{prefix}_hours_at_least"] = grouped[f"tc{radius}_positive_snapshot_hours"]
        grouped[f"{prefix}_bounded_possible_hours"] = grouped[f"main_tc{radius}_possible_hours"]-grouped.main_unbounded_hours
        grouped[f"{prefix}_hours_at_most"] = grouped[f"{prefix}_hours_at_least"]+grouped[f"main_tc{radius}_possible_hours"]
        grouped[f"{prefix}_at_least_pct"] = 100*safe_ratio(grouped[f"{prefix}_hours_at_least"], grouped.voyage_hours)
        grouped[f"{prefix}_at_most_pct"] = 100*safe_ratio(grouped[f"{prefix}_hours_at_most"], grouped.voyage_hours)
    grouped["single_ship"] = grouped.ships.eq(1)
    grouped["sampling_interval_unavailable_fewer_than_3_ships"] = grouped.ships.lt(3)
    selected = grouped.loc[grouped.voyages.ge(5)].copy()
    boot_sources = []
    for mode in ("base", "wave"):
        folder = root / f"bootstrap_{mode}"
        receipt = json.loads((folder / "status.json").read_text())
        if receipt["state"] != "completed" or receipt["failures"] or receipt["combinations"] != len(selected):
            raise ValueError("bootstrap does not cover the complete selected population")
        if record(Path(receipt["output"]["path"])) != receipt["output"]: raise ValueError("bootstrap payload changed")
        boot = release_bootstrap_fields(pd.read_parquet(receipt["output"]["path"]))
        check = selected[KEY+["ships"]].merge(boot[KEY+["ships"]], on=KEY, validate="one_to_one", suffixes=("", "_boot"))
        if not check.ships.eq(check.ships_boot).all(): raise ValueError("bootstrap ship clusters differ")
        metrics = ["share_wave_hours_ge4m", "tc300_at_least_pct", "tc500_at_least_pct"] if mode == "base" else ["mean_swh_matched_m"]
        joined = selected[KEY+metrics].merge(boot[KEY+metrics], on=KEY, validate="one_to_one", suffixes=("", "_boot"))
        for metric in metrics: numeric_close(joined[metric], joined[metric+"_boot"], metric)
        intervals = [c for c in boot if c.endswith("_ci95_low") or c.endswith("_ci95_high")]
        selected = selected.merge(boot[KEY+intervals], on=KEY, validate="one_to_one")
        boot_sources.append(record(folder / "status.json"))
    private = root / "assembly"; private.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(private / "voyage_intervals.parquet", index=False)
    grouped.to_parquet(private / "all_combinations.parquet", index=False)
    selected[RELEASE_COLUMNS].sort_values(KEY).to_parquet(private / "release_full_precision.parquet", index=False)
    below = grouped.loc[grouped.voyages.lt(5)]
    summable = ["voyages", *aggregate.WEATHER_SUMS, "unbounded_hours", "wave_unknown_counted_hours",
                "tc300_bounded_possible_hours", "tc500_bounded_possible_hours", "tc300_hours_at_most", "tc500_hours_at_most"]
    summary = {"combinations": len(below), **{c: float(below[c].sum()) for c in summable}}
    write_json(private / "below_threshold_summary.json", summary)
    quantiles = []
    flags = grouped[KEY+["voyages"]].assign(release=lambda d:d.voyages.ge(5))
    panel = panel.merge(flags[KEY+["release"]], on=KEY, validate="many_to_one")
    for population, part in (("all", panel), ("release_ge5", panel.loc[panel.release])):
        for rule in ("main", "zero"):
            for metric in ("wave_range_m", "tc300_range_pp", "tc500_range_pp"):
                values = part[f"{rule}_{metric}"]
                row = {"population": population, "rule": rule, "metric": metric, "voyages": len(part), "available": int(values.notna().sum())}
                for q in (.5,.9,.95,.99): row[f"p{int(q*100)}"] = float(values.quantile(q))
                quantiles.append(row)
    pd.DataFrame(quantiles).to_csv(private / "missing_exposure_quantiles.csv", index=False, lineterminator="\n")
    result = {"state": "assembled", "candidates": len(panel), "all_combinations": len(grouped),
              "release_combinations": len(selected), "release_voyages": int(selected.voyages.sum()),
              "below_combinations": len(below), "below_voyages": int(below.voyages.sum()),
              "release_fields": len(RELEASE_COLUMNS), "fewer_than_3_ships": int(selected.ships.lt(3).sum()),
              "bounds_weather_rows": sum(r["weather_rows"] for r in receipts), "unknown_hours": float(panel.unknown_hours.sum()),
              "totals": {c: float(panel[c].sum()) for c in aggregate.WEATHER_SUMS},
              "authorization": source["authorization"], "bootstrap_sources": boot_sources,
              "network_requests": 0, "products_modified": 0, "adopted": False}
    result.update(provenance([HERE / "step6_core.py", Path(__file__), DECISION, REVISION]))
    result["outputs"] = {p.name: record(p) for p in private.iterdir() if p.is_file()}
    write_json(private / "status.json", result)
    print(json.dumps({k: result[k] for k in ("state", "release_combinations", "release_voyages", "release_fields")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", type=Path, required=True)
    assemble(parser.parse_args().run)
