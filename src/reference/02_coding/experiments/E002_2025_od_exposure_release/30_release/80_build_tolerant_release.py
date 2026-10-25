#!/usr/bin/env python3
"""Close T014 tolerant voyage intervals and assemble a comparison release draft."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[3]
RUN: Path
PRIOR = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release"
PRIOR_REVIEW = CODING / "outputs/2026-09-24_0135_E002-uncertainty-review"
KEY = ["origin_city", "dest_city", "shiptype_group"]
PERCENTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: float, expected: float, label: str, atol: float = 1e-4) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f"{label} does not close: {actual} != {expected}")


def _sum_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in ["candidate_id", *KEY]]


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    pieces = sorted((RUN / "30_census_full").glob("shard_*_voyage_bounds.parquet"))
    if len(pieces) != 253:
        raise ValueError(f"C2 census has {len(pieces)} of 253 shard files")
    run_status = json.loads((RUN / "30_census_full/full_run_status.json").read_text())
    if run_status["failures"]:
        raise ValueError("C2 full run has failed shards")
    shard_reports = [json.loads(path.read_text()) for path in sorted(
        (RUN / "30_census_full").glob("shard_*_closure.json")
    )]
    if len(shard_reports) != 253:
        raise ValueError("C2 shard closure reports are incomplete")
    total_weather_rows = sum(item["weather_rows"] for item in shard_reports)
    if total_weather_rows != 174_615_386:
        raise ValueError("C2 scanned weather rows do not match the adopted product")
    additions = pd.concat([pd.read_parquet(path) for path in pieces], ignore_index=True)
    if additions.candidate_id.duplicated().any():
        raise ValueError("C2 candidate_id repeated across shards")
    panel = pd.read_parquet(PRIOR / "10_u1/10_voyage_exposure_panel.parquet")
    if len(panel) != 1_354_168 or panel.candidate_id.duplicated().any():
        raise ValueError("U1 panel grain changed")
    panel = panel.merge(additions, on="candidate_id", how="left", validate="one_to_one",
                        indicator=True)
    expected_with_unknown = panel.unresolved_track_hours.gt(1e-10)
    if not panel._merge.eq("both").equals(expected_with_unknown):
        raise ValueError("C2 voyage coverage does not match U1 unknown hours")
    panel.drop(columns="_merge", inplace=True)
    for column in _sum_columns(additions):
        panel[column] = panel[column].fillna(0)
    _close(panel.unknown_segments.sum(), 18_074_055, "unknown segments", atol=0)
    _close(panel.unknown_hours.sum(), 16_744_981.166388888, "unknown hours")
    _close(panel.old_unbounded_segments.sum(), 436_213, "old unbounded segments", atol=0)
    _close(panel.old_unbounded_hours.sum(), 377_819.27222222264, "old unbounded hours")
    _close(panel.unresolved_track_hours.sum(), panel.unknown_hours.sum(), "U1/C2 unknown hours")
    for rule in ("old", "new"):
        count_parts = (panel[f"{rule}_unbounded_segments"]
                       + panel[f"{rule}_wave_counted_segments"]
                       + panel[f"{rule}_wave_no_wet_segments"]
                       + panel[f"{rule}_wave_speed_infeasible_segments"])
        if not count_parts.eq(panel.unknown_segments).all():
            raise ValueError(f"{rule} segment classification fails per voyage")
        panel[f"{rule}_wave_uncounted_hours"] = (
            panel.unknown_hours - panel[f"{rule}_unbounded_hours"]
            - panel[f"{rule}_wave_counted_hours"]
        )
        if (panel[f"{rule}_wave_uncounted_hours"] < -1e-7).any():
            raise ValueError(f"{rule} counted wave hours exceed unknown hours")
        if (panel[f"{rule}_wave_min_integral"] >
                panel[f"{rule}_wave_max_integral"] + 1e-7).any():
            raise ValueError(f"{rule} wave integral lower exceeds upper")
        for radius in (300, 500):
            observed = panel[f"tc{radius}_positive_snapshot_hours"]
            possible = panel[f"{rule}_tc{radius}_possible_hours"]
            panel[f"{rule}_tc{radius}_lower_pp"] = 100.0 * observed / panel.voyage_hours
            panel[f"{rule}_tc{radius}_upper_pp"] = 100.0 * (observed + possible) / panel.voyage_hours
            panel[f"{rule}_tc{radius}_width_pp"] = 100.0 * possible / panel.voyage_hours
            if (panel[f"{rule}_tc{radius}_lower_pp"] >
                    panel[f"{rule}_tc{radius}_upper_pp"] + 1e-8).any():
                raise ValueError(f"{rule} TC{radius} voyage bounds reversed")
            if (panel[f"{rule}_tc{radius}_upper_pp"] > 100 + 1e-7).any():
                raise ValueError(f"{rule} TC{radius} voyage upper exceeds full duration")
        wave_denominator = panel.wave_matched_hours + panel[f"{rule}_wave_counted_hours"]
        panel[f"{rule}_wave_denominator_hours"] = wave_denominator
        panel[f"{rule}_wave_lower_m"] = (
            (panel.wave_integral_m_hours + panel[f"{rule}_wave_min_integral"])
            / wave_denominator.where(wave_denominator > 0)
        )
        panel[f"{rule}_wave_upper_m"] = (
            (panel.wave_integral_m_hours + panel[f"{rule}_wave_max_integral"])
            / wave_denominator.where(wave_denominator > 0)
        )
        panel[f"{rule}_wave_width_m"] = panel[f"{rule}_wave_upper_m"] - panel[f"{rule}_wave_lower_m"]
        if (panel[f"{rule}_wave_width_m"].dropna() < -1e-8).any():
            raise ValueError(f"{rule} wave voyage bounds reversed")
    panel.to_parquet(RUN / "50_voyage_exposure_intervals_internal.parquet", index=False)

    aggregate_columns = [
        "voyage_hours", "track_hours", "wave_matched_hours", "wave_integral_m_hours",
        "wave_ge4m_matched_hours", "tc300_positive_snapshot_hours", "tc500_positive_snapshot_hours",
        "unresolved_track_hours", "wave_unmatched_hours", "old_wave_uncounted_hours",
        "new_wave_uncounted_hours", *_sum_columns(additions),
    ]
    aggregate_columns = list(dict.fromkeys(aggregate_columns))
    groups = panel.groupby(KEY, dropna=False, observed=True, sort=False)[aggregate_columns].sum().reset_index()
    u1 = pd.read_parquet(PRIOR / "10_u1/20_od_shiptype_all_combinations.parquet")
    if len(groups) != len(u1):
        raise ValueError("C3 combination count differs from U1")
    groups = u1.merge(groups, on=KEY, how="left", validate="one_to_one", suffixes=("", "_check"))
    for column in ("voyage_hours", "track_hours", "wave_matched_hours", "wave_integral_m_hours",
                   "tc300_positive_snapshot_hours", "tc500_positive_snapshot_hours"):
        if not np.allclose(groups[column], groups[f"{column}_check"], rtol=0, atol=1e-5):
            raise ValueError(f"C3 group aggregation differs from U1: {column}")
        groups.drop(columns=f"{column}_check", inplace=True)
    for column in ("wave_ge4m_matched_hours", "unresolved_track_hours", "wave_unmatched_hours"):
        if f"{column}_check" in groups:
            if not np.allclose(groups[column], groups[f"{column}_check"], rtol=0, atol=1e-5):
                raise ValueError(f"C3 group aggregation differs from U1: {column}")
            groups.drop(columns=f"{column}_check", inplace=True)
    for rule in ("old", "new"):
        for radius in (300, 500):
            lower = groups[f"tc{radius}_positive_snapshot_hours"]
            possible = groups[f"{rule}_tc{radius}_possible_hours"]
            groups[f"{rule}_tc{radius}_hours_lower"] = lower
            groups[f"{rule}_tc{radius}_hours_upper"] = lower + possible
            groups[f"{rule}_tc{radius}_lower_pp"] = 100.0 * lower / groups.voyage_hours
            groups[f"{rule}_tc{radius}_upper_pp"] = 100.0 * (lower + possible) / groups.voyage_hours
            groups[f"{rule}_tc{radius}_width_pp"] = 100.0 * possible / groups.voyage_hours
            groups[f"{rule}_tc{radius}_bounded_possible_hours"] = possible - groups[f"{rule}_unbounded_hours"]
            if (groups[f"{rule}_tc{radius}_bounded_possible_hours"] < -1e-8).any():
                raise ValueError("TC possible hours below unbounded contribution")
        wave_denominator = groups.wave_matched_hours + groups[f"{rule}_wave_counted_hours"]
        groups[f"{rule}_wave_denominator_hours"] = wave_denominator
        groups[f"{rule}_wave_lower_m"] = (
            (groups.wave_integral_m_hours + groups[f"{rule}_wave_min_integral"])
            / wave_denominator.where(wave_denominator > 0)
        )
        groups[f"{rule}_wave_upper_m"] = (
            (groups.wave_integral_m_hours + groups[f"{rule}_wave_max_integral"])
            / wave_denominator.where(wave_denominator > 0)
        )
        groups[f"{rule}_wave_width_m"] = groups[f"{rule}_wave_upper_m"] - groups[f"{rule}_wave_lower_m"]
        if (groups[f"{rule}_wave_width_m"].dropna() < -1e-8).any():
            raise ValueError("group wave lower exceeds upper")

    old_u2 = pd.read_csv(PRIOR_REVIEW / "19_u2_full_od_shiptype_exposure_bounds.csv")
    previous = groups[KEY + ["old_tc300_hours_upper", "old_tc500_hours_upper"]].merge(
        old_u2[KEY + ["tc300_hours_upper", "tc500_hours_upper"]], on=KEY,
        validate="one_to_one", how="left"
    )
    for radius in (300, 500):
        if not np.allclose(previous[f"old_tc{radius}_hours_upper"],
                           previous[f"tc{radius}_hours_upper"], rtol=0, atol=1e-5):
            raise ValueError(f"old TC{radius} voyage extension differs from accepted U2 census")

    threshold = u1[KEY + ["voyages"]].copy()
    threshold["published"] = threshold.voyages.ge(5)
    panel = panel.merge(threshold[KEY + ["published"]], on=KEY, how="left", validate="many_to_one")
    quantile_rows = []
    for subset_name, selected in (("all", panel), ("release_voyages_ge5", panel.loc[panel.published])):
        for rule in ("old", "new"):
            for metric, unit in (("tc300_width_pp", "percentage_points"),
                                 ("tc500_width_pp", "percentage_points"),
                                 ("wave_width_m", "m")):
                values = selected[f"{rule}_{metric}"].dropna().to_numpy(dtype=float)
                row = {"population": subset_name, "rule": rule, "metric": metric, "unit": unit,
                       "voyages_total": int(len(selected)), "voyages_with_bound": int(len(values)),
                       "voyages_without_bound": int(len(selected) - len(values))}
                row.update({f"p{int(q*100)}": float(np.quantile(values, q)) if len(values) else np.nan
                            for q in PERCENTILES})
                quantile_rows.append(row)
    pd.DataFrame(quantile_rows).to_csv(out / "01_voyage_error_quantiles.csv", index=False)
    # Treat an unavailable wave interval as worse than every finite width.
    # This gives a valid 90%-of-all-voyages statement when the missing share
    # is below 10%, while leaving higher percentiles undefined if needed.
    guarantee_rows = []
    for population, selected in (("all", panel), ("release_voyages_ge5", panel.loc[panel.published])):
        for rule in ("old", "new"):
            finite = np.sort(selected[f"{rule}_wave_width_m"].dropna().to_numpy(dtype=float))
            total = len(selected)
            row = {"population": population, "rule": rule, "metric": "wave_width_m",
                   "unit": "m", "voyages_total": total, "voyages_with_bound": len(finite),
                   "voyages_without_bound": total - len(finite),
                   "unavailable_treatment": "worst_tail_unbounded"}
            for q in PERCENTILES:
                position = (total - 1) * q
                lo, hi = int(np.floor(position)), int(np.ceil(position))
                row[f"p{int(q*100)}"] = (
                    float(finite[lo] + (position - lo) * (finite[hi] - finite[lo]))
                    if hi < len(finite) else np.nan
                )
            guarantee_rows.append(row)
    pd.DataFrame(guarantee_rows).to_csv(out / "01a_wave_population_guarantee.csv", index=False)

    point_diagnostic = []
    for population, combo_subset, voyage_subset in (
        ("all", groups, panel),
        ("release_voyages_ge5", groups.loc[groups.voyages.ge(5)], panel.loc[panel.published]),
    ):
        for rule in ("old", "new"):
            for radius in (300, 500):
                rate = combo_subset[f"tc{radius}_rate_per_position_hour"] * 100.0
                outside = rate.notna() & (
                    (rate < combo_subset[f"{rule}_tc{radius}_lower_pp"] - 1e-9)
                    | (rate > combo_subset[f"{rule}_tc{radius}_upper_pp"] + 1e-9)
                )
                voyage_point = 100.0 * voyage_subset[f"tc{radius}_positive_snapshot_hours"] / voyage_subset.track_hours.where(voyage_subset.track_hours > 0)
                voyage_outside = voyage_point.notna() & (
                    (voyage_point < voyage_subset[f"{rule}_tc{radius}_lower_pp"] - 1e-9)
                    | (voyage_point > voyage_subset[f"{rule}_tc{radius}_upper_pp"] + 1e-9)
                )
                point_diagnostic.append({"population": population, "rule": rule, "radius_km": radius,
                                         "combinations_outside": int(outside.sum()),
                                         "combinations_with_point": int(rate.notna().sum()),
                                         "voyages_in_outside_combinations": int(combo_subset.loc[outside, "voyages"].sum()),
                                         "voyage_share_in_population": float(combo_subset.loc[outside, "voyages"].sum() / len(voyage_subset)),
                                         "voyages_with_individual_point_outside": int(voyage_outside.sum()),
                                         "individual_point_outside_share_in_population": float(voyage_outside.mean())})
    pd.DataFrame(point_diagnostic).to_csv(out / "02_tc_point_outside_bounds.csv", index=False)

    public_group_columns = [column for column in groups.columns
                            if column not in {"voyages_missing_ship_id"}
                            and not column.endswith("_check")]
    groups[public_group_columns].to_csv(out / "03_all_combinations_exposure_bounds.csv", index=False)
    groups.to_parquet(RUN / "50_all_combinations_exposure_bounds_internal.parquet", index=False)
    release = groups.loc[groups.voyages.ge(5)].copy()
    u4 = pd.read_parquet(PRIOR / "30_u4/30_cluster_bootstrap_intervals.parquet")
    u4_closure = json.loads((PRIOR_REVIEW / "06_u4_closure.json").read_text())
    if u4_closure["seed"] != 20260923 or u4_closure["replicates_per_combination"] != 1000 \
            or not u4.bootstrap_replicates.eq(1000).all():
        raise ValueError("U4 bootstrap provenance differs from T011")
    # U4 records a stable per-combination seed derived from the global seed.
    def expected_seed(row: tuple) -> int:
        key = "|".join("<NA>" if pd.isna(value) else str(value) for value in row)
        return int.from_bytes(hashlib.sha256(f"20260923|{key}".encode()).digest()[:4], "big")
    expected_seeds = [expected_seed(row) for row in u4[KEY].itertuples(index=False, name=None)]
    if not np.array_equal(u4.bootstrap_seed.to_numpy(), np.asarray(expected_seeds)):
        raise ValueError("U4 combination seeds do not derive from the accepted global seed")
    release = release.merge(u4, on=KEY, how="left", validate="one_to_one", suffixes=("", "_u4"))
    if len(release) != 46_981 or int(release.voyages.sum()) != 1_034_602:
        raise ValueError("UG1 release set differs from T011")
    for column in ("voyages", "ships"):
        if not release[column].eq(release[f"{column}_u4"]).all():
            raise ValueError(f"U4 merge differs on {column}")
    for column in ("track_hours", "wave_matched_hours"):
        if not np.allclose(release[column], release[f"{column}_u4"], rtol=0, atol=1e-5):
            raise ValueError(f"U4 merge differs on {column}")
    if not np.allclose(release.voyage_hours, release.total_voyage_hours, rtol=0, atol=1e-5):
        raise ValueError("U4 merge differs on voyage_hours")
    release["sampling_interval_unavailable_fewer_than_3_ships"] = release.ships.lt(3)
    release["single_ship"] = release.ships.eq(1)
    for column in release.columns:
        if column.endswith("_ci95_low") or column.endswith("_ci95_high"):
            release.loc[release.ships.lt(3), column] = np.nan
    selected = [*KEY, "voyages", "ships", "voyage_hours", "track_hours", "position_coverage",
                "unresolved_track_hours", "wave_matched_hours", "wave_unmatched_hours",
                "wave_ge4m_matched_hours", "wave_coverage", "mean_swh_matched_m",
                "mean_swh_m_ci95_low", "mean_swh_m_ci95_high",
                "share_wave_hours_ge4m", "share_wave_hours_ge4m_ci95_low",
                "share_wave_hours_ge4m_ci95_high", "tc300_hours_observed", "tc500_hours_observed",
                "tc300_rate_per_position_hour", "tc300_rate_per_position_hour_ci95_low",
                "tc300_rate_per_position_hour_ci95_high", "tc500_rate_per_position_hour",
                "tc500_rate_per_position_hour_ci95_low", "tc500_rate_per_position_hour_ci95_high",
                "voyages_with_tc300", "voyages_with_tc500", "single_ship",
                "sampling_interval_unavailable_fewer_than_3_ships",
                "new_unbounded_hours", "new_wave_counted_hours", "new_wave_lower_m",
                "new_wave_upper_m", "new_wave_width_m"]
    for radius in (300, 500):
        selected.extend([f"new_tc{radius}_hours_lower", f"new_tc{radius}_bounded_possible_hours",
                         f"new_tc{radius}_hours_upper", f"new_tc{radius}_lower_pp",
                         f"new_tc{radius}_upper_pp", f"new_tc{radius}_width_pp"])
    release[selected].to_csv(out / "04_release_table_draft_voyages_ge5.csv", index=False)
    below = groups.loc[groups.voyages.lt(5)]
    summable = ["voyages", "voyage_hours", "track_hours", "wave_matched_hours",
                "wave_unmatched_hours", "wave_integral_m_hours", "wave_ge4m_matched_hours", "unresolved_track_hours",
                "tc300_hours_observed", "tc500_hours_observed", "new_unbounded_hours",
                "new_wave_counted_hours", "new_tc300_bounded_possible_hours",
                "new_tc500_bounded_possible_hours", "new_tc300_hours_upper", "new_tc500_hours_upper"]
    unrelease = {"combinations": int(len(below))}
    unrelease.update({column: float(below[column].sum()) for column in summable})
    pd.DataFrame([unrelease]).to_csv(out / "05_below_threshold_summary.csv", index=False)

    for column in ["voyages", "voyage_hours", "track_hours", "wave_matched_hours",
                   "wave_unmatched_hours", "wave_ge4m_matched_hours",
                   "unresolved_track_hours", "tc300_hours_observed", "tc500_hours_observed"]:
        _close(float(release[column].sum() + below[column].sum()), float(u1[column].sum()),
               f"release plus omitted {column}")
    if len(release) + len(below) != len(u1):
        raise ValueError("release plus omitted combination count differs from U1")
    if release.loc[release.ships.lt(3), [c for c in selected if c.endswith("_ci95_low") or c.endswith("_ci95_high")]].notna().any().any():
        raise ValueError("bootstrap interval remains on fewer than three ships")

    report = {
        "status": "closed_internal_draft", "census_shards": len(pieces),
        "weather_rows_scanned": total_weather_rows,
        "voyages": len(panel), "combinations": len(groups),
        "unknown_segments": int(panel.unknown_segments.sum()),
        "unknown_hours": float(panel.unknown_hours.sum()),
        "old_unbounded_voyages": int(panel.old_unbounded_segments.gt(0).sum()),
        "new_unbounded_voyages": int(panel.new_unbounded_segments.gt(0).sum()),
        "old_no_position_voyages_port_bounded": int(
            panel.old_unbounded_segments.gt(0).sum() - panel.new_unbounded_segments.gt(0).sum()
        ),
        "old_no_position_voyages_still_unbounded": int(panel.new_unbounded_segments.gt(0).sum()),
        "old_unbounded_segments": int(panel.old_unbounded_segments.sum()),
        "new_unbounded_segments": int(panel.new_unbounded_segments.sum()),
        "no_position_segments_port_bounded": int(
            panel.old_unbounded_segments.sum() - panel.new_unbounded_segments.sum()
        ),
        "former_one_sided_segments_port_completed": int(
            panel.port_added_segments.sum() - panel.old_unbounded_segments.sum()
            + panel.new_unbounded_segments.sum()
        ),
        "old_bounded_segments": int(panel.unknown_segments.sum() - panel.old_unbounded_segments.sum()),
        "new_bounded_segments": int(panel.unknown_segments.sum() - panel.new_unbounded_segments.sum()),
        "old_unbounded_hours": float(panel.old_unbounded_hours.sum()),
        "new_unbounded_hours": float(panel.new_unbounded_hours.sum()),
        "no_position_hours_port_bounded": float(
            panel.old_unbounded_hours.sum() - panel.new_unbounded_hours.sum()
        ),
        "former_one_sided_hours_port_completed": float(
            panel.port_added_hours.sum() - panel.old_unbounded_hours.sum()
            + panel.new_unbounded_hours.sum()
        ),
        "old_wave_counted_hours": float(panel.old_wave_counted_hours.sum()),
        "new_wave_counted_hours": float(panel.new_wave_counted_hours.sum()),
        "old_wave_uncounted_hours": float(panel.old_wave_uncounted_hours.sum()),
        "new_wave_uncounted_hours": float(panel.new_wave_uncounted_hours.sum()),
        "old_wave_no_wet_segments": int(panel.old_wave_no_wet_segments.sum()),
        "new_wave_no_wet_segments": int(panel.new_wave_no_wet_segments.sum()),
        "old_wave_speed_infeasible_segments": int(panel.old_wave_speed_infeasible_segments.sum()),
        "new_wave_speed_infeasible_segments": int(panel.new_wave_speed_infeasible_segments.sum()),
        "release_combinations": int(len(release)), "release_voyages": int(release.voyages.sum()),
        "below_threshold_combinations": int(len(below)),
        "below_threshold_voyages": int(below.voyages.sum()),
        "fewer_than_3_ships_in_release": int(release.ships.lt(3).sum()),
        "old_tc_upper_matches_prior_u2": True,
        "sources": {
            "u1_voyage_panel_sha256": _sha256(PRIOR / "10_u1/10_voyage_exposure_panel.parquet"),
            "u1_combinations_sha256": _sha256(PRIOR / "10_u1/20_od_shiptype_all_combinations.parquet"),
            "u2_full_tc_upper_sha256": _sha256(PRIOR_REVIEW / "19_u2_full_od_shiptype_exposure_bounds.csv"),
            "u4_bootstrap_sha256": _sha256(PRIOR / "30_u4/30_cluster_bootstrap_intervals.parquet"),
        },
    }
    (out / "06_closure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    RUN = arguments.run.resolve()
    if ".runs" not in RUN.parts:
        parser.error("per-voyage results must stay under .runs")
    print(json.dumps(build(arguments.out), indent=2))
