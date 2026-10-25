#!/usr/bin/env python3
"""Independently close T014's release, clipping, calibration and GFW units."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[3]
U1 = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release/10_u1"
U4 = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release/30_u4/30_cluster_bootstrap_intervals.parquet"
T013 = CODING / "outputs/2026-09-24_2048_E002-T013-full-voyage-share-review/01_release_table_voyages_ge5.csv"
T012_VOYAGES = CODING / ".runs/2026-09-24_1500_E002-T012-exposure-units/50_voyage_exposure_intervals_internal.parquet"
KEY = ["origin_city", "dest_city", "shiptype_group"]
METRICS = ["wave_integral_m_hours", "wave_matched_hours", "new_wave_min_integral",
           "new_wave_max_integral", "new_wave_counted_hours"]


def same(a, b, label: str, atol: float = 1e-5) -> None:
    if not np.allclose(a, b, rtol=0, atol=atol, equal_nan=True):
        raise ValueError(f"T014 closure mismatch: {label}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_wave_bootstrap(voyages: pd.DataFrame, release: pd.DataFrame,
                               wave_internal: pd.DataFrame) -> int:
    eligible = release.loc[release.ships.ge(3) & release.mean_swh_full_voyage_m.notna()]
    keys = pd.concat([eligible.head(5)[KEY], eligible.nlargest(5, "ships")[KEY]],
                     ignore_index=True).drop_duplicates()
    sample = voyages.merge(keys, on=KEY, how="inner", validate="many_to_one")
    row_source = wave_internal.merge(keys, on=KEY, how="inner", validate="one_to_one")
    count = 0
    for key, block in sample.groupby(KEY, dropna=False, observed=True, sort=True):
        key = tuple(key)
        row = row_source.loc[(row_source[KEY] == pd.Series(key, index=KEY)).all(axis=1)]
        if len(row) != 1:
            raise ValueError("bootstrap sample key missing")
        clusters = block.groupby("vessel_uid_clean", sort=True, observed=True)[METRICS].sum().to_numpy(float)
        rng = np.random.default_rng(int(row.bootstrap_seed_wave.iloc[0]))
        draws = rng.integers(0, len(clusters), size=(1000, len(clusters)))
        totals = clusters[draws].sum(axis=1)
        matched = totals[:, 1]
        denominator = matched + totals[:, 4]
        observed = np.divide(totals[:, 0], matched,
                             out=np.full(1000, np.nan), where=matched > 0)
        lower = np.divide(totals[:, 0] + totals[:, 2], denominator,
                          out=np.full(1000, np.nan), where=denominator > 0)
        upper = np.divide(totals[:, 0] + totals[:, 3], denominator,
                          out=np.full(1000, np.nan), where=denominator > 0)
        clipped = np.clip(observed, lower, upper)
        valid = clipped[np.isfinite(clipped)]
        expected = np.quantile(valid, [.025, .975], method="linear")
        same(expected, row[["mean_swh_full_voyage_ci95_low",
                            "mean_swh_full_voyage_ci95_high"]].iloc[0].to_numpy(float),
             "independent paired wave bootstrap", atol=1e-10)
        count += 1
    return count


def verify(out: Path, run: Path, internal: Path, gfw_run: Path) -> dict:
    release = pd.read_csv(out / "01_release_table_voyages_ge5.csv")
    below = pd.read_csv(out / "02_below_threshold_summary.csv")
    source = pd.read_parquet(run / "50_all_combinations_exposure_bounds_internal.parquet")
    voyages = pd.read_parquet(run / "50_voyage_exposure_intervals_internal.parquet",
                              columns=["candidate_id", *KEY, "vessel_uid_clean", *METRICS])
    old_columns = ["old_unbounded_segments", "old_unbounded_hours", "old_wave_counted_segments",
                   "old_wave_counted_hours", "old_wave_no_wet_segments",
                   "old_wave_speed_infeasible_segments", "old_wave_min_integral",
                   "old_wave_max_integral", "old_tc300_possible_hours", "old_tc500_possible_hours"]
    old_prior = pd.read_parquet(T012_VOYAGES, columns=["candidate_id", *old_columns])
    old_new = pd.read_parquet(run / "50_voyage_exposure_intervals_internal.parquet",
                              columns=["candidate_id", *old_columns])
    old_join = old_prior.merge(old_new, on="candidate_id", validate="one_to_one",
                               suffixes=("_prior", "_current"))
    if len(old_join) != 1_354_168:
        raise ValueError("zero-tolerance old-rule voyage grain changed")
    for column in old_columns:
        same(old_join[column + "_prior"], old_join[column + "_current"],
             "old rule " + column)
    wave_internal = pd.read_parquet(internal / "wave_clipping_by_combination.parquet")
    clip_report = json.loads((out / "08_wave_clipping_summary.json").read_text())
    u1 = pd.read_parquet(U1 / "20_od_shiptype_all_combinations.parquet")
    u4 = pd.read_parquet(U4, columns=[*KEY, "bootstrap_seed"])
    calibration = pd.read_csv(out / "03_port_tolerance_reachability.csv")
    conditional = pd.read_csv(out / "03b_port_tolerance_without_conflict_voyages.csv")
    gfw = json.loads((out / "04_gfw_exposure_summary.json").read_text())
    gfw_months = pd.read_csv(out / "04_gfw_exposure_by_month.csv")
    gfw_position = json.loads((gfw_run / "summary.json").read_text())
    if len(release) != 46_981 or release.duplicated(KEY).any() or not release.voyages.ge(5).all():
        raise ValueError("release grain/threshold changed")
    if len(release) + int(below.combinations.iloc[0]) != len(u1):
        raise ValueError("release/omitted combination count does not close")
    if int(release.voyages.sum() + below.voyages.iloc[0]) != 1_354_168:
        raise ValueError("release/omitted voyage count does not close")
    restricted = {"candidate_id", "mmsi", "vessel_uid_clean", "lat", "lon"}
    for csv in out.glob("*.csv"):
        if restricted.intersection(pd.read_csv(csv, nrows=0).columns):
            raise ValueError(f"restricted field in {csv.name}")
    if any(name.startswith(("new_", "old_")) or "rate_per_position_hour" in name
           for name in release.columns):
        raise ValueError("comparison/internal field leaked to release")
    joined = release.merge(source, on=KEY, validate="one_to_one", suffixes=("", "_source"))
    for field in ("voyages", "ships", "voyage_hours", "track_hours", "wave_matched_hours"):
        same(joined[field], joined[f"{field}_source"], field)
    for public, source_name in (("unbounded_hours", "new_unbounded_hours"),
                                ("wave_unknown_counted_hours", "new_wave_counted_hours"),
                                ("mean_swh_min_possible_m", "new_wave_lower_m"),
                                ("mean_swh_max_possible_m", "new_wave_upper_m"),
                                ("mean_swh_possible_range_m", "new_wave_width_m")):
        same(joined[public], joined[source_name], public)
    for radius in (300, 500):
        name = f"tc{radius}"
        same(joined[f"{name}_hours_at_least"], joined[f"{name}_positive_snapshot_hours"], name + " least hours")
        same(joined[f"{name}_hours_at_most"], joined[f"new_{name}_hours_upper"], name + " most hours")
        same(joined[f"{name}_bounded_possible_hours"], joined[f"new_{name}_bounded_possible_hours"], name + " possible hours")
        same(joined[f"{name}_hours_at_most"],
             joined[f"{name}_hours_at_least"] + joined[f"{name}_bounded_possible_hours"] + joined.unbounded_hours,
             name + " hour decomposition")
        same(joined[f"{name}_at_least_pct"], 100 * joined[f"{name}_hours_at_least"] / joined.voyage_hours,
             name + " least share")
        same(joined[f"{name}_at_most_pct"], 100 * joined[f"{name}_hours_at_most"] / joined.voyage_hours,
             name + " most share")
        if (joined[f"{name}_at_least_pct"] > joined[f"{name}_at_most_pct"] + 1e-9).any():
            raise ValueError(name + " least exceeds most")
    wave = release.merge(wave_internal, on=KEY, validate="one_to_one")
    same(wave.mean_swh_full_voyage_m_x, wave.mean_swh_full_voyage_m_y, "clipped point")
    for end in ("low", "high"):
        field = f"mean_swh_full_voyage_ci95_{end}"
        same(wave[field + "_x"], wave[field + "_y"], field)
    expected_point = wave.mean_swh_observed_before_clip_m.clip(
        lower=wave.mean_swh_min_possible_m, upper=wave.mean_swh_max_possible_m)
    same(wave.mean_swh_full_voyage_m_x, expected_point, "clipping formula")
    clip_delta = wave_internal.wave_clip_change_m.dropna().abs()
    changed = clip_delta.gt(1e-10)
    if int(changed.sum()) != clip_report["changed_combinations"] or len(clip_delta) != clip_report["with_observed_mean"]:
        raise ValueError("clipping before/after counts do not close")
    same(float(clip_delta[changed].quantile(.5)), clip_report["absolute_change_p50_changed_m"],
         "clipping median", atol=1e-12)
    same(float(clip_delta[changed].quantile(.9)), clip_report["absolute_change_p90_changed_m"],
         "clipping p90", atol=1e-12)
    ci = [name for name in release if name.endswith(("_ci95_low", "_ci95_high"))]
    if release.loc[release.ships.lt(3), ci].notna().any().any():
        raise ValueError("sampling interval on fewer than three ships")
    eligible_tc = release.loc[release.ships.ge(3)]
    tc_ci = [f"tc{radius}_at_least_pct_ci95_{end}"
             for radius in (300, 500) for end in ("low", "high")]
    if eligible_tc[tc_ci].isna().any().any():
        raise ValueError("eligible TC sampling interval is blank")
    for radius in (300, 500):
        name = f"tc{radius}_at_least_pct"
        if (eligible_tc[name].lt(eligible_tc[name + "_ci95_low"] - 1e-8)
                | eligible_tc[name].gt(eligible_tc[name + "_ci95_high"] + 1e-8)).any():
            raise ValueError("TC sampling interval excludes estimate")
    eligible_wave = release.loc[release.ships.ge(3) & release.mean_swh_full_voyage_m.notna()]
    if eligible_wave[["mean_swh_full_voyage_ci95_low", "mean_swh_full_voyage_ci95_high"]].isna().any().any():
        raise ValueError("eligible wave sampling interval is blank")
    if (eligible_wave.mean_swh_full_voyage_m.lt(eligible_wave.mean_swh_full_voyage_ci95_low - 1e-8)
            | eligible_wave.mean_swh_full_voyage_m.gt(eligible_wave.mean_swh_full_voyage_ci95_high + 1e-8)).any():
        raise ValueError("wave sampling interval excludes clipped estimate")
    seeds = wave_internal.merge(u4, on=KEY, validate="one_to_one")
    if not seeds.bootstrap_seed_wave.eq(seeds.bootstrap_seed).all():
        raise ValueError("wave bootstrap seed differs from U4")
    tc_prior = pd.read_csv(T013)
    tc = release.merge(tc_prior[KEY + [f"tc{r}_at_least_pct_ci95_{e}"
                                        for r in (300, 500) for e in ("low", "high")]],
                       on=KEY, validate="one_to_one", suffixes=("", "_t013"))
    for r in (300, 500):
        for e in ("low", "high"):
            name = f"tc{r}_at_least_pct_ci95_{e}"
            same(tc[name], tc[name + "_t013"], name)
    sampled = independent_wave_bootstrap(voyages, release, wave_internal)
    if len(calibration) != 12 or len(conditional) != 12:
        raise ValueError("port tolerance grid incomplete")
    if not (calibration.gfw_hours.eq(969).all() and conditional.gfw_hours.eq(920).all()):
        raise ValueError("port tolerance denominators changed")
    port_hours = pd.read_parquet(run.parent / "port_tolerance_hour_audit.parquet")
    if len(port_hours) != 969:
        raise ValueError("GFW port calibration source population changed")
    conflict_ids = set(port_hours.loc[~port_hours.reachable_50km_2h, "candidate_id"])
    if len(conflict_ids) != 5 or int(port_hours.candidate_id.isin(conflict_ids).sum()) != 49:
        raise ValueError("five-voyage port-conflict exception changed")
    conflict_voyages = pd.read_parquet(run / "50_voyage_exposure_intervals_internal.parquet",
                                       columns=["candidate_id", "unknown_hours", "new_unbounded_hours",
                                                "new_wave_counted_hours"])
    conflict_voyages = conflict_voyages.loc[conflict_voyages.candidate_id.isin(conflict_ids)]
    if len(conflict_voyages) != 5:
        raise ValueError("port-conflict voyages missing from census")
    same(conflict_voyages.new_unbounded_hours, conflict_voyages.unknown_hours,
         "port-conflict voyages stay unbounded")
    if conflict_voyages.new_wave_counted_hours.gt(1e-8).any():
        raise ValueError("port-conflict voyage gained wave-bound hours")
    hour = pd.to_datetime(port_hours.hour_utc, utc=True)
    departure = pd.to_datetime(port_hours.departure_time, utc=True)
    arrival = pd.to_datetime(port_hours.arrival_time, utc=True)
    keep = ~port_hours.candidate_id.isin(conflict_ids)
    for original in calibration.itertuples(index=False):
        space, time = float(original.space_km), float(original.time_h)
        first = departure - pd.Timedelta(hours=time)
        last = arrival + pd.Timedelta(hours=time)
        origin_travel = pd.to_timedelta(np.maximum(0, port_hours.origin_distance_km - space)
                                        / (25 * 1.852), unit="h")
        destination_travel = pd.to_timedelta(np.maximum(0, port_hours.dest_distance_km - space)
                                             / (25 * 1.852), unit="h")
        earliest = pd.concat([hour, first, first + origin_travel], axis=1).max(axis=1)
        latest = pd.concat([hour + pd.Timedelta(hours=1), last,
                            last - destination_travel], axis=1).min(axis=1)
        reachable = earliest.le(latest)
        if int(reachable.sum()) != int(original.reachable_hours):
            raise ValueError("original 969-hour port tolerance rate differs")
        conditional_row = conditional.loc[conditional.space_km.eq(space)
                                          & conditional.time_h.eq(time)]
        if len(conditional_row) != 1 or int(reachable[keep].sum()) != int(conditional_row.reachable_hours.iloc[0]):
            raise ValueError("conditional 920-hour port tolerance rate differs")
    if int(calibration.loc[calibration.space_km.eq(50) & calibration.time_h.eq(2),
                           "reachable_hours"].iloc[0]) != 945:
        raise ValueError("full GFW maximum tolerance reachability changed")
    if int(conditional.loc[conditional.space_km.eq(0) & conditional.time_h.eq(1),
                           "reachable_hours"].iloc[0]) != 911:
        raise ValueError("conditional selected reachability changed")
    if gfw["hours"] != 2_293_663 or gfw_position["comparison_class"]["gfw_supported_interpolation"] != gfw["hours"]:
        raise ValueError("GFW held-out exposure population changed")
    if len(gfw_months) != 12 or int(gfw_months.hours.sum()) != gfw["hours"]:
        raise ValueError("GFW month inventory does not close")
    if int(gfw_months.swh_absolute_difference_m_valid.sum()) != gfw["swh_absolute_difference_m"]["valid_hours"]:
        raise ValueError("GFW paired-wave hours do not close")
    for radius in (300, 500):
        if int(gfw_months[f"tc{radius}_flip_hours"].sum()) != gfw[f"tc{radius}"]["flip_hours"]:
            raise ValueError("GFW TC flips do not close")
    result = {"status": "pass", "release_combinations": len(release),
              "release_voyages": int(release.voyages.sum()),
              "below_threshold_combinations": int(below.combinations.iloc[0]),
              "below_threshold_voyages": int(below.voyages.iloc[0]),
              "wave_points_outside_possible_range": int((release.mean_swh_full_voyage_m.lt(release.mean_swh_min_possible_m - 1e-9)
                                                         | release.mean_swh_full_voyage_m.gt(release.mean_swh_max_possible_m + 1e-9)).sum()),
              "wave_bootstrap_groups_independently_redrawn": sampled,
              "old_rule_voyages_independently_closed": len(old_join),
              "port_calibration_original_hours": 969,
              "port_calibration_conditional_hours": 920,
              "port_conflict_voyages_still_unbounded": len(conflict_voyages),
              "port_conflict_unbounded_hours": float(conflict_voyages.new_unbounded_hours.sum()),
              "gfw_exposure_hours": gfw["hours"],
              "sources": {"t013_release_sha256": sha256(T013),
                          "t014_voyages_sha256": sha256(run / "50_voyage_exposure_intervals_internal.parquet"),
                          "t014_all_combinations_sha256": sha256(run / "50_all_combinations_exposure_bounds_internal.parquet"),
                          "gfw_derived_manifest_sha256": sha256(gfw_run / "climate_derived_source_manifest.json"),
                          "release_sha256": sha256(out / "01_release_table_voyages_ge5.csv")}}
    (out / "07_independent_closure.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--gfw-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.out, args.run, args.internal, args.gfw_run), indent=2))
