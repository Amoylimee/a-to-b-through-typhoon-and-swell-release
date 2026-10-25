#!/usr/bin/env python3
"""Independently close T013's published rows against T012, U1, and U4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[3]
PRIOR = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release"
T012 = CODING / "outputs/2026-09-24_1648_E002-T012-exposure-unit-review"
T012_RUN = CODING / ".runs/2026-09-24_1500_E002-T012-exposure-units"
KEY = ["origin_city", "dest_city", "shiptype_group"]


def require_close(actual, expected, label: str, atol: float = 1e-5) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True):
        raise ValueError(f"T013 closure failed: {label}")


def verify(out: Path, scratch: Path) -> dict:
    release = pd.read_csv(out / "01_release_table_voyages_ge5.csv")
    below = pd.read_csv(out / "02_below_threshold_summary.csv")
    source = pd.read_parquet(T012_RUN / "50_all_combinations_exposure_bounds_internal.parquet")
    u1 = pd.read_parquet(PRIOR / "10_u1/20_od_shiptype_all_combinations.parquet")
    old = pd.read_csv(T012 / "04_release_table_draft_voyages_ge5.csv")
    boot = pd.read_parquet(scratch / "10_tc_at_least_bootstrap_internal.parquet")
    u4 = pd.read_parquet(PRIOR / "30_u4/30_cluster_bootstrap_intervals.parquet")
    if len(release) != 46_981 or release.duplicated(KEY).any() or not release.voyages.ge(5).all():
        raise ValueError("release grain or UG1 threshold changed")
    if len(release) + int(below.combinations.iloc[0]) != len(u1):
        raise ValueError("publication and below-threshold combination count do not close")
    if int(release.voyages.sum() + below.voyages.iloc[0]) != 1_354_168:
        raise ValueError("publication and below-threshold voyages do not close")
    if any(column.startswith(("new_", "old_")) or "rate_per_position_hour" in column
           for column in release.columns):
        raise ValueError("legacy rule or positioned-hour TC field leaked into release")
    if any(column in release.columns for column in ("candidate_id", "mmsi", "vessel_uid_clean", "lat", "lon")):
        raise ValueError("internal identifier or position leaked into release")

    reference = release[KEY].merge(source, on=KEY, how="left", validate="one_to_one")
    if reference.voyages.isna().any():
        raise ValueError("T012 combination source missing")
    joined = release.merge(reference, on=KEY, validate="one_to_one", suffixes=("", "_source"))
    for column in ("voyages", "ships", "voyage_hours", "track_hours", "wave_matched_hours"):
        require_close(joined[column], joined[f"{column}_source"], f"T012 {column}")
    require_close(joined.unbounded_hours, joined.new_unbounded_hours, "unbounded hours")
    require_close(joined.wave_unknown_counted_hours, joined.new_wave_counted_hours, "wave unknown counted hours")
    for public, internal in (("mean_swh_min_possible_m", "new_wave_lower_m"),
                             ("mean_swh_max_possible_m", "new_wave_upper_m"),
                             ("mean_swh_possible_range_m", "new_wave_width_m")):
        require_close(joined[public], joined[internal], public)
    for radius in (300, 500):
        prefix = f"tc{radius}"
        require_close(joined[f"{prefix}_hours_at_least"], joined[f"{prefix}_positive_snapshot_hours"],
                      f"{prefix} observed hours")
        require_close(joined[f"{prefix}_bounded_possible_hours"],
                      joined[f"new_{prefix}_bounded_possible_hours"], f"{prefix} possible hours")
        require_close(joined[f"{prefix}_hours_at_most"], joined[f"new_{prefix}_hours_upper"],
                      f"{prefix} maximum hours")
        require_close(joined[f"{prefix}_at_least_pct"],
                      100 * joined[f"{prefix}_hours_at_least"] / joined.voyage_hours,
                      f"{prefix} minimum share")
        require_close(joined[f"{prefix}_at_most_pct"],
                      100 * joined[f"{prefix}_hours_at_most"] / joined.voyage_hours,
                      f"{prefix} maximum share")
        if (joined[f"{prefix}_at_least_pct"] > joined[f"{prefix}_at_most_pct"] + 1e-9).any():
            raise ValueError(f"{prefix} least share exceeds most")
        if (joined[f"{prefix}_at_most_pct"] > 100 + 1e-8).any():
            raise ValueError(f"{prefix} most share exceeds 100 percent")

    old_comparison = release.merge(old, on=KEY, how="left", validate="one_to_one", suffixes=("", "_old"))
    for column in ("mean_swh_matched_m", "mean_swh_m_ci95_low", "mean_swh_m_ci95_high",
                   "share_wave_hours_ge4m", "share_wave_hours_ge4m_ci95_low",
                   "share_wave_hours_ge4m_ci95_high"):
        require_close(old_comparison[column], old_comparison[f"{column}_old"],
                      f"unchanged U4 wave interval {column}")
    boot_joined = release.merge(boot, on=KEY, how="left", validate="one_to_one", suffixes=("", "_boot"))
    if boot_joined.bootstrap_seed.isna().any() or not boot_joined.bootstrap_replicates.eq(1000).all():
        raise ValueError("T013 bootstrap results incomplete")
    for radius in (300, 500):
        name = f"tc{radius}_at_least_pct"
        require_close(boot_joined[name], boot_joined[f"{name}_estimate"], f"{name} bootstrap estimate")
        for end in ("low", "high"):
            column = f"{name}_ci95_{end}"
            require_close(boot_joined.loc[boot_joined.ships.ge(3), column],
                          boot_joined.loc[boot_joined.ships.ge(3), f"{column}_boot"],
                          f"{name} bootstrap {end}")
    u4_check = boot.merge(u4[KEY + ["bootstrap_seed", "bootstrap_replicates"]], on=KEY,
                          validate="one_to_one", suffixes=("", "_u4"))
    if not u4_check.bootstrap_seed.eq(u4_check.bootstrap_seed_u4).all() or \
            not u4_check.bootstrap_replicates.eq(u4_check.bootstrap_replicates_u4).all():
        raise ValueError("T013 bootstrap method provenance differs from U4")
    # Recalculate a deterministic mix of ordinary and large-vessel groups
    # from the voyage panel with one vectorized draw, independent of the
    # production script's 100-draw chunks.
    sample_keys = pd.concat([release.head(5)[KEY], release.nlargest(5, "ships")[KEY]],
                            ignore_index=True).drop_duplicates()
    sample_panel = pd.read_parquet(PRIOR / "10_u1/10_voyage_exposure_panel.parquet",
                                   columns=[*KEY, "vessel_uid_clean", "voyage_hours",
                                            "tc300_positive_snapshot_hours",
                                            "tc500_positive_snapshot_hours"])
    sample_panel = sample_panel.merge(sample_keys, on=KEY, how="inner", validate="many_to_one")
    sample_boot = boot.merge(sample_keys, on=KEY, how="inner", validate="one_to_one")
    sampled = 0
    for key, group in sample_panel.groupby(KEY, dropna=False, observed=True, sort=True):
        row = sample_boot.loc[(sample_boot[KEY] == pd.Series(key, index=KEY)).all(axis=1)]
        if len(row) != 1:
            raise ValueError("independent bootstrap sample key missing")
        clusters = group.groupby("vessel_uid_clean", sort=True, observed=True)[
            ["tc300_positive_snapshot_hours", "tc500_positive_snapshot_hours", "voyage_hours"]
        ].sum().to_numpy(dtype=float)
        rng = np.random.default_rng(int(row.bootstrap_seed.iloc[0]))
        indexes = rng.integers(0, len(clusters), size=(1000, len(clusters)))
        sums = clusters[indexes].sum(axis=1)
        sample_limits = np.quantile(100 * sums[:, :2] / sums[:, 2, None],
                                    [0.025, 0.975], axis=0, method="linear")
        for index, radius in enumerate((300, 500)):
            name = f"tc{radius}_at_least_pct"
            require_close(sample_limits[:, index],
                          row[[f"{name}_ci95_low", f"{name}_ci95_high"]].iloc[0].to_numpy(dtype=float),
                          f"independent bootstrap {name}", atol=1e-12)
        sampled += 1
    ci = [column for column in release if column.endswith(("_ci95_low", "_ci95_high"))]
    if release.loc[release.ships.lt(3), ci].notna().any().any():
        raise ValueError("bootstrap CI present for fewer than three ships")
    if release.loc[release.ships.ge(3), [f"tc{r}_at_least_pct_ci95_{e}"
                                            for r in (300, 500) for e in ("low", "high")]].isna().any().any():
        raise ValueError("TC lower-share bootstrap CI missing for eligible ships")

    no_data = release.wave_mean_missing_reason.eq("no_matched_or_reachable_wave_hours")
    only_bounds = release.wave_mean_missing_reason.eq("no_matched_wave_hours_reachable_bounds_available")
    if int(no_data.sum()) != 515 or int(only_bounds.sum()) != 408:
        raise ValueError("wave missingness categories changed")
    if not release.loc[no_data, ["wave_matched_hours", "wave_unknown_counted_hours"]].eq(0).all().all():
        raise ValueError("no-wave-data category has wave hours")
    wave_estimates = ["mean_swh_matched_m", "mean_swh_m_ci95_low", "mean_swh_m_ci95_high",
                      "share_wave_hours_ge4m", "share_wave_hours_ge4m_ci95_low",
                      "share_wave_hours_ge4m_ci95_high", "mean_swh_min_possible_m",
                      "mean_swh_max_possible_m", "mean_swh_possible_range_m"]
    if release.loc[no_data, wave_estimates].notna().any().any():
        raise ValueError("no-wave-data category has wave estimate")
    if release.loc[only_bounds, ["mean_swh_min_possible_m", "mean_swh_max_possible_m"]].isna().any().any():
        raise ValueError("bounds-only wave category lacks bound")

    full_hours = {"voyages": "voyages", "voyage_hours": "voyage_hours",
                  "track_hours": "track_hours", "unresolved_track_hours": "unresolved_track_hours",
                  "wave_matched_hours": "wave_matched_hours", "wave_unmatched_hours": "wave_unmatched_hours",
                  "wave_ge4m_matched_hours": "wave_ge4m_matched_hours",
                  "unbounded_hours": "new_unbounded_hours",
                  "wave_unknown_counted_hours": "new_wave_counted_hours",
                  "tc300_hours_at_least": "tc300_positive_snapshot_hours",
                  "tc500_hours_at_least": "tc500_positive_snapshot_hours",
                  "tc300_bounded_possible_hours": "new_tc300_bounded_possible_hours",
                  "tc500_bounded_possible_hours": "new_tc500_bounded_possible_hours",
                  "tc300_hours_at_most": "new_tc300_hours_upper",
                  "tc500_hours_at_most": "new_tc500_hours_upper"}
    for public, internal in full_hours.items():
        expected = source[internal].sum() if public not in u1.columns else u1[public].sum()
        require_close(release[public].sum() + below[public].iloc[0], expected,
                      f"published plus below threshold {public}", atol=1e-4)
    for path in out.glob("*.csv"):
        header = pd.read_csv(path, nrows=0).columns
        if any(column in header for column in ("candidate_id", "mmsi", "vessel_uid_clean", "lat", "lon")):
            raise ValueError(f"restricted field in {path.name}")
    result = {"status": "pass", "release_combinations": len(release),
              "release_voyages": int(release.voyages.sum()),
              "below_threshold_combinations": int(below.combinations.iloc[0]),
              "below_threshold_voyages": int(below.voyages.iloc[0]),
              "recalculated_full_population_hour_columns": len(full_hours),
              "new_tc_bootstrap_combinations": len(boot),
              "bootstrap_combinations_independently_redrawn": sampled,
              "few_ship_interval_blank_combinations": int(release.ships.lt(3).sum()),
              "no_wave_data_combinations": int(no_data.sum()),
              "no_observed_wave_but_bounds_combinations": int(only_bounds.sum()),
              "review_csv_files_checked_for_restricted_columns": len(list(out.glob("*.csv")))}
    (out / "07_independent_closure.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.out, args.scratch), indent=2))
