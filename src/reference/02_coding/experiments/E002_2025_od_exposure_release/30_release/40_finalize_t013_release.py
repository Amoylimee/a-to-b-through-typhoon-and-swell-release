#!/usr/bin/env python3
"""Finalize the T013 release draft from closed T012 and U1/U4 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[3]
PRIOR = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release"
T012 = CODING / "outputs/2026-09-24_1648_E002-T012-exposure-unit-review"
T012_RUN = CODING / ".runs/2026-09-24_1500_E002-T012-exposure-units"
KEY = ["origin_city", "dest_city", "shiptype_group"]
SHIP = "vessel_uid_clean"
SEED = 20260923
B = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(values: tuple[object, ...]) -> int:
    text = "|".join("<NA>" if pd.isna(value) else str(value) for value in values)
    return int.from_bytes(hashlib.sha256(f"{SEED}|{text}".encode()).digest()[:4], "big")


def close(actual: np.ndarray | float, expected: np.ndarray | float, label: str,
          atol: float = 1e-5) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True):
        raise ValueError(f"{label} does not close")


def bootstrap_at_least(panel: pd.DataFrame, old_release: pd.DataFrame,
                       u4: pd.DataFrame, scratch: Path) -> pd.DataFrame:
    """Use U4's ship clusters, per-key seed, B, chunks, and percentile method."""
    selected = panel.merge(old_release[KEY], on=KEY, how="inner", validate="many_to_one")
    if len(selected) != 1_034_602 or selected[SHIP].isna().any():
        raise ValueError("release voyage set or ship clusters changed")
    metrics = ["tc300_positive_snapshot_hours", "tc500_positive_snapshot_hours", "voyage_hours"]
    clusters = selected.groupby([*KEY, SHIP], dropna=False, observed=True, sort=True)[metrics].sum().reset_index()
    rows = []
    for group_key, group in clusters.groupby(KEY, dropna=False, observed=True, sort=True):
        group_key = tuple(group_key)
        values = group[metrics].to_numpy(dtype=np.float64)
        if (values[:, 2] <= 0).any():
            raise ValueError(f"nonpositive voyage duration in {group_key}")
        seed = stable_seed(group_key)
        rng = np.random.default_rng(seed)
        boot = np.empty((B, 2), dtype=np.float64)
        for start in range(0, B, 100):
            stop = min(B, start + 100)
            indexes = rng.integers(0, len(values), size=(stop - start, len(values)))
            sums = values[indexes].sum(axis=1)
            boot[start:stop] = 100.0 * sums[:, :2] / sums[:, 2, None]
        limits = np.quantile(boot, [0.025, 0.975], axis=0, method="linear")
        totals = values.sum(axis=0)
        rows.append({**dict(zip(KEY, group_key)), "ships": len(values),
                     "bootstrap_seed": seed, "bootstrap_replicates": B,
                     "tc300_at_least_pct_estimate": 100.0 * totals[0] / totals[2],
                     "tc300_at_least_pct_ci95_low": float(limits[0, 0]),
                     "tc300_at_least_pct_ci95_high": float(limits[1, 0]),
                     "tc500_at_least_pct_estimate": 100.0 * totals[1] / totals[2],
                     "tc500_at_least_pct_ci95_low": float(limits[0, 1]),
                     "tc500_at_least_pct_ci95_high": float(limits[1, 1])})
        if len(rows) % 5_000 == 0:
            print(f"T013 bootstrap: {len(rows):,}/{len(old_release):,} combinations", flush=True)
    results = pd.DataFrame(rows)
    if len(results) != 46_981 or results.duplicated(KEY).any():
        raise ValueError("T013 bootstrap combination grain changed")
    provenance = results.merge(u4[KEY + ["ships", "bootstrap_seed", "bootstrap_replicates"]],
                               on=KEY, how="left", validate="one_to_one", suffixes=("", "_u4"))
    for column in ("ships", "bootstrap_seed", "bootstrap_replicates"):
        if not provenance[column].eq(provenance[f"{column}_u4"]).all():
            raise ValueError(f"T013 bootstrap differs from U4 on {column}")
    results.to_parquet(scratch / "10_tc_at_least_bootstrap_internal.parquet", index=False)
    return results


def build_release(old: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    merged = old.merge(boot.drop(columns=["ships", "bootstrap_seed", "bootstrap_replicates"]),
                       on=KEY, how="left", validate="one_to_one")
    if len(merged) != len(old) or merged.tc300_at_least_pct_estimate.isna().any():
        raise ValueError("T013 bootstrap merge incomplete")
    columns = [*KEY, "voyages", "ships", "voyage_hours", "track_hours",
               "position_coverage", "unresolved_track_hours", "wave_matched_hours",
               "wave_unmatched_hours", "wave_ge4m_matched_hours", "wave_coverage",
               "mean_swh_matched_m", "mean_swh_m_ci95_low", "mean_swh_m_ci95_high",
               "share_wave_hours_ge4m", "share_wave_hours_ge4m_ci95_low",
               "share_wave_hours_ge4m_ci95_high", "voyages_with_tc300",
               "voyages_with_tc500", "single_ship",
               "sampling_interval_unavailable_fewer_than_3_ships"]
    release = merged[columns].copy()
    release["unbounded_hours"] = merged.new_unbounded_hours
    release["wave_unknown_counted_hours"] = merged.new_wave_counted_hours
    release["mean_swh_min_possible_m"] = merged.new_wave_lower_m
    release["mean_swh_max_possible_m"] = merged.new_wave_upper_m
    release["mean_swh_possible_range_m"] = merged.new_wave_width_m
    no_observed_wave = merged.wave_matched_hours.eq(0)
    no_wave_anywhere = no_observed_wave & merged.new_wave_counted_hours.eq(0)
    release["wave_mean_missing_reason"] = pd.Series(pd.NA, index=release.index, dtype="string")
    release.loc[no_observed_wave & ~no_wave_anywhere, "wave_mean_missing_reason"] = "no_matched_wave_hours_reachable_bounds_available"
    release.loc[no_wave_anywhere, "wave_mean_missing_reason"] = "no_matched_or_reachable_wave_hours"
    for radius in (300, 500):
        name = f"tc{radius}"
        release[f"{name}_hours_at_least"] = merged[f"new_{name}_hours_lower"]
        release[f"{name}_bounded_possible_hours"] = merged[f"new_{name}_bounded_possible_hours"]
        release[f"{name}_hours_at_most"] = merged[f"new_{name}_hours_upper"]
        release[f"{name}_at_least_pct"] = 100.0 * release[f"{name}_hours_at_least"] / release.voyage_hours
        release[f"{name}_at_most_pct"] = 100.0 * release[f"{name}_hours_at_most"] / release.voyage_hours
        release[f"{name}_at_least_pct_ci95_low"] = merged[f"{name}_at_least_pct_ci95_low"]
        release[f"{name}_at_least_pct_ci95_high"] = merged[f"{name}_at_least_pct_ci95_high"]
        close(release[f"{name}_hours_at_least"], merged[f"{name}_hours_observed"], f"{name} observed hours")
        close(release[f"{name}_at_least_pct"], merged[f"new_{name}_lower_pp"], f"{name} lower percentage")
        close(release[f"{name}_at_most_pct"], merged[f"new_{name}_upper_pp"], f"{name} upper percentage")
        close(release[f"{name}_at_least_pct"], merged[f"{name}_at_least_pct_estimate"], f"{name} bootstrap point")
        close(release[f"{name}_hours_at_most"],
              release[f"{name}_hours_at_least"] + release[f"{name}_bounded_possible_hours"] + release.unbounded_hours,
              f"{name} hour decomposition")
        if (release[f"{name}_at_least_pct"] > release[f"{name}_at_most_pct"] + 1e-9).any():
            raise ValueError(f"{name} least exceeds most")
        if (release[f"{name}_at_most_pct"] > 100 + 1e-8).any():
            raise ValueError(f"{name} most exceeds 100 percent")
        for end in ("low", "high"):
            release.loc[release.ships.lt(3), f"{name}_at_least_pct_ci95_{end}"] = np.nan
    if int(no_wave_anywhere.sum()) != 515:
        raise ValueError("no-wave-data combination count differs from T013")
    if int((no_observed_wave & ~no_wave_anywhere).sum()) != 408:
        raise ValueError("wave-only-bounds combination count changed")
    if release.loc[no_wave_anywhere, ["mean_swh_matched_m", "mean_swh_m_ci95_low",
                                       "mean_swh_m_ci95_high", "share_wave_hours_ge4m",
                                       "share_wave_hours_ge4m_ci95_low", "share_wave_hours_ge4m_ci95_high",
                                       "mean_swh_min_possible_m", "mean_swh_max_possible_m",
                                       "mean_swh_possible_range_m"]].notna().any().any():
        raise ValueError("no-wave-data combinations have nonempty wave estimates")
    if any("new_" in name or "rate_per_position_hour" in name for name in release.columns):
        raise ValueError("internal or positioned-hour TC field leaked to release")
    return release


def build(out: Path, scratch: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=True)
    old_path = T012 / "04_release_table_draft_voyages_ge5.csv"
    panel_path = PRIOR / "10_u1/10_voyage_exposure_panel.parquet"
    u4_path = PRIOR / "30_u4/30_cluster_bootstrap_intervals.parquet"
    old = pd.read_csv(old_path)
    u4 = pd.read_parquet(u4_path)
    panel = pd.read_parquet(panel_path, columns=["candidate_id", *KEY, SHIP,
                                                 "tc300_positive_snapshot_hours",
                                                 "tc500_positive_snapshot_hours", "voyage_hours"])
    if len(old) != 46_981 or int(old.voyages.sum()) != 1_034_602 or len(panel) != 1_354_168:
        raise ValueError("T012/U1 source populations changed")
    if old.duplicated(KEY).any() or u4.duplicated(KEY).any():
        raise ValueError("combination key not unique")
    boot = bootstrap_at_least(panel, old, u4, scratch)
    release = build_release(old, boot)
    release_path = out / "01_release_table_voyages_ge5.csv"
    release.to_csv(release_path, index=False)
    below = pd.read_csv(T012 / "05_below_threshold_summary.csv")
    below.rename(columns={"tc300_hours_observed": "tc300_hours_at_least",
                          "tc500_hours_observed": "tc500_hours_at_least",
                          "new_unbounded_hours": "unbounded_hours",
                          "new_wave_counted_hours": "wave_unknown_counted_hours",
                          "new_tc300_bounded_possible_hours": "tc300_bounded_possible_hours",
                          "new_tc500_bounded_possible_hours": "tc500_bounded_possible_hours",
                          "new_tc300_hours_upper": "tc300_hours_at_most",
                          "new_tc500_hours_upper": "tc500_hours_at_most"}, inplace=True)
    if any(name.startswith("new_") for name in below.columns):
        raise ValueError("internal prefix remains in below-threshold summary")
    below.to_csv(out / "02_below_threshold_summary.csv", index=False)
    quantiles = pd.read_csv(T012 / "01_voyage_error_quantiles.csv")
    quantiles.loc[quantiles.rule.eq("new")].drop(columns="rule").to_csv(
        out / "04_voyage_error_quantiles.csv", index=False)
    guarantee = pd.read_csv(T012 / "01a_wave_population_guarantee.csv")
    guarantee.loc[guarantee.rule.eq("new")].drop(columns="rule").to_csv(
        out / "05_wave_population_guarantee.csv", index=False)
    report = {"status": "closed_internal_draft_pending_review", "seed": SEED,
              "bootstrap_replicates_per_combination": B,
              "bootstrap_method": "whole-ship cluster resampling within directed OD x ship type; U4 stable per-key seed and linear percentile quantiles",
              "release_combinations": len(release), "release_voyages": int(release.voyages.sum()),
              "below_threshold_combinations": int(below.combinations.iloc[0]),
              "below_threshold_voyages": int(below.voyages.iloc[0]),
              "fewer_than_3_ships": int(release.ships.lt(3).sum()),
              "no_wave_data_combinations": int(release.wave_mean_missing_reason.eq("no_matched_or_reachable_wave_hours").sum()),
              "no_observed_wave_but_bounds_combinations": int(release.wave_mean_missing_reason.eq("no_matched_wave_hours_reachable_bounds_available").sum()),
              "sources": {"t012_release_sha256": sha256(old_path),
                          "u1_panel_sha256": sha256(panel_path),
                          "u4_intervals_sha256": sha256(u4_path),
                          "t012_closure_sha256": sha256(T012 / "06_closure.json"),
                          "t012_all_combinations_sha256": sha256(T012_RUN / "50_all_combinations_exposure_bounds_internal.parquet")},
              "internal_bootstrap_sha256": sha256(scratch / "10_tc_at_least_bootstrap_internal.parquet"),
              "release_sha256": sha256(release_path)}
    (out / "06_closure.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.out, arguments.scratch), indent=2, ensure_ascii=False))
