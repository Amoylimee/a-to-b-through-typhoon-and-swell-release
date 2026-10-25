#!/usr/bin/env python3
"""T014 P1: clip whole-voyage wave means and bootstrap the clipped estimand."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["origin_city", "dest_city", "shiptype_group"]
SHIP = "vessel_uid_clean"
METRICS = ["wave_integral_m_hours", "wave_matched_hours", "new_wave_min_integral",
           "new_wave_max_integral", "new_wave_counted_hours"]
SEED = 20260923
B = 1000


def stable_seed(key: tuple[object, ...]) -> int:
    value = "|".join("<NA>" if pd.isna(part) else str(part) for part in key)
    return int.from_bytes(hashlib.sha256(f"{SEED}|{value}".encode()).digest()[:4], "big")


def estimates(sums: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One or many paired ship-resample totals; all means use the same draw."""
    observed = np.divide(sums[..., 0], sums[..., 1], out=np.full_like(sums[..., 0], np.nan),
                         where=sums[..., 1] > 0)
    denominator = sums[..., 1] + sums[..., 4]
    lower = np.divide(sums[..., 0] + sums[..., 2], denominator,
                      out=np.full_like(observed, np.nan), where=denominator > 0)
    upper = np.divide(sums[..., 0] + sums[..., 3], denominator,
                      out=np.full_like(observed, np.nan), where=denominator > 0)
    if np.any(lower > upper + 1e-8):
        raise ValueError("wave lower exceeds upper")
    return observed, lower, upper


def build(voyages_path: Path, release_path: Path, out: Path, internal: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    internal.mkdir(parents=True, exist_ok=True)
    release = pd.read_csv(release_path)
    if release.duplicated(KEY).any():
        raise ValueError("duplicate release combination")
    voyages = pd.read_parquet(voyages_path, columns=["candidate_id", *KEY, SHIP, *METRICS])
    selected = voyages.merge(release[KEY], on=KEY, how="inner", validate="many_to_one")
    if len(selected) != int(release.voyages.sum()):
        raise ValueError("release voyage population changed")
    if selected[SHIP].isna().any():
        raise ValueError("missing ship cluster")
    clusters = selected.groupby([*KEY, SHIP], dropna=False, observed=True,
                                sort=True)[METRICS].sum().reset_index()
    rows = []
    for index, (key, block) in enumerate(clusters.groupby(KEY, dropna=False, observed=True, sort=True), 1):
        key = tuple(key)
        values = block[METRICS].to_numpy(dtype=np.float64)
        total = values.sum(axis=0)
        observed, lower, upper = estimates(total)
        point = np.clip(observed, lower, upper) if np.isfinite(observed) else np.nan
        row = {**dict(zip(KEY, key)), "ships_bootstrap": len(values),
               "bootstrap_seed_wave": stable_seed(key), "bootstrap_replicates_wave": B,
               "mean_swh_observed_before_clip_m": float(observed),
               "mean_swh_full_voyage_m": float(point),
               "mean_swh_min_possible_m_check": float(lower),
               "mean_swh_max_possible_m_check": float(upper),
               "wave_clip_change_m": float(point - observed) if np.isfinite(point) else np.nan,
               "wave_bootstrap_valid_replicates": 0,
               "mean_swh_full_voyage_ci95_low": np.nan,
               "mean_swh_full_voyage_ci95_high": np.nan}
        if len(values) >= 3 and np.isfinite(point):
            rng = np.random.default_rng(stable_seed(key))
            boot = np.empty(B, dtype=np.float64)
            for start in range(0, B, 100):
                stop = min(B, start + 100)
                indexes = rng.integers(0, len(values), size=(stop - start, len(values)))
                sums = values[indexes].sum(axis=1)
                obs, lo, hi = estimates(sums)
                boot[start:stop] = np.clip(obs, lo, hi)
            valid = boot[np.isfinite(boot)]
            row["wave_bootstrap_valid_replicates"] = len(valid)
            if len(valid):
                limits = np.quantile(valid, [.025, .975], method="linear")
                row["mean_swh_full_voyage_ci95_low"] = float(limits[0])
                row["mean_swh_full_voyage_ci95_high"] = float(limits[1])
        rows.append(row)
        if index % 5000 == 0:
            print(f"P1 wave bootstrap {index:,}/{len(release):,}", flush=True)
    calculated = pd.DataFrame(rows)
    if len(calculated) != len(release) or calculated.duplicated(KEY).any():
        raise ValueError("bootstrap combination population changed")
    calculated.to_parquet(internal / "wave_clipping_by_combination.parquet", index=False)
    result = release.merge(calculated, on=KEY, how="left", validate="one_to_one")
    if not np.allclose(result.mean_swh_min_possible_m, result.mean_swh_min_possible_m_check,
                       rtol=0, atol=1e-5, equal_nan=True):
        raise ValueError("wave lower does not close against per-voyage source")
    if not np.allclose(result.mean_swh_max_possible_m, result.mean_swh_max_possible_m_check,
                       rtol=0, atol=1e-5, equal_nan=True):
        raise ValueError("wave upper does not close against per-voyage source")
    if not np.allclose(result.mean_swh_matched_m, result.mean_swh_observed_before_clip_m,
                       rtol=0, atol=1e-5, equal_nan=True):
        raise ValueError("observed wave mean does not close against per-voyage source")
    result = result.drop(columns=["mean_swh_min_possible_m_check", "mean_swh_max_possible_m_check",
                                  "mean_swh_observed_before_clip_m", "ships_bootstrap",
                                  "bootstrap_seed_wave", "bootstrap_replicates_wave",
                                  "wave_bootstrap_valid_replicates", "wave_clip_change_m",
                                  "mean_swh_matched_m", "mean_swh_m_ci95_low", "mean_swh_m_ci95_high"])
    wave_front = ["mean_swh_full_voyage_m", "mean_swh_min_possible_m",
                  "mean_swh_max_possible_m", "mean_swh_possible_range_m",
                  "mean_swh_full_voyage_ci95_low", "mean_swh_full_voyage_ci95_high",
                  "wave_mean_missing_reason"]
    columns = [name for name in result if name not in wave_front]
    at = columns.index("wave_coverage") + 1
    result = result[columns[:at] + wave_front + columns[at:]]
    result.to_csv(out / "01_release_table_voyages_ge5.csv", index=False)
    delta = calculated.wave_clip_change_m.dropna().abs()
    changed = delta.gt(1e-10)
    boot_eligible = calculated.loc[calculated.ships_bootstrap.ge(3)
                                   & calculated.mean_swh_full_voyage_m.notna()]
    summary = {"status": "p1_wave_clipping_closed", "combinations": len(result),
               "release_voyages": int(result.voyages.sum()), "with_observed_mean": len(delta),
               "changed_combinations": int(changed.sum()),
               "changed_pct_of_observed": 100 * float(changed.mean()),
               "absolute_change_p50_all_m": float(delta.quantile(.5)),
               "absolute_change_p90_all_m": float(delta.quantile(.9)),
               "absolute_change_p50_changed_m": float(delta[changed].quantile(.5)),
               "absolute_change_p90_changed_m": float(delta[changed].quantile(.9)),
               "absolute_change_max_m": float(delta.max()),
               "fewer_than_3_ships": int(result.ships.lt(3).sum()),
               "wave_interval_nonnull": int(result.mean_swh_full_voyage_ci95_low.notna().sum()),
               "eligible_wave_bootstrap_groups": len(boot_eligible),
               "groups_with_fewer_than_1000_valid_draws": int(boot_eligible.wave_bootstrap_valid_replicates.lt(B).sum()),
               "minimum_valid_draws_in_eligible_group": int(boot_eligible.wave_bootstrap_valid_replicates.min()),
               "wave_point_outside_possible_range": int((result.mean_swh_full_voyage_m.lt(result.mean_swh_min_possible_m - 1e-8)
                                                           | result.mean_swh_full_voyage_m.gt(result.mean_swh_max_possible_m + 1e-8)).sum()),
               "source_voyages": str(voyages_path), "source_release": str(release_path)}
    (out / "08_wave_clipping_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voyages", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--internal", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.voyages, args.release, args.out, args.internal), indent=2))
