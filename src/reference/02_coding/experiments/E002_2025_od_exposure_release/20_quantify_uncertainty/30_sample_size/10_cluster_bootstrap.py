#!/usr/bin/env python3
"""Cluster-bootstrap U4 metrics by ship within each directed OD × ship type."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260923
B = 1_000
GROUP_COLUMNS = ["origin_city", "dest_city", "shiptype_group"]
CLUSTER = "vessel_uid_clean"
METRIC_COLUMNS = [
    "wave_integral_m_hours",
    "wave_matched_hours",
    "wave_ge4m_matched_hours",
    "tc300_positive_snapshot_hours",
    "tc500_positive_snapshot_hours",
    "track_hours",
]
THRESHOLDS = (5, 10, 20, 30)


def _stable_seed(values: tuple[object, ...]) -> int:
    text = "|".join("<NA>" if pd.isna(x) else str(x) for x in values)
    return int.from_bytes(hashlib.sha256(f"{SEED}|{text}".encode()).digest()[:4], "big")


def _derived_metrics(sums: np.ndarray) -> np.ndarray:
    """Return (mean Hs, >=4 m share, TC300 rate, TC500 rate)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack((
            sums[:, 0] / sums[:, 1],
            sums[:, 2] / sums[:, 1],
            sums[:, 3] / sums[:, 5],
            sums[:, 4] / sums[:, 5],
        ))


def bootstrap_one(group: pd.DataFrame, group_key: tuple[object, ...]) -> dict[str, object]:
    if group[CLUSTER].isna().any():
        raise ValueError(f"missing ship cluster for combination {group_key}")
    clusters = (
        group.groupby(CLUSTER, sort=True, observed=True)[METRIC_COLUMNS]
        .sum()
        .to_numpy(dtype=np.float64)
    )
    if len(clusters) == 0:
        raise ValueError(f"combination {group_key} has no ship clusters")
    rng = np.random.default_rng(_stable_seed(group_key))
    boot_sums = np.empty((B, len(METRIC_COLUMNS)), dtype=np.float64)
    chunk_size = 100
    for start in range(0, B, chunk_size):
        stop = min(B, start + chunk_size)
        indexes = rng.integers(0, len(clusters), size=(stop - start, len(clusters)))
        boot_sums[start:stop] = clusters[indexes].sum(axis=1)
    boot_metrics = _derived_metrics(boot_sums)
    point = _derived_metrics(clusters.sum(axis=0, keepdims=True))[0]
    names = ("mean_swh_m", "share_wave_hours_ge4m", "tc300_rate_per_position_hour", "tc500_rate_per_position_hour")
    result: dict[str, object] = {
        **dict(zip(GROUP_COLUMNS, group_key)),
        "voyages": int(group.candidate_id.nunique()),
        "ships": int(group[CLUSTER].nunique()),
        "total_voyage_hours": float(group.voyage_hours.sum()),
        "track_hours": float(group.track_hours.sum()),
        "wave_matched_hours": float(group.wave_matched_hours.sum()),
        "bootstrap_replicates": B,
        "bootstrap_seed": _stable_seed(group_key),
    }
    for column, name, estimate in zip(range(4), names, point):
        finite = boot_metrics[:, column][np.isfinite(boot_metrics[:, column])]
        if len(finite):
            lo, hi = np.quantile(finite, [0.025, 0.975], method="linear")
            relative_width = (hi - lo) / abs(estimate) if estimate != 0 else np.nan
            result[f"{name}_estimate"] = float(estimate)
            result[f"{name}_ci95_low"] = float(lo)
            result[f"{name}_ci95_high"] = float(hi)
            result[f"{name}_ci95_width"] = float(hi - lo)
            result[f"{name}_relative_width"] = float(relative_width) if np.isfinite(relative_width) else np.nan
            result[f"{name}_valid_bootstrap_replicates"] = int(len(finite))
        else:
            result[f"{name}_estimate"] = float(estimate) if np.isfinite(estimate) else np.nan
            result[f"{name}_ci95_low"] = np.nan
            result[f"{name}_ci95_high"] = np.nan
            result[f"{name}_ci95_width"] = np.nan
            result[f"{name}_relative_width"] = np.nan
            result[f"{name}_valid_bootstrap_replicates"] = 0
    return result


def candidate_threshold_table(all_groups: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in THRESHOLDS:
        for comparator, mask in (("at_least_n", all_groups.voyages.ge(n)), ("strictly_more_than_n", all_groups.voyages.gt(n))):
            chosen = all_groups.loc[mask]
            selected_keys = chosen[GROUP_COLUMNS].merge(intervals[GROUP_COLUMNS], on=GROUP_COLUMNS, how="inner").drop_duplicates()
            selected_intervals = intervals.merge(selected_keys, on=GROUP_COLUMNS, how="inner")
            row: dict[str, object] = {
                "n": n,
                "comparator": comparator,
                "combinations": int(len(chosen)),
                "combinations_with_bootstrap_intervals": int(len(selected_keys)),
                "voyages": int(chosen.voyages.sum()),
                "ship_combination_memberships": int(chosen.ships.sum()),
                "voyage_hours": float(chosen.voyage_hours.sum()),
                "position_hours": float(chosen.track_hours.sum()),
                "wave_matched_hours": float(chosen.wave_matched_hours.sum()),
            }
            for metric in (
                "mean_swh_m", "share_wave_hours_ge4m",
                "tc300_rate_per_position_hour", "tc500_rate_per_position_hour",
            ):
                col = f"{metric}_relative_width"
                values = selected_intervals[col].dropna()
                row[f"{metric}_relative_width_median"] = float(values.median()) if len(values) else np.nan
                row[f"{metric}_relative_width_p90"] = float(values.quantile(.9)) if len(values) else np.nan
                row[f"{metric}_relative_width_p95"] = float(values.quantile(.95)) if len(values) else np.nan
                row[f"{metric}_relative_width_n"] = int(len(values))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--all-combinations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel)
    all_groups = pd.read_parquet(args.all_combinations)
    group_sizes = panel.groupby(GROUP_COLUMNS, dropna=False, observed=True).size()
    eligible_keys = group_sizes.loc[group_sizes.ge(min(THRESHOLDS))].reset_index()[GROUP_COLUMNS]
    eligible = panel.merge(eligible_keys, on=GROUP_COLUMNS, how="inner", validate="many_to_one")
    results = []
    for group_key, group in eligible.groupby(GROUP_COLUMNS, dropna=False, sort=True, observed=True):
        results.append(bootstrap_one(group, tuple(group_key)))
        if len(results) % 1_000 == 0:
            print(f"bootstrapped {len(results):,} combinations", flush=True)
    intervals = pd.DataFrame(results)
    threshold_table = candidate_threshold_table(all_groups, intervals)
    interval_path = args.out / "30_cluster_bootstrap_intervals.parquet"
    threshold_path = args.out / "31_candidate_n_thresholds.csv"
    intervals.to_parquet(interval_path, index=False)
    threshold_table.to_csv(threshold_path, index=False)
    report = {
        "ok": True,
        "method": "resample vessel_uid_clean whole-ship clusters with replacement within directed OD x ship type",
        "seed": SEED,
        "replicates_per_combination": B,
        "eligible_minimum_voyages": min(THRESHOLDS),
        "interval_combination_count": int(len(intervals)),
        "candidate_thresholds": list(THRESHOLDS),
        "threshold_semantics": "both at_least_n and strictly_more_than_n retained pending UG1",
        "publish_threshold_applied": False,
        "intervals_parquet": str(interval_path),
        "candidate_thresholds_csv": str(threshold_path),
    }
    (args.out / "32_u4_closure.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
