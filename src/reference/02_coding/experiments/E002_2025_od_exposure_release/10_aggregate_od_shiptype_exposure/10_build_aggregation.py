#!/usr/bin/env python3
"""Build the full 2025 directed OD × ship-type exposure panel (U1)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CODING_ROOT = Path(__file__).resolve().parents[3]
TRACK_ROOT = CODING_ROOT / "data/processed/products/companion_tracks/2025"
WEATHER_ROOT = CODING_ROOT / "data/processed/products/voyage_weather_facts/2025"
EXPECTED = {
    "candidate_count": 1_354_168,
    "voyage_hours": 173_282_514.955,
    "track_hours": 156_537_533.788,
    "wave_matched_hours": 155_842_974.788,
}
HOUR_TOLERANCE = 5e-4
GROUP_COLUMNS = ["origin_city", "dest_city", "shiptype_group"]
WEATHER_SUMS = [
    "voyage_hours",
    "track_hours",
    "wave_matched_hours",
    "wave_integral_m_hours",
    "wave_ge4m_matched_hours",
    "tc500_positive_snapshot_hours",
    "tc300_positive_snapshot_hours",
    "unresolved_track_hours",
    "wave_unmatched_hours",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weighted numerators and denominators without imputing unknowns."""
    work = panel.copy()
    for column in WEATHER_SUMS:
        work[column] = pd.to_numeric(work[column], errors="raise")
    work["_has_tc500"] = work["tc500_positive_snapshot_hours"].gt(0).astype("int64")
    work["_has_tc300"] = work["tc300_positive_snapshot_hours"].gt(0).astype("int64")
    aggregations: dict[str, tuple[str, str]] = {
        "voyages": ("candidate_id", "nunique"),
        "ships": ("vessel_uid_clean", "nunique"),
        "voyages_missing_ship_id": ("vessel_uid_clean", lambda x: int(x.isna().sum())),
        **{column: (column, "sum") for column in WEATHER_SUMS},
        "voyages_with_tc300": ("_has_tc300", "sum"),
        "voyages_with_tc500": ("_has_tc500", "sum"),
    }
    grouped = (
        work.groupby(GROUP_COLUMNS, dropna=False, observed=True, sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    grouped["position_coverage"] = np.divide(
        grouped.track_hours,
        grouped.voyage_hours,
        out=np.full(len(grouped), np.nan),
        where=grouped.voyage_hours.to_numpy() > 0,
    )
    grouped["wave_coverage"] = np.divide(
        grouped.wave_matched_hours,
        grouped.voyage_hours,
        out=np.full(len(grouped), np.nan),
        where=grouped.voyage_hours.to_numpy() > 0,
    )
    grouped["mean_swh_matched_m"] = np.divide(
        grouped.wave_integral_m_hours,
        grouped.wave_matched_hours,
        out=np.full(len(grouped), np.nan),
        where=grouped.wave_matched_hours.to_numpy() > 0,
    )
    grouped["share_wave_hours_ge4m"] = np.divide(
        grouped.wave_ge4m_matched_hours,
        grouped.wave_matched_hours,
        out=np.full(len(grouped), np.nan),
        where=grouped.wave_matched_hours.to_numpy() > 0,
    )
    for radius, source in (
        (300, "tc300_positive_snapshot_hours"),
        (500, "tc500_positive_snapshot_hours"),
    ):
        grouped[f"tc{radius}_hours_observed"] = grouped[source]
        grouped[f"tc{radius}_rate_per_position_hour"] = np.divide(
            grouped[source],
            grouped.track_hours,
            out=np.full(len(grouped), np.nan),
            where=grouped.track_hours.to_numpy() > 0,
        )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    hash_report = json.loads(args.hash_report.read_text(encoding="utf-8"))
    if not hash_report.get("ok"):
        raise SystemExit("formal adoption-manifest hash verification did not pass")

    shards = sorted((TRACK_ROOT / "run/shards").glob("shard_*/candidate_summary.parquet"))
    if len(shards) != 253:
        raise SystemExit(f"expected 253 candidate_summary shards, found {len(shards)}")
    candidates = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    expected_candidates = {
        "candidate_id",
        "vessel_uid_clean",
        "origin_city",
        "dest_city",
        "shiptype_group",
    }
    missing = sorted(expected_candidates.difference(candidates.columns))
    if missing:
        raise SystemExit(f"candidate_summary lacks required fields: {missing}")
    if candidates.candidate_id.isna().any() or candidates.candidate_id.duplicated().any():
        raise SystemExit("candidate_summary candidate_id is null or duplicated")

    annual_weather = WEATHER_ROOT / "voyage_weather_facts_2025.parquet"
    weather_columns = ["candidate_id", *WEATHER_SUMS]
    weather = pq.read_table(annual_weather, columns=weather_columns).to_pandas()
    if weather.candidate_id.isna().any() or weather.candidate_id.duplicated().any():
        raise SystemExit("annual weather candidate_id is null or duplicated")
    panel = candidates.merge(
        weather,
        on="candidate_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = panel[panel["_merge"].ne("both")]
    if not unmatched.empty:
        raise SystemExit(
            "formal products do not close by candidate_id: "
            f"left_only={int(unmatched['_merge'].eq('left_only').sum())}, "
            f"right_only={int(unmatched['_merge'].eq('right_only').sum())}"
        )
    panel.drop(columns="_merge", inplace=True)
    if len(panel) != EXPECTED["candidate_count"]:
        raise SystemExit(f"candidate panel has {len(panel):,} rows")

    grouped = aggregate_panel(panel)
    totals = {
        "candidate_count": int(panel.candidate_id.nunique()),
        "combination_count": int(len(grouped)),
        "voyage_hours": float(panel.voyage_hours.sum()),
        "track_hours": float(panel.track_hours.sum()),
        "wave_matched_hours": float(panel.wave_matched_hours.sum()),
        "wave_integral_m_hours": float(panel.wave_integral_m_hours.sum()),
        "wave_ge4m_matched_hours": float(panel.wave_ge4m_matched_hours.sum()),
        "unresolved_track_hours": float(panel.unresolved_track_hours.sum()),
        "wave_unmatched_hours": float(panel.wave_unmatched_hours.sum()),
        "tc300_positive_snapshot_hours": float(
            panel.tc300_positive_snapshot_hours.sum()
        ),
        "tc500_positive_snapshot_hours": float(
            panel.tc500_positive_snapshot_hours.sum()
        ),
        "missing_origin_city": int(panel.origin_city.isna().sum()),
        "missing_dest_city": int(panel.dest_city.isna().sum()),
        "missing_shiptype_group": int(panel.shiptype_group.isna().sum()),
        "missing_vessel_uid_clean": int(panel.vessel_uid_clean.isna().sum()),
    }
    checks = {
        "candidate_count": totals["candidate_count"] == EXPECTED["candidate_count"],
        "voyage_hours": abs(totals["voyage_hours"] - EXPECTED["voyage_hours"])
        <= HOUR_TOLERANCE,
        "track_hours": abs(totals["track_hours"] - EXPECTED["track_hours"])
        <= HOUR_TOLERANCE,
        "wave_matched_hours": abs(
            totals["wave_matched_hours"] - EXPECTED["wave_matched_hours"]
        )
        <= HOUR_TOLERANCE,
        "group_voyages_close": int(grouped.voyages.sum()) == totals["candidate_count"],
    }
    if not all(checks.values()):
        raise SystemExit(f"U1 closure failed: {checks}; totals={totals}")

    # Independently recompute the weighted mean on a deterministic set of groups.
    rng = np.random.default_rng(args.seed)
    sample_count = min(30, len(grouped))
    selected = rng.choice(len(grouped), size=sample_count, replace=False)
    independent_rows = []
    for index in selected:
        row = grouped.iloc[int(index)]
        mask = pd.Series(True, index=panel.index)
        for column in GROUP_COLUMNS:
            value = row[column]
            mask &= panel[column].isna() if pd.isna(value) else panel[column].eq(value)
        subset = panel.loc[mask]
        matched_h = float(subset.wave_matched_hours.sum())
        expected_mean = (
            float(subset.wave_integral_m_hours.sum() / matched_h)
            if matched_h > 0
            else np.nan
        )
        actual_mean = float(row.mean_swh_matched_m) if pd.notna(row.mean_swh_matched_m) else np.nan
        if not (
            (np.isnan(expected_mean) and np.isnan(actual_mean))
            or np.isclose(expected_mean, actual_mean, atol=1e-12, rtol=1e-12)
        ):
            raise SystemExit(f"independent weighted-mean check failed for {tuple(row[c] for c in GROUP_COLUMNS)}")
        independent_rows.append({
            **{column: row[column] for column in GROUP_COLUMNS},
            "voyages": int(len(subset)),
            "wave_matched_hours": matched_h,
            "recomputed_mean_swh_matched_m": expected_mean,
        })

    args.out.mkdir(parents=True, exist_ok=True)
    panel_path = args.out / "10_voyage_exposure_panel.parquet"
    group_path = args.out / "20_od_shiptype_all_combinations.parquet"
    check_path = args.out / "30_u1_closure.json"
    panel.to_parquet(panel_path, index=False)
    grouped.to_parquet(group_path, index=False)
    report = {
        "ok": True,
        "seed_for_independent_group_check": args.seed,
        "track_adoption_manifest_sha256": sha256_file(TRACK_ROOT / "adoption_manifest.json"),
        "weather_adoption_manifest_sha256": sha256_file(WEATHER_ROOT / "adoption_manifest.json"),
        "inputs": {
            "candidate_summary_files": len(shards),
            "annual_weather_summary": str(annual_weather),
            "candidate_id_join": "one_to_one; full outer; no unmatched IDs",
        },
        "expected_totals": EXPECTED,
        "actual_totals": totals,
        "checks": checks,
        "independent_group_mean_checks": independent_rows,
        "outputs": {
            "voyage_panel": str(panel_path),
            "combinations": str(group_path),
        },
        "combination_count_by_candidate_n": {
            str(n): {
                "at_least_n": int(grouped.voyages.ge(n).sum()),
                "strictly_more_than_n": int(grouped.voyages.gt(n).sum()),
            }
            for n in (5, 10, 20, 30)
        },
    }
    check_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
