#!/usr/bin/env python3
"""Trace E030 bridge ODs through stage 30/20 and compare old/new qtrim."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CODING_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parent
BUILDER_PATH = PACKAGE_ROOT / "10_build_adjacent_commercial_episode_voyages.py"
SPEC = importlib.util.spec_from_file_location("adjacent_episode_builder", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load stage builder: {BUILDER_PATH}")
p0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p0)

YEARS = (2021, 2022, 2023, 2024, 2025)
BUFFERED_YEARS = (2021, 2022, 2023, 2024)
OLD_THRESHOLD_PATHS = {
    2021: CODING_ROOT
    / ".runs/2026-08-23_E030-one-leg-main-pipeline/voyages/full_calendar/"
    "manifests/qtrim_thresholds_2021.json",
    2022: CODING_ROOT
    / ".runs/2026-08-23_E030-one-leg-main-pipeline/voyages/full_cross_year/"
    "run_09ebcbfcda88/manifests/qtrim_thresholds_2022.json",
    2023: CODING_ROOT
    / ".runs/2026-08-23_E030-one-leg-main-pipeline/voyages/full_cross_year/"
    "run_09ebcbfcda88/manifests/qtrim_thresholds_2023.json",
    2024: CODING_ROOT
    / ".runs/2026-08-23_E030-one-leg-main-pipeline/voyages/full_cross_year/"
    "run_09ebcbfcda88/manifests/qtrim_thresholds_2024.json",
    2025: CODING_ROOT
    / ".runs/2026-08-23_E030-one-leg-main-pipeline/voyages/full_calendar/"
    "manifests/qtrim_thresholds_2025.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--od-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_parts(directory: Path, columns: list[str] | None = None) -> pd.DataFrame:
    paths = sorted(directory.glob("part-*.feather"))
    if len(paths) != 16:
        raise ValueError(f"expected 16 parts under {directory}; found {len(paths)}")
    return pd.concat(
        [pd.read_feather(path, columns=columns) for path in paths],
        ignore_index=True,
    )


def episode_manifest_path(run_root: Path, year: int) -> Path:
    if year in BUFFERED_YEARS:
        return run_root / f"intermediate/{year}/city_episodes/output_manifest.json"
    return (
        run_root
        / f"annual_sources/intermediate/{year}/city_episodes/output_manifest.json"
    )


def manifest_output(manifest: dict[str, Any], key: str) -> Path:
    return p0.resolve_manifest_record_path(manifest["outputs"][key])


def december_profile(frame: pd.DataFrame) -> dict[str, Any]:
    departure = pd.to_datetime(frame["departure_time"], utc=True)
    selected = departure.loc[departure.dt.month.eq(12)]
    counts = selected.dt.day.value_counts().reindex(range(1, 32), fill_value=0)
    baseline = float(counts.loc[1:20].mean())
    percentages = {
        str(day): (float(counts.loc[day] / baseline * 100.0) if baseline else None)
        for day in range(28, 32)
    }
    values = [percentages[str(day)] for day in range(28, 32)]
    monotone_collapse = bool(
        all(value is not None for value in values)
        and values[0] > values[1] > values[2] > values[3]
        and values[3] < 25.0
    )
    return {
        "december_1_20_daily_mean": baseline,
        "december_28_31_percent": percentages,
        "monotone_collapse": monotone_collapse,
    }


def map_bridge_legs_to_candidates(
    bridge: pd.DataFrame,
    legs: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    bridge_legs = bridge.merge(
        legs[["mmsi", "trip_id", "leg_sequence"]],
        on=["mmsi", "trip_id"],
        how="left",
        validate="one_to_one",
    )
    bridge_legs["candidate_id"] = pd.Series(pd.NA, index=bridge_legs.index, dtype="string")
    candidate_groups = {
        str(mmsi): group.sort_values("outbound_leg_sequence", kind="stable")
        for mmsi, group in candidates.groupby("mmsi", sort=False)
    }
    for mmsi, indices in bridge_legs.loc[
        bridge_legs["leg_sequence"].notna()
    ].groupby("mmsi", sort=False).groups.items():
        group = candidate_groups.get(str(mmsi))
        if group is None or group.empty:
            continue
        outbound = group["outbound_leg_sequence"].to_numpy(np.int64)
        inbound = group["inbound_leg_sequence"].to_numpy(np.int64)
        if len(group) > 1 and np.any(outbound[1:] <= inbound[:-1]):
            raise ValueError(f"candidate leg intervals overlap for MMSI {mmsi}")
        candidate_ids = group["candidate_id"].astype(str).to_numpy()
        sequences = bridge_legs.loc[indices, "leg_sequence"].to_numpy(np.int64)
        positions = np.searchsorted(outbound, sequences, side="right") - 1
        valid = positions >= 0
        valid &= np.where(valid, sequences <= inbound[np.maximum(positions, 0)], False)
        mapped = np.full(len(sequences), None, dtype=object)
        mapped[valid] = candidate_ids[positions[valid]]
        bridge_legs.loc[indices, "candidate_id"] = mapped
    if bridge_legs["candidate_id"].dropna().duplicated().any():
        raise ValueError("one candidate represents more than one bridge OD")
    return bridge_legs


def qtrim_keep(frame: pd.DataFrame, lower: float, upper: float) -> pd.Series:
    eligible = (
        frame["temporal_scope_eligible"].astype(bool)
        & np.isfinite(frame["normalized_time_ratio"])
        & frame["d_yardstick"].gt(0)
    )
    trimmed = eligible & (
        frame["normalized_time_ratio"].lt(lower)
        | frame["normalized_time_ratio"].gt(upper)
    )
    return frame["temporal_scope_eligible"].astype(bool) & ~trimmed


def reason_counts(lineage: pd.DataFrame, stage: str) -> dict[str, int]:
    values = lineage.loc[lineage["first_drop_stage"].eq(stage), "first_drop_reason"]
    return {str(key): int(value) for key, value in values.value_counts().items()}


def validate(run_root: Path, od_root: Path, output_dir: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    od_root = od_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assembly = p0.od_input_assembly.verify_od_input_root(od_root)
    if int(assembly["total_bridge_voyages"]) != 80_489:
        raise ValueError("validation requires the frozen 80,489-bridge OD assembly")
    bridge_registry = pd.read_feather(Path(assembly["bridge_registry"]["path"]))
    bridge_registry["mmsi"] = bridge_registry["mmsi"].astype(str)

    lineage_frames = []
    annual_records = []
    threshold_records = []
    flip_frames = []
    namespace_rows = 0
    namespace_double_prefix = 0
    final_ids = []
    raw_ids = []

    for year in YEARS:
        manifest = load_json(episode_manifest_path(run_root, year))
        events = pd.read_feather(
            manifest_output(manifest, "endpoint_events"),
            columns=["mmsi", "trip_id", "event_type", "visit_id"],
        )
        namespace_rows += len(events)
        namespace_double_prefix += int(
            events["visit_id"]
            .astype("string")
            .str.match(r"^\d{4}:\d{4}:", na=False)
            .sum()
        )
        event_counts = events.groupby("trip_id", observed=True).agg(
            event_rows=("event_type", "size"),
            event_types=("event_type", "nunique"),
        )
        episodes, legs, _ = p0.load_candidate_inputs(manifest)
        del episodes
        candidates = load_parts(
            run_root / f"intermediate/{year}/candidate_occurrences/candidate_passages",
            columns=[
                "candidate_id",
                "mmsi",
                "departure_time",
                "arrival_time",
                "origin_episode_sequence",
                "dest_episode_sequence",
                "n_intermediate_city_episodes",
                "outbound_leg_sequence",
                "inbound_leg_sequence",
            ],
        )
        if candidates["candidate_id"].duplicated().any():
            raise ValueError(f"{year} raw candidate IDs duplicate")
        if candidates["dest_episode_sequence"].sub(
            candidates["origin_episode_sequence"]
        ).ne(1).any() or candidates["n_intermediate_city_episodes"].ne(0).any():
            raise ValueError(f"{year} raw candidates violate adjacency")
        raw_ids.append(candidates["candidate_id"].astype(str))

        c2 = load_parts(
            run_root / f"intermediate/{year}/filters/c2_retained",
            columns=["candidate_id"],
        )
        c3 = load_parts(
            run_root / f"intermediate/{year}/filters/c3_decisions_{year}",
            columns=[
                "candidate_id",
                "retain_after_route_circuity",
                "route_circuity_decision_basis",
                "exclude_extreme_duration_speed_final",
                "extreme_duration_speed_decision_reason",
                "retain_after_c3_before_same_node_backstop",
                "exclude_same_node_route_backstop",
                "same_node_route_backstop_decision_reason",
                "retain_after_c3_final",
                "c3_final_decision_reason",
            ],
        )
        pretrim = load_parts(
            run_root / f"intermediate/{year}/filters/c4_pretrim_{year}",
            columns=[
                "candidate_id",
                "departure_time",
                "arrival_time",
                "normalized_time_ratio",
                "temporal_scope_eligible",
                "d_yardstick",
            ],
        )
        final = load_parts(
            run_root / f"adjacent_commercial_episode_voyages_{year}",
            columns=[
                "candidate_id",
                "departure_time",
                "arrival_time",
                "origin_episode_sequence",
                "dest_episode_sequence",
                "n_intermediate_city_episodes",
            ],
        )
        if final["candidate_id"].duplicated().any():
            raise ValueError(f"{year} final candidate IDs duplicate")
        final_ids.append(final["candidate_id"].astype(str))

        bridge = bridge_registry.loc[
            bridge_registry["analysis_year"].eq(year)
        ].copy()
        if year in BUFFERED_YEARS:
            mapped = map_bridge_legs_to_candidates(bridge, legs, candidates)
            mapped = mapped.merge(
                event_counts,
                left_on="trip_id",
                right_index=True,
                how="left",
                validate="one_to_one",
            )
            mapped["episode_survived"] = mapped["event_rows"].eq(2) & mapped[
                "event_types"
            ].eq(2)
            mapped["candidate_survived"] = mapped["candidate_id"].notna()
            c2_ids = set(c2["candidate_id"].astype(str))
            mapped["c2_survived"] = mapped["candidate_id"].isin(c2_ids)
            c3_index = c3.set_index("candidate_id")
            for column, target in (
                ("retain_after_route_circuity", "route_survived"),
                ("retain_after_c3_before_same_node_backstop", "duration_survived"),
                ("retain_after_c3_final", "c3_survived"),
            ):
                mapped[target] = mapped["candidate_id"].map(c3_index[column]).eq(True)
            pretrim_index = pretrim.set_index("candidate_id")
            mapped["temporal_survived"] = (
                mapped["candidate_id"]
                .map(pretrim_index["temporal_scope_eligible"])
                .eq(True)
            )
            new_threshold = load_json(
                run_root / f"manifests/qtrim_thresholds_{year}.json"
            )
            ratio = mapped["candidate_id"].map(pretrim_index["normalized_time_ratio"])
            mapped["q01_survived"] = mapped["temporal_survived"] & ~ratio.lt(
                float(new_threshold["normalized_time_ratio_q01"])
            ).fillna(False)
            mapped["q99_survived"] = mapped["q01_survived"] & ~ratio.gt(
                float(new_threshold["normalized_time_ratio_q99"])
            ).fillna(False)
            final_set = set(final["candidate_id"].astype(str))
            mapped["final_survived"] = mapped["candidate_id"].isin(final_set)

            mapped["first_drop_stage"] = ""
            mapped["first_drop_reason"] = ""
            ordered_rules = [
                ("episode", "episode_survived", "trip_not_retained_as_complete_episode_leg"),
                ("candidate", "candidate_survived", "no_adjacent_commercial_candidate"),
                ("c2_idle_gap", "c2_survived", "idle_gap_gt_30_days"),
                ("c3_route", "route_survived", "route_circuity_excluded"),
                ("c3_duration_speed", "duration_survived", "duration_speed_excluded"),
                ("c3_same_node", "c3_survived", "same_node_backstop_excluded"),
                ("temporal_scope", "temporal_survived", "outside_common_followup"),
                ("q01_trim", "q01_survived", "below_normalized_time_ratio_q01"),
                ("q99_trim", "q99_survived", "above_normalized_time_ratio_q99"),
                ("final", "final_survived", "final_membership_mismatch"),
            ]
            for stage, column, reason in ordered_rules:
                dropped = mapped["first_drop_stage"].eq("") & ~mapped[column]
                mapped.loc[dropped, "first_drop_stage"] = stage
                mapped.loc[dropped, "first_drop_reason"] = reason
            same_city = mapped["origin_city"].astype(str).eq(
                mapped["dest_city"].astype(str)
            )
            mapped.loc[
                mapped["first_drop_stage"].eq("candidate") & same_city,
                "first_drop_reason",
            ] = "same_analytical_city_merged_into_one_episode"
            route_drop = mapped["first_drop_stage"].eq("c3_route")
            mapped.loc[route_drop, "first_drop_reason"] = mapped.loc[
                route_drop, "candidate_id"
            ].map(c3_index["route_circuity_decision_basis"])
            duration_drop = mapped["first_drop_stage"].eq("c3_duration_speed")
            mapped.loc[duration_drop, "first_drop_reason"] = mapped.loc[
                duration_drop, "candidate_id"
            ].map(c3_index["extreme_duration_speed_decision_reason"])
            same_node_drop = mapped["first_drop_stage"].eq("c3_same_node")
            mapped.loc[same_node_drop, "first_drop_reason"] = mapped.loc[
                same_node_drop, "candidate_id"
            ].map(c3_index["same_node_route_backstop_decision_reason"])
            lineage_frames.append(mapped)
        else:
            mapped = pd.DataFrame(columns=["candidate_id"])

        old_threshold = load_json(OLD_THRESHOLD_PATHS[year])
        new_threshold = load_json(run_root / f"manifests/qtrim_thresholds_{year}.json")
        bridge_candidate_ids = set(mapped.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str))
        nonbridge = pretrim.loc[
            ~pretrim["candidate_id"].astype(str).isin(bridge_candidate_ids)
        ].copy()
        old_keep = qtrim_keep(
            nonbridge,
            float(old_threshold["normalized_time_ratio_q01"]),
            float(old_threshold["normalized_time_ratio_q99"]),
        )
        new_keep = qtrim_keep(
            nonbridge,
            float(new_threshold["normalized_time_ratio_q01"]),
            float(new_threshold["normalized_time_ratio_q99"]),
        )
        flips = nonbridge.loc[old_keep.ne(new_keep)].copy()
        if not flips.empty:
            flips["analysis_year"] = year
            flips["old_keep"] = old_keep.loc[flips.index].to_numpy()
            flips["new_keep"] = new_keep.loc[flips.index].to_numpy()
            flip_frames.append(flips)
        threshold_records.append(
            {
                "year": year,
                "old_q01": float(old_threshold["normalized_time_ratio_q01"]),
                "new_q01": float(new_threshold["normalized_time_ratio_q01"]),
                "old_q99": float(old_threshold["normalized_time_ratio_q99"]),
                "new_q99": float(new_threshold["normalized_time_ratio_q99"]),
                "nonbridge_pretrim_rows": int(len(nonbridge)),
                "nonbridge_retention_flips": int(old_keep.ne(new_keep).sum()),
                "old_only_retained": int((old_keep & ~new_keep).sum()),
                "new_only_retained": int((~old_keep & new_keep).sum()),
            }
        )

        raw_departure = pd.to_datetime(candidates["departure_time"], utc=True)
        raw_arrival = pd.to_datetime(candidates["arrival_time"], utc=True)
        raw_cohort_mask = raw_departure.dt.year.eq(year)
        raw_cohort = candidates.loc[raw_cohort_mask].copy()
        final_departure = pd.to_datetime(final["departure_time"], utc=True)
        final_arrival = pd.to_datetime(final["arrival_time"], utc=True)
        annual_records.append(
            {
                "year": year,
                "raw_buffer_candidate_rows": int(len(candidates)),
                "raw_cohort_candidate_rows": int(raw_cohort_mask.sum()),
                "final_candidate_rows": int(len(final)),
                "raw_dep_y_arr_y1": int(
                    (raw_departure.dt.year.eq(year) & raw_arrival.dt.year.eq(year + 1)).sum()
                ),
                "final_dep_y_arr_y1": int(
                    (
                        final_departure.dt.year.eq(year)
                        & final_arrival.dt.year.eq(year + 1)
                    ).sum()
                ),
                "raw_december_profile": december_profile(raw_cohort),
                "final_december_profile": december_profile(final),
            }
        )

    if namespace_double_prefix:
        raise ValueError(
            f"real episode data contains {namespace_double_prefix} double-prefixed visit IDs"
        )
    combined_raw_ids = pd.concat(raw_ids, ignore_index=True)
    combined_final_ids = pd.concat(final_ids, ignore_index=True)
    if combined_raw_ids.duplicated().any() or combined_final_ids.duplicated().any():
        raise ValueError("candidate IDs duplicate across years")
    for item in annual_records:
        if item["year"] in BUFFERED_YEARS:
            if item["raw_dep_y_arr_y1"] < 1_000:
                raise ValueError(f"{item['year']} raw next-year candidates remain too small")
            if item["raw_december_profile"]["monotone_collapse"]:
                raise ValueError(f"{item['year']} raw candidate December cliff remains")

    lineage = pd.concat(lineage_frames, ignore_index=True)
    if len(lineage) != 80_489 or lineage["trip_id"].duplicated().any():
        raise ValueError("bridge lineage does not contain exactly 80,489 unique ODs")
    stage_columns = [
        ("od_input", None),
        ("episode", "episode_survived"),
        ("candidate", "candidate_survived"),
        ("c2_idle_gap", "c2_survived"),
        ("c3_route", "route_survived"),
        ("c3_duration_speed", "duration_survived"),
        ("c3_same_node", "c3_survived"),
        ("temporal_scope", "temporal_survived"),
        ("q01_trim", "q01_survived"),
        ("q99_trim", "q99_survived"),
        ("final", "final_survived"),
    ]
    attrition_rows = []
    previous = len(lineage)
    for stage, column in stage_columns:
        retained = len(lineage) if column is None else int(lineage[column].sum())
        attrition_rows.append(
            {
                "stage": stage,
                "bridge_ods_retained": retained,
                "dropped_from_previous": previous - retained,
                "drop_reasons": {} if column is None else reason_counts(lineage, stage),
            }
        )
        previous = retained
    attrition = pd.DataFrame(attrition_rows)
    thresholds = pd.DataFrame(threshold_records)
    annual = pd.DataFrame(
        [
            {
                **{key: value for key, value in item.items() if "profile" not in key},
                "raw_dec_28_31_pct": json.dumps(
                    item["raw_december_profile"]["december_28_31_percent"]
                ),
                "raw_december_monotone_collapse": item["raw_december_profile"][
                    "monotone_collapse"
                ],
                "final_dec_28_31_pct": json.dumps(
                    item["final_december_profile"]["december_28_31_percent"]
                ),
                "final_december_monotone_collapse": item[
                    "final_december_profile"
                ]["monotone_collapse"],
            }
            for item in annual_records
        ]
    )
    attrition.to_csv(output_dir / "01_bridge_attrition.csv", index=False)
    lineage.to_feather(
        output_dir / "02_bridge_lineage.feather",
        compression="zstd",
        compression_level=6,
    )
    thresholds.to_csv(output_dir / "03_threshold_shift.csv", index=False)
    annual.to_csv(output_dir / "04_annual_candidate_checks.csv", index=False)
    if flip_frames:
        pd.concat(flip_frames, ignore_index=True).to_feather(
            output_dir / "05_nonbridge_threshold_flips.feather",
            compression="zstd",
            compression_level=6,
        )
    total_flips = int(thresholds["nonbridge_retention_flips"].sum())
    payload = {
        "status": "review_required" if total_flips else "pass",
        "headline": {
            "bridge_od_input": 80_489,
            "bridge_final_retained": int(lineage["final_survived"].sum()),
            "bridge_final_retention_rate": float(lineage["final_survived"].mean()),
        },
        "attrition": attrition_rows,
        "threshold_comparison": threshold_records,
        "nonbridge_retention_flips": total_flips,
        "annual": annual_records,
        "five_year": {
            "raw_candidate_rows": int(len(combined_raw_ids)),
            "final_candidate_rows": int(len(combined_final_ids)),
            "raw_unique_candidate_ids": int(combined_raw_ids.nunique()),
            "final_unique_candidate_ids": int(combined_final_ids.nunique()),
        },
        "namespace": {
            "visit_id_rows_scanned": int(namespace_rows),
            "double_year_prefix_rows": int(namespace_double_prefix),
        },
        "execution_boundary": {
            "candidate_root_run": False,
            "ordered_suez_run": False,
            "strict_run": False,
            "reachability_run": False,
            "weather_run": False,
            "adoption_run": False,
        },
    }
    p0.write_json_atomic(output_dir / "validation_manifest.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_root / "validation/e030_stage30_20"
    )
    result = validate(run_root, args.od_root, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
