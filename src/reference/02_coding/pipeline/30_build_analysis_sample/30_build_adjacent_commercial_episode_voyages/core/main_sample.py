#!/usr/bin/env python3
"""Build the 2021 multi-call main sample from frozen quality-stage products.

C4 deliberately does not add a new physical cap.  The frozen C3 decision
manifest is the physical-rule boundary.  This module joins the C3-final
candidate decisions to the full C2 candidate records and the C0 yardstick
table, limits the 2021 main sample to the configured departure-period scope,
computes strict q01/q99 normalized-time trimming thresholds by the formal P1
ship-type group, and writes the full C4 decision table plus the retained main
sample and aggregate panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


CODING_ROOT = Path(__file__).resolve().parents[4]
C4_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = C4_ROOT / "config/frozen/main_sample_2021.json"
JOINED_ROOT = DATA_ROOT / "main/provenance/main_sample_2021/joined"
TRIM_ROOT = DATA_ROOT / "main/provenance/main_sample_2021/qtrim"
MAIN_ROOT = DATA_ROOT / "main/provenance/main_sample_2021/final"
REPORT_ROOT = DATA_ROOT / "main/validation/main_sample_2021"
RUN_ROOT = (
    CODING_ROOT
    / ".runs/build_multicall_first_passage_voyages/"
    "main_sample_2021"
)

C3_DECISION_COLUMNS = [
    "candidate_id",
    "pair_id",
    "vessel_uid_clean",
    "origin_city",
    "dest_city",
    "observed_sea_hours",
    "route_sequence_d_base_km",
    "route_circuity_decision_basis",
    "retain_after_route_circuity",
    "observed_sea_days",
    "implied_route_sequence_speed_knots",
    "extreme_duration_speed_decision_reason",
    "exclude_extreme_duration_speed_final",
    "retain_after_c3_final",
    "direct_d_base_km",
    "direct_endpoint_same_network_node",
    "same_node_route_backstop_km",
    "exclude_same_node_route_backstop",
    "same_node_route_backstop_decision_reason",
    "c3_final_decision_reason",
]

YARDSTICK_COLUMNS = [
    "pair_id",
    "d_base",
    "d_canal",
    "d_redsea",
    "d_yardstick",
    "regime_used",
    "feasible_flags",
    "same_network_node",
]

SHIPTYPE_GROUP_RULE_VERSION = "p1_p5_shiptype_group_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, encoding="utf-8", delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def write_feather_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".feather", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.reset_index(drop=True).to_feather(
            temporary, compression="zstd", compression_level=6
        )
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, encoding="utf-8", newline="", delete=False
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False)
    os.replace(temporary, path)
    path.chmod(0o644)


def validate_config(config: dict[str, Any]) -> None:
    if int(config["analysis_year"]) != 2021:
        raise ValueError("C4 is restricted to the 2021 development sample")
    physical_cap = config["physical_cap"]
    if physical_cap.get("enabled") is not False:
        raise ValueError("C4 must not enable a new physical cap")
    if physical_cap.get("rule_version") != "c3_frozen_no_additional_cap":
        raise ValueError("Unexpected C4 physical-cap rule version")
    trim = config["normalized_time_trim"]
    if trim.get("enabled") is not True:
        raise ValueError("The C4 main specification requires q01/q99 trimming")
    if not 0 < float(trim["lower_quantile"]) < float(trim["upper_quantile"]) < 1:
        raise ValueError("Invalid q01/q99 quantiles")
    if trim["group_columns"] != ["shiptype_group", "period_id"]:
        raise ValueError("C4 trim grouping must remain formal ship type by period")
    if trim.get("shiptype_group_rule_version") != SHIPTYPE_GROUP_RULE_VERSION:
        raise ValueError("Unexpected ship-type grouping rule version")
    temporal_scope = config["temporal_scope"]
    if temporal_scope.get("eligible_period_ids") != ["2021_full_year"]:
        raise ValueError("C4 main-sample temporal scope must remain 2021_full_year")
    if temporal_scope.get("exclude_from_main_sample") is not True:
        raise ValueError("Out-of-scope departures must not enter the C4 main sample")


def verify_manifest_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} manifest hash mismatch: {observed}")


def c2_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = manifest.get("parts")
    if raw_items is None:
        raw_items = manifest.get("retained_candidate_parts")
    if raw_items is None:
        raise ValueError(
            "C2 manifest must contain 'parts' or 'retained_candidate_parts'"
        )
    items = sorted(
        raw_items,
        key=lambda item: int(item["task_id"]),
    )
    if [int(item["task_id"]) for item in items] != list(range(len(items))):
        raise ValueError("C2 retained parts do not have contiguous task IDs")
    required = {
        "task_id",
        "retained_path",
        "retained_rows",
        "retained_sha256",
    }
    for item in items:
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"C2 retained part {item.get('task_id')} missing keys: "
                f"{sorted(missing)}"
            )
    return items


def map_shiptype_group(detailed_type: Any) -> str:
    """Reproduce the formal P1/p5 shiptype_group mapping."""

    text = " ".join(str(detailed_type).strip().upper().split())
    if "CONTAINER" in text:
        return "container"
    if "BULK" in text or text in {"ORE CARRIER", "OBO CARRIER"}:
        return "bulk_carrier"
    if "OIL PRODUCTS" in text or "CHEMICAL" in text or "OIL/CHEMICAL" in text:
        return "product_chemical_tanker"
    if "CRUDE OIL" in text:
        return "crude_oil_tanker"
    if "LNG" in text or "LPG" in text:
        return "gas_tanker"
    if "GENERAL CARGO" in text:
        return "general_cargo"
    if "RO-RO" in text or "VEHICLE" in text:
        return "ro_ro_vehicle"
    if "REEFER" in text:
        return "reefer"
    return "other_retained"


def load_c3_final_decisions(
    manifest: dict[str, Any],
    *,
    expected_rows: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    items = sorted(manifest["parts"], key=lambda item: int(item["task_id"]))
    for item in items:
        path = CODING_ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"C3 part hash mismatch: {path}")
        frame = pd.read_feather(path)
        missing = set(C3_DECISION_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"C3 part missing columns: {sorted(missing)}")
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"Duplicate C3 candidate IDs: {path}")
        final = frame.loc[frame["retain_after_c3_final"].astype(bool)].copy()
        expected_part_rows = int(item["final_retained_candidates"])
        if len(final) != expected_part_rows:
            raise ValueError(
                f"C3 final row count mismatch for task {item['task_id']}: "
                f"{len(final)} != {expected_part_rows}"
            )
        frames.append(final[C3_DECISION_COLUMNS])
    output = pd.concat(frames, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs across C3 final parts")
    if len(output) != expected_rows:
        raise ValueError(f"C3 final row count mismatch: {len(output)} != {expected_rows}")
    return output


def load_yardstick(path: Path, expected_sha256: str) -> pd.DataFrame:
    if sha256_file(path) != expected_sha256:
        raise ValueError("C0 yardstick table hash mismatch")
    frame = pd.read_feather(path, columns=YARDSTICK_COLUMNS)
    return validate_yardstick(frame)


def validate_yardstick(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the max over feasible C0 route regimes."""

    if frame["pair_id"].duplicated().any():
        raise ValueError("C0 yardstick table contains duplicate pair IDs")
    distances = frame[["d_base", "d_canal", "d_redsea"]].to_numpy(float)
    finite = np.isfinite(distances)
    if (~finite.any(axis=1)).any():
        raise ValueError("C0 yardstick row has no feasible route regime")
    if (distances[finite] < 0).any():
        raise ValueError("C0 route-regime distances cannot be negative")

    parsed_flags: list[dict[str, Any]] = []
    for value in frame["feasible_flags"]:
        try:
            flags = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("C0 feasible_flags is not valid JSON") from error
        if not isinstance(flags, dict):
            raise ValueError("C0 feasible_flags must decode to an object")
        parsed_flags.append(flags)
    for index, regime in enumerate(["base", "canal", "redsea"]):
        declared = np.array(
            [flags.get(regime) is True for flags in parsed_flags],
            dtype=bool,
        )
        if not np.array_equal(declared, finite[:, index]):
            raise ValueError(
                f"C0 {regime} feasibility flags disagree with route distances"
            )

    expected = np.max(np.where(finite, distances, -np.inf), axis=1)
    observed = frame["d_yardstick"].to_numpy(float)
    if not np.isfinite(observed).all():
        raise ValueError("C0 d_yardstick must be finite")
    if not np.allclose(
        observed, expected, rtol=1e-12, atol=1e-9
    ):
        raise ValueError("C0 d_yardstick is not the max of feasible regimes")
    if (observed < 0).any():
        raise ValueError("C0 yardstick distances cannot be negative")
    return frame


def _assert_equal_column(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    tolerance: float = 1e-9,
) -> None:
    left_values = frame[left]
    right_values = frame[right]
    if pd.api.types.is_numeric_dtype(left_values) and pd.api.types.is_numeric_dtype(
        right_values
    ):
        if not np.allclose(
            left_values.to_numpy(float),
            right_values.to_numpy(float),
            rtol=1e-12,
            atol=tolerance,
            equal_nan=True,
        ):
            raise ValueError(f"Joined columns disagree: {left} vs {right}")
        return
    left_text = left_values.astype("string").fillna("<NA>")
    right_text = right_values.astype("string").fillna("<NA>")
    if not left_text.eq(right_text).all():
        raise ValueError(f"Joined columns disagree: {left} vs {right}")


def _add_nullable_p4_flags(frame: pd.DataFrame) -> None:
    frame["suez_box_flag"] = pd.Series(
        pd.array([pd.NA] * len(frame), dtype="boolean"), index=frame.index
    )
    frame["cape_box_flag"] = pd.Series(
        pd.array([pd.NA] * len(frame), dtype="boolean"), index=frame.index
    )
    frame["p4_corridor_flag_status"] = "not_available_in_current_c2_product"


def add_c4_fields(
    frame: pd.DataFrame,
    *,
    eligible_period_ids: Sequence[str] = ("2021_full_year",),
) -> pd.DataFrame:
    result = frame.copy()
    result["analysis_year"] = 2021
    result["passage_id"] = result["candidate_id"]
    result["shiptype_group"] = result["detailed_vessel_type_normalized"].map(
        map_shiptype_group
    )
    result["shiptype_group_rule_version"] = SHIPTYPE_GROUP_RULE_VERSION
    result["ship_type"] = result["shiptype_group"]
    result["period_window"] = result["period_id"]
    result["total_voyage_time_hours"] = result["total_hours"].astype(float)
    result["total_voyage_time_days"] = result["total_voyage_time_hours"] / 24.0
    result["observed_sea_sailing_time_hours"] = result[
        "observed_sea_hours"
    ].astype(float)
    result["n_intermediate_calls"] = result["n_intermediate_raw_stops"].astype(
        "int64"
    )
    result["n_intermediate_cities"] = result[
        "n_intermediate_city_episodes"
    ].astype("int64")
    result["r1_valid"] = True
    result["r2_loop_risk"] = result["r2_repeated_intermediate_city"].astype(bool)
    result["r3_support_count"] = result[
        "r3_recurrence_count_post_r4"
    ].astype("int64")
    result["r3_singleton"] = result["r3_singleton_post_r4"].astype(bool)
    result["r4_idle_break"] = False
    result["congestion_exempt"] = False
    result["singleton"] = result["r3_singleton_post_r4"].astype(bool)
    result["recurrence_count"] = result[
        "r3_recurrence_count_post_r4"
    ].astype("int64")
    result["transition_location_quality"] = np.select(
        [
            result["n_strict_unresolved_raw_stops"].gt(0),
            result["n_blank_topology_raw_stops"].gt(0),
            result["n_conflicting_city_raw_stops"].gt(0),
        ],
        ["strict_unresolved", "blank_topology", "conflicting_city"],
        default="resolved_or_not_applicable",
    )
    result["resolved_stationary_time_hours"] = np.nan
    result["unresolved_transition_gap_time_hours"] = np.nan
    result["time_breakdown_status"] = "not_available_in_current_c2_product"
    result["departure_month"] = pd.to_datetime(
        result["departure_time"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m")
    result["departure_year"] = pd.to_datetime(
        result["departure_time"], utc=True, errors="coerce"
    ).dt.year.astype("Int64")
    period_eligible = result["period_id"].isin(list(eligible_period_ids))
    window_eligible = result["r3_window_eligible"].astype(bool)
    result["temporal_scope_eligible"] = period_eligible & window_eligible
    result["temporal_scope_excluded"] = ~result["temporal_scope_eligible"]
    result["temporal_scope_decision_reason"] = np.where(
        result["temporal_scope_eligible"],
        "retained_2021_departure_period",
        "excluded_outside_2021_departure_period",
    )
    _add_nullable_p4_flags(result)

    result["cap_exceeded"] = False
    result["retain_after_physical_cap"] = True
    result["cap_rule_version"] = "c3_frozen_no_additional_cap"
    result["cap_ratio"] = np.nan
    result["cap_status"] = "not_applied_c3_boundary"

    route_sequence = result["route_sequence_d_base_km"].to_numpy(float)
    direct = result["direct_d_base_km"].to_numpy(float)
    route_ratio = np.full(len(result), np.nan, dtype=np.float64)
    route_ratio_eligible = (
        np.isfinite(route_sequence) & np.isfinite(direct) & (direct > 0)
    )
    np.divide(
        route_sequence,
        direct,
        out=route_ratio,
        where=route_ratio_eligible,
    )
    result["route_sequence_d_base"] = route_sequence
    result["direct_d_base"] = direct
    result["route_circuity_ratio"] = route_ratio
    result["route_circuity_status"] = result[
        "route_circuity_decision_basis"
    ].astype("string")
    result["route_circuity_exceeded"] = ~result[
        "retain_after_route_circuity"
    ].astype(bool)
    result["implied_speed_knots"] = result[
        "implied_route_sequence_speed_knots"
    ].astype(float)

    total_days = result["total_voyage_time_days"].to_numpy(float)
    yardstick = result["d_yardstick"].to_numpy(float)
    ratio = np.full(len(result), np.nan, dtype=np.float64)
    eligible = np.isfinite(total_days) & np.isfinite(yardstick) & (yardstick > 0)
    np.divide(total_days, yardstick, out=ratio, where=eligible)
    result["normalized_time_ratio"] = ratio
    result["normalized_time_ratio_unit"] = "days_per_yardstick_km"
    return result.reset_index(drop=True)


def join_c2_with_c3_and_yardstick(
    candidates: pd.DataFrame,
    c3_final: pd.DataFrame,
    yardstick: pd.DataFrame,
    *,
    eligible_period_ids: Sequence[str] = ("2021_full_year",),
) -> pd.DataFrame:
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("C2 candidate part contains duplicate IDs")
    decision = c3_final.copy()
    overlap = [
        "pair_id",
        "vessel_uid_clean",
        "origin_city",
        "dest_city",
        "observed_sea_hours",
    ]
    decision = decision.rename(columns={column: f"{column}__c3" for column in overlap})
    merged = candidates.merge(
        decision,
        left_on="candidate_id",
        right_on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    for column in overlap:
        _assert_equal_column(merged, column, f"{column}__c3")
        merged = merged.drop(columns=f"{column}__c3")
    joined = merged.merge(
        yardstick,
        on="pair_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "__yardstick"),
    )
    if joined["d_yardstick"].isna().any():
        raise ValueError("C3-final candidates missing C0 yardstick rows")
    return add_c4_fields(joined, eligible_period_ids=eligible_period_ids)


def apply_no_additional_physical_cap(frame: pd.DataFrame) -> pd.DataFrame:
    """Audit the C3 boundary without applying a new C4 exclusion."""

    result = frame.copy()
    result["cap_exceeded"] = False
    result["retain_after_physical_cap"] = True
    result["cap_rule_version"] = "c3_frozen_no_additional_cap"
    result["cap_status"] = "not_applied_c3_boundary"
    return result


def compute_quantile_thresholds(
    frames: Iterable[pd.DataFrame],
    *,
    group_columns: Sequence[str],
    lower_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    compact = pd.concat(
        [
            frame[
                list(group_columns)
                + ["normalized_time_ratio", "temporal_scope_eligible"]
            ]
            for frame in frames
        ],
        ignore_index=True,
    )
    eligible = compact.loc[
        compact["temporal_scope_eligible"].astype(bool)
        & np.isfinite(compact["normalized_time_ratio"])
    ].copy()
    if eligible.empty:
        raise ValueError("No positive-yardstick rows available for q trimming")
    grouped = eligible.groupby(
        list(group_columns), dropna=False, observed=True, sort=True
    )["normalized_time_ratio"]
    thresholds = grouped.quantile([lower_quantile, upper_quantile]).unstack(
        level=-1
    )
    thresholds = thresholds.rename(
        columns={
            lower_quantile: "normalized_time_ratio_q01",
            upper_quantile: "normalized_time_ratio_q99",
        }
    ).reset_index()
    counts = (
        eligible.groupby(
            list(group_columns), dropna=False, observed=True, sort=True
        )
        .size()
        .rename("quantile_eligible_rows")
        .reset_index()
    )
    thresholds = thresholds.merge(
        counts, on=list(group_columns), how="left", validate="one_to_one"
    )
    thresholds["lower_quantile"] = float(lower_quantile)
    thresholds["upper_quantile"] = float(upper_quantile)
    return thresholds


def apply_quantile_trim(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    result = frame.merge(
        thresholds[
            list(group_columns)
            + ["normalized_time_ratio_q01", "normalized_time_ratio_q99"]
        ],
        on=list(group_columns),
        how="left",
        validate="many_to_one",
    )
    ratio = result["normalized_time_ratio"]
    eligible = (
        result["temporal_scope_eligible"].astype(bool)
        &
        ratio.notna()
        & np.isfinite(ratio.to_numpy(float))
        & result["d_yardstick"].gt(0)
    )
    lower = eligible & ratio.lt(result["normalized_time_ratio_q01"])
    upper = eligible & ratio.gt(result["normalized_time_ratio_q99"])
    result["trimmed"] = lower | upper
    result["retain_after_c4"] = (
        result["temporal_scope_eligible"].astype(bool) & ~result["trimmed"]
    )
    result["c4_trim_rule_version"] = "strict_q01_q99_ship_type_period"
    result["c4_decision_reason"] = np.select(
        [
            result["temporal_scope_excluded"].astype(bool),
            lower,
            upper,
            ~eligible,
        ],
        [
            "excluded_outside_2021_departure_period",
            "normalized_time_below_q01",
            "normalized_time_above_q99",
            "ratio_undefined_not_trimmed",
        ],
        default="retained_within_q01_q99",
    )
    return result.reset_index(drop=True)


def build_aggregate_panel(
    decisions: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (
        "origin_city",
        "dest_city",
        "shiptype_group",
        "departure_month",
    ),
) -> pd.DataFrame:
    all_summary = (
        decisions.groupby(list(group_columns), dropna=False, observed=True, sort=True)
        .agg(
            c4_input_passages=("candidate_id", "size"),
            c4_temporal_scope_excluded_passages=(
                "temporal_scope_excluded",
                "sum",
            ),
            c4_trimmed_passages=("trimmed", "sum"),
            cap_exceeded_passages=("cap_exceeded", "sum"),
        )
        .reset_index()
    )
    retained = decisions.loc[decisions["retain_after_c4"]].copy()
    if retained.empty:
        raise ValueError("C4 main sample is empty")
    retained["_transition_location_unresolved"] = retained[
        "transition_location_quality"
    ].ne("resolved_or_not_applicable")
    retained_grouped = retained.groupby(
        list(group_columns),
        dropna=False,
        observed=True,
        sort=True,
    )
    main_summary = (
        retained_grouped
        .agg(
            c4_retained_passages=("candidate_id", "size"),
            independent_vessels=("vessel_uid_clean", "nunique"),
            total_voyage_time_p50_days=("total_voyage_time_days", "median"),
            observed_sea_time_p50_hours=(
                "observed_sea_sailing_time_hours", "median"
            ),
            intermediate_gap_time_p50_hours=("intermediate_gap_hours", "median"),
            unresolved_location_share=(
                "_transition_location_unresolved",
                "mean",
            ),
        )
        .reset_index()
    )
    voyage_time_quantiles = (
        retained_grouped["total_voyage_time_days"]
        .quantile([0.25, 0.75])
        .unstack(level=-1)
        .rename(
            columns={
                0.25: "total_voyage_time_p25_days",
                0.75: "total_voyage_time_p75_days",
            }
        )
        .reset_index()
    )
    main_summary = main_summary.merge(
        voyage_time_quantiles,
        on=list(group_columns),
        how="left",
        validate="one_to_one",
    )
    output = main_summary.merge(
        all_summary,
        on=list(group_columns),
        how="left",
        validate="one_to_one",
    )
    output["c4_trimmed_share"] = np.divide(
        output["c4_trimmed_passages"],
        output["c4_input_passages"],
        out=np.zeros(len(output), dtype=float),
        where=output["c4_input_passages"].to_numpy() > 0,
    )
    output["c4_cap_rule_version"] = "c3_frozen_no_additional_cap"
    output["c4_trim_rule_version"] = "strict_q01_q99_ship_type_period"
    output["shiptype_group_rule_version"] = SHIPTYPE_GROUP_RULE_VERSION
    return output


def build_attrition_by_period_regime(decisions: pd.DataFrame) -> pd.DataFrame:
    """Build the available attrition cross-tab without inventing route-exposure flags."""

    output = (
        decisions.groupby(
            ["period_id", "regime_used"],
            dropna=False,
            observed=True,
            sort=True,
        )
        .agg(
            c3_final_input_passages=("candidate_id", "size"),
            temporal_scope_excluded_passages=("temporal_scope_excluded", "sum"),
            additional_cap_excluded_passages=("cap_exceeded", "sum"),
            q_trim_excluded_passages=("trimmed", "sum"),
            c4_retained_passages=("retain_after_c4", "sum"),
        )
        .reset_index()
    )
    denominator = output["c3_final_input_passages"].to_numpy(float)
    for count_column, share_column in [
        ("temporal_scope_excluded_passages", "temporal_scope_excluded_share"),
        ("additional_cap_excluded_passages", "additional_cap_excluded_share"),
        ("q_trim_excluded_passages", "q_trim_excluded_share"),
        ("c4_retained_passages", "c4_retained_share"),
    ]:
        output[share_column] = np.divide(
            output[count_column],
            denominator,
            out=np.zeros(len(output), dtype=float),
            where=denominator > 0,
        )
    output["dependency_audit_basis"] = (
        "c0_route_envelope_regime_used_not_p4_suez_exposure"
    )
    return output


def part_complete(
    output_path: Path,
    sidecar_path: Path,
    *,
    contract_hash: str,
    source_sha256: str,
) -> bool:
    if not output_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        sidecar = load_json(sidecar_path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        sidecar.get("contract_hash") == contract_hash
        and sidecar.get("source_sha256") == source_sha256
        and sidecar.get("output_sha256") == sha256_file(output_path)
    )


def load_verified_c2_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = CODING_ROOT / config["c2_output_manifest"]
    verify_manifest_hash(
        path, config["c2_output_manifest_sha256"], "C2 output"
    )
    manifest = load_json(path)
    items = c2_items(manifest)
    if len(items) != int(config["expected_parts"]):
        raise ValueError("Unexpected C2 retained part count")
    observed_rows = sum(int(item["retained_rows"]) for item in items)
    declared_rows = manifest.get("retained_candidate_rows")
    if declared_rows is not None and int(declared_rows) != observed_rows:
        raise ValueError("C2 manifest declared row count does not match its parts")
    if observed_rows != int(config["expected_c2_retained_rows"]):
        raise ValueError("Unexpected C2 retained row count")
    return manifest


def load_verified_c3_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = CODING_ROOT / config["c3_final_output_manifest"]
    verify_manifest_hash(
        path, config["c3_final_output_manifest_sha256"], "C3 final output"
    )
    manifest = load_json(path)
    items = sorted(manifest["parts"], key=lambda item: int(item["task_id"]))
    if len(items) != int(config["expected_parts"]):
        raise ValueError("Unexpected C3 final part count")
    if [int(item["task_id"]) for item in items] != list(range(len(items))):
        raise ValueError("C3 final parts do not have contiguous task IDs")
    final_rows = sum(int(item["final_retained_candidates"]) for item in items)
    if final_rows != int(config["expected_c3_final_rows"]):
        raise ValueError("Unexpected C3 final retained row count")
    if int(manifest["summary"]["final_retained_candidates"]) != final_rows:
        raise ValueError("C3 final summary does not match its parts")
    return manifest


def prepare_joined_parts(
    config: dict[str, Any],
    *,
    contract_hash: str,
    c2_manifest: dict[str, Any],
    c3_final: pd.DataFrame,
    yardstick: pd.DataFrame,
) -> list[dict[str, Any]]:
    output_parts: list[dict[str, Any]] = []
    sidecar_root = RUN_ROOT / "c4_1_joined_parts"
    for item in c2_items(c2_manifest):
        task_id = int(item["task_id"])
        source_path = CODING_ROOT / item["retained_path"]
        source_sha256 = item["retained_sha256"]
        if sha256_file(source_path) != source_sha256:
            raise ValueError(f"C2 retained part hash mismatch: {source_path}")
        output_path = JOINED_ROOT / "parts" / f"part-{task_id:03d}.feather"
        sidecar_path = sidecar_root / f"part-{task_id:03d}.json"
        if not part_complete(
            output_path,
            sidecar_path,
            contract_hash=contract_hash,
            source_sha256=source_sha256,
        ):
            candidates = pd.read_feather(source_path)
            joined = join_c2_with_c3_and_yardstick(
                candidates,
                c3_final,
                yardstick,
                eligible_period_ids=config["temporal_scope"]["eligible_period_ids"],
            )
            joined = apply_no_additional_physical_cap(joined)
            write_feather_atomic(output_path, joined)
            write_json_atomic(
                sidecar_path,
                {
                    "task_id": task_id,
                    "contract_hash": contract_hash,
                    "source_sha256": source_sha256,
                    "output_sha256": sha256_file(output_path),
                    "c2_input_rows": int(len(candidates)),
                    "c3_final_rows": int(len(joined)),
                    "cap_exceeded_rows": int(joined["cap_exceeded"].sum()),
                },
            )
        sidecar = load_json(sidecar_path)
        output_parts.append(
            {
                "task_id": task_id,
                "path": str(output_path.relative_to(CODING_ROOT)),
                "rows": int(sidecar["c3_final_rows"]),
                "sha256": sidecar["output_sha256"],
                "source_sha256": source_sha256,
            }
        )
    if sum(item["rows"] for item in output_parts) != int(
        config["expected_c3_final_rows"]
    ):
        raise ValueError("Joined C4 parts do not reproduce the C3 final count")
    joined_candidate_ids = pd.concat(
        [
            pd.read_feather(CODING_ROOT / item["path"], columns=["candidate_id"])[
                "candidate_id"
            ]
            for item in output_parts
        ],
        ignore_index=True,
    )
    if joined_candidate_ids.duplicated().any():
        raise ValueError("Duplicate candidate IDs across joined C4 parts")
    c3_candidate_index = pd.Index(c3_final["candidate_id"])
    joined_candidate_index = pd.Index(joined_candidate_ids)
    if (
        not c3_candidate_index.difference(joined_candidate_index).empty
        or not joined_candidate_index.difference(c3_candidate_index).empty
    ):
        raise ValueError("Joined C4 candidate IDs do not exactly reproduce C3 final")
    manifest = {
        "analysis_year": 2021,
        "contract_hash": contract_hash,
        "parts": output_parts,
        "rows": sum(item["rows"] for item in output_parts),
        "cap_exceeded_rows": 0,
        "status": "pass",
    }
    write_json_atomic(JOINED_ROOT / "joined_output_manifest.json", manifest)
    return output_parts


def read_joined_parts(parts: Sequence[dict[str, Any]]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for item in sorted(parts, key=lambda value: int(value["task_id"])):
        path = CODING_ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"C4 joined part hash mismatch: {path}")
        frame = pd.read_feather(path)
        if len(frame) != int(item["rows"]):
            raise ValueError(f"C4 joined row count mismatch: {path}")
        frames.append(frame)
    return frames


def apply_trimmed_parts(
    config: dict[str, Any],
    *,
    contract_hash: str,
    joined_parts: Sequence[dict[str, Any]],
    thresholds: pd.DataFrame,
    threshold_sha256: str,
) -> dict[str, Any]:
    decision_root = TRIM_ROOT / "decision_parts"
    main_root = MAIN_ROOT / "parts"
    sidecar_root = RUN_ROOT / "c4_2_trimmed_parts"
    output_parts: list[dict[str, Any]] = []
    for item in sorted(joined_parts, key=lambda value: int(value["task_id"])):
        task_id = int(item["task_id"])
        joined_path = CODING_ROOT / item["path"]
        decision_path = decision_root / f"part-{task_id:03d}.feather"
        main_path = main_root / f"part-{task_id:03d}.feather"
        sidecar_path = sidecar_root / f"part-{task_id:03d}.json"
        source_sha256 = item["sha256"]
        complete = (
            decision_path.is_file()
            and main_path.is_file()
            and sidecar_path.is_file()
        )
        if complete:
            sidecar = load_json(sidecar_path)
            complete = (
                sidecar.get("contract_hash") == contract_hash
                and sidecar.get("joined_sha256") == source_sha256
                and sidecar.get("threshold_sha256") == threshold_sha256
                and sidecar.get("decision_sha256") == sha256_file(decision_path)
                and sidecar.get("main_sha256") == sha256_file(main_path)
            )
        if not complete:
            joined = pd.read_feather(joined_path)
            decision = apply_quantile_trim(
                joined,
                thresholds,
                group_columns=config["normalized_time_trim"]["group_columns"],
            )
            main = decision.loc[decision["retain_after_c4"]].reset_index(drop=True)
            write_feather_atomic(decision_path, decision)
            write_feather_atomic(main_path, main)
            sidecar = {
                "task_id": task_id,
                "contract_hash": contract_hash,
                "joined_sha256": source_sha256,
                "threshold_sha256": threshold_sha256,
                "decision_sha256": sha256_file(decision_path),
                "main_sha256": sha256_file(main_path),
                "input_rows": int(len(decision)),
                "trimmed_rows": int(decision["trimmed"].sum()),
                "retained_rows": int(decision["retain_after_c4"].sum()),
            }
            write_json_atomic(sidecar_path, sidecar)
        else:
            sidecar = load_json(sidecar_path)
        output_parts.append(
            {
                "task_id": task_id,
                "decision_path": str(decision_path.relative_to(CODING_ROOT)),
                "main_path": str(main_path.relative_to(CODING_ROOT)),
                "input_rows": int(sidecar["input_rows"]),
                "trimmed_rows": int(sidecar["trimmed_rows"]),
                "retained_rows": int(sidecar["retained_rows"]),
                "decision_sha256": sidecar["decision_sha256"],
                "main_sha256": sidecar["main_sha256"],
            }
        )
    return {
        "analysis_year": 2021,
        "contract_hash": contract_hash,
        "threshold_sha256": threshold_sha256,
        "parts": output_parts,
        "input_rows": sum(item["input_rows"] for item in output_parts),
        "trimmed_rows": sum(item["trimmed_rows"] for item in output_parts),
        "retained_rows": sum(item["retained_rows"] for item in output_parts),
        "status": "pass",
    }


def render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# C4 2021 Main-Sample Construction",
            "",
            "## Contract",
            "",
            "- The C3 final decision manifest is the physical-rule boundary.",
            "- C4 adds no physical cap, W term, or standalone speed gate.",
            "- q01/q99 trimming is strict and grouped by ship type and period.",
            "",
            "## Accounting",
            "",
            f"- C3-final input: {summary['c3_final_rows']:,}.",
            f"- Outside 2021 departure-period scope: "
            f"{summary['temporal_scope_excluded_rows']:,}.",
            f"- Additional cap exclusions: {summary['cap_excluded_rows']:,}.",
            f"- q-trim exclusions: {summary['trimmed_rows']:,}.",
            f"- Main-sample retained: {summary['retained_rows']:,}.",
            f"- Undefined normalized ratios retained: {summary['undefined_ratio_rows']:,}.",
            "",
            "The period-by-regime attrition table uses the C0 route-envelope",
            "`regime_used` field. It is not relabeled as Suez exposure.",
            "Corridor flags and resolved/unresolved time decomposition remain",
            "explicitly unavailable because they are not present in the frozen",
            "C2 retained candidate product.",
        ]
    ) + "\n"


def run(config_path: Path, *, stage: str) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    script_hash = sha256_file(Path(__file__).resolve())
    contract_hash = hashlib.sha256(
        json.dumps(
            {"config_hash": config_hash, "script_sha256": script_hash},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    c2_manifest = load_verified_c2_manifest(config)
    c3_manifest = load_verified_c3_manifest(config)
    yardstick_path = CODING_ROOT / config["yardstick_table"]
    if stage == "preflight":
        verify_manifest_hash(
            yardstick_path,
            config["yardstick_table_sha256"],
            "C0 yardstick table",
        )
        return {
            "analysis_year": 2021,
            "contract_hash": contract_hash,
            "c2_retained_rows": sum(
                int(item["retained_rows"]) for item in c2_items(c2_manifest)
            ),
            "c3_final_rows": sum(
                int(item["final_retained_candidates"])
                for item in c3_manifest["parts"]
            ),
            "c2_parts": len(c2_items(c2_manifest)),
            "c3_parts": len(c3_manifest["parts"]),
            "status": "preflight_complete",
        }
    c3_final = load_c3_final_decisions(
        c3_manifest, expected_rows=int(config["expected_c3_final_rows"])
    )
    yardstick = load_yardstick(
        yardstick_path, config["yardstick_table_sha256"]
    )
    joined_parts = prepare_joined_parts(
        config,
        contract_hash=contract_hash,
        c2_manifest=c2_manifest,
        c3_final=c3_final,
        yardstick=yardstick,
    )
    if stage == "prepare":
        return {
            "analysis_year": 2021,
            "contract_hash": contract_hash,
            "c3_final_rows": int(config["expected_c3_final_rows"]),
            "joined_rows": int(config["expected_c3_final_rows"]),
            "status": "prepare_complete",
        }

    joined_frames = read_joined_parts(joined_parts)
    trim_config = config["normalized_time_trim"]
    thresholds = compute_quantile_thresholds(
        joined_frames,
        group_columns=trim_config["group_columns"],
        lower_quantile=float(trim_config["lower_quantile"]),
        upper_quantile=float(trim_config["upper_quantile"]),
    )
    threshold_path = TRIM_ROOT / "quantile_thresholds_2021.csv"
    write_csv_atomic(threshold_path, thresholds)
    threshold_sha256 = sha256_file(threshold_path)
    write_json_atomic(
        TRIM_ROOT / "quantile_thresholds_2021.json",
        {
            "analysis_year": 2021,
            "contract_hash": contract_hash,
            "threshold_sha256": threshold_sha256,
            "rows": int(len(thresholds)),
            "path": str(threshold_path.relative_to(CODING_ROOT)),
        },
    )
    if stage == "trim":
        return {
            "analysis_year": 2021,
            "contract_hash": contract_hash,
            "threshold_rows": int(len(thresholds)),
            "status": "thresholds_complete",
        }

    trim_manifest = apply_trimmed_parts(
        config,
        contract_hash=contract_hash,
        joined_parts=joined_parts,
        thresholds=thresholds,
        threshold_sha256=threshold_sha256,
    )
    write_json_atomic(TRIM_ROOT / "trim_output_manifest.json", trim_manifest)
    decision_frames = [
        pd.read_feather(CODING_ROOT / item["decision_path"])
        for item in trim_manifest["parts"]
    ]
    decisions = pd.concat(decision_frames, ignore_index=True)
    aggregate = build_aggregate_panel(decisions)
    aggregate_path = MAIN_ROOT / "aggregate_panel_2021.csv"
    write_csv_atomic(aggregate_path, aggregate)
    attrition = build_attrition_by_period_regime(decisions)
    attrition_path = REPORT_ROOT / "c4_attrition_by_period_regime_2021.csv"
    write_csv_atomic(attrition_path, attrition)
    main_manifest = {
        "analysis_year": 2021,
        "contract_hash": contract_hash,
        "trim_manifest": trim_manifest,
        "aggregate_path": str(aggregate_path.relative_to(CODING_ROOT)),
        "aggregate_sha256": sha256_file(aggregate_path),
        "attrition_by_period_regime_path": str(
            attrition_path.relative_to(CODING_ROOT)
        ),
        "attrition_by_period_regime_sha256": sha256_file(attrition_path),
        "c3_final_rows": int(len(decisions)),
        "temporal_scope_excluded_rows": int(
            decisions["temporal_scope_excluded"].sum()
        ),
        "cap_excluded_rows": int(decisions["cap_exceeded"].sum()),
        "trimmed_rows": int(decisions["trimmed"].sum()),
        "retained_rows": int(decisions["retain_after_c4"].sum()),
        "undefined_ratio_rows": int(decisions["normalized_time_ratio"].isna().sum()),
        "status": "pass",
    }
    write_json_atomic(MAIN_ROOT / "main_sample_output_manifest.json", main_manifest)
    write_text_atomic(REPORT_ROOT / "c4_main_sample_2021.md", render_report(main_manifest))
    write_json_atomic(RUN_ROOT / "run_summary.json", main_manifest)
    return main_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="C4 JSON configuration"
    )
    parser.add_argument(
        "--stage",
        choices=["preflight", "prepare", "trim", "all"],
        default="all",
        help="Run only the joined-input stage, threshold stage, or full C4",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.config, stage=arguments.stage)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
