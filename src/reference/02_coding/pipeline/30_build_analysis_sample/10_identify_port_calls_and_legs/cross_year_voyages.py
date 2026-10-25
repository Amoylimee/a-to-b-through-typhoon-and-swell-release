"""Pair accepted annual port visits across a calendar boundary.

The AIS state machine deliberately remains annual.  This module starts from its
accepted port-visit products, appends year ``Y + 1`` visits while pairing, and
retains only voyages whose departure falls in ``Y``.  Existing annual voyage rows
are preserved exactly; only the newly observable boundary pair is appended.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


VISIT_REQUIRED_COLUMNS = [
    "year",
    "vessel_uid",
    "registry_imo_clean",
    "visit_id_internal",
    "anchorage_id",
    "anchorage_iso3",
    "anchorage_label",
    "anchorage_sublabel",
    "analytical_city_id",
    "city_name",
    "entry_time",
    "exit_time",
    "visit_confidence",
]

VOYAGE_OUTPUT_COLUMNS = [
    "year", "vessel_uid", "registry_imo_clean",
    "internal_voyage_id", "voyage_sequence",
    "trip_start", "trip_end", "duration_hours", "duration_days",
    "origin_visit_id_internal", "dest_visit_id_internal",
    "origin_anchorage_id", "dest_anchorage_id",
    "origin_iso3", "dest_iso3",
    "origin_label", "dest_label",
    "origin_sublabel", "dest_sublabel",
    "origin_analytical_city_id", "dest_analytical_city_id",
    "origin_city_name", "dest_city_name",
    "od_pair", "city_pair", "country_pair", "is_international",
    "shiptype_group", "size_bin",
    "source_pipeline_generation",
]

VESSEL_META_COLUMNS = [
    "vessel_uid",
    "registry_imo_clean",
    "shiptype_group",
    "size_bin",
]

_PAIR_ENDPOINT_COLUMNS = [
    "entry_time",
    "visit_id_internal",
    "anchorage_id",
    "anchorage_iso3",
    "anchorage_label",
    "anchorage_sublabel",
    "analytical_city_id",
    "city_name",
    "_visit_source_year",
]


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _assert_single_annual_value(
    frame: pd.DataFrame,
    expected_year: int,
    label: str,
) -> None:
    observed = set(pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int))
    if observed != {expected_year}:
        raise ValueError(
            f"{label} must contain exactly year {expected_year}; observed {sorted(observed)}"
        )


def namespace_internal_visit_id(value: Any, source_year: int) -> Any:
    """Namespace one nonblank annual visit ID without changing missing values."""

    if pd.isna(value) or str(value).strip() == "":
        return value
    text = str(value)
    if len(text) >= 5 and text[:4].isdigit() and text[4] == ":":
        return text
    return f"{source_year}:{text}"


def enumerate_departure_year_pairs(
    current_visits: pd.DataFrame,
    next_visits: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Append next-year visits, pair adjacent high-confidence calls, keep Y starts.

    ``_voyage_sequence`` is assigned after sorting the concatenated high-confidence
    stream.  Because every high-confidence visit has an exit, appending next-year
    rows must not move any current-year sequence.  The caller proves that property
    against the accepted annual voyage product before writing output.
    """

    _require_columns(current_visits, VISIT_REQUIRED_COLUMNS, "current-year visits")
    _require_columns(next_visits, VISIT_REQUIRED_COLUMNS, "next-year visits")
    _assert_single_annual_value(current_visits, year, "current-year visits")
    _assert_single_annual_value(next_visits, year + 1, "next-year visits")

    frames = []
    for source_year, frame in ((year, current_visits), (year + 1, next_visits)):
        high = frame.loc[
            frame["visit_confidence"].eq("high"), VISIT_REQUIRED_COLUMNS
        ].copy()
        high["vessel_uid"] = high["vessel_uid"].astype(str)
        high["entry_time"] = pd.to_datetime(high["entry_time"], errors="coerce")
        high["exit_time"] = pd.to_datetime(high["exit_time"], errors="coerce")
        if high[["entry_time", "exit_time"]].isna().any(axis=None):
            raise ValueError(
                f"high-confidence visits for {source_year} contain missing entry/exit times"
            )
        high["_visit_source_year"] = source_year
        frames.append(high)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["vessel_uid", "exit_time"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)
    combined["_voyage_sequence"] = (
        combined.groupby("vessel_uid", sort=False).cumcount() + 1
    )

    grouped = combined.groupby("vessel_uid", sort=False)
    destination = grouped[_PAIR_ENDPOINT_COLUMNS].shift(-1)
    destination = destination.rename(
        columns={column: f"_dest_{column}" for column in _PAIR_ENDPOINT_COLUMNS}
    )
    paired = pd.concat([combined, destination], axis=1)

    start = paired["exit_time"]
    end = paired["_dest_entry_time"]
    start_floor = pd.Timestamp(year=year, month=1, day=1)
    start_ceiling = pd.Timestamp(year=year + 1, month=1, day=1)
    keep = (
        paired["_visit_source_year"].eq(year)
        & start.ge(start_floor)
        & start.lt(start_ceiling)
        & end.notna()
        & end.gt(start)
    )
    paired = paired.loc[keep].copy().reset_index(drop=True)
    paired["internal_voyage_id"] = (
        str(year)
        + "_"
        + paired["vessel_uid"].astype(str)
        + "_"
        + paired["_voyage_sequence"].astype(int).astype(str).str.zfill(4)
    )
    if paired["internal_voyage_id"].duplicated().any():
        raise ValueError("buffered pairing produced duplicate internal_voyage_id values")
    return paired


def assert_existing_voyage_identity(
    baseline_voyages: pd.DataFrame,
    paired: pd.DataFrame,
    year: int,
) -> dict[str, Any]:
    """Prove every accepted annual ID retains its exact start and end timestamp."""

    _require_columns(
        baseline_voyages,
        ["internal_voyage_id", "trip_start", "trip_end"],
        "baseline voyages",
    )
    if baseline_voyages["internal_voyage_id"].duplicated().any():
        raise ValueError("baseline voyages contain duplicate internal_voyage_id values")

    expected = paired.loc[
        paired["_dest__visit_source_year"].eq(year),
        ["internal_voyage_id", "exit_time", "_dest_entry_time"],
    ].rename(columns={"exit_time": "trip_start", "_dest_entry_time": "trip_end"})
    if expected["internal_voyage_id"].duplicated().any():
        raise ValueError("regenerated within-year pairs contain duplicate voyage IDs")

    old_ids = set(baseline_voyages["internal_voyage_id"].astype(str))
    regenerated_ids = set(expected["internal_voyage_id"].astype(str))
    missing = sorted(old_ids - regenerated_ids)
    extra = sorted(regenerated_ids - old_ids)
    if missing or extra:
        raise ValueError(
            "existing voyage ID stability failed: "
            f"missing={len(missing)} sample={missing[:5]}; "
            f"extra={len(extra)} sample={extra[:5]}"
        )

    old = baseline_voyages[
        ["internal_voyage_id", "trip_start", "trip_end"]
    ].copy()
    old["trip_start"] = pd.to_datetime(old["trip_start"], errors="coerce")
    old["trip_end"] = pd.to_datetime(old["trip_end"], errors="coerce")
    checked = old.merge(
        expected,
        on="internal_voyage_id",
        how="inner",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    start_mismatch = checked["trip_start_old"].ne(checked["trip_start_new"])
    end_mismatch = checked["trip_end_old"].ne(checked["trip_end_new"])
    if start_mismatch.any() or end_mismatch.any():
        bad = checked.loc[
            start_mismatch | end_mismatch,
            [
                "internal_voyage_id",
                "trip_start_old",
                "trip_start_new",
                "trip_end_old",
                "trip_end_new",
            ],
        ].head(5)
        raise ValueError(
            "existing voyage timestamp stability failed: "
            + bad.to_dict("records").__repr__()
        )
    return {
        "baseline_ids": len(old_ids),
        "regenerated_within_year_ids": len(regenerated_ids),
        "missing_ids": 0,
        "extra_ids": 0,
        "trip_start_mismatches": 0,
        "trip_end_mismatches": 0,
        "status": "pass",
    }


def _build_bridge_rows(
    paired: pd.DataFrame,
    vessel_meta: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    bridges = paired.loc[
        paired["_dest__visit_source_year"].eq(year + 1)
    ].copy()
    if bridges.empty:
        return pd.DataFrame(columns=VOYAGE_OUTPUT_COLUMNS)

    _require_columns(vessel_meta, VESSEL_META_COLUMNS, "vessel metadata")
    if vessel_meta["vessel_uid"].duplicated().any():
        raise ValueError("vessel metadata repeats vessel_uid")
    metadata = vessel_meta.copy()
    metadata["vessel_uid"] = metadata["vessel_uid"].astype(str)
    bridges = bridges.merge(
        metadata,
        on="vessel_uid",
        how="left",
        validate="many_to_one",
        suffixes=("", "_meta"),
    )
    if bridges[["shiptype_group", "size_bin"]].isna().any(axis=None):
        missing = bridges.loc[
            bridges[["shiptype_group", "size_bin"]].isna().any(axis=1),
            "vessel_uid",
        ].drop_duplicates().head(10).tolist()
        raise ValueError(f"boundary voyages lack accepted vessel metadata: {missing}")

    duration_hours = (
        bridges["_dest_entry_time"] - bridges["exit_time"]
    ) / np.timedelta64(1, "h")
    output = pd.DataFrame(
        {
            "year": year,
            "vessel_uid": bridges["vessel_uid"],
            "registry_imo_clean": bridges["registry_imo_clean_meta"],
            "internal_voyage_id": bridges["internal_voyage_id"],
            "voyage_sequence": bridges["_voyage_sequence"].astype(int),
            "trip_start": bridges["exit_time"],
            "trip_end": bridges["_dest_entry_time"],
            "duration_hours": duration_hours.astype(float),
            "duration_days": (duration_hours / 24.0).astype(float),
            "origin_visit_id_internal": [
                namespace_internal_visit_id(value, year)
                for value in bridges["visit_id_internal"]
            ],
            "dest_visit_id_internal": [
                namespace_internal_visit_id(value, year + 1)
                for value in bridges["_dest_visit_id_internal"]
            ],
            "origin_anchorage_id": bridges["anchorage_id"],
            "dest_anchorage_id": bridges["_dest_anchorage_id"],
            "origin_iso3": bridges["anchorage_iso3"],
            "dest_iso3": bridges["_dest_anchorage_iso3"],
            "origin_label": bridges["anchorage_label"],
            "dest_label": bridges["_dest_anchorage_label"],
            "origin_sublabel": bridges["anchorage_sublabel"],
            "dest_sublabel": bridges["_dest_anchorage_sublabel"],
            "origin_analytical_city_id": bridges["analytical_city_id"],
            "dest_analytical_city_id": bridges["_dest_analytical_city_id"],
            "origin_city_name": bridges["city_name"],
            "dest_city_name": bridges["_dest_city_name"],
            "od_pair": (
                bridges["anchorage_id"].astype(str)
                + "__"
                + bridges["_dest_anchorage_id"].astype(str)
            ),
            "city_pair": (
                bridges["analytical_city_id"].astype(str)
                + "__"
                + bridges["_dest_analytical_city_id"].astype(str)
            ),
            "country_pair": (
                bridges["anchorage_iso3"].astype(str)
                + "__"
                + bridges["_dest_anchorage_iso3"].astype(str)
            ),
            "is_international": bridges["anchorage_iso3"].ne(
                bridges["_dest_anchorage_iso3"]
            ),
            "shiptype_group": bridges["shiptype_group"],
            "size_bin": bridges["size_bin"],
            "source_pipeline_generation": "internal_harmonized_cross_year_buffered_v1",
        }
    )
    return output[VOYAGE_OUTPUT_COLUMNS].reset_index(drop=True)


def build_buffered_voyages(
    current_visits: pd.DataFrame,
    next_visits: pd.DataFrame,
    baseline_voyages: pd.DataFrame,
    vessel_meta: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return exact old annual rows plus newly observable cross-year rows."""

    _require_columns(baseline_voyages, VOYAGE_OUTPUT_COLUMNS, "baseline voyages")
    paired = enumerate_departure_year_pairs(current_visits, next_visits, year)
    identity = assert_existing_voyage_identity(baseline_voyages, paired, year)
    bridges = _build_bridge_rows(paired, vessel_meta, year)

    collisions = set(baseline_voyages["internal_voyage_id"].astype(str)) & set(
        bridges["internal_voyage_id"].astype(str)
    )
    if collisions:
        raise ValueError(
            "new boundary voyage IDs collide with accepted annual IDs: "
            f"{sorted(collisions)[:5]}"
        )

    output = pd.concat(
        [baseline_voyages[VOYAGE_OUTPUT_COLUMNS], bridges],
        ignore_index=True,
    )
    if output["internal_voyage_id"].duplicated().any():
        raise ValueError("buffered output contains duplicate internal_voyage_id values")
    pd.testing.assert_frame_equal(
        baseline_voyages[VOYAGE_OUTPUT_COLUMNS].reset_index(drop=True),
        output.iloc[: len(baseline_voyages)].reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )

    diagnostics = {
        "existing_voyage_identity": identity,
        "baseline_voyages": int(len(baseline_voyages)),
        "new_cross_year_voyages": int(len(bridges)),
        "buffered_voyages": int(len(output)),
        "cross_year_duration_hours_median": (
            float(bridges["duration_hours"].median()) if len(bridges) else None
        ),
        "cross_year_duration_hours_max": (
            float(bridges["duration_hours"].max()) if len(bridges) else None
        ),
        "cross_year_voyages_within_30_days": int(
            bridges["duration_hours"].le(30 * 24).sum()
        ),
    }
    return output, bridges, diagnostics
