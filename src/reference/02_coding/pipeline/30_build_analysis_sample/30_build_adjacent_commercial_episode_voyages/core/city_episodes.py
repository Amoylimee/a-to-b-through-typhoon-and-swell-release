"""Build clean city episodes and their adjacent directed-pair universe.

The implementation is intentionally independent of retired earlier code. It
uses the reviewed v3 mapping as frozen data and merges consecutive commercial
activity within the same topology city before enumerating adjacent endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_pairs import (
    ADJACENT_COMMERCIAL_EPISODES,
    destination_episode_indices,
    validate_candidate_enumeration_mode,
)


SCRIPT_PATH = Path(__file__).resolve()
C0_ROOT = SCRIPT_PATH.parents[1]
P1X_ROOT = SCRIPT_PATH.parents[1]
CODING_ROOT = SCRIPT_PATH.parents[4]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)

DEFAULT_CONFIG = C0_ROOT / "config/frozen/city_episodes_2021.json"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "main/provenance/city_episodes_2021"

OD_COLUMNS = [
    "mmsi",
    "vessel_id",
    "trip_id",
    "trip_start",
    "trip_end",
    "trip_start_anchorage_s2id",
    "trip_end_anchorage_s2id",
    "trip_start_visit_id",
    "trip_end_visit_id",
]

VESSEL_COLUMNS = [
    "mmsi_clean",
    "registry_imo_clean",
    "vessel_uid_clean",
    "vessel_name",
    "detailed_vessel_type_normalized",
    "ship_type_sample",
]

MAPPING_COLUMNS = [
    "anchorage_id",
    "country_iso3",
    "city_name",
    "raw_city_id",
    "analytical_city_id",
    "topology_city_id",
    "location_role",
    "endpoint_eligible",
    "role_review_status",
    "canal_id",
    "canal_event_type",
    "canal_side",
    "canal_crossing_evidence",
]

ANCHORAGE_SENTINELS = {"10000001", "NO_PREVIOUS_DATA"}


class DataContractError(RuntimeError):
    """Raised when an approved C0A hard gate fails."""


@dataclass
class BuildResult:
    excluded_trips: pd.DataFrame
    source_identity_decisions: pd.DataFrame
    endpoint_events: pd.DataFrame
    calls: pd.DataFrame
    episodes: pd.DataFrame
    pair_manifest: pd.DataFrame
    diagnostics: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*tokens: object, length: int = 20) -> str:
    payload = "|".join(str(token) for token in tokens)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalize_token(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def parse_utc_timestamp(series: pd.Series) -> pd.Series:
    """Parse the two approved OD formats: whole or fractional UTC seconds."""
    values = normalize_token(series)
    has_fraction = values.str.contains(r"\.\d+ UTC$", regex=True)
    normalized = values.where(
        has_fraction,
        values.str.replace(r" UTC$", ".000000 UTC", regex=True),
    )
    return pd.to_datetime(
        normalized,
        format="%Y-%m-%d %H:%M:%S.%f UTC",
        utc=True,
        errors="coerce",
    )


def parse_boolean(series: pd.Series, name: str) -> pd.Series:
    normalized = normalize_token(series).str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        values = sorted(normalized.loc[invalid].unique())[:10]
        raise DataContractError(f"{name} contains invalid booleans: {values}")
    return normalized.isin({"true", "1"})


def source_sentinel_reasons(trips: pd.DataFrame) -> pd.Series:
    timestamp_sentinel = trips["trip_start"].str.startswith("0001-") | trips[
        "trip_end"
    ].str.startswith("0001-")
    anchorage_sentinel = trips["trip_start_anchorage_s2id"].isin(
        ANCHORAGE_SENTINELS
    ) | trips["trip_end_anchorage_s2id"].isin(ANCHORAGE_SENTINELS)
    reasons = pd.Series("", index=trips.index, dtype="object")
    reasons.loc[timestamp_sentinel & ~anchorage_sentinel] = (
        "source_timestamp_sentinel"
    )
    reasons.loc[~timestamp_sentinel & anchorage_sentinel] = (
        "source_anchorage_sentinel"
    )
    reasons.loc[timestamp_sentinel & anchorage_sentinel] = (
        "source_timestamp_and_anchorage_sentinel"
    )
    return reasons


def count_internal_trip_overlaps(trips: pd.DataFrame) -> int:
    ordered = trips.sort_values(["trip_start", "trip_end", "trip_id"])
    prior_end = ordered["trip_end"].cummax().shift()
    return int((ordered["trip_start"] < prior_end).sum())


def resolve_source_identity_overlaps(
    trips: pd.DataFrame,
    anchorage_country: pd.Series,
    analysis_year: int,
    rule_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select one coherent source track only when source tracks overlap.

    The approved rule first prefers the track with more international trips,
    preserving the sample-definition signal, then more valid trips. Ties are
    resolved by more active endpoint months in the analysis year, shorter
    maximum trip duration, wider in-year coverage, then source ID. Strictly
    sequential source tracks are retained together.
    """
    if rule_name != "coherent_source_track_v1":
        raise DataContractError(f"Unsupported source-identity rule: {rule_name}")

    source_counts = trips.groupby("mmsi", sort=True)["vessel_id"].nunique()
    multi_source_mmsi = set(source_counts[source_counts > 1].index)
    affected_mmsi: set[str] = set()
    overlapping_source_activity_windows = 0

    for mmsi, vessel_trips in trips.loc[
        trips["mmsi"].isin(multi_source_mmsi)
    ].groupby("mmsi", sort=True):
        track_windows = {
            source_id: (
                source_trips["trip_start"].min(),
                source_trips["trip_end"].max(),
            )
            for source_id, source_trips in vessel_trips.groupby(
                "vessel_id", sort=True
            )
        }
        for source_a, source_b in itertools.combinations(
            sorted(track_windows), 2
        ):
            start_a, end_a = track_windows[source_a]
            start_b, end_b = track_windows[source_b]
            if start_a < end_b and start_b < end_a:
                affected_mmsi.add(str(mmsi))
                overlapping_source_activity_windows += 1

    year_start = pd.Timestamp(f"{analysis_year}-01-01", tz="UTC")
    year_end = pd.Timestamp(f"{analysis_year + 1}-01-01", tz="UTC")
    decision_rows: list[dict[str, Any]] = []
    selected_source_by_mmsi: dict[str, str] = {}

    for mmsi, vessel_trips in trips.loc[
        trips["mmsi"].isin(affected_mmsi)
    ].groupby("mmsi", sort=True):
        candidates: list[dict[str, Any]] = []
        for source_id, source_trips in vessel_trips.groupby(
            "vessel_id", sort=True
        ):
            internal_overlap_rows = count_internal_trip_overlaps(source_trips)
            if internal_overlap_rows:
                raise DataContractError(
                    "A candidate source track has internally overlapping trips: "
                    f"mmsi={mmsi}, vessel_id={source_id}, "
                    f"overlap_rows={internal_overlap_rows}"
                )
            start_months = set(
                source_trips.loc[
                    source_trips["trip_start"].between(
                        year_start, year_end, inclusive="left"
                    ),
                    "trip_start",
                ].dt.strftime("%Y-%m")
            )
            end_months = set(
                source_trips.loc[
                    source_trips["trip_end"].between(
                        year_start, year_end, inclusive="left"
                    ),
                    "trip_end",
                ].dt.strftime("%Y-%m")
            )
            coverage_start = max(source_trips["trip_start"].min(), year_start)
            coverage_end = min(source_trips["trip_end"].max(), year_end)
            coverage_hours = max(
                0.0, (coverage_end - coverage_start).total_seconds() / 3600.0
            )
            trip_hours = (
                source_trips["trip_end"] - source_trips["trip_start"]
            ).dt.total_seconds() / 3600.0
            start_country = source_trips[
                "trip_start_anchorage_s2id"
            ].map(anchorage_country)
            end_country = source_trips[
                "trip_end_anchorage_s2id"
            ].map(anchorage_country)
            international = (
                start_country.notna()
                & end_country.notna()
                & start_country.ne(end_country)
            )
            candidates.append(
                {
                    "mmsi": str(mmsi),
                    "vessel_id": str(source_id),
                    "n_international_trips": int(international.sum()),
                    "n_trips": int(len(source_trips)),
                    "active_endpoint_months_2021": int(
                        len(start_months | end_months)
                    ),
                    "max_trip_hours": float(trip_hours.max()),
                    "coverage_hours_2021": float(coverage_hours),
                    "first_trip_start": source_trips["trip_start"].min(),
                    "last_trip_end": source_trips["trip_end"].max(),
                    "internal_overlap_rows": 0,
                }
            )
        ranked = pd.DataFrame(candidates).sort_values(
            [
                "n_international_trips",
                "n_trips",
                "active_endpoint_months_2021",
                "max_trip_hours",
                "coverage_hours_2021",
                "vessel_id",
            ],
            ascending=[False, False, False, True, False, True],
            kind="mergesort",
        )
        ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
        selected_source = str(ranked.iloc[0]["vessel_id"])
        selected_source_by_mmsi[str(mmsi)] = selected_source
        ranked["selected"] = ranked["vessel_id"].eq(selected_source)
        ranked["rule_name"] = rule_name
        decision_rows.extend(ranked.to_dict(orient="records"))

    decisions = pd.DataFrame(decision_rows)
    if decisions.empty:
        decisions = pd.DataFrame(
            columns=[
                "mmsi",
                "vessel_id",
                "n_international_trips",
                "n_trips",
                "active_endpoint_months_2021",
                "max_trip_hours",
                "coverage_hours_2021",
                "first_trip_start",
                "last_trip_end",
                "internal_overlap_rows",
                "selection_rank",
                "selected",
                "rule_name",
            ]
        )

    selected_source = trips["mmsi"].map(selected_source_by_mmsi)
    excluded_mask = selected_source.notna() & trips["vessel_id"].ne(selected_source)
    exclusion_columns = [
        "mmsi",
        "vessel_id",
        "trip_id",
        "trip_start",
        "trip_end",
        "source_file",
        "source_row",
    ]
    excluded = trips.loc[excluded_mask, exclusion_columns].copy()
    excluded["exclusion_reason"] = "source_identity_nonselected_track"
    retained = trips.loc[~excluded_mask].copy().reset_index(drop=True)

    residual_overlap_windows = 0
    retained_multi = retained.groupby("mmsi", sort=True)["vessel_id"].nunique()
    retained_multi_mmsi = set(retained_multi[retained_multi > 1].index)
    for _, vessel_trips in retained.loc[
        retained["mmsi"].isin(retained_multi_mmsi)
    ].groupby("mmsi", sort=True):
        track_windows = {
            source_id: (
                source_trips["trip_start"].min(),
                source_trips["trip_end"].max(),
            )
            for source_id, source_trips in vessel_trips.groupby(
                "vessel_id", sort=True
            )
        }
        for source_a, source_b in itertools.combinations(
            sorted(track_windows), 2
        ):
            start_a, end_a = track_windows[source_a]
            start_b, end_b = track_windows[source_b]
            if start_a < end_b and start_b < end_a:
                residual_overlap_windows += 1
    if residual_overlap_windows:
        raise DataContractError(
            "Source-identity resolution left overlapping source activity "
            f"windows: {residual_overlap_windows} source pairs"
        )

    diagnostics = {
        "source_identity_rule": rule_name,
        "multiple_source_id_vessels": int(len(multi_source_mmsi)),
        "overlapping_source_activity_window_vessels": int(len(affected_mmsi)),
        "overlapping_source_activity_window_pairs": int(
            overlapping_source_activity_windows
        ),
        "source_tracks_ranked": int(len(decisions)),
        "source_tracks_selected": int(
            decisions["selected"].sum() if not decisions.empty else 0
        ),
        "source_identity_trips_excluded": int(len(excluded)),
        "unique_trips": int(len(retained)),
        "analysis_mmsi_after_source_identity_resolution": int(
            retained["mmsi"].nunique()
        ),
        "residual_overlapping_source_activity_windows": 0,
    }
    return retained, excluded, decisions, diagnostics


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in config["input_sha256"].items():
        path = CODING_ROOT / relative
        if not path.is_file():
            raise DataContractError(f"Missing frozen input: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise DataContractError(
                f"Frozen input hash mismatch for {relative}: "
                f"{actual_hash} != {expected_hash}"
            )
        stat = path.stat()
        manifest[relative] = {
            "sha256": actual_hash,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return manifest


def load_vessel_universe(path: Path, expected_count: int) -> pd.DataFrame:
    vessels = pd.read_csv(
        path,
        usecols=VESSEL_COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    for column in VESSEL_COLUMNS:
        vessels[column] = normalize_token(vessels[column])
    vessels = vessels.rename(columns={"mmsi_clean": "mmsi", "registry_imo_clean": "imo"})
    if len(vessels) != expected_count:
        raise DataContractError(
            f"Expected {expected_count} vessel rows, found {len(vessels)}"
        )
    if vessels["mmsi"].eq("").any() or vessels["imo"].eq("").any():
        raise DataContractError("Analysis-vessel specification has blank MMSI or IMO")
    if vessels["mmsi"].duplicated().any():
        examples = vessels.loc[vessels["mmsi"].duplicated(False), "mmsi"].head()
        raise DataContractError(f"Duplicate analysis MMSI values: {examples.tolist()}")
    if vessels["imo"].duplicated().any():
        examples = vessels.loc[vessels["imo"].duplicated(False), "imo"].head()
        raise DataContractError(f"Duplicate analysis IMO values: {examples.tolist()}")
    return vessels.sort_values("mmsi").reset_index(drop=True)


def load_v3_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(
        path,
        usecols=MAPPING_COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    for column in MAPPING_COLUMNS:
        mapping[column] = normalize_token(mapping[column])
    mapping["endpoint_eligible"] = parse_boolean(
        mapping["endpoint_eligible"], "endpoint_eligible"
    )
    if mapping["anchorage_id"].eq("").any():
        raise DataContractError("V3 mapping contains blank anchorage IDs")
    if mapping["anchorage_id"].duplicated().any():
        examples = mapping.loc[
            mapping["anchorage_id"].duplicated(False), "anchorage_id"
        ].head()
        raise DataContractError(f"Duplicate V3 anchorage IDs: {examples.tolist()}")
    invalid_endpoint = mapping["endpoint_eligible"] & mapping["topology_city_id"].eq("")
    if invalid_endpoint.any():
        raise DataContractError(
            "V3 mapping has endpoint-eligible rows with blank topology_city_id"
        )
    return mapping.sort_values("anchorage_id").reset_index(drop=True)


def load_filtered_od(
    od_paths: list[Path],
    analysis_mmsi: set[str],
    chunk_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    retained: list[pd.DataFrame] = []
    file_counts: list[dict[str, Any]] = []
    for path in sorted(od_paths):
        raw_rows = 0
        kept_rows = 0
        source_row = 2
        reader = pd.read_csv(
            path,
            usecols=OD_COLUMNS,
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_rows,
            low_memory=False,
        )
        for chunk in reader:
            n_rows = len(chunk)
            raw_rows += n_rows
            chunk["mmsi"] = normalize_token(chunk["mmsi"])
            mask = chunk["mmsi"].isin(analysis_mmsi)
            selected = chunk.loc[mask].copy()
            if not selected.empty:
                selected["source_file"] = path.name
                selected["source_row"] = np.arange(
                    source_row, source_row + n_rows, dtype=np.int64
                )[mask.to_numpy()]
                retained.append(selected)
                kept_rows += len(selected)
            source_row += n_rows
        file_counts.append(
            {
                "source_file": path.name,
                "raw_rows": raw_rows,
                "analysis_vessel_rows": kept_rows,
            }
        )
    if not retained:
        raise DataContractError("No 2021 OD rows matched the analysis-vessel universe")

    trips = pd.concat(retained, ignore_index=True)
    for column in OD_COLUMNS:
        trips[column] = normalize_token(trips[column])
    if trips["trip_id"].eq("").any():
        raise DataContractError("Retained OD data contain blank trip IDs")

    core_columns = [column for column in OD_COLUMNS if column != "trip_id"]
    duplicate_trip_rows = int(trips["trip_id"].duplicated(False).sum())
    unique_variants = trips.drop_duplicates(["trip_id", *core_columns])
    conflict_counts = unique_variants.groupby("trip_id", sort=False).size()
    conflict_trip_ids = conflict_counts[conflict_counts > 1].index
    if len(conflict_trip_ids):
        examples = list(conflict_trip_ids[:10])
        raise DataContractError(f"Conflicting duplicate trip IDs: {examples}")

    trips = (
        trips.sort_values(["trip_id", "source_file", "source_row"])
        .drop_duplicates("trip_id", keep="first")
        .reset_index(drop=True)
    )
    exclusion_reasons = source_sentinel_reasons(trips)
    source_sentinel = exclusion_reasons.ne("")
    excluded_trips = trips.loc[
        source_sentinel,
        [
            "mmsi",
            "vessel_id",
            "trip_id",
            "trip_start",
            "trip_end",
            "source_file",
            "source_row",
        ],
    ].copy()
    excluded_trips["exclusion_reason"] = exclusion_reasons.loc[source_sentinel]
    trips = trips.loc[~source_sentinel].copy()

    raw_trip_start = trips["trip_start"].copy()
    raw_trip_end = trips["trip_end"].copy()
    trips["trip_start"] = parse_utc_timestamp(raw_trip_start)
    trips["trip_end"] = parse_utc_timestamp(raw_trip_end)
    invalid_timestamp = trips["trip_start"].isna() | trips["trip_end"].isna()
    if invalid_timestamp.any():
        examples = pd.DataFrame(
            {
                "trip_start": raw_trip_start.loc[invalid_timestamp],
                "trip_end": raw_trip_end.loc[invalid_timestamp],
            }
        ).head(10)
        raise DataContractError(
            f"Invalid non-sentinel trip timestamps: "
            f"{int(invalid_timestamp.sum())} rows; "
            f"examples={examples.to_dict(orient='records')}"
        )
    reversed_trip = trips["trip_start"] > trips["trip_end"]
    if reversed_trip.any():
        raise DataContractError(
            f"Trip start is after trip end: {int(reversed_trip.sum())} rows"
        )
    blank_anchorages = trips[
        ["trip_start_anchorage_s2id", "trip_end_anchorage_s2id"]
    ].eq("").any(axis=1)
    if blank_anchorages.any():
        raise DataContractError(
            f"Retained trips with blank anchorage IDs: {int(blank_anchorages.sum())}"
        )

    diagnostics = {
        "source_files": file_counts,
        "raw_rows": int(sum(item["raw_rows"] for item in file_counts)),
        "analysis_vessel_rows_before_trip_dedup": int(
            sum(item["analysis_vessel_rows"] for item in file_counts)
        ),
        "duplicate_trip_rows": duplicate_trip_rows,
        "source_sentinel_trips_excluded": int(len(excluded_trips)),
        "source_sentinel_exclusions_by_reason": {
            str(reason): int(count)
            for reason, count in excluded_trips["exclusion_reason"]
            .value_counts()
            .sort_index()
            .items()
        },
        "unique_trips_before_source_identity_resolution": int(len(trips)),
        "analysis_mmsi_with_trips": int(trips["mmsi"].nunique()),
    }
    return trips, excluded_trips, diagnostics


def build_endpoint_events(
    trips: pd.DataFrame, vessels: pd.DataFrame
) -> pd.DataFrame:
    vessel_fields = vessels[
        [
            "mmsi",
            "imo",
            "vessel_uid_clean",
            "vessel_name",
            "detailed_vessel_type_normalized",
            "ship_type_sample",
        ]
    ]
    enriched = trips.merge(
        vessel_fields, on="mmsi", how="left", validate="many_to_one"
    )
    if enriched["imo"].isna().any():
        raise DataContractError("Retained trip failed the analysis-vessel join")

    common = [
        "mmsi",
        "imo",
        "vessel_uid_clean",
        "vessel_name",
        "detailed_vessel_type_normalized",
        "ship_type_sample",
        "vessel_id",
        "trip_id",
        "source_file",
        "source_row",
    ]
    departures = enriched[common].copy()
    departures["event_type"] = "departure"
    departures["event_time"] = enriched["trip_start"]
    departures["anchorage_id"] = enriched["trip_start_anchorage_s2id"]
    departures["visit_id"] = enriched["trip_start_visit_id"]

    arrivals = enriched[common].copy()
    arrivals["event_type"] = "arrival"
    arrivals["event_time"] = enriched["trip_end"]
    arrivals["anchorage_id"] = enriched["trip_end_anchorage_s2id"]
    arrivals["visit_id"] = enriched["trip_end_visit_id"]

    events = pd.concat([arrivals, departures], ignore_index=True)
    events["visit_id_missing"] = events["visit_id"].eq("")
    fallback = (
        "__MISSING_VISIT__"
        + events["trip_id"]
        + "__"
        + events["event_type"]
    )
    events["visit_key"] = events["visit_id"].where(
        ~events["visit_id_missing"], fallback
    )
    events["call_key"] = (
        events["mmsi"]
        + "|"
        + events["visit_key"]
        + "|"
        + events["anchorage_id"]
    )
    events["call_id"] = [
        stable_id(key, length=24) for key in events["call_key"].tolist()
    ]
    events["event_id"] = (
        events["trip_id"] + "|" + events["event_type"].str.slice(0, 1)
    )
    events["event_order"] = events["event_type"].map(
        {"arrival": 0, "departure": 1}
    )
    events = events.sort_values(
        ["mmsi", "event_time", "event_order", "event_id", "source_file", "source_row"]
    ).reset_index(drop=True)
    events["event_sequence"] = events.groupby("mmsi", sort=False).cumcount()
    return events


def build_calls(
    events: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    working = events.copy()
    working["arrival_time_value"] = working["event_time"].where(
        working["event_type"].eq("arrival")
    )
    working["departure_time_value"] = working["event_time"].where(
        working["event_type"].eq("departure")
    )
    working["is_arrival"] = working["event_type"].eq("arrival").astype(np.int8)
    working["is_departure"] = working["event_type"].eq("departure").astype(np.int8)

    calls = (
        working.groupby("call_id", sort=True, observed=True)
        .agg(
            call_key=("call_key", "first"),
            mmsi=("mmsi", "first"),
            imo=("imo", "first"),
            vessel_uid_clean=("vessel_uid_clean", "first"),
            vessel_name=("vessel_name", "first"),
            detailed_vessel_type_normalized=(
                "detailed_vessel_type_normalized",
                "first",
            ),
            ship_type_sample=("ship_type_sample", "first"),
            vessel_id=("vessel_id", "first"),
            vessel_id_count=("vessel_id", "nunique"),
            visit_id=("visit_id", "first"),
            visit_id_missing=("visit_id_missing", "first"),
            anchorage_id=("anchorage_id", "first"),
            arrival_time=("arrival_time_value", "min"),
            departure_time=("departure_time_value", "max"),
            event_start=("event_time", "min"),
            event_end=("event_time", "max"),
            n_events=("event_id", "size"),
            n_arrivals=("is_arrival", "sum"),
            n_departures=("is_departure", "sum"),
            first_source_file=("source_file", "first"),
            first_source_row=("source_row", "min"),
        )
        .reset_index()
    )
    if (calls["vessel_id_count"] > 1).any():
        raise DataContractError("A call key maps to multiple vessel_id values")

    mapping_fields = mapping[
        [
            "anchorage_id",
            "country_iso3",
            "city_name",
            "raw_city_id",
            "analytical_city_id",
            "topology_city_id",
            "location_role",
            "endpoint_eligible",
            "role_review_status",
            "canal_id",
            "canal_event_type",
            "canal_side",
            "canal_crossing_evidence",
        ]
    ]
    calls = calls.merge(
        mapping_fields, on="anchorage_id", how="left", validate="many_to_one"
    )
    missing_mapping = calls["topology_city_id"].isna()
    missing_rows = calls.loc[
        missing_mapping,
        ["anchorage_id", "mmsi", "imo", "call_id", "event_start", "event_end"],
    ].copy()
    if missing_mapping.any():
        return calls, missing_rows, {
            "missing_mapping_calls": int(missing_mapping.sum()),
            "missing_mapping_anchorages": int(
                calls.loc[missing_mapping, "anchorage_id"].nunique()
            ),
        }

    calls["is_commercial_endpoint"] = (
        calls["endpoint_eligible"].astype(bool)
        & calls["topology_city_id"].ne("")
    )
    reversed_call = (
        calls["arrival_time"].notna()
        & calls["departure_time"].notna()
        & (calls["arrival_time"] > calls["departure_time"])
    )
    if reversed_call.any():
        raise DataContractError(
            f"Call arrival occurs after departure: {int(reversed_call.sum())} calls"
        )

    calls = calls.sort_values(
        ["mmsi", "event_start", "event_end", "call_id"]
    ).reset_index(drop=True)
    calls["call_sequence"] = calls.groupby("mmsi", sort=False).cumcount()
    calls["previous_call_end"] = calls.groupby("mmsi", sort=False)[
        "event_end"
    ].shift()
    calls["overlaps_previous_call"] = (
        calls["previous_call_end"].notna()
        & (calls["event_start"] < calls["previous_call_end"])
    )
    diagnostics = {
        "missing_mapping_calls": 0,
        "missing_mapping_anchorages": 0,
        "port_calls": int(len(calls)),
        "commercial_endpoint_calls": int(calls["is_commercial_endpoint"].sum()),
        "technical_or_nonendpoint_calls": int(
            (~calls["is_commercial_endpoint"]).sum()
        ),
        "missing_visit_id_calls": int(calls["visit_id_missing"].sum()),
        "overlapping_call_count": int(calls["overlaps_previous_call"].sum()),
    }
    return calls, missing_rows, diagnostics


def merge_same_city_episodes(
    calls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calls = calls.copy()
    commercial = calls.loc[calls["is_commercial_endpoint"]].copy()
    if commercial.empty:
        raise DataContractError("No endpoint-eligible commercial calls remain")
    previous_city = commercial.groupby("mmsi", sort=False)["topology_city_id"].shift()
    commercial["starts_new_episode"] = (
        previous_city.isna()
        | commercial["topology_city_id"].ne(previous_city)
    )
    commercial["episode_sequence"] = (
        commercial.groupby("mmsi", sort=False)["starts_new_episode"].cumsum() - 1
    ).astype(np.int64)

    episode_keys = ["mmsi", "episode_sequence"]
    episodes = (
        commercial.groupby(episode_keys, sort=True, observed=True)
        .agg(
            imo=("imo", "first"),
            vessel_uid_clean=("vessel_uid_clean", "first"),
            vessel_name=("vessel_name", "first"),
            detailed_vessel_type_normalized=(
                "detailed_vessel_type_normalized",
                "first",
            ),
            ship_type_sample=("ship_type_sample", "first"),
            topology_city_id=("topology_city_id", "first"),
            country_iso3=("country_iso3", "first"),
            city_name=("city_name", "first"),
            arrival_time=("arrival_time", "min"),
            departure_time=("departure_time", "max"),
            episode_event_start=("event_start", "min"),
            episode_event_end=("event_end", "max"),
            first_call_sequence=("call_sequence", "min"),
            last_call_sequence=("call_sequence", "max"),
            n_commercial_calls=("call_id", "size"),
            n_distinct_anchorages=("anchorage_id", "nunique"),
            first_call_id=("call_id", "first"),
            last_call_id=("call_id", "last"),
        )
        .reset_index()
    )
    first_flags = commercial[
        episode_keys + ["call_id", "arrival_time", "departure_time"]
    ].drop_duplicates(episode_keys, keep="first")
    first_flags = first_flags.rename(
        columns={
            "call_id": "first_flag_call_id",
            "arrival_time": "first_call_arrival_time",
            "departure_time": "first_call_departure_time",
        }
    )
    last_flags = commercial[
        episode_keys + ["call_id", "arrival_time", "departure_time"]
    ].drop_duplicates(episode_keys, keep="last")
    last_flags = last_flags.rename(
        columns={
            "call_id": "last_flag_call_id",
            "arrival_time": "last_call_arrival_time",
            "departure_time": "last_call_departure_time",
        }
    )
    episodes = episodes.merge(
        first_flags, on=episode_keys, how="left", validate="one_to_one"
    ).merge(last_flags, on=episode_keys, how="left", validate="one_to_one")
    episodes["arrival_complete"] = episodes["first_call_arrival_time"].notna()
    episodes["departure_complete"] = episodes["last_call_departure_time"].notna()
    episodes["passage_arrival_time"] = episodes["arrival_time"].where(
        episodes["arrival_complete"]
    )
    episodes["passage_departure_time"] = episodes["departure_time"].where(
        episodes["departure_complete"]
    )
    episodes["episode_id"] = [
        f"{imo}_2021_{int(sequence):06d}_{stable_id(city, length=8)}"
        for imo, sequence, city in episodes[
            ["imo", "episode_sequence", "topology_city_id"]
        ].itertuples(index=False, name=None)
    ]

    commercial_assignment = commercial[
        ["call_id", "mmsi", "episode_sequence"]
    ].merge(
        episodes[["mmsi", "episode_sequence", "episode_id"]],
        on=["mmsi", "episode_sequence"],
        how="left",
        validate="many_to_one",
    )
    assignment = commercial_assignment.set_index("call_id")["episode_id"]
    calls["episode_id"] = calls["call_id"].map(assignment).fillna("")
    nonblank_episode = calls["episode_id"].replace("", pd.NA)
    previous_episode = nonblank_episode.groupby(calls["mmsi"], sort=False).ffill()
    next_episode = nonblank_episode.groupby(calls["mmsi"], sort=False).bfill()
    calls["technical_within_same_city_episode"] = (
        ~calls["is_commercial_endpoint"]
        & previous_episode.notna()
        & previous_episode.eq(next_episode)
    )
    calls.loc[
        calls["technical_within_same_city_episode"], "episode_id"
    ] = previous_episode.loc[calls["technical_within_same_city_episode"]]

    technical_counts = (
        calls.loc[calls["technical_within_same_city_episode"]]
        .groupby("episode_id", sort=True)
        .size()
        .rename("n_internal_technical_calls")
    )
    episodes = episodes.merge(
        technical_counts, left_on="episode_id", right_index=True, how="left"
    )
    episodes["n_internal_technical_calls"] = (
        episodes["n_internal_technical_calls"].fillna(0).astype(np.int64)
    )
    episodes = episodes.sort_values(["mmsi", "episode_sequence"]).reset_index(
        drop=True
    )

    adjacent_duplicate = episodes["topology_city_id"].eq(
        episodes.groupby("mmsi", sort=False)["topology_city_id"].shift()
    )
    if adjacent_duplicate.any():
        raise DataContractError(
            f"Adjacent duplicate city episodes remain: {int(adjacent_duplicate.sum())}"
        )
    if episodes["episode_id"].duplicated().any():
        raise DataContractError("Duplicate episode IDs were generated")

    diagnostics = {
        "city_episodes": int(len(episodes)),
        "episodes_with_multiple_commercial_calls": int(
            (episodes["n_commercial_calls"] > 1).sum()
        ),
        "commercial_calls_merged_by_same_city_rule": int(
            len(commercial) - len(episodes)
        ),
        "technical_calls_inside_same_city_episode": int(
            calls["technical_within_same_city_episode"].sum()
        ),
        "arrival_incomplete_episodes": int((~episodes["arrival_complete"]).sum()),
        "departure_incomplete_episodes": int(
            (~episodes["departure_complete"]).sum()
        ),
        "adjacent_duplicate_city_episodes": 0,
    }
    return calls, episodes, diagnostics


def enumerate_adjacent_episode_pairs(
    episodes: pd.DataFrame,
    candidate_enumeration_mode: str = ADJACENT_COMMERCIAL_EPISODES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_enumeration_mode = validate_candidate_enumeration_mode(
        candidate_enumeration_mode
    )
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    vessels_by_pair: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    first_departure: dict[tuple[str, str], pd.Timestamp] = {}
    last_arrival: dict[tuple[str, str], pd.Timestamp] = {}
    examples: dict[tuple[str, str], tuple[str, str]] = {}
    invalid_time_examples: list[dict[str, Any]] = []
    candidate_count = 0

    for _, vessel_episodes in episodes.groupby("mmsi", sort=True):
        ordered = vessel_episodes.sort_values("episode_sequence")
        cities = ordered["topology_city_id"].tolist()
        departures = ordered["passage_departure_time"].tolist()
        arrivals = ordered["passage_arrival_time"].tolist()
        episode_ids = ordered["episode_id"].tolist()
        imo = str(ordered["imo"].iloc[0])
        for origin_index, origin_city in enumerate(cities):
            departure_time = departures[origin_index]
            if pd.isna(departure_time):
                continue
            for destination_index in destination_episode_indices(
                cities,
                origin_index,
                candidate_enumeration_mode,
            ):
                destination_city = cities[destination_index]
                arrival_time = arrivals[destination_index]
                if pd.isna(arrival_time):
                    continue
                if arrival_time <= departure_time:
                    if len(invalid_time_examples) < 100:
                        invalid_time_examples.append(
                            {
                                "imo": imo,
                                "origin_episode_id": episode_ids[origin_index],
                                "dest_episode_id": episode_ids[destination_index],
                                "origin_city": origin_city,
                                "dest_city": destination_city,
                                "departure_time": departure_time.isoformat(),
                                "arrival_time": arrival_time.isoformat(),
                            }
                        )
                    continue
                key = (origin_city, destination_city)
                counts[key] += 1
                vessels_by_pair[key].add(imo)
                candidate_count += 1
                if key not in first_departure or departure_time < first_departure[key]:
                    first_departure[key] = departure_time
                if key not in last_arrival or arrival_time > last_arrival[key]:
                    last_arrival[key] = arrival_time
                examples.setdefault(
                    key,
                    (episode_ids[origin_index], episode_ids[destination_index]),
                )

    if invalid_time_examples:
        raise DataContractError(
            "Adjacent-episode enumeration found non-positive elapsed times; "
            f"examples={invalid_time_examples[:3]}"
        )

    rows: list[dict[str, Any]] = []
    for origin_city, destination_city in sorted(counts):
        key = (origin_city, destination_city)
        origin_episode_id, dest_episode_id = examples[key]
        rows.append(
            {
                "pair_id": f"{origin_city}__{destination_city}",
                "origin_city": origin_city,
                "dest_city": destination_city,
                "candidate_count_2021": counts[key],
                "vessel_count_2021": len(vessels_by_pair[key]),
                "first_departure_time": first_departure[key],
                "last_arrival_time": last_arrival[key],
                "example_origin_episode_id": origin_episode_id,
                "example_dest_episode_id": dest_episode_id,
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise DataContractError("No valid adjacent-episode pairs were enumerated")
    if manifest["pair_id"].duplicated().any():
        raise DataContractError("Duplicate pair IDs were generated")
    self_pairs = manifest["origin_city"].eq(manifest["dest_city"])
    if self_pairs.any():
        raise DataContractError("Origin-equals-destination pair entered the manifest")
    diagnostics = {
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "candidate_pair_count": int(len(manifest)),
        "candidate_occurrences": int(candidate_count),
        "adjacent_episode_pair_count": int(len(manifest)),
        "adjacent_episode_candidate_occurrences": int(candidate_count),
        "invalid_nonpositive_candidate_times": 0,
    }
    return manifest, diagnostics


def write_pair_manifest_metadata(
    diagnostics: dict[str, Any],
    pair_manifest_path: Path,
    metadata_path: Path,
) -> None:
    payload = {
        "status": "frozen",
        "schema_version": "p1x_c0a_pair_manifest_v1",
        "analysis_year": int(diagnostics["analysis_year"]),
        "pair_manifest": {
            "filename": pair_manifest_path.name,
            "sha256": sha256_file(pair_manifest_path),
            "row_count": int(diagnostics["adjacent_episode_pair_count"]),
            "key": "pair_id",
            "origin_field": "origin_city",
            "destination_field": "dest_city",
        },
        "construction_scope": {
            "rule": "adjacent commercial episode endpoint pairing",
            "same_city_activity_merged": True,
            "origin_equals_destination_allowed": False,
            "r2_r3_r4_r5_applied": False,
            "cap_or_trim_applied": False,
            "ais_used": False,
            "outcomes_or_regressions_used": False,
        },
        "c1_replay_contract": {
            "require_exact_distinct_pair_id_set": True,
            "expected_distinct_pair_count": int(
                diagnostics["adjacent_episode_pair_count"]
            ),
            "comparison_stage": "C1 after rebuilding full candidate records",
            "mismatch_action": "stop_and_report",
        },
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(
    config: dict[str, Any],
    verify_hashes: bool = True,
) -> tuple[BuildResult, dict[str, dict[str, Any]]]:
    input_manifest = verify_frozen_inputs(config) if verify_hashes else {}
    vessel_path = CODING_ROOT / (
        "data/P1_prepare_od_panel/p4_build_analysis_vessel_specs/"
        "analysis_vessel_specs_2021_2021_2025_present_2022_2024.csv"
    )
    mapping_path = CODING_ROOT / (
        "data/named_anchorages/anchorage_to_analytical_city_candidate_v3.csv"
    )
    od_paths = sorted((CODING_ROOT / "data/od/2021").glob("voyages_2021??.csv"))
    if len(od_paths) != 12:
        raise DataContractError(f"Expected 12 monthly OD files, found {len(od_paths)}")

    vessels = load_vessel_universe(
        vessel_path, int(config["expected_analysis_vessels"])
    )
    mapping = load_v3_mapping(mapping_path)
    trips, excluded_trips, od_diagnostics = load_filtered_od(
        od_paths,
        set(vessels["mmsi"]),
        int(config["od_chunk_rows"]),
    )
    source_rule = config.get("source_identity_resolution", {}).get("rule_name", "")
    (
        trips,
        source_identity_exclusions,
        source_identity_decisions,
        source_identity_diagnostics,
    ) = resolve_source_identity_overlaps(
        trips,
        mapping.set_index("anchorage_id")["country_iso3"],
        int(config["analysis_year"]),
        str(source_rule),
    )
    excluded_trips = pd.concat(
        [excluded_trips, source_identity_exclusions],
        ignore_index=True,
    )
    source_identity_decisions = source_identity_decisions.merge(
        vessels[["mmsi", "imo", "vessel_name"]],
        on="mmsi",
        how="left",
        validate="many_to_one",
    )
    endpoint_events = build_endpoint_events(trips, vessels)
    calls, missing_mapping, call_diagnostics = build_calls(endpoint_events, mapping)
    if not missing_mapping.empty:
        examples = missing_mapping["anchorage_id"].drop_duplicates().head(10).tolist()
        raise DataContractError(
            "V3 mapping does not cover retained OD anchorages: "
            f"{call_diagnostics['missing_mapping_anchorages']} unique; "
            f"examples={examples}"
        )
    calls, episodes, episode_diagnostics = merge_same_city_episodes(calls)
    pair_manifest, pair_diagnostics = enumerate_adjacent_episode_pairs(episodes)
    diagnostics = {
        "analysis_year": int(config["analysis_year"]),
        "analysis_vessels": int(len(vessels)),
        "v3_mapping_rows": int(len(mapping)),
        "endpoint_events": int(len(endpoint_events)),
        **od_diagnostics,
        **source_identity_diagnostics,
        **call_diagnostics,
        **episode_diagnostics,
        **pair_diagnostics,
    }
    return (
        BuildResult(
            excluded_trips=excluded_trips,
            source_identity_decisions=source_identity_decisions,
            endpoint_events=endpoint_events,
            calls=calls,
            episodes=episodes,
            pair_manifest=pair_manifest,
            diagnostics=diagnostics,
        ),
        input_manifest,
    )


def write_outputs(
    result: BuildResult,
    input_manifest: dict[str, dict[str, Any]],
    output_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "excluded_trips": output_dir / "excluded_trip_records_2021.csv",
        "source_identity_decisions": (
            output_dir / "source_identity_resolution_2021.csv"
        ),
        "endpoint_events": output_dir / "endpoint_events_2021.feather",
        "calls": output_dir / "ordered_port_calls_2021.feather",
        "episodes": output_dir / "ordered_city_episodes_2021.feather",
        "pair_manifest": output_dir / "adjacent_episode_pair_manifest_2021.csv",
        "pair_manifest_metadata": (
            output_dir / "adjacent_episode_pair_manifest_2021.metadata.json"
        ),
        "input_manifest": output_dir / "input_manifest.json",
        "audit_json": output_dir / "c0a_lineage_and_schema_audit.json",
        "audit_md": output_dir / "c0a_lineage_and_schema_audit.md",
        "output_manifest": output_dir / "output_manifest.json",
    }
    result.excluded_trips.to_csv(paths["excluded_trips"], index=False)
    result.source_identity_decisions.to_csv(
        paths["source_identity_decisions"], index=False
    )
    result.endpoint_events.to_feather(
        paths["endpoint_events"], compression="zstd", compression_level=9
    )
    result.calls.to_feather(
        paths["calls"], compression="zstd", compression_level=9
    )
    result.episodes.to_feather(
        paths["episodes"], compression="zstd", compression_level=9
    )
    result.pair_manifest.to_csv(
        paths["pair_manifest"], index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
    )
    write_pair_manifest_metadata(
        result.diagnostics,
        paths["pair_manifest"],
        paths["pair_manifest_metadata"],
    )
    paths["input_manifest"].write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "status": "pass",
        "config": str(config_path.relative_to(CODING_ROOT)),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "diagnostics": result.diagnostics,
    }
    paths["audit_json"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    diagnostics = result.diagnostics
    report = f"""# C0A 2021 City-Episode Audit

Status: PASS

## Main Result

- Analysis vessels: {diagnostics['analysis_vessels']:,}
- Unique OD trips: {diagnostics['unique_trips']:,}
- Source-sentinel trips excluded: {diagnostics['source_sentinel_trips_excluded']:,}
- Vessels requiring source-track selection: {diagnostics['overlapping_source_activity_window_vessels']:,}
- Nonselected source-track trips excluded: {diagnostics['source_identity_trips_excluded']:,}
- Ordered port calls: {diagnostics['port_calls']:,}
- City episodes: {diagnostics['city_episodes']:,}
- Same-city calls merged: {diagnostics['commercial_calls_merged_by_same_city_rule']:,}
- Episodes containing multiple commercial calls: {diagnostics['episodes_with_multiple_commercial_calls']:,}
- Technical calls retained inside a same-city episode: {diagnostics['technical_calls_inside_same_city_episode']:,}
- Observed directed adjacent-episode pairs: {diagnostics['adjacent_episode_pair_count']:,}
- Adjacent-episode candidate occurrences: {diagnostics['adjacent_episode_candidate_occurrences']:,}

## Hard Gates

- Missing v3 mapping calls: {diagnostics['missing_mapping_calls']:,}
- Residual overlapping source activity windows: {diagnostics['residual_overlapping_source_activity_windows']:,}
- Adjacent duplicate city episodes: {diagnostics['adjacent_duplicate_city_episodes']:,}
- Non-positive candidate times: {diagnostics['invalid_nonpositive_candidate_times']:,}
- Overlapping ordered calls: {diagnostics['overlapping_call_count']:,}
- Arrival-incomplete boundary episodes: {diagnostics['arrival_incomplete_episodes']:,}
- Departure-incomplete boundary episodes: {diagnostics['departure_incomplete_episodes']:,}

Boundary-incomplete episodes remain explicit and are not used on the missing
side of adjacent-episode pairing.
"""
    paths["audit_md"].write_text(report, encoding="utf-8")

    manifest_outputs = {
        key: {
            "path": str(path.relative_to(CODING_ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
        if key != "output_manifest"
    }
    output_manifest = {
        "status": "pass",
        "outputs": manifest_outputs,
    }
    paths["output_manifest"].write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-input-hash-check",
        action="store_true",
        help="Tests only; formal C0A runs must not use this option.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)
    result, input_manifest = build(
        config, verify_hashes=not args.skip_input_hash_check
    )
    output_manifest = write_outputs(
        result, input_manifest, output_dir, config_path
    )
    print(json.dumps({"diagnostics": result.diagnostics, **output_manifest}, indent=2))


if __name__ == "__main__":
    main()
