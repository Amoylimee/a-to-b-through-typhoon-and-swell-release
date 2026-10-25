"""Pure C1 candidate and raw-stop construction helpers."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .candidate_pairs import (
    ADJACENT_COMMERCIAL_EPISODES,
    destination_episode_indices,
    validate_candidate_enumeration_mode,
)


class DataContractError(RuntimeError):
    """Raised when the frozen C1 execution contract is violated."""


def stable_id(*tokens: object, length: int = 24) -> str:
    payload = "|".join(str(token) for token in tokens)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def service_regime_proxy(
    vessel_type: str,
    liner_types: set[str],
    tramp_types: set[str],
) -> str:
    if vessel_type in liner_types:
        return "liner_proxy"
    if vessel_type in tramp_types:
        return "tramp_proxy"
    return "unclassified"


def vessel_family_proxy(vessel_type: str) -> str:
    if "TANKER" in vessel_type or vessel_type in {"LNG TANKER", "LPG TANKER"}:
        return "tanker_proxy"
    if any(
        token in vessel_type
        for token in ("BULK", "ORE", "OBO", "CEMENT", "AGGREGATES", "WOOD CHIPS", "TIMBER")
    ):
        return "bulk_proxy"
    if vessel_type in {
        "GENERAL CARGO",
        "HEAVY LIFT VESSEL",
        "HEAVY LOAD CARRIER",
        "LIVESTOCK CARRIER",
    }:
        return "general_cargo_proxy"
    return "other_proxy"


def assign_period_id(
    timestamp: pd.Timestamp,
    windows: Iterable[dict[str, Any]],
) -> str:
    matches = [
        str(window["period_id"])
        for window in windows
        if window["start"] <= timestamp < window["end"]
    ]
    if not matches:
        return "outside_configured_period"
    if len(matches) != 1:
        raise DataContractError(
            f"Expected exactly one period for {timestamp.isoformat()}, found {matches}"
        )
    return matches[0]


def resolve_stop_location(
    arrival_call_id: str,
    departure_call_id: str,
    arrival_meta: dict[str, str],
    departure_meta: dict[str, str],
    episode_city: dict[str, str],
) -> tuple[str, str, bool, bool]:
    """Resolve a between-leg stationary interval without hiding ambiguity.

    Returns `(category, resolved_city, strict_eligible, one_sided_eligible)`.
    The strict C1.5 fit only uses a nonblank city supported on both sides or by
    one shared C0 city episode. The one-sided flag is retained for sensitivity.
    """

    arrival_city = str(arrival_meta.get("topology_city_id", "") or "")
    departure_city = str(departure_meta.get("topology_city_id", "") or "")
    arrival_episode = str(arrival_meta.get("episode_id", "") or "")
    departure_episode = str(departure_meta.get("episode_id", "") or "")

    if arrival_call_id == departure_call_id:
        city = arrival_city or departure_city
        return "same_call", city, bool(city), bool(city)
    if arrival_city and arrival_city == departure_city:
        return "same_city", arrival_city, True, True
    if arrival_episode and arrival_episode == departure_episode:
        city = episode_city.get(arrival_episode, arrival_city or departure_city)
        return "same_episode", city, bool(city), bool(city)
    if arrival_city and not departure_city:
        return "arrival_city_only", arrival_city, False, True
    if departure_city and not arrival_city:
        return "departure_city_only", departure_city, False, True
    if not arrival_city and not departure_city:
        return "blank_topology", "", False, False
    return "conflicting_cities", "", False, False


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def _range_sum(prefix: np.ndarray, start: int, stop: int) -> float:
    return float(prefix[stop] - prefix[start])


def enumerate_vessel_candidates(
    episodes: pd.DataFrame,
    legs: pd.DataFrame,
    call_meta: dict[str, dict[str, str]],
    windows: list[dict[str, Any]],
    liner_types: set[str],
    tramp_types: set[str],
    candidate_enumeration_mode: str = ADJACENT_COMMERCIAL_EPISODES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate candidates and candidate-used raw stops for one vessel.

    ``episodes`` and ``legs`` must each contain exactly one vessel and be
    ordered. The formal E030 rule joins each commercial city episode only to
    the immediately following commercial city episode.
    """

    candidate_enumeration_mode = validate_candidate_enumeration_mode(
        candidate_enumeration_mode
    )
    if episodes.empty:
        return pd.DataFrame(), pd.DataFrame()
    episodes = episodes.sort_values("episode_sequence").reset_index(drop=True)
    legs = legs.sort_values("leg_sequence").reset_index(drop=True)
    if legs.empty:
        return pd.DataFrame(), pd.DataFrame()
    if episodes["mmsi"].nunique() != 1 or legs["mmsi"].nunique() != 1:
        raise DataContractError("Vessel enumerator received multiple MMSI values")
    if str(episodes.iloc[0]["mmsi"]) != str(legs.iloc[0]["mmsi"]):
        raise DataContractError("Episode and leg vessel keys do not match")

    mmsi = str(episodes.iloc[0]["mmsi"])
    imo = str(episodes.iloc[0]["imo"])
    vessel_type = str(episodes.iloc[0]["detailed_vessel_type_normalized"])
    regime = service_regime_proxy(vessel_type, liner_types, tramp_types)
    family = vessel_family_proxy(vessel_type)

    departure_lookup: dict[tuple[str, int], int] = {}
    arrival_lookup: dict[tuple[str, int], int] = {}
    for index, row in legs.iterrows():
        departure_lookup[(str(row["departure_call_id"]), row["departure_time"].value)] = int(index)
        arrival_lookup[(str(row["arrival_call_id"]), row["arrival_time"].value)] = int(index)

    duration_hours = (
        (legs["arrival_time"] - legs["departure_time"]).dt.total_seconds().to_numpy()
        / 3600.0
    )
    if np.any(duration_hours <= 0):
        raise DataContractError(f"Non-positive OD leg for MMSI {mmsi}")
    sea_prefix = _prefix(duration_hours)

    episode_city = dict(
        zip(episodes["episode_id"].astype(str), episodes["topology_city_id"].astype(str))
    )
    n_stops = max(0, len(legs) - 1)
    stop_rows: list[dict[str, Any]] = []
    stop_hours = np.zeros(n_stops, dtype=np.float64)
    strict_unresolved = np.zeros(n_stops, dtype=np.float64)
    blank_topology = np.zeros(n_stops, dtype=np.float64)
    conflicting_city = np.zeros(n_stops, dtype=np.float64)
    stop_usage_diff = np.zeros(len(legs) + 1, dtype=np.int64)

    for stop_index in range(n_stops):
        arrival_leg = legs.iloc[stop_index]
        departure_leg = legs.iloc[stop_index + 1]
        arrival_time = arrival_leg["arrival_time"]
        departure_time = departure_leg["departure_time"]
        dwell_hours = float((departure_time - arrival_time).total_seconds() / 3600.0)
        if dwell_hours < 0:
            raise DataContractError(
                f"Overlapping adjacent OD legs for MMSI {mmsi}, index {stop_index}"
            )
        arrival_call = str(arrival_leg["arrival_call_id"])
        departure_call = str(departure_leg["departure_call_id"])
        arrival_meta = call_meta.get(arrival_call, {})
        departure_meta = call_meta.get(departure_call, {})
        category, city, strict, one_sided = resolve_stop_location(
            arrival_call,
            departure_call,
            arrival_meta,
            departure_meta,
            episode_city,
        )
        stop_hours[stop_index] = dwell_hours
        strict_unresolved[stop_index] = 0.0 if strict else 1.0
        blank_topology[stop_index] = 1.0 if category == "blank_topology" else 0.0
        conflicting_city[stop_index] = 1.0 if category == "conflicting_cities" else 0.0
        stop_rows.append(
            {
                "stop_id": stable_id(
                    imo,
                    arrival_leg["trip_id"],
                    departure_leg["trip_id"],
                ),
                "mmsi": mmsi,
                "imo": imo,
                "detailed_vessel_type_normalized": vessel_type,
                "ship_type_sample": str(episodes.iloc[0]["ship_type_sample"]),
                "service_regime_proxy": regime,
                "vessel_family_proxy": family,
                "stop_sequence": stop_index,
                "arrival_trip_id": str(arrival_leg["trip_id"]),
                "departure_trip_id": str(departure_leg["trip_id"]),
                "arrival_call_id": arrival_call,
                "departure_call_id": departure_call,
                "arrival_time": arrival_time,
                "departure_time": departure_time,
                "dwell_hours": dwell_hours,
                "dwell_days": dwell_hours / 24.0,
                "arrival_topology_city_id": str(
                    arrival_meta.get("topology_city_id", "") or ""
                ),
                "departure_topology_city_id": str(
                    departure_meta.get("topology_city_id", "") or ""
                ),
                "resolved_topology_city_id": city,
                "location_resolution": category,
                "r5_fit_eligible_strict": strict,
                "r5_fit_eligible_one_sided": one_sided,
            }
        )

    stop_prefix = _prefix(stop_hours)
    unresolved_prefix = _prefix(strict_unresolved)
    blank_prefix = _prefix(blank_topology)
    conflicting_prefix = _prefix(conflicting_city)

    cities = episodes["topology_city_id"].astype(str).tolist()
    episode_ids = episodes["episode_id"].astype(str).tolist()
    departures = episodes["passage_departure_time"].tolist()
    arrivals = episodes["passage_arrival_time"].tolist()
    first_calls = episodes["first_call_id"].astype(str).tolist()
    last_calls = episodes["last_call_id"].astype(str).tolist()

    candidate_rows: list[dict[str, Any]] = []
    for origin_index, origin_city in enumerate(cities):
        departure_time = departures[origin_index]
        if pd.isna(departure_time):
            continue
        departure_key = (last_calls[origin_index], departure_time.value)
        if departure_key not in departure_lookup:
            raise DataContractError(
                f"Origin episode boundary has no OD leg: {episode_ids[origin_index]}"
            )
        outbound_leg = departure_lookup[departure_key]
        for destination_index in destination_episode_indices(
            cities,
            origin_index,
            candidate_enumeration_mode,
        ):
            destination_city = cities[destination_index]
            arrival_time = arrivals[destination_index]
            if pd.isna(arrival_time):
                continue
            arrival_key = (first_calls[destination_index], arrival_time.value)
            if arrival_key not in arrival_lookup:
                raise DataContractError(
                    f"Destination episode boundary has no OD leg: {episode_ids[destination_index]}"
                )
            inbound_leg = arrival_lookup[arrival_key]
            if inbound_leg < outbound_leg:
                raise DataContractError(
                    f"Reversed leg span {episode_ids[origin_index]} -> {episode_ids[destination_index]}"
                )
            total_hours = float((arrival_time - departure_time).total_seconds() / 3600.0)
            if total_hours <= 0:
                raise DataContractError(
                    f"Non-positive candidate time {episode_ids[origin_index]} -> {episode_ids[destination_index]}"
                )
            sea_hours = _range_sum(sea_prefix, outbound_leg, inbound_leg + 1)
            gap_hours = _range_sum(stop_prefix, outbound_leg, inbound_leg)
            residual_hours = total_hours - sea_hours - gap_hours
            middle = cities[origin_index + 1 : destination_index]
            middle_counts = Counter(middle)
            repeated = sorted(city for city, count in middle_counts.items() if count > 1)
            period_id = assign_period_id(departure_time, windows)
            if inbound_leg > outbound_leg:
                stop_usage_diff[outbound_leg] += 1
                stop_usage_diff[inbound_leg] -= 1
            candidate_rows.append(
                {
                    "candidate_id": stable_id(
                        imo,
                        episode_ids[origin_index],
                        episode_ids[destination_index],
                    ),
                    "mmsi": mmsi,
                    "imo": imo,
                    "vessel_uid_clean": str(episodes.iloc[0]["vessel_uid_clean"]),
                    "vessel_name": str(episodes.iloc[0]["vessel_name"]),
                    "detailed_vessel_type_normalized": vessel_type,
                    "ship_type_sample": str(episodes.iloc[0]["ship_type_sample"]),
                    "service_regime_proxy": regime,
                    "vessel_family_proxy": family,
                    "period_id": period_id,
                    "r3_window_eligible": period_id != "outside_configured_period",
                    "origin_city": origin_city,
                    "dest_city": destination_city,
                    "pair_id": f"{origin_city}__{destination_city}",
                    "origin_episode_id": episode_ids[origin_index],
                    "dest_episode_id": episode_ids[destination_index],
                    "origin_episode_sequence": int(episodes.iloc[origin_index]["episode_sequence"]),
                    "dest_episode_sequence": int(episodes.iloc[destination_index]["episode_sequence"]),
                    "departure_time": departure_time,
                    "arrival_time": arrival_time,
                    "total_hours": total_hours,
                    "observed_sea_hours": sea_hours,
                    "intermediate_gap_hours": gap_hours,
                    "time_identity_residual_hours": residual_hours,
                    "outbound_leg_sequence": outbound_leg,
                    "inbound_leg_sequence": inbound_leg,
                    "n_observed_sea_legs": inbound_leg - outbound_leg + 1,
                    "n_intermediate_raw_stops": inbound_leg - outbound_leg,
                    "n_strict_unresolved_raw_stops": int(
                        round(_range_sum(unresolved_prefix, outbound_leg, inbound_leg))
                    ),
                    "n_blank_topology_raw_stops": int(
                        round(_range_sum(blank_prefix, outbound_leg, inbound_leg))
                    ),
                    "n_conflicting_city_raw_stops": int(
                        round(_range_sum(conflicting_prefix, outbound_leg, inbound_leg))
                    ),
                    "n_intermediate_city_episodes": len(middle),
                    "intermediate_city_sequence": ">".join(middle),
                    "r2_repeated_intermediate_city": bool(repeated),
                    "r2_repeated_city_count": len(repeated),
                    "r2_repeated_cities": ";".join(repeated),
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        recurrence = candidates.groupby(
            ["imo", "origin_city", "dest_city", "period_id"],
            sort=False,
        )["candidate_id"].transform("size")
        candidates["r3_recurrence_count"] = recurrence.astype(np.int64)
        candidates["r3_singleton"] = (
            candidates["r3_window_eligible"]
            & candidates["r3_recurrence_count"].eq(1)
        )

    if stop_rows:
        usage = np.cumsum(stop_usage_diff[:-2], dtype=np.int64)
        stops = pd.DataFrame(stop_rows)
        if len(usage) != len(stops):
            raise DataContractError("Raw-stop usage counter length mismatch")
        stops["candidate_intermediate_usage_count"] = usage
        stops["used_by_any_r1_candidate"] = usage > 0
    else:
        stops = pd.DataFrame()
    return candidates, stops
