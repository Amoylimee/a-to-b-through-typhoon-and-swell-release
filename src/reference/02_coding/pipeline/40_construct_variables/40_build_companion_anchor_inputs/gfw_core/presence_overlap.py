"""Pure helpers for the bounded GFW/reference AIS AIS-presence overlap probe.

GFW Vessel Presence is a second AIS feed, not an independent satellite
observation.  These helpers therefore retain the distinction between an exact
MMSI/hour hit, a two-sided-motion-feasible grid cell, and an unresolved hour.
They never choose a point from multiple feasible cells.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .dark_gaps import KM_PER_NAUTICAL_MILE, haversine_km
from .sar_matching import canonical_gfw_mmsi, great_circle_point


ONE_HOUR = pd.Timedelta(hours=1)


def utc_timestamp(value: Any) -> pd.Timestamp:
    """Return one timezone-aware UTC timestamp."""

    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def full_utc_hours_within(start: Any, end: Any) -> pd.DatetimeIndex:
    """Return UTC clock-hour bins fully contained in ``[start, end]``.

    GFW Presence contains one gridded position per vessel and UTC hour, without
    the original message timestamp.  Requiring the whole clock hour to lie in
    the reference AIS gap prevents a partially observed boundary hour from being called
    repaired.
    """

    left = utc_timestamp(start)
    right = utc_timestamp(end)
    if right <= left:
        return pd.DatetimeIndex([], tz="UTC")
    first = left.ceil("h")
    last = (right - ONE_HOUR).floor("h")
    if last < first:
        return pd.DatetimeIndex([], tz="UTC")
    hours = pd.date_range(first, last, freq="h", tz="UTC")
    return hours[(hours >= left) & ((hours + ONE_HOUR) <= right)]


def point_region(
    lat: float,
    lon: float,
    regions: Mapping[str, list[float]],
) -> tuple[str | None, int]:
    """Assign a point to the first configured box and report overlap count."""

    labels: list[str] = []
    for name, bounds in regions.items():
        south, north, west, east = map(float, bounds)
        if south <= float(lat) <= north and west <= float(lon) <= east:
            labels.append(str(name))
    return (labels[0] if labels else None), len(labels)


def expand_bounded_gap_hours(
    gaps: pd.DataFrame,
    regions: Mapping[str, list[float]],
    *,
    break_utc: Any,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Expand bounded gaps to conservative full UTC hours in protected boxes."""

    columns = [
        "target_hour_id",
        "gap_id",
        "candidate_id",
        "mmsi",
        "route",
        "period",
        "hour_utc",
        "region",
        "region_overlap_count",
        "direct_path_lat",
        "direct_path_lon",
        "source_gap_start_utc",
        "source_gap_end_utc",
        "lat_a",
        "lon_a",
        "lat_b",
        "lon_b",
        "effective_speed_kn",
    ]
    if gaps.empty:
        return pd.DataFrame(columns=columns), {
            "input_gaps": 0,
            "unbounded_gaps": 0,
            "bounded_gaps_without_full_hour": 0,
            "full_hours_outside_regions": 0,
            "duplicate_target_hours": 0,
        }

    cutoff = utc_timestamp(break_utc)
    rows: list[dict[str, Any]] = []
    unbounded = 0
    no_full_hour = 0
    outside = 0
    for gap in gaps.to_dict(orient="records"):
        if not bool(gap["bounded"]):
            unbounded += 1
            continue
        hours = full_utc_hours_within(gap["gap_start_utc"], gap["gap_end_utc"])
        if len(hours) == 0:
            no_full_hour += 1
            continue
        source_start = utc_timestamp(gap["source_gap_start_utc"])
        source_end = utc_timestamp(gap["source_gap_end_utc"])
        duration_seconds = (source_end - source_start).total_seconds()
        if duration_seconds <= 0:
            raise ValueError("source gap must have positive duration")
        for hour in hours:
            midpoint = hour + ONE_HOUR / 2
            fraction = (midpoint - source_start).total_seconds() / duration_seconds
            if not -1e-12 <= fraction <= 1.0 + 1e-12:
                raise ValueError("full hour midpoint lies outside source gap")
            direct_lat, direct_lon = great_circle_point(
                float(gap["lat_a"]),
                float(gap["lon_a"]),
                float(gap["lat_b"]),
                float(gap["lon_b"]),
                float(fraction),
            )
            region, overlap_count = point_region(direct_lat, direct_lon, regions)
            if region is None:
                outside += 1
                continue
            mmsi = str(gap["mmsi"])
            hour_iso = hour.isoformat()
            rows.append(
                {
                    "target_hour_id": f"{mmsi}|{hour_iso}",
                    "gap_id": str(gap["gap_id"]),
                    "candidate_id": str(gap["candidate_id"]),
                    "mmsi": mmsi,
                    "route": str(gap["route"]),
                    "period": "pre" if hour < cutoff else "post",
                    "hour_utc": hour,
                    "region": region,
                    "region_overlap_count": int(overlap_count),
                    "direct_path_lat": direct_lat,
                    "direct_path_lon": direct_lon,
                    "source_gap_start_utc": source_start,
                    "source_gap_end_utc": source_end,
                    "lat_a": float(gap["lat_a"]),
                    "lon_a": float(gap["lon_a"]),
                    "lat_b": float(gap["lat_b"]),
                    "lon_b": float(gap["lon_b"]),
                    "effective_speed_kn": float(gap["effective_speed_kn"]),
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    duplicate_hours = (
        int(result["target_hour_id"].duplicated(keep=False).sum()) if len(result) else 0
    )
    if duplicate_hours:
        duplicated = result.loc[result["target_hour_id"].duplicated(keep=False)]
        conflict = duplicated.groupby("target_hour_id", observed=True).agg(
            gaps=("gap_id", "nunique"),
            regions=("region", "nunique"),
        )
        if (conflict[["gaps", "regions"]] > 1).any(axis=None):
            raise ValueError("one MMSI/hour belongs to conflicting voyage gaps or regions")
        result = result.drop_duplicates("target_hour_id", keep="first").reset_index(drop=True)
    return result, {
        "input_gaps": int(len(gaps)),
        "unbounded_gaps": int(unbounded),
        "bounded_gaps_without_full_hour": int(no_full_hour),
        "full_hours_outside_regions": int(outside),
        "duplicate_target_hours": int(duplicate_hours),
    }


def expand_all_gap_hours(
    gaps: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Expand every dark gap into full UTC bins without a spatial filter.

    Bounded internal gaps retain the two bracketing fixes needed for the GFW
    motion check.  Leading, trailing, and no-track gaps stay in the denominator
    but carry no invented geometry and therefore can never become a feasible
    two-sided witness under the protected rule.
    """

    columns = [
        "target_hour_id",
        "gap_id",
        "candidate_id",
        "mmsi",
        "route",
        "gap_type",
        "bounded",
        "hour_utc",
        "direct_path_lat",
        "direct_path_lon",
        "source_gap_start_utc",
        "source_gap_end_utc",
        "lat_a",
        "lon_a",
        "lat_b",
        "lon_b",
        "effective_speed_kn",
    ]
    if gaps.empty:
        return pd.DataFrame(columns=columns), {
            "input_gaps": 0,
            "bounded_gaps": 0,
            "unbounded_gaps": 0,
            "bounded_full_hours": 0,
            "unbounded_full_hours": 0,
            "dark_duration_hours": 0.0,
            "full_utc_dark_hours": 0,
            "partial_boundary_dark_hours": 0.0,
            "gaps_without_full_hour": 0,
            "duplicate_target_hours": 0,
        }

    rows: list[dict[str, Any]] = []
    without_full = 0
    bounded_gaps = 0
    for gap in gaps.to_dict(orient="records"):
        bounded = bool(gap["bounded"])
        bounded_gaps += int(bounded)
        hours = full_utc_hours_within(gap["gap_start_utc"], gap["gap_end_utc"])
        if len(hours) == 0:
            without_full += 1
            continue
        source_start = utc_timestamp(gap["source_gap_start_utc"])
        source_end = utc_timestamp(gap["source_gap_end_utc"])
        duration_seconds = (source_end - source_start).total_seconds()
        if duration_seconds <= 0:
            raise ValueError("source gap must have positive duration")
        for hour in hours:
            direct_lat = np.nan
            direct_lon = np.nan
            if bounded:
                midpoint = hour + ONE_HOUR / 2
                fraction = (midpoint - source_start).total_seconds() / duration_seconds
                if not -1e-12 <= fraction <= 1.0 + 1e-12:
                    raise ValueError("full hour midpoint lies outside source gap")
                direct_lat, direct_lon = great_circle_point(
                    float(gap["lat_a"]),
                    float(gap["lon_a"]),
                    float(gap["lat_b"]),
                    float(gap["lon_b"]),
                    float(fraction),
                )
            mmsi = str(gap["mmsi"])
            rows.append(
                {
                    "target_hour_id": f"{mmsi}|{hour.isoformat()}",
                    "gap_id": str(gap["gap_id"]),
                    "candidate_id": str(gap["candidate_id"]),
                    "mmsi": mmsi,
                    "route": str(gap["route"]),
                    "gap_type": str(gap["gap_type"]),
                    "bounded": bounded,
                    "hour_utc": hour,
                    "direct_path_lat": direct_lat,
                    "direct_path_lon": direct_lon,
                    "source_gap_start_utc": source_start,
                    "source_gap_end_utc": source_end,
                    "lat_a": float(gap["lat_a"]) if bounded else np.nan,
                    "lon_a": float(gap["lon_a"]) if bounded else np.nan,
                    "lat_b": float(gap["lat_b"]) if bounded else np.nan,
                    "lon_b": float(gap["lon_b"]) if bounded else np.nan,
                    "effective_speed_kn": (
                        float(gap["effective_speed_kn"]) if bounded else np.nan
                    ),
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    duplicate_hours = (
        int(result["target_hour_id"].duplicated(keep=False).sum()) if len(result) else 0
    )
    if duplicate_hours:
        duplicated = result.loc[result["target_hour_id"].duplicated(keep=False)]
        conflict = duplicated.groupby("target_hour_id", observed=True).agg(
            gaps=("gap_id", "nunique"),
            routes=("route", "nunique"),
            bounded_states=("bounded", "nunique"),
        )
        if (conflict[["gaps", "routes", "bounded_states"]] > 1).any(axis=None):
            raise ValueError("one MMSI/hour belongs to conflicting voyage gaps")
        result = result.drop_duplicates("target_hour_id", keep="first").reset_index(drop=True)

    dark_duration = float(pd.to_numeric(gaps["dark_hours_in_probe"]).sum())
    full_hours = int(len(result))
    partial = dark_duration - full_hours
    if partial < -1e-8:
        raise AssertionError("full UTC dark bins exceed continuous dark duration")
    return result, {
        "input_gaps": int(len(gaps)),
        "bounded_gaps": int(bounded_gaps),
        "unbounded_gaps": int(len(gaps) - bounded_gaps),
        "bounded_full_hours": int(result["bounded"].sum()) if len(result) else 0,
        "unbounded_full_hours": int((~result["bounded"]).sum()) if len(result) else 0,
        "dark_duration_hours": dark_duration,
        "full_utc_dark_hours": full_hours,
        "partial_boundary_dark_hours": float(max(partial, 0.0)),
        "gaps_without_full_hour": int(without_full),
        "duplicate_target_hours": int(duplicate_hours),
    }


def extract_presence_rows_with_identity_audit(
    payload: Mapping[str, Any], dataset: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize canonical rows while counting provider rows without an MMSI.

    Unfiltered spatial reports may contain activity whose provider MMSI is
    blank.  Such rows can never satisfy this project's exact-target identity
    rule.  They are excluded with an explicit count rather than being allowed
    to erase every valid row in the same response.

    A row whose ``hours`` is not one is counted here and carried forward, not
    rejected.  ``group-by=MMSI`` sums presence across every provider entity
    sharing one MMSI, so such a row reports an identity that is not unique.
    Over 92% of an unfiltered spatial response is discarded by the exact-target
    postfilter, and a non-target row must never be able to stop a block; only
    :func:`classify_admissible_anchors` knows the target keys, so only it can
    decide.  The strict :func:`extract_presence_rows` keeps the older global
    rule for the closed single-purpose probes that rely on it.
    """

    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("unexpected 4Wings report wrapper")
    rows = entries[0].get(dataset)
    if not isinstance(rows, list):
        raise ValueError(f"4Wings response does not contain fixed dataset {dataset}")
    frame = pd.DataFrame(rows)
    required = {"date", "hours", "lat", "lon", "mmsi"}
    missing = sorted(required.difference(frame.columns))
    if missing and len(frame):
        raise ValueError(f"presence response missing columns: {missing}")
    if frame.empty:
        return (
            pd.DataFrame(columns=["hour_utc", "mmsi", "hours", "lat", "lon"]),
            {
                "provider_rows": 0,
                "canonical_identity_rows": 0,
                "noncanonical_identity_rows": 0,
                "blank_identity_rows": 0,
                "hour_multiplicity_rows": 0,
            },
        )
    frame = frame.loc[:, ["date", "mmsi", "hours", "lat", "lon"]].copy()
    raw_mmsi = frame["mmsi"].copy()
    frame["mmsi"] = frame["mmsi"].map(canonical_gfw_mmsi)
    canonical = frame["mmsi"].notna()
    identity_audit = {
        "provider_rows": int(len(frame)),
        "canonical_identity_rows": int(canonical.sum()),
        "noncanonical_identity_rows": int((~canonical).sum()),
        "blank_identity_rows": int(raw_mmsi.fillna("").astype(str).str.strip().eq("").sum()),
    }
    frame = frame.loc[canonical].copy()
    parsed = pd.to_datetime(frame.pop("date"), utc=True, errors="raise")
    if not parsed.dt.minute.eq(0).all() or not parsed.dt.second.eq(0).all():
        raise ValueError("presence response is not aligned to UTC clock hours")
    frame.insert(0, "hour_utc", parsed)
    for column in ("hours", "lat", "lon"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    identity_audit["hour_multiplicity_rows"] = int(frame["hours"].ne(1).sum())
    return frame, identity_audit


def extract_presence_rows(payload: Mapping[str, Any], dataset: str) -> pd.DataFrame:
    """Normalize one strict 4Wings response whose every row carries an MMSI."""

    frame, identity_audit = extract_presence_rows_with_identity_audit(payload, dataset)
    if identity_audit["noncanonical_identity_rows"]:
        raise ValueError("presence response contains a non-canonical MMSI")
    if identity_audit["hour_multiplicity_rows"]:
        raise ValueError("hourly unaggregated presence rows must each carry one hour")
    return frame


def cell_reachable_during_hour(
    dark_hour: Mapping[str, Any],
    cell_lat: float,
    cell_lon: float,
    *,
    cell_degrees: float,
    tolerance_km: float = 0.001,
) -> bool:
    """Whether the reported GFW grid centre satisfies both motion leashes.

    The report does not expose the original AIS timestamp within the hour.  We
    therefore ask whether the centre's feasible-time interval intersects the
    full UTC hour.  The centre is a conservative positive witness.  Expanding
    both leash radii by the cell half diagonal would be unsafe because the
    nearest point to each anchor could be a different point in the cell.
    """

    if float(cell_degrees) <= 0:
        raise ValueError("cell resolution must be positive")
    speed_kmh = float(dark_hour["effective_speed_kn"]) * KM_PER_NAUTICAL_MILE
    if not np.isfinite(speed_kmh) or speed_kmh <= 0:
        return False
    distance_a = max(
        0.0,
        haversine_km(
            float(dark_hour["lat_a"]),
            float(dark_hour["lon_a"]),
            float(cell_lat),
            float(cell_lon),
        )
        - float(tolerance_km),
    )
    distance_b = max(
        0.0,
        haversine_km(
            float(cell_lat),
            float(cell_lon),
            float(dark_hour["lat_b"]),
            float(dark_hour["lon_b"]),
        )
        - float(tolerance_km),
    )
    source_start = utc_timestamp(dark_hour["source_gap_start_utc"])
    source_end = utc_timestamp(dark_hour["source_gap_end_utc"])
    hour_start = utc_timestamp(dark_hour["hour_utc"])
    hour_end = hour_start + ONE_HOUR
    earliest = source_start + pd.to_timedelta(distance_a / speed_kmh, unit="h")
    latest = source_end - pd.to_timedelta(distance_b / speed_kmh, unit="h")
    return max(hour_start, earliest) < min(hour_end, latest)


def match_presence_to_dark_hours(
    dark_hours: pd.DataFrame,
    presence: pd.DataFrame,
    *,
    cell_degrees: float,
    tolerance_km: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return hour-level verdicts and all exact-MMSI/hour evidence cells."""

    if dark_hours["target_hour_id"].duplicated().any():
        raise ValueError("dark-hour denominator must be unique by MMSI/hour")
    target_keys = dark_hours.loc[
        :,
        [
            "mmsi",
            "hour_utc",
            "target_hour_id",
            "gap_id",
            "candidate_id",
            "period",
            "region",
        ],
    ].drop_duplicates(["mmsi", "hour_utc"])
    cells = presence.merge(
        target_keys,
        on=["mmsi", "hour_utc"],
        how="inner",
        validate="many_to_one",
    )
    if len(cells):
        geometry = dark_hours.set_index(["mmsi", "hour_utc"], verify_integrity=True)
        verdicts: list[bool] = []
        for row in cells.itertuples(index=False):
            dark_hour = geometry.loc[(row.mmsi, row.hour_utc)].to_dict()
            dark_hour["hour_utc"] = row.hour_utc
            verdicts.append(
                cell_reachable_during_hour(
                    dark_hour,
                    row.lat,
                    row.lon,
                    cell_degrees=cell_degrees,
                    tolerance_km=tolerance_km,
                )
            )
        cells["motion_feasible"] = verdicts
        counts = cells.groupby(["mmsi", "hour_utc"], observed=True).agg(
            exact_cell_count=("motion_feasible", "size"),
            feasible_cell_count=("motion_feasible", "sum"),
        )
    else:
        cells = cells.assign(motion_feasible=pd.Series(dtype=bool))
        counts = pd.DataFrame(columns=["exact_cell_count", "feasible_cell_count"])
        counts.index = pd.MultiIndex.from_arrays([[], []], names=["mmsi", "hour_utc"])
    result = dark_hours.merge(
        counts.reset_index(),
        on=["mmsi", "hour_utc"],
        how="left",
        validate="one_to_one",
    )
    result[["exact_cell_count", "feasible_cell_count"]] = result[
        ["exact_cell_count", "feasible_cell_count"]
    ].fillna(0).astype("int64")
    result["exact_mmsi_hour_hit"] = result["exact_cell_count"].gt(0)
    result["motion_feasible_hit"] = result["feasible_cell_count"].gt(0)
    result["presence_status"] = np.select(
        [
            result["feasible_cell_count"].gt(1),
            result["feasible_cell_count"].eq(1),
            result["exact_cell_count"].gt(0),
        ],
        ["multiple_feasible_cells", "one_feasible_cell", "exact_mmsi_motion_conflict"],
        default="both_provider_dark",
    )
    return result, cells


def summarize_overlap(hours: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Summarize denominators and overlap outcomes at requested strata."""

    def one(group: pd.DataFrame) -> pd.Series:
        total = int(len(group))
        exact = int(group["exact_mmsi_hour_hit"].sum())
        feasible = int(group["motion_feasible_hit"].sum())
        gaps = int(group["gap_id"].nunique())
        feasible_gaps = int(group.loc[group["motion_feasible_hit"], "gap_id"].nunique())
        return pd.Series(
            {
                "target_dark_hours": total,
                "target_dark_gaps": gaps,
                "exact_mmsi_hour_hits": exact,
                "motion_feasible_hours": feasible,
                "residual_both_provider_dark_hours": total - feasible,
                "gaps_with_motion_feasible_hour": feasible_gaps,
                "exact_mmsi_hour_share_pct": 100.0 * exact / total if total else np.nan,
                "motion_feasible_hour_share_pct": 100.0 * feasible / total if total else np.nan,
                "motion_feasible_gap_share_pct": 100.0 * feasible_gaps / gaps if gaps else np.nan,
                "motion_conflict_hours": int(
                    group["presence_status"].eq("exact_mmsi_motion_conflict").sum()
                ),
                "multiple_feasible_cell_hours": int(
                    group["presence_status"].eq("multiple_feasible_cells").sum()
                ),
            }
        )

    if by:
        records: list[dict[str, Any]] = []
        for keys, group in hours.groupby(by, observed=True, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            record = dict(zip(by, key_values))
            record.update(one(group).to_dict())
            records.append(record)
        return pd.DataFrame(records)
    return one(hours).to_frame().T
