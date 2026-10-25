"""IBTrACS loading, interpolation, matching, and aggregation for E020.

The implementation deliberately keeps cyclone-centre availability separate from
wind availability.  A missing intensity therefore never removes a measurable
distance exposure.  Track interpolation uses the original within-SID row order,
so a filtered lifecycle record creates a hard phase break.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .geo import great_circle_km, normalize_longitude


IBTRACS_COLUMNS = (
    "SID",
    "ISO_TIME",
    "NATURE",
    "LAT",
    "LON",
    "USA_WIND",
    "WMO_WIND",
    "TRACK_TYPE",
)
DEFAULT_TRACK_TYPES = ("main", "PROVISIONAL", "US-PROVISIONAL")
DEFAULT_PRIMARY_NATURE = ("TS", "SS", "MX")
DEFAULT_PROVISIONAL_NR_TRACK_TYPES = ("PROVISIONAL", "US-PROVISIONAL")
NS_PER_HOUR = 3_600_000_000_000
# Sub-millimetre tolerance protects an analytically exact spherical-radius
# boundary from arccos roundoff; it has no material geographic extent.
RADIUS_COMPARISON_TOLERANCE_KM = 1e-9


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_series(values: Iterable[object], *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    result = pd.Series(parsed, copy=False)
    if result.isna().any():
        raise ValueError(f"{name} contains missing or invalid UTC timestamps")
    return result


def _json_counts(values: pd.Series) -> dict[str, int]:
    counts = values.astype("string").fillna("<NA>").value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.sort_index().items()}


def load_ibtracs_snapshot(
    path: str | Path,
    *,
    time_start: str | pd.Timestamp = "2021-01-01T00:00:00Z",
    time_end_exclusive: str | pd.Timestamp = "2026-01-01T00:00:00Z",
    track_types: Sequence[str] = DEFAULT_TRACK_TYPES,
    primary_nature: Sequence[str] = DEFAULT_PRIMARY_NATURE,
    provisional_nr_track_types: Sequence[str] = DEFAULT_PROVISIONAL_NR_TRACK_TYPES,
    expected_sha256: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read and freeze the E020 subset of the IBTrACS v04r01 CSV.

    The second CSV row contains units and is skipped.  Original row adjacency is
    recorded before any lifecycle filter is applied; retained phases therefore
    cannot bridge an excluded record.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing IBTrACS snapshot: {source}")
    # The caller must explicitly supply the frozen config hash.  ``None`` is
    # available only for synthetic fixtures and cannot occur by omission.
    actual_hash = _file_sha256(source) if expected_sha256 is not None else None
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError(
            f"IBTrACS SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )

    raw = pd.read_csv(
        source,
        skiprows=[1],
        usecols=list(IBTRACS_COLUMNS),
        dtype="string",
        low_memory=False,
    )
    raw["_source_row"] = np.arange(len(raw), dtype=np.int64)
    for column in ("SID", "NATURE", "TRACK_TYPE"):
        raw[column] = raw[column].str.strip()
    raw["fix_time_utc"] = _utc_series(raw["ISO_TIME"], name="ISO_TIME").array
    if raw["SID"].isna().any() or raw["SID"].eq("").any():
        raise ValueError("IBTrACS SID contains missing or blank values")
    raw["_sid_source_order"] = raw.groupby("SID", sort=False).cumcount().astype("int64")

    # Original file order must also be chronological within SID.  This is the
    # ordering used to decide whether an excluded lifecycle row broke a phase.
    time_ns = raw["fix_time_utc"].astype("int64")
    if (time_ns.groupby(raw["SID"], sort=False).diff().dropna() < 0).any():
        raise ValueError("IBTrACS fixes are not strictly increasing within SID source order")

    start = pd.Timestamp(time_start)
    end = pd.Timestamp(time_end_exclusive)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end <= start:
        raise ValueError("time_end_exclusive must be later than time_start")
    window = raw.loc[(raw["fix_time_utc"] >= start) & (raw["fix_time_utc"] < end)].copy()
    if window.duplicated(["SID", "fix_time_utc"]).any():
        example = window.loc[
            window.duplicated(["SID", "fix_time_utc"], keep=False), ["SID", "fix_time_utc"]
        ].head(5)
        raise ValueError(f"IBTrACS SID + ISO_TIME is not unique: {example.to_dict('records')}")

    track_ok = window["TRACK_TYPE"].isin(tuple(track_types))
    primary = window["NATURE"].isin(tuple(primary_nature))
    provisional_nr = window["NATURE"].eq("NR") & window["TRACK_TYPE"].isin(
        tuple(provisional_nr_track_types)
    )
    retained = window.loc[track_ok & (primary | provisional_nr)].copy()

    for source_column, output_column in (("LAT", "center_lat"), ("LON", "center_lon_raw")):
        retained[output_column] = pd.to_numeric(retained[source_column], errors="coerce")
    retained["center_valid"] = (
        retained["center_lat"].between(-90.0, 90.0, inclusive="both")
        & retained["center_lon_raw"].between(-180.0, 360.0, inclusive="both")
        & np.isfinite(retained["center_lat"])
        & np.isfinite(retained["center_lon_raw"])
    )
    retained["center_lon"] = np.where(
        retained["center_valid"], normalize_longitude(retained["center_lon_raw"]), np.nan
    )

    usa = pd.to_numeric(retained["USA_WIND"], errors="coerce").astype("float64")
    wmo = pd.to_numeric(retained["WMO_WIND"], errors="coerce").astype("float64")
    usa_valid = np.isfinite(usa) & (usa >= 0.0)
    wmo_valid = np.isfinite(wmo) & (wmo >= 0.0)
    retained["wind_kt"] = np.where(usa_valid, usa, np.where(wmo_valid, wmo, np.nan))
    retained["wind_source"] = pd.Series(
        np.where(usa_valid, "USA_WIND", np.where(wmo_valid, "WMO_WIND", pd.NA)),
        index=retained.index,
        dtype="string",
    )
    retained["is_provisional_nr"] = provisional_nr.loc[retained.index].astype(bool)

    retained.sort_values("_source_row", kind="stable", inplace=True)
    adjacent = retained.groupby("SID", sort=False)["_sid_source_order"].diff().eq(1)
    retained["retained_phase"] = (
        (~adjacent).groupby(retained["SID"], sort=False).cumsum().astype("int32") - 1
    )

    retained = retained.rename(
        columns={"NATURE": "nature", "TRACK_TYPE": "track_type"}
    )[
        [
            "SID",
            "fix_time_utc",
            "nature",
            "track_type",
            "center_lat",
            "center_lon",
            "center_valid",
            "wind_kt",
            "wind_source",
            "is_provisional_nr",
            "retained_phase",
            "_source_row",
            "_sid_source_order",
        ]
    ].reset_index(drop=True)

    window_year = window["fix_time_utc"].dt.year.astype(int)
    retained_year = retained["fix_time_utc"].dt.year.astype(int)
    year_rows: dict[str, dict[str, int]] = {}
    for year in range(start.year, end.year + 1):
        if year >= end.year and end == pd.Timestamp(f"{end.year}-01-01", tz="UTC"):
            continue
        raw_mask = window_year.eq(year)
        kept_mask = retained_year.eq(year)
        year_rows[str(year)] = {
            "window_rows": int(raw_mask.sum()),
            "retained_rows": int(kept_mask.sum()),
            "retained_sids": int(retained.loc[kept_mask, "SID"].nunique()),
            "provisional_nr_rows": int(retained.loc[kept_mask, "is_provisional_nr"].sum()),
            "track_type_counts": {
                track_type: int(
                    retained.loc[kept_mask, "track_type"].eq(track_type).sum()
                )
                for track_type in track_types
            },
        }
    audit: dict[str, object] = {
        "path": str(source.resolve()),
        "sha256": actual_hash,
        "time_start": start.isoformat(),
        "time_end_exclusive": end.isoformat(),
        "window_rows": int(len(window)),
        "window_sids": int(window["SID"].nunique()),
        "window_track_type_counts": _json_counts(window["TRACK_TYPE"]),
        "window_nature_counts": _json_counts(window["NATURE"]),
        "retained_rows": int(len(retained)),
        "retained_sids": int(retained["SID"].nunique()),
        "retained_phase_count": int(retained.groupby(["SID", "retained_phase"]).ngroups),
        "retained_invalid_center_rows": int((~retained["center_valid"]).sum()),
        "retained_missing_wind_rows": int(retained["wind_kt"].isna().sum()),
        "retained_wind_source_counts": _json_counts(retained["wind_source"]),
        "provisional_nr_rows": int(retained["is_provisional_nr"].sum()),
        "year_rows": year_rows,
    }
    return retained, audit


@dataclass(frozen=True)
class _TrackIntervals:
    sid: np.ndarray
    start_ns: np.ndarray
    end_ns: np.ndarray
    start_lat: np.ndarray
    end_lat: np.ndarray
    start_lon: np.ndarray
    lon_delta: np.ndarray
    start_wind: np.ndarray
    end_wind: np.ndarray


class CycloneTrackIndex:
    """A compact interval index for exact and interpolated cyclone positions."""

    def __init__(self, fixes: pd.DataFrame, *, maximum_fix_gap_hours: float = 3.0):
        required = {
            "SID",
            "fix_time_utc",
            "center_lat",
            "center_lon",
            "center_valid",
            "wind_kt",
            "retained_phase",
            "_source_row",
        }
        missing = sorted(required - set(fixes.columns))
        if missing:
            raise ValueError(f"retained IBTrACS table missing columns: {missing}")
        if maximum_fix_gap_hours <= 0:
            raise ValueError("maximum_fix_gap_hours must be positive")
        work = fixes.copy()
        work["fix_time_utc"] = _utc_series(
            work["fix_time_utc"], name="fix_time_utc"
        ).array
        if work.duplicated(["SID", "fix_time_utc"]).any():
            raise ValueError("retained fixes contain duplicate SID + fix_time_utc")
        work.sort_values(["SID", "_source_row"], kind="stable", inplace=True)
        self.maximum_fix_gap_hours = float(maximum_fix_gap_hours)
        self._fixes = work.reset_index(drop=True)

        next_fix = work.groupby("SID", sort=False).shift(-1)
        gap_ns = (
            next_fix["fix_time_utc"].astype("int64") - work["fix_time_utc"].astype("int64")
        )
        link = (
            next_fix["retained_phase"].eq(work["retained_phase"])
            & work["center_valid"].eq(True)
            & next_fix["center_valid"].eq(True)
            & gap_ns.gt(0)
            & gap_ns.le(int(round(maximum_fix_gap_hours * NS_PER_HOUR)))
        )
        left = work.loc[link]
        right = next_fix.loc[link]
        start_lon = left["center_lon"].to_numpy(np.float64)
        end_lon = right["center_lon"].to_numpy(np.float64)
        lon_delta = normalize_longitude(end_lon - start_lon)
        self._intervals = _TrackIntervals(
            sid=left["SID"].astype(str).to_numpy(),
            start_ns=left["fix_time_utc"].astype("int64").to_numpy(),
            end_ns=right["fix_time_utc"].astype("int64").to_numpy(),
            start_lat=left["center_lat"].to_numpy(np.float64),
            end_lat=right["center_lat"].to_numpy(np.float64),
            start_lon=start_lon,
            lon_delta=np.asarray(lon_delta, dtype=np.float64),
            start_wind=left["wind_kt"].to_numpy(np.float64),
            end_wind=right["wind_kt"].to_numpy(np.float64),
        )

    @property
    def n_fixes(self) -> int:
        return int(len(self._fixes))

    @property
    def n_intervals(self) -> int:
        return int(len(self._intervals.start_ns))

    def positions_at_times(self, times: Iterable[object]) -> pd.DataFrame:
        """Return one active centre per target-time × SID.

        Exact source fixes take their own recorded wind.  Strictly interior
        targets interpolate wind only when both endpoints are valid.
        """

        parsed = _utc_series(times, name="target times")
        if parsed.empty:
            return pd.DataFrame(
                {
                    "target_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "SID": pd.Series(dtype="string"),
                    "storm_lat": pd.Series(dtype="float64"),
                    "storm_lon": pd.Series(dtype="float64"),
                    "storm_wind_kt": pd.Series(dtype="float64"),
                    "position_source": pd.Series(dtype="string"),
                }
            )
        target_ns = np.unique(parsed.astype("int64").to_numpy())
        pieces: list[pd.DataFrame] = []

        valid_fixes = self._fixes.loc[self._fixes["center_valid"].fillna(False)].copy()
        fix_ns = valid_fixes["fix_time_utc"].astype("int64").to_numpy()
        target_locations = np.searchsorted(target_ns, fix_ns)
        exact = (target_locations < len(target_ns)) & (
            target_ns[np.minimum(target_locations, len(target_ns) - 1)] == fix_ns
        )
        if exact.any():
            exact_fix = valid_fixes.loc[exact]
            pieces.append(
                pd.DataFrame(
                    {
                        "target_ns": exact_fix["fix_time_utc"].astype("int64").to_numpy(),
                        "SID": exact_fix["SID"].astype(str).to_numpy(),
                        "storm_lat": exact_fix["center_lat"].to_numpy(np.float64),
                        "storm_lon": exact_fix["center_lon"].to_numpy(np.float64),
                        "storm_wind_kt": exact_fix["wind_kt"].to_numpy(np.float64),
                        "position_source": "exact_fix",
                    }
                )
            )

        interval = self._intervals
        for index in range(len(interval.start_ns)):
            # Endpoints are supplied by exact fixes.  Only strictly interior
            # target times are interpolated, eliminating endpoint duplicates.
            first = int(np.searchsorted(target_ns, interval.start_ns[index], side="right"))
            last = int(np.searchsorted(target_ns, interval.end_ns[index], side="left"))
            if first >= last:
                continue
            selected = target_ns[first:last]
            fraction = (selected - interval.start_ns[index]) / (
                interval.end_ns[index] - interval.start_ns[index]
            )
            wind = np.full(len(selected), np.nan, dtype=np.float64)
            if np.isfinite(interval.start_wind[index]) and np.isfinite(
                interval.end_wind[index]
            ):
                wind = interval.start_wind[index] + fraction * (
                    interval.end_wind[index] - interval.start_wind[index]
                )
            pieces.append(
                pd.DataFrame(
                    {
                        "target_ns": selected,
                        "SID": interval.sid[index],
                        "storm_lat": interval.start_lat[index]
                        + fraction * (interval.end_lat[index] - interval.start_lat[index]),
                        "storm_lon": normalize_longitude(
                            interval.start_lon[index] + fraction * interval.lon_delta[index]
                        ),
                        "storm_wind_kt": wind,
                        "position_source": "interpolated",
                    }
                )
            )

        if not pieces:
            return pd.DataFrame(
                {
                    "target_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "SID": pd.Series(dtype="string"),
                    "storm_lat": pd.Series(dtype="float64"),
                    "storm_lon": pd.Series(dtype="float64"),
                    "storm_wind_kt": pd.Series(dtype="float64"),
                    "position_source": pd.Series(dtype="string"),
                }
            )
        result = pd.concat(pieces, ignore_index=True)
        if result.duplicated(["target_ns", "SID"]).any():
            raise RuntimeError("track index emitted duplicate target-time × SID centres")
        result["target_time_utc"] = pd.to_datetime(result.pop("target_ns"), utc=True)
        result["SID"] = result["SID"].astype("string")
        result["position_source"] = result["position_source"].astype("string")
        return result.sort_values(["target_time_utc", "SID"], kind="stable").reset_index(
            drop=True
        )

    def interpolate_hourly_fixes(
        self, start: str | pd.Timestamp, end_exclusive: str | pd.Timestamp
    ) -> pd.DataFrame:
        start_time = pd.Timestamp(start)
        end_time = pd.Timestamp(end_exclusive)
        start_time = (
            start_time.tz_localize("UTC")
            if start_time.tzinfo is None
            else start_time.tz_convert("UTC")
        )
        end_time = (
            end_time.tz_localize("UTC")
            if end_time.tzinfo is None
            else end_time.tz_convert("UTC")
        )
        if start_time != start_time.floor("h") or end_time != end_time.floor("h"):
            raise ValueError("hourly interpolation bounds must be exact UTC hours")
        hours = pd.date_range(start_time, end_time, freq="h", inclusive="left")
        return self.positions_at_times(hours)

    def match_positions(
        self,
        positions: pd.DataFrame,
        *,
        time_col: str = "position_time_utc",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        radius_500_km: float = 500.0,
        radius_300_km: float = 300.0,
    ) -> pd.DataFrame:
        """Match a batch of ship positions to all simultaneous cyclone centres."""

        required = {time_col, lat_col, lon_col}
        missing = sorted(required - set(positions.columns))
        if missing:
            raise ValueError(f"ship position table missing columns: {missing}")
        if "_input_index" in positions.columns:
            raise ValueError("ship position table reserves the column name _input_index")
        if not (0 < radius_300_km <= radius_500_km):
            raise ValueError("cyclone radii must satisfy 0 < radius_300 <= radius_500")
        output = positions.copy().reset_index(drop=True)
        output.insert(0, "_input_index", positions.index.to_numpy(copy=True))
        if output.empty:
            return _empty_position_match(output)
        parsed_time = pd.to_datetime(output[time_col], utc=True, errors="coerce")
        raw_lat = pd.to_numeric(output[lat_col], errors="coerce").to_numpy(np.float64)
        raw_lon = pd.to_numeric(output[lon_col], errors="coerce").to_numpy(np.float64)
        valid = (
            parsed_time.notna().to_numpy()
            & np.isfinite(raw_lat)
            & np.isfinite(raw_lon)
            & (raw_lat >= -90.0)
            & (raw_lat <= 90.0)
            & (raw_lon >= -180.0)
            & (raw_lon <= 180.0)
        )
        n = len(output)
        min_distance = np.full(n, np.nan, dtype=np.float64)
        within_500 = np.zeros(n, dtype=bool)
        within_300 = np.zeros(n, dtype=bool)
        max_wind = np.full(n, np.nan, dtype=np.float64)
        sid_sets: list[tuple[str, ...]] = [tuple() for _ in range(n)]
        storm_present = np.zeros(n, dtype=bool)

        if valid.any():
            valid_rows = np.flatnonzero(valid)
            ship = pd.DataFrame(
                {
                    "_row": valid_rows,
                    "target_time_utc": parsed_time.iloc[valid_rows].array,
                    "ship_lat": raw_lat[valid_rows],
                    "ship_lon": normalize_longitude(raw_lon[valid_rows]),
                }
            )
            storm = self.positions_at_times(ship["target_time_utc"].drop_duplicates())
            if not storm.empty:
                pairs = ship.merge(storm, how="inner", on="target_time_utc", sort=False)
                pairs["distance_km"] = great_circle_km(
                    pairs["ship_lon"].to_numpy(np.float64),
                    pairs["ship_lat"].to_numpy(np.float64),
                    pairs["storm_lon"].to_numpy(np.float64),
                    pairs["storm_lat"].to_numpy(np.float64),
                )
                for row_id, group in pairs.groupby("_row", sort=False, observed=True):
                    row_id = int(row_id)
                    distance = group["distance_km"].to_numpy(np.float64)
                    min_distance[row_id] = float(np.min(distance))
                    storm_present[row_id] = True
                    inside500 = distance <= (
                        float(radius_500_km) + RADIUS_COMPARISON_TOLERANCE_KM
                    )
                    inside300 = distance <= (
                        float(radius_300_km) + RADIUS_COMPARISON_TOLERANCE_KM
                    )
                    within_500[row_id] = bool(inside500.any())
                    within_300[row_id] = bool(inside300.any())
                    if inside500.any():
                        sids = group.loc[inside500, "SID"].dropna().astype(str).unique()
                        sid_sets[row_id] = tuple(sorted(sids))
                        winds = group.loc[inside500, "storm_wind_kt"].to_numpy(np.float64)
                        if np.isfinite(winds).any():
                            max_wind[row_id] = float(np.nanmax(winds))

        output["position_valid"] = valid
        output["tc_storm_present"] = storm_present
        output["tc_min_center_distance_km"] = min_distance
        output["tc_within_500km"] = within_500
        output["tc_within_300km"] = within_300
        output["tc_max_storm_wind_kt_500km"] = max_wind
        output["tc_sids_500km"] = sid_sets
        output["tc_match_status"] = np.where(
            ~valid, "invalid_position", np.where(storm_present, "valid_storm", "valid_no_storm")
        )
        return output


def _empty_position_match(output: pd.DataFrame) -> pd.DataFrame:
    output["position_valid"] = pd.Series(dtype="bool")
    output["tc_storm_present"] = pd.Series(dtype="bool")
    output["tc_min_center_distance_km"] = pd.Series(dtype="float64")
    output["tc_within_500km"] = pd.Series(dtype="bool")
    output["tc_within_300km"] = pd.Series(dtype="bool")
    output["tc_max_storm_wind_kt_500km"] = pd.Series(dtype="float64")
    output["tc_sids_500km"] = pd.Series(dtype="object")
    output["tc_match_status"] = pd.Series(dtype="string")
    return output


TC_OUTPUT_COLUMNS = (
    "tropical_cyclone_exposure_hours_500km",
    "tropical_cyclone_exposure_hours_300km",
    "tropical_cyclone_min_center_distance_km",
    "tropical_cyclone_max_storm_wind_kt_500km",
    "tropical_cyclone_count_500km",
    "high_wave_tc_overlap_hours_ge4m_500km",
)


def aggregate_candidate_tc_segments(
    segments: pd.DataFrame,
    *,
    candidate_col: str = "candidate_id",
    weight_col: str = "segment_weight_hours",
    high_wave_threshold_m: float = 4.0,
) -> pd.DataFrame:
    """Aggregate matched, non-overlapping sea segments to voyage-level TC facts."""

    required = {
        candidate_col,
        weight_col,
        "position_valid",
        "tc_min_center_distance_km",
        "tc_within_500km",
        "tc_within_300km",
        "tc_max_storm_wind_kt_500km",
        "tc_sids_500km",
        "swh_m",
    }
    missing = sorted(required - set(segments.columns))
    if missing:
        raise ValueError(f"matched segment table missing columns: {missing}")
    if high_wave_threshold_m < 0:
        raise ValueError("high-wave threshold must be nonnegative")
    rows: list[dict[str, object]] = []
    for candidate_id, group in segments.groupby(candidate_col, sort=False, observed=True):
        weight = pd.to_numeric(group[weight_col], errors="coerce").to_numpy(np.float64)
        if (~np.isfinite(weight) | (weight <= 0.0)).any():
            raise ValueError(f"candidate {candidate_id} has invalid segment weights")
        position_valid = group["position_valid"].fillna(False).to_numpy(bool)
        weather_track_hours = float(weight[position_valid].sum(dtype=np.float64))
        row: dict[str, object] = {
            candidate_col: candidate_id,
            "weather_track_hours": weather_track_hours,
        }
        if weather_track_hours == 0.0:
            row.update({column: np.nan for column in TC_OUTPUT_COLUMNS})
            row["tropical_cyclone_count_500km"] = pd.NA
            rows.append(row)
            continue

        within500 = position_valid & group["tc_within_500km"].fillna(False).to_numpy(bool)
        within300 = position_valid & group["tc_within_300km"].fillna(False).to_numpy(bool)
        if np.any(within300 & ~within500):
            raise ValueError(
                f"candidate {candidate_id} violates nested 300km <= 500km exposure"
            )
        exposure500 = float(weight[within500].sum(dtype=np.float64))
        exposure300 = float(weight[within300].sum(dtype=np.float64))
        distance = pd.to_numeric(
            group["tc_min_center_distance_km"], errors="coerce"
        ).to_numpy(np.float64)
        wind = pd.to_numeric(
            group["tc_max_storm_wind_kt_500km"], errors="coerce"
        ).to_numpy(np.float64)
        row["tropical_cyclone_exposure_hours_500km"] = exposure500
        row["tropical_cyclone_exposure_hours_300km"] = exposure300
        row["tropical_cyclone_min_center_distance_km"] = (
            float(np.nanmin(distance[position_valid]))
            if np.isfinite(distance[position_valid]).any()
            else np.nan
        )
        row["tropical_cyclone_max_storm_wind_kt_500km"] = (
            float(np.nanmax(wind[within500]))
            if np.isfinite(wind[within500]).any()
            else np.nan
        )
        distinct_sids: set[str] = set()
        for value in group.loc[within500, "tc_sids_500km"]:
            if value is None or value is pd.NA:
                continue
            if isinstance(value, str):
                distinct_sids.add(value)
            else:
                distinct_sids.update(str(sid) for sid in value)
        row["tropical_cyclone_count_500km"] = len(distinct_sids)

        if exposure500 == 0.0:
            overlap = 0.0
        else:
            exposed_swh = pd.to_numeric(group.loc[within500, "swh_m"], errors="coerce").to_numpy(
                np.float64
            )
            if not np.isfinite(exposed_swh).all():
                overlap = np.nan
            else:
                overlap = float(
                    weight[within500][exposed_swh >= float(high_wave_threshold_m)].sum(
                        dtype=np.float64
                    )
                )
        row["high_wave_tc_overlap_hours_ge4m_500km"] = overlap
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=[candidate_col, "weather_track_hours", *TC_OUTPUT_COLUMNS])
    result["tropical_cyclone_count_500km"] = pd.array(
        result["tropical_cyclone_count_500km"], dtype="UInt16"
    )
    return result


def snapshot_census(
    fixes: pd.DataFrame, index: CycloneTrackIndex, audit: Mapping[str, object]
) -> dict[str, object]:
    """Return JSON-safe source and interpolation counts for a run manifest."""

    return {
        **dict(audit),
        "interpolable_interval_count": index.n_intervals,
        "retained_exact_fix_count": index.n_fixes,
        "retained_sid_count": int(fixes["SID"].nunique()),
    }
