"""Centered UTC-hour sea-leg segments and AIS interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd


HOUR = pd.Timedelta(hours=1)
HALF_HOUR = pd.Timedelta(minutes=30)


def centered_hour_segments(legs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per non-empty sea-leg intersection with a centered hour.

    The integer-hour anchor ``h`` labels ``[h-30m, h+30m)``.  A segment's
    position/TC representative time is its own intersection midpoint, while
    ERA5 is later matched exactly at ``h``.  Segments are deliberately not
    collapsed here because one candidate-hour may contain two sea legs with a
    stop gap between them.
    """

    required = {"physical_leg_id", "mmsi", "departure_time", "arrival_time"}
    missing = sorted(required - set(legs.columns))
    if missing:
        raise ValueError(f"legs missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for leg in legs.itertuples(index=False):
        values = leg._asdict()
        start = pd.Timestamp(values["departure_time"])
        end = pd.Timestamp(values["arrival_time"])
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        if not end > start:
            raise ValueError("sea legs must have positive duration")
        # The inequalities are strict at the empty-intersection boundary:
        # h+30m > start and h-30m < end.  These forms therefore exclude an
        # anchor whose centered hour merely touches a leg at exactly :30.
        first_anchor = (start + HALF_HOUR).floor("h")
        last_anchor = (end - HALF_HOUR).ceil("h")
        for anchor in pd.date_range(first_anchor, last_anchor, freq="h"):
            segment_start = max(start, anchor - HALF_HOUR)
            segment_end = min(end, anchor + HALF_HOUR)
            if segment_end <= segment_start:
                continue
            weight_ns = int((segment_end - segment_start).value)
            weight = weight_ns / float(HOUR.value)
            if not 0.0 < weight <= 1.0 + 1e-12:
                raise ValueError("centered-hour segment weight is outside (0,1]")
            rows.append(
                {
                    "physical_leg_id": values["physical_leg_id"],
                    "mmsi": str(values["mmsi"]),
                    "hour_anchor_utc": anchor,
                    "segment_start_utc": segment_start,
                    "segment_end_utc": segment_end,
                    "position_time_utc": segment_start + (segment_end - segment_start) / 2,
                    "segment_weight_ns": weight_ns,
                    "segment_weight_hours": float(weight),
                }
            )
    return pd.DataFrame(rows)


def required_hour_anchor_bounds(legs: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return anchor bounds directly from leg endpoints without materializing hours."""

    if legs.empty:
        raise ValueError("cannot determine hour bounds from zero sea legs")
    departure = pd.to_datetime(legs["departure_time"], utc=True, errors="coerce")
    arrival = pd.to_datetime(legs["arrival_time"], utc=True, errors="coerce")
    if departure.isna().any() or arrival.isna().any():
        raise ValueError("leg endpoints contain invalid timestamps")
    first = (departure.min() + HALF_HOUR).floor("h")
    last = (arrival.max() - HALF_HOUR).ceil("h")
    if last < first:
        raise ValueError("leg endpoints do not produce a valid hour range")
    return first, last


def assign_continuity_segments(points: pd.DataFrame, maximum_gap_hours: float = 4.0) -> pd.DataFrame:
    """Sort/deduplicate AIS and rebuild E020 continuity independently of source traj_id."""

    required = {"vessel_key", "UTC", "lat", "lon"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"AIS points missing columns: {missing}")
    work = points.copy()
    vessel = work["vessel_key"].astype("string").str.strip()
    if vessel.isna().any() or vessel.eq("").any() or vessel.nunique() != 1:
        raise ValueError("assign_continuity_segments requires exactly one nonblank vessel_key")
    work["UTC"] = pd.to_datetime(work["UTC"], utc=True, errors="coerce")
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.loc[
        work["UTC"].notna()
        & np.isfinite(work["lat"])
        & np.isfinite(work["lon"])
        & work["lat"].between(-90.0, 90.0)
        & work["lon"].between(-180.0, 180.0)
    ].sort_values("UTC", kind="stable")
    work = work.drop_duplicates("UTC", keep="first").reset_index(drop=True)
    gap = work["UTC"].diff().dt.total_seconds().div(3600.0)
    work["e020_segment"] = gap.gt(float(maximum_gap_hours)).cumsum().astype(np.int64)
    return work


def interpolate_positions(points: pd.DataFrame, targets: pd.Series, maximum_gap_hours: float = 4.0) -> pd.DataFrame:
    """Interpolate one vessel at target instants only inside one E020 segment."""

    work = assign_continuity_segments(points, maximum_gap_hours)
    target = pd.Series(pd.to_datetime(targets, utc=True, errors="coerce")).reset_index(drop=True)
    result_lat = np.full(len(target), np.nan, dtype=np.float64)
    result_lon = np.full(len(target), np.nan, dtype=np.float64)
    result = pd.DataFrame({"position_time_utc": target, "lat": result_lat, "lon": result_lon})
    if work.empty:
        return result
    times = work["UTC"].astype("int64").to_numpy(np.int64)
    target_ns = target.astype("int64").to_numpy(np.int64)
    right = np.searchsorted(times, target_ns, side="left")
    exact = (right < len(times)) & (times[np.minimum(right, len(times) - 1)] == target_ns)
    if exact.any():
        result_lat[exact] = work["lat"].to_numpy(float)[right[exact]]
        exact_lon = work["lon"].to_numpy(float)[right[exact]]
        result_lon[exact] = ((exact_lon + 180.0) % 360.0) - 180.0
    between = (~exact) & (right > 0) & (right < len(times)) & target.notna().to_numpy()
    if between.any():
        idx = np.flatnonzero(between)
        left_index = right[idx] - 1
        right_index = right[idx]
        segments = work["e020_segment"].to_numpy(np.int64)
        same = segments[left_index] == segments[right_index]
        idx, left_index, right_index = idx[same], left_index[same], right_index[same]
        if len(idx):
            denominator = (times[right_index] - times[left_index]).astype(np.float64)
            fraction = (target_ns[idx] - times[left_index]) / denominator
            lat = work["lat"].to_numpy(float)
            lon = work["lon"].to_numpy(float)
            delta_lon = ((lon[right_index] - lon[left_index] + 180.0) % 360.0) - 180.0
            result_lat[idx] = lat[left_index] + fraction * (lat[right_index] - lat[left_index])
            result_lon[idx] = ((lon[left_index] + fraction * delta_lon + 180.0) % 360.0) - 180.0
    result["lat"] = result_lat
    result["lon"] = result_lon
    return result
