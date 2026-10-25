#!/usr/bin/env python3
"""Bound wave and cyclone exposure over positions reachable during long AIS gaps.

Read-only diagnostic.  It writes no product root and computes no accepted value.

`20_locate_dark_hours.py` provides a route-neutral grid attribution of long AIS
gaps. This script tests how wide weather exposure bounds become when each
unknown position is constrained by its two bounding fixes and the speed cap.
It does not assign routes, corridors, or treatment categories. Unknown exposure
remains unknown in the accepted products.

That proposal lives or dies on one number nobody here has measured -- **how wide
the bounds actually are.**  A bound of [1.8 m, 2.1 m] is as good as a point
estimate.  A bound of [0.4 m, 6.2 m] is a statement that we know nothing,
dressed up as a result.

Method, per dark gap with bounding fixes A at t0 and B at t1, at each whole
hour h in between:

    reachable(h) = { x : gc(A,x) <= v*(h-t0)  and  gc(x,B) <= v*(t1-h) }

Two leashes, one lengthening from where the ship was, one shortening towards
where it must arrive.  Both are short near the ends, so the region pinches to a
point at each fix and is widest in the middle.  `v` is not invented here: it is
The per-voyage speed cap, min(max(p95 * 1.25, 12 kn), 25 kn), comes from the
retained speed-bound settings so this diagnostic uses one consistent limit.

The region is then **enumerated, not sampled.**  Weather is only ever known per
ERA5 cell, so the number of distinct values the region can take is finite and
small: every 0.5-degree cell the region touches, once.  Touching is measured to
a cell's nearest edge rather than to its centre, in both directions -- a region
clipping a cell's corner really could put the ship there, and a region that ends
inside one cell must not pick up its neighbours.  Where a six-hour gap fits
inside one cell there is exactly one possible value and the interval is a point,
correctly and not by accident.

The straight line between the two fixes outranks the speed cap where they
disagree, because the vessel provably covered that distance in that time.  Legs
needing more than their own cap are marked and excluded from the headline: at a
median implied 24.8 knots and a maximum of 337, they are bad fixes rather than
fast ships, and their intervals are valid arithmetic over a false premise.

Dry cells are dropped rather than snapped to the nearest wet cell.  Snapping is
right when a position is known and the grid is coarse; here a dry cell is simply
somewhere the ship could not have been, and importing a wave height from outside
the region would widen the interval with a value no possible position holds.

Cyclones use the same enumerated cells and a different question, because 500 km
is a threshold rather than a field.  Each hour is classified three ways:

    every reachable cell within 500 km   -> exposed, whatever the ship did
    no reachable cell within 500 km      -> not exposed, whatever the ship did
    otherwise                            -> undetermined by this evidence

For cyclones the share of undetermined hours *is* the result; a width in
kilometres would hide it.

One deliberate looseness: each hour is bounded independently, so the interval
admits a ship that is at the roughest reachable water at hour 5 and the roughest
again at hour 6 even when no single track could visit both.  The true interval
is therefore narrower than the one reported.  That is the safe direction, and it
gives a cheap decision rule -- if this loose interval is already tight enough to
use, the expensive path-optimisation version is unnecessary.

    python3 30_bound_dark_hour_wave_exposure.py --months 2025-03 2025-09 \\
        --coverage-run <run> --out <dir>

Environment: sinbue_env (Python 3.10)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODING_ROOT = HERE.parents[3]
WEATHER_STAGE_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
LEGACY_SPEED_CONFIG = HERE / "config/legacy_speed_bound.json"
WEATHER_CONFIG = json.loads(
    (WEATHER_STAGE_ROOT / "config/2025_ten_minute_weather.json").read_text(encoding="utf-8")
)

if str(WEATHER_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(WEATHER_STAGE_ROOT))
ENVIRONMENT_STAGE_ROOT = CODING_ROOT / "pipeline/00_environment_and_config"
if str(ENVIRONMENT_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_STAGE_ROOT))

from weather_core.era5_match import Era5MonthlyMatcher  # noqa: E402
from weather_core.tropical_cyclone import (  # noqa: E402
    CycloneTrackIndex,
    load_ibtracs_snapshot,
)

from ais_identity import (  # noqa: E402
    deduplicate_vessel_ais,
    normalise_required_identifier,
    stable_vessel_buckets,
    validate_ais_physical_identity,
)
from accepted_candidate_root import load_accepted_2025_candidates  # noqa: E402
from weather_assets import (  # noqa: E402
    resolve_monthly_era5_assets,
    validate_month_time_grid,
)

MAXIMUM_INTERPOLATION_GAP_HOURS = 4.0
NS_PER_HOUR = 3_600 * 1_000_000_000
GAP_LIMIT_NS = np.int64(MAXIMUM_INTERPOLATION_GAP_HOURS * NS_PER_HOUR)

EARTH_RADIUS_KM = 6371.0088
KM_PER_NAUTICAL_MILE = 1.852
HIGH_WAVE_THRESHOLD_M = 4.0
TC_RADIUS_500_KM = 500.0
TC_RADIUS_300_KM = 300.0
TC_MAXIMUM_FIX_GAP_HOURS = 3.0


@dataclass(frozen=True)
class SpeedBoundDefinition:
    speed_floor_kn: float
    speed_ceiling_kn: float
    speed_margin: float
    continuous_fix_max_gap_hours: float
    minimum_speed_intervals: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SpeedBoundDefinition":
        speed = config["speed_bound"]
        return cls(
            speed_floor_kn=float(speed["floor_kn"]),
            speed_ceiling_kn=float(speed["ceiling_kn"]),
            speed_margin=float(speed["p95_margin"]),
            continuous_fix_max_gap_hours=float(speed["continuous_fix_max_gap_hours"]),
            minimum_speed_intervals=int(speed["minimum_intervals"]),
        )


def _nautical_miles(lat1, lon1, lat2, lon2):
    first = np.radians(lat1)
    second = np.radians(lat2)
    dlat = second - first
    dlon = np.radians(lon2 - lon1)
    haversine = (
        np.sin(dlat / 2) ** 2
        + np.cos(first) * np.cos(second) * np.sin(dlon / 2) ** 2
    )
    return (
        2
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        / KM_PER_NAUTICAL_MILE
    )


def voyage_speed_cap(
    lat: np.ndarray,
    lon: np.ndarray,
    times_ns: np.ndarray,
    definition: SpeedBoundDefinition,
) -> float:
    """Preserve the legacy dark-gap probe's generic per-voyage speed cap."""
    if len(lat) < 3:
        return definition.speed_ceiling_kn
    hours = np.diff(times_ns) / 3_600_000_000_000
    good = (hours > 0.05) & (
        hours < definition.continuous_fix_max_gap_hours
    )
    if int(good.sum()) < definition.minimum_speed_intervals:
        return definition.speed_ceiling_kn
    distance = _nautical_miles(
        lat[:-1][good], lon[:-1][good], lat[1:][good], lon[1:][good]
    )
    p95 = float(np.nanpercentile(distance / hours[good], 95))
    if not np.isfinite(p95):
        return definition.speed_ceiling_kn
    return float(
        np.clip(
            p95 * definition.speed_margin,
            definition.speed_floor_kn,
            definition.speed_ceiling_kn,
        )
    )

# ERA5 is the only source of wave values, so its grid defines how many distinct
# answers a region can possibly have.  A cell counts when the region touches any
# part of it, so the centre test is expanded by the cell's half-diagonal.
ERA5_CELL_DEGREES = 0.5
CELL_HALF_DIAGONAL_KM = float(
    np.hypot(ERA5_CELL_DEGREES / 2.0, ERA5_CELL_DEGREES / 2.0)
    * np.pi
    * EARTH_RADIUS_KM
    / 180.0
)
# Dry cells must fall out rather than borrow a neighbour's value.
NO_WET_CELL_SNAP_KM = 0.0

# The region is closed, and a leg with no slack -- one whose fixes demand its
# whole speed budget -- pinches onto the great circle exactly.  Every cell it
# passes through then sits on the boundary, where a bare `<=` on haversine
# arithmetic can round the entire region away and leave nothing enumerated.
# One metre of tolerance keeps the boundary inside without widening anything.
REACH_TOLERANCE_KM = 1e-3

# Enumeration is exhaustive until a region grows large enough that it is already
# uninformative; past this it is thinned and the hour is flagged, never silently
# treated as if it had been enumerated.
MAX_CELLS_PER_HOUR = 800
MAX_HOURS_PER_GAP = 24

IBTRACS_PATH = Path(WEATHER_CONFIG["ibtracs_path"])
IBTRACS_SHA256 = WEATHER_CONFIG["ibtracs_sha256"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def great_circle_km(
    lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray
) -> np.ndarray:
    """Haversine distance in kilometres, broadcasting like the inputs."""

    phi_a, phi_b = np.radians(lat_a), np.radians(lat_b)
    d_phi = phi_b - phi_a
    d_lam = np.radians(np.asarray(lon_b) - np.asarray(lon_a))
    h = (
        np.sin(d_phi / 2.0) ** 2
        + np.cos(phi_a) * np.cos(phi_b) * np.sin(d_lam / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def unit_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    cos_lat = np.cos(lat_r)
    return np.stack(
        [cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)], axis=-1
    )


def great_circle_point(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float, fraction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The constant-speed shortest-path position -- what interpolation assumes."""

    a = unit_xyz(np.array([lon_a]), np.array([lat_a]))[0]
    b = unit_xyz(np.array([lon_b]), np.array([lat_b]))[0]
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = np.arccos(dot)
    if np.sin(omega) < 1e-12:
        return np.full(fraction.shape, lat_a), np.full(fraction.shape, lon_a)
    w_a = np.sin((1.0 - fraction) * omega) / np.sin(omega)
    w_b = np.sin(fraction * omega) / np.sin(omega)
    point = w_a[:, None] * a[None, :] + w_b[:, None] * b[None, :]
    point /= np.linalg.norm(point, axis=1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(point[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(point[:, 1], point[:, 0]))
    return lat, lon


def reachable_cells(
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    radius_a_km: float,
    radius_b_km: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Every ERA5 cell the reachable region touches, as centres.

    Returns ``(latitudes, longitudes, capped)``.  ``capped`` says the region was
    too large to enumerate and the returned cells are a thinned subset, so the
    interval built from them is a floor on the true one rather than the truth.
    """

    reach_km = max(radius_a_km, radius_b_km) + CELL_HALF_DIAGONAL_KM + REACH_TOLERANCE_KM
    span_deg = min(np.degrees(reach_km / EARTH_RADIUS_KM), 90.0)
    lat_lo = max(min(lat_a, lat_b) - span_deg, -90.0)
    lat_hi = min(max(lat_a, lat_b) + span_deg, 90.0)
    mid_lat = np.clip((lat_a + lat_b) / 2.0, -85.0, 85.0)
    lon_span = min(span_deg / max(np.cos(np.radians(mid_lat)), 0.09), 180.0)

    # Snap the search box outwards onto the ERA5 grid so no touched cell centre
    # can fall outside it.
    step = ERA5_CELL_DEGREES
    lats = np.arange(np.floor(lat_lo / step) * step, lat_hi + step, step)
    lats = lats[(lats >= -90.0) & (lats <= 90.0)]
    # Work in a longitude frame centred on A so a region crossing the date line
    # stays contiguous; wrap once at the end.
    lon_b_rel = ((lon_b - lon_a + 180.0) % 360.0) - 180.0
    lon_lo = min(0.0, lon_b_rel) - lon_span
    lon_hi = max(0.0, lon_b_rel) + lon_span
    offsets = np.arange(np.floor(lon_lo / step) * step, lon_hi + step, step)
    lons = np.round((offsets + lon_a) / step) * step
    lons = ((lons + 180.0) % 360.0) - 180.0
    if lats.size == 0 or lons.size == 0:
        return np.array([lat_a]), np.array([lon_a]), False

    grid_lat, grid_lon = (x.ravel() for x in np.meshgrid(lats, lons, indexing="ij"))
    grid_lon_rel = ((grid_lon - lon_a + 180.0) % 360.0) - 180.0

    def touches(lat_f: float, lon_f_rel: float, radius_km: float) -> np.ndarray:
        """Does a cap of this radius around a focus reach into each cell?

        Measured to the *nearest corner or edge* of the cell rather than to its
        centre.  Expanding every centre by a fixed half-diagonal instead would
        admit neighbours the region never reaches, and would destroy the case
        that matters most: a short gap confined to one cell has exactly one
        possible value, and must report a width of zero rather than a spurious
        spread borrowed from cells the ship could not enter.
        """

        half = ERA5_CELL_DEGREES / 2.0
        nearest_lat = np.clip(lat_f, grid_lat - half, grid_lat + half)
        nearest_lon = np.clip(lon_f_rel, grid_lon_rel - half, grid_lon_rel + half)
        return (
            great_circle_km(lat_f, lon_f_rel, nearest_lat, nearest_lon)
            <= radius_km + REACH_TOLERANCE_KM
        )

    touched = touches(lat_a, 0.0, radius_a_km) & touches(
        lat_b, lon_b_rel, radius_b_km
    )
    keep_lat, keep_lon = grid_lat[touched], grid_lon[touched]
    if keep_lat.size == 0:
        # Both fixes sit in one cell and the box rounded away from it; the two
        # observed positions are then the whole of what is reachable.
        return np.array([lat_a, lat_b]), np.array([lon_a, lon_b]), False
    if keep_lat.size > MAX_CELLS_PER_HOUR:
        take = np.linspace(0, keep_lat.size - 1, MAX_CELLS_PER_HOUR).astype(int)
        return keep_lat[take], keep_lon[take], True
    return keep_lat, keep_lon, False


def extract_gaps(
    bucket_dir: Path, candidates: pd.DataFrame, definition: Any
) -> pd.DataFrame:
    """Every >4 h gap inside a candidate window, with its two bounding fixes."""

    parts = sorted(bucket_dir.glob("*.feather"))
    if not parts:
        return pd.DataFrame()
    points = pd.concat(
        [feather.read_table(p, memory_map=True).to_pandas() for p in parts],
        ignore_index=True,
    )
    identity = candidates[["mmsi", "vessel_uid_clean"]].drop_duplicates()
    points, _ = validate_ais_physical_identity(points, identity)
    points, _ = deduplicate_vessel_ais(points)

    by_vessel = {k: g for k, g in points.groupby("mmsi", sort=False, observed=True)}
    rows: list[dict[str, Any]] = []
    for mmsi, group in candidates.groupby("mmsi", sort=False, observed=True):
        vessel = by_vessel.get(mmsi)
        if vessel is None or vessel.empty:
            continue
        vessel = vessel.sort_values("UTC", kind="stable")
        utc = vessel["UTC"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        lat_all = vessel["lat"].to_numpy(dtype=np.float64)
        lon_all = vessel["lon"].to_numpy(dtype=np.float64)

        starts = group["departure_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        ends = group["arrival_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        left = np.searchsorted(utc, starts, side="left")
        right = np.searchsorted(utc, ends, side="right")

        for i, candidate_id in enumerate(group["candidate_id"].to_numpy()):
            lo, hi = left[i], right[i]
            if hi - lo < 2:
                continue
            t = utc[lo:hi]
            la, ln = lat_all[lo:hi], lon_all[lo:hi]
            # The speed cap is a property of the voyage, not of the gap: it is
            # estimated from the fixes we do have, exactly as E024 does.
            cap_kn = voyage_speed_cap(la, ln, t, definition)
            for j in np.flatnonzero(np.diff(t) > GAP_LIMIT_NS):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "gap_ordinal": int(j),
                        "t0_ns": int(t[j]),
                        "t1_ns": int(t[j + 1]),
                        "lat0": float(la[j]),
                        "lon0": float(ln[j]),
                        "lat1": float(la[j + 1]),
                        "lon1": float(ln[j + 1]),
                        "gap_hours": float((t[j + 1] - t[j]) / NS_PER_HOUR),
                        "direct_km": float(
                            great_circle_km(la[j], ln[j], la[j + 1], ln[j + 1])
                        ),
                        "speed_cap_kn": float(cap_kn),
                        "implied_kn": float(
                            great_circle_km(la[j], ln[j], la[j + 1], ln[j + 1])
                            / KM_PER_NAUTICAL_MILE
                            / ((t[j + 1] - t[j]) / NS_PER_HOUR)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_reachable_positions(gaps: pd.DataFrame) -> pd.DataFrame:
    """One row per (gap, hour, reachable ERA5 cell), plus the direct-path cell."""

    frames: list[pd.DataFrame] = []
    for gap in gaps.itertuples(index=False):
        t0, t1 = pd.Timestamp(gap.t0_ns, tz="UTC"), pd.Timestamp(gap.t1_ns, tz="UTC")
        hours = pd.date_range(t0.ceil("h"), t1.floor("h"), freq="h")
        hours = hours[(hours > t0) & (hours < t1)]
        if hours.empty:
            continue
        if len(hours) > MAX_HOURS_PER_GAP:
            hours = hours[np.linspace(0, len(hours) - 1, MAX_HOURS_PER_GAP).astype(int)]

        span_ns = gap.t1_ns - gap.t0_ns
        fraction = np.array([(h.value - gap.t0_ns) / span_ns for h in hours])
        gc_lat, gc_lon = great_circle_point(
            gap.lat0, gap.lon0, gap.lat1, gap.lon1, fraction
        )
        # The ship demonstrably covered this distance in this time, so the
        # straight line between the fixes is evidence about what it can do and
        # outranks the cap estimated from its other intervals.  Without this the
        # region would exclude a path the vessel provably took, and the interval
        # would not contain the truth.  `implied_exceeds_cap` marks these: a
        # 337-knot leg is a bad fix or a shared MMSI, not a fast ship, and the
        # interval it produces should be read as a defect flag, not a bound.
        effective_kn = max(gap.speed_cap_kn, gap.implied_kn)
        speed_km_h = effective_kn * KM_PER_NAUTICAL_MILE

        for k, hour in enumerate(hours):
            elapsed = (hour.value - gap.t0_ns) / NS_PER_HOUR
            remaining = (gap.t1_ns - hour.value) / NS_PER_HOUR
            lat_c, lon_c, capped = reachable_cells(
                gap.lat0,
                gap.lon0,
                gap.lat1,
                gap.lon1,
                speed_km_h * elapsed,
                speed_km_h * remaining,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "candidate_id": gap.candidate_id,
                        "gap_ordinal": gap.gap_ordinal,
                        "hour_anchor_utc": hour,
                        "lat": np.concatenate([[gc_lat[k]], lat_c]),
                        "lon": np.concatenate([[gc_lon[k]], lon_c]),
                        "is_direct_path": np.concatenate(
                            [[True], np.zeros(lat_c.size, dtype=bool)]
                        ),
                        "cells_capped": capped,
                    }
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarise(matched: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one interval per gap-hour, then to one row per gap.

    ``matched`` must already be restricted to water: a dry cell is somewhere the
    ship could not have been, so it carries neither a wave height nor a cyclone
    distance.
    """

    if matched.empty:
        return pd.DataFrame()
    key = ["candidate_id", "gap_ordinal", "hour_anchor_utc"]
    region = matched[~matched["is_direct_path"]]
    if region.empty:
        return pd.DataFrame()

    hourly = (
        region.groupby(key, sort=False)
        .agg(
            swh_lo=("swh_m", "min"),
            swh_hi=("swh_m", "max"),
            swh_mid=("swh_m", "median"),
            cells=("swh_m", "size"),
            capped=("cells_capped", "max"),
            # A cyclone hour is settled only when every reachable cell agrees.
            tc500_any=("within_500", "max"),
            tc500_all=("within_500", "min"),
            tc300_any=("within_300", "max"),
            tc300_all=("within_300", "min"),
            tc_distance_lo=("tc_min_distance_km", "min"),
            tc_distance_hi=("tc_min_distance_km", "max"),
            tc_wind_lo=("tc_max_wind_kt", "min"),
            tc_wind_hi=("tc_max_wind_kt", "max"),
        )
        .reset_index()
    )
    direct = (
        matched[matched["is_direct_path"]]
        .groupby(key, sort=False)["swh_m"]
        .first()
        .rename("swh_direct")
        .reset_index()
    )
    hourly = hourly.merge(direct, on=key, how="left")

    # With exhaustive enumeration the direct path's own cell is in the region by
    # construction, so this must hold wherever the region was not thinned.  It is
    # an invariant check, not a tuning knob.
    hourly["direct_inside"] = (
        hourly["swh_direct"].isna()
        | hourly["swh_direct"].between(hourly["swh_lo"], hourly["swh_hi"])
    )
    hourly["width_m"] = hourly["swh_hi"] - hourly["swh_lo"]
    hourly["wave_settled"] = hourly["cells"] == 1
    hourly["tc_undetermined"] = hourly["tc500_any"] & ~hourly["tc500_all"]
    hourly["wave_must_exceed"] = hourly["swh_lo"] >= HIGH_WAVE_THRESHOLD_M
    hourly["wave_could_exceed"] = hourly["swh_hi"] >= HIGH_WAVE_THRESHOLD_M

    return (
        hourly.groupby(["candidate_id", "gap_ordinal"], sort=False)
        .agg(
            hours_bounded=("hour_anchor_utc", "size"),
            cells_median=("cells", "median"),
            cells_max=("cells", "max"),
            capped_share=("capped", "mean"),
            single_cell_share=("wave_settled", "mean"),
            direct_inside_share=("direct_inside", "mean"),
            swh_lo_mean=("swh_lo", "mean"),
            swh_hi_mean=("swh_hi", "mean"),
            swh_mid_mean=("swh_mid", "mean"),
            swh_direct_mean=("swh_direct", "mean"),
            width_mean_m=("width_m", "mean"),
            width_max_m=("width_m", "max"),
            high_wave_hours_lo=("wave_must_exceed", "sum"),
            high_wave_hours_hi=("wave_could_exceed", "sum"),
            tc500_hours_lo=("tc500_all", "sum"),
            tc500_hours_hi=("tc500_any", "sum"),
            tc300_hours_lo=("tc300_all", "sum"),
            tc300_hours_hi=("tc300_any", "sum"),
            tc_undetermined_hours=("tc_undetermined", "sum"),
            # Closest approach is itself a minimum over the voyage, so even the
            # least favourable position still takes the min across hours.
            tc_min_distance_lo=("tc_distance_lo", "min"),
            tc_min_distance_hi=("tc_distance_hi", "min"),
            tc_max_wind_lo=("tc_wind_lo", "max"),
            tc_max_wind_hi=("tc_wind_hi", "max"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--months",
        nargs="+",
        required=True,
        help="explicit 2025 calendar months to inspect",
    )
    parser.add_argument("--coverage-run", type=Path, required=True)
    parser.add_argument("--gaps-per-month", type=int, default=1500)
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.workers > 12:
        raise SystemExit("CONTRACT caps this project at 12 workers")

    bucket_root = args.coverage_run / "vessel_buckets"
    if not (bucket_root / "_shuffle_complete.json").exists():
        raise SystemExit(f"no completed shuffle at {bucket_root}")
    args.out.mkdir(parents=True, exist_ok=True)

    definition = SpeedBoundDefinition.from_config(
        json.loads(LEGACY_SPEED_CONFIG.read_text(encoding="utf-8"))
    )
    print(
        f"speed cap: p95 x {definition.speed_margin} clipped to "
        f"[{definition.speed_floor_kn}, {definition.speed_ceiling_kn}] kn "
        "(retained speed-bound settings)"
    )
    print(f"cell half-diagonal: {CELL_HALF_DIAGONAL_KM:.2f} km")

    fixes, ibtracs_audit = load_ibtracs_snapshot(
        IBTRACS_PATH, expected_sha256=IBTRACS_SHA256
    )
    cyclones = CycloneTrackIndex(
        fixes, maximum_fix_gap_hours=TC_MAXIMUM_FIX_GAP_HOURS
    )
    print(f"IBTrACS: {cyclones.n_fixes:,} fixes, {cyclones.n_intervals:,} intervals")

    if any(not month.startswith("2025-") for month in args.months):
        raise SystemExit("E001 exposure bounds read the accepted 2025 candidate root only")
    candidates = load_accepted_2025_candidates(
        ("candidate_id", "mmsi", "vessel_uid_clean", "departure_time", "arrival_time")
    )
    candidates["mmsi"] = normalise_required_identifier(candidates["mmsi"], "mmsi")
    candidates["vessel_uid_clean"] = normalise_required_identifier(
        candidates["vessel_uid_clean"], "vessel_uid_clean"
    )
    candidates["vessel_bucket"] = stable_vessel_buckets(candidates["mmsi"], args.buckets)

    # Only voyages overlapping a requested month can contribute a matched hour,
    # and the buckets hold whole years.
    overlaps = np.zeros(len(candidates), dtype=bool)
    for period in pd.PeriodIndex(args.months, freq="M"):
        overlaps |= (
            candidates["departure_time"] <= period.end_time.tz_localize("UTC")
        ) & (candidates["arrival_time"] >= period.start_time.tz_localize("UTC"))
    candidates = candidates[overlaps].reset_index(drop=True)
    print(f"extracting >4 h gaps from {len(candidates):,} candidates ...")

    gap_frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                extract_gaps,
                bucket_root / f"bucket_{int(bucket):04d}",
                group.reset_index(drop=True),
                definition,
            ): bucket
            for bucket, group in candidates.groupby("vessel_bucket", sort=True)
        }
        for n, future in enumerate(as_completed(futures), start=1):
            got = future.result()
            if not got.empty:
                gap_frames.append(got)
            if n % 16 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} buckets")
    gaps = pd.concat(gap_frames, ignore_index=True)
    gaps["month"] = pd.to_datetime(gaps["t0_ns"], utc=True).dt.strftime("%Y-%m")
    print(f"total >4 h gaps: {len(gaps):,}")

    rng = np.random.default_rng(args.seed)
    picked = []
    for month in args.months:
        pool_month = gaps[gaps["month"] == month]
        if pool_month.empty:
            print(f"  {month}: no gaps")
            continue
        take = min(len(pool_month), max(args.gaps_per_month, 1))
        picked.append(pool_month.iloc[rng.choice(len(pool_month), size=take, replace=False)])
        print(f"  {month}: pool {len(pool_month):,}")
    sample = pd.concat(picked, ignore_index=True)
    print(f"sampled gaps: {len(sample):,}")
    print(sample.groupby("month").size().to_string())

    results, audits = [], []
    for month in args.months:
        block = sample[sample["month"] == month]
        if block.empty:
            continue
        year, mon = (int(x) for x in month.split("-"))
        assets = resolve_monthly_era5_assets(year=year, month=mon)
        positions = build_reachable_positions(block)
        # A gap can run past the end of its month; ERA5 tiles are monthly, so
        # those hours are dropped rather than matched against the wrong file.
        inside = positions["hour_anchor_utc"].dt.strftime("%Y-%m") == month
        outside_month = int((~inside).sum())
        positions = positions[inside].reset_index(drop=True)
        print(
            f"{month}: {len(positions):,} reachable cells "
            f"({outside_month:,} outside the month, dropped), {len(assets)} ERA5 tiles"
        )

        with Era5MonthlyMatcher(
            assets, maximum_wet_cell_snap_km=NO_WET_CELL_SNAP_KM
        ) as matcher:
            validate_month_time_grid(matcher, year, mon)
            matched = matcher.match(positions)
        dry = int(matched["swh_m"].isna().sum())
        matched = matched[matched["swh_m"].notna()].reset_index(drop=True)
        print(f"  {dry:,} dry or unmatched cells dropped, {len(matched):,} remain")

        tc = cyclones.match_positions(
            matched.rename(
                columns={
                    "hour_anchor_utc": "position_time_utc",
                    "lat": "latitude",
                    "lon": "longitude",
                }
            )[["position_time_utc", "latitude", "longitude"]],
            radius_500_km=TC_RADIUS_500_KM,
            radius_300_km=TC_RADIUS_300_KM,
        )
        matched["within_500"] = tc["tc_within_500km"].to_numpy()
        matched["within_300"] = tc["tc_within_300km"].to_numpy()
        # No storm within 500 km leaves the distance NaN and the wind NaN.  The
        # distance stays NaN so it cannot masquerade as a close approach; the
        # wind becomes zero because "no storm" is a wind of zero, not a gap.
        matched["tc_min_distance_km"] = tc["tc_min_center_distance_km"].to_numpy()
        matched["tc_max_wind_kt"] = (
            tc["tc_max_storm_wind_kt_500km"].fillna(0.0).to_numpy()
        )

        per_gap = summarise(matched)
        per_gap["month"] = month
        results.append(
            per_gap.merge(
                block[
                    [
                        "candidate_id",
                        "gap_ordinal",
                        "gap_hours",
                        "direct_km",
                        "speed_cap_kn",
                        "implied_kn",
                    ]
                ],
                on=["candidate_id", "gap_ordinal"],
                how="left",
            )
        )
        audits.append(
            {
                "month": month,
                "gaps": int(len(block)),
                "reachable_cells_matched": int(len(matched)),
                "cells_outside_month_dropped": outside_month,
                "dry_cells_dropped": dry,
                "era5_tiles": [str(p) for p in assets],
            }
        )

    out = pd.concat(results, ignore_index=True)
    out["implied_exceeds_cap"] = out["implied_kn"] > out["speed_cap_kn"] + 1e-9
    out.to_parquet(args.out / "20_dark_hour_exposure_bounds.parquet", index=False)

    # Defect legs are kept in the parquet and excluded from the headline table.
    # Their interval is valid arithmetic over a false premise, so averaging it
    # in would let a handful of bad fixes set the reported width.
    defects = out[out["implied_exceeds_cap"]]
    out = out[~out["implied_exceeds_cap"]].copy()

    out["band"] = pd.cut(
        out["gap_hours"],
        [4, 12, 24, 48, 1e9],
        labels=["4-12h", "12-24h", "24-48h", "48h+"],
        right=False,
    )
    table = (
        out.groupby(["month", "band"], observed=True)
        .agg(
            gaps=("candidate_id", "size"),
            cells=("cells_median", "median"),
            wave_width_m=("width_mean_m", "median"),
            swh_lo=("swh_lo_mean", "median"),
            swh_hi=("swh_hi_mean", "median"),
            single_cell=("single_cell_share", "mean"),
            capped=("capped_share", "mean"),
            tc_undet_hours=("tc_undetermined_hours", "mean"),
            tc500_lo=("tc500_hours_lo", "mean"),
            tc500_hi=("tc500_hours_hi", "mean"),
            invariant=("direct_inside_share", "mean"),
        )
        .reset_index()
    )
    table.to_csv(args.out / "21_exposure_bounds_by_band.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": sha256_file(Path(__file__)),
        "read_only": True,
        "products_modified": 0,
        "months": args.months,
        "gaps_sampled": int(len(sample)),
        "enumeration": "every ERA5 cell the reachable region touches, once",
        "era5_cell_degrees": ERA5_CELL_DEGREES,
        "cell_half_diagonal_km": CELL_HALF_DIAGONAL_KM,
        "wet_cell_snap_km": NO_WET_CELL_SNAP_KM,
        "max_cells_per_hour": MAX_CELLS_PER_HOUR,
        "max_hours_per_gap": MAX_HOURS_PER_GAP,
        "hours_bounded_independently": True,
        "tc_radius_500_km": TC_RADIUS_500_KM,
        "tc_radius_300_km": TC_RADIUS_300_KM,
        "tc_maximum_fix_gap_hours": TC_MAXIMUM_FIX_GAP_HOURS,
        "ibtracs": {"path": str(IBTRACS_PATH), "sha256": IBTRACS_SHA256},
        "ibtracs_audit": {
            k: v for k, v in ibtracs_audit.items() if isinstance(v, (int, float, str))
        },
        "speed_cap_source": str(LEGACY_SPEED_CONFIG),
        "speed_floor_kn": definition.speed_floor_kn,
        "speed_ceiling_kn": definition.speed_ceiling_kn,
        "seed": args.seed,
        "gaps_excluded_as_defects": int(len(defects)),
        "defect_rule": "straight-line speed exceeds the voyage speed cap",
        "months_audit": audits,
    }
    (args.out / "bound_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n--- exposure bounds by gap length ---")
    print(table.round(3).to_string(index=False))
    breached = out[out["direct_inside_share"] < 1.0]
    print(
        f"\ninvariant: direct path inside its own interval on "
        f"{100 * (1 - len(breached) / max(len(out), 1)):.2f}% of feasible gaps"
    )
    print(
        f"excluded as defects: {len(defects):,} gaps whose straight line needs "
        f"more than the voyage speed cap "
        f"(median implied {defects['implied_kn'].median():.1f} kn "
        f"against cap {defects['speed_cap_kn'].median():.1f} kn)"
    )
    print(f"outputs: {args.out}")


if __name__ == "__main__":
    main()
