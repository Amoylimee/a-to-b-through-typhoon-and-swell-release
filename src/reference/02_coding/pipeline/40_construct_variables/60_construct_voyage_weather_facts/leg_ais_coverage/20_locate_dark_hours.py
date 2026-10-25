#!/usr/bin/env python3
"""Locate long AIS gaps in a route-neutral one-degree grid.

Read-only diagnostic for the accepted 2025 candidate root. It computes no
weather and writes no product root. For each within-voyage inter-fix gap longer
than the weather interpolation limit, it distributes the gap duration along
the great-circle arc between the bounding fixes. This is only a spatial
attribution assumption: none of these grid cells becomes an observed position
or a weather lookup location. Gaps that can be interpolated are attributed to
the midpoint of their two fixes by the same convention.

The result contains cell coordinates and observed/dark hour totals. It does not
group cells into routes, corridors, named seas, or treatment categories.

    python3 20_locate_dark_hours.py --years 2025 --workers 12

Environment: sinbue_env (Python 3.10)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather

HERE = Path(__file__).resolve().parent
CODING_ROOT = HERE.parents[3]
WEATHER_STAGE_ROOT = HERE.parent
if str(WEATHER_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(WEATHER_STAGE_ROOT))
ENVIRONMENT_STAGE_ROOT = CODING_ROOT / "pipeline/00_environment_and_config"
if str(ENVIRONMENT_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_STAGE_ROOT))

from ais_identity import (  # noqa: E402
    deduplicate_vessel_ais,
    normalise_required_identifier,
    stable_vessel_buckets,
    validate_ais_physical_identity,
)
from accepted_candidate_root import load_accepted_2025_candidates  # noqa: E402

MAXIMUM_INTERPOLATION_GAP_HOURS = 4.0
NS_PER_HOUR = 3_600 * 1_000_000_000
GAP_LIMIT_NS = np.int64(MAXIMUM_INTERPOLATION_GAP_HOURS * NS_PER_HOUR)
CELL_DEGREES = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def unit_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Longitude and latitude in degrees to unit vectors on the sphere."""

    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    cos_lat = np.cos(lat_r)
    return np.stack(
        [cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)], axis=-1
    )


def great_circle_samples(
    lon_a: np.ndarray,
    lat_a: np.ndarray,
    lon_b: np.ndarray,
    lat_b: np.ndarray,
    fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate along the great circle at the given fractions.

    ``fractions`` has shape (k,) and the endpoint arrays shape (n,); the result
    has shape (n, k).  Antipodal or coincident endpoints fall back to the first
    endpoint, which is the only defensible choice when the path is undefined.
    """

    a = unit_xyz(lon_a, lat_a)
    b = unit_xyz(lon_b, lat_b)
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    f = fractions[None, :, None]
    degenerate = sin_omega < 1e-12
    safe = np.where(degenerate, 1.0, sin_omega)[:, None, None]
    w_a = np.sin((1.0 - f) * omega[:, None, None]) / safe
    w_b = np.sin(f * omega[:, None, None]) / safe
    point = w_a * a[:, None, :] + w_b * b[:, None, :]
    point = np.where(degenerate[:, None, None], a[:, None, :], point)

    norm = np.linalg.norm(point, axis=-1, keepdims=True)
    point = point / np.where(norm == 0, 1.0, norm)
    lat = np.degrees(np.arcsin(np.clip(point[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(point[..., 1], point[..., 0]))
    return lon, lat


def cell_index(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Pack a position into a single integer one-degree cell key."""

    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    lat = np.clip(np.asarray(lat), -90.0, 90.0)
    lon_i = np.clip(np.floor(lon / CELL_DEGREES).astype(np.int64), -180, 179)
    lat_i = np.clip(np.floor(lat / CELL_DEGREES).astype(np.int64), -90, 89)
    return (lat_i + 90) * 360 + (lon_i + 180)


def decode_cell(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key = np.asarray(key, dtype=np.int64)
    lat = (key // 360 - 90) * CELL_DEGREES + CELL_DEGREES / 2.0
    lon = (key % 360 - 180) * CELL_DEGREES + CELL_DEGREES / 2.0
    return lat, lon


def accumulate_bucket(
    bucket_dir: Path, candidates: pd.DataFrame, output_dir: Path
) -> dict[str, Any]:
    """Bin one bucket's observed and dark hours into one-degree cells."""

    out_path = output_dir / f"{bucket_dir.name}.parquet"
    if out_path.exists():
        return {"resumed": 1}

    parts = sorted(bucket_dir.glob("*.feather"))
    if not parts:
        pd.DataFrame(columns=["cell", "observed_hours", "dark_hours"]).to_parquet(
            out_path, index=False
        )
        return {"resumed": 0, "candidates": 0}

    points = pd.concat(
        [feather.read_table(p, memory_map=True).to_pandas() for p in parts],
        ignore_index=True,
    )
    identity = candidates[["mmsi", "vessel_uid_clean"]].drop_duplicates()
    points, _ = validate_ais_physical_identity(points, identity)
    points, _ = deduplicate_vessel_ais(points)

    observed: dict[int, float] = {}
    dark: dict[int, float] = {}
    audit = {"gaps_over_limit": 0, "dark_hours_total": 0.0, "candidates": 0}

    def deposit(store: dict[int, float], keys: np.ndarray, hours: np.ndarray) -> None:
        if keys.size == 0:
            return
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        hours = hours[order]
        uniq, start = np.unique(keys, return_index=True)
        summed = np.add.reduceat(hours, start)
        for key, value in zip(uniq.tolist(), summed.tolist()):
            store[key] = store.get(key, 0.0) + value

    by_vessel = {k: g for k, g in points.groupby("mmsi", sort=False, observed=True)}

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

        for i in range(len(starts)):
            lo, hi = left[i], right[i]
            audit["candidates"] += 1
            if hi - lo < 2:
                continue
            t = utc[lo:hi]
            la = lat_all[lo:hi]
            lo_ = lon_all[lo:hi]
            gaps_ns = np.diff(t)
            hours = gaps_ns / NS_PER_HOUR
            over = gaps_ns > GAP_LIMIT_NS

            # Observed legs: attribute their duration to the great-circle midpoint.
            if (~over).any():
                idx = np.flatnonzero(~over)
                mid_lon, mid_lat = great_circle_samples(
                    lo_[idx], la[idx], lo_[idx + 1], la[idx + 1], np.array([0.5])
                )
                deposit(
                    observed,
                    cell_index(mid_lon[:, 0], mid_lat[:, 0]),
                    hours[idx],
                )

            # Dark legs: spread the hours along the great circle, one sample per
            # hour of gap, capped so a very long gap cannot explode the work.
            if over.any():
                idx = np.flatnonzero(over)
                audit["gaps_over_limit"] += int(idx.size)
                audit["dark_hours_total"] += float(hours[idx].sum())
                for j in idx:
                    k = int(min(max(np.ceil(hours[j]), 2), 240))
                    fractions = (np.arange(k) + 0.5) / k
                    s_lon, s_lat = great_circle_samples(
                        lo_[j : j + 1],
                        la[j : j + 1],
                        lo_[j + 1 : j + 2],
                        la[j + 1 : j + 2],
                        fractions,
                    )
                    deposit(
                        dark,
                        cell_index(s_lon[0], s_lat[0]),
                        np.full(k, hours[j] / k),
                    )

    rows = []
    for key in set(observed) | set(dark):
        rows.append(
            {
                "cell": int(key),
                "observed_hours": observed.get(key, 0.0),
                "dark_hours": dark.get(key, 0.0),
            }
        )
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    audit["resumed"] = 0
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=[2025])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument(
        "--coverage-run",
        type=Path,
        required=True,
        help="run root from 10_measure_leg_ais_coverage.py; its vessel_buckets are reused",
    )
    parser.add_argument(
        "--calendar-window",
        nargs=2,
        metavar=("MM-DD", "MM-DD"),
        help=(
            "keep only departures inside this calendar window, inclusive of both ends"
        ),
    )
    parser.add_argument(
        "--out-name",
        default="dark_cells",
        help="subdirectory for the per-bucket cells; use a fresh name per comparison",
    )
    args = parser.parse_args()
    if args.workers > 12:
        raise SystemExit("CONTRACT caps this project at 12 workers")

    bucket_root = args.coverage_run / "vessel_buckets"
    if not (bucket_root / "_shuffle_complete.json").exists():
        raise SystemExit(f"no completed shuffle at {bucket_root}")

    out_root = args.coverage_run / args.out_name
    out_root.mkdir(exist_ok=True)

    if tuple(sorted(set(args.years))) != (2025,):
        raise SystemExit("E001 dark-hour diagnostics read the accepted 2025 candidate root only")
    candidates = load_accepted_2025_candidates(
        ("candidate_id", "mmsi", "vessel_uid_clean", "departure_time", "arrival_time")
    )
    candidates["mmsi"] = normalise_required_identifier(candidates["mmsi"], "mmsi")
    candidates["vessel_uid_clean"] = normalise_required_identifier(
        candidates["vessel_uid_clean"], "vessel_uid_clean"
    )
    if args.calendar_window:
        start_md, end_md = args.calendar_window
        departure = candidates["departure_time"].dt.tz_convert("UTC")
        month_day = departure.dt.strftime("%m-%d")
        before = len(candidates)
        candidates = candidates[
            (month_day >= start_md) & (month_day <= end_md)
        ].reset_index(drop=True)
        print(
            f"calendar window {start_md}..{end_md}: kept {len(candidates):,} "
            f"of {before:,} candidates"
        )

    candidates["vessel_bucket"] = stable_vessel_buckets(
        candidates["mmsi"], args.buckets
    )
    print(f"locating dark hours for {len(candidates):,} candidates")

    totals = {"gaps_over_limit": 0, "dark_hours_total": 0.0, "candidates": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                accumulate_bucket,
                bucket_root / f"bucket_{int(bucket):04d}",
                group.reset_index(drop=True),
                out_root,
            ): bucket
            for bucket, group in candidates.groupby("vessel_bucket", sort=True)
        }
        for n, future in enumerate(as_completed(futures), start=1):
            audit = future.result()
            for key in totals:
                totals[key] += audit.get(key, 0)
            if n % 8 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} buckets")

    cells = pd.concat(
        [pd.read_parquet(p) for p in sorted(out_root.glob("bucket_*.parquet"))],
        ignore_index=True,
    )
    cells = cells.groupby("cell", as_index=False)[
        ["observed_hours", "dark_hours"]
    ].sum()
    lat, lon = decode_cell(cells["cell"].to_numpy())
    cells["cell_lat"] = lat
    cells["cell_lon"] = lon
    cells["total_hours"] = cells["observed_hours"] + cells["dark_hours"]
    cells["dark_share_pct"] = 100 * cells["dark_hours"] / cells["total_hours"]
    cells.insert(0, "departure_year", 2025)

    tables = args.coverage_run / "tables"
    tables.mkdir(exist_ok=True)
    stem = "" if args.out_name == "dark_cells" else f"_{args.out_name}"
    cells.to_parquet(tables / f"10_dark_cells_1deg{stem}.parquet", index=False)
    cells.sort_values(["dark_hours", "cell"], ascending=[False, True]).to_csv(
        tables / f"11_dark_share_by_cell{stem}.csv", index=False
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": sha256_file(Path(__file__)),
        "read_only": True,
        "products_modified": 0,
        "cell_degrees": CELL_DEGREES,
        "attribution": "great-circle samples, one per hour of gap, capped at 240",
        "maximum_interpolation_gap_hours": MAXIMUM_INTERPOLATION_GAP_HOURS,
        "years": args.years,
        "calendar_window": args.calendar_window,
        "candidates_after_window": int(len(candidates)),
        "totals": totals,
    }
    (args.coverage_run / f"dark_location_manifest{stem}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n--- highest dark-hour grid cells ---")
    print(cells.nlargest(20, "dark_hours")[
        ["cell_lat", "cell_lon", "observed_hours", "dark_hours", "dark_share_pct"]
    ].round(2).to_string(index=False))
    print(f"\ntables: {tables}")


if __name__ == "__main__":
    main()
