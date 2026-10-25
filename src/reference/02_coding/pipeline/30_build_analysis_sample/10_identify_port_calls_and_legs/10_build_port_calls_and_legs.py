"""Build harmonized port calls and adjacent movement legs for 2021 and 2025.

See identify_port_calls_and_legs/README.md for the design rationale.
This script consumes cleaned AIS shards plus the fixed 2022 anchorage
catalog and produces, for each shard, a port-call table and an adjacent-leg
table using GFW-style 3 km / 4 km / stop-or-gap port-visit logic.

Each AIS shard is vessel-preserving, so per-shard workers produce complete
adjacent legs. These legs are inputs to multi-call first-passage construction,
not direct-OD research outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

import iogenius as iog
import natsort
import numpy as np
import pandas as pd
import sinbue as sb

from sklearn.neighbors import BallTree


# --- Project / pipeline constants -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vessel_group_helpers import (  # noqa: E402
    build_size_bin,
    map_shiptype_group,
)
PIPELINE_NAME = "identify_port_calls_and_legs"
STEP_NAME = "build_port_calls_and_legs"

# Built by 20_clean_and_harmonize/20_build_anchorage_city_catalog: the frozen 2022
# coordinates joined to the reviewed v3 analytical-city crosswalk, so anchorage
# geometry and city identity come from one file for all five years.
# The accepted catalog since E026 (2026-08-22): canal cells removed, non-dock cells
# removed at locations that already hold a dock cell, and manually adjudicated
# cargo-handling locations without a dock flag retained. Not "dock-only" — it keeps
# 6,582 dock=False cells across 1,681 locations holding no dock cell at all.
# The frozen v3 catalog beside it is retained as the pre-E026 reference and is what
# `--anchorage-file` should point at to reproduce anything built before that date.
ANCHORAGE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "products" / "anchorage_catalog"
    / "anchorages_analytical_city_v3_dock_adjusted_inclusive.csv"
)
FROZEN_V3_ANCHORAGE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "products" / "anchorage_catalog"
    / "anchorages_analytical_city_v3.csv"
)

# The five years this chain cleaned itself, not the two inherited products.
AIS_INPUT_DIR_TEMPLATE = (
    PROJECT_ROOT / "data" / "processed" / "clean_ais_tracks" / "ais_cleaning_{year}"
    / "30_apply_masks_dynamic_ais"
)

# Emitted by 10_ingest_data/10_define_observed_fleet for all five years, so every
# year resolves the same observed fleet rather than the two published files.
VESSEL_SPECS_PATH_TEMPLATE = (
    PROJECT_ROOT / "data" / "processed" / "products" / "vessel_universe"
    / "analysis_vessel_specs_{year}.csv"
)

OUTPUT_BASE = PROJECT_ROOT / "data" / "processed" / PIPELINE_NAME / STEP_NAME
LOG_BASE = PROJECT_ROOT / ".runs" / PIPELINE_NAME / STEP_NAME

YEARS = [2021, 2022, 2023, 2024, 2025]
MAX_WORKERS = 12
RUN_SERIAL = False


# --- GFW-style port-visit thresholds --------------------------------------

ENTRY_RADIUS_KM = 3.0
EXIT_RADIUS_KM = 4.0
STOP_SPEED_ENTER_KN = 0.2
STOP_SPEED_EXIT_KN = 0.5
GAP_HOURS = 4.0
EARTH_RADIUS_KM = 6371.0088


# --- AIS columns ----------------------------------------------------------

AIS_REQUIRED_COLUMNS = [
    "vessel_uid",
    "UTC",
    "lon",
    "lat",
    "speed",
    "speed_masked",
    "calc_speed_knots_masked",
    "course_masked",
    "heading_masked",
    "traj_id",
]


# --- Helpers --------------------------------------------------------------


def load_vessel_specs(year: int) -> pd.DataFrame:
    """Load the analysis vessel specs and attach shiptype_group / size_bin.

    Uses the frozen local copies of the legacy grouping definitions.
    """
    path = Path(str(VESSEL_SPECS_PATH_TEMPLATE).format(year=year))
    df = pd.read_csv(path, low_memory=False)
    df = df[df["vessel_uid_clean"].notna()].copy()
    df["vessel_uid_clean"] = df["vessel_uid_clean"].astype(str)
    df["registry_imo_clean"] = df["registry_imo_clean"].astype(str)

    type_source = (
        df["detailed_vessel_type_normalized"]
        if "detailed_vessel_type_normalized" in df.columns
        else df["detailed_vessel_type"]
    )
    df["shiptype_group"] = type_source.map(map_shiptype_group)
    df["size_bin"] = build_size_bin(df)

    keep = [
        "vessel_uid_clean",
        "registry_imo_clean",
        "shiptype_group",
        "size_bin",
    ]
    return df[keep].drop_duplicates(subset=["vessel_uid_clean"]).reset_index(drop=True)


def load_anchorages(anchorage_file: Path = ANCHORAGE_FILE) -> tuple[pd.DataFrame, BallTree]:
    """Load the anchorage catalog and index it for nearest-neighbour search.

    Coordinates are range-checked, not merely tested for nulls. An earlier version
    dropped nulls only, which let an anchorage whose longitude was recorded as 1001
    into the tree: converted to radians it wrapped to 79W in the Caribbean while still
    labelled POINTE NOIRE, COG, and produced 71 false calls in 2021 and 17 in 2025.
    The catalog builder already excludes that record, so this raises rather than
    filters — a bad coordinate reaching here means the catalog itself is wrong.
    """
    df = pd.read_csv(anchorage_file, low_memory=False)
    df = df[
        [
            "s2id",
            "anchorage_iso3",
            "anchorage_label",
            "anchorage_sublabel",
            "lat",
            "lon",
            "analytical_city_id",
            "city_name",
            "country_iso3",
        ]
    ].copy()

    if df[["lat", "lon"]].isna().any().any():
        raise ValueError(f"{anchorage_file} contains null coordinates")
    out_of_range = ~(df["lat"].between(-90.0, 90.0) & df["lon"].between(-180.0, 180.0))
    if out_of_range.any():
        bad = df.loc[out_of_range, ["s2id", "lat", "lon"]].head(5).to_dict("records")
        raise ValueError(f"{anchorage_file} contains out-of-range coordinates: {bad}")
    if df["analytical_city_id"].isna().any():
        raise ValueError(f"{anchorage_file} contains anchorages with no analytical city")
    if df["s2id"].duplicated().any():
        raise ValueError(f"{anchorage_file} repeats an s2id")

    df = df.reset_index(drop=True)
    coords_rad = np.radians(df[["lat", "lon"]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")
    return df, tree


def query_nearest_anchorage(
    lat: np.ndarray,
    lon: np.ndarray,
    tree: BallTree,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (nearest_anchorage_idx, distance_km) for each AIS point."""
    if len(lat) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    q = np.radians(np.column_stack([lat, lon]))
    dist_rad, idx = tree.query(q, k=1)
    dist_km = dist_rad[:, 0] * EARTH_RADIUS_KM
    return idx[:, 0].astype(np.int64), dist_km


def distance_to_anchorage_km(
    lat: float,
    lon: float,
    anchorage_lat: float,
    anchorage_lon: float,
) -> float:
    """Return the haversine distance from one AIS point to one anchorage cell."""
    lat1 = radians(lat)
    lon1 = radians(lon)
    lat2 = radians(anchorage_lat)
    lon2 = radians(anchorage_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        sin(dlat / 2.0) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    )
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * asin(sqrt(a))


# --- Per-vessel state machine --------------------------------------------


def build_vessel_visits(
    vessel_uid: str,
    registry_imo_clean: str,
    times: np.ndarray,
    speeds: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    nearest_idx: np.ndarray,
    nearest_dist_km: np.ndarray,
    anchorage_latitudes: np.ndarray,
    anchorage_longitudes: np.ndarray,
) -> list[dict[str, Any]]:
    """Apply GFW-style 3 km entry / 4 km exit / stop-or-gap rules.

    Each visit is opened on entry into ENTRY_RADIUS_KM of its nearest anchorage
    cell and bound to that cell for its lifetime. Every subsequent entry/exit,
    stop and gap decision uses distance to the opening cell, not distance to
    whichever catalog cell happens to be nearest later. This prevents a visit
    opened at one anchorage from borrowing distance coverage or a later stop
    from another. A high-confidence visit requires entry, exit, and (stop OR
    gap).
    """
    n = len(times)
    visits: list[dict[str, Any]] = []
    if n == 0:
        return visits

    times_ns = times.astype("datetime64[ns]")

    open_visit: dict[str, Any] | None = None
    in_stop = False
    visit_seq = 0

    def open_new(i: int) -> dict[str, Any]:
        anchorage_idx = int(nearest_idx[i])
        return {
            "anchorage_idx": anchorage_idx,
            "first_seen_idx": i,
            "entry_idx": i,
            "exit_idx": None,
            "has_entry": True,
            "has_exit": False,
            "has_stop": False,
            "has_gap": False,
            "n_points": 0,
            "last_within_entry_idx": i,
        }

    for i in range(n):
        nearest_d = nearest_dist_km[i]
        sp = speeds[i]

        if open_visit is None:
            if nearest_d <= ENTRY_RADIUS_KM:
                open_visit = open_new(i)
                in_stop = False
            else:
                continue

        v = open_visit
        v["n_points"] += 1
        anchorage_idx = int(v["anchorage_idx"])
        visit_d = distance_to_anchorage_km(
            float(latitudes[i]),
            float(longitudes[i]),
            float(anchorage_latitudes[anchorage_idx]),
            float(anchorage_longitudes[anchorage_idx]),
        )

        if visit_d <= ENTRY_RADIUS_KM:
            # Detect AIS gap while staying near the anchorage.
            if v["last_within_entry_idx"] != i and i > 0:
                prev_t = times_ns[v["last_within_entry_idx"]]
                cur_t = times_ns[i]
                gap_h = (cur_t - prev_t) / np.timedelta64(1, "h")
                if gap_h >= GAP_HOURS:
                    v["has_gap"] = True
            v["last_within_entry_idx"] = i

            # Stop detection.
            if not in_stop and pd.notna(sp) and sp < STOP_SPEED_ENTER_KN:
                in_stop = True
                v["has_stop"] = True
            elif in_stop and pd.notna(sp) and sp > STOP_SPEED_EXIT_KN:
                in_stop = False

        if visit_d > EXIT_RADIUS_KM:
            v["exit_idx"] = i
            v["has_exit"] = True
            confidence = "high" if (v["has_stop"] or v["has_gap"]) else "low"
            visit_seq += 1
            visits.append(
                {
                    "vessel_uid": vessel_uid,
                    "registry_imo_clean": registry_imo_clean,
                    "visit_id_internal": f"{vessel_uid}_v{visit_seq:05d}",
                    "anchorage_idx": v["anchorage_idx"],
                    "entry_time": times_ns[v["entry_idx"]],
                    "exit_time": times_ns[v["exit_idx"]],
                    "first_seen_time": times_ns[v["first_seen_idx"]],
                    "last_seen_time": times_ns[v["last_within_entry_idx"]],
                    "has_entry": True,
                    "has_exit": True,
                    "has_stop": v["has_stop"],
                    "has_gap": v["has_gap"],
                    "visit_confidence": confidence,
                    "n_ais_points_in_visit": v["n_points"],
                }
            )
            open_visit = None
            in_stop = False

    if open_visit is not None:
        v = open_visit
        confidence = "low"
        visit_seq += 1
        visits.append(
            {
                "vessel_uid": vessel_uid,
                "registry_imo_clean": registry_imo_clean,
                "visit_id_internal": f"{vessel_uid}_v{visit_seq:05d}",
                "anchorage_idx": v["anchorage_idx"],
                "entry_time": times_ns[v["entry_idx"]],
                "exit_time": pd.NaT,
                "first_seen_time": times_ns[v["first_seen_idx"]],
                "last_seen_time": times_ns[v["last_within_entry_idx"]],
                "has_entry": True,
                "has_exit": False,
                "has_stop": v["has_stop"],
                "has_gap": v["has_gap"],
                "visit_confidence": confidence,
                "n_ais_points_in_visit": v["n_points"],
            }
        )

    return visits


def visits_to_voyages(
    visits_df: pd.DataFrame,
    year: int,
    anchorages: pd.DataFrame,
    vessel_meta: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Pair consecutive high-confidence visits within a vessel into voyages."""
    if visits_df.empty:
        return pd.DataFrame()

    hc = visits_df.loc[visits_df["visit_confidence"] == "high"].copy()
    hc = hc.sort_values(["vessel_uid", "exit_time"]).reset_index(drop=True)

    voyages: list[dict[str, Any]] = []
    for ship_id, group in hc.groupby("vessel_uid", sort=False):
        group = group.reset_index(drop=True)
        for k in range(len(group) - 1):
            origin = group.iloc[k]
            dest = group.iloc[k + 1]
            if pd.isna(origin["exit_time"]) or pd.isna(dest["entry_time"]):
                continue
            if dest["entry_time"] <= origin["exit_time"]:
                continue
            duration_h = (
                dest["entry_time"] - origin["exit_time"]
            ) / np.timedelta64(1, "h")
            o_anchor = anchorages.iloc[int(origin["anchorage_idx"])]
            d_anchor = anchorages.iloc[int(dest["anchorage_idx"])]
            meta = vessel_meta.get(ship_id, {})
            voyages.append(
                {
                    "year": year,
                    "vessel_uid": ship_id,
                    "registry_imo_clean": meta.get("registry_imo_clean", ""),
                    "internal_voyage_id": f"{year}_{ship_id}_{k + 1:04d}",
                    "voyage_sequence": k + 1,
                    "trip_start": origin["exit_time"],
                    "trip_end": dest["entry_time"],
                    "duration_hours": float(duration_h),
                    "duration_days": float(duration_h / 24.0),
                    "origin_visit_id_internal": origin["visit_id_internal"],
                    "dest_visit_id_internal": dest["visit_id_internal"],
                    "origin_anchorage_id": o_anchor["s2id"],
                    "dest_anchorage_id": d_anchor["s2id"],
                    "origin_iso3": o_anchor["anchorage_iso3"],
                    "dest_iso3": d_anchor["anchorage_iso3"],
                    "origin_label": o_anchor["anchorage_label"],
                    "dest_label": d_anchor["anchorage_label"],
                    "origin_sublabel": o_anchor["anchorage_sublabel"],
                    "dest_sublabel": d_anchor["anchorage_sublabel"],
                    # City identity comes from the reviewed v3 crosswalk. The
                    # anchorage labels above are anchorage-level names kept for
                    # provenance and must not be read as a city.
                    "origin_analytical_city_id": o_anchor["analytical_city_id"],
                    "dest_analytical_city_id": d_anchor["analytical_city_id"],
                    "origin_city_name": o_anchor["city_name"],
                    "dest_city_name": d_anchor["city_name"],
                    "od_pair": f"{o_anchor['s2id']}__{d_anchor['s2id']}",
                    "city_pair": (
                        f"{o_anchor['analytical_city_id']}__"
                        f"{d_anchor['analytical_city_id']}"
                    ),
                    "country_pair": (
                        f"{o_anchor['anchorage_iso3']}__{d_anchor['anchorage_iso3']}"
                    ),
                    "is_international": bool(
                        o_anchor["anchorage_iso3"] != d_anchor["anchorage_iso3"]
                    ),
                    "shiptype_group": meta.get("shiptype_group", "other_retained"),
                    "size_bin": meta.get("size_bin", "unknown"),
                    "source_pipeline_generation": "internal_harmonized",
                }
            )

    return pd.DataFrame(voyages)


# --- Empty-output schemas (used when a shard yields no rows) -------------


VISIT_OUTPUT_COLUMNS = [
    "year", "vessel_uid", "registry_imo_clean", "visit_id_internal",
    "anchorage_id", "anchorage_iso3", "anchorage_label", "anchorage_sublabel",
    "analytical_city_id", "city_name",
    "entry_time", "exit_time", "first_seen_time", "last_seen_time",
    "has_entry", "has_exit", "has_stop", "has_gap",
    "visit_confidence", "n_ais_points_in_visit",
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


def empty_visits() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in VISIT_OUTPUT_COLUMNS})


def empty_voyages() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in VOYAGE_OUTPUT_COLUMNS})


# --- Worker ---------------------------------------------------------------


def build_task(
    input_file: Path,
    year: int,
    output_dir: Path,
    task_results_dir: Path,
    anchorage_file: Path = ANCHORAGE_FILE,
) -> dict[str, Any]:
    voyages_dir = output_dir / "shard_voyages"
    visits_dir = output_dir / "shard_port_visits"
    task_id = f"{year}_{input_file.stem}"
    return {
        "task_id": task_id,
        "year": year,
        "input_path": str(input_file),
        # Carried on the task rather than read from a module global: workers may be
        # spawned rather than forked, and a silently defaulted catalog would produce
        # a plausible but wrong run.
        "anchorage_file": str(anchorage_file),
        "voyages_output_path": str(
            voyages_dir / f"{input_file.stem}_harmonized_od_voyages.feather"
        ),
        "visits_output_path": str(
            visits_dir / f"{input_file.stem}_port_visits.feather"
        ),
        "task_result_path": str(task_results_dir / f"{task_id}.json"),
    }


def process_task(task: dict[str, Any], log_dir) -> dict[str, Any]:
    input_path = Path(task["input_path"])
    voyages_path = Path(task["voyages_output_path"])
    visits_path = Path(task["visits_output_path"])
    year = int(task["year"])
    log_path = Path(log_dir) / f"{task['task_id']}.log"
    task_result_path = Path(task["task_result_path"])
    task_result_path.parent.mkdir(parents=True, exist_ok=True)

    voyages_path.parent.mkdir(parents=True, exist_ok=True)
    visits_path.parent.mkdir(parents=True, exist_ok=True)
    voyages_tmp = voyages_path.with_suffix(voyages_path.suffix + ".tmp")
    visits_tmp = visits_path.with_suffix(visits_path.suffix + ".tmp")

    result: dict[str, Any] = {
        "task_id": task["task_id"],
        "year": year,
        "input_path": str(input_path),
        "voyages_output_path": str(voyages_path),
        "visits_output_path": str(visits_path),
        "status": "failed",
        "n_input_rows": 0,
        "n_filtered_rows": 0,
        "n_unique_vessels": 0,
        "n_visits": 0,
        "n_high_confidence_visits": 0,
        "n_voyages": 0,
        "error": "",
    }

    try:
        with sb.PrintRedirector(log_path):
            print(f"[{task['task_id']}] reading {input_path}")
            anchorage_file = Path(task["anchorage_file"])
            print(f"[{task['task_id']}] anchorage catalog {anchorage_file}")
            anchorages, tree = load_anchorages(anchorage_file)
            specs = load_vessel_specs(year)

            df = pd.read_feather(input_path, columns=AIS_REQUIRED_COLUMNS)
            result["n_input_rows"] = int(len(df))

            keep_ids = set(specs["vessel_uid_clean"].astype(str))
            df["vessel_uid"] = df["vessel_uid"].astype(str)
            df = df.loc[df["vessel_uid"].isin(keep_ids)].copy()
            result["n_filtered_rows"] = int(len(df))

            if df.empty:
                empty_visits().to_feather(visits_tmp)
                empty_voyages().to_feather(voyages_tmp)
                voyages_tmp.replace(voyages_path)
                visits_tmp.replace(visits_path)
                result["status"] = "success"
                task_result_path.write_text(json.dumps(result, indent=2, default=str))
                return result

            df = df.sort_values(["vessel_uid", "UTC"]).reset_index(drop=True)
            df = df.dropna(subset=["lat", "lon", "UTC"]).reset_index(drop=True)
            # Use the cleaned calculated SOG as the canonical speed signal
            # for stop detection. NaN values are tolerated by the state
            # machine (no transition fires on a NaN sample).
            speed_for_state = pd.to_numeric(
                df["calc_speed_knots_masked"], errors="coerce"
            )

            nearest_idx, nearest_dist = query_nearest_anchorage(
                df["lat"].to_numpy(), df["lon"].to_numpy(), tree
            )

            vessel_meta = specs.set_index("vessel_uid_clean")[
                ["registry_imo_clean", "shiptype_group", "size_bin"]
            ].to_dict(orient="index")

            all_visits: list[dict[str, Any]] = []
            unique_ships = df["vessel_uid"].unique()
            result["n_unique_vessels"] = int(len(unique_ships))

            times_arr = df["UTC"].to_numpy()
            speeds_arr = speed_for_state.to_numpy()
            latitudes_arr = df["lat"].to_numpy()
            longitudes_arr = df["lon"].to_numpy()
            anchorage_latitudes = anchorages["lat"].to_numpy()
            anchorage_longitudes = anchorages["lon"].to_numpy()
            ship_arr = df["vessel_uid"].to_numpy()

            change = np.flatnonzero(ship_arr[1:] != ship_arr[:-1]) + 1
            starts = np.concatenate([[0], change])
            ends = np.concatenate([change, [len(df)]])

            for s, e in zip(starts, ends):
                ship_id = str(ship_arr[s])
                meta = vessel_meta.get(ship_id, {})
                imo = meta.get("registry_imo_clean", "")
                visits = build_vessel_visits(
                    ship_id,
                    imo,
                    times_arr[s:e],
                    speeds_arr[s:e],
                    latitudes_arr[s:e],
                    longitudes_arr[s:e],
                    nearest_idx[s:e],
                    nearest_dist[s:e],
                    anchorage_latitudes,
                    anchorage_longitudes,
                )
                all_visits.extend(visits)

            visits_df = pd.DataFrame(all_visits)
            if not visits_df.empty:
                visits_df["year"] = year
                visits_df["anchorage_id"] = visits_df["anchorage_idx"].map(
                    anchorages["s2id"]
                )
                visits_df["anchorage_iso3"] = visits_df["anchorage_idx"].map(
                    anchorages["anchorage_iso3"]
                )
                visits_df["anchorage_label"] = visits_df["anchorage_idx"].map(
                    anchorages["anchorage_label"]
                )
                visits_df["anchorage_sublabel"] = visits_df["anchorage_idx"].map(
                    anchorages["anchorage_sublabel"]
                )
                visits_df["analytical_city_id"] = visits_df["anchorage_idx"].map(
                    anchorages["analytical_city_id"]
                )
                visits_df["city_name"] = visits_df["anchorage_idx"].map(
                    anchorages["city_name"]
                )
                visits_out = visits_df[
                    # one list, not a second copy: this selection and
                    # VISIT_OUTPUT_COLUMNS had drifted apart, which is how the
                    # analytical-city fields reached the voyages but not the visits
                    VISIT_OUTPUT_COLUMNS
                ].reset_index(drop=True)
            else:
                visits_out = pd.DataFrame()

            voyages_df = visits_to_voyages(
                visits_df, year, anchorages, vessel_meta
            )

            result["n_visits"] = int(len(visits_out))
            result["n_high_confidence_visits"] = int(
                (visits_df["visit_confidence"] == "high").sum()
            ) if not visits_df.empty else 0
            result["n_voyages"] = int(len(voyages_df))

            visits_out.to_feather(visits_tmp)
            voyages_df.reset_index(drop=True).to_feather(voyages_tmp)
            voyages_tmp.replace(voyages_path)
            visits_tmp.replace(visits_path)

            result["status"] = "success"
            print(
                f"[{task['task_id']}] visits={result['n_visits']} "
                f"hc_visits={result['n_high_confidence_visits']} "
                f"voyages={result['n_voyages']}"
            )

    except Exception as exc:
        for tmp in (voyages_tmp, visits_tmp):
            if tmp.exists():
                tmp.unlink()
        result["error"] = f"{exc}\n{traceback.format_exc()}"

    task_result_path.write_text(json.dumps(result, indent=2, default=str))
    return result


# --- Main / aggregation ---------------------------------------------------


def aggregate_year(year: int, output_dir: Path) -> dict[str, Any]:
    voyages_dir = output_dir / "shard_voyages"
    visits_dir = output_dir / "shard_port_visits"

    voyage_files = natsort.natsorted(voyages_dir.glob("*.feather"))
    visit_files = natsort.natsorted(visits_dir.glob("*.feather"))

    voyages = pd.concat(
        [pd.read_feather(f) for f in voyage_files], ignore_index=True
    ) if voyage_files else empty_voyages()
    visits = pd.concat(
        [pd.read_feather(f) for f in visit_files], ignore_index=True
    ) if visit_files else empty_visits()

    voyages_path = output_dir / f"harmonized_od_voyages_{year}.feather"
    visits_path = output_dir / f"port_visits_{year}.feather"
    voyages.to_feather(voyages_path)
    visits.to_feather(visits_path)

    summary = {
        "year": year,
        "n_shards": len(voyage_files),
        "n_voyages": int(len(voyages)),
        "n_port_visits": int(len(visits)),
        "n_unique_vessels": int(voyages["vessel_uid"].nunique()) if not voyages.empty else 0,
        "n_international_voyages": int(voyages["is_international"].sum()) if not voyages.empty else 0,
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / f"harmonized_od_voyages_{year}_summary.csv", index=False
    )
    return summary


def main(
    years: list[int],
    force: bool = False,
    limit_shards: int | None = None,
    skip_year_aggregate: bool = False,
    anchorage_file: Path = ANCHORAGE_FILE,
    output_base: Path = OUTPUT_BASE,
    log_base: Path = LOG_BASE,
) -> None:
    pd.set_option("display.max_columns", None)
    iog.set_working_directory(str(PROJECT_ROOT))

    if not Path(anchorage_file).exists():
        raise FileNotFoundError(f"anchorage catalog not found: {anchorage_file}")
    # A non-default catalog writing into the default output root would overwrite the
    # accepted port-call products with a differently-defined measurement.
    if Path(anchorage_file) != ANCHORAGE_FILE and Path(output_base) == OUTPUT_BASE:
        raise ValueError(
            "refusing to write a non-default anchorage catalog into the accepted "
            f"output root {OUTPUT_BASE}; pass --output-base to a separate location"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for year in years:
        ais_dir = Path(str(AIS_INPUT_DIR_TEMPLATE).format(year=year))
        year_output_dir = Path(output_base) / str(year)
        year_log_dir = Path(log_base) / str(year)
        run_summary_dir = year_log_dir / "run_summary" / run_id
        task_results_dir = run_summary_dir / "task_results"

        for d in (
            year_output_dir / "shard_voyages",
            year_output_dir / "shard_port_visits",
            year_log_dir,
            run_summary_dir,
            task_results_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        input_files = natsort.natsorted(ais_dir.glob("shard_*.feather"))
        if limit_shards is not None and limit_shards > 0:
            input_files = input_files[:limit_shards]
        all_tasks = [
            build_task(p, year, year_output_dir, task_results_dir, anchorage_file)
            for p in input_files
        ]

        if force:
            skipped = []
            pending = all_tasks
        else:
            skipped = [
                t for t in all_tasks
                if Path(t["voyages_output_path"]).exists()
                and Path(t["visits_output_path"]).exists()
            ]
            pending = [
                t for t in all_tasks
                if not (
                    Path(t["voyages_output_path"]).exists()
                    and Path(t["visits_output_path"]).exists()
                )
            ]

        print(
            f"[{year}] all={len(all_tasks)} pending={len(pending)} skipped={len(skipped)}"
        )

        run_config = {
            "run_id": run_id,
            "year": year,
            "max_workers": MAX_WORKERS,
            "run_serial": RUN_SERIAL,
            "force": force,
            "n_all_tasks": len(all_tasks),
            "n_pending_tasks": len(pending),
            "n_skipped_tasks": len(skipped),
            "ais_input_dir": str(ais_dir),
            "output_dir": str(year_output_dir),
            "anchorage_file": str(anchorage_file),
            "anchorage_file_is_accepted_default": Path(anchorage_file) == ANCHORAGE_FILE,
        }
        (run_summary_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2)
        )
        (run_summary_dir / "all_tasks.json").write_text(
            json.dumps(all_tasks, indent=2)
        )
        (run_summary_dir / "skipped_existing_output_tasks.json").write_text(
            json.dumps(skipped, indent=2)
        )

        if pending:
            if RUN_SERIAL:
                for t in pending:
                    process_task(t, year_log_dir)
            else:
                sb.process_files_parallel(
                    process_task,
                    input_files=pending,
                    log_dir=year_log_dir,
                    max_workers=MAX_WORKERS,
                    show_progress=True,
                )

        results = []
        for t in all_tasks:
            rp = Path(t["task_result_path"])
            if rp.exists():
                results.append(json.loads(rp.read_text()))

        failed = [r for r in results if r.get("status") != "success"]
        (run_summary_dir / "failed_tasks.json").write_text(
            json.dumps(failed, indent=2, default=str)
        )

        if skip_year_aggregate:
            print(f"[{year}] skipping year-level aggregation (test mode)")
        else:
            summary = aggregate_year(year, year_output_dir)
            (run_summary_dir / "year_aggregation_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            print(f"[{year}] aggregation summary: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Process only the first N shards per year (for test runs).",
    )
    parser.add_argument(
        "--skip-year-aggregate",
        action="store_true",
        help="Skip building the year-level concatenated feather (useful for partial test runs).",
    )
    parser.add_argument(
        "--anchorage-file",
        type=Path,
        default=ANCHORAGE_FILE,
        help=(
            "Anchorage catalog to resolve port calls against. Defaults to the accepted "
            "frozen v3 catalog. Any other value also requires --output-base."
        ),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=OUTPUT_BASE,
        help="Root directory for shard outputs. Defaults to the accepted product root.",
    )
    parser.add_argument(
        "--log-base",
        type=Path,
        default=LOG_BASE,
        help="Root directory for run logs and task results.",
    )
    args = parser.parse_args()

    if args.serial:
        RUN_SERIAL = True
    if args.max_workers:
        MAX_WORKERS = args.max_workers

    main(
        args.years,
        force=args.force,
        limit_shards=args.limit_shards,
        skip_year_aggregate=args.skip_year_aggregate,
        anchorage_file=args.anchorage_file,
        output_base=args.output_base,
        log_base=args.log_base,
    )
