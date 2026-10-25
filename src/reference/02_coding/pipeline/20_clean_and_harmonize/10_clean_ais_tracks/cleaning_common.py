import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import natsort
import numpy as np
import pandas as pd
import yaml
import iogenius as iog


SCRIPT_DIR = Path(__file__).resolve().parent
# clean_ais_tracks/ -> code root
SUBPROJECT_ROOT = SCRIPT_DIR.parents[2]

# Column names for this project's AIS data
SHIP_ID_COL = "vessel_uid"
MMSI_COL = "mmsi"
IMO_COL = "imo"
SHIPNAME_COL = "SHIPNAME"
TIMESTAMP_COL = "UTC"
LON_COL = "lon"
LAT_COL = "lat"
SPEED_COL = "speed"
COURSE_COL = "course"
HEADING_COL = "heading"

CORE_DYNAMIC_COLUMNS = [
    SHIP_ID_COL,
    TIMESTAMP_COL,
    LON_COL,
    LAT_COL,
    SPEED_COL,
    COURSE_COL,
]
OPTIONAL_DYNAMIC_COLUMNS = [HEADING_COL, MMSI_COL, IMO_COL, SHIPNAME_COL]

RAW_TYPE_CANDIDATE_COLUMNS = [
    "TYPE_NAME",
    "SHIPTYPE_NAME",
    "SHIPTYPE",
    "VESSEL_TYPE",
    "VESSELTYPE",
]

SPECS_REQUIRED_COLUMNS = [SHIP_ID_COL]
SPECS_GENERAL_TYPE_CANDIDATE_COLUMNS = [
    "general_vessel_type",
]
SPECS_DETAILED_TYPE_CANDIDATE_COLUMNS = [
    "detailed_vessel_type",
]
SPECS_SERVICE_SPEED_CANDIDATE_COLUMNS = [
    "speed_service",
]

IMO_LEVEL2_DEFAULT_MAX_SPEED = {
    "Bulk carrier": 18.0,
    "Chemical tanker": 17.0,
    "Container": 28.0,
    "General cargo": 18.0,
    "Liquefied gas tanker": 20.0,
    "Oil tanker": 17.0,
    "Other liquids tankers": 15.0,
    "Ferry-pax only": 25.0,
    "Cruise": 25.0,
    "Ferry-RoPax": 25.0,
    "Refrigerated bulk": 20.0,
    "Ro-Ro": 22.0,
    "Vehicle": 20.0,
    "Yacht": 30.0,
    "Service tug": 14.0,
    "Miscellaneous-fishing": 15.0,
    "Offshore": 20.0,
    "Service-other": 20.0,
    "Miscellaneous-other": 25.0,
}

IMO_TYPE_CANONICAL_LOOKUP = {
    "bulk carrier": "Bulk carrier",
    "bulk": "Bulk carrier",
    "chemical tanker": "Chemical tanker",
    "chemical": "Chemical tanker",
    "container ship": "Container",
    "container": "Container",
    "containership": "Container",
    "general cargo": "General cargo",
    "deck cargo ship": "General cargo",
    "cargo": "General cargo",
    "lng tanker": "Liquefied gas tanker",
    "lpg tanker": "Liquefied gas tanker",
    "liquefied gas tanker": "Liquefied gas tanker",
    "gas tanker": "Liquefied gas tanker",
    "lng": "Liquefied gas tanker",
    "lpg": "Liquefied gas tanker",
    "crude oil tanker": "Oil tanker",
    "oil tanker": "Oil tanker",
    "product tanker": "Oil tanker",
    "oil/chemical tanker": "Chemical tanker",
    "lpg/chemical tanker": "Chemical tanker",
    "other liquids tankers": "Other liquids tankers",
    "asphalt tanker": "Other liquids tankers",
    "bitumen tanker": "Other liquids tankers",
    "passenger vessel": "Ferry-pax only",
    "passenger ship": "Ferry-pax only",
    "ferry-pax only": "Ferry-pax only",
    "passenger ferry": "Ferry-pax only",
    "pax": "Ferry-pax only",
    "cruise ship": "Cruise",
    "cruise": "Cruise",
    "ferry-ropax": "Ferry-RoPax",
    "ropax": "Ferry-RoPax",
    "refrigerated bulk": "Refrigerated bulk",
    "reefer": "Refrigerated bulk",
    "ro-ro": "Ro-Ro",
    "roro": "Ro-Ro",
    "vehicle carrier": "Vehicle",
    "car carrier": "Vehicle",
    "vehicle": "Vehicle",
    "yacht": "Yacht",
    # tug must come before any broad "other" rule
    "tug": "Service tug",
    "service tug": "Service tug",
    "miscellaneous-fishing": "Miscellaneous-fishing",
    "fishing": "Miscellaneous-fishing",
    "floating storage/production": "Offshore",
    "offshore": "Offshore",
    "cable layer": "Service-other",
    "pilot": "Service-other",
    "patrol": "Service-other",
    "research": "Service-other",
    "service-other": "Service-other",
    "miscellaneous-other": "Miscellaneous-other",
    # "other" is last so it does not swallow tug / offshore / etc.
    "other": "Miscellaneous-other",
}


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("CLEAN_AIS_CONFIG")
    if not config_path:
        raise RuntimeError(
            "Set CLEAN_AIS_CONFIG to an explicitly approved year-specific YAML file; "
            "no processing run is authorized by the current CONTRACT."
        )
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
    config["_config_path"] = str(config_path)
    return config


def require_config_value(config, dotted_key):
    value = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing required config key: {dotted_key}")
        value = value[key]
    return value


def get_subproject_path(relative_path_value):
    return (SUBPROJECT_ROOT / Path(relative_path_value)).resolve()


def list_input_files(input_dir):
    return natsort.natsorted(Path(input_dir).glob("*.feather"))


def list_cleaned_point_shard_files(input_dir):
    """List cleaned AIS point shards, excluding audit sidecars."""
    return [
        path
        for path in list_input_files(input_dir)
        if not path.stem.endswith("_gap_log")
        and not path.stem.endswith("_removed_points")
    ]


def ensure_directories(*paths):
    for path in paths:
        iog.create_new_directory(Path(path))


def normalize_ship_id_series(series):
    """Convert vessel_uid (Int64) to clean string. NA → empty string."""
    return series.astype("Int64").astype(str).replace({"<NA>": ""})


def normalize_identifier_series(series):
    series = series.fillna("").astype(str).str.strip()
    series.loc[series.isin(["", "nan", "None", "<NA>"])] = ""
    return series


def normalize_shipname_series(series):
    series = series.fillna("").astype(str).str.strip()
    series.loc[series.isin(["", "nan", "None", "<NA>"])] = "not_available"
    return series


def normalize_dynamic_df(df):
    missing_columns = [column for column in CORE_DYNAMIC_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dynamic AIS shard missing required columns: {missing_columns}")

    df = df.copy()
    # vessel_uid is Int64 in source data — convert to string key
    df[SHIP_ID_COL] = normalize_ship_id_series(df[SHIP_ID_COL])
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    for numeric_column in [LON_COL, LAT_COL, SPEED_COL, COURSE_COL]:
        df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce")
    if HEADING_COL in df.columns:
        df[HEADING_COL] = pd.to_numeric(df[HEADING_COL], errors="coerce")
    if MMSI_COL in df.columns:
        df[MMSI_COL] = normalize_identifier_series(df[MMSI_COL].astype(str))
    if IMO_COL in df.columns:
        df[IMO_COL] = normalize_identifier_series(df[IMO_COL].astype(str))
    if SHIPNAME_COL in df.columns:
        df[SHIPNAME_COL] = normalize_shipname_series(df[SHIPNAME_COL])
    else:
        df[SHIPNAME_COL] = "not_available"
    return df


def find_raw_type_column(df):
    for column in RAW_TYPE_CANDIDATE_COLUMNS:
        if column in df.columns:
            return column
    return None


def find_first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_type_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text


def map_to_imo_level2(*values):
    candidates = []
    for value in values:
        text = normalize_type_text(value)
        if text:
            candidates.append(text)
    for text in candidates:
        lowered = text.lower()
        # exact match first
        if lowered in IMO_TYPE_CANONICAL_LOOKUP:
            return IMO_TYPE_CANONICAL_LOOKUP[lowered]
        # then substring, preserving insertion order (tug before other)
        for key, canonical in IMO_TYPE_CANONICAL_LOOKUP.items():
            if key in lowered:
                return canonical
    return "Miscellaneous-other"


def parse_service_speed_knots(value):
    text = normalize_type_text(value)
    if not text:
        return math.nan
    cleaned = "".join(ch if ch.isdigit() or ch == "." else " " for ch in text)
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return math.nan
    try:
        return float(tokens[0])
    except ValueError:
        return math.nan


def load_specs_table(specs_path):
    specs_path = Path(specs_path)
    if not specs_path.exists():
        raise FileNotFoundError(f"Required specs table not found: {specs_path}")

    if specs_path.suffix.lower() == ".csv":
        specs_df = pd.read_csv(specs_path, dtype={SHIP_ID_COL: str}, encoding="utf-8-sig")
    elif specs_path.suffix.lower() == ".feather":
        specs_df = pd.read_feather(specs_path)
    else:
        raise ValueError(f"Unsupported specs table format: {specs_path.suffix}")

    missing_columns = [column for column in SPECS_REQUIRED_COLUMNS if column not in specs_df.columns]
    if missing_columns:
        raise ValueError(f"Specs table missing required columns: {missing_columns}")

    general_type_column = find_first_existing_column(specs_df, SPECS_GENERAL_TYPE_CANDIDATE_COLUMNS)
    detailed_type_column = find_first_existing_column(specs_df, SPECS_DETAILED_TYPE_CANDIDATE_COLUMNS)
    service_speed_column = find_first_existing_column(specs_df, SPECS_SERVICE_SPEED_CANDIDATE_COLUMNS)
    if general_type_column is None or detailed_type_column is None or service_speed_column is None:
        raise ValueError(
            "Specs table missing required vessel-type/speed columns. "
            f"Expected one of {SPECS_GENERAL_TYPE_CANDIDATE_COLUMNS}, "
            f"one of {SPECS_DETAILED_TYPE_CANDIDATE_COLUMNS}, and "
            f"one of {SPECS_SERVICE_SPEED_CANDIDATE_COLUMNS}."
        )

    specs_df = specs_df.copy()
    specs_df[SHIP_ID_COL] = specs_df[SHIP_ID_COL].fillna("").astype(str).str.strip()
    specs_df = specs_df.loc[specs_df[SHIP_ID_COL] != ""].copy()
    specs_df["specs_general_vessel_type"] = specs_df[general_type_column].map(normalize_type_text)
    specs_df["specs_detailed_vessel_type"] = specs_df[detailed_type_column].map(normalize_type_text)
    specs_df["specs_speed_service_raw"] = specs_df[service_speed_column].map(normalize_type_text)
    specs_df["specs_speed_service_knots"] = specs_df["specs_speed_service_raw"].map(
        parse_service_speed_knots
    )
    specs_df = (
        specs_df.sort_values(
            by=[SHIP_ID_COL, "specs_speed_service_knots"],
            ascending=[True, False],
            na_position="last",
            kind="stable",
        )
        .drop_duplicates(subset=[SHIP_ID_COL], keep="first")
        .reset_index(drop=True)
    )
    return specs_df[
        [
            SHIP_ID_COL,
            "specs_general_vessel_type",
            "specs_detailed_vessel_type",
            "specs_speed_service_knots",
        ]
    ].copy()


def enrich_with_specs(df, specs_df):
    raw_type_column = find_raw_type_column(df)
    enriched = df.merge(specs_df, on=SHIP_ID_COL, how="left")

    if raw_type_column is None:
        enriched["_raw_type_candidate"] = ""
    else:
        enriched["_raw_type_candidate"] = enriched[raw_type_column].map(normalize_type_text)

    enriched["imo_ship_type"] = [
        map_to_imo_level2(specs_general, specs_detailed, raw_type)
        for specs_general, specs_detailed, raw_type in zip(
            enriched.get("specs_general_vessel_type", pd.Series([""] * len(enriched))),
            enriched.get("specs_detailed_vessel_type", pd.Series([""] * len(enriched))),
            enriched["_raw_type_candidate"],
        )
    ]

    enriched["max_speed_knots"] = enriched.get("specs_speed_service_knots", pd.Series(dtype="float64"))
    enriched["max_speed_source"] = "specs_speed_service"

    missing_speed_mask = enriched["max_speed_knots"].isna()
    enriched.loc[missing_speed_mask, "max_speed_knots"] = enriched.loc[
        missing_speed_mask, "imo_ship_type"
    ].map(IMO_LEVEL2_DEFAULT_MAX_SPEED)
    enriched.loc[missing_speed_mask, "max_speed_source"] = "imo_type_default"

    hard_fallback_mask = enriched["max_speed_knots"].isna()
    enriched.loc[hard_fallback_mask, "max_speed_knots"] = 30.0
    enriched.loc[hard_fallback_mask, "max_speed_source"] = "hard_fallback"

    enriched["max_speed_knots"] = pd.to_numeric(enriched["max_speed_knots"], errors="coerce")
    enriched = enriched.drop(columns=["_raw_type_candidate"], errors="ignore")
    return enriched


def haversine_series_nm(lon1, lat1, lon2, lat2):
    """Great-circle distance in nautical miles between two series of points."""
    R_NM = 3440.065
    lo1 = np.radians(lon1.to_numpy(dtype=float, na_value=np.nan))
    la1 = np.radians(lat1.to_numpy(dtype=float, na_value=np.nan))
    lo2 = np.radians(lon2.to_numpy(dtype=float, na_value=np.nan))
    la2 = np.radians(lat2.to_numpy(dtype=float, na_value=np.nan))
    dlon = lo2 - lo1
    dlat = la2 - la1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return 2.0 * R_NM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def drop_invalid_basic_rows(df):
    masks = {
        "null_ship_id": df[SHIP_ID_COL] == "",
        "null_timestamp": df[TIMESTAMP_COL].isna(),
        "invalid_lon": ~df[LON_COL].between(-180, 180, inclusive="both"),
        "invalid_lat": ~df[LAT_COL].between(-90, 90, inclusive="both"),
        "negative_speed": df[SPEED_COL] < 0,
    }
    kept = df.copy()
    counts = {}
    for label, mask in masks.items():
        mask = mask.fillna(False)
        counts[label] = int(mask.sum())
        kept = kept.loc[~mask].copy()
    counts["after_basic_filters"] = int(len(kept))
    return kept, counts


def drop_exact_duplicates(df):
    subset = [
        column
        for column in [SHIP_ID_COL, TIMESTAMP_COL, LON_COL, LAT_COL, SPEED_COL, COURSE_COL, HEADING_COL]
        if column in df.columns
    ]
    before_count = len(df)
    deduped = df.drop_duplicates(subset=subset, keep="first").copy()
    return deduped, before_count - len(deduped)


def resolve_duplicate_ship_timestamp_positions(df):
    df = df.copy()
    duplicate_mask = df.duplicated(subset=[SHIP_ID_COL, TIMESTAMP_COL], keep=False)
    non_duplicate_df = df.loc[~duplicate_mask].copy()
    duplicate_df = df.loc[duplicate_mask].copy()
    if duplicate_df.empty:
        return df, 0

    duplicate_df["_hour_bucket"] = duplicate_df[TIMESTAMP_COL].dt.floor("h")
    vessel_hour_medians = (
        duplicate_df.groupby([SHIP_ID_COL, "_hour_bucket"], dropna=False)
        .agg(median_lon=(LON_COL, "median"), median_lat=(LAT_COL, "median"))
        .reset_index()
    )
    duplicate_df = duplicate_df.merge(vessel_hour_medians, on=[SHIP_ID_COL, "_hour_bucket"], how="left")
    duplicate_df["_distance_to_hour_median"] = (
        (duplicate_df[LON_COL] - duplicate_df["median_lon"]).pow(2)
        + (duplicate_df[LAT_COL] - duplicate_df["median_lat"]).pow(2)
    )
    duplicate_df = (
        duplicate_df.sort_values(
            by=[SHIP_ID_COL, TIMESTAMP_COL, "_distance_to_hour_median"],
            ascending=[True, True, True],
            kind="stable",
        )
        .drop_duplicates(subset=[SHIP_ID_COL, TIMESTAMP_COL], keep="first")
        .copy()
    )
    duplicate_removed = int(duplicate_mask.sum() - len(duplicate_df))
    resolved = pd.concat([non_duplicate_df, duplicate_df], ignore_index=True)
    resolved = resolved.drop(
        columns=["_hour_bucket", "median_lon", "median_lat", "_distance_to_hour_median"],
        errors="ignore",
    )
    return resolved, duplicate_removed


def shard_output_paths(output_dir, input_file):
    input_file = Path(input_file)
    return {
        "cleaned": Path(output_dir) / input_file.name,
        "gap_log": Path(output_dir) / f"{input_file.stem}_gap_log.feather",
        "removed": Path(output_dir) / f"{input_file.stem}_removed_points.feather",
        "stats": Path(output_dir) / f"{input_file.stem}_stats.json",
    }


def shard_outputs_exist(output_dir, input_file):
    return all(path.exists() for path in shard_output_paths(output_dir, input_file).values())


def build_run_log_dir(log_root_dir, step_tag, input_files, config_fingerprint):
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shard_names = [Path(file_path).stem for file_path in input_files]
    hash_source = "|".join(shard_names) + "|" + config_fingerprint
    run_hash = hashlib.md5(hash_source.encode("utf-8")).hexdigest()[:8]
    run_dir = (
        Path(log_root_dir)
        / f"{step_tag}_{run_timestamp}_shards{len(input_files):03d}_{run_hash}"
    )
    ensure_directories(run_dir)
    return run_dir


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_text_lines(path, lines):
    content = "\n".join(lines)
    if content:
        content += "\n"
    Path(path).write_text(content, encoding="utf-8")


def read_stats_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
