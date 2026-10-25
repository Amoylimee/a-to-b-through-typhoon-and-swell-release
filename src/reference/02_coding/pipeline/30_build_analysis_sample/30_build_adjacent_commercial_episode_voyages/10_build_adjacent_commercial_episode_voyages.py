#!/usr/bin/env python3
"""Build formal adjacent-commercial-episode voyages."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

CODING_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import (  # noqa: E402
    city_episodes as c0a,
    city_snaps as c0_snap,
    pair_routes as c0_route,
)
from core import (  # noqa: E402
    candidate_pairs,
    endpoint_city_episodes as c5_city,
    candidate_workers as c1,
    global_qtrim as qtrim,
    od_input_assembly,
    rebuild_main as rebuild,
)


PIPELINE_NAME = "build_adjacent_commercial_episode_voyages"
STEP_NAME = "build_adjacent_commercial_episode_voyages"
FORMAL_OUTPUT_ROOT = (
    CODING_ROOT
    / "data"
    / "processed"
    / "products"
    / "adjacent_commercial_episode_voyages"
)
DEFAULT_OUTPUT_ROOT = FORMAL_OUTPUT_ROOT
FROZEN_SUPPORT_ROOT = (
    CODING_ROOT
    / "data/frozen_inputs/stage30_support_2025/data/processed/build_multicall_first_passage_voyages/main"
)
CITY_REFERENCE_PATH = (
    CODING_ROOT / "data/frozen_inputs/e001_2025_pretrim_golfito_corrected/city_snap_table_2025.csv"
)
MAPPING_PATH = (
    CODING_ROOT / "data/frozen_inputs/stage30_support_2025/data/legacy_inputs/anchorage_to_analytical_city_candidate_v3.csv"
)
NAMED_ANCHORAGE_ROOT = CODING_ROOT / "data/frozen_inputs/named_anchorages"
DEFAULT_LOG_ROOT = CODING_ROOT / ".runs" / PIPELINE_NAME

# Where the monthly OD files are read from. Rebound once in main() from --od-root.
DEFAULT_OD_ROOT = CODING_ROOT / "data" / "processed" / "products" / "our_od"
OD_ROOT = DEFAULT_OD_ROOT
# Calendar cohorts retained here. The middle three analysis cohorts are built by
# the cross-year-buffered entry point, while all five years still contribute to
# the shared city-to-network-node support table.
DEFAULT_YEARS = (2021, 2025)
DEFAULT_CITY_SUPPORT_YEARS = (2021, 2022, 2023, 2024, 2025)
SUPPORTED_YEARS = frozenset(DEFAULT_CITY_SUPPORT_YEARS)
ROUTE_REGIMES = {
    "base": ["northwest"],
    "canal": ["northwest", "suez", "panama"],
    "redsea": ["northwest", "babalmandab", "panama"],
}


OD_COLUMNS = list(c0a.OD_COLUMNS)
FROZEN_DEPENDENCIES = [
    str(path.relative_to(CODING_ROOT))
    for path in (
        sorted(PACKAGE_ROOT.glob("core/*.py"))
        + [PACKAGE_ROOT / "config/main.json",
           PACKAGE_ROOT / "config/frozen/city_snap_2021.json",
           PACKAGE_ROOT / "config/frozen/upstream_2021_enumeration_inputs.json",
           CITY_REFERENCE_PATH,
           CODING_ROOT / "data/frozen_inputs/stage30_support_2025/manifest.json",
           MAPPING_PATH,
           NAMED_ANCHORAGE_ROOT / "named_anchorages_v2_20201104.csv",
           NAMED_ANCHORAGE_ROOT / "named_anchorages_v2_20221206.csv",
           FROZEN_SUPPORT_ROOT / "support/frozen_2021/node_pair_routes_2021.feather",
           FROZEN_SUPPORT_ROOT / "support/endpoint_2025/new_node_pair_routes_2025.feather"]
    )
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if path.suffix == ".feather":
            frame.to_feather(
                temporary,
                compression="zstd",
                compression_level=6,
            )
        elif path.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError(f"Unsupported output format: {path}")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=CODING_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_years(years: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(int(year) for year in years)))
    if not values or not set(values).issubset(SUPPORTED_YEARS):
        raise ValueError(f"Years must be drawn from {sorted(SUPPORTED_YEARS)}")
    return values


def validate_city_support_years(
    analysis_years: Iterable[int],
    city_support_years: Iterable[int],
) -> tuple[int, ...]:
    analysis = validate_years(analysis_years)
    support = validate_years(city_support_years)
    if not set(analysis).issubset(support):
        raise ValueError("City-support years must contain every analysis year")
    return support


def years_tag(years: Iterable[int]) -> str:
    values = tuple(sorted(set(years)))
    if values == DEFAULT_CITY_SUPPORT_YEARS:
        return "2021_2025"
    return "_".join(str(year) for year in values)


def vessel_spec_path(year: int) -> Path:
    return (
        CODING_ROOT / "data" / "processed" / "products" / "vessel_universe"
        / f"analysis_vessel_specs_{year}.csv"
    )


@lru_cache(maxsize=1)
def analysis_vessel_count() -> int:
    """How many vessels the analysis universe holds, read from its definition.

    Five separate assertions in this step check a frame against this number. They
    used to carry the literal 20,319, which meant the approved narrowing to the
    20,134 observed vessels had to be applied in five places and in two config files
    without any of them knowing about the others. Reading it once from the file that
    defines the universe keeps those assertions meaningful — they still fail loudly
    on an unexpected fleet — while leaving one place to change if it is revised again.
    """
    path = (
        CODING_ROOT / "data" / "processed" / "products" / "vessel_universe"
        / "observed_fleet.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"analysis universe is not defined: {path}")
    with path.open(encoding="utf-8") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def od_paths(year: int) -> list[Path]:
    """Monthly OD files for one year, produced by this chain rather than by GFW.

    `15_adapt_od_for_first_passage` rewrites our AIS-derived voyages into GFW's
    monthly column layout, so this reader is unchanged and the frozen identification
    code below never learns that its input changed hands.

    OD_ROOT is set once in main() before any pool starts. The pools fork, so workers
    inherit it. It is read through the module global rather than passed down because
    the frozen identification code between here and the call sites must not change.
    """
    paths = sorted(
        (OD_ROOT / str(year)).glob(
            f"voyages_{year}??.csv"
        )
    )
    if len(paths) != 12:
        raise c0a.DataContractError(
            f"Expected 12 monthly OD files for {year}; found {len(paths)}"
        )
    return paths


def is_headerless_legacy(path: Path) -> bool:
    first = pd.read_csv(path, header=None, nrows=1, dtype=str)
    if first.shape[1] < len(OD_COLUMNS):
        return False
    values = first.iloc[0, : len(OD_COLUMNS)].fillna("").astype(str)
    return (
        values.iloc[0].strip().isdigit()
        and "UTC" in values.iloc[3]
        and "UTC" in values.iloc[4]
    )


def resolve_od_reader(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if is_headerless_legacy(path):
        return (
            {"header": None, "names": OD_COLUMNS},
            {column: column for column in OD_COLUMNS},
        )
    columns = set(pd.read_csv(path, nrows=0).columns)
    aliases = {
        "mmsi": "mmsi" if "mmsi" in columns else "ssvid",
        "vessel_id": "vessel_id",
        "trip_id": "trip_id",
        "trip_start": "trip_start",
        "trip_end": "trip_end",
        "trip_start_anchorage_s2id": (
            "trip_start_anchorage_s2id"
            if "trip_start_anchorage_s2id" in columns
            else "trip_start_anchorage_id"
        ),
        "trip_end_anchorage_s2id": (
            "trip_end_anchorage_s2id"
            if "trip_end_anchorage_s2id" in columns
            else "trip_end_anchorage_id"
        ),
        "trip_start_visit_id": "trip_start_visit_id",
        "trip_end_visit_id": "trip_end_visit_id",
    }
    missing = sorted(set(aliases.values()) - columns)
    if missing:
        raise c0a.DataContractError(f"{path} lacks OD columns: {missing}")
    return {}, aliases


def build_input_manifest(
    years: tuple[int, ...],
    city_support_years: tuple[int, ...],
    candidate_enumeration_mode: str = candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> dict[str, Any]:
    city_support_years = validate_city_support_years(years, city_support_years)
    candidate_enumeration_mode = (
        candidate_pairs.validate_candidate_enumeration_mode(
            candidate_enumeration_mode
        )
    )
    od_assembly = None
    if OD_ROOT.resolve() != DEFAULT_OD_ROOT.resolve():
        verified_assembly = od_input_assembly.verify_od_input_root(OD_ROOT)
        assembly_manifest_path = OD_ROOT / "assembly_manifest.json"
        od_assembly = {
            "path": str(assembly_manifest_path.relative_to(CODING_ROOT)),
            "sha256": sha256_file(assembly_manifest_path),
            "total_source_rows": int(verified_assembly["total_source_rows"]),
            "total_bridge_voyages": int(
                verified_assembly["total_bridge_voyages"]
            ),
            "expected_years": verified_assembly["expected_years"],
        }
    files: list[Path] = []
    for year in city_support_years:
        files.extend(od_paths(year))
        files.append(vessel_spec_path(year))
    files.extend(CODING_ROOT / value for value in FROZEN_DEPENDENCIES)
    records = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": str(path.relative_to(CODING_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "status": "pass",
        "analysis_years": list(years),
        "city_support_years": list(city_support_years),
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "branch": git_value("branch", "--show-current"),
        "base_commit": git_value("rev-parse", "HEAD"),
        "od_assembly": od_assembly,
        "files": records,
    }


def verify_or_write_input_manifest(
    years: tuple[int, ...],
    city_support_years: tuple[int, ...],
    output_root: Path,
    candidate_enumeration_mode: str = candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> dict[str, Any]:
    manifest = build_input_manifest(
        years,
        city_support_years,
        candidate_enumeration_mode,
    )
    path = output_root / f"manifests/input_manifest_{years_tag(years)}.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("candidate_enumeration_mode") != candidate_enumeration_mode:
            raise c0a.DataContractError(
                "Existing input manifest lacks or disagrees with the formal "
                f"candidate mode: {path}"
            )
        stable_keys = (
            "status",
            "analysis_years",
            "city_support_years",
            "candidate_enumeration_mode",
            "branch",
            "od_assembly",
            "files",
        )
        if any(prior.get(key) != manifest.get(key) for key in stable_keys):
            raise c0a.DataContractError("Frozen input manifest changed")
        manifest = prior
    else:
        write_json_atomic(path, manifest)
    return manifest


def load_filtered_od(
    year: int,
    paths: list[Path],
    analysis_mmsi: set[str],
    chunk_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    retained: list[pd.DataFrame] = []
    file_counts = []
    for path in paths:
        reader_options, aliases = resolve_od_reader(path)
        usecols = list(aliases.values())
        raw_rows = 0
        kept_rows = 0
        source_row = 1 if reader_options else 2
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_rows,
            low_memory=False,
            **reader_options,
        ):
            n_rows = len(chunk)
            raw_rows += n_rows
            rename = {source: target for target, source in aliases.items()}
            chunk = chunk.rename(columns=rename)[OD_COLUMNS]
            chunk["mmsi"] = c0a.normalize_token(chunk["mmsi"])
            mask = chunk["mmsi"].isin(analysis_mmsi)
            selected = chunk.loc[mask].copy()
            if not selected.empty:
                selected["source_file"] = path.name
                selected["source_row"] = np.arange(
                    source_row,
                    source_row + n_rows,
                    dtype=np.int64,
                )[mask.to_numpy()]
                retained.append(selected)
                kept_rows += len(selected)
            source_row += n_rows
        file_counts.append(
            {
                "source_file": path.name,
                "headerless_legacy": bool(reader_options),
                "raw_rows": raw_rows,
                "analysis_vessel_rows": kept_rows,
            }
        )
    if not retained:
        raise c0a.DataContractError(f"No {year} rows match the analysis universe")
    trips = pd.concat(retained, ignore_index=True)
    for column in OD_COLUMNS:
        trips[column] = c0a.normalize_token(trips[column])
    if trips["trip_id"].eq("").any():
        raise c0a.DataContractError(f"{year} retained rows contain blank trip IDs")

    trips, evolving_excluded, duplicate_diagnostics = (
        c5_city.resolve_evolving_trip_snapshots(trips)
    )
    reasons = c5_city.source_sentinel_reasons_2025(trips)
    sentinel = reasons.ne("")
    sentinel_excluded = trips.loc[
        sentinel,
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
    sentinel_excluded["exclusion_reason"] = reasons.loc[sentinel]
    excluded = pd.concat(
        [evolving_excluded, sentinel_excluded],
        ignore_index=True,
    )
    trips = trips.loc[~sentinel].copy()
    trips["trip_start"] = c0a.parse_utc_timestamp(trips["trip_start"])
    trips["trip_end"] = c0a.parse_utc_timestamp(trips["trip_end"])
    if trips[["trip_start", "trip_end"]].isna().any(axis=None):
        raise c0a.DataContractError(f"{year} has invalid non-sentinel timestamps")
    if trips["trip_start"].gt(trips["trip_end"]).any():
        raise c0a.DataContractError(f"{year} trip start occurs after trip end")
    if trips[
        ["trip_start_anchorage_s2id", "trip_end_anchorage_s2id"]
    ].eq("").any(axis=None):
        raise c0a.DataContractError(f"{year} retained trip has blank anchorage")
    diagnostics = {
        "source_files": file_counts,
        "raw_rows": int(sum(item["raw_rows"] for item in file_counts)),
        "analysis_vessel_rows_before_trip_dedup": int(
            sum(item["analysis_vessel_rows"] for item in file_counts)
        ),
        **duplicate_diagnostics,
        "source_sentinel_trips_excluded": int(len(sentinel_excluded)),
        "source_sentinel_exclusions_by_reason": {
            str(key): int(value)
            for key, value in sentinel_excluded[
                "exclusion_reason"
            ].value_counts().items()
        },
        "unique_trips_before_source_identity_resolution": int(len(trips)),
        "analysis_mmsi_with_trips": int(trips["mmsi"].nunique()),
    }
    return trips, excluded, diagnostics


def build_year_episodes(
    year: int,
    output_root: Path,
    chunk_rows: int,
    candidate_enumeration_mode: str = candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> dict[str, Any]:
    output_dir = output_root / f"intermediate/{year}/city_episodes"
    manifest_path = output_dir / "output_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_mode = manifest.get("candidate_enumeration_mode")
        outputs_are_current = all(
            (CODING_ROOT / item["path"]).is_file()
            and sha256_file(CODING_ROOT / item["path"]) == item["sha256"]
            for item in manifest["outputs"].values()
        )
        if outputs_are_current and manifest_mode != candidate_enumeration_mode:
            raise c0a.DataContractError(
                "City-episode output root already holds candidate mode "
                f"{manifest_mode!r}; use a separate output root for "
                f"{candidate_enumeration_mode!r}"
            )
        if outputs_are_current:
            return manifest

    vessels = c0a.load_vessel_universe(vessel_spec_path(year), analysis_vessel_count())
    mapping = c0a.load_v3_mapping(MAPPING_PATH)
    trips, excluded, od_diagnostics = load_filtered_od(
        year,
        od_paths(year),
        set(vessels["mmsi"]),
        chunk_rows,
    )
    trips, incoherent_excluded, incoherent_diagnostics = (
        c5_city.remove_incoherent_source_tracks(trips)
    )
    excluded = pd.concat([excluded, incoherent_excluded], ignore_index=True)
    (
        trips,
        identity_excluded,
        identity_decisions,
        identity_diagnostics,
    ) = c0a.resolve_source_identity_overlaps(
        trips,
        mapping.set_index("anchorage_id")["country_iso3"],
        year,
        "coherent_source_track_v1",
    )
    excluded = pd.concat([excluded, identity_excluded], ignore_index=True)
    identity_decisions = identity_decisions.merge(
        vessels[["mmsi", "imo", "vessel_name"]],
        on="mmsi",
        how="left",
        validate="many_to_one",
    )
    identity_decisions = identity_decisions.rename(
        columns={
            "active_endpoint_months_2021": f"active_endpoint_months_{year}",
            "coverage_hours_2021": f"coverage_hours_{year}",
        }
    )
    events = c0a.build_endpoint_events(trips, vessels)
    calls, missing_mapping, call_diagnostics = c0a.build_calls(events, mapping)
    if not missing_mapping.empty:
        examples = (
            missing_mapping["anchorage_id"].drop_duplicates().head(10).tolist()
        )
        raise c0a.DataContractError(
            f"{year} retained calls lack v3 mapping: {examples}"
        )
    trips, overlap_excluded, overlap_diagnostics = (
        c5_city.remove_mmsi_with_overlapping_calls(trips, calls)
    )
    excluded = pd.concat([excluded, overlap_excluded], ignore_index=True)
    if overlap_diagnostics["unsupported_overlapping_call_mmsi"]:
        events = c0a.build_endpoint_events(trips, vessels)
        calls, missing_mapping, call_diagnostics = c0a.build_calls(events, mapping)
        if not missing_mapping.empty:
            raise c0a.DataContractError(
                f"{year} mapping gap remains after overlap exclusion"
            )
    if calls["overlaps_previous_call"].any():
        raise c0a.DataContractError(
            f"{year} has overlapping reconstructed calls after exclusion"
        )
    calls, episodes, episode_diagnostics = c0a.merge_same_city_episodes(calls)
    calls["episode_id"] = calls["episode_id"].str.replace(
        "_2021_", f"_{year}_", regex=False
    )
    episodes["episode_id"] = episodes["episode_id"].str.replace(
        "_2021_", f"_{year}_", regex=False
    )
    pairs, pair_diagnostics = c0a.enumerate_adjacent_episode_pairs(
        episodes,
        candidate_enumeration_mode,
    )
    pairs = pairs.rename(
        columns={
            "candidate_count_2021": f"candidate_count_{year}",
            "vessel_count_2021": f"vessel_count_{year}",
        }
    )
    diagnostics = {
        "status": "pass",
        "analysis_year": year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "operational_parameter_year": 2021,
        "analysis_vessels": int(len(vessels)),
        "supported_analysis_vessels": int(episodes["imo"].nunique()),
        "endpoint_events": int(len(events)),
        **od_diagnostics,
        **incoherent_diagnostics,
        **identity_diagnostics,
        **overlap_diagnostics,
        **call_diagnostics,
        **episode_diagnostics,
        **pair_diagnostics,
    }
    paths = {
        "excluded_trips": output_dir / "excluded_trip_records.csv",
        "source_identity_decisions": output_dir
        / "source_identity_resolution.csv",
        "endpoint_events": output_dir / "endpoint_events.feather",
        "calls": output_dir / "ordered_port_calls.feather",
        "episodes": output_dir / "ordered_city_episodes.feather",
        "pair_manifest": output_dir / "adjacent_episode_pair_manifest.csv",
        "summary": output_dir / "summary.json",
    }
    write_frame_atomic(paths["excluded_trips"], excluded)
    write_frame_atomic(
        paths["source_identity_decisions"],
        identity_decisions,
    )
    write_frame_atomic(paths["endpoint_events"], events)
    write_frame_atomic(paths["calls"], calls)
    write_frame_atomic(paths["episodes"], episodes)
    write_frame_atomic(paths["pair_manifest"], pairs)
    write_json_atomic(paths["summary"], diagnostics)
    outputs = {
        name: {
            "path": str(path.relative_to(CODING_ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    manifest = {
        "status": "pass",
        "analysis_year": year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "diagnostics": diagnostics,
        "outputs": outputs,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def verified_manifest_output(
    manifest: dict[str, Any],
    key: str,
) -> Path:
    record = manifest["outputs"][key]
    path = resolve_manifest_record_path(record)
    if sha256_file(path) != record["sha256"]:
        raise c0a.DataContractError(f"Manifest output changed: {path}")
    return path


def resolve_manifest_record_path(record: dict[str, Any]) -> Path:
    recorded = Path(record["path"])
    return CODING_ROOT / recorded


def episode_manifest_identity(manifest: dict[str, Any]) -> str:
    return canonical_hash(
        {
            name: {
                "path": record["path"],
                "size_bytes": int(record["size_bytes"]),
                "sha256": record["sha256"],
            }
            for name, record in sorted(manifest["outputs"].items())
        }
    )


def load_candidate_inputs(
    episode_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    episodes = pd.read_feather(
        verified_manifest_output(episode_manifest, "episodes"),
        columns=c1.EPISODE_COLUMNS,
    ).sort_values(["mmsi", "episode_sequence"], kind="stable").reset_index(
        drop=True
    )
    events = pd.read_feather(
        verified_manifest_output(episode_manifest, "endpoint_events"),
        columns=c1.EVENT_COLUMNS,
    )
    departures = events.loc[
        events["event_type"].eq("departure"),
        ["mmsi", "trip_id", "event_time", "call_id"],
    ].rename(
        columns={
            "event_time": "departure_time",
            "call_id": "departure_call_id",
        }
    )
    arrivals = events.loc[
        events["event_type"].eq("arrival"),
        ["mmsi", "trip_id", "event_time", "call_id"],
    ].rename(
        columns={
            "event_time": "arrival_time",
            "call_id": "arrival_call_id",
        }
    )
    if (
        departures["trip_id"].duplicated().any()
        or arrivals["trip_id"].duplicated().any()
    ):
        raise c0a.DataContractError("Endpoint events duplicate a trip boundary")
    legs = departures.merge(
        arrivals,
        on=["mmsi", "trip_id"],
        how="inner",
        validate="one_to_one",
    ).sort_values(
        ["mmsi", "departure_time", "arrival_time", "trip_id"],
        kind="stable",
    ).reset_index(drop=True)
    if len(legs) != len(departures) or len(legs) != len(arrivals):
        raise c0a.DataContractError("A retained trip lacks an endpoint event")
    if (legs["arrival_time"] <= legs["departure_time"]).any():
        raise c0a.DataContractError("A retained trip has non-positive duration")
    legs["leg_sequence"] = legs.groupby("mmsi", sort=False).cumcount().astype(
        np.int64
    )
    calls = pd.read_feather(
        verified_manifest_output(episode_manifest, "calls"),
        columns=c1.CALL_COLUMNS,
    )
    if calls["call_id"].duplicated().any():
        raise c0a.DataContractError("Ordered calls duplicate call_id")
    call_meta = {
        str(call_id): {
            "topology_city_id": str(city or ""),
            "episode_id": str(episode or ""),
        }
        for call_id, city, episode in calls.itertuples(index=False, name=None)
    }
    return episodes, legs, call_meta


def candidate_execution_config(
    year: int,
    candidate_enumeration_mode: str = candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> dict[str, Any]:
    frozen = json.loads(
        (
            CODING_ROOT
            / "pipeline/30_build_analysis_sample/30_build_adjacent_commercial_episode_voyages/config/frozen/"
            "upstream_2021_enumeration_inputs.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "period_windows": [
            {
                "period_id": f"{year}_full_year",
                "start": f"{year}-01-01T00:00:00Z",
                "end": f"{year + 1}-01-01T00:00:00Z",
            }
        ],
        "liner_proxy_types": frozen["liner_proxy_types"],
        "tramp_proxy_types": frozen["tramp_proxy_types"],
    }


def run_candidate_tasks(
    year: int,
    output_root: Path,
    episodes: pd.DataFrame,
    legs: pd.DataFrame,
    call_meta: dict[str, dict[str, str]],
    max_workers: int,
    task_count: int,
    task_limit: int | None,
    source_identity_sha256: str,
    candidate_enumeration_mode: str = candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> list[dict[str, Any]]:
    output_dir = output_root / f"intermediate/{year}/candidate_occurrences"
    candidate_dir = output_dir / "candidate_passages"
    stop_dir = output_dir / "raw_stationary_episodes"
    sidecar_dir = output_dir / "parts"
    for directory in (candidate_dir, stop_dir, sidecar_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config = candidate_execution_config(year, candidate_enumeration_mode)
    contract = {
        "analysis_year": year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "operational_parameter_year": 2021,
        "episode_rows": len(episodes),
        "leg_rows": len(legs),
        "task_count": task_count,
        "source_identity_sha256": source_identity_sha256,
        "config": config,
        "frozen_c1_core_sha256": sha256_file(
            CODING_ROOT
            / "pipeline/30_build_analysis_sample/30_build_adjacent_commercial_episode_voyages/core/candidate_core.py"
        ),
    }
    contract_hash = canonical_hash(contract)
    contract_path = output_dir / "execution_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise c0a.DataContractError(
                f"{year} candidate execution contract changed"
            )
    else:
        write_json_atomic(contract_path, contract)
    tasks = c1.partition_vessels(episodes, task_count)
    c1.initialize_globals(episodes, legs, call_meta, config, contract_hash)
    complete = []
    pending = []
    for task_id, mmsi_values in enumerate(tasks):
        candidate_path = candidate_dir / f"part-{task_id:03d}.feather"
        stop_path = stop_dir / f"part-{task_id:03d}.feather"
        sidecar_path = sidecar_dir / f"part-{task_id:03d}.json"
        if c1.part_is_complete(
            candidate_path,
            stop_path,
            sidecar_path,
            contract_hash,
        ):
            complete.append(
                json.loads(sidecar_path.read_text(encoding="utf-8"))
            )
        else:
            pending.append(
                (
                    task_id,
                    mmsi_values,
                    str(candidate_path),
                    str(stop_path),
                    str(sidecar_path),
                )
            )
    selected = pending if task_limit is None else pending[:task_limit]
    if selected:
        with ProcessPoolExecutor(
            max_workers=min(max_workers, len(selected), 16),
            mp_context=mp.get_context("fork"),
        ) as executor:
            futures = {
                executor.submit(c1.process_task, *task): task[0]
                for task in selected
            }
            for future in as_completed(futures):
                payload = future.result()
                complete.append(payload)
                print(
                    f"[adjacent-episode {year}] candidate task "
                    f"{payload['task_id']:03d}: "
                    f"{payload['candidate_rows']:,}",
                    flush=True,
                )
    write_json_atomic(
        output_dir / "run_status.json",
        {
            "status": "partial" if len(complete) != task_count else "pass",
            "analysis_year": year,
            "candidate_enumeration_mode": candidate_enumeration_mode,
            "task_count": task_count,
            "completed_task_count": len(complete),
            "pending_task_count": task_count - len(complete),
        },
    )
    return sorted(complete, key=lambda item: int(item["task_id"]))


def validate_candidate_outputs(
    year: int,
    output_root: Path,
    episode_manifest: dict[str, Any],
    payloads: list[dict[str, Any]],
    task_count: int,
    candidate_enumeration_mode: str = candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> dict[str, Any]:
    candidate_enumeration_mode = (
        candidate_pairs.validate_candidate_enumeration_mode(
            candidate_enumeration_mode
        )
    )
    manifest_mode = episode_manifest.get("candidate_enumeration_mode")
    if manifest_mode != candidate_enumeration_mode:
        raise c0a.DataContractError(
            "Candidate validation mode differs from the city-episode manifest: "
            f"{candidate_enumeration_mode!r} != {manifest_mode!r}"
        )
    if len(payloads) != task_count:
        return {
            "status": "partial",
            "analysis_year": year,
            "completed_task_count": len(payloads),
            "pending_task_count": task_count - len(payloads),
        }
    output_dir = output_root / f"intermediate/{year}/candidate_occurrences"
    pair_frames = []
    total_rows = 0
    max_residual = 0.0
    duplicate_ids = 0
    nonadjacent_candidates = 0
    intermediate_city_candidates = 0
    all_ids = []
    for task_id in range(task_count):
        path = output_dir / "candidate_passages" / f"part-{task_id:03d}.feather"
        columns = [
            "candidate_id",
            "pair_id",
            "imo",
            "time_identity_residual_hours",
        ]
        if (
            candidate_enumeration_mode
            == candidate_pairs.ADJACENT_COMMERCIAL_EPISODES
        ):
            columns.extend(
                [
                    "origin_episode_sequence",
                    "dest_episode_sequence",
                    "n_intermediate_city_episodes",
                ]
            )
        frame = pd.read_feather(
            path,
            columns=columns,
        )
        total_rows += len(frame)
        max_residual = max(
            max_residual,
            float(frame["time_identity_residual_hours"].abs().max()),
        )
        duplicate_ids += int(frame["candidate_id"].duplicated().sum())
        if (
            candidate_enumeration_mode
            == candidate_pairs.ADJACENT_COMMERCIAL_EPISODES
        ):
            nonadjacent_candidates += int(
                frame["dest_episode_sequence"]
                .sub(frame["origin_episode_sequence"])
                .ne(1)
                .sum()
            )
            intermediate_city_candidates += int(
                frame["n_intermediate_city_episodes"].ne(0).sum()
            )
        all_ids.append(frame["candidate_id"])
        pair_frames.append(
            frame.groupby("pair_id", sort=True)
            .agg(
                observed_candidates=("imo", "size"),
                observed_vessels=("imo", "nunique"),
            )
            .reset_index()
        )
    if pd.concat(all_ids, ignore_index=True).duplicated().any():
        raise c0a.DataContractError(f"{year} candidate IDs duplicate across parts")
    observed = (
        pd.concat(pair_frames, ignore_index=True)
        .groupby("pair_id", sort=True)[
            ["observed_candidates", "observed_vessels"]
        ]
        .sum()
        .reset_index()
    )
    expected = pd.read_csv(
        verified_manifest_output(episode_manifest, "pair_manifest"),
        usecols=[
            "pair_id",
            f"candidate_count_{year}",
            f"vessel_count_{year}",
        ],
    )
    audit = expected.merge(
        observed,
        on="pair_id",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    difference = (
        audit["_merge"].ne("both")
        | audit[f"candidate_count_{year}"].ne(audit["observed_candidates"])
        | audit[f"vessel_count_{year}"].ne(audit["observed_vessels"])
    )
    if difference.any():
        raise c0a.DataContractError(
            f"{year} pair reproduction differs for {int(difference.sum())} rows"
        )
    if max_residual > 1e-6:
        raise c0a.DataContractError(
            f"{year} time decomposition residual exceeds tolerance"
        )
    if nonadjacent_candidates or intermediate_city_candidates:
        raise c0a.DataContractError(
            f"{year} adjacent-mode invariant failed: "
            f"nonadjacent={nonadjacent_candidates}, "
            f"with_intermediate_city={intermediate_city_candidates}"
        )
    summary = {
        "status": "pass",
        "analysis_year": year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "task_count": task_count,
        "candidate_rows": total_rows,
        "pair_rows": len(observed),
        "raw_stop_rows": int(sum(item["raw_stop_rows"] for item in payloads)),
        "max_abs_time_identity_residual_hours": max_residual,
        "duplicate_candidate_ids": duplicate_ids,
        "nonadjacent_candidate_rows": nonadjacent_candidates,
        "candidate_rows_with_intermediate_city": intermediate_city_candidates,
        "pair_reproduction_difference_rows": int(difference.sum()),
    }
    outputs = {}
    for folder in ("candidate_passages", "raw_stationary_episodes"):
        for path in sorted((output_dir / folder).glob("part-*.feather")):
            outputs[str(path.relative_to(CODING_ROOT))] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest = {
        "status": "pass",
        "analysis_year": year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "diagnostics": summary,
        "outputs": outputs,
    }
    write_json_atomic(output_dir / "output_manifest.json", manifest)
    return manifest


def build_combined_city_snaps(
    city_support_years: tuple[int, ...],
    output_root: Path,
    episode_manifests: dict[int, dict[str, Any]],
) -> Path:
    tag = years_tag(city_support_years)
    target = output_root / f"support/city_snap_table_{tag}.csv"
    if target.is_file():
        return target
    pair_frames = [
        pd.read_csv(
            verified_manifest_output(episode_manifests[year], "pair_manifest")
        )[c0_snap.PAIR_COLUMNS]
        for year in city_support_years
    ]
    pairs = pd.concat(pair_frames, ignore_index=True).drop_duplicates("pair_id")
    cities, _ = c0_snap.build_pair_city_universe(pairs)
    # Adopted coordinates include the reviewed Golfito correction. Reusing this
    # reference does not process voyages from its historical reference year.
    frozen_frames = [pd.read_csv(CITY_REFERENCE_PATH)]
    frozen = pd.concat(frozen_frames, ignore_index=True)
    conflicts = frozen.groupby("city_id")["network_node_id"].nunique()
    if conflicts.gt(1).any():
        raise c0a.DataContractError("Endpoint city snaps disagree across frozen years")
    frozen = frozen.drop_duplicates("city_id", keep="first")
    required = set(cities["city_id"])
    reused = frozen.loc[frozen["city_id"].isin(required)].copy()
    missing = cities.loc[~cities["city_id"].isin(set(reused["city_id"]))].copy()
    if missing.empty:
        combined = reused
        extension_diagnostics: dict[str, Any] = {"extension_city_count": 0}
    else:
        calls = pd.concat(
            [
                pd.read_feather(
                    verified_manifest_output(episode_manifests[year], "calls"),
                    columns=c0_snap.CALL_COLUMNS,
                )
                for year in city_support_years
            ],
            ignore_index=True,
        )
        config = json.loads(
            (
                CODING_ROOT
                / "pipeline/30_build_analysis_sample/30_build_adjacent_commercial_episode_voyages/config/frozen/"
                "city_snap_2021.json"
            ).read_text(encoding="utf-8")
        )
        lookup, _, _ = c0_snap.load_coordinate_lookup(coordinate_specifications(config))
        overrides = [
            item
            for item in config["coordinate_role_overrides"]
            if item["city_id"] in set(missing["city_id"])
        ]
        coordinates, lineage, coordinate_diagnostics = (
            c0_snap.build_city_coordinates(
                calls,
                missing,
                lookup,
                float(config["coordinate_spread_review_threshold_deg"]),
                overrides,
            )
        )
        extension, snap_diagnostics = c0_snap.add_network_snaps(
            coordinates,
            float(config["snap_review_threshold_km"]),
        )
        if list(extension.columns) != list(reused.columns):
            extension = extension[reused.columns]
        write_frame_atomic(
            output_root / f"support/city_snap_extension_{tag}.csv",
            extension,
        )
        write_frame_atomic(
            output_root / f"support/city_snap_extension_lineage_{tag}.csv",
            lineage,
        )
        combined = pd.concat([reused, extension], ignore_index=True)
        extension_diagnostics = {
            "extension_city_count": int(len(extension)),
            "coordinate": coordinate_diagnostics,
            "snap": snap_diagnostics,
        }
    combined = combined.sort_values("city_id", kind="stable").reset_index(drop=True)
    if combined["city_id"].duplicated().any() or set(combined["city_id"]) != required:
        raise c0a.DataContractError("Combined city snaps do not cover the pair universe")
    write_frame_atomic(target, combined)
    write_json_atomic(
        output_root / "support/city_snap_summary.json",
        {
            "status": "pass",
            "city_support_years": list(city_support_years),
            "required_city_count": int(len(required)),
            "reused_frozen_city_count": int(len(reused)),
            "candidate_enumeration_mode": (
                candidate_pairs.MAIN_CANDIDATE_ENUMERATION_MODE
            ),
            **extension_diagnostics,
            "output_sha256": sha256_file(target),
        },
    )
    return target


def known_node_routes() -> pd.DataFrame:
    paths = [
        FROZEN_SUPPORT_ROOT
        / "support/frozen_2021/node_pair_routes_2021.feather",
        FROZEN_SUPPORT_ROOT
        / "support/endpoint_2025/new_node_pair_routes_2025.feather",
    ]
    paths.extend(
        sorted(
            FROZEN_SUPPORT_ROOT.glob(
                "support/*/direct_route_tasks/new_node_pair_routes.feather"
            )
        )
    )
    paths.extend(
        sorted(
            FROZEN_SUPPORT_ROOT.glob(
                "intermediate/*/filters/transition_route_tasks/"
                "new_node_pair_routes.feather"
            )
        )
    )
    routes = pd.concat([pd.read_feather(path) for path in paths], ignore_index=True)
    distance_columns = ["d_base", "d_canal", "d_redsea", "d_yardstick"]
    duplicate = routes.loc[routes["node_pair_id"].duplicated(keep=False)]
    if not duplicate.empty:
        spread = duplicate.groupby("node_pair_id")[distance_columns].nunique(
            dropna=False
        )
        if spread.gt(1).any(axis=None):
            raise c0a.DataContractError("Frozen node-route distances disagree")
    return routes.drop_duplicates("node_pair_id", keep="first")


def coordinate_specifications(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the fixed coordinate sources to A-owned copies without fallback."""
    return [
        {**item, "path": str(NAMED_ANCHORAGE_ROOT / Path(item["path"]).name)}
        for item in config["coordinate_source_precedence"]
    ]


def attach_city_nodes(
    pairs: pd.DataFrame,
    city_snap_path: Path,
) -> pd.DataFrame:
    cities = pd.read_csv(city_snap_path)
    origin = cities.rename(
        columns={
            "city_id": "origin_city",
            "network_node_id": "origin_network_node_id",
            "network_lon": "origin_network_lon",
            "network_lat": "origin_network_lat",
            "snap_distance_km": "snap_dist_o",
            "routing_gateway_override": "origin_routing_gateway_override",
        }
    )
    destination = cities.rename(
        columns={
            "city_id": "dest_city",
            "network_node_id": "dest_network_node_id",
            "network_lon": "dest_network_lon",
            "network_lat": "dest_network_lat",
            "snap_distance_km": "snap_dist_d",
            "routing_gateway_override": "dest_routing_gateway_override",
        }
    )
    output = pairs.merge(
        origin[
            [
                "origin_city",
                "origin_network_node_id",
                "origin_network_lon",
                "origin_network_lat",
                "snap_dist_o",
                "origin_routing_gateway_override",
            ]
        ],
        on="origin_city",
        how="left",
        validate="many_to_one",
    ).merge(
        destination[
            [
                "dest_city",
                "dest_network_node_id",
                "dest_network_lon",
                "dest_network_lat",
                "snap_dist_d",
                "dest_routing_gateway_override",
            ]
        ],
        on="dest_city",
        how="left",
        validate="many_to_one",
    )
    node_columns = [
        "origin_network_node_id",
        "origin_network_lon",
        "origin_network_lat",
        "dest_network_node_id",
        "dest_network_lon",
        "dest_network_lat",
    ]
    if output[node_columns].isna().any(axis=None):
        raise c0a.DataContractError("A route endpoint lacks a city snap")
    output["node_pair_id"] = (
        output["origin_network_node_id"]
        + "__"
        + output["dest_network_node_id"]
    )
    output["same_network_node"] = output["origin_network_node_id"].eq(
        output["dest_network_node_id"]
    )
    return output


def ensure_city_snap_coverage(
    required_cities: set[str],
    city_snap_path: Path,
    output_root: Path,
    source_years: tuple[int, ...],
) -> None:
    existing = pd.read_csv(city_snap_path)
    missing_ids = sorted(required_cities - set(existing["city_id"]))
    if not missing_ids:
        return
    split = [city_id.split(":", 1) for city_id in missing_ids]
    missing = pd.DataFrame(
        {
            "city_id": missing_ids,
            "iso3": [value[0] for value in split],
            "city_name": [value[1] for value in split],
            "origin_pair_count": 0,
            "dest_pair_count": 0,
            "total_pair_endpoint_roles": 0,
        }
    )
    calls = pd.concat(
        [
            pd.read_feather(
                output_root / f"intermediate/{year}/city_episodes/ordered_port_calls.feather",
                columns=c0_snap.CALL_COLUMNS,
            )
            for year in source_years
        ],
        ignore_index=True,
    )
    config = json.loads(
        (
            CODING_ROOT
            / "pipeline/30_build_analysis_sample/30_build_adjacent_commercial_episode_voyages/config/frozen/"
            "city_snap_2021.json"
        ).read_text(encoding="utf-8")
    )
    lookup, _, _ = c0_snap.load_coordinate_lookup(coordinate_specifications(config))
    overrides = [
        item
        for item in config["coordinate_role_overrides"]
        if item["city_id"] in set(missing_ids)
    ]
    coordinates, lineage, _ = c0_snap.build_city_coordinates(
        calls,
        missing,
        lookup,
        float(config["coordinate_spread_review_threshold_deg"]),
        overrides,
    )
    extension, _ = c0_snap.add_network_snaps(
        coordinates,
        float(config["snap_review_threshold_km"]),
    )
    extension = extension[existing.columns]
    combined = (
        pd.concat([existing, extension], ignore_index=True)
        .sort_values("city_id", kind="stable")
        .reset_index(drop=True)
    )
    if combined["city_id"].duplicated().any():
        raise c0a.DataContractError("Supplementary city snaps duplicate a city")
    write_frame_atomic(city_snap_path, combined)
    write_frame_atomic(
        output_root / "support/transition_city_snap_extension.csv",
        extension,
    )
    write_frame_atomic(
        output_root / "support/transition_city_snap_extension_lineage.csv",
        lineage,
    )
    summary_path = output_root / "support/city_snap_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["transition_extension_city_count"] = int(len(extension))
    summary["final_supported_city_count"] = int(len(combined))
    summary["output_sha256"] = sha256_file(city_snap_path)
    write_json_atomic(summary_path, summary)


def route_missing_nodes(
    node_pairs: pd.DataFrame,
    route_dir: Path,
    max_workers: int,
) -> pd.DataFrame:
    known = known_node_routes()
    required_ids = set(node_pairs["node_pair_id"])
    reused = known.loc[known["node_pair_id"].isin(required_ids)].copy()
    unresolved = node_pairs.loc[
        ~node_pairs["node_pair_id"].isin(set(reused["node_pair_id"]))
    ].copy()
    if unresolved.empty:
        return reused
    task_manifest, tasks = c0_route.build_task_manifest(
        unresolved,
        route_dir,
        origin_nodes_per_task=20,
    )
    write_frame_atomic(route_dir / "full_task_list.csv", task_manifest)
    pending = [task for task in tasks if not Path(task["output_path"]).is_file()]
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(max_workers, len(pending), 16),
            initializer=c0_route.initialize_route_worker,
            initargs=(ROUTE_REGIMES,),
        ) as executor:
            futures = {
                executor.submit(c0_route.run_route_task, task): task
                for task in pending
            }
            failures = []
            for future in as_completed(futures):
                result = future.result()
                if result["status"] != "success":
                    failures.append(result)
            if failures:
                raise c0a.DataContractError(
                    f"{len(failures)} route tasks failed in {route_dir}"
                )
    chunks = [pd.read_feather(task["output_path"]) for task in tasks]
    routed = c0_route.add_yardstick_fields(
        pd.concat(chunks, ignore_index=True),
        list(ROUTE_REGIMES),
    )
    write_frame_atomic(route_dir / "new_node_pair_routes.feather", routed)
    combined = pd.concat([reused, routed], ignore_index=True)
    if set(combined["node_pair_id"]) != required_ids:
        raise c0a.DataContractError("Node-route support remains incomplete")
    return combined.drop_duplicates("node_pair_id")


def build_year_yardstick(
    year: int,
    output_root: Path,
    episode_manifest: dict[str, Any],
    city_snap_path: Path,
    max_workers: int,
) -> Path:
    target = output_root / f"support/{year}/yardstick_table_{year}.feather"
    if target.is_file():
        return target
    pairs = pd.read_csv(verified_manifest_output(episode_manifest, "pair_manifest"))
    pairs = attach_city_nodes(pairs, city_snap_path)
    node_columns = [
        "node_pair_id",
        "origin_network_node_id",
        "origin_network_lon",
        "origin_network_lat",
        "dest_network_node_id",
        "dest_network_lon",
        "dest_network_lat",
    ]
    nodes = pairs[node_columns].drop_duplicates("node_pair_id")
    routes = route_missing_nodes(
        nodes,
        output_root / f"support/{year}/direct_route_tasks",
        max_workers,
    )
    route_columns = [
        "node_pair_id",
        "d_base",
        "d_canal",
        "d_redsea",
        "d_yardstick",
        "regime_used",
        "feasible_flags",
    ]
    output = pairs.merge(
        routes[route_columns],
        on="node_pair_id",
        how="left",
        validate="many_to_one",
    )
    if output["d_yardstick"].isna().any():
        raise c0a.DataContractError(f"{year} direct yardstick remains incomplete")
    columns = [
        "pair_id",
        "origin_city",
        "dest_city",
        f"candidate_count_{year}",
        f"vessel_count_{year}",
        "d_base",
        "d_canal",
        "d_redsea",
        "d_yardstick",
        "regime_used",
        "snap_dist_o",
        "snap_dist_d",
        "feasible_flags",
        "same_network_node",
        "origin_network_node_id",
        "dest_network_node_id",
        "origin_routing_gateway_override",
        "dest_routing_gateway_override",
    ]
    output = output[columns].sort_values("pair_id", kind="stable").reset_index(
        drop=True
    )
    write_frame_atomic(target, output)
    return target


def materialize_c2(
    year: int,
    output_root: Path,
    task_count: int,
    candidate_enumeration_mode: str,
) -> dict[str, Any]:
    c2_filter = rebuild.c2_filter
    source_dir = output_root / f"intermediate/{year}/candidate_occurrences"
    artifact_dir = output_root / f"intermediate/{year}/filters"
    records = []
    totals = {"input_candidates": 0, "retained_candidates": 0}
    for task_id in range(task_count):
        candidate_path = source_dir / f"candidate_passages/part-{task_id:03d}.feather"
        stop_path = source_dir / f"raw_stationary_episodes/part-{task_id:03d}.feather"
        target = artifact_dir / f"c2_retained/part-{task_id:03d}.feather"
        sidecar = artifact_dir / f"c2_sidecars/part-{task_id:03d}.json"
        source_hash = canonical_hash(
            {
                "candidate": sha256_file(candidate_path),
                "stop": sha256_file(stop_path),
                "rule": "strict_same_location_gt_30_days",
            }
        )
        if target.is_file() and sidecar.is_file():
            prior = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                prior.get("contract_hash") == source_hash
                and prior.get("retained_sha256") == sha256_file(target)
            ):
                records.append(prior)
                totals["input_candidates"] += prior["input_candidates"]
                totals["retained_candidates"] += prior["retained_candidates"]
                continue
        candidates = pd.read_feather(candidate_path)
        stops = pd.read_feather(stop_path)
        stops["is_strict_same_location"] = (
            stops["location_resolution"].isin(
                ["same_call", "same_city", "same_episode"]
            )
            & stops["resolved_topology_city_id"].fillna("").ne("")
        )
        stops["idle_gap_exceeds_unadjusted_threshold"] = stops[
            "is_strict_same_location"
        ] & stops["dwell_days"].gt(30.0)
        stops["idle_gap_threshold_days"] = 30.0
        annotated = c2_filter.annotate_candidate_breaks(candidates, stops)
        annotated["r4_rule_version"] = "strict_same_location_gt_30_days"
        retained = c2_filter.add_post_r4_support(
            annotated.loc[~annotated["r4_idle_break"]].reset_index(drop=True)
        )
        write_frame_atomic(target, retained)
        record = {
            "task_id": task_id,
            "contract_hash": source_hash,
            "input_candidates": int(len(candidates)),
            "retained_candidates": int(len(retained)),
            "excluded_candidates": int(len(candidates) - len(retained)),
            "retained_path": str(target.relative_to(CODING_ROOT)),
            "retained_sha256": sha256_file(target),
        }
        write_json_atomic(sidecar, record)
        records.append(record)
        totals["input_candidates"] += len(candidates)
        totals["retained_candidates"] += len(retained)
    totals["excluded_candidates"] = (
        totals["input_candidates"] - totals["retained_candidates"]
    )
    manifest = {
        "status": "pass",
        "analysis_year": year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "rule": "strict_same_location_gt_30_days",
        "parts": records,
        "summary": {key: int(value) for key, value in totals.items()},
    }
    write_json_atomic(artifact_dir / "c2_manifest.json", manifest)
    return manifest


def build_transition_yardstick(
    year: int,
    output_root: Path,
    c2_manifest: dict[str, Any],
    city_snap_path: Path,
    direct_yardstick: Path,
    max_workers: int,
    source_years: tuple[int, ...],
) -> Path:
    artifact_dir = output_root / f"intermediate/{year}/filters"
    target = artifact_dir / "transition_yardstick.feather"
    if target.is_file():
        return target
    usage = rebuild.c3.enumerate_transition_usage(c2_manifest, ">")
    direct = pd.read_feather(
        direct_yardstick,
        columns=["pair_id", "d_base", "same_network_node"],
    )
    transition = usage.merge(
        direct, on="pair_id", how="left", validate="one_to_one"
    )
    missing = transition.loc[
        transition["d_base"].isna(),
        ["pair_id", "origin_city", "dest_city"],
    ].copy()
    if not missing.empty:
        ensure_city_snap_coverage(
            set(missing["origin_city"]) | set(missing["dest_city"]),
            city_snap_path,
            output_root,
            source_years,
        )
        pairs = attach_city_nodes(missing, city_snap_path)
        node_columns = [
            "node_pair_id",
            "origin_network_node_id",
            "origin_network_lon",
            "origin_network_lat",
            "dest_network_node_id",
            "dest_network_lon",
            "dest_network_lat",
        ]
        routes = route_missing_nodes(
            pairs[node_columns].drop_duplicates("node_pair_id"),
            artifact_dir / "transition_route_tasks",
            max_workers,
        )
        extension = pairs.merge(
            routes[["node_pair_id", "d_base"]],
            on="node_pair_id",
            how="left",
            validate="many_to_one",
        )[["pair_id", "d_base", "same_network_node"]]
        transition = transition.merge(
            extension.rename(
                columns={
                    "d_base": "extension_d_base",
                    "same_network_node": "extension_same_node",
                }
            ),
            on="pair_id",
            how="left",
            validate="one_to_one",
        )
        transition["d_base"] = transition["d_base"].fillna(
            transition["extension_d_base"]
        )
        transition["same_network_node"] = transition["same_network_node"].where(
            transition["same_network_node"].notna(),
            transition["extension_same_node"],
        )
        transition = transition.drop(
            columns=["extension_d_base", "extension_same_node"]
        )
    if transition["d_base"].isna().any():
        raise c0a.DataContractError(f"{year} transition yardstick is incomplete")
    transition["same_network_node"] = transition["same_network_node"].astype(bool)
    write_frame_atomic(target, transition)
    return target


def downstream_config(
    year: int,
    yardstick: Path,
    max_workers: int,
    candidate_enumeration_mode: str,
) -> dict[str, Any]:
    return {
        "candidate_enumeration_mode": candidate_enumeration_mode,
        f"yardstick_{year}": str(yardstick.relative_to(CODING_ROOT)),
        "sequence_separator": ">",
        "route_circuity_threshold": 2.0,
        "sensitivity_thresholds": [1.75, 2.5, 3.0],
        "duration_threshold_days": 30.0,
        "positive_route_speed_threshold_knots": 1.0,
        "exclude_zero_route_above_duration_threshold": True,
        "same_node_route_backstop_km": 2000.0,
        "worker_count": max_workers,
        "q_lower": 0.01,
        "q_upper": 0.99,
    }


def build_final_year(
    year: int,
    output_root: Path,
    yardstick: Path,
    transition: Path,
    c2_manifest: dict[str, Any],
    max_workers: int,
    candidate_enumeration_mode: str,
) -> dict[str, Any]:
    artifact_dir = output_root / f"intermediate/{year}/filters"
    config = downstream_config(
        year, yardstick, max_workers, candidate_enumeration_mode
    )
    c3_manifest = rebuild.run_c3(
        year, config, c2_manifest, transition, artifact_dir
    )
    pretrim = rebuild.prepare_c4_parts(year, config, artifact_dir)
    thresholds = qtrim.fit_global_thresholds(pretrim, 0.01, 0.99, year)
    thresholds["candidate_enumeration_mode"] = candidate_enumeration_mode
    write_json_atomic(
        output_root / f"manifests/qtrim_thresholds_{year}.json",
        thresholds,
    )
    exposure = None
    final_dir = output_root / f"adjacent_commercial_episode_voyages_{year}"
    sidecar_dir = output_root / f"sidecars/{year}"
    contract_hash = canonical_hash(
        {
            "year": year,
            "thresholds": thresholds,
            "yardstick_sha256": sha256_file(yardstick),
        }
    )
    records = []
    for source in pretrim:
        output = final_dir / source.name
        sidecar = sidecar_dir / f"{source.stem}.json"
        record = qtrim.process_part(
            source,
            output,
            sidecar,
            exposure,
            thresholds["normalized_time_ratio_q01"],
            thresholds["normalized_time_ratio_q99"],
            "unsupported",
            contract_hash,
        )
        frame = pd.read_feather(output)
        frame["global_qtrim_version"] = (
            f"pooled_{year}_q01_q99_no_grouping"
        )
        write_frame_atomic(output, frame)
        record["output_sha256"] = sha256_file(output)
        write_json_atomic(sidecar, record)
        records.append(record)
    summary = {
        key: int(sum(item[key] for item in records))
        for key in (
            "input_rows",
            "temporal_scope_excluded",
            "q01_trimmed",
            "q99_trimmed",
            "final_retained",
        )
    }
    manifest = {
        "status": "pass",
        "analysis_year": year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "c2_summary": c2_manifest["summary"],
        "c3_summary": c3_manifest["summary"],
        "qtrim_thresholds": thresholds,
        "summary": summary,
        "parts": records,
    }
    write_json_atomic(
        output_root / f"manifests/final_sample_manifest_{year}.json",
        manifest,
    )
    return manifest


def run_full_pipeline(
    years: tuple[int, ...],
    city_support_years: tuple[int, ...],
    output_root: Path,
    episode_manifests: dict[int, dict[str, Any]],
    max_workers: int,
    task_count: int,
    candidate_enumeration_mode: str,
) -> dict[str, Any]:
    city_support_years = validate_city_support_years(years, city_support_years)
    candidate_enumeration_mode = candidate_pairs.validate_candidate_enumeration_mode(
        candidate_enumeration_mode
    )
    city_snap = build_combined_city_snaps(
        city_support_years,
        output_root,
        episode_manifests,
    )
    results = {}
    for year in years:
        yardstick = build_year_yardstick(
            year,
            output_root,
            episode_manifests[year],
            city_snap,
            max_workers,
        )
        c2_manifest = materialize_c2(
            year, output_root, task_count, candidate_enumeration_mode
        )
        transition = build_transition_yardstick(
            year,
            output_root,
            c2_manifest,
            city_snap,
            yardstick,
            max_workers,
            city_support_years,
        )
        results[str(year)] = build_final_year(
            year,
            output_root,
            yardstick,
            transition,
            c2_manifest,
            max_workers,
            candidate_enumeration_mode,
        )
    results["delivery"] = build_delivery_products(
        years, output_root, results, candidate_enumeration_mode
    )
    results["analysis_years"] = list(years)
    results["city_support_years"] = list(city_support_years)
    results["candidate_enumeration_mode"] = candidate_enumeration_mode
    return results


def build_delivery_products(
    years: tuple[int, ...],
    output_root: Path,
    manifests: dict[str, Any],
    candidate_enumeration_mode: str,
) -> dict[str, Any]:
    tag = years_tag(years)
    year_frames = []
    monthly_frames = []
    year_paths = []
    for year in years:
        parts = sorted(
            (output_root / f"adjacent_commercial_episode_voyages_{year}").glob("part-*.feather")
        )
        if not parts:
            raise c0a.DataContractError(f"{year} final voyage parts are missing")
        frame = pd.concat([pd.read_feather(path) for path in parts], ignore_index=True)
        if frame["candidate_id"].duplicated().any():
            raise c0a.DataContractError(f"{year} final candidate IDs duplicate")
        if len(frame) != manifests[str(year)]["summary"]["final_retained"]:
            raise c0a.DataContractError(f"{year} final row count differs from manifest")
        frame["analysis_year"] = year
        year_path = output_root / f"adjacent_commercial_episode_voyages_{year}.feather"
        write_frame_atomic(year_path, frame)
        year_paths.append(year_path)
        year_frames.append(frame)
        supported = frame
        monthly = (
            supported.groupby(
                [
                    "analysis_year",
                    "departure_month",
                    "pair_id",
                    "shiptype_group",
                ],
                observed=True,
            )
            .agg(
                voyage_count=("candidate_id", "size"),
                vessel_count=("imo", "nunique"),
                median_voyage_days=("total_voyage_time_days", "median"),
                mean_voyage_days=("total_voyage_time_days", "mean"),
            )
            .reset_index()
        )
        monthly_frames.append(monthly)
    combined = pd.concat(year_frames, ignore_index=True)
    if combined["candidate_id"].duplicated().any():
        raise c0a.DataContractError("Candidate IDs duplicate across analysis years")
    combined_path = output_root / f"adjacent_commercial_episode_voyages_{tag}.feather"
    write_frame_atomic(combined_path, combined)
    write_frame_atomic(
        output_root / f"adjacent_commercial_episode_voyages_{tag}_sample_first_10000.csv",
        combined.head(10_000),
    )
    panel = pd.concat(monthly_frames, ignore_index=True)
    panel_path = (
        output_root
        / f"adjacent_commercial_episode_voyages_{tag}_monthly_group_panel.feather"
    )
    write_frame_atomic(panel_path, panel)
    write_frame_atomic(
        output_root / f"adjacent_commercial_episode_voyages_{tag}_monthly_group_panel.csv",
        panel,
    )
    summary = pd.DataFrame(
        [
            {
                "analysis_year": year,
                **manifests[str(year)]["c2_summary"],
                **{
                    f"c3_{key}": value
                    for key, value in manifests[str(year)]["c3_summary"].items()
                },
                **{
                    f"final_{key}": value
                    for key, value in manifests[str(year)]["summary"].items()
                },
            }
            for year in years
        ]
    )
    summary_path = output_root / f"adjacent_episode_voyage_summary_{tag}.csv"
    write_frame_atomic(summary_path, summary)
    outputs = [*year_paths, combined_path, panel_path, summary_path]
    payload = {
        "status": "pass",
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "analysis_years": list(years),
        "combined_final_rows": int(len(combined)),
        "monthly_panel_rows": int(len(panel)),
        "outputs": {
            str(path.relative_to(CODING_ROOT)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        },
    }
    write_json_atomic(output_root / "manifests/delivery_manifest.json", payload)
    build_validation_outputs(combined, output_root)
    return payload


def build_validation_outputs(
    combined: pd.DataFrame,
    output_root: Path,
) -> None:
    key_pairs = [
        ("BEL:ANTWERP__CHN:SHANGHAI", "container", "Antwerp-Shanghai"),
        ("CHN:NINGBO__NLD:ROTTERDAM", "container", "Ningbo-Rotterdam"),
        ("CHN:SHANGHAI__NLD:ROTTERDAM", "container", "Shanghai-Rotterdam"),
        ("MAR:TANGER MED__CHN:NINGBO", "container", "Tanger Med-Ningbo"),
        ("EGY:PORT SAID__CHN:NINGBO", "container", "Port Said-Ningbo"),
        (
            "EGY:PORT SAID__CHN:SHANGHAI",
            "bulk_carrier",
            "Port Said-Shanghai bulk",
        ),
        (
            "EGY:SUEZ PORT__SGP:SINGAPORE",
            "bulk_carrier",
            "Suez Port-Singapore bulk",
        ),
        (
            "SAU:JEDDAH__CHN:NINGBO",
            "container",
            "Jeddah-Ningbo",
        ),
    ]
    records = []
    for pair_id, shiptype, label in key_pairs:
        for year in sorted(combined["analysis_year"].unique()):
            selected = combined.loc[
                combined["analysis_year"].eq(year)
                & combined["pair_id"].eq(pair_id)
                & combined["shiptype_group"].eq(shiptype)
            ]
            records.append(
                {
                    "analysis_year": int(year),
                    "city_pair": label,
                    "pair_id": pair_id,
                    "shiptype_group": shiptype,
                    "voyage_count": int(len(selected)),
                    "vessel_count": int(selected["imo"].nunique()),
                    "median_voyage_days": (
                        float(selected["total_voyage_time_days"].median())
                        if len(selected)
                        else np.nan
                    ),
                    "p25_voyage_days": (
                        float(selected["total_voyage_time_days"].quantile(0.25))
                        if len(selected)
                        else np.nan
                    ),
                    "p75_voyage_days": (
                        float(selected["total_voyage_time_days"].quantile(0.75))
                        if len(selected)
                        else np.nan
                    ),
                }
            )
    write_frame_atomic(
        output_root / "validation/key_city_pair_voyage_times_2022_2024.csv",
        pd.DataFrame(records),
    )
    checks = []
    for year, frame in combined.groupby("analysis_year", sort=True):
        lower = frame["global_normalized_time_ratio_q01"].dropna().unique()
        upper = frame["global_normalized_time_ratio_q99"].dropna().unique()
        checks.append(
            {
                "analysis_year": int(year),
                "final_rows": int(len(frame)),
                "duplicate_candidate_ids": int(
                    frame["candidate_id"].duplicated().sum()
                ),
                "missing_direct_yardstick": int(frame["d_yardstick"].isna().sum()),
                "q01_value_count": int(len(lower)),
                "q99_value_count": int(len(upper)),
                "below_retained_q01": int(
                    frame["normalized_time_ratio"].lt(lower[0]).sum()
                    if len(lower)
                    else -1
                ),
                "above_retained_q99": int(
                    frame["normalized_time_ratio"].gt(upper[0]).sum()
                    if len(upper)
                    else -1
                ),
            }
        )
    checks_frame = pd.DataFrame(checks)
    failure_columns = [
        "duplicate_candidate_ids",
        "missing_direct_yardstick",
        "below_retained_q01",
        "above_retained_q99",
    ]
    if checks_frame[failure_columns].ne(0).any(axis=None):
        raise c0a.DataContractError("Final middle-year validation checks failed")
    write_frame_atomic(
        output_root / "validation/final_sample_checks_2022_2024.csv",
        checks_frame,
    )


def run_early_stages(
    years: tuple[int, ...],
    city_support_years: tuple[int, ...],
    output_root: Path,
    chunk_rows: int,
    max_workers: int,
    task_count: int,
    task_limit: int | None,
    stop_after: str,
    candidate_enumeration_mode: str = candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> dict[str, Any]:
    city_support_years = validate_city_support_years(years, city_support_years)
    candidate_enumeration_mode = (
        candidate_pairs.validate_candidate_enumeration_mode(
            candidate_enumeration_mode
        )
    )
    input_manifest = verify_or_write_input_manifest(
        years,
        city_support_years,
        output_root,
        candidate_enumeration_mode,
    )
    if stop_after == "preflight":
        return input_manifest
    results: dict[str, Any] = {
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "analysis_years": list(years),
        "city_support_years": list(city_support_years),
        "input_manifest": input_manifest,
        "years": {},
    }
    episode_manifests: dict[int, dict[str, Any]] = {}
    for year in city_support_years:
        episode_manifest = build_year_episodes(
            year,
            output_root,
            chunk_rows,
            candidate_enumeration_mode,
        )
        episode_manifests[year] = episode_manifest
        results["years"][str(year)] = {
            "role": "analysis" if year in years else "city_support_only",
            "episodes": episode_manifest,
        }
        if stop_after == "episodes":
            continue
        if year not in years:
            continue
        episodes, legs, call_meta = load_candidate_inputs(episode_manifest)
        payloads = run_candidate_tasks(
            year,
            output_root,
            episodes,
            legs,
            call_meta,
            max_workers,
            task_count,
            task_limit,
            episode_manifest_identity(episode_manifest),
            candidate_enumeration_mode,
        )
        candidate_result = validate_candidate_outputs(
            year,
            output_root,
            episode_manifest,
            payloads,
            task_count,
            candidate_enumeration_mode,
        )
        results["years"][str(year)]["candidates"] = candidate_result
        if candidate_result["status"] != "pass":
            continue
    if stop_after == "final":
        if any(
            results["years"][str(year)]["candidates"]["status"] != "pass"
            for year in years
        ):
            raise c0a.DataContractError("Candidate construction is incomplete")
        results["final"] = run_full_pipeline(
            years,
            city_support_years,
            output_root,
            episode_manifests,
            max_workers,
            task_count,
            candidate_enumeration_mode,
        )
    return results


def build_log_dir(log_root: Path, years: tuple[int, ...]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "-".join(str(year) for year in years)
    path = log_root / f"{STEP_NAME}_{timestamp}_years-{tag}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument(
        "--city-support-years",
        nargs="+",
        type=int,
        default=list(DEFAULT_CITY_SUPPORT_YEARS),
        help=(
            "Years used to freeze the shared city-to-network-node table. Defaults "
            "to all five project years and must contain every analysis year. "
            "Support-only years "
            "build city episodes but do not enumerate or retain voyages."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Isolated run root. Accepted products are adopted only after review.",
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--od-root",
        type=Path,
        default=DEFAULT_OD_ROOT,
        help=(
            "Root holding {year}/voyages_{year}MM.csv. Defaults to the accepted our_od "
            "product. Any other value also requires --output-root."
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--task-count", type=int, default=16)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument(
        "--stop-after",
        choices=["preflight", "episodes", "candidates", "final"],
        default="final",
    )
    return parser.parse_args()


def main() -> None:
    global OD_ROOT
    arguments = parse_args()

    # Bound before any pool starts, so the forked workers inherit it. Reading a
    # non-default OD while writing to the accepted output root would silently
    # relabel a differently-defined sample as the accepted one.
    OD_ROOT = arguments.od_root.resolve()
    if OD_ROOT != DEFAULT_OD_ROOT.resolve() and (
        arguments.output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve()
    ):
        raise SystemExit(
            f"refusing to build a non-default OD into the accepted output root "
            f"{DEFAULT_OUTPUT_ROOT}; pass --output-root to a separate location"
        )
    years = validate_years(arguments.years)
    city_support_years = validate_city_support_years(
        years,
        arguments.city_support_years,
    )
    if not 1 <= arguments.max_workers <= 16:
        raise ValueError("--max-workers must be between 1 and 16")
    if not 1 <= arguments.task_count <= 64:
        raise ValueError("--task-count must be between 1 and 64")
    output_root = arguments.output_root.resolve()
    log_root = arguments.log_root.resolve()
    log_dir = build_log_dir(log_root, years)
    configuration = {
        "analysis_years": list(years),
        "city_support_years": list(city_support_years),
        "chunk_rows": arguments.chunk_rows,
        "max_workers": arguments.max_workers,
        "task_count": arguments.task_count,
        "task_limit": arguments.task_limit,
        "candidate_enumeration_mode": candidate_pairs.MAIN_CANDIDATE_ENUMERATION_MODE,
        "stop_after": arguments.stop_after,
        "output_root": str(output_root.relative_to(CODING_ROOT)),
        "branch": git_value("branch", "--show-current"),
        "base_commit": git_value("rev-parse", "HEAD"),
    }
    write_json_atomic(log_dir / "run_config.json", configuration)
    try:
        result = run_early_stages(
            years,
            city_support_years,
            output_root,
            arguments.chunk_rows,
            arguments.max_workers,
            arguments.task_count,
            arguments.task_limit,
            arguments.stop_after,
            candidate_pairs.MAIN_CANDIDATE_ENUMERATION_MODE,
        )
    except Exception as error:
        write_json_atomic(
            log_dir / "run_summary.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    write_json_atomic(
        log_dir / "run_summary.json",
        {"status": "pass", "result": result},
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
