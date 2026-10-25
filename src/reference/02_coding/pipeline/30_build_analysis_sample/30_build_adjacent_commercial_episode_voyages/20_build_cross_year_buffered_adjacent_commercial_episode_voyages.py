#!/usr/bin/env python3
"""Build the guarded E030 five-year adjacent-commercial-episode cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import pandas as pd


CODING_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parent
builder_path = PACKAGE_ROOT / "10_build_adjacent_commercial_episode_voyages.py"
builder_spec = importlib.util.spec_from_file_location("adjacent_episode_builder", builder_path)
if builder_spec is None or builder_spec.loader is None:
    raise ImportError(f"Cannot load adjacent-episode builder: {builder_path}")
p0 = importlib.util.module_from_spec(builder_spec)
builder_spec.loader.exec_module(p0)


PIPELINE = "build_adjacent_commercial_episode_voyages"
STEP = "build_cross_year_buffered_adjacent_commercial_episode_voyages"
SPEC_TAG = "next_calendar_year_v4_common_one_year_followup"
SOURCE_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages/main"
)
DEFAULT_OUTPUT_ROOT = CODING_ROOT / "data" / "processed" / PIPELINE / STEP / SPEC_TAG
DEFAULT_LOG_ROOT = CODING_ROOT / ".runs" / PIPELINE / STEP / SPEC_TAG
DEFAULT_YEARS = (2021, 2022, 2023, 2024, 2025)
BUFFERED_YEARS = (2021, 2022, 2023, 2024)
SUPPORTED_YEARS = frozenset(DEFAULT_YEARS)
MAPPING_PATH = (
    CODING_ROOT
    / "data/legacy_inputs/anchorage_to_analytical_city_candidate_v3.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument(
        "--od-root",
        type=Path,
        required=True,
        help="Guarded root produced by 05_assemble_stage_30_20_od_input.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Isolated run root. Accepted products are adopted only after review.",
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--task-count", type=int, default=16)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument(
        "--stop-after",
        choices=("episodes", "candidates", "final"),
        default="final",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else CODING_ROOT / path


def resolve_cli_output(path: Path) -> Path:
    """Resolve operator-chosen output/log paths from the launch directory."""

    return path.expanduser().resolve()


def source_episode_manifest(source_root: Path, year: int) -> Path:
    return (
        source_root
        / f"intermediate/{year}/city_episodes/output_manifest.json"
    )


def verified_source_event_record(
    source_root: Path,
    year: int,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = source_episode_manifest(source_root, year)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing source episode manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise ValueError(f"source episode manifest is not complete: {manifest_path}")
    record = manifest["outputs"]["endpoint_events"]
    path = p0.resolve_manifest_record_path(record)
    if not path.is_file():
        migrated_sibling = manifest_path.parent / Path(record["path"]).name
        if migrated_sibling.is_file():
            path = migrated_sibling
    if (
        not path.is_file()
        or int(path.stat().st_size) != int(record["size_bytes"])
        or p0.sha256_file(path) != record["sha256"]
    ):
        raise ValueError(f"source endpoint events fail manifest validation: {path}")
    identity = {
        "year": year,
        "manifest_path": str(manifest_path.relative_to(CODING_ROOT)),
        "manifest_sha256": p0.sha256_file(manifest_path),
        "endpoint_events": {
            "recorded_path": record["path"],
            "path": str(path.relative_to(CODING_ROOT)),
            "size_bytes": int(path.stat().st_size),
            "sha256": record["sha256"],
        },
    }
    return path, identity


def verified_source_events(
    source_root: Path,
    year: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, identity = verified_source_event_record(source_root, year)
    return pd.read_feather(path), identity


def namespace_visit_ids(frame: pd.DataFrame, source_year: int) -> pd.DataFrame:
    """Make annual visit IDs unique while preserving an existing year prefix."""

    result = frame.copy()
    values = result["visit_id"].astype("string")
    nonblank = values.notna() & values.str.strip().ne("")
    already_namespaced = values.str.match(r"^\d{4}:", na=False)
    needs_namespace = nonblank & ~already_namespaced
    result.loc[needs_namespace, "visit_id"] = (
        str(source_year) + ":" + values.loc[needs_namespace].astype(str)
    )
    return result


def apply_common_followup_window(frame: pd.DataFrame) -> pd.DataFrame:
    """Give each departure no more than one calendar year of follow-up."""

    result = frame.copy()
    departure = pd.to_datetime(result["departure_time"], utc=True)
    arrival = pd.to_datetime(result["arrival_time"], utc=True)
    deadline = departure + pd.DateOffset(years=1)
    result["e017_followup_deadline"] = deadline
    result["e017_common_followup_eligible"] = arrival.le(deadline)
    result["temporal_scope_eligible"] = (
        result["temporal_scope_eligible"].astype(bool)
        & result["e017_common_followup_eligible"]
    )
    return result


def code_identity() -> dict[str, dict[str, Any]]:
    paths = {
        "driver": Path(__file__).resolve(),
        "formal_driver": (
            CODING_ROOT
            / "pipeline/30_build_analysis_sample/"
            "20_build_adjacent_commercial_episode_voyages/"
            "10_build_adjacent_commercial_episode_voyages.py"
        ),
        "city_episode_core": (
            CODING_ROOT
            / "pipeline/30_build_analysis_sample/20_build_adjacent_commercial_episode_voyages/core/city_episodes.py"
        ),
        "endpoint_city_episode_core": (
            CODING_ROOT
            / "pipeline/30_build_analysis_sample/20_build_adjacent_commercial_episode_voyages/core/"
            "endpoint_city_episodes.py"
        ),
        "candidate_core": (
            CODING_ROOT
            / "pipeline/30_build_analysis_sample/20_build_adjacent_commercial_episode_voyages/core/candidate_core.py"
        ),
        "candidate_pairs": (
            CODING_ROOT
            / "pipeline/30_build_analysis_sample/20_build_adjacent_commercial_episode_voyages/core/candidate_pairs.py"
        ),
        "od_input_assembly": (
            CODING_ROOT
            / "pipeline/30_build_analysis_sample/20_build_adjacent_commercial_episode_voyages/core/"
            "od_input_assembly.py"
        ),
    }
    return {
        name: {
            "path": str(path.relative_to(CODING_ROOT)),
            "size_bytes": int(path.stat().st_size),
            "sha256": p0.sha256_file(path),
        }
        for name, path in paths.items()
    }


def source_identity(
    *,
    analysis_year: int,
    source_records: list[dict[str, Any]],
    candidate_enumeration_mode: str = p0.candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> tuple[dict[str, Any], str]:
    vessel_specs = [
        p0.vessel_spec_path(year) for year in (analysis_year, analysis_year + 1)
    ]
    payload = {
        "specification": SPEC_TAG,
        "analysis_year": analysis_year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "source_years": [analysis_year, analysis_year + 1],
        "endpoint_event_sources": source_records,
        "vessel_specs": [
            {
                "path": str(path.relative_to(CODING_ROOT)),
                "size_bytes": int(path.stat().st_size),
                "sha256": p0.sha256_file(path),
            }
            for path in vessel_specs
        ],
        "mapping": {
            "path": str(MAPPING_PATH.relative_to(CODING_ROOT)),
            "size_bytes": int(MAPPING_PATH.stat().st_size),
            "sha256": p0.sha256_file(MAPPING_PATH),
        },
        "code": code_identity(),
    }
    return payload, p0.canonical_hash(payload)


def global_run_identity(
    *,
    years: tuple[int, ...],
    source_root: Path,
    candidate_enumeration_mode: str = p0.candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> tuple[dict[str, Any], str]:
    source_years = sorted({year for value in years for year in (value, value + 1)})
    source_records = [
        verified_source_event_record(source_root, year)[1]
        for year in source_years
    ]
    vessel_years = source_years
    payload = {
        "specification": SPEC_TAG,
        "analysis_years": list(years),
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "endpoint_event_sources": source_records,
        "vessel_specs": [
            {
                "path": str(p0.vessel_spec_path(year).relative_to(CODING_ROOT)),
                "size_bytes": int(p0.vessel_spec_path(year).stat().st_size),
                "sha256": p0.sha256_file(p0.vessel_spec_path(year)),
            }
            for year in vessel_years
        ],
        "mapping_sha256": p0.sha256_file(MAPPING_PATH),
        "code": code_identity(),
    }
    return payload, p0.canonical_hash(payload)


def validate_stable_vessel_keys(analysis_year: int) -> None:
    columns = ["registry_imo_clean", "mmsi_clean", "vessel_uid_clean"]
    left = pd.read_csv(
        p0.vessel_spec_path(analysis_year),
        usecols=columns,
        dtype=str,
    )
    right = pd.read_csv(
        p0.vessel_spec_path(analysis_year + 1),
        usecols=columns,
        dtype=str,
    )
    comparison = left.merge(
        right,
        on="registry_imo_clean",
        suffixes=("_analysis", "_buffer"),
        validate="one_to_one",
    )
    expected_vessels = p0.analysis_vessel_count()
    if len(comparison) != expected_vessels:
        raise ValueError(
            "cross-year vessel specifications do not cover the accepted "
            f"{expected_vessels:,}-vessel universe"
        )
    for column in ("mmsi_clean", "vessel_uid_clean"):
        if comparison[f"{column}_analysis"].ne(
            comparison[f"{column}_buffer"]
        ).any():
            raise ValueError(
                f"{column} changes across {analysis_year} and "
                f"{analysis_year + 1}; explicit identity remapping is required"
            )


def select_latest_trip_snapshots(
    frames: list[pd.DataFrame],
    source_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tagged = [
        namespace_visit_ids(frame, year).assign(_source_year=year)
        for frame, year in zip(frames, source_years, strict=True)
    ]
    combined = pd.concat(tagged, ignore_index=True)
    arrivals = combined.loc[
        combined["event_type"].eq("arrival"),
        ["trip_id", "_source_year", "event_time"],
    ]
    if arrivals.duplicated(["trip_id", "_source_year"]).any():
        raise ValueError("a source year contains duplicate trip arrivals")
    choices = (
        arrivals.sort_values(
            ["trip_id", "event_time", "_source_year"],
            kind="stable",
        )
        .drop_duplicates("trip_id", keep="last")
        .rename(
            columns={
                "_source_year": "selected_source_year",
                "event_time": "selected_arrival_time",
            }
        )
    )
    versions = (
        arrivals.groupby("trip_id", observed=True)
        .agg(
            n_source_versions=("_source_year", "nunique"),
            earliest_arrival_time=("event_time", "min"),
            latest_arrival_time=("event_time", "max"),
        )
        .reset_index()
    )
    audit = choices.merge(
        versions,
        on="trip_id",
        how="left",
        validate="one_to_one",
    )
    audit = audit.loc[audit["n_source_versions"].gt(1)].reset_index(drop=True)
    selected = combined.merge(
        choices[["trip_id", "selected_source_year"]],
        left_on=["trip_id", "_source_year"],
        right_on=["trip_id", "selected_source_year"],
        how="inner",
        validate="many_to_one",
    ).drop(columns=["_source_year", "selected_source_year"])
    if selected.duplicated(["trip_id", "event_type"]).any():
        raise ValueError("selected cross-year events duplicate a trip boundary")
    event_counts = selected.groupby("trip_id", observed=True)["event_type"].nunique()
    if not event_counts.eq(2).all():
        raise ValueError("selected cross-year trip lacks departure or arrival")
    selected = selected.sort_values(
        ["mmsi", "event_time", "event_order", "event_id"],
        kind="stable",
    ).reset_index(drop=True)
    selected["event_sequence"] = selected.groupby(
        "mmsi", sort=False
    ).cumcount()
    return selected, audit


def events_to_trips(events: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the OD trip contract before cross-year identity cleaning."""

    common = [
        "mmsi",
        "vessel_id",
        "trip_id",
        "source_file",
        "source_row",
    ]
    departures = events.loc[
        events["event_type"].eq("departure"),
        [
            *common,
            "event_time",
            "anchorage_id",
            "visit_id",
        ],
    ].rename(
        columns={
            "event_time": "trip_start",
            "anchorage_id": "trip_start_anchorage_s2id",
            "visit_id": "trip_start_visit_id",
        }
    )
    arrivals = events.loc[
        events["event_type"].eq("arrival"),
        [
            *common,
            "event_time",
            "anchorage_id",
            "visit_id",
        ],
    ].rename(
        columns={
            "event_time": "trip_end",
            "anchorage_id": "trip_end_anchorage_s2id",
            "visit_id": "trip_end_visit_id",
        }
    )
    trips = departures.merge(
        arrivals,
        on=["mmsi", "vessel_id", "trip_id"],
        how="inner",
        suffixes=("_departure", "_arrival"),
        validate="one_to_one",
    )
    if len(trips) * 2 != len(events):
        raise ValueError("cross-year events do not reconstruct one OD trip each")
    for column in ("source_file", "source_row"):
        left = f"{column}_departure"
        right = f"{column}_arrival"
        if trips[left].ne(trips[right]).any():
            raise ValueError(
                f"cross-year trip boundaries disagree on {column}"
            )
        trips[column] = trips[left]
    trips = trips.drop(
        columns=[
            "source_file_departure",
            "source_file_arrival",
            "source_row_departure",
            "source_row_arrival",
        ]
    )
    return trips[p0.OD_COLUMNS + ["source_file", "source_row"]]


def episode_manifest_is_current(
    manifest_path: Path,
    *,
    identity_digest: str,
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if (
        manifest.get("status") != "pass"
        or manifest.get("source_identity_sha256") != identity_digest
    ):
        return False
    for record in manifest.get("outputs", {}).values():
        path = CODING_ROOT / record["path"]
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(record["size_bytes"])
            or p0.sha256_file(path) != record["sha256"]
        ):
            return False
    return True


def build_buffered_year_episodes(
    *,
    analysis_year: int,
    source_root: Path,
    output_root: Path,
    candidate_enumeration_mode: str = p0.candidate_pairs.ADJACENT_COMMERCIAL_EPISODES,
) -> dict[str, Any]:
    output_dir = output_root / f"intermediate/{analysis_year}/city_episodes"
    manifest_path = output_dir / "output_manifest.json"
    validate_stable_vessel_keys(analysis_year)
    event_frames: list[pd.DataFrame] = []
    source_records: list[dict[str, Any]] = []
    for source_year in (analysis_year, analysis_year + 1):
        events, record = verified_source_events(source_root, source_year)
        event_frames.append(events)
        source_records.append(record)
    identity_payload, identity_digest = source_identity(
        analysis_year=analysis_year,
        source_records=source_records,
        candidate_enumeration_mode=candidate_enumeration_mode,
    )
    if episode_manifest_is_current(
        manifest_path,
        identity_digest=identity_digest,
    ):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    selected_events, duplicate_audit = select_latest_trip_snapshots(
        event_frames,
        [analysis_year, analysis_year + 1],
    )
    mapping = p0.c0a.load_v3_mapping(MAPPING_PATH)
    vessels = p0.c0a.load_vessel_universe(
        p0.vessel_spec_path(analysis_year),
        p0.analysis_vessel_count(),
    )
    trips = events_to_trips(selected_events)
    trips_before_cleaning = len(trips)
    trips, incoherent_excluded, incoherent_diagnostics = (
        p0.c5_city.remove_incoherent_source_tracks(trips)
    )
    (
        trips,
        identity_excluded,
        identity_decisions,
        identity_diagnostics,
    ) = p0.c0a.resolve_source_identity_overlaps(
        trips,
        mapping.set_index("anchorage_id")["country_iso3"],
        analysis_year,
        "coherent_source_track_v1",
    )
    identity_decisions = identity_decisions.merge(
        vessels[["mmsi", "imo", "vessel_name"]],
        on="mmsi",
        how="left",
        validate="many_to_one",
    )
    identity_decisions = identity_decisions.rename(
        columns={
            "active_endpoint_months_2021": (
                f"active_endpoint_months_{analysis_year}"
            ),
            "coverage_hours_2021": f"coverage_hours_{analysis_year}",
        }
    )
    events = p0.c0a.build_endpoint_events(trips, vessels)
    calls, missing_mapping, call_diagnostics = p0.c0a.build_calls(
        events,
        mapping,
    )
    if not missing_mapping.empty:
        raise ValueError(
            "cross-year calls contain unmapped anchorages: "
            f"{missing_mapping['anchorage_id'].drop_duplicates().head(10).tolist()}"
        )
    trips, overlap_excluded, overlap_diagnostics = (
        p0.c5_city.remove_mmsi_with_overlapping_calls(trips, calls)
    )
    if overlap_diagnostics["unsupported_overlapping_call_mmsi"]:
        events = p0.c0a.build_endpoint_events(trips, vessels)
        calls, missing_mapping, call_diagnostics = p0.c0a.build_calls(
            events,
            mapping,
        )
        if not missing_mapping.empty:
            raise ValueError(
                "cross-year mapping gap remains after call-overlap exclusion"
            )
    if calls["overlaps_previous_call"].any():
        raise ValueError(
            "cross-year calls still overlap after formal cleaning sequence"
        )
    excluded = pd.concat(
        [incoherent_excluded, identity_excluded, overlap_excluded],
        ignore_index=True,
    )
    calls, episodes, episode_diagnostics = p0.c0a.merge_same_city_episodes(
        calls
    )
    calls["episode_id"] = calls["episode_id"].str.replace(
        "_2021_",
        f"_{analysis_year}_",
        regex=False,
    )
    episodes["episode_id"] = episodes["episode_id"].str.replace(
        "_2021_",
        f"_{analysis_year}_",
        regex=False,
    )
    pairs, pair_diagnostics = p0.c0a.enumerate_adjacent_episode_pairs(
        episodes,
        candidate_enumeration_mode,
    )
    pairs = pairs.rename(
        columns={
            "candidate_count_2021": f"candidate_count_{analysis_year}",
            "vessel_count_2021": f"vessel_count_{analysis_year}",
        }
    )
    diagnostics = {
        "status": "pass",
        "specification": SPEC_TAG,
        "analysis_year": analysis_year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "buffer_year": analysis_year + 1,
        "input_event_rows": int(sum(len(frame) for frame in event_frames)),
        "selected_snapshot_event_rows": int(len(selected_events)),
        "trips_before_formal_cleaning": int(trips_before_cleaning),
        "retained_event_rows": int(len(events)),
        "selected_trips": int(len(trips)),
        "duplicate_trip_snapshots_resolved": int(
            len(duplicate_audit)
        ),
        "event_time_min": str(events["event_time"].min()),
        "event_time_max": str(events["event_time"].max()),
        **incoherent_diagnostics,
        **identity_diagnostics,
        **overlap_diagnostics,
        **call_diagnostics,
        **episode_diagnostics,
        **pair_diagnostics,
    }
    paths = {
        "endpoint_events": output_dir / "endpoint_events.feather",
        "calls": output_dir / "ordered_port_calls.feather",
        "episodes": output_dir / "ordered_city_episodes.feather",
        "pair_manifest": output_dir / "adjacent_episode_pair_manifest.csv",
        "duplicate_trip_resolution": (
            output_dir / "duplicate_trip_resolution.csv"
        ),
        "excluded_trips": output_dir / "excluded_trip_records.csv",
        "source_identity_decisions": (
            output_dir / "source_identity_resolution.csv"
        ),
        "summary": output_dir / "summary.json",
    }
    p0.write_frame_atomic(paths["endpoint_events"], events)
    p0.write_frame_atomic(paths["calls"], calls)
    p0.write_frame_atomic(paths["episodes"], episodes)
    p0.write_frame_atomic(paths["pair_manifest"], pairs)
    p0.write_frame_atomic(
        paths["duplicate_trip_resolution"],
        duplicate_audit,
    )
    p0.write_frame_atomic(paths["excluded_trips"], excluded)
    p0.write_frame_atomic(
        paths["source_identity_decisions"],
        identity_decisions,
    )
    p0.write_json_atomic(paths["summary"], diagnostics)
    outputs = {
        name: {
            "path": str(path.relative_to(CODING_ROOT)),
            "size_bytes": int(path.stat().st_size),
            "sha256": p0.sha256_file(path),
        }
        for name, path in paths.items()
    }
    manifest = {
        "status": "pass",
        "analysis_year": analysis_year,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "source_identity": identity_payload,
        "source_identity_sha256": identity_digest,
        "diagnostics": diagnostics,
        "outputs": outputs,
    }
    p0.write_json_atomic(manifest_path, manifest)
    return manifest


def create_link(link: Path, source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"missing reused consumer input: {source}")
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise ValueError(f"consumer link points to the wrong source: {link}")
        return
    if link.exists():
        raise FileExistsError(f"consumer path already exists and is not a link: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{link.name}.",
        suffix=".link",
        dir=link.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        temporary.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        temporary.replace(link)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def assemble_consumer_inputs(
    *,
    output_root: Path,
    source_root: Path,
    rebuilt_years: tuple[int, ...],
) -> dict[str, Any]:
    del source_root
    records: list[dict[str, Any]] = []
    layout = output_root / "cross_year_candidate_layout"
    for year in rebuilt_years:
        source = output_root / f"adjacent_commercial_episode_voyages_{year}"
        target = layout / str(year)
        create_link(target, source)
        record = {
            "analysis_year": year,
            "status": "cross_year_buffered_candidate",
            "consumer_path": str(target.relative_to(CODING_ROOT)),
            "source_path": str(source.relative_to(CODING_ROOT)),
        }
        records.append(record)
    payload = {
        "status": "complete",
        "specification": SPEC_TAG,
        "candidate_enumeration_mode": (
            p0.candidate_pairs.MAIN_CANDIDATE_ENUMERATION_MODE
        ),
        "rebuilt_years": list(rebuilt_years),
        "layout_root": str(layout.relative_to(CODING_ROOT)),
        "inputs": records,
        "interpretation": (
            "Each 2022--2024 departure cohort uses its own endpoint events plus "
            "the next calendar year, with source-year visit-ID namespacing and "
            "a common one-calendar-year follow-up before pooled q01/q99."
        ),
    }
    p0.write_json_atomic(output_root / "consumer_input_manifest.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_enumeration_mode = p0.candidate_pairs.MAIN_CANDIDATE_ENUMERATION_MODE
    years = tuple(sorted(set(args.years)))
    invalid = sorted(set(years) - SUPPORTED_YEARS)
    if years != DEFAULT_YEARS or invalid:
        raise ValueError(
            "the E030 stage-30/20 entry requires exactly "
            f"{list(DEFAULT_YEARS)}; got {list(years)}"
        )
    if not 1 <= args.max_workers <= 16:
        raise ValueError("--max-workers must be between 1 and 16")
    if not 1 <= args.task_count <= 64:
        raise ValueError("--task-count must be between 1 and 64")
    od_root = resolve_cli_output(args.od_root)
    assembly = p0.od_input_assembly.verify_od_input_root(od_root)
    if int(assembly["total_bridge_voyages"]) != 80_489:
        raise ValueError(
            "guarded OD assembly bridge total changed: "
            f"{assembly['total_bridge_voyages']} != 80489"
        )
    p0.OD_ROOT = od_root
    output_base = resolve_cli_output(args.output_root)
    if output_base == od_root or od_root in output_base.parents:
        raise ValueError("stage output must not overwrite or nest inside its OD input root")
    accepted_products = (CODING_ROOT / "data/processed/products").resolve()
    if output_base == accepted_products or accepted_products in output_base.parents:
        raise ValueError("stage output must not write inside accepted products")

    original_prepare_c4_parts = p0.rebuild.prepare_c4_parts

    def prepare_common_followup_c4_parts(
        year: int,
        config: dict[str, Any],
        artifact_dir: Path,
    ) -> list[Path]:
        paths = original_prepare_c4_parts(year, config, artifact_dir)
        for path in paths:
            frame = pd.read_feather(path)
            frame = apply_common_followup_window(frame)
            p0.write_frame_atomic(path, frame)
        return paths

    p0.rebuild.prepare_c4_parts = prepare_common_followup_c4_parts
    output_root = output_base
    output_root.mkdir(parents=True, exist_ok=True)
    input_manifest = p0.verify_or_write_input_manifest(
        years,
        years,
        output_root,
        candidate_enumeration_mode,
    )
    source_root = output_root / "annual_sources"
    source_manifests: dict[int, dict[str, Any]] = {}
    for year in years:
        source_manifests[year] = p0.build_year_episodes(
            year,
            source_root,
            args.chunk_rows,
            candidate_enumeration_mode,
        )
    run_identity, run_identity_digest = global_run_identity(
        years=BUFFERED_YEARS,
        source_root=source_root,
        candidate_enumeration_mode=candidate_enumeration_mode,
    )

    episode_manifests: dict[int, dict[str, Any]] = {}
    results: dict[str, Any] = {
        "status": "partial",
        "specification": SPEC_TAG,
        "candidate_enumeration_mode": candidate_enumeration_mode,
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_digest,
        "output_root": str(output_root.relative_to(CODING_ROOT)),
        "od_assembly": {
            "path": str((od_root / "assembly_manifest.json").relative_to(CODING_ROOT)),
            "sha256": p0.sha256_file(od_root / "assembly_manifest.json"),
            "total_source_rows": int(assembly["total_source_rows"]),
            "total_bridge_voyages": int(assembly["total_bridge_voyages"]),
        },
        "input_manifest": input_manifest,
        "annual_source_episode_manifests": {
            str(year): manifest for year, manifest in source_manifests.items()
        },
        "years": {},
    }
    for year in years:
        if year in BUFFERED_YEARS:
            manifest = build_buffered_year_episodes(
                analysis_year=year,
                source_root=source_root,
                output_root=output_root,
                candidate_enumeration_mode=candidate_enumeration_mode,
            )
        else:
            manifest = source_manifests[year]
        episode_manifests[year] = manifest
        results["years"][str(year)] = {"episodes": manifest}
        if args.stop_after == "episodes":
            continue
        episodes, legs, call_meta = p0.load_candidate_inputs(manifest)
        payloads = p0.run_candidate_tasks(
            year,
            output_root,
            episodes,
            legs,
            call_meta,
            args.max_workers,
            args.task_count,
            args.task_limit,
            p0.episode_manifest_identity(manifest),
            candidate_enumeration_mode,
        )
        candidates = p0.validate_candidate_outputs(
            year,
            output_root,
            manifest,
            payloads,
            args.task_count,
            candidate_enumeration_mode,
        )
        results["years"][str(year)]["candidates"] = candidates
    if args.stop_after == "final":
        if any(
            results["years"][str(year)]["candidates"]["status"] != "pass"
            for year in years
        ):
            raise ValueError("buffered candidate construction is incomplete")
        results["final"] = p0.run_full_pipeline(
            years,
            years,
            output_root,
            episode_manifests,
            args.max_workers,
            args.task_count,
            candidate_enumeration_mode,
        )
        results["status"] = "complete"
        results["execution_boundary"] = {
            "candidate_root_run": False,
            "ordered_suez_run": False,
            "strict_run": False,
            "reachability_run": False,
            "weather_run": False,
            "adoption_run": False,
        }
    elif args.stop_after == "candidates":
        results["status"] = (
            "complete"
            if all(
                results["years"][str(year)]["candidates"]["status"] == "pass"
                for year in years
            )
            else "partial"
        )
    return results


def main() -> None:
    args = parse_args()

    if resolve_cli_output(args.output_root) == DEFAULT_OUTPUT_ROOT.resolve():
        raise SystemExit(
            f"refusing to build E030 into the accepted output root "
            f"{DEFAULT_OUTPUT_ROOT}; pass --output-root to a separate location"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    years = tuple(sorted(set(args.years)))
    log_root = resolve_cli_output(args.log_root)
    log_dir = log_root / f"{timestamp}_years-{'-'.join(map(str, years))}"
    log_dir.mkdir(parents=True, exist_ok=False)
    p0.write_json_atomic(
        log_dir / "run_config.json",
        {
            "specification": SPEC_TAG,
            "years": list(years),
            "od_root": str(resolve_cli_output(args.od_root)),
            "output_root": str(resolve_cli_output(args.output_root)),
            "max_workers": args.max_workers,
            "task_count": args.task_count,
            "task_limit": args.task_limit,
            "chunk_rows": args.chunk_rows,
            "candidate_enumeration_mode": (
                p0.candidate_pairs.MAIN_CANDIDATE_ENUMERATION_MODE
            ),
            "stop_after": args.stop_after,
        },
    )
    try:
        result = run(args)
    except Exception as error:
        p0.write_json_atomic(
            log_dir / "run_summary.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    p0.write_json_atomic(log_dir / "run_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
