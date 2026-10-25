"""Build 2025 city episodes with the frozen C0A rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import city_episodes as c0a
from .endpoint_preflight import (
    C5_ROOT,
    normalize_token,
    resolve_od_schema,
)

CODING_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = C5_ROOT / "config/frozen/endpoint_validation_2025.json"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "main/provenance/endpoint_city_episodes_2025"
CANONICAL_OD_COLUMNS = [
    "mmsi",
    "vessel_id",
    "trip_id",
    "trip_start",
    "trip_end",
    "trip_start_anchorage_s2id",
    "trip_end_anchorage_s2id",
    "trip_start_visit_id",
    "trip_end_visit_id",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if suffix == ".feather":
            frame.to_feather(temporary, compression="zstd", compression_level=9)
        elif suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError(f"Unsupported output suffix: {suffix}")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def verify_frozen_manifest(config: dict[str, Any]) -> pd.DataFrame:
    path = CODING_ROOT / config["frozen_input_manifest"]
    observed = sha256_file(path)
    expected = config["frozen_input_manifest_sha256"]
    if observed != expected:
        raise ValueError(
            f"C5.0 input manifest hash mismatch: expected {expected}; got {observed}"
        )
    manifest = pd.read_csv(path)
    if manifest["content_hash_skipped"].astype(str).str.lower().ne("false").any():
        raise ValueError("C5.1 requires a content-hashed C5.0 input manifest")
    for row in manifest.itertuples(index=False):
        source = CODING_ROOT / row.path
        if not source.exists() or source.stat().st_size != int(row.size_bytes):
            raise ValueError(f"Frozen input missing or size changed: {row.path}")
    return manifest


def resolve_evolving_trip_snapshots(
    trips: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Keep the latest completed snapshot when one trip ID evolves over time."""

    core = [column for column in CANONICAL_OD_COLUMNS if column != "trip_id"]
    duplicate_trip_rows = int(trips["trip_id"].duplicated(False).sum())
    variants = trips.drop_duplicates(["trip_id", *core])
    variant_counts = variants.groupby("trip_id", sort=False).size()
    conflict_ids = set(variant_counts.loc[variant_counts.gt(1)].index)
    if conflict_ids:
        conflicts = trips.loc[trips["trip_id"].isin(conflict_ids)].copy()
        stable_columns = [
            "mmsi",
            "vessel_id",
            "trip_start",
            "trip_start_anchorage_s2id",
            "trip_start_visit_id",
        ]
        unstable = conflicts.groupby("trip_id", sort=False)[stable_columns].nunique()
        if unstable.gt(1).any(axis=None):
            bad = unstable.loc[unstable.gt(1).any(axis=1)].index[:10].tolist()
            raise c0a.DataContractError(
                f"Conflicting trip snapshots change start identity: {bad}"
            )
        conflicts["_parsed_end"] = c0a.parse_utc_timestamp(conflicts["trip_end"])
        if conflicts["_parsed_end"].isna().any():
            raise c0a.DataContractError(
                "Conflicting trip snapshots contain invalid end timestamps"
            )
        maximum_end = conflicts.groupby("trip_id", sort=False)[
            "_parsed_end"
        ].transform("max")
        finalists = conflicts.loc[conflicts["_parsed_end"].eq(maximum_end)]
        final_variants = finalists.drop_duplicates(["trip_id", *core])
        final_counts = final_variants.groupby("trip_id", sort=False).size()
        if final_counts.gt(1).any():
            bad = final_counts.loc[final_counts.gt(1)].index[:10].tolist()
            raise c0a.DataContractError(
                f"Conflicting trip snapshots have tied final versions: {bad}"
            )

    working = trips.copy()
    working["_parsed_end"] = c0a.parse_utc_timestamp(working["trip_end"])
    selected = (
        working.sort_values(
            ["trip_id", "_parsed_end", "source_file", "source_row"],
            kind="stable",
        )
        .drop_duplicates("trip_id", keep="last")
    )
    evolving_excluded = working.loc[
        working["trip_id"].isin(conflict_ids)
        & ~working.index.isin(selected.index),
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
    evolving_excluded["exclusion_reason"] = "superseded_evolving_trip_snapshot"
    selected = selected.drop(columns="_parsed_end").reset_index(drop=True)
    diagnostics = {
        "duplicate_trip_rows": duplicate_trip_rows,
        "evolving_snapshot_trip_ids": len(conflict_ids),
        "evolving_snapshot_rows_excluded": len(evolving_excluded),
    }
    return selected, evolving_excluded, diagnostics


def source_sentinel_reasons_2025(trips: pd.DataFrame) -> pd.Series:
    """Extend the frozen sentinel rule to explicit blank 2025 timestamps."""

    reasons = c0a.source_sentinel_reasons(trips)
    blank_start = trips["trip_start"].eq("")
    blank_end = trips["trip_end"].eq("")
    blank_timestamp = blank_start | blank_end
    existing_reason = blank_timestamp & reasons.ne("")
    no_existing_reason = blank_timestamp & reasons.eq("")
    reasons.loc[no_existing_reason] = "source_blank_timestamp"
    reasons.loc[existing_reason] = (
        reasons.loc[existing_reason]
        + "_and_source_blank_timestamp"
    )
    return reasons


def remove_incoherent_source_tracks(
    trips: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Keep coherent tracks and mark vessels without one as unsupported.

    A bad source track is removed when the MMSI has a coherent alternative.
    If no coherent alternative exists, all trips for that MMSI are excluded
    from C5 rather than resolving the overlap by guessing which trip is valid.
    """

    rows: list[dict[str, Any]] = []
    for (mmsi, vessel_id), group in trips.groupby(
        ["mmsi", "vessel_id"], sort=False
    ):
        rows.append(
            {
                "mmsi": mmsi,
                "vessel_id": vessel_id,
                "overlap_rows": c0a.count_internal_trip_overlaps(group),
            }
        )
    audit = pd.DataFrame(rows)
    bad = audit.loc[audit["overlap_rows"].gt(0)].copy()
    if bad.empty:
        return trips, pd.DataFrame(), {
            "incoherent_source_tracks_excluded": 0,
            "incoherent_source_track_trips_excluded": 0,
            "unsupported_internal_overlap_mmsi": 0,
            "unsupported_internal_overlap_trips_excluded": 0,
        }
    coherent_mmsi = set(audit.loc[audit["overlap_rows"].eq(0), "mmsi"])
    unsupported_mmsi = set(bad.loc[~bad["mmsi"].isin(coherent_mmsi), "mmsi"])
    replaceable_bad = bad.loc[bad["mmsi"].isin(coherent_mmsi)]
    bad_keys = set(
        zip(
            replaceable_bad["mmsi"],
            replaceable_bad["vessel_id"],
            strict=True,
        )
    )
    replaceable_bad_mask = pd.Series(
        [
            (mmsi, vessel_id) in bad_keys
            for mmsi, vessel_id in zip(
                trips["mmsi"], trips["vessel_id"], strict=True
            )
        ],
        index=trips.index,
    )
    unsupported_mask = trips["mmsi"].isin(unsupported_mmsi)
    source_excluded = trips.loc[
        replaceable_bad_mask,
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
    source_excluded["exclusion_reason"] = (
        "incoherent_source_track_internal_overlap"
    )
    unsupported_excluded = trips.loc[
        unsupported_mask,
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
    unsupported_excluded["exclusion_reason"] = (
        "unsupported_internal_overlap_no_coherent_source"
    )
    excluded = pd.concat(
        [source_excluded, unsupported_excluded],
        ignore_index=True,
    )
    retained = trips.loc[~replaceable_bad_mask & ~unsupported_mask].copy()
    return retained, excluded, {
        "incoherent_source_tracks_excluded": int(len(replaceable_bad)),
        "incoherent_source_track_trips_excluded": int(len(source_excluded)),
        "unsupported_internal_overlap_mmsi": int(len(unsupported_mmsi)),
        "unsupported_internal_overlap_trips_excluded": int(
            len(unsupported_excluded)
        ),
    }


def remove_mmsi_with_overlapping_calls(
    trips: pd.DataFrame,
    calls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Mark a vessel unsupported when its reconstructed calls overlap."""

    unsupported_mmsi = set(
        calls.loc[calls["overlaps_previous_call"], "mmsi"]
    )
    unsupported_mask = trips["mmsi"].isin(unsupported_mmsi)
    excluded = trips.loc[
        unsupported_mask,
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
    excluded["exclusion_reason"] = (
        "unsupported_overlapping_reconstructed_calls"
    )
    return trips.loc[~unsupported_mask].copy(), excluded, {
        "unsupported_overlapping_call_mmsi": int(len(unsupported_mmsi)),
        "unsupported_overlapping_call_trips_excluded": int(len(excluded)),
        "initial_overlapping_call_count": int(
            calls["overlaps_previous_call"].sum()
        ),
    }


def merge_same_city_episodes_2025(
    calls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply frozen episode logic while using year-correct identifiers."""

    calls, episodes, diagnostics = c0a.merge_same_city_episodes(calls)
    episodes["episode_id"] = episodes["episode_id"].str.replace(
        "_2021_", "_2025_", regex=False
    )
    calls["episode_id"] = calls["episode_id"].str.replace(
        "_2021_", "_2025_", regex=False
    )
    return calls, episodes, diagnostics


def enumerate_adjacent_episode_pairs_2025(
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply frozen first-passage logic with year-correct count labels."""

    manifest, diagnostics = c0a.enumerate_adjacent_episode_pairs(episodes)
    return manifest.rename(
        columns={
            "candidate_count_2021": "candidate_count_2025",
            "vessel_count_2021": "vessel_count_2025",
        }
    ), diagnostics


def load_filtered_od_2025(
    paths: list[Path],
    analysis_mmsi: set[str],
    aliases: dict[str, list[str]],
    chunk_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    retained: list[pd.DataFrame] = []
    file_counts: list[dict[str, Any]] = []
    fixed_columns = [
        "vessel_id",
        "trip_id",
        "trip_start",
        "trip_end",
        "trip_start_visit_id",
        "trip_end_visit_id",
    ]
    for path in paths:
        schema = resolve_od_schema(path, aliases)
        usecols = [*fixed_columns, *schema.values()]
        rename = {
            schema["mmsi"]: "mmsi",
            schema["origin_anchorage"]: "trip_start_anchorage_s2id",
            schema["dest_anchorage"]: "trip_end_anchorage_s2id",
        }
        raw_rows = 0
        kept_rows = 0
        source_row = 2
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_rows,
            low_memory=False,
        ):
            n_rows = len(chunk)
            raw_rows += n_rows
            chunk = chunk.rename(columns=rename)
            chunk["mmsi"] = normalize_token(chunk["mmsi"])
            mask = chunk["mmsi"].isin(analysis_mmsi)
            selected = chunk.loc[mask, CANONICAL_OD_COLUMNS].copy()
            if not selected.empty:
                selected["source_file"] = path.name
                selected["source_row"] = np.arange(
                    source_row, source_row + n_rows, dtype=np.int64
                )[mask.to_numpy()]
                retained.append(selected)
                kept_rows += len(selected)
            source_row += n_rows
        file_counts.append(
            {
                "source_file": path.name,
                "raw_rows": raw_rows,
                "analysis_vessel_rows": kept_rows,
            }
        )
    if not retained:
        raise c0a.DataContractError(
            "No 2025 OD rows matched the analysis-vessel universe"
        )
    trips = pd.concat(retained, ignore_index=True)
    for column in CANONICAL_OD_COLUMNS:
        trips[column] = normalize_token(trips[column])
    if trips["trip_id"].eq("").any():
        raise c0a.DataContractError("Retained OD data contain blank trip IDs")

    (
        trips,
        evolving_excluded,
        duplicate_diagnostics,
    ) = resolve_evolving_trip_snapshots(trips)
    reasons = source_sentinel_reasons_2025(trips)
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
        raise c0a.DataContractError("Invalid non-sentinel 2025 timestamps")
    if trips["trip_start"].gt(trips["trip_end"]).any():
        raise c0a.DataContractError("2025 trip start is after trip end")
    if trips[
        ["trip_start_anchorage_s2id", "trip_end_anchorage_s2id"]
    ].eq("").any(axis=None):
        raise c0a.DataContractError("Retained 2025 trip has blank anchorage ID")
    diagnostics = {
        "source_files": file_counts,
        "raw_rows": int(sum(row["raw_rows"] for row in file_counts)),
        "analysis_vessel_rows_before_trip_dedup": int(
            sum(row["analysis_vessel_rows"] for row in file_counts)
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


def build(config: dict[str, Any]) -> tuple[c0a.BuildResult, pd.DataFrame]:
    manifest = verify_frozen_manifest(config)
    vessel_path = CODING_ROOT / config["analysis_vessel_specs_2025"]
    mapping_path = CODING_ROOT / config["anchorage_mapping"]
    vessels = c0a.load_vessel_universe(
        vessel_path, int(config["expected_analysis_vessels"])
    )
    mapping = c0a.load_v3_mapping(mapping_path)
    trips, excluded, od_diagnostics = load_filtered_od_2025(
        [CODING_ROOT / value for value in config["od_files"]],
        set(vessels["mmsi"]),
        config["od_column_aliases"],
        int(config["chunk_rows"]),
    )
    (
        trips,
        incoherent_source_exclusions,
        incoherent_source_diagnostics,
    ) = remove_incoherent_source_tracks(trips)
    excluded = pd.concat(
        [excluded, incoherent_source_exclusions],
        ignore_index=True,
    )
    (
        trips,
        identity_exclusions,
        identity_decisions,
        identity_diagnostics,
    ) = c0a.resolve_source_identity_overlaps(
        trips,
        mapping.set_index("anchorage_id")["country_iso3"],
        2025,
        config["source_identity_rule"],
    )
    excluded = pd.concat([excluded, identity_exclusions], ignore_index=True)
    identity_decisions = identity_decisions.merge(
        vessels[["mmsi", "imo", "vessel_name"]],
        on="mmsi",
        how="left",
        validate="many_to_one",
    )
    identity_decisions = identity_decisions.rename(
        columns={
            "active_endpoint_months_2021": "active_endpoint_months_2025",
            "coverage_hours_2021": "coverage_hours_2025",
        }
    )
    events = c0a.build_endpoint_events(trips, vessels)
    calls, missing_mapping, call_diagnostics = c0a.build_calls(events, mapping)
    if not missing_mapping.empty:
        raise c0a.DataContractError(
            "Frozen v3 mapping does not cover retained 2025 OD calls"
        )
    (
        trips,
        overlapping_call_exclusions,
        overlapping_call_diagnostics,
    ) = remove_mmsi_with_overlapping_calls(trips, calls)
    excluded = pd.concat(
        [excluded, overlapping_call_exclusions],
        ignore_index=True,
    )
    if overlapping_call_diagnostics["unsupported_overlapping_call_mmsi"]:
        events = c0a.build_endpoint_events(trips, vessels)
        calls, missing_mapping, call_diagnostics = c0a.build_calls(
            events, mapping
        )
        if not missing_mapping.empty:
            raise c0a.DataContractError(
                "Frozen v3 mapping does not cover retained 2025 OD calls"
            )
    if calls["overlaps_previous_call"].any():
        raise c0a.DataContractError(
            "Overlapping reconstructed calls remain after unsupported-vessel "
            "exclusion"
        )
    calls, episodes, episode_diagnostics = merge_same_city_episodes_2025(calls)
    pair_manifest, pair_diagnostics = enumerate_adjacent_episode_pairs_2025(
        episodes
    )
    diagnostics = {
        "analysis_year": 2025,
        "operational_parameter_year": 2021,
        "analysis_vessels": int(len(vessels)),
        "v3_mapping_rows": int(len(mapping)),
        "endpoint_events": int(len(events)),
        **od_diagnostics,
        **incoherent_source_diagnostics,
        **identity_diagnostics,
        **overlapping_call_diagnostics,
        **call_diagnostics,
        **episode_diagnostics,
        **pair_diagnostics,
    }
    result = c0a.BuildResult(
        excluded_trips=excluded,
        source_identity_decisions=identity_decisions,
        endpoint_events=events,
        calls=calls,
        episodes=episodes,
        pair_manifest=pair_manifest,
        diagnostics=diagnostics,
    )
    return result, manifest


def write_outputs(
    result: c0a.BuildResult,
    input_manifest: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    paths = {
        "excluded_trips": output_dir / "excluded_trip_records_2025.csv",
        "source_identity_decisions": output_dir
        / "source_identity_resolution_2025.csv",
        "endpoint_events": output_dir / "endpoint_events_2025.feather",
        "calls": output_dir / "ordered_port_calls_2025.feather",
        "episodes": output_dir / "ordered_city_episodes_2025.feather",
        "pair_manifest": output_dir / "adjacent_episode_pair_manifest_2025.csv",
        "diagnostics": output_dir / "c5_1_city_episode_summary.json",
        "input_manifest": output_dir / "c5_1_frozen_input_manifest.csv",
    }
    write_frame_atomic(paths["excluded_trips"], result.excluded_trips)
    write_frame_atomic(
        paths["source_identity_decisions"], result.source_identity_decisions
    )
    write_frame_atomic(paths["endpoint_events"], result.endpoint_events)
    write_frame_atomic(paths["calls"], result.calls)
    write_frame_atomic(paths["episodes"], result.episodes)
    write_frame_atomic(paths["pair_manifest"], result.pair_manifest)
    write_json_atomic(paths["diagnostics"], result.diagnostics)
    write_frame_atomic(paths["input_manifest"], input_manifest)
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
        "analysis_year": 2025,
        "operational_parameter_year": 2021,
        "outputs": outputs,
        "diagnostics": result.diagnostics,
    }
    write_json_atomic(output_dir / "c5_1_output_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config_path = (
        arguments.config
        if arguments.config.is_absolute()
        else CODING_ROOT / arguments.config
    )
    output_dir = (
        arguments.output_dir
        if arguments.output_dir.is_absolute()
        else CODING_ROOT / arguments.output_dir
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["analysis_year"]) != 2025:
        raise ValueError("C5.1 is restricted to 2025")
    result, input_manifest = build(config)
    manifest = write_outputs(result, input_manifest, output_dir)
    print(json.dumps(manifest["diagnostics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
