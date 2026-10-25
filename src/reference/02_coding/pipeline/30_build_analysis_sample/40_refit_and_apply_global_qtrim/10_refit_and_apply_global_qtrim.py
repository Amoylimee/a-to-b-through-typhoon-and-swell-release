#!/usr/bin/env python3
"""Stage 30d: fit the frozen pooled q-trim rule and publish full retained parts.

The stage consumes one hash-bound input manifest. Its run mode is explicit:
``formal_annual`` is the only mode that may claim a complete annual q-trim;
``regression`` and ``synthetic_test`` remain non-formal artifacts. Thresholds
are fitted by the migrated stage-30c implementation, while this stage preserves
the complete voyage rows needed by 30e and 40a.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather


STAGE = Path(__file__).resolve().parent
UPSTREAM_QTRIM = STAGE.parent / "30_build_adjacent_commercial_episode_voyages"
sys.path.insert(0, str(UPSTREAM_QTRIM))
from core import global_qtrim as qtrim  # noqa: E402

sys.path.insert(0, str(STAGE.parents[1] / "00_environment_and_config"))
from owned_paths import local_input

YEAR = 2025
PART_COUNT = 16
ENUMERATION_MODE = "adjacent_commercial_episodes"
RUN_MODES = {"formal_annual", "regression", "synthetic_test"}
FORMAL_LAYERS = {"L1", "L2", "L3"}
IDENTITY_COLUMNS = ["candidate_id", "vessel_uid_clean", "mmsi", "imo"]
QTRIM_COLUMNS = ["temporal_scope_eligible", "normalized_time_ratio", "d_yardstick"]
MEASUREMENT_COLUMNS = [
    "origin_city", "dest_city", "pair_id", "departure_time", "arrival_time",
    "shiptype_group",
]
REQUIRED_COLUMNS = IDENTITY_COLUMNS + QTRIM_COLUMNS + MEASUREMENT_COLUMNS
OPTIONAL_HISTORICAL_FIELDS = {
    "e017_common_followup_eligible": pa.bool_(),
    "e017_followup_deadline": pa.timestamp("ns", tz="UTC"),
}
FROZEN_BASELINE = {
    "candidates": 857_105,
    "eligible_rows": 816_168,
    "q01": 0.0003400554702184352,
    "q99": 0.04599518624920419,
    "trimmed": 16_324,
    "retained": 840_781,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_record(record: dict[str, Any], manifest_path: Path) -> Path:
    raw = record.get("path")
    if not raw:
        raise ValueError("Every input record must declare path")
    path = Path(str(raw))
    path = local_input(path) if path.is_absolute() else (manifest_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest input is absent: {path}")
    if not record.get("sha256") or sha256(path) != record["sha256"]:
        raise ValueError(f"Manifest input hash mismatch: {path}")
    if record.get("bytes") is not None and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Manifest input byte count mismatch: {path}")
    return path


def apply_thresholds(frame: pd.DataFrame, lower: float, upper: float, *, year: int = YEAR) -> pd.DataFrame:
    """Apply the Contract rule; equality with either threshold is retained."""
    eligible = (
        frame["temporal_scope_eligible"].astype(bool)
        & np.isfinite(frame["normalized_time_ratio"])
        & frame["d_yardstick"].gt(0)
    )
    below = eligible & frame["normalized_time_ratio"].lt(lower)
    above = eligible & frame["normalized_time_ratio"].gt(upper)
    output = frame.copy()
    output["global_normalized_time_ratio_q01"] = lower
    output["global_normalized_time_ratio_q99"] = upper
    output["global_trimmed_below_q01"] = below
    output["global_trimmed_above_q99"] = above
    output["global_qtrimmed"] = below | above
    output["retain_after_global_qtrim"] = (
        output["temporal_scope_eligible"].astype(bool) & ~output["global_qtrimmed"]
    )
    output["global_qtrim_version"] = f"pooled_{year}_q01_q99_no_grouping"
    return output


def verify_pool_manifest(
    manifest_path: Path, run_mode: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run_mode not in RUN_MODES or manifest.get("run_mode") != run_mode:
        raise ValueError("30d run mode is invalid or disagrees with the input manifest")
    if manifest.get("status") != "pass":
        raise ValueError("30d input manifest status must be pass")
    if manifest.get("producer_stage") != "30c_build_adjacent_commercial_episode_voyages":
        raise ValueError("30d input manifest must come from stage 30c")
    year = int(manifest.get("analysis_year", -1))
    if year not in {2024, YEAR}:
        raise ValueError("30d requires an explicitly supported independent annual year")
    if year == 2024 and run_mode != "formal_annual":
        raise ValueError("2024 is supported only with verified complete annual inputs")
    part_count = 64 if year == 2024 else PART_COUNT
    formal_layers = {"Y2024"} if year == 2024 else FORMAL_LAYERS
    if manifest.get("candidate_enumeration_mode") != ENUMERATION_MODE:
        raise ValueError(f"30d input must use {ENUMERATION_MODE}")

    input_version = manifest.get("input_version", {})
    version_path = resolve_record(input_version, manifest_path)
    version_hash = input_version["sha256"]
    verification = manifest.get("verification_artifact", {})
    verification_path = resolve_record(verification, manifest_path)
    verification_payload = json.loads(verification_path.read_text(encoding="utf-8"))
    verification_checks = verification_payload.get("checks")
    if (
        verification_payload.get("status") != "pass"
        or not isinstance(verification_checks, dict)
        or not verification_checks
        or not all(verification_checks.values())
    ):
        raise ValueError("30d input verification artifact is not pass")

    partitions = sorted(
        manifest.get("partitions", []), key=lambda item: int(item.get("part_id", -1))
    )
    if len(partitions) != part_count or [int(item.get("part_id", -1)) for item in partitions] != list(range(part_count)):
        raise ValueError(f"30d requires exactly {part_count} contiguous output partitions")

    sources: list[dict[str, Any]] = []
    actual_layers: set[str] = set()
    seen_paths: set[Path] = set()
    for partition in partitions:
        partition_sources = partition.get("sources", [])
        if not partition_sources:
            raise ValueError(f"30d partition {partition['part_id']} has no sources")
        for record in partition_sources:
            layer = str(record.get("layer", ""))
            cohort = str(record.get("cohort", ""))
            if not layer or cohort not in {"old", "new"}:
                raise ValueError("Every 30d source must declare layer and old/new cohort")
            if record.get("input_version_sha256") != version_hash:
                raise ValueError("30d source parts mix or omit upstream input versions")
            path = resolve_record(record, manifest_path)
            if path in seen_paths:
                raise ValueError(f"30d source part is listed more than once: {path}")
            seen_paths.add(path)
            actual_layers.add(layer)
            sources.append({**record, "path": str(path), "part_id": partition["part_id"]})

    declared_layers = set(map(str, manifest.get("completed_layers", [])))
    if actual_layers != declared_layers:
        raise ValueError(f"30d declared and observed layers disagree: {declared_layers} != {actual_layers}")
    for partition in partitions:
        partition_layers = [str(record.get("layer", "")) for record in partition["sources"]]
        if len(partition_layers) != len(set(partition_layers)) or set(partition_layers) != declared_layers:
            raise ValueError(
                f"30d partition {partition['part_id']} layer parts disagree with completed_layers"
            )
    formal = run_mode == "formal_annual"
    if formal:
        if manifest.get("annual_pool_complete") is not True or actual_layers != formal_layers:
            raise ValueError("Formal 30d requires a complete L1/L2/L3 annual pool")
        expected_scope = "complete_2024_annual_pretrim" if year == 2024 else "complete_2025_L1_L2_L3_pretrim"
        if verification_payload.get("scope") != expected_scope:
            raise ValueError("Formal 30d requires complete annual input-verification scope")
        if year == 2024:
            annual = verification_payload.get("annual_source", {})
            annual_path = resolve_record(annual, manifest_path)
            annual_state = json.loads(annual_path.read_text())
            expected_shards = [f"shard_{i:04d}" for i in range(314)]
            if (annual_state.get("annual_run") is not True or annual_state.get("status") != "pass"
                or annual_state.get("year") != 2024 or annual_state.get("shards") != expected_shards
                or annual_state.get("failures") or len(annual_state.get("results", [])) != 314
                or sum(item["n_input_rows"] for item in annual_state["results"]) != 764795042
                or {item["task_id"] for item in annual_state["results"]} != {f"2024_{x}" for x in expected_shards}):
                raise ValueError("2024 annual q-trim requires actual complete 314-shard port inputs")
            if any(source["cohort"] != "new" for source in sources):
                raise ValueError("Independent 2024 pool has no inherited 2025 old cohort")
    elif manifest.get("annual_pool_complete") is True:
        raise ValueError("Regression and synthetic-test inputs cannot claim annual_pool_complete")
    elif run_mode == "synthetic_test" and actual_layers != FORMAL_LAYERS:
        raise ValueError("Synthetic interface test must exercise L1, L2, and L3")

    historical_record = manifest.get("historical_retention", {})
    historical_path = resolve_record(historical_record, manifest_path) if year == YEAR else None

    return manifest, sources, {
        "path": str(version_path),
        "bytes": version_path.stat().st_size,
        "sha256": version_hash,
        "verification_path": str(verification_path),
        "verification_sha256": verification["sha256"],
        "completed_layers": sorted(actual_layers),
        "historical_retention_manifest": {
            "path": str(historical_path),
            "bytes": historical_path.stat().st_size,
            "sha256": historical_record["sha256"],
        } if historical_path is not None else None,
    }


def _read_candidate_ids(
    section: dict[str, Any], historical_manifest_path: Path, label: str
) -> tuple[pd.Index, list[dict[str, Any]]]:
    parts = section.get("parts", [])
    if not parts:
        raise ValueError(f"Historical retention {label} has no parts")
    values: list[pd.Series] = []
    verified: list[dict[str, Any]] = []
    for item in parts:
        path = resolve_record(item, historical_manifest_path)
        table = feather.read_table(path, columns=["candidate_id"])
        series = table.column("candidate_id").to_pandas()
        if series.isna().any():
            raise ValueError(f"Historical retention {label} contains missing candidate_id: {path}")
        values.append(series.astype(str))
        verified.append({
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": item["sha256"], "rows": int(len(series)),
        })
    combined = pd.concat(values, ignore_index=True)
    expected_rows = section.get("expected_rows")
    if expected_rows is None or int(expected_rows) != len(combined):
        raise ValueError(f"Historical retention {label} row count disagrees with its manifest")
    if combined.duplicated().any():
        duplicate = combined.loc[combined.duplicated(keep=False)].iloc[0]
        raise ValueError(f"Historical retention {label} duplicates candidate_id: {duplicate}")
    return pd.Index(combined), verified


def load_historical_retention(
    record: dict[str, Any], old_candidate_ids: pd.Series
) -> tuple[pd.Series, dict[str, Any]]:
    """Verify a complete historical universe before treating absence as not retained."""
    manifest_path = Path(record["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "pass"
        or manifest.get("mode") != "complete_retained_candidate_list"
        or int(manifest.get("analysis_year", -1)) != YEAR
    ):
        raise ValueError("Historical retention manifest is not a verified complete 2025 list")
    universe, universe_parts = _read_candidate_ids(
        manifest.get("old_candidate_universe", {}), manifest_path, "old candidate universe"
    )
    retained, retained_parts = _read_candidate_ids(
        manifest.get("historically_retained", {}), manifest_path, "retained candidates"
    )
    retained_outside = retained.difference(universe)
    if len(retained_outside):
        raise ValueError("Historical retained candidates are outside the declared old universe")
    old = old_candidate_ids.astype(str)
    if old.duplicated().any():
        raise ValueError("Current old cohort duplicates candidate_id")
    current = pd.Index(old)
    missing_from_history = current.difference(universe)
    missing_from_current = universe.difference(current)
    if len(missing_from_history) or len(missing_from_current):
        raise ValueError(
            "Historical old-candidate identity range does not close: "
            f"current_not_in_history={len(missing_from_history)}, "
            f"history_not_in_current={len(missing_from_current)}"
        )
    decisions = old.isin(retained)
    return pd.Series(decisions, index=old_candidate_ids.index, dtype=bool), {
        "manifest": record,
        "mode": manifest["mode"],
        "source_description": manifest.get("source_description"),
        "old_candidate_universe_rows": int(len(universe)),
        "historically_retained_rows": int(len(retained)),
        "current_old_cohort_rows": int(len(current)),
        "identity_range_closed": True,
        "old_candidate_universe_parts": universe_parts,
        "historically_retained_parts": retained_parts,
    }


def validate_and_harmonize_sources(
    sources: list[dict[str, Any]]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Allow only the named E017 history fields to differ across source schemas."""
    schemas: list[tuple[dict[str, Any], pa.Schema]] = []
    for source in sources:
        path = Path(source["path"])
        schema = feather.read_table(path).schema
        missing = sorted(set(REQUIRED_COLUMNS) - set(schema.names))
        if missing:
            raise ValueError(f"30d source lacks required columns {missing}: {path}")
        schemas.append((source, schema))

    reference_names = set(schemas[0][1].names) - set(OPTIONAL_HISTORICAL_FIELDS)
    reference_types = {field.name: field.type for field in schemas[0][1] if field.name in reference_names}
    presence: dict[str, dict[str, Any]] = {}
    for source, schema in schemas:
        path = Path(source["path"])
        ordinary_names = set(schema.names) - set(OPTIONAL_HISTORICAL_FIELDS)
        if ordinary_names != reference_names:
            raise ValueError(f"30d source non-optional fields differ across layers or partitions: {path}")
        for name in reference_names:
            if schema.field(name).type != reference_types[name]:
                raise ValueError(f"30d source field type differs for {name}: {path}")
        for name, required_type in OPTIONAL_HISTORICAL_FIELDS.items():
            if name in schema.names and schema.field(name).type != required_type:
                raise ValueError(f"30d optional historical field type differs for {name}: {path}")
        key = f"{source['layer']}:{int(source['part_id']):03d}"
        presence[key] = {name: name in schema.names for name in OPTIONAL_HISTORICAL_FIELDS}

    frames: dict[str, pd.DataFrame] = {}
    for source, _ in schemas:
        path = Path(source["path"])
        frame = feather.read_table(path).to_pandas()
        if "e017_common_followup_eligible" not in frame:
            frame["e017_common_followup_eligible"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        else:
            frame["e017_common_followup_eligible"] = frame["e017_common_followup_eligible"].astype("boolean")
        if "e017_followup_deadline" not in frame:
            frame["e017_followup_deadline"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        else:
            frame["e017_followup_deadline"] = pd.to_datetime(frame["e017_followup_deadline"], utc=True)
        frame["_qtrim_input_cohort"] = source["cohort"]
        frames[str(path)] = frame
    return frames, {
        "required_columns": REQUIRED_COLUMNS,
        "strict_non_optional_schema": True,
        "optional_historical_fields": {key: str(value) for key, value in OPTIONAL_HISTORICAL_FIELDS.items()},
        "source_presence": presence,
        "missing_optional_values_remain_null": True,
    }


def run_from_manifest(
    input_manifest: Path,
    run_dir: Path,
    run_mode: str,
    *,
    baseline_check: bool = False,
) -> dict[str, Any]:
    input_manifest = input_manifest.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"30d run directory is not empty: {run_dir}")
    manifest, sources, provenance = verify_pool_manifest(input_manifest, run_mode)
    year = int(manifest["analysis_year"])
    if baseline_check and (run_mode != "regression" or year != YEAR):
        raise ValueError("--baseline-check is only valid in regression mode")

    source_frames, schema_compatibility = validate_and_harmonize_sources(sources)
    all_frames = list(source_frames.values())
    pooled = pd.concat(all_frames, ignore_index=True)
    if pooled[IDENTITY_COLUMNS].isna().any(axis=None):
        raise ValueError("30d pooled input contains missing identity fields")
    if pooled["candidate_id"].duplicated().any():
        duplicate = pooled.loc[pooled["candidate_id"].duplicated(keep=False), "candidate_id"].iloc[0]
        raise ValueError(f"30d pooled input duplicates candidate_id: {duplicate}")

    paths = [Path(source["path"]) for source in sources]
    eligible = (
        pooled["temporal_scope_eligible"].astype(bool)
        & np.isfinite(pooled["normalized_time_ratio"])
        & pooled["d_yardstick"].gt(0)
    )
    thresholds = qtrim.fit_global_thresholds(paths, 0.01, 0.99, year)
    if int(thresholds["eligible_rows"]) != int(eligible.sum()):
        raise ValueError("30d eligibility count disagrees with the threshold implementation")
    lower = float(thresholds["normalized_time_ratio_q01"])
    upper = float(thresholds["normalized_time_ratio_q99"])
    applied = apply_thresholds(pooled, lower, upper, year=year)
    old = applied.loc[applied["_qtrim_input_cohort"].eq("old")].copy()
    if year == YEAR:
        historically_retained, historical_provenance = load_historical_retention(
            provenance["historical_retention_manifest"], old["candidate_id"])
    else:
        if len(old):
            raise ValueError("Independent 2024 annual pool cannot inherit old candidates")
        historically_retained = pd.Series(False, index=old.index, dtype=bool)
        historical_provenance = {"status": "not_applicable", "reason": "independent_2024_population"}
    corrected_ratio_under_frozen_thresholds_retained = (
        old["temporal_scope_eligible"].astype(bool)
        & ~(old["normalized_time_ratio"].lt(FROZEN_BASELINE["q01"])
            | old["normalized_time_ratio"].gt(FROZEN_BASELINE["q99"]))
    )

    final_dir = run_dir / f"final_sample_{year}"
    final_dir.mkdir(parents=True)
    part_records = []
    for partition in sorted(manifest["partitions"], key=lambda item: int(item["part_id"])):
        part_sources = [source for source in sources if int(source["part_id"]) == int(partition["part_id"])]
        frame = pd.concat([source_frames[source["path"]] for source in part_sources], ignore_index=True)
        retained = apply_thresholds(frame, lower, upper, year=year)
        retained = retained.loc[retained["retain_after_global_qtrim"]].drop(columns=["_qtrim_input_cohort"])
        output = final_dir / f"part-{int(partition['part_id']):03d}.feather"
        retained.reset_index(drop=True).to_feather(output, compression="zstd", compression_level=6)
        part_records.append({
            "part_id": int(partition["part_id"]),
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "rows": int(len(retained)),
            "input_rows": int(len(frame)),
            "source_records": part_sources,
            "upstream_input_version_sha256": provenance["sha256"],
        })

    cells = {
        "old_retained_to_new_retained": int((historically_retained & old["retain_after_global_qtrim"]).sum()),
        "old_retained_to_new_trimmed": int((historically_retained & ~old["retain_after_global_qtrim"]).sum()),
        "old_trimmed_to_new_retained": int((~historically_retained & old["retain_after_global_qtrim"]).sum()),
        "old_trimmed_to_new_trimmed": int((~historically_retained & ~old["retain_after_global_qtrim"]).sum()),
    }
    new = applied.loc[applied["_qtrim_input_cohort"].eq("new")]
    status = {"formal_annual": "pass", "regression": "regression_pass", "synthetic_test": "test_pass"}[run_mode]
    formal = run_mode == "formal_annual"
    result = {
        "status": status,
        "stage": "30d_refit_and_apply_global_qtrim",
        "run_mode": run_mode,
        "analysis_year": year,
        "candidate_enumeration_mode": ENUMERATION_MODE,
        "annual_pool_complete": formal,
        "completed_layers": provenance["completed_layers"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "state": "candidate run artifact under .runs/, not adopted",
        "input_manifest": {"path": str(input_manifest), "bytes": input_manifest.stat().st_size, "sha256": sha256(input_manifest)},
        "upstream_input_version": {key: provenance[key] for key in ("path", "bytes", "sha256")},
        "input_verification": {
            "path": provenance["verification_path"],
            "bytes": Path(provenance["verification_path"]).stat().st_size,
            "sha256": provenance["verification_sha256"],
        },
        "schema_compatibility": schema_compatibility,
        "historical_retention": historical_provenance,
        "qtrim": {
            "status": "pass",
            "fitted_on_complete_annual_pool": formal,
            "threshold_scope": f"{year}_pooled_no_grouping",
            "q_lower": 0.01,
            "q_upper": 0.99,
            "equality_retained": True,
            "normalized_time_ratio_q01": lower,
            "normalized_time_ratio_q99": upper,
            "threshold_implementation": str((UPSTREAM_QTRIM / "core/global_qtrim.py").resolve()),
        },
        "summary": {
            "input_rows": int(len(applied)),
            "eligible_rows": int(eligible.sum()),
            "global_qtrimmed": int(applied["global_qtrimmed"].sum()),
            "temporal_scope_excluded": int((~applied["temporal_scope_eligible"].astype(bool)).sum()),
            "final_retained": int(applied["retain_after_global_qtrim"].sum()),
            "unique_candidate_id": int(applied["candidate_id"].nunique()),
            "unique_vessel_uid": int(applied["vessel_uid_clean"].nunique()),
        },
        "transition_matrix_old_candidates": cells,
        "corrected_ratio_under_frozen_thresholds_diagnostic": {
            "meaning": "diagnostic only; not the historical retention decision",
            "retained": int(corrected_ratio_under_frozen_thresholds_retained.sum()),
            "disagrees_with_verified_history": int(
                (corrected_ratio_under_frozen_thresholds_retained != historically_retained).sum()
            ),
        },
        "new_candidates_reported_separately": {
            "new_candidates": int(len(new)),
            "new_retained": int(new["retain_after_global_qtrim"].sum()),
            "new_trimmed": int((~new["retain_after_global_qtrim"]).sum()),
        },
        "parts": part_records,
    }
    if sum(item["rows"] for item in part_records) != result["summary"]["final_retained"]:
        raise ValueError("30d retained part counts do not close")
    if sum(cells.values()) != len(old):
        raise ValueError("30d old-candidate transition matrix does not close")

    if baseline_check:
        actual = {
            "candidates": result["summary"]["input_rows"],
            "eligible_rows": result["summary"]["eligible_rows"],
            "q01": lower,
            "q99": upper,
            "trimmed": result["summary"]["global_qtrimmed"],
            "retained": result["summary"]["final_retained"],
        }
        checks = {key: actual[key] == required for key, required in FROZEN_BASELINE.items()}
        result["baseline_checks"] = checks
        if not all(checks.values()):
            raise ValueError(f"Frozen baseline regression failed: {checks}")

    historical_by_id = dict(zip(old["candidate_id"].astype(str), historically_retained))
    diagnostic_by_id = dict(zip(old["candidate_id"].astype(str), corrected_ratio_under_frozen_thresholds_retained))
    applied["historically_retained"] = applied["candidate_id"].astype(str).map(historical_by_id).astype("boolean")
    applied["corrected_ratio_under_frozen_thresholds_retained"] = (
        applied["candidate_id"].astype(str).map(diagnostic_by_id).astype("boolean")
    )
    decisions = applied[
        IDENTITY_COLUMNS
        + ["_qtrim_input_cohort", "temporal_scope_eligible", "normalized_time_ratio", "d_yardstick",
           "historically_retained", "corrected_ratio_under_frozen_thresholds_retained",
           "global_qtrimmed", "retain_after_global_qtrim"]
    ].rename(columns={"_qtrim_input_cohort": "cohort"})
    decisions_dir = run_dir / "global_qtrim_refit"
    decisions_dir.mkdir()
    decisions.to_parquet(decisions_dir / f"candidate_retain_decisions_{year}.parquet", index=False)
    manifest_path = run_dir / f"manifests/final_sample_manifest_{year}.json"
    write_json(manifest_path, result)
    write_json(decisions_dir / f"global_qtrim_refit_{year}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-mode", choices=sorted(RUN_MODES), required=True)
    parser.add_argument("--baseline-check", action="store_true")
    args = parser.parse_args()
    result = run_from_manifest(
        args.input_manifest,
        args.run_dir,
        args.run_mode,
        baseline_check=args.baseline_check,
    )
    print(json.dumps({
        "status": result["status"],
        "run_mode": result["run_mode"],
        "summary": result["summary"],
        "manifest": str(args.run_dir.resolve() / "manifests/final_sample_manifest_2025.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
