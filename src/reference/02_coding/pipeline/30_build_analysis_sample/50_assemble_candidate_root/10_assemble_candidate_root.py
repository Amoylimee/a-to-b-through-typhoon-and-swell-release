#!/usr/bin/env python3
"""Stage 30e: assemble a hash-bound candidate root from an explicit 30d manifest.

Formal mode accepts only a complete annual L1/L2/L3 result after pooled q-trim.
Drill mode is explicit, remains marked unusable downstream, and exists only to
exercise the same directory, lineage, identity, and failure contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.feather as feather


REPO_ROOT = Path(__file__).resolve().parents[4]
YEAR = 2025
PART_COUNT = 16
ENUMERATION_MODE = "adjacent_commercial_episodes"
FORMAL_STAGE = "30d_refit_and_apply_global_qtrim"
FORMAL_LAYERS = {"L1", "L2", "L3"}
FORMAL_COLUMNS = {
    "candidate_id", "vessel_uid_clean", "mmsi", "imo",
    "origin_city", "dest_city", "pair_id", "departure_time", "arrival_time",
    "shiptype_group", "retain_after_global_qtrim", "global_qtrimmed",
    "global_normalized_time_ratio_q01", "global_normalized_time_ratio_q99",
}
DRILL_COLUMNS = {
    "candidate_id", "vessel_uid_clean", "mmsi", "imo",
    "origin_city", "dest_city", "pair_id", "departure_time", "arrival_time",
    "shiptype_group", "retain_after_c3_final", "normalized_time_ratio",
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_record_path(item: dict[str, Any], manifest_path: Path) -> Path:
    raw = item.get("path") or item.get("output_path")
    if not raw:
        raise ValueError("Every source part must declare path or output_path")
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def verify_record(item: dict[str, Any], manifest_path: Path) -> Path:
    path = resolve_record_path(item, manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest source part is absent: {path}")
    expected_hash = item.get("sha256") or item.get("output_sha256")
    if not expected_hash:
        raise ValueError(f"Manifest source part has no SHA-256: {path}")
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Source part hash mismatch: {path}")
    if item.get("bytes") is not None and path.stat().st_size != int(item["bytes"]):
        raise ValueError(f"Source part byte count mismatch: {path}")
    return path


def inspect_part(path: Path, required_columns: set[str], formal: bool, year: int = YEAR) -> dict[str, Any]:
    table = feather.read_table(path)
    columns = set(table.schema.names)
    missing = sorted(required_columns - columns)
    if missing:
        raise ValueError(f"Candidate part lacks required columns {missing}: {path}")
    frame = table.select([
        "candidate_id", "vessel_uid_clean", "mmsi", "imo",
        "departure_time", "arrival_time",
        *(["retain_after_global_qtrim"] if formal else ["retain_after_c3_final"]),
    ]).to_pandas()
    identity_columns = ["candidate_id", "vessel_uid_clean", "mmsi", "imo"]
    if frame[identity_columns].isna().any(axis=None):
        raise ValueError(f"Candidate part contains missing identity fields: {path}")
    if frame.candidate_id.duplicated().any():
        raise ValueError(f"Candidate part contains duplicate candidate_id: {path}")
    if formal and not frame.retain_after_global_qtrim.astype(bool).all():
        raise ValueError(f"Formal source contains a non-retained q-trim row: {path}")
    departure = pd.to_datetime(frame.departure_time, utc=True, errors="raise")
    arrival = pd.to_datetime(frame.arrival_time, utc=True, errors="raise")
    if year == 2024 and not (departure.dt.year.eq(year) & arrival.dt.year.eq(year)).all():
        raise ValueError("2024 calendar cohort contains a departure/arrival outside its input window")
    if (arrival <= departure).any():
        raise ValueError(f"Candidate part contains non-positive voyage time: {path}")
    identities = frame[identity_columns].copy()
    identities["source_part"] = str(path)
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        "rows": len(frame), "columns": table.schema.names,
        "next_year_arrival_rows": int(arrival.dt.year.eq(year + 1).sum()),
        "latest_arrival": arrival.max().isoformat() if len(frame) else None,
        "identities": identities,
    }


def validate_source_manifest(
    manifest_path: Path, *, drill: bool, interface_test: bool, workers: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_status = "pass" if not interface_test else "test_pass"
    if manifest.get("status") != expected_status:
        raise ValueError(f"Source manifest status must be {expected_status}")
    year = int(manifest.get("analysis_year", -1))
    if year not in {2024, YEAR} or (year == 2024 and (drill or interface_test)):
        raise ValueError("Source manifest must describe an explicitly supported annual year/mode")
    part_count = 64 if year == 2024 else PART_COUNT
    formal_layers = {"Y2024"} if year == 2024 else FORMAL_LAYERS
    if manifest.get("candidate_enumeration_mode") != ENUMERATION_MODE:
        raise ValueError(f"Source manifest must declare {ENUMERATION_MODE}")
    if drill:
        if manifest.get("assembly_purpose") != "pretrim_structure_drill":
            raise ValueError("Drill input must explicitly declare assembly_purpose=pretrim_structure_drill")
        if manifest.get("qtrim", {}).get("status") != "not_run":
            raise ValueError("Drill input must explicitly declare qtrim status not_run")
        required_columns = DRILL_COLUMNS
    else:
        if manifest.get("stage") != FORMAL_STAGE:
            raise ValueError(f"Formal source must declare stage={FORMAL_STAGE}")
        if set(manifest.get("completed_layers", [])) != formal_layers:
            raise ValueError("30d candidate root input requires completed_layers L1, L2, and L3")
        qtrim = manifest.get("qtrim", {})
        if qtrim.get("status") != "pass":
            raise ValueError("30d candidate root input requires a passing q-trim")
        if (
            qtrim.get("threshold_scope") != f"{year}_pooled_no_grouping"
            or qtrim.get("q_lower") != 0.01
            or qtrim.get("q_upper") != 0.99
            or qtrim.get("equality_retained") is not True
        ):
            raise ValueError("30d q-trim method metadata disagrees with the Contract")
        pool_manifest_path = verify_record(manifest.get("input_manifest", {}), manifest_path)
        pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
        verification_path = verify_record(manifest.get("input_verification", {}), manifest_path)
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification_checks = verification.get("checks")
        if (
            pool_manifest.get("producer_stage") != "30c_build_adjacent_commercial_episode_voyages"
            or pool_manifest.get("run_mode") != manifest.get("run_mode")
            or int(pool_manifest.get("analysis_year", -1)) != year
            or pool_manifest.get("candidate_enumeration_mode") != ENUMERATION_MODE
            or pool_manifest.get("annual_pool_complete") is not manifest.get("annual_pool_complete")
            or set(pool_manifest.get("completed_layers", [])) != formal_layers
        ):
            raise ValueError("30d output disagrees with its verified input manifest")
        if (
            verification.get("status") != "pass"
            or not isinstance(verification_checks, dict)
            or not verification_checks
            or not all(verification_checks.values())
        ):
            raise ValueError("30d input verification artifact is not pass")
        if interface_test:
            if manifest.get("run_mode") != "synthetic_test" or manifest.get("annual_pool_complete") is not False:
                raise ValueError("Interface-test assembly requires a non-annual synthetic_test 30d result")
            if qtrim.get("fitted_on_complete_annual_pool") is not False:
                raise ValueError("Interface-test q-trim cannot claim a complete annual fit")
            if verification.get("scope") != "synthetic_2025_L1_L2_L3_pretrim":
                raise ValueError("Interface-test input verification scope is not synthetic L1/L2/L3")
        else:
            if manifest.get("run_mode") != "formal_annual":
                raise ValueError("Formal candidate root requires run_mode=formal_annual")
            if manifest.get("annual_pool_complete") is not True:
                raise ValueError("Formal candidate root requires annual_pool_complete=true")
            if qtrim.get("fitted_on_complete_annual_pool") is not True:
                raise ValueError("Formal candidate root requires completed pooled annual q-trim")
            expected_scope = "complete_2024_annual_pretrim" if year == 2024 else "complete_2025_L1_L2_L3_pretrim"
            if verification.get("scope") != expected_scope:
                raise ValueError("Formal candidate root requires complete annual input verification")
            if year == 2024:
                annual_path = verify_record(verification.get("annual_source", {}), verification_path)
                annual = json.loads(annual_path.read_text())
                shards = [f"shard_{i:04d}" for i in range(314)]
                if (annual.get("status") != "pass" or annual.get("annual_run") is not True
                    or annual.get("year") != year or annual.get("shards") != shards
                    or annual.get("failures") or len(annual.get("results", [])) != 314
                    or sum(item["n_input_rows"] for item in annual["results"]) != 764795042
                    or {item["task_id"] for item in annual["results"]} != {f"2024_{x}" for x in shards}):
                    raise ValueError("2024 assembly is not bound to complete annual port inputs")
        version = manifest.get("upstream_input_version", {})
        version_path = resolve_record_path(version, manifest_path)
        if not version_path.is_file() or sha256(version_path) != version.get("sha256"):
            raise ValueError("Formal upstream input version path/hash is absent or mismatched")
        version_hash = version["sha256"]
        mismatched_parts = [
            item.get("part_id") for item in manifest.get("parts", [])
            if item.get("upstream_input_version_sha256") != version_hash
        ]
        if mismatched_parts:
            raise ValueError(f"Formal source parts mix or omit upstream input versions: {mismatched_parts}")
        required_columns = FORMAL_COLUMNS
    part_items = sorted(manifest.get("parts", []), key=lambda item: int(item.get("part_id", -1)))
    if len(part_items) != part_count or [int(item.get("part_id", -1)) for item in part_items] != list(range(part_count)):
        raise ValueError(f"Source manifest must declare exactly {part_count} contiguous parts")
    part_paths = [verify_record(item, manifest_path) for item in part_items]
    inspected: list[dict[str, Any] | None] = [None] * part_count
    failures = []
    with ThreadPoolExecutor(max_workers=min(workers, part_count)) as executor:
        futures = {executor.submit(inspect_part, path, required_columns, not drill, year): index for index, path in enumerate(part_paths)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                inspected[index] = future.result()
            except Exception as exc:
                failures.append((index, repr(exc)))
    if failures:
        raise RuntimeError(f"{len(failures)} candidate part validation failures: {failures}")
    results = [item for item in inspected if item is not None]
    identities = pd.concat([item.pop("identities") for item in results], ignore_index=True)
    duplicate_ids = identities.loc[identities.candidate_id.duplicated(keep=False), "candidate_id"]
    if not duplicate_ids.empty:
        raise ValueError(f"Duplicate candidate_id across source parts: {duplicate_ids.iloc[0]}")
    declared_rows = manifest.get("summary", {}).get("final_retained" if not drill else "pretrim_rows")
    actual_rows = sum(item["rows"] for item in results)
    if declared_rows is None or int(declared_rows) != actual_rows:
        raise ValueError(f"Source manifest row count mismatch: declared={declared_rows}, actual={actual_rows}")
    return manifest, results


def link_or_verify(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(f"Existing link points to a different source: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    os.symlink(str(source), str(destination))


def assemble_candidate_root(
    candidate_root: Path,
    source_manifest: Path,
    *,
    drill: bool = False,
    interface_test: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if drill and interface_test:
        raise ValueError("drill and interface_test modes are mutually exclusive")
    candidate_root = candidate_root.resolve()
    source_manifest = source_manifest.resolve()
    accepted_root = (REPO_ROOT / "02_coding/data/processed/products").resolve()
    if candidate_root == accepted_root or accepted_root in candidate_root.parents:
        raise ValueError("Candidate root may not be created under accepted products")
    manifest, parts = validate_source_manifest(
        source_manifest, drill=drill, interface_test=interface_test, workers=workers
    )
    year = int(manifest["analysis_year"])
    year_dir = candidate_root / str(year)
    if candidate_root.exists() and any(candidate_root.iterdir()):
        raise FileExistsError(f"Candidate root is not empty: {candidate_root}")
    year_dir.mkdir(parents=True)
    for part_id, part in enumerate(parts):
        link_or_verify(Path(part["path"]), year_dir / f"part-{part_id:03d}.feather")
    link_or_verify(source_manifest, year_dir / f"final_sample_manifest_{year}.json")
    entry = {
        "cohort": "calendar", "calendar_truncated": True,
        "candidate_enumeration_mode": ENUMERATION_MODE,
        "source_path": str(Path(parts[0]["path"]).parent),
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256(source_manifest),
        "link_path": str(year_dir), "row_count": sum(item["rows"] for item in parts),
        "part_count": len(parts), "parts": parts,
        "next_year_arrival_rows": sum(item["next_year_arrival_rows"] for item in parts),
        "has_next_year_arrival": any(item["next_year_arrival_rows"] for item in parts),
        "qtrim_status": manifest["qtrim"]["status"],
        "annual_pool_complete": bool(manifest.get("annual_pool_complete", False)),
    }
    payload = {
        "status": "drill_only" if drill else ("interface_test_only" if interface_test else "pass"),
        "stage": "30e_assemble_candidate_root", "analysis_years": [year],
        "assembly_mode": "pretrim_drill" if drill else ("synthetic_interface_test" if interface_test else "formal_annual"),
        "candidate_enumeration_mode": ENUMERATION_MODE,
        "candidate_root": str(candidate_root), "years": {str(year): entry},
        "source_30d_manifest_sha256": sha256(source_manifest),
        "downstream_ready": not (drill or interface_test),
        "warning": ("UNTRIMMED L1/L2 STRUCTURE DRILL; L3 missing; annual q-trim not run; "
                    "must not be used by formal downstream stages") if drill else (
                    "SYNTHETIC INTERFACE TEST ONLY; not an annual result; must not be used by formal downstream stages"
                    if interface_test else None),
        "downstream_contract": {
            "year_directory": f"{year}/", "part_pattern": "part-*.feather",
            "required_part_count": len(parts),
            "source_manifest_link": f"{year}/final_sample_manifest_{year}.json",
        },
    }
    write_json(candidate_root / "assembly_manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--drill", action="store_true", help="accept only an explicitly untrimmed structure-drill manifest")
    modes.add_argument("--interface-test", action="store_true", help="accept only an actual synthetic-test 30d result")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(assemble_candidate_root(
        args.candidate_root,
        args.source_manifest,
        drill=args.drill,
        interface_test=args.interface_test,
        workers=args.workers,
    ), indent=2, sort_keys=True))
