#!/usr/bin/env python3
"""Prepare the complete 2025 ten-minute candidate assembly.

The package binds one complete candidate-voyage table, the CP1 source-gap
table, the verified 253-shard cleaned-AIS view, and the two already completed
supplement runs.  It is an offline preparation step: no network request and
no accepted-product write is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
CODING = HERE.parents[2]
STAGE = CODING / "pipeline/40_construct_variables/50_reconstruct_companion_tracks"
import sys

sys.path.insert(0, str(STAGE))
from candidate_full_ten_minute import GRID_NS, strict_grid_ns  # noqa: E402


EXPECTED_SHARDS = [f"shard_{number:04d}" for number in range(253)]
ALLOWED_GAP_CLASSES = {
    "gfw_candidate_4_24h",
    "unresolved_internal_gt24h",
    "endpoint_or_no_ais_unknown",
}
PREPARED_CANDIDATE_COLUMNS = [
    "candidate_id",
    "mmsi",
    "vessel_uid_clean",
    "layer",
    "source_shard",
    "departure_time",
    "arrival_time",
    "origin_city",
    "dest_city",
    "shiptype_group",
    "departure_month",
    "window_hours",
    "total_hours",
    "ais_points_in_window",
    "gfw_candidate_gap_count",
    "unresolved_gt24_gap_count",
    "endpoint_gap_count",
    "has_gfw_gap",
    "has_gt24_gap",
    "has_endpoint_unknown",
    "expected_grid_rows",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalise_ids(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{column} contains null identity values")
        frame[column] = frame[column].astype("string")


def strict_grid_count(start: pd.Series, end: pd.Series) -> pd.Series:
    start_ns = pd.to_datetime(start, utc=True).astype("int64")
    end_ns = pd.to_datetime(end, utc=True).astype("int64")
    if (end_ns <= start_ns).any():
        raise ValueError("a voyage or gap has a non-positive time window")
    first = start_ns // GRID_NS + 1
    last = (end_ns + GRID_NS - 1) // GRID_NS - 1
    return (last - first + 1).clip(lower=0).astype("int64")


def _read_state(root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    state_path = root / "run_state.json"
    if not state_path.is_file():
        raise ValueError(f"{label} run state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("state") != "completed":
        raise ValueError(f"{label} run is not completed")
    return state_path, state


def _check_cleaned_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("shards") != 253:
        raise ValueError("cleaned-AIS manifest is not the verified 253-shard view")
    checks = manifest.get("checks", {})
    required_checks = (
        "all_253_shards_contiguous",
        "all_source_hashes_match_declared_records",
        "vessel_uid_unique_across_shards",
    )
    if not all(checks.get(name) is True for name in required_checks):
        raise ValueError("cleaned-AIS manifest has not passed its identity checks")
    by_shard: dict[str, dict[str, Any]] = {}
    for item in manifest.get("files", []):
        link = Path(item["link_path"])
        shard = link.stem
        if shard in by_shard:
            raise ValueError(f"duplicate cleaned-AIS shard in manifest: {shard}")
        if not link.is_file():
            raise ValueError(f"cleaned-AIS link is missing: {link}")
        if not item.get("sha256") or not item.get("path"):
            raise ValueError(f"cleaned-AIS file is not hash-bound: {shard}")
        by_shard[shard] = {
            "path": str(link.resolve()),
            "sha256": item["sha256"],
            "layer": item["layer"],
            "rows": int(item["rows"]),
            "bytes": int(item["bytes"]),
        }
    if sorted(by_shard) != EXPECTED_SHARDS:
        raise ValueError("cleaned-AIS manifest does not contain exactly shards 0000--0252")
    return by_shard


def _check_patch_files(root: Path, filename: str, label: str) -> None:
    missing = [
        str(root / "shards" / shard / filename)
        for shard in EXPECTED_SHARDS
        if not (root / "shards" / shard / filename).is_file()
    ]
    if missing:
        raise ValueError(f"{label} is missing {len(missing)} shard files; first: {missing[0]}")


def main(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    if not output.is_relative_to((CODING / ".runs").resolve()):
        raise SystemExit("complete candidate preparation must stay under 02_coding/.runs")

    scope_path = args.candidate_scope.resolve()
    quality_path = args.candidate_quality.resolve()
    gap_path = args.gap_intervals.resolve()
    cleaned_manifest_path = args.cleaned_manifest.resolve()
    base_root = args.base_run.resolve()
    endpoint_root = args.endpoint_run.resolve()

    source_files = {
        "candidate_scope": file_record(scope_path),
        "candidate_quality": file_record(quality_path),
        "gap_intervals": file_record(gap_path),
        "cleaned_ais_manifest": file_record(cleaned_manifest_path),
    }
    cleaned_by_shard = _check_cleaned_manifest(cleaned_manifest_path)
    base_state_path, base_state = _read_state(base_root, "4--24h base")
    endpoint_state_path, endpoint_state = _read_state(endpoint_root, "24--48h endpoint")
    base_semantics = str(base_state.get("output_semantics", ""))
    endpoint_semantics = str(endpoint_state.get("output_semantics", ""))
    if "4-24" not in base_semantics or "24-48" in base_semantics:
        raise ValueError("the base run is not the completed 4--24h supplement")
    if "24" not in endpoint_semantics or "endpoint" not in endpoint_semantics:
        raise ValueError("the endpoint run is not the completed 24--48h candidate")
    _check_patch_files(base_root, "ten_minute_gap_supplement.parquet", "4--24h base run")
    _check_patch_files(endpoint_root, "ten_minute_endpoint_extension.parquet", "24--48h endpoint run")

    scope_columns = [
        "candidate_id",
        "mmsi",
        "vessel_uid_clean",
        "origin_city",
        "dest_city",
        "departure_time",
        "arrival_time",
        "total_hours",
        "shiptype_group",
        "departure_month",
    ]
    quality_columns = [
        "candidate_id",
        "mmsi",
        "vessel_uid_clean",
        "layer",
        "source_shard",
        "departure_time",
        "arrival_time",
        "window_hours",
        "ais_points_in_window",
        "gfw_candidate_gap_count",
        "unresolved_gt24_gap_count",
        "endpoint_gap_count",
    ]
    scope = pd.read_parquet(scope_path, columns=scope_columns)
    quality = pd.read_parquet(quality_path, columns=quality_columns)
    _normalise_ids(scope, ["candidate_id", "mmsi", "vessel_uid_clean"])
    _normalise_ids(quality, ["candidate_id", "mmsi", "vessel_uid_clean", "source_shard"])
    if scope.candidate_id.duplicated().any() or quality.candidate_id.duplicated().any():
        raise ValueError("candidate scope or candidate quality contains duplicate candidate IDs")
    if len(scope) != len(quality) or set(scope.candidate_id) != set(quality.candidate_id):
        raise ValueError("candidate scope and quality do not describe the same voyages")

    merged = scope.merge(
        quality,
        on="candidate_id",
        how="inner",
        suffixes=("_scope", "_quality"),
        validate="one_to_one",
    )
    for left, right in (
        ("mmsi_scope", "mmsi_quality"),
        ("vessel_uid_clean_scope", "vessel_uid_clean_quality"),
    ):
        if not merged[left].astype(str).eq(merged[right].astype(str)).all():
            raise ValueError(f"candidate scope and quality disagree in {left}/{right}")
    for left, right in (
        ("departure_time_scope", "departure_time_quality"),
        ("arrival_time_scope", "arrival_time_quality"),
    ):
        if not pd.to_datetime(merged[left], utc=True).eq(
            pd.to_datetime(merged[right], utc=True)
        ).all():
            raise ValueError(f"candidate scope and quality disagree in {left}/{right}")

    candidates = pd.DataFrame(
        {
            "candidate_id": merged.candidate_id.astype("string"),
            "mmsi": merged.mmsi_scope.astype("string"),
            "vessel_uid_clean": merged.vessel_uid_clean_scope.astype("string"),
            "layer": merged.layer.astype("string"),
            "source_shard": merged.source_shard.astype("string"),
            "departure_time": pd.to_datetime(merged.departure_time_scope, utc=True),
            "arrival_time": pd.to_datetime(merged.arrival_time_scope, utc=True),
            "origin_city": merged.origin_city.astype("string"),
            "dest_city": merged.dest_city.astype("string"),
            "shiptype_group": merged.shiptype_group.astype("string"),
            "departure_month": merged.departure_month.astype("string"),
            "window_hours": merged.window_hours.astype(float),
            "total_hours": merged.total_hours.astype(float),
            "ais_points_in_window": merged.ais_points_in_window.astype("int64"),
            "gfw_candidate_gap_count": merged.gfw_candidate_gap_count.astype("int64"),
            "unresolved_gt24_gap_count": merged.unresolved_gt24_gap_count.astype("int64"),
            "endpoint_gap_count": merged.endpoint_gap_count.astype("int64"),
        }
    )
    if candidates.source_shard.isna().any() or ~candidates.source_shard.isin(EXPECTED_SHARDS).all():
        raise ValueError("candidate quality has an unknown source shard")
    candidates["has_gfw_gap"] = candidates.gfw_candidate_gap_count.gt(0)
    candidates["has_gt24_gap"] = candidates.unresolved_gt24_gap_count.gt(0)
    candidates["has_endpoint_unknown"] = candidates.endpoint_gap_count.gt(0)
    candidates["expected_grid_rows"] = strict_grid_count(
        candidates.departure_time, candidates.arrival_time
    ).to_numpy()
    if candidates.expected_grid_rows.eq(0).all():
        raise ValueError("candidate scope contains no ten-minute grid rows")
    if candidates.expected_grid_rows.lt(0).any():
        raise ValueError("candidate grid row count is negative")

    gaps = pd.read_parquet(gap_path)
    _normalise_ids(gaps, ["gap_id", "candidate_id", "mmsi", "vessel_uid_clean", "source_shard"])
    if gaps.gap_id.duplicated().any():
        raise ValueError("source gap table contains duplicate gap IDs")
    if not set(gaps.gap_class.astype(str)).issubset(ALLOWED_GAP_CLASSES):
        raise ValueError("source gap table contains an unexpected gap class")
    candidate_lookup = candidates.set_index("candidate_id")
    if not set(gaps.candidate_id).issubset(set(candidate_lookup.index)):
        raise ValueError("source gap table contains an unknown candidate")
    gap_candidate = candidate_lookup.loc[gaps.candidate_id]
    if not gaps.mmsi.astype(str).eq(gap_candidate.mmsi.astype(str).to_numpy()).all():
        raise ValueError("source gaps disagree with candidate MMSI")
    if not gaps.vessel_uid_clean.astype(str).eq(gap_candidate.vessel_uid_clean.astype(str).to_numpy()).all():
        raise ValueError("source gaps disagree with candidate API identity")
    if not gaps.source_shard.astype(str).eq(gap_candidate.source_shard.astype(str).to_numpy()).all():
        raise ValueError("source gaps disagree with candidate source shard")
    gaps["gap_start"] = pd.to_datetime(gaps.gap_start, utc=True)
    gaps["gap_end"] = pd.to_datetime(gaps.gap_end, utc=True)
    gaps["gap_grid_rows"] = strict_grid_count(gaps.gap_start, gaps.gap_end).to_numpy()
    if not gaps.gap_hours.astype(float).sub(
        (gaps.gap_end - gaps.gap_start).dt.total_seconds() / 3600.0
    ).abs().lt(1e-6).all():
        raise ValueError("source gap duration does not match its timestamps")

    candidates["_month"] = candidates.departure_time.dt.month.astype("int8")
    candidates["_selection_rank"] = candidates.candidate_id.map(
        lambda value: hashlib.sha256(f"E001-full-grid-probe-v1|{value}".encode()).hexdigest()
    )
    probe = (
        candidates.sort_values(["_selection_rank", "candidate_id"], kind="stable")
        .groupby(["layer", "_month", "has_gfw_gap", "has_gt24_gap"], sort=True, dropna=False)
        .head(8)
        .sort_values(["layer", "_month", "_selection_rank", "candidate_id"], kind="stable")
        .head(720)
        .drop(columns=["_month", "_selection_rank"])
        .reset_index(drop=True)
    )
    if probe.empty or not set(probe.candidate_id).issubset(set(candidates.candidate_id)):
        raise ValueError("probe scope was not constructed from the candidate scope")

    output.mkdir(parents=True)
    (output / "candidates").mkdir()
    (output / "gaps").mkdir()
    probe_path = output / "probe_scope.parquet"
    probe.to_parquet(probe_path, index=False, compression="zstd")

    candidate_records: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for shard, candidate_group in candidates.groupby("source_shard", sort=True):
        shard = str(shard)
        candidate_group = candidate_group[PREPARED_CANDIDATE_COLUMNS].sort_values(
            "candidate_id", kind="stable"
        )
        gap_group = gaps.loc[gaps.source_shard.eq(shard)].sort_values(
            ["candidate_id", "gap_start", "gap_end", "gap_id"], kind="stable"
        )
        candidate_file = output / "candidates" / f"{shard}.parquet"
        gap_file = output / "gaps" / f"{shard}.parquet"
        candidate_group.to_parquet(candidate_file, index=False, compression="zstd")
        gap_group.to_parquet(gap_file, index=False, compression="zstd")
        source = cleaned_by_shard[shard]
        base_patch = base_root / "shards" / shard / "ten_minute_gap_supplement.parquet"
        endpoint_patch = endpoint_root / "shards" / shard / "ten_minute_endpoint_extension.parquet"
        base_patch_record = file_record(base_patch)
        endpoint_patch_record = file_record(endpoint_patch)
        candidate_records.append(
            {
                "shard": shard,
                "candidates": int(len(candidate_group)),
                "expected_grid_rows": int(candidate_group.expected_grid_rows.sum()),
                "path": str(candidate_file.resolve()),
                "sha256": sha256(candidate_file),
            }
        )
        shard_records.append(
            {
                "shard": shard,
                "candidates": int(len(candidate_group)),
                "expected_grid_rows": int(candidate_group.expected_grid_rows.sum()),
                "gaps": int(len(gap_group)),
                "gap_grid_rows": int(gap_group.gap_grid_rows.sum()),
                "candidate_path": str(candidate_file.resolve()),
                "candidate_sha256": sha256(candidate_file),
                "gap_path": str(gap_file.resolve()),
                "gap_sha256": sha256(gap_file),
                "cleaned_ais_path": source["path"],
                "cleaned_ais_sha256": source["sha256"],
                "cleaned_ais_bytes": source["bytes"],
                "cleaned_ais_layer": source["layer"],
                "base_patch_path": base_patch_record["path"],
                "base_patch_sha256": base_patch_record["sha256"],
                "base_patch_bytes": base_patch_record["bytes"],
                "endpoint_patch_path": endpoint_patch_record["path"],
                "endpoint_patch_sha256": endpoint_patch_record["sha256"],
                "endpoint_patch_bytes": endpoint_patch_record["bytes"],
            }
        )

    candidates = candidates.drop(columns=["_month", "_selection_rank"])
    expected = {
        "candidates": int(len(candidates)),
        "source_gaps": int(len(gaps)),
        "expected_grid_rows": int(candidates.expected_grid_rows.sum()),
        "source_gap_grid_rows": int(gaps.gap_grid_rows.sum()),
        "source_shards": int(candidates.source_shard.nunique()),
        "layers": candidates.layer.value_counts().sort_index().to_dict(),
        "gap_classes": gaps.gap_class.value_counts().sort_index().to_dict(),
        "gap_grid_rows_by_class": gaps.groupby("gap_class").gap_grid_rows.sum().sort_index().to_dict(),
    }
    if expected["source_shards"] != 253 or len(shard_records) != 253:
        raise ValueError("prepared candidate scope is not distributed over all 253 shards")

    manifest = {
        "status": "pass",
        "operation": "prepare_complete_2025_ten_minute_candidate",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "year": 2025,
        "candidate_only": True,
        "network_requests": 0,
        "products_modified": 0,
        "sources": {
            **source_files,
            "base_run_state": {**file_record(base_state_path), "state": base_state["state"]},
            "endpoint_run_state": {**file_record(endpoint_state_path), "state": endpoint_state["state"]},
        },
        "source_runs": {
            "base_4_24h": {"root": str(base_root), "run_state_sha256": sha256(base_state_path), "semantics": base_semantics},
            "endpoint_24_48h": {"root": str(endpoint_root), "run_state_sha256": sha256(endpoint_state_path), "semantics": endpoint_semantics},
        },
        "policy": {
            "grid_seconds": 600,
            "grid_window": "strictly inside departure_time and arrival_time",
            "short_gap_max_hours": 4.0,
            "short_gap_method": "linear latitude and shortest-path longitude between actual AIS fixes",
            "no_extrapolation": True,
            "precedence": [
                "observed_ais",
                "ais_interpolated_le4h",
                "gfw_candidate_4_24h patch",
                "unresolved_internal_gt24h endpoint patch",
                "unresolved",
            ],
            "patch_rule": "supplements fill unresolved grid rows only; observed AIS is never overwritten",
            "endpoint_label": "companion_endpoint_24_48h",
            "endpoint_label_is_candidate_only": True,
        },
        "expected": expected,
        "candidate_scope": {
            "path": str(scope_path),
            "sha256": source_files["candidate_scope"]["sha256"],
            "rows": int(len(scope)),
        },
        "probe_scope": {
            "path": str(probe_path.resolve()),
            "sha256": sha256(probe_path),
            "candidates": int(len(probe)),
        },
        "candidate_files": candidate_records,
        "shards": shard_records,
        "output_semantics": (
            "complete 2025 ten-minute voyage-grid candidate: observed AIS plus "
            "short-gap interpolation, completed 4--24h GFW candidate supplement, "
            "completed 24--48h endpoint candidate patch, and explicit unresolved rows; "
            "not an accepted research product"
        ),
    }
    write_json(output / "input_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], **expected, "probe_candidates": len(probe)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scope", type=Path, required=True)
    parser.add_argument("--candidate-quality", type=Path, required=True)
    parser.add_argument("--gap-intervals", type=Path, required=True)
    parser.add_argument("--cleaned-manifest", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--endpoint-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
