#!/usr/bin/env python3
"""Build the complete 2025 ten-minute voyage-grid candidate.

This runner reads only hash-bound, already completed inputs.  It writes one
complete grid per source shard and keeps the result separate from the formal
measurement product.  The endpoint evidence label is intentionally preserved
as a candidate-only label for later review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import resource
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
CODING = HERE.parents[2]
STAGE = CODING / "pipeline/40_construct_variables/50_reconstruct_companion_tracks"
sys.path.insert(0, str(STAGE))

from candidate_full_ten_minute import (  # noqa: E402
    ALLOWED_EVIDENCE,
    ENDPOINT_LABEL,
    UNRESOLVED_LABEL,
    build_full_candidate_grid,
    validate_candidate_frame,
)


RAW_COLUMNS = [
    "vessel_uid",
    "mmsi",
    "UTC",
    "UTC_10min",
    "lat",
    "lon",
    "source_provider",
    "source_resolution_original",
]
PATCH_COLUMNS = [
    "gap_id",
    "candidate_id",
    "mmsi",
    "utc",
    "lat",
    "lon",
    "track_evidence",
    "anchor_source",
]
EXPECTED_SHARDS = [f"shard_{number:04d}" for number in range(253)]
TOTAL_KEYS = [
    "candidates",
    "expected_grid_rows",
    "grid_rows",
    "raw_rows",
    "relevant_raw_rows",
    "identity_mismatch_rows",
    "observed_ais_rows",
    "ais_interpolated_le4h_rows",
    "gfw_anchor_rows",
    "gfw_supported_interpolation_rows",
    "endpoint_24_48h_rows",
    "unresolved_rows",
    "source_gap_grid_rows_assigned",
    "base_valid_patch_rows",
    "base_applied_rows",
    "base_skipped_existing_rows",
    "endpoint_valid_patch_rows",
    "endpoint_applied_rows",
    "endpoint_skipped_existing_rows",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _string_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        if frame[column].isna().any():
            raise ValueError(f"{column} contains null values")
        frame[column] = frame[column].astype("string")


def _check_size(path: Path, expected: int, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if expected and path.stat().st_size != int(expected):
        raise ValueError(f"{label} size changed: {path}")


def _read_patch(
    path: Path,
    candidates: pd.DataFrame,
    gaps: pd.DataFrame,
    expected_labels: set[str],
    label: str,
) -> pd.DataFrame:
    patch = pq.read_table(path, columns=PATCH_COLUMNS).to_pandas()
    _string_columns(patch, ["gap_id", "candidate_id", "mmsi", "track_evidence"])
    if patch.empty:
        return patch
    patch["utc"] = pd.to_datetime(patch.utc, utc=True, errors="raise")
    if patch.duplicated(["candidate_id", "utc"]).any():
        raise ValueError(f"{label} has duplicate candidate/time keys")
    if not patch.utc.eq(patch.utc.dt.floor("10min")).all():
        raise ValueError(f"{label} has off-grid timestamps")
    labels = set(patch.track_evidence.astype(str))
    if not labels.issubset(expected_labels):
        raise ValueError(f"{label} has unexpected evidence labels: {labels - expected_labels}")
    resolved = patch.track_evidence.ne(UNRESOLVED_LABEL).to_numpy()
    coords = patch[["lat", "lon"]].to_numpy(dtype=float)
    if resolved.any() and not np.isfinite(coords[resolved]).all():
        raise ValueError(f"{label} has non-finite resolved coordinates")
    if (~resolved).any() and np.isfinite(coords[~resolved]).any():
        raise ValueError(f"{label} has coordinates on unresolved rows")

    candidate_ids = set(candidates.candidate_id.astype(str))
    gap_ids = set(gaps.gap_id.astype(str))
    if not set(patch.candidate_id.astype(str)).issubset(candidate_ids):
        raise ValueError(f"{label} contains an unknown candidate")
    if not set(patch.gap_id.astype(str)).issubset(gap_ids):
        raise ValueError(f"{label} contains an unknown source gap")
    candidate_mmsi = candidates.set_index("candidate_id").mmsi.astype(str)
    if not patch.mmsi.astype(str).eq(patch.candidate_id.map(candidate_mmsi).astype(str)).all():
        raise ValueError(f"{label} disagrees with candidate MMSI")
    gap_candidate = gaps.set_index("gap_id").candidate_id.astype(str)
    if not patch.candidate_id.astype(str).eq(patch.gap_id.map(gap_candidate).astype(str)).all():
        raise ValueError(f"{label} disagrees with source gap ownership")
    return patch


def _read_raw_groups(
    path: Path, candidates: pd.DataFrame
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, int]]:
    raw = feather.read_table(path, columns=RAW_COLUMNS, memory_map=True).to_pandas()
    raw_rows = int(len(raw))
    _string_columns(raw, ["vessel_uid", "mmsi"])
    raw["UTC"] = pd.to_datetime(raw.UTC, utc=True, errors="raise")
    raw["UTC_10min"] = pd.to_datetime(raw.UTC_10min, utc=True, errors="raise")
    candidate_apis = set(candidates.vessel_uid_clean.astype(str))
    api_relevant = raw.vessel_uid.astype(str).isin(candidate_apis)
    raw_api = raw.loc[api_relevant].copy()
    pair_keys = set(
        zip(candidates.vessel_uid_clean.astype(str), candidates.mmsi.astype(str))
    )
    pair_key = raw.vessel_uid.astype(str) + "\x1f" + raw.mmsi.astype(str)
    wanted_key = {f"{api}\x1f{mmsi}" for api, mmsi in pair_keys}
    exact = pair_key.isin(wanted_key)
    relevant = raw.loc[exact].copy()
    identity_mismatch_rows = int(len(raw_api) - len(relevant))
    if relevant["UTC"].isna().any() or relevant["UTC_10min"].isna().any():
        raise ValueError(f"raw AIS has null timestamps: {path}")
    if not relevant["UTC_10min"].eq(relevant.UTC.dt.floor("10min")).all():
        raise ValueError(f"raw AIS UTC_10min violates the frozen floor rule: {path}")
    coordinates = relevant[["lat", "lon"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"relevant raw AIS has non-finite coordinates: {path}")

    groups: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in relevant.groupby(["vessel_uid", "mmsi"], sort=False):
        api, mmsi = (str(key[0]), str(key[1]))
        group = group.sort_values(["UTC", "UTC_10min"], kind="stable").reset_index(drop=True)
        if group.duplicated("UTC").any() or group.duplicated("UTC_10min").any():
            raise ValueError(f"relevant raw AIS has duplicate time bins for {(api, mmsi)}")
        groups[(api, mmsi)] = group
    return groups, {
        "raw_rows": raw_rows,
        "relevant_raw_rows": int(len(relevant)),
        "identity_mismatch_rows": identity_mismatch_rows,
    }


def _write_table_chunk(
    frame: pd.DataFrame,
    path: Path,
    writer: pq.ParquetWriter | None,
    schema: pa.Schema | None,
) -> tuple[pq.ParquetWriter, pa.Schema]:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        schema = table.schema
    else:
        assert schema is not None
        if table.schema != schema:
            if table.schema.names != schema.names or [str(t) for t in table.schema.types] != [str(t) for t in schema.types]:
                raise ValueError(f"inconsistent parquet schema while writing {path}")
            table = table.replace_schema_metadata(schema.metadata)
    writer.write_table(table)
    return writer, schema


def _flush_pending(
    pending_tracks: list[pd.DataFrame],
    pending_summaries: list[dict[str, Any]],
    track_path: Path,
    summary_path: Path,
    track_writer: pq.ParquetWriter | None,
    track_schema: pa.Schema | None,
    summary_writer: pq.ParquetWriter | None,
    summary_schema: pa.Schema | None,
) -> tuple[pq.ParquetWriter | None, pa.Schema | None, pq.ParquetWriter | None, pa.Schema | None]:
    if pending_tracks:
        track_chunk = pd.concat(pending_tracks, ignore_index=True)
        track_writer, track_schema = _write_table_chunk(
            track_chunk, track_path, track_writer, track_schema
        )
    if pending_summaries:
        summary_chunk = pd.DataFrame.from_records(pending_summaries)
        summary_writer, summary_schema = _write_table_chunk(
            summary_chunk, summary_path, summary_writer, summary_schema
        )
    pending_tracks.clear()
    pending_summaries.clear()
    return track_writer, track_schema, summary_writer, summary_schema


def work(task: tuple[dict[str, Any], str, list[str] | None, str]) -> dict[str, Any]:
    record, output_root, selected_ids, endpoint_policy = task
    shard = str(record["shard"])
    out = Path(output_root) / "shards" / shard
    if out.exists():
        raise RuntimeError(f"output exists: {out}")
    out.mkdir(parents=True)
    started = time.monotonic()
    track_path = out / "ten_minute_complete_candidate.parquet"
    summary_path = out / "candidate_summary.parquet"
    try:
        candidate_path = Path(record["candidate_path"])
        gap_path = Path(record["gap_path"])
        raw_path = Path(record["cleaned_ais_path"])
        base_patch_path = Path(record["base_patch_path"])
        endpoint_patch_path = Path(record["endpoint_patch_path"])
        _check_size(candidate_path, 0, "prepared candidate file")
        _check_size(gap_path, 0, "prepared gap file")
        _check_size(raw_path, int(record["cleaned_ais_bytes"]), "cleaned AIS source")
        _check_size(base_patch_path, int(record["base_patch_bytes"]), "4--24h patch")
        _check_size(endpoint_patch_path, int(record["endpoint_patch_bytes"]), "24--48h patch")
        if sha256(candidate_path) != record["candidate_sha256"]:
            raise ValueError("prepared candidate file hash changed")
        if sha256(gap_path) != record["gap_sha256"]:
            raise ValueError("prepared gap file hash changed")
        candidates = pq.read_table(candidate_path).to_pandas()
        gaps = pq.read_table(gap_path).to_pandas()
        _string_columns(candidates, ["candidate_id", "mmsi", "vessel_uid_clean", "layer", "source_shard"])
        _string_columns(gaps, ["gap_id", "candidate_id", "mmsi", "vessel_uid_clean", "source_shard", "gap_class"])
        all_candidates = candidates.copy()
        all_gaps = gaps.copy()
        if selected_ids is not None:
            wanted = set(selected_ids)
            candidates = candidates.loc[candidates.candidate_id.astype(str).isin(wanted)].copy()
            if set(candidates.candidate_id.astype(str)) != wanted:
                raise ValueError(f"probe candidate identity is incomplete in {shard}")
            gaps = gaps.loc[gaps.candidate_id.astype(str).isin(wanted)].copy()
        candidates = candidates.sort_values("candidate_id", kind="stable").reset_index(drop=True)
        gaps = gaps.sort_values(["candidate_id", "gap_start", "gap_end", "gap_id"], kind="stable").reset_index(drop=True)

        raw_groups, raw_status = _read_raw_groups(raw_path, candidates)
        base_patch = _read_patch(
            base_patch_path,
            all_candidates,
            all_gaps,
            {"unresolved", "gfw_anchor", "gfw_supported_interpolation"},
            "4--24h patch",
        )
        endpoint_patch = _read_patch(
            endpoint_patch_path,
            all_candidates,
            all_gaps,
            {"unresolved", ENDPOINT_LABEL},
            "24--48h endpoint patch",
        )
        if endpoint_policy == "unresolved":
            # The Contract keeps every >24 h internal gap unknown. Retain the
            # hash-bound endpoint candidate as evidence, but do not apply it to
            # the contract-compliant 2025 baseline.
            endpoint_patch = endpoint_patch.iloc[0:0].copy()
        if selected_ids is not None:
            base_patch = base_patch.loc[base_patch.candidate_id.astype(str).isin(wanted)].copy()
            endpoint_patch = endpoint_patch.loc[
                endpoint_patch.candidate_id.astype(str).isin(wanted)
            ].copy()

        empty_gaps = gaps.iloc[0:0].copy()
        empty_base = base_patch.iloc[0:0].copy()
        empty_endpoint = endpoint_patch.iloc[0:0].copy()
        gaps_by_candidate = {
            str(key): group
            for key, group in gaps.groupby("candidate_id", sort=False)
        }
        base_by_candidate = {
            str(key): group
            for key, group in base_patch.groupby("candidate_id", sort=False)
        }
        endpoint_by_candidate = {
            str(key): group
            for key, group in endpoint_patch.groupby("candidate_id", sort=False)
        }

        pending_tracks: list[pd.DataFrame] = []
        pending_summaries: list[dict[str, Any]] = []
        track_writer: pq.ParquetWriter | None = None
        track_schema: pa.Schema | None = None
        summary_writer: pq.ParquetWriter | None = None
        summary_schema: pa.Schema | None = None
        totals: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        completed_candidates = 0

        for candidate in candidates.itertuples(index=False):
            candidate_id = str(candidate.candidate_id)
            points = raw_groups.get((str(candidate.vessel_uid_clean), str(candidate.mmsi)))
            if points is None:
                points = pd.DataFrame(columns=RAW_COLUMNS)
            frame, summary = build_full_candidate_grid(
                candidate,
                points,
                gaps_by_candidate.get(candidate_id, empty_gaps),
                base_by_candidate.get(candidate_id, empty_base),
                endpoint_by_candidate.get(candidate_id, empty_endpoint),
            )
            expected_rows = int(candidate.expected_grid_rows)
            if summary["grid_rows"] != expected_rows or len(frame) != expected_rows:
                raise ValueError(f"grid closure failed for {candidate_id}")
            validate_candidate_frame(frame)
            summary.update(
                {
                    "expected_grid_rows": expected_rows,
                    "window_hours": float(candidate.window_hours),
                    "origin_city": candidate.origin_city,
                    "dest_city": candidate.dest_city,
                    "shiptype_group": candidate.shiptype_group,
                    "has_gfw_gap": bool(candidate.has_gfw_gap),
                    "has_gt24_gap": bool(candidate.has_gt24_gap),
                    "has_endpoint_unknown": bool(candidate.has_endpoint_unknown),
                }
            )
            pending_tracks.append(frame)
            pending_summaries.append(summary)
            completed_candidates += 1
            counts = frame.track_evidence.astype(str).value_counts()
            for label, count in counts.items():
                label_counts[str(label)] += int(count)
            for key in (
                "grid_rows",
                "observed_ais_rows",
                "ais_interpolated_le4h_rows",
                "gfw_anchor_rows",
                "gfw_supported_interpolation_rows",
                "endpoint_24_48h_rows",
                "unresolved_rows",
                "source_gap_grid_rows_assigned",
                "base_valid_patch_rows",
                "base_applied_rows",
                "base_skipped_existing_rows",
                "endpoint_valid_patch_rows",
                "endpoint_applied_rows",
                "endpoint_skipped_existing_rows",
            ):
                totals[key] += int(summary[key])
            if len(pending_tracks) >= 256:
                (
                    track_writer,
                    track_schema,
                    summary_writer,
                    summary_schema,
                ) = _flush_pending(
                    pending_tracks,
                    pending_summaries,
                    track_path,
                    summary_path,
                    track_writer,
                    track_schema,
                    summary_writer,
                    summary_schema,
                )
            if completed_candidates % 500 == 0:
                write_json(
                    out / "status.json",
                    {
                        "state": "assembling",
                        "shard": shard,
                        "completed_candidates": completed_candidates,
                        "total_candidates": len(candidates),
                        "grid_rows_written": int(totals["grid_rows"]),
                    },
                )

        (
            track_writer,
            track_schema,
            summary_writer,
            summary_schema,
        ) = _flush_pending(
            pending_tracks,
            pending_summaries,
            track_path,
            summary_path,
            track_writer,
            track_schema,
            summary_writer,
            summary_schema,
        )
        if track_writer is None or summary_writer is None:
            raise ValueError(f"no output rows were assembled for {shard}")
        track_writer.close()
        summary_writer.close()

        expected_grid_rows = int(candidates.expected_grid_rows.sum())
        if totals["grid_rows"] != expected_grid_rows:
            raise ValueError(f"shard grid total does not close for {shard}")
        result: dict[str, Any] = {
            "state": "completed",
            "shard": shard,
            "candidates": int(len(candidates)),
            "expected_grid_rows": expected_grid_rows,
            "grid_rows": int(totals["grid_rows"]),
            **raw_status,
            **{key: int(totals[key]) for key in TOTAL_KEYS if key in totals},
            "label_counts": dict(sorted(label_counts.items())),
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "outputs": {
                track_path.name: sha256(track_path),
                summary_path.name: sha256(summary_path),
            },
        }
        write_json(out / "status.json", result)
        return result
    except BaseException as exc:
        write_json(
            out / "status.json",
            {"state": "failed", "shard": shard, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


def _parse_shards(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or not set(names).issubset(set(EXPECTED_SHARDS)) or len(set(names)) != len(names):
        raise SystemExit("--only-shards must contain unique shard_0000--shard_0252 names")
    return names


def _verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("candidate_only") is not True:
        raise SystemExit("prepared full candidate input is not a closed candidate package")
    if manifest.get("network_requests") != 0 or manifest.get("products_modified") != 0:
        raise SystemExit("prepared full candidate input is not offline and read-only")
    if manifest.get("expected", {}).get("source_shards") != 253:
        raise SystemExit("prepared package does not cover all 253 source shards")
    for source_name in ("candidate_scope", "candidate_quality", "gap_intervals", "cleaned_ais_manifest"):
        source = manifest["sources"][source_name]
        path = Path(source["path"])
        if sha256(path) != source["sha256"]:
            raise SystemExit(f"prepared source changed: {path}")
    for source_name in ("base_run_state", "endpoint_run_state"):
        source = manifest["sources"][source_name]
        path = Path(source["path"])
        if sha256(path) != source["sha256"]:
            raise SystemExit(f"source run state changed: {path}")
        if json.loads(path.read_text(encoding="utf-8")).get("state") != "completed":
            raise SystemExit(f"source run is no longer completed: {path}")
    if len(manifest.get("shards", [])) != 253:
        raise SystemExit("prepared package does not contain exactly 253 shard records")
    return manifest


def main(args: argparse.Namespace) -> int:
    if not 1 <= args.workers <= 32:
        raise SystemExit("workers must be between 1 and 32")
    prepared = args.prepared_input.resolve()
    manifest_path = prepared / "input_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"prepared manifest is missing: {manifest_path}")
    manifest = _verify_manifest(manifest_path)
    sys.path.insert(0, str(CODING / "pipeline/00_environment_and_config"))
    from ten_minute_review_gates import require_prepared_clearance
    clearance = require_prepared_clearance(CODING, manifest_path)
    only_shards = _parse_shards(args.only_shards)
    records = list(manifest["shards"])
    if args.mode == "probe":
        probe_path = prepared / "probe_scope.parquet"
        probe_record = manifest["probe_scope"]
        if sha256(probe_path) != probe_record["sha256"]:
            raise SystemExit("probe scope changed")
        probe = pq.read_table(probe_path, columns=["candidate_id", "source_shard"]).to_pandas()
        if probe.candidate_id.astype(str).duplicated().any():
            raise SystemExit("probe scope contains duplicate candidate IDs")
        selected_by_shard: dict[str, list[str]] = {
            str(shard): group.candidate_id.astype(str).tolist()
            for shard, group in probe.groupby("source_shard", sort=True)
        }
        records = [record for record in records if record["shard"] in selected_by_shard]
    else:
        selected_by_shard = {}
    if only_shards is not None:
        allowed = set(only_shards)
        records = [record for record in records if record["shard"] in allowed]
    if not records:
        raise SystemExit("no candidate shards selected")

    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    output.mkdir(parents=True)
    (output / "shards").mkdir()
    git = lambda *parts: subprocess.check_output(["git", *parts], cwd=HERE, text=True).strip()
    context: dict[str, Any] = {
        "state": "running",
        "mode": args.mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "workers": args.workers,
        "expected_shards": len(records),
        "selected_shards": [record["shard"] for record in records],
        "prepared_manifest_sha256": sha256(manifest_path),
        "prepared_manifest_path": str(manifest_path),
        "review_clearance": clearance,
        "source_branch": git("branch", "--show-current"),
        "source_commit": git("rev-parse", "HEAD"),
        "network_requests": 0,
        "products_modified": 0,
        "candidate_only": True,
        "endpoint_policy": args.endpoint_policy,
        "output_semantics": (
            manifest["output_semantics"]
            if args.endpoint_policy == "candidate"
            else (
                "contract-compliant complete 2025 ten-minute voyage-grid candidate: "
                "observed AIS plus short-gap interpolation, completed 4--24h GFW "
                "candidate supplement, and explicit unresolved rows; the 24--48h "
                "endpoint candidate is retained upstream but not applied"
            )
        ),
    }
    write_json(output / "run_state.json", context)
    tasks = [
        (
            record,
            str(output),
            None if args.mode == "full" else selected_by_shard[record["shard"]],
            args.endpoint_policy,
        )
        for record in records
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = {pool.submit(work, task): task[0]["shard"] for task in tasks}
        for future in as_completed(futures):
            shard = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append({"shard": shard, "error": f"{type(exc).__name__}: {exc}"})
            write_json(
                output / "run_state.json",
                {
                    **context,
                    "state": "running",
                    "completed_shards": len(results),
                    "failed_shards": len(failures),
                    "remaining_shards": len(tasks) - len(results) - len(failures),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

    if failures:
        write_json(
            output / "run_state.json",
            {
                **context,
                "state": "failed",
                "completed_shards": len(results),
                "failed_shards": len(failures),
                "remaining_shards": len(tasks) - len(results) - len(failures),
                "failures": failures,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise SystemExit(1)

    totals = {key: int(sum(int(result.get(key, 0)) for result in results)) for key in TOTAL_KEYS}
    label_counts: Counter[str] = Counter()
    for result in results:
        label_counts.update({str(k): int(v) for k, v in result.get("label_counts", {}).items()})
    totals["resolved_grid_rows"] = totals["grid_rows"] - totals["unresolved_rows"]
    totals["resolved_grid_pct"] = (
        100.0 * totals["resolved_grid_rows"] / totals["grid_rows"]
        if totals["grid_rows"]
        else 0.0
    )
    final = {
        **context,
        "state": "completed",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "completed_shards": len(results),
        "failed_shards": 0,
        "remaining_shards": 0,
        "totals": totals,
        "label_counts": dict(sorted(label_counts.items())),
        "max_shard_seconds": max(result["elapsed_seconds"] for result in results),
        "max_peak_rss_gib": max(result["peak_rss_gib"] for result in results),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "run_state.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-input", type=Path, required=True)
    parser.add_argument("--mode", choices=["probe", "full"], required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-shards", default=None)
    parser.add_argument(
        "--endpoint-policy",
        choices=["candidate", "unresolved"],
        default="candidate",
        help="apply the 24--48h endpoint candidate or keep those rows unresolved",
    )
    raise SystemExit(main(parser.parse_args()))
