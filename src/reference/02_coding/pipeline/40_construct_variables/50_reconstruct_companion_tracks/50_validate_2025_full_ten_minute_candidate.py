#!/usr/bin/env python3
"""Independently validate a complete 2025 ten-minute candidate run."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
CODING = HERE.parents[2]
EXPECTED_SHARDS = [f"shard_{number:04d}" for number in range(253)]
ALLOWED_EVIDENCE = {
    "observed_ais",
    "ais_interpolated_le4h",
    "gfw_anchor",
    "gfw_supported_interpolation",
    "companion_endpoint_24_48h",
    "unresolved",
}
LONG_GAP_CLASSES = {
    "gfw_candidate_4_24h",
    "unresolved_internal_gt24h",
    "endpoint_or_no_ais_unknown",
}
TRACK_COLUMNS = [
    "candidate_id",
    "mmsi",
    "vessel_uid_clean",
    "layer",
    "utc",
    "lat",
    "lon",
    "track_evidence",
    "position_source",
    "gap_id",
    "gap_class",
    "anchor_source",
    "observation_time_utc",
    "source_provider",
    "source_resolution_original",
    "source_shard",
]


_WORKER_RUN_DIR: Path | None = None
_WORKER_PREPARED: dict[str, dict[str, Any]] | None = None


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


def _load_prepared(prepared: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = prepared / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("candidate_only") is not True:
        raise ValueError("prepared input is not a closed candidate package")
    by_shard: dict[str, dict[str, Any]] = {}
    for record in manifest["shards"]:
        shard = str(record["shard"])
        candidates = pd.read_parquet(record["candidate_path"])
        gaps = pd.read_parquet(record["gap_path"], columns=["gap_id", "candidate_id", "gap_class"])
        if candidates.candidate_id.astype(str).duplicated().any():
            raise ValueError(f"prepared candidates are not unique in {shard}")
        meta = candidates.copy()
        meta["candidate_id"] = meta.candidate_id.astype(str)
        meta["mmsi"] = meta.mmsi.astype(str)
        meta["vessel_uid_clean"] = meta.vessel_uid_clean.astype(str)
        meta["source_shard"] = meta.source_shard.astype(str)
        meta["departure_time"] = pd.to_datetime(meta.departure_time, utc=True)
        meta["arrival_time"] = pd.to_datetime(meta.arrival_time, utc=True)
        if not meta.source_shard.eq(shard).all():
            raise ValueError(f"prepared candidates have the wrong source shard: {shard}")
        gap_ids = gaps.gap_id.astype(str)
        if gap_ids.duplicated().any():
            raise ValueError(f"prepared gaps are not unique in {shard}")
        by_shard[shard] = {
            "record": record,
            "candidates": meta,
            "candidate_map": meta.set_index("candidate_id").to_dict("index"),
            "gap_map": gaps.assign(
                gap_id=gap_ids,
                candidate_id=gaps.candidate_id.astype(str),
                gap_class=gaps.gap_class.astype(str),
            ).set_index("gap_id").to_dict("index"),
        }
    if sorted(by_shard) != EXPECTED_SHARDS:
        raise ValueError("prepared package does not contain exactly 253 shards")
    return manifest, by_shard


def _batch_validation(
    batch: pd.DataFrame,
    shard: str,
    candidate_map: dict[str, dict[str, Any]],
    gap_map: dict[str, dict[str, Any]],
    previous: tuple[str, int] | None,
) -> tuple[dict[str, int], tuple[str, int] | None, list[tuple[str, int]]]:
    missing = sorted(set(TRACK_COLUMNS).difference(batch.columns))
    if missing:
        raise ValueError(f"{shard} output is missing columns: {missing}")
    batch["candidate_id"] = batch.candidate_id.astype(str)
    batch["mmsi"] = batch.mmsi.astype(str)
    batch["vessel_uid_clean"] = batch.vessel_uid_clean.astype(str)
    batch["layer"] = batch.layer.astype(str)
    batch["source_shard"] = batch.source_shard.astype(str)
    batch["track_evidence"] = batch.track_evidence.astype(str)
    batch["gap_class"] = batch.gap_class.astype("string")
    batch["gap_id"] = batch.gap_id.astype("string")
    batch["utc"] = pd.to_datetime(batch.utc, utc=True, errors="raise")
    batch["observation_time_utc"] = pd.to_datetime(
        batch.observation_time_utc, utc=True, errors="raise"
    )
    ids = batch.candidate_id.to_numpy(dtype=str)
    if not set(ids).issubset(candidate_map):
        raise ValueError(f"{shard} output contains an unknown candidate")
    if not batch.source_shard.eq(shard).all():
        raise ValueError(f"{shard} output contains a wrong source_shard value")
    if not set(batch.track_evidence).issubset(ALLOWED_EVIDENCE):
        raise ValueError(f"{shard} output contains an unexpected evidence label")
    utc_ns = batch.utc.astype("int64").to_numpy(dtype=np.int64)
    if len(batch) and not batch.utc.eq(batch.utc.dt.floor("10min")).all():
        raise ValueError(f"{shard} output contains off-grid UTC values")
    if len(ids) > 1:
        if np.any(ids[1:] < ids[:-1]):
            raise ValueError(f"{shard} output is not sorted by candidate_id")
        same = ids[1:] == ids[:-1]
        if np.any(utc_ns[1:][same] <= utc_ns[:-1][same]):
            raise ValueError(f"{shard} output contains duplicate or unsorted candidate/time keys")
    if previous is not None and len(batch):
        first = (str(ids[0]), int(utc_ns[0]))
        if first[0] < previous[0] or (first[0] == previous[0] and first[1] <= previous[1]):
            raise ValueError(f"{shard} output is not globally sorted across row groups")

    start_ns = np.asarray([candidate_map[cid]["departure_time"].value for cid in ids], dtype=np.int64)
    end_ns = np.asarray([candidate_map[cid]["arrival_time"].value for cid in ids], dtype=np.int64)
    if np.any(utc_ns <= start_ns) or np.any(utc_ns >= end_ns):
        raise ValueError(f"{shard} output contains a UTC outside its voyage window")
    expected_mmsi = np.asarray([str(candidate_map[cid]["mmsi"]) for cid in ids], dtype=object)
    expected_api = np.asarray([str(candidate_map[cid]["vessel_uid_clean"]) for cid in ids], dtype=object)
    if not np.array_equal(batch.mmsi.to_numpy(dtype=object), expected_mmsi):
        raise ValueError(f"{shard} output disagrees with candidate MMSI")
    if not np.array_equal(batch.vessel_uid_clean.to_numpy(dtype=object), expected_api):
        raise ValueError(f"{shard} output disagrees with candidate API identity")
    expected_layer = np.asarray([str(candidate_map[cid]["layer"]) for cid in ids], dtype=object)
    if not np.array_equal(batch.layer.to_numpy(dtype=object), expected_layer):
        raise ValueError(f"{shard} output disagrees with candidate layer")

    evidence = batch.track_evidence.to_numpy(dtype=str)
    coordinates = batch[["lat", "lon"]].to_numpy(dtype=float)
    resolved = evidence != "unresolved"
    if resolved.any() and not np.isfinite(coordinates[resolved]).all():
        raise ValueError(f"{shard} has a resolved row without finite coordinates")
    if (~resolved).any() and np.isfinite(coordinates[~resolved]).any():
        raise ValueError(f"{shard} has an unresolved row with coordinates")
    if np.any(resolved & ((coordinates[:, 0] < -90) | (coordinates[:, 0] > 90) | (coordinates[:, 1] < -180) | (coordinates[:, 1] > 180))):
        raise ValueError(f"{shard} has coordinates outside geographic bounds")

    observation = batch.observation_time_utc
    if np.any((evidence == "observed_ais") & observation.isna().to_numpy()):
        raise ValueError(f"{shard} has observed AIS rows without actual observation time")
    if np.any((evidence != "observed_ais") & observation.notna().to_numpy()):
        raise ValueError(f"{shard} has a non-observed row with actual observation time")
    if observation.notna().any():
        observation_ns = observation.astype("int64").to_numpy(dtype=np.int64)
        observed = evidence == "observed_ais"
        if np.any(observation_ns[observed] < start_ns[observed]) or np.any(observation_ns[observed] > end_ns[observed]):
            raise ValueError(f"{shard} has an actual AIS observation outside its voyage window")
    nonobserved = evidence != "observed_ais"
    provider_present = batch.source_provider.notna().to_numpy()
    resolution_present = batch.source_resolution_original.notna().to_numpy()
    if np.any(nonobserved & (provider_present | resolution_present)):
        raise ValueError(f"{shard} carries raw AIS provenance on a non-observed row")

    position_source = batch.position_source.astype("string").to_numpy(dtype=object)
    expected_position_sources = {
        "observed_ais": "observed_ais",
        "ais_interpolated_le4h": "ais_linear_between_fixes_le4h",
        "gfw_anchor": "gfw_anchor",
        "gfw_supported_interpolation": "gfw_sea_path_interpolation",
        "companion_endpoint_24_48h": "ais_endpoint_sea_path_24_48h",
        "unresolved": None,
    }
    for index, label in enumerate(evidence):
        expected_source = expected_position_sources[label]
        if expected_source is None:
            if not bool(pd.isna(position_source[index])):
                raise ValueError(f"{shard} unresolved row carries a position source")
        elif bool(pd.isna(position_source[index])) or str(position_source[index]) != expected_source:
            raise ValueError(f"{shard} evidence and position source disagree")
    anchor_present = batch.anchor_source.notna().to_numpy()
    patch_labels = np.isin(
        evidence,
        ["gfw_anchor", "gfw_supported_interpolation", "companion_endpoint_24_48h"],
    )
    if np.any(patch_labels & ~anchor_present) or np.any(~patch_labels & anchor_present):
        raise ValueError(f"{shard} anchor provenance does not match patch evidence")

    gap_class = batch.gap_class.astype("string").to_numpy(dtype=object)
    gap_id = batch.gap_id.astype("string").to_numpy(dtype=object)
    for index, label in enumerate(evidence):
        gap_id_missing = bool(pd.isna(gap_id[index]))
        gap_class_missing = bool(pd.isna(gap_class[index]))
        if label == "observed_ais":
            if not gap_id_missing or not gap_class_missing:
                raise ValueError(f"{shard} observed row carries a source gap")
        elif label == "ais_interpolated_le4h":
            if gap_class_missing or gap_class[index] != "ais_supported_le4h" or gap_id_missing:
                raise ValueError(f"{shard} short interpolation lacks its source gap ID")
        elif label in {"gfw_anchor", "gfw_supported_interpolation"}:
            if gap_class_missing or gap_class[index] != "gfw_candidate_4_24h" or gap_id_missing:
                raise ValueError(f"{shard} GFW row lacks its 4--24h source gap")
            if str(gap_id[index]) not in gap_map:
                raise ValueError(f"{shard} GFW row references an unknown source gap")
        elif label == "companion_endpoint_24_48h":
            if gap_class_missing or gap_class[index] != "unresolved_internal_gt24h" or gap_id_missing:
                raise ValueError(f"{shard} endpoint row lacks its >24h source gap")
            if str(gap_id[index]) not in gap_map:
                raise ValueError(f"{shard} endpoint row references an unknown source gap")
        elif label == "unresolved":
            if gap_id_missing or gap_class_missing or gap_class[index] not in LONG_GAP_CLASSES:
                raise ValueError(f"{shard} unresolved row lacks its source gap classification")
            if str(gap_id[index]) not in gap_map:
                raise ValueError(f"{shard} unresolved row references an unknown source gap")

    label_counts = Counter(evidence.tolist())
    long_gap_rows = int(np.isin(gap_class.astype(str), list(LONG_GAP_CLASSES)).sum())
    last = previous
    if len(batch):
        last = (str(ids[-1]), int(utc_ns[-1]))
    run_lengths: list[tuple[str, int]] = []
    if len(ids):
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        ends = np.r_[starts[1:], len(ids)]
        run_lengths = [(str(ids[left]), int(right - left)) for left, right in zip(starts, ends)]
    return {
        "rows": int(len(batch)),
        "long_gap_rows": long_gap_rows,
        **{f"label_{key}": int(value) for key, value in label_counts.items()},
    }, last, run_lengths


def _validate_shard(shard: str, run_dir: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Validate one shard without relying on any other shard's output."""
    record = info["record"]
    candidates = info["candidates"]
    candidate_map = info["candidate_map"]
    out = run_dir / "shards" / shard
    status_path = out / "status.json"
    track_path = out / "ten_minute_complete_candidate.parquet"
    summary_path = out / "candidate_summary.parquet"
    if not status_path.is_file() or not track_path.is_file() or not summary_path.is_file():
        raise ValueError(f"{shard} is missing an output or status file")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"{shard} status is not completed")
    if status.get("outputs", {}).get(track_path.name) != sha256(track_path):
        raise ValueError(f"{shard} track output hash does not match status")
    if status.get("outputs", {}).get(summary_path.name) != sha256(summary_path):
        raise ValueError(f"{shard} summary output hash does not match status")

    summary = pd.read_parquet(summary_path)
    summary["candidate_id"] = summary.candidate_id.astype(str)
    if summary.candidate_id.duplicated().any() or set(summary.candidate_id) != set(candidate_map):
        raise ValueError(f"{shard} candidate summary does not match prepared candidates")
    summary = summary.set_index("candidate_id").loc[list(candidate_map)]
    expected_rows = candidates.set_index("candidate_id").loc[list(candidate_map)].expected_grid_rows.astype(int)
    if not summary.expected_grid_rows.astype(int).eq(expected_rows).all():
        raise ValueError(f"{shard} candidate summary expected rows changed")

    previous: tuple[str, int] | None = None
    actual_candidate_rows: Counter[str] = Counter()
    shard_totals: Counter[str] = Counter()
    shard_label_counts: Counter[str] = Counter()
    parquet_file = pq.ParquetFile(track_path)
    for record_batch in parquet_file.iter_batches(batch_size=250_000, columns=TRACK_COLUMNS):
        batch = record_batch.to_pandas()
        batch_counts, previous, runs = _batch_validation(
            batch, shard, candidate_map, info["gap_map"], previous
        )
        shard_totals.update(batch_counts)
        batch_labels = {
            key.removeprefix("label_"): value
            for key, value in batch_counts.items()
            if key.startswith("label_")
        }
        shard_label_counts.update(batch_labels)
        for candidate_id, rows in runs:
            actual_candidate_rows[candidate_id] += rows
    positive_candidate_ids = {
        key for key, value in expected_rows.items() if int(value) > 0
    }
    if set(actual_candidate_rows) != positive_candidate_ids:
        missing = sorted(positive_candidate_ids - set(actual_candidate_rows))[:3]
        raise ValueError(f"{shard} output is missing candidate rows: {missing}")
    expected_by_candidate = {
        key: int(value)
        for key, value in summary.grid_rows.astype(int).to_dict().items()
        if int(value) > 0
    }
    if actual_candidate_rows != Counter({key: int(value) for key, value in expected_by_candidate.items()}):
        raise ValueError(f"{shard} per-candidate grid rows do not match candidate summary")
    summary_label_columns = {
        "observed_ais": "observed_ais_rows",
        "ais_interpolated_le4h": "ais_interpolated_le4h_rows",
        "gfw_anchor": "gfw_anchor_rows",
        "gfw_supported_interpolation": "gfw_supported_interpolation_rows",
        "companion_endpoint_24_48h": "endpoint_24_48h_rows",
        "unresolved": "unresolved_rows",
    }
    for label, column in summary_label_columns.items():
        if int(summary[column].sum()) != shard_label_counts[label]:
            raise ValueError(f"{shard} label count does not match candidate summaries: {label}")
    if int(summary.source_gap_grid_rows_assigned.sum()) != shard_totals["long_gap_rows"]:
        raise ValueError(f"{shard} source-gap assignment count does not match output")
    if int(summary.base_applied_rows.sum()) != shard_label_counts["gfw_anchor"] + shard_label_counts["gfw_supported_interpolation"]:
        raise ValueError(f"{shard} applied 4--24h patch rows do not match output labels")
    if int(summary.endpoint_applied_rows.sum()) != shard_label_counts["companion_endpoint_24_48h"]:
        raise ValueError(f"{shard} applied endpoint rows do not match output labels")
    return {
        "shard": shard,
        "candidate_count": len(candidate_map),
        "totals": dict(shard_totals),
        "label_counts": dict(shard_label_counts),
        "output_hashes": {
            track_path.name: sha256(track_path),
            summary_path.name: sha256(summary_path),
        },
    }


def _validate_shard_worker(shard: str) -> dict[str, Any]:
    if _WORKER_RUN_DIR is None or _WORKER_PREPARED is None:
        raise RuntimeError("validation worker was not initialized")
    return _validate_shard(shard, _WORKER_RUN_DIR, _WORKER_PREPARED[shard])


def validate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "run_state.json"
    if not state_path.is_file():
        raise ValueError(f"run state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("state") != "completed":
        raise ValueError("candidate run is not completed")
    if state.get("candidate_only") is not True or state.get("network_requests") != 0 or state.get("products_modified") != 0:
        raise ValueError("candidate run is not an offline candidate-only run")
    manifest, prepared = _load_prepared(args.prepared_input.resolve())
    selected = list(state.get("selected_shards", []))
    if sorted(selected) != selected or not set(selected).issubset(set(EXPECTED_SHARDS)):
        raise ValueError("run state has an invalid selected-shard list")
    if args.require_full and selected != EXPECTED_SHARDS:
        raise ValueError("full validation requires all 253 selected shards")
    if state.get("prepared_manifest_sha256") != sha256(args.prepared_input.resolve() / "input_manifest.json"):
        raise ValueError("run state points to a different prepared input")
    if state.get("mode") == "probe":
        probe_path = Path(manifest["probe_scope"]["path"])
        probe = pd.read_parquet(probe_path, columns=["candidate_id", "source_shard"])
        probe_ids_by_shard = {
            str(shard): set(group.candidate_id.astype(str))
            for shard, group in probe.groupby("source_shard", sort=True)
        }
        for shard in selected:
            wanted = probe_ids_by_shard.get(shard, set())
            info = prepared[shard]
            candidates = info["candidates"]
            candidates = candidates.loc[candidates.candidate_id.astype(str).isin(wanted)].copy()
            if set(candidates.candidate_id.astype(str)) != wanted:
                raise ValueError(f"probe candidate scope does not match prepared candidates: {shard}")
            info["candidates"] = candidates
            info["candidate_map"] = candidates.set_index("candidate_id").to_dict("index")

    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    global _WORKER_RUN_DIR, _WORKER_PREPARED
    _WORKER_RUN_DIR = run_dir
    _WORKER_PREPARED = prepared
    shard_results: list[dict[str, Any]] = []
    if args.workers == 1:
        shard_results = [_validate_shard(shard, run_dir, prepared[shard]) for shard in selected]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise ValueError("parallel validation requires the fork start method on this host")
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = {
                executor.submit(_validate_shard_worker, shard): shard
                for shard in selected
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                shard = futures[future]
                result = future.result()
                shard_results.append(result)
                print(
                    json.dumps(
                        {"validated_shards": completed, "total_shards": len(selected), "shard": shard},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    totals: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    output_hashes: dict[str, dict[str, str]] = {}
    candidate_seen = 0
    for result in shard_results:
        totals.update(result["totals"])
        label_counts.update(result["label_counts"])
        output_hashes[result["shard"]] = result["output_hashes"]
        candidate_seen += int(result["candidate_count"])

    totals["resolved_grid_rows"] = totals["rows"] - totals["label_unresolved"]
    totals["candidates"] = candidate_seen
    totals["grid_rows"] = totals["rows"]
    totals["expected_grid_rows"] = int(
        sum(int(prepared[shard]["candidates"].expected_grid_rows.sum()) for shard in selected)
    )
    totals["resolved_grid_pct"] = (
        100.0 * totals["resolved_grid_rows"] / totals["rows"] if totals["rows"] else 0.0
    )
    expected_state_totals = state.get("totals", {})
    state_checks = {
        key: bool(expected_state_totals.get(key) == totals[key])
        for key in ("candidates", "expected_grid_rows", "grid_rows")
    }
    state_checks["grid_rows_matches_independent_rows"] = bool(
        expected_state_totals.get("grid_rows") == totals["rows"]
    )
    if not all(state_checks.values()):
        raise ValueError(f"run state totals disagree with independent validation: {state_checks}")

    expected_labels = state.get("label_counts", {})
    if {key: int(value) for key, value in expected_labels.items()} != dict(sorted(label_counts.items())):
        raise ValueError("run state label totals disagree with independent validation")
    checks = {
        "state_completed": True,
        "all_selected_shards_present": True,
        "per_candidate_grid_closure": True,
        "unique_candidate_time_keys_and_sorted_grid": True,
        "strict_voyage_window": True,
        "observed_ais_provenance_and_no_overwrite_shape": True,
        "resolved_rows_have_coordinates": True,
        "unresolved_rows_have_no_coordinates": True,
        "gap_labels_align_with_source_gap_classes": True,
        "state_totals_match_independent_counts": True,
        "state_label_totals_match_independent_counts": True,
    }
    result = {
        "status": "pass",
        "operation": "validate_complete_2025_ten_minute_candidate",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "prepared_manifest_sha256": sha256(args.prepared_input.resolve() / "input_manifest.json"),
        "selected_shards": selected,
        "checks": checks,
        "totals": dict(totals),
        "label_counts": dict(sorted(label_counts.items())),
        "output_hashes": output_hashes,
        "candidate_only": True,
        "formal_product_adoption": False,
        "open_promotion_issue": (
            "companion_endpoint_24_48h remains a candidate-only evidence label"
            if totals["label_companion_endpoint_24_48h"]
            else None
        ),
        "endpoint_policy": state.get("endpoint_policy", "candidate"),
    }
    return result


def main(args: argparse.Namespace) -> int:
    validation_path = args.run_dir.resolve() / "validation.json"
    try:
        result = validate(args)
    except BaseException as exc:
        write_json(
            validation_path,
            {
                "status": "fail",
                "validated_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    write_json(validation_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-input", type=Path, required=True)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    raise SystemExit(main(parser.parse_args()))
