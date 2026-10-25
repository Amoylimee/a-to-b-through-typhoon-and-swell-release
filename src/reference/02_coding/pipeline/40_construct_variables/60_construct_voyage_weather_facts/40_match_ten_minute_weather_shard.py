#!/usr/bin/env python3
"""Match one contract-compliant 2025 ten-minute shard to wave and cyclone weather."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


HERE = Path(__file__).resolve().parent
CODING = HERE.parents[2]
STAGE = HERE
sys.path.insert(0, str(STAGE))

from weather_core.era5_match import Era5MonthlyMatcher  # noqa: E402
from weather_core.ten_minute_track_adapter import build_sampling_positions  # noqa: E402
from weather_core.track_adapter import mask_uncertified_tc, verify_tc_calendar_audit  # noqa: E402
from weather_core.tropical_cyclone import CycloneTrackIndex, load_ibtracs_snapshot  # noqa: E402


TRACK_COLUMNS = [
    "candidate_id", "mmsi", "vessel_uid_clean", "utc", "lat", "lon",
    "track_evidence", "position_source", "gap_id", "gap_class", "anchor_source",
]
SUMMARY_COLUMNS = [
    "candidate_id", "mmsi", "vessel_uid_clean", "departure_time", "arrival_time",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    baseline = args.baseline_run.resolve()
    output = args.output.resolve()
    config_path = args.config.resolve()
    sys.path.insert(0, str(CODING / "pipeline/00_environment_and_config"))
    from ten_minute_review_gates import require_weather_clearance
    weather_config = json.loads(config_path.read_text(encoding="utf-8"))
    clearance = require_weather_clearance(CODING, baseline, weather_config["year"],
                                          getattr(args, "decision", None),
                                          getattr(args, "prepared_input", None))
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("weather shard output must stay under 02_coding/.runs")
    state_path = baseline / "run_state.json"
    validation_path = baseline / "validation.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if state.get("state") != "completed" or validation.get("status") != "pass":
        raise ValueError("weather requires a completed, independently validated track baseline")
    if state.get("endpoint_policy") != "unresolved":
        raise ValueError("weather requires the Contract baseline with >24 h gaps unresolved")
    if validation.get("endpoint_policy") != "unresolved" or validation.get("open_promotion_issue") is not None:
        raise ValueError("track validation has an unresolved endpoint-policy issue")
    if args.shard not in state.get("selected_shards", []):
        raise ValueError(f"baseline does not contain {args.shard}")

    shard_root = baseline / "shards" / args.shard
    shard_status_path = shard_root / "status.json"
    track_path = shard_root / "ten_minute_complete_candidate.parquet"
    summary_path = shard_root / "candidate_summary.parquet"
    shard_status = json.loads(shard_status_path.read_text(encoding="utf-8"))
    for path in (track_path, summary_path):
        if shard_status.get("outputs", {}).get(path.name) != sha256(path):
            raise ValueError(f"baseline shard hash mismatch: {path}")

    return match_validated_weather_inputs(args, clearance, {
        "baseline_state": file_record(state_path),
        "baseline_validation": file_record(validation_path),
        "shard_status": file_record(shard_status_path),
        "track": file_record(track_path),
        "summary": file_record(summary_path),
    }, started=started)


def match_validated_weather_inputs(args: argparse.Namespace, clearance: dict,
                                   sources: dict, *, started: float | None = None) -> dict:
    """Reuse the frozen matcher after an entrypoint has checked its own scope.

    E001 keeps its original CP1/CP2/WX1 checks in run(). E006 binds its distinct
    GFW baseline and explicit step-5 authorization before calling this engine.
    The engine has no command-line route that bypasses those entrypoint checks.
    """
    started = time.perf_counter() if started is None else started
    output = args.output.resolve()
    config_path = args.config.resolve()
    track_path = Path(sources["track"]["path"])
    summary_path = Path(sources["summary"]["path"])
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("weather shard output must stay under 02_coding/.runs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = {
        "year": 2025,
        "maximum_wet_cell_snap_km": 117.94,
        "maximum_tc_fix_gap_hours": 3.0,
        "tc_radius_km": [300.0, 500.0],
        "ibtracs_best_track_final": False,
    }
    for key, expected in frozen.items():
        if config.get(key) != expected:
            raise ValueError(f"protected weather parameter mismatch: {key}")
    tc_calendar_audit = verify_tc_calendar_audit(config)

    voyages = pd.read_parquet(summary_path, columns=SUMMARY_COLUMNS).sort_values("candidate_id", kind="stable")
    if args.max_candidates is not None:
        if args.max_candidates < 1:
            raise ValueError("--max-candidates must be positive")
        voyages = voyages.head(args.max_candidates).copy()
    wanted = set(voyages.candidate_id.astype(str))
    track = pd.read_parquet(track_path, columns=TRACK_COLUMNS,
                            filters=[("candidate_id", "in", sorted(wanted))])
    track = track.loc[track.candidate_id.astype(str).isin(wanted)].copy()
    samples = build_sampling_positions(voyages, track)
    del track

    output.mkdir(parents=True)
    audit: dict[str, object] = {
        "state": "running",
        "year": 2025,
        "shard": args.shard,
        "mode": "probe" if args.max_candidates is not None else "full",
        "max_candidates": args.max_candidates,
        "candidate_only": True,
        "review_clearance": clearance,
        "network_requests": 0,
        "products_modified": 0,
        "source_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=HERE, text=True).strip(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "sources": {**sources, "config": file_record(config_path)},
        "code_sha256": {
            str(path.relative_to(CODING)): sha256(path)
            for path in [Path(__file__).resolve(), STAGE / "weather_core/ten_minute_track_adapter.py"]
        },
        "months": [],
        "candidates": int(len(voyages)),
        "rows": int(len(samples)),
        "voyage_hours": float(samples.segment_weight_hours.sum()),
        "track_hours_by_evidence": {
            str(key): float(value)
            for key, value in samples.groupby("track_evidence").segment_weight_hours.sum().items()
        },
    }
    write_json(output / "status.json", audit)

    tc_path = Path(config["ibtracs_path"])
    fixes, tc_audit = load_ibtracs_snapshot(
        tc_path,
        expected_sha256=config["ibtracs_sha256"],
        time_end_exclusive="2026-01-02T00:00:00Z",
    )
    cyclone = CycloneTrackIndex(fixes, maximum_fix_gap_hours=config["maximum_tc_fix_gap_hours"])
    audit["sources"]["ibtracs"] = file_record(tc_path)
    audit["tc_snapshot_audit"] = tc_audit
    audit["tc_calendar_audit"] = tc_calendar_audit["new"] if tc_calendar_audit else None

    summaries: list[pd.DataFrame] = []
    for month, part in samples.groupby(samples.hour_anchor_utc.dt.strftime("%Y-%m"), sort=True):
        month_started = time.perf_counter()
        part = part.reset_index(drop=True)
        wave_path = Path(config["wave_directory"]) / config["wave_filename_template"].format(month=int(month[-2:]))
        eligible = month.startswith("2025-") and wave_path.is_file()
        preload_bytes = 0
        if eligible:
            with xr.open_dataset(wave_path) as dataset:
                if dataset.swh.attrs.get("units") != "m":
                    raise ValueError("wave units are not metres")
            with Era5MonthlyMatcher(
                [wave_path], maximum_wet_cell_snap_km=config["maximum_wet_cell_snap_km"], cached_hours=2
            ) as matcher:
                preload_bytes = matcher.preload_swh()
                part = matcher.match(part)
            part = part.rename(
                columns={
                    "match_status": "wave_match_status",
                    "snap_distance_km": "wave_snap_distance_km",
                    "source_path": "wave_source_path",
                }
            )
            audit["sources"][f"era5_{month}"] = file_record(wave_path)
        else:
            part["swh_m"] = np.nan
            part["matched_lat"] = np.nan
            part["matched_lon"] = np.nan
            part["wave_match_status"] = "weather_time_out_of_snapshot"
            part["wave_snap_distance_km"] = np.nan
            part["wave_source_path"] = pd.NA
        if (part.swh_m.dropna() < 0).any():
            raise ValueError("negative SWH")
        tc_parts = [
            cyclone.match_positions(
                part.iloc[offset : offset + 50_000],
                time_col="position_time_utc",
                lat_col="lat",
                lon_col="lon",
            )
            for offset in range(0, len(part), 50_000)
        ]
        part = pd.concat(tc_parts, ignore_index=True).drop(columns=["_input_index"])
        part = mask_uncertified_tc(
            part, full_year_verified=config["ibtracs_full_year_coverage_verified"]
        )
        month_path = output / f"weather_{month}.parquet"
        part.to_parquet(month_path, index=False, compression="zstd")
        weights = part.segment_weight_hours
        summary = pd.DataFrame(
            {
                "candidate_id": part.candidate_id,
                "voyage_hours": weights,
                "track_hours": weights.where(part.lat.notna(), 0),
                "wave_matched_hours": weights.where(part.swh_m.notna(), 0),
                "wave_integral_m_hours": (weights * part.swh_m).fillna(0),
                "wave_ge4m_matched_hours": weights.where(part.swh_m.ge(4), 0),
                "tc500_positive_snapshot_hours": weights.where(part.tc_within_500km_snapshot.fillna(False), 0),
                "tc300_positive_snapshot_hours": weights.where(part.tc_within_300km_snapshot.fillna(False), 0),
            }
        )
        summaries.append(summary.groupby("candidate_id", as_index=False).sum())
        entry = {
            "month": month,
            "rows": int(len(part)),
            "seconds": time.perf_counter() - month_started,
            "monthly_swh_preload_bytes": int(preload_bytes),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "wave_status_counts": {str(k): int(v) for k, v in part.wave_match_status.value_counts().items()},
            "wave_matched_hours": float(summary.wave_matched_hours.sum()),
        }
        audit["months"].append(entry)
        write_json(output / "status.json", audit)

    combined = pd.concat(summaries).groupby("candidate_id", as_index=False).sum()
    combined["wave_mean_on_matched_hours_m"] = (
        combined.wave_integral_m_hours / combined.wave_matched_hours.replace(0, np.nan)
    )
    combined["unresolved_track_hours"] = combined.voyage_hours - combined.track_hours
    combined["wave_unmatched_hours"] = combined.voyage_hours - combined.wave_matched_hours
    summary_output = output / "voyage_weather_facts.parquet"
    combined.to_parquet(summary_output, index=False)
    outputs = {
        path.name: file_record(path)
        for path in sorted(output.glob("*.parquet"))
    }
    audit.update(
        state="completed",
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        wave_matched_hours=float(combined.wave_matched_hours.sum()),
        outputs=outputs,
    )
    write_json(output / "status.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=STAGE / "config/2025_ten_minute_weather.json")
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--prepared-input", type=Path,
                        help="Actual prepared input_manifest.json for a historical baseline without a clearance receipt")
    args = parser.parse_args()
    output_existed = args.output.exists()
    try:
        result = run(args)
    except BaseException as exc:
        if args.output.exists() and not output_existed:
            status_path = args.output / "status.json"
            prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            prior.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(status_path, prior)
        raise
    print(json.dumps({key: result[key] for key in ["state", "shard", "candidates", "rows", "elapsed_seconds", "peak_rss_gib"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
