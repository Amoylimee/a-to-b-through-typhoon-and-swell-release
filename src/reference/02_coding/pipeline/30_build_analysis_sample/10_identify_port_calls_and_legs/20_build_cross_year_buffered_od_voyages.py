#!/usr/bin/env python3
"""Rebuild annual adjacent OD voyages from accepted visits with a Y+1 buffer.

This entry never reads AIS and never writes accepted product roots.  For each
requested departure year Y it reads accepted Y and Y+1 port visits, proves that
every existing annual voyage ID retains its exact timestamps, and writes the old
rows plus newly observable cross-boundary voyages into an isolated run root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd


CODING_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from cross_year_voyages import (  # noqa: E402
    VOYAGE_OUTPUT_COLUMNS,
    build_buffered_voyages,
)
from vessel_group_helpers import build_size_bin, map_shiptype_group  # noqa: E402


ACCEPTED_PORT_CALL_ROOT = (
    CODING_ROOT / "data/processed/products/port_calls_and_legs"
)
VESSEL_SPECS_TEMPLATE = (
    CODING_ROOT / "data/processed/products/vessel_universe"
    / "analysis_vessel_specs_{year}.csv"
)
BUFFERED_YEARS = (2021, 2022, 2023, 2024)
KNOWN_TRUNCATED_OUTER_YEAR = 2025
MAX_YEAR_WORKERS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ACCEPTED_PORT_CALL_ROOT,
        help="Accepted annual port-call product root; substitutions are refused.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Isolated run root. Accepted product roots are always refused.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Maximum concurrent annual builds. Capped at 12; effective workers "
            "cannot exceed the number of requested years."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        Path(name).replace(path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def write_feather_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_feather(temporary, compression="zstd", compression_level=6)
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


def load_vessel_meta(year: int) -> tuple[pd.DataFrame, Path]:
    path = Path(str(VESSEL_SPECS_TEMPLATE).format(year=year))
    specs = pd.read_csv(path, low_memory=False)
    specs = specs.loc[specs["vessel_uid_clean"].notna()].copy()
    specs["vessel_uid_clean"] = specs["vessel_uid_clean"].astype(str)
    specs["registry_imo_clean"] = specs["registry_imo_clean"].astype(str)
    type_source = (
        specs["detailed_vessel_type_normalized"]
        if "detailed_vessel_type_normalized" in specs.columns
        else specs["detailed_vessel_type"]
    )
    specs["shiptype_group"] = type_source.map(map_shiptype_group)
    specs["size_bin"] = build_size_bin(specs)
    result = specs[
        ["vessel_uid_clean", "registry_imo_clean", "shiptype_group", "size_bin"]
    ].rename(columns={"vessel_uid_clean": "vessel_uid"})
    result = result.drop_duplicates("vessel_uid").reset_index(drop=True)
    return result, path


def december_profile(frame: pd.DataFrame, year: int) -> dict[str, Any]:
    starts = pd.to_datetime(frame["trip_start"], errors="coerce")
    december = starts.loc[(starts.dt.year == year) & (starts.dt.month == 12)]
    counts = december.dt.day.value_counts().reindex(range(1, 32), fill_value=0).sort_index()
    early_mean = float(counts.loc[1:20].mean())
    return {
        "december_1_20_daily_mean": early_mean,
        "daily_counts": {str(day): int(count) for day, count in counts.items()},
        "december_28_31_percent_of_december_1_20_mean": {
            str(day): (float(counts.loc[day] / early_mean * 100.0) if early_mean else None)
            for day in range(28, 32)
        },
    }


def validate_roots(source_root: Path, output_root: Path) -> tuple[Path, Path]:
    source = source_root.expanduser().resolve()
    accepted = ACCEPTED_PORT_CALL_ROOT.resolve()
    if source != accepted:
        raise ValueError(
            f"source substitution refused: required accepted product root {accepted}; got {source}"
        )
    output = output_root.expanduser().resolve()
    products = (CODING_ROOT / "data/processed/products").resolve()
    if output == products or products in output.parents:
        raise ValueError(f"refusing to write buffered probe inside accepted products: {output}")
    if output == source or source in output.parents:
        raise ValueError(f"output must be isolated from accepted source root: {output}")
    return source, output


def build_one_year(year: int, source_root: Path, output_root: Path) -> dict[str, Any]:
    if year not in BUFFERED_YEARS:
        boundary = (
            f"{KNOWN_TRUNCATED_OUTER_YEAR} has no accepted "
            f"{KNOWN_TRUNCATED_OUTER_YEAR + 1} visits and remains calendar-truncated"
        )
        raise ValueError(
            f"buffered years must be drawn from {BUFFERED_YEARS}; {boundary}"
        )

    current_visits_path = source_root / str(year) / f"port_visits_{year}.feather"
    next_visits_path = (
        source_root / str(year + 1) / f"port_visits_{year + 1}.feather"
    )
    baseline_path = (
        source_root / str(year) / f"harmonized_od_voyages_{year}.feather"
    )
    for path in (current_visits_path, next_visits_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(f"accepted stage-30/10 input is missing: {path}")

    print(f"[{year}] reading accepted current visits: {current_visits_path}", flush=True)
    current_visits = pd.read_feather(current_visits_path)
    print(f"[{year}] reading accepted next-year visits: {next_visits_path}", flush=True)
    next_visits = pd.read_feather(next_visits_path)
    print(f"[{year}] reading accepted annual voyage baseline: {baseline_path}", flush=True)
    baseline = pd.read_feather(baseline_path)
    vessel_meta, vessel_meta_path = load_vessel_meta(year)

    buffered, bridges, diagnostics = build_buffered_voyages(
        current_visits,
        next_visits,
        baseline,
        vessel_meta,
        year,
    )
    baseline_median = float(baseline["duration_hours"].median())
    bridge_median = diagnostics["cross_year_duration_hours_median"]
    if not diagnostics["existing_voyage_identity"]["status"] == "pass":
        raise ValueError("existing voyage identity assertion did not pass")
    if year == 2022 and len(bridges) < 10_000:
        raise ValueError(
            f"2022 cross-year recovery is not order-ten-thousand: {len(bridges):,}"
        )
    if bridge_median is None or bridge_median <= baseline_median:
        raise ValueError(
            "new cross-year duration median is not above the annual baseline: "
            f"bridge={bridge_median}, baseline={baseline_median}"
        )

    year_dir = output_root / str(year)
    output_path = year_dir / f"harmonized_od_voyages_{year}.feather"
    print(f"[{year}] writing isolated buffered voyages: {output_path}", flush=True)
    write_feather_atomic(output_path, buffered[VOYAGE_OUTPUT_COLUMNS])

    reloaded = pd.read_feather(
        output_path,
        columns=["internal_voyage_id", "trip_start", "trip_end", "duration_hours"],
    )
    if len(reloaded) != len(buffered) or reloaded["internal_voyage_id"].duplicated().any():
        raise ValueError("serialized buffered voyage output failed row/ID validation")

    report = {
        "status": "pass",
        "stage": "30_build_analysis_sample/10_identify_port_calls_and_legs",
        "mode": "cross_year_buffered_adjacent_high_confidence_visits",
        "departure_year": year,
        "buffer_year": year + 1,
        "reads_ais": False,
        "writes_accepted_products": False,
        "source_root_is_contract_accepted": True,
        "inputs": {
            "current_visits": file_record(current_visits_path),
            "next_year_visits": file_record(next_visits_path),
            "baseline_voyages": file_record(baseline_path),
            "vessel_metadata": file_record(vessel_meta_path),
        },
        "input_rows": {
            "current_visits": int(len(current_visits)),
            "next_year_visits": int(len(next_visits)),
            "baseline_voyages": int(len(baseline)),
        },
        "diagnostics": diagnostics,
        "duration_hours_median": {
            "baseline_all_voyages": baseline_median,
            "new_cross_year_voyages": bridge_median,
        },
        "december_departure_profile_before": december_profile(baseline, year),
        "december_departure_profile_after": december_profile(buffered, year),
        "output": file_record(output_path),
        "known_outer_boundary": {
            "year": KNOWN_TRUNCATED_OUTER_YEAR,
            "status": "calendar_truncated",
            "reason": "no accepted 2026 port visits are available",
        },
        "code": {
            "entry": file_record(Path(__file__).resolve()),
            "core": file_record(PACKAGE_ROOT / "cross_year_voyages.py"),
            "git_branch": git_value("branch", "--show-current"),
            "git_commit_at_run_start": git_value("rev-parse", "HEAD"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = year_dir / f"cross_year_buffered_od_manifest_{year}.json"
    write_json_atomic(manifest_path, report)
    print(
        f"[{year}] PASS old={len(baseline):,} new_cross_year={len(bridges):,} "
        f"total={len(buffered):,} bridge_median={bridge_median:.1f}h",
        flush=True,
    )
    return report


def run_years(
    years: tuple[int, ...],
    source_root: Path,
    output_root: Path,
    max_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Run independent annual builds and aggregate every failure explicitly."""

    if max_workers < 1 or max_workers > MAX_YEAR_WORKERS:
        raise ValueError(f"max_workers must be between 1 and {MAX_YEAR_WORKERS}")
    effective_workers = min(max_workers, len(years))
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if effective_workers == 1:
        for year in years:
            try:
                reports.append(build_one_year(year, source_root, output_root))
            except Exception as exc:  # pragma: no cover - exercised by real-run gate
                failures.append({"year": year, "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_year = {
                executor.submit(build_one_year, year, source_root, output_root): year
                for year in years
            }
            for future in as_completed(future_year):
                year = future_year[future]
                try:
                    reports.append(future.result())
                except Exception as exc:
                    failures.append({"year": year, "error": repr(exc)})

    reports.sort(key=lambda report: int(report["departure_year"]))
    failures.sort(key=lambda failure: int(failure["year"]))
    return reports, failures, effective_workers


def main() -> int:
    args = parse_args()
    source_root, output_root = validate_roots(args.source_root, args.output_root)
    years = tuple(sorted(set(args.years)))
    if not years:
        raise ValueError("at least one departure year is required")
    reports, failures, effective_workers = run_years(
        years,
        source_root,
        output_root,
        args.max_workers,
    )
    write_json_atomic(
        output_root / "run_summary.json",
        {
            "status": "failed" if failures else "pass",
            "years": years,
            "requested_max_workers": args.max_workers,
            "effective_year_workers": effective_workers,
            "reports": reports,
            "failures": failures,
            "downstream_stage_30_20_run": False,
        },
    )
    if failures:
        raise RuntimeError(f"annual buffered builds failed: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
