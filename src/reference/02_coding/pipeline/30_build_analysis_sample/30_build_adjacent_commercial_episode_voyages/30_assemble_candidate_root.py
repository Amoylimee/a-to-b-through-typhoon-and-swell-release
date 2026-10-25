#!/usr/bin/env python3
"""Assemble the per-year candidate root consumed by stage 40a.

The voyage builders intentionally keep calendar-bounded and cross-year outputs in
different locations.  This script is the only supported way to select the cohort
that the ordered-passage measurement reads.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
YEARS = (2021, 2022, 2023, 2024, 2025)
COHORTS = {
    2021: "cross_year",
    2022: "cross_year",
    2023: "cross_year",
    2024: "cross_year",
    2025: "calendar",
}
REQUIRED_ENUMERATION_MODE = "adjacent_commercial_episodes"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _link_or_verify(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(
                f"Existing link resolves to the wrong source: {destination} -> "
                f"{destination.resolve()} (expected {source})"
            )
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    os.symlink(str(source), str(destination))


def _inspect_source(source_dir: Path, year: int) -> dict[str, Any]:
    parts = sorted(source_dir.glob("part-*.feather"))
    if not parts:
        raise FileNotFoundError(f"No voyage parts found under {source_dir}")

    row_count = 0
    next_year_arrivals = 0
    latest_arrival: pd.Timestamp | None = None
    for part in parts:
        arrivals = pd.read_feather(part, columns=["arrival_time"])["arrival_time"]
        parsed = pd.to_datetime(arrivals, utc=True, errors="raise")
        row_count += int(len(parsed))
        next_year = parsed.dt.year.eq(year + 1)
        next_year_arrivals += int(next_year.sum())
        if len(parsed):
            part_latest = parsed.max()
            if latest_arrival is None or part_latest > latest_arrival:
                latest_arrival = part_latest

    return {
        "row_count": row_count,
        "part_count": len(parts),
        "next_year_arrival_rows": next_year_arrivals,
        "has_next_year_arrival": bool(next_year_arrivals),
        "latest_arrival": latest_arrival.isoformat() if latest_arrival is not None else None,
        "parts": parts,
    }


def _source_manifest(source_root: Path, year: int) -> Path:
    path = source_root / "manifests" / f"final_sample_manifest_{year}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source final manifest: {path}")
    return path


def assemble_candidate_root(
    candidate_root: Path,
    calendar_root: Path | None = None,
    cross_year_root: Path | None = None,
    years: tuple[int, ...] = YEARS,
    *,
    run_root: Path | None = None,
) -> dict[str, Any]:
    candidate_root = candidate_root.resolve()
    if run_root is not None:
        run_root = run_root.resolve()
        calendar_root = run_root
        cross_year_root = run_root
    if calendar_root is None or cross_year_root is None:
        raise ValueError("provide run_root or both calendar_root and cross_year_root")
    calendar_root = calendar_root.resolve()
    cross_year_root = cross_year_root.resolve()
    accepted_root = (REPO_ROOT / "02_coding/data/processed/products").resolve()
    if candidate_root == accepted_root or accepted_root in candidate_root.parents:
        raise ValueError(f"candidate_root must not be an accepted product root: {candidate_root}")
    if tuple(sorted(years)) != YEARS:
        raise ValueError(f"E030 requires exactly years {YEARS}; received {years}")

    candidate_root.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    for year in years:
        cohort = COHORTS[year]
        source_root = calendar_root if cohort == "calendar" else cross_year_root
        source_dir = source_root / f"adjacent_commercial_episode_voyages_{year}"
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Required {cohort} cohort for {year} is absent: {source_dir}"
            )
        source_manifest = _source_manifest(source_root, year)
        source_stats = _inspect_source(source_dir, year)

        manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("status") != "pass":
            raise ValueError(f"Source manifest is not pass for {year}: {source_manifest}")
        if int(manifest_payload.get("analysis_year", -1)) != year:
            raise ValueError(
                f"Source manifest year disagrees for {year}: "
                f"{manifest_payload.get('analysis_year')!r}"
            )
        mode = manifest_payload.get("candidate_enumeration_mode")
        if mode != REQUIRED_ENUMERATION_MODE:
            raise ValueError(
                f"Source manifest mode disagrees for {year}: "
                f"{mode!r} != {REQUIRED_ENUMERATION_MODE!r}"
            )
        expected_rows = manifest_payload.get("rows")
        if expected_rows is None:
            expected_rows = manifest_payload.get("summary", {}).get(
                "final_retained"
            )
        if expected_rows is None:
            raise ValueError(
                f"Source manifest has no final row count for {year}: {source_manifest}"
            )
        if expected_rows is not None and int(expected_rows) != source_stats["row_count"]:
            raise ValueError(
                f"Source manifest row count disagrees for {year}: "
                f"manifest={expected_rows}, data={source_stats['row_count']}"
            )

        destination_dir = candidate_root / str(year)
        destination_dir.mkdir(parents=True, exist_ok=True)
        for part in source_stats["parts"]:
            _link_or_verify(part, destination_dir / part.name)
        _link_or_verify(source_manifest, destination_dir / source_manifest.name)

        entries[str(year)] = {
            "cohort": cohort,
            "calendar_truncated": year == 2025,
            "source_path": str(source_dir),
            "source_manifest_path": str(source_manifest.resolve()),
            "link_path": str(destination_dir),
            "row_count": source_stats["row_count"],
            "part_count": source_stats["part_count"],
            "next_year_arrival_rows": source_stats["next_year_arrival_rows"],
            "has_next_year_arrival": source_stats["has_next_year_arrival"],
            "latest_arrival": source_stats["latest_arrival"],
            "candidate_enumeration_mode": mode,
        }

    payload = {
        "status": "pass",
        "experiment": "E030",
        "candidate_enumeration_mode": REQUIRED_ENUMERATION_MODE,
        "candidate_root": str(candidate_root),
        "years": entries,
    }
    manifest_path = candidate_root / "assembly_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"Existing assembly manifest differs; refusing to overwrite: {manifest_path}"
            )
    else:
        _write_json_atomic(manifest_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        "--output-root",
        dest="candidate_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="The single five-year E030 stage-30/20 run tree used for every cohort.",
    )
    parser.add_argument("--calendar-root", type=Path)
    parser.add_argument("--cross-year-root", type=Path)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = assemble_candidate_root(
        args.candidate_root,
        args.calendar_root,
        args.cross_year_root,
        tuple(args.years),
        run_root=args.run_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
