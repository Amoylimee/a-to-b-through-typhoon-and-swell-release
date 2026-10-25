#!/usr/bin/env python3
"""Assemble and verify the five-year OD root consumed by E030 stage 30/20."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.od_input_assembly import (
    DEFAULT_SOURCES,
    EXPECTED_YEARS,
    assemble_od_input_root,
    verify_od_input_root,
)


def parse_source(value: str) -> tuple[int, Path]:
    try:
        year_text, path_text = value.split("=", 1)
        year = int(year_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "--source must have the form YEAR=/absolute/or/relative/path.feather"
        ) from error
    if year not in EXPECTED_YEARS or not path_text:
        raise argparse.ArgumentTypeError(f"unsupported source year: {year}")
    return year, Path(path_text).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        help="Override one default source as YEAR=PATH; may be repeated.",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.verify_only:
        manifest = verify_od_input_root(output_root)
    else:
        sources = dict(DEFAULT_SOURCES)
        for year, path in args.source or []:
            sources[year] = path
        manifest = assemble_od_input_root(output_root, sources)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
