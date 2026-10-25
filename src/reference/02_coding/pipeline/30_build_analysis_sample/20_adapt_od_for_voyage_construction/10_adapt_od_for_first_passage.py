#!/usr/bin/env python3
"""Present our AIS-derived OD voyages in the format the first-passage builder reads.

The builder was written against GFW's monthly OD extracts and asks for nine columns
under GFW's names. Our port-call step produces the same voyages with twenty-five
columns under our own names. The information matches — both sides identify a voyage
by its two bracketing port visits and name anchorages by 8-hex-character S2 cell id —
so this step is a rename plus one join, not a conversion.

Writing monthly CSVs rather than teaching the builder a second input format keeps the
frozen voyage-identification code untouched, which matters because it verifies its own
dependencies by SHA-256. `od_paths()` requires exactly twelve files per year, so a
month with no voyages still gets a header-only file.

Two details the builder cares about:

- `mmsi` is absent from our product, which keys vessels by `vessel_uid`. The builder
  filters rows against the analysis MMSI set, so the value is joined back from the
  vessel specs rather than left blank.
- GFW timestamps carry a trailing ` UTC`. Ours do not. The suffix is added so both
  sides parse identically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CODING_ROOT = Path(__file__).resolve().parents[3]

# The voyages this chain built, not the two inherited products.
OD_ROOT = (
    CODING_ROOT / "data" / "processed" / "identify_port_calls_and_legs"
    / "build_port_calls_and_legs"
)
OUR_OD_TEMPLATE = OD_ROOT / "{year}" / "harmonized_od_voyages_{year}.feather"
SPECS_TEMPLATE = (
    CODING_ROOT / "data" / "processed" / "products" / "vessel_universe"
    / "analysis_vessel_specs_{year}.csv"
)
DEFAULT_OUT = CODING_ROOT / "data" / "processed" / "products" / "our_od"

# the universe step emits specs for every year, all carrying the same observed fleet
SPECS_YEARS = (2021, 2022, 2023, 2024, 2025)

# builder column <- our column
RENAME = {
    "vessel_id": "vessel_uid",
    "trip_id": "internal_voyage_id",
    "trip_start": "trip_start",
    "trip_end": "trip_end",
    "trip_start_anchorage_s2id": "origin_anchorage_id",
    "trip_end_anchorage_s2id": "dest_anchorage_id",
    "trip_start_visit_id": "origin_visit_id_internal",
    "trip_end_visit_id": "dest_visit_id_internal",
}
OD_COLUMNS = ["mmsi"] + list(RENAME)


def specs_year(year: int) -> int:
    return year if year in SPECS_YEARS else 2021


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--od-root",
        type=Path,
        default=OD_ROOT,
        help=(
            "Root holding {year}/harmonized_od_voyages_{year}.feather. Defaults to the "
            "accepted port-call product root. Any other value also requires --out."
        ),
    )
    args = parser.parse_args()

    # Adapting a non-default OD into the accepted our_od root would replace the
    # accepted first-passage input with a differently-defined measurement.
    if args.od_root.resolve() != OD_ROOT.resolve() and args.out.resolve() == DEFAULT_OUT.resolve():
        raise SystemExit(
            f"refusing to write a non-default OD source into the accepted output root "
            f"{DEFAULT_OUT}; pass --out to a separate location"
        )

    od_template = args.od_root / "{year}" / "harmonized_od_voyages_{year}.feather"

    reports = []
    for year in args.years:
        source = Path(str(od_template).format(year=year))
        if not source.exists():
            raise SystemExit(f"missing OD product: {source}")

        voyages = pd.read_feather(source)
        n_in = len(voyages)

        specs_path = Path(str(SPECS_TEMPLATE).format(year=specs_year(year)))
        specs = pd.read_csv(
            specs_path, usecols=["vessel_uid_clean", "mmsi_clean"], dtype=str
        ).dropna()
        specs = specs.drop_duplicates("vessel_uid_clean")

        voyages["vessel_uid"] = voyages["vessel_uid"].astype(str)
        merged = voyages.merge(
            specs.rename(columns={"vessel_uid_clean": "vessel_uid", "mmsi_clean": "mmsi"}),
            on="vessel_uid",
            how="left",
        )
        unmatched = int(merged["mmsi"].isna().sum())
        if unmatched:
            raise SystemExit(
                f"{year}: {unmatched:,} voyages have no MMSI in {specs_path.name}; "
                "the builder filters on MMSI, so these would vanish silently"
            )

        out = pd.DataFrame({"mmsi": merged["mmsi"].astype(str)})
        for target, ours in RENAME.items():
            out[target] = merged[ours].astype(str)

        # GFW writes '2021-01-01 06:16:23 UTC'; ours lack the suffix
        starts = pd.to_datetime(merged["trip_start"], errors="coerce")
        ends = pd.to_datetime(merged["trip_end"], errors="coerce")
        if starts.isna().any() or ends.isna().any():
            raise SystemExit(f"{year}: unparseable trip_start/trip_end values")
        out["trip_start"] = starts.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        out["trip_end"] = ends.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        out = out[OD_COLUMNS]

        year_dir = args.out / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        month_of = starts.dt.month
        written = 0
        per_month = {}
        for month in range(1, 13):
            chunk = out.loc[month_of == month]
            path = year_dir / f"voyages_{year}{month:02d}.csv"
            chunk.to_csv(path, index=False)
            per_month[f"{year}{month:02d}"] = int(len(chunk))
            written += len(chunk)

        if written != n_in:
            raise SystemExit(
                f"{year}: wrote {written:,} rows from {n_in:,} voyages; "
                "a trip_start fell outside months 1-12"
            )

        reports.append(
            {
                "year": year,
                "source": str(source),
                "od_root_is_accepted_default": args.od_root.resolve() == OD_ROOT.resolve(),
                "specs_year_used": specs_year(year),
                "voyages_in": n_in,
                "rows_written": written,
                "files": 12,
                "rows_per_month": per_month,
            }
        )
        print(f"{year}: {n_in:,} voyages -> 12 monthly files in {year_dir}")
        print(
            "     "
            + "  ".join(f"{m[-2:]}:{c:,}" for m, c in sorted(per_month.items()))
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adapt_report.json").write_text(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
