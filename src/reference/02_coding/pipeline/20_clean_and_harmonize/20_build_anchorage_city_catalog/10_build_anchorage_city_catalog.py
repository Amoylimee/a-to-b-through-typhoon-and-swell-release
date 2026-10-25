#!/usr/bin/env python3
"""Build the single project-owned anchorage catalog.

Port-call detection needs anchorage coordinates, because a call opens when a
vessel comes within 3 km of an anchorage and closes when it leaves 4 km. City
identity needs the reviewed analytical-city crosswalk. Those two facts live in
two different legacy files: the frozen 2022 named-anchorage catalog carries
coordinates but only the old anchorage-level names, and the v3 crosswalk carries
the reviewed city grouping but no coordinates at all.

This step joins them into one file, so the project holds exactly one anchorage
reference and every year of the chain resolves anchorages and cities the same
way. The 2022 coordinate catalog is used unchanged, as the accepted method
boundary requires: holding the catalog fixed means a change in port calls between
2021 and 2025 reflects vessel behaviour rather than a revision of the catalog.

The old anchorage-level names are carried through as `anchorage_label` and
`anchorage_sublabel` and are deliberately not called city fields. City identity is
`analytical_city_id` from the crosswalk, and nothing downstream should infer a
city from the anchorage label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CODING_ROOT = Path(__file__).resolve().parents[3]
LEGACY = CODING_ROOT / "data" / "legacy_inputs"

COORDINATE_CATALOG = LEGACY / "named_anchorages_v2_20221206.csv"
CITY_CROSSWALK = LEGACY / "anchorage_to_analytical_city_candidate_v3.csv"

DEFAULT_OUTPUT = (
    CODING_ROOT / "data" / "processed" / "products" / "anchorage_catalog"
    / "anchorages_analytical_city_v3.csv"
)

# coordinates and anchorage-level identity, from the frozen 2022 catalog
FROM_CATALOG = {
    "s2id": "s2id",
    "lat": "lat",
    "lon": "lon",
    "iso3": "anchorage_iso3",
    "label": "anchorage_label",
    "sublabel": "anchorage_sublabel",
    "distance_from_shore_m": "distance_from_shore_m",
    "drift_radius": "drift_radius",
    "dock": "dock",
}

# reviewed city identity and role, from the v3 crosswalk
FROM_CROSSWALK = {
    "country_iso3": "country_iso3",
    "city_name": "city_name",
    "analytical_city_id": "analytical_city_id",
    "topology_city_id": "topology_city_id",
    "canonical_port_entity": "canonical_port_entity",
    "location_role": "location_role",
    "endpoint_eligible": "endpoint_eligible",
    "role_review_status": "role_review_status",
    "mapping_changed": "mapping_changed",
    "recommendation": "recommendation",
    "canal_id": "canal_id",
    "canal_event_type": "canal_event_type",
    "canal_side": "canal_side",
    "crosswalk_version": "crosswalk_version",
}

OUTPUT_COLUMNS = list(FROM_CATALOG.values()) + list(FROM_CROSSWALK.values())

# Anchorages known to carry unusable coordinates in the frozen 2022 catalog.
# Listing them here keeps the exclusion visible and lets an unknown bad row still
# fail the build instead of being waved through by the same rule.
KNOWN_BAD_COORDINATES = {
    "8efe7543": (
        "lon recorded as 1001 by an anchorage_overrides edit; wraps to 79W 11.8N "
        "in the Caribbean while still labelled POINTE NOIRE, COG. Excluding it "
        "keeps 165 other Pointe-Noire anchorages, so the port is not lost."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path in (COORDINATE_CATALOG, CITY_CROSSWALK):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    csv.field_size_limit(10**7)

    with CITY_CROSSWALK.open(newline="") as fh:
        crosswalk = {}
        for row in csv.DictReader(fh):
            key = row["anchorage_id"].strip()
            if key in crosswalk:
                raise SystemExit(f"crosswalk repeats anchorage {key}")
            crosswalk[key] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_catalog = 0
    n_written = 0
    dropped_no_city: list[str] = []
    bad_coordinates: list[str] = []
    unexpected_bad: list[str] = []
    seen: set[str] = set()
    blank_city = 0

    with COORDINATE_CATALOG.open(newline="") as fh, args.out.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in csv.DictReader(fh):
            n_catalog += 1
            s2id = row["s2id"].strip()
            if s2id in seen:
                raise SystemExit(f"coordinate catalog repeats s2id {s2id}")
            seen.add(s2id)

            city = crosswalk.get(s2id)
            if city is None:
                dropped_no_city.append(s2id)
                continue

            usable = True
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (TypeError, ValueError):
                usable = False
            else:
                usable = -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
            if not usable:
                bad_coordinates.append(s2id)
                if s2id not in KNOWN_BAD_COORDINATES:
                    unexpected_bad.append(s2id)
                continue

            record = {out_name: row.get(src, "") for src, out_name in FROM_CATALOG.items()}
            record.update(
                {out_name: city.get(src, "") for src, out_name in FROM_CROSSWALK.items()}
            )
            if not record["analytical_city_id"]:
                blank_city += 1
            writer.writerow(record)
            n_written += 1

    unused_crosswalk = len(crosswalk) - n_written - len(bad_coordinates)

    report = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_catalog": {
            "path": str(COORDINATE_CATALOG.resolve()),
            "rows": n_catalog,
            "sha256": sha256_file(COORDINATE_CATALOG),
        },
        "city_crosswalk": {
            "path": str(CITY_CROSSWALK.resolve()),
            "rows": len(crosswalk),
            "sha256": sha256_file(CITY_CROSSWALK),
        },
        "output": {
            "path": str(args.out.resolve()),
            "rows": n_written,
            "columns": OUTPUT_COLUMNS,
            "sha256": sha256_file(args.out),
        },
        "catalog_anchorages_without_a_crosswalk_row": len(dropped_no_city),
        "catalog_anchorages_with_invalid_coordinates": len(bad_coordinates),
        "excluded_known_bad_coordinates": {
            s2id: KNOWN_BAD_COORDINATES[s2id]
            for s2id in bad_coordinates
            if s2id in KNOWN_BAD_COORDINATES
        },
        "unexpected_invalid_coordinates": unexpected_bad,
        "crosswalk_anchorages_without_coordinates": unused_crosswalk,
        "rows_with_blank_analytical_city_id": blank_city,
    }
    report_path = args.out.with_name(args.out.stem + "_build_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    print(f"coordinate catalog rows          : {n_catalog:,}")
    print(f"city crosswalk rows              : {len(crosswalk):,}")
    print(f"catalog rows with no crosswalk   : {len(dropped_no_city):,}")
    print(f"crosswalk rows with no coords    : {unused_crosswalk:,}")
    print(f"rows with blank city id          : {blank_city:,}")
    print(f"excluded, known bad coordinates  : {len(bad_coordinates) - len(unexpected_bad):,}")
    for s2id in bad_coordinates:
        if s2id in KNOWN_BAD_COORDINATES:
            print(f"    {s2id}: {KNOWN_BAD_COORDINATES[s2id]}")
    print(f"UNEXPECTED bad coordinates       : {len(unexpected_bad):,}")
    for s2id in unexpected_bad[:20]:
        print(f"    {s2id}")
    print(f"\nwritten {n_written:,} rows -> {args.out}")
    print(f"report  -> {report_path}")

    if dropped_no_city or unexpected_bad or blank_city:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
