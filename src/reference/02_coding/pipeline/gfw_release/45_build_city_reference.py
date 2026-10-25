#!/usr/bin/env python3
"""E007 V3: city and anchorage reference tables for the GFW-version release.

city_reference.csv: one row per city key used in the release, with its country, the
call-weighted port coordinate recomputed from GFW-version port calls, the number of
anchorage cells and calls behind it, and how often it appears in released records.

city_anchorages.csv: every anchorage cell of the port-call catalog with a GFW-version call in
2025: GFW anchorage label and sublabel, coordinate, dock flag, the city it is assigned to
(after the T034 corrections), and its number of 2025 port visits.

    python 45_build_city_reference.py --out <new .runs dir>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pyarrow.feather as ft

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
BASE = CODING / ".runs/2026-10-06_1728_E006-gfw-voyage-base-t034/input_view"
CITY_TABLE = BASE / "results/support/city_snap_table_2025.csv"
CROSSWALK = BASE / "support_inputs/anchorage_to_analytical_city_corrected.csv"
CATALOG = CODING / "data/frozen_inputs/anchorage_catalog/anchorages_analytical_city_v3_dock_adjusted_inclusive.csv"
VISITS = CODING / ".runs/2026-10-06_1650_E006-gfw-port-calls/shards"
RELEASE = CODING / ".runs/2026-10-07_1410_E006-gfw-release-step6/assembly/release_full_precision.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    out.mkdir(parents=True, exist_ok=False)

    release = pd.read_parquet(RELEASE, columns=["origin_city", "dest_city", "voyages"])
    cities = pd.read_csv(CITY_TABLE)
    used = pd.Index(pd.unique(pd.concat([release.origin_city, release.dest_city])))
    missing = used.difference(cities.city_id)
    if len(missing):
        raise ValueError(f"released cities without a GFW city coordinate: {list(missing[:5])}")
    role = pd.DataFrame({
        "released_records_as_origin": release.groupby("origin_city").size(),
        "released_records_as_destination": release.groupby("dest_city").size(),
        "released_voyages_from": release.groupby("origin_city").voyages.sum(),
        "released_voyages_to": release.groupby("dest_city").voyages.sum(),
    }).fillna(0).astype(int)
    city = cities.loc[cities.city_id.isin(used), ["city_id", "iso3", "city_name", "port_lat", "port_lon", "n_anchorages",
                                                   "source_call_count", "coordinate_spread_deg"]].rename(
        columns={"city_id": "city_key", "source_call_count": "gfw_2025_calls"})
    city = city.merge(role, left_on="city_key", right_index=True, how="left")
    city.sort_values("city_key").to_csv(out / "city_reference.csv", index=False)

    visits = pd.concat([ft.read_table(f, columns=["anchorage_id"]).to_pandas() for f in glob.glob(str(VISITS / "*/shard_port_visits/*.feather"))])
    counts = visits.anchorage_id.astype(str).value_counts().rename("gfw_2025_port_visits")
    catalog = pd.read_csv(CATALOG, dtype=str, usecols=["s2id", "lat", "lon", "anchorage_iso3", "anchorage_label", "anchorage_sublabel", "dock"])
    catalog["dock"] = catalog.dock.str.lower().eq("true")
    crosswalk = pd.read_csv(CROSSWALK, dtype=str, usecols=["anchorage_id", "analytical_city_id", "location_role"])
    cells = catalog.merge(crosswalk, left_on="s2id", right_on="anchorage_id", how="left", validate="one_to_one").drop(columns="anchorage_id")
    cells = cells.merge(counts, left_on="s2id", right_index=True, how="inner")
    if cells.analytical_city_id.isna().any():
        raise ValueError("visited anchorage cell without a city assignment")
    cells = cells.rename(columns={"s2id": "anchorage_id", "analytical_city_id": "city_key", "dock": "dock_flag"})
    cells["city_in_release"] = cells.city_key.isin(used)
    cells.sort_values(["city_key", "anchorage_id"]).to_csv(out / "city_anchorages.csv", index=False)

    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()
    execution = {"step": "E007 V3: city and anchorage reference tables", "state": "pass", "git_head": git("rev-parse", "HEAD"),
                 "git_dirty_experiment": bool(git("status", "--porcelain", "--", ".")),
                 "inputs": {str(p.relative_to(CODING)): sha256(p) for p in (CITY_TABLE, CROSSWALK, CATALOG, RELEASE)},
                 "released_cities": len(used), "anchorage_cells_with_visits": len(cells),
                 "anchorage_cells_in_released_cities": int(cells.city_in_release.sum()),
                 "port_visits": int(counts.sum()), "visits_at_cells_outside_catalog": int(counts.loc[~counts.index.isin(catalog.s2id)].sum())}
    (out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps({k: v for k, v in execution.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
