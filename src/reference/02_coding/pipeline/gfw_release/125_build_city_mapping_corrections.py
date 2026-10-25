#!/usr/bin/env python3
"""E006 (T034): list the anchorages the frozen anchorage-to-city mapping files under a distant city.

Step 3 recomputed every city coordinate from GFW-version calls (T032/T033). Four cities then
landed on land because the mapping, following GFW's own anchorage labels, files anchorages of
another port under them. The user asked for these four to be fixed (T034). Rules:

- An anchorage cluster lying among other cities' anchorages takes the city of its nearest
  correctly mapped, endpoint-eligible anchorage (all within 6 km):
  - `USA:GRAYS HARBOR`: the Florida anchorage (GFW sublabel PRETTY BAYOU) -> Panama City;
  - `USA:SACRAMENTO`: the three New Haven anchorages;
  - `CHN:HUANG HUA`: the 32 Yangtze anchorages near Suzhou (GFW sublabel SUZHOU).
- `CHN:TAIZHOU` holds two real ports that share the pinyin name: Taizhou, Zhejiang (台州) and
  Taizhou, Jiangsu (泰州). The Jiangsu cluster is a port of its own (up to 10 km from any other
  city's anchorage), so it becomes the new city `CHN:TAIZHOU JIANGSU`; `CHN:TAIZHOU` keeps the
  Zhejiang anchorages, where the accepted reference AIS version coordinate also lies.

Only the five city fields the 30c builder reads change; each anchorage keeps its own location
role and endpoint eligibility. The frozen mapping is not modified: `130_build_gfw_voyage_base.py
--mapping-corrections` applies this list to a copy inside its run. No network use.

    python 125_build_city_mapping_corrections.py --out config/t034_anchorage_city_corrections.csv
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
MAPPING = CODING / "data/frozen_inputs/stage30_support_2025/data/legacy_inputs/anchorage_to_analytical_city_candidate_v3.csv"
CATALOGS = [CODING / "data/frozen_inputs/named_anchorages/named_anchorages_v2_20221206.csv",
            CODING / "data/frozen_inputs/named_anchorages/named_anchorages_v2_20201104.csv"]
EXPECTED = {MAPPING: "a06ef572b8a3689b3e5b1ea418810bc5cd3fa59531b69ca7f2ebdcd99ef3a030"}
CITY_FIELDS = ["country_iso3", "city_name", "raw_city_id", "analytical_city_id", "topology_city_id"]
NEAREST_MAX_KM = 6.0
# Stray cluster of each city, separated by a coordinate bound far from both clusters.
STRAYS = {
    "USA:GRAYS HARBOR": ("lat", "<", 40.0, "nearest"),
    "USA:SACRAMENTO": ("lon", ">", -100.0, "nearest"),
    "CHN:HUANG HUA": ("lat", "<", 35.0, "nearest"),
    "CHN:TAIZHOU": ("lat", ">", 31.0, "CHN:TAIZHOU JIANGSU"),
}
EARTH_KM = 6371.0088


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build() -> pd.DataFrame:
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ValueError(f"frozen input changed: {path}")
    mapping = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
    catalog = (pd.concat([pd.read_csv(p, dtype={"s2id": str}, usecols=["s2id", "lat", "lon", "label", "sublabel"])
                          for p in CATALOGS]).drop_duplicates("s2id"))
    located = mapping.merge(catalog, left_on="anchorage_id", right_on="s2id", how="inner", validate="one_to_one")
    reference = located.loc[~located.topology_city_id.isin(STRAYS) & located.endpoint_eligible.eq("True")].reset_index(drop=True)
    tree = BallTree(np.radians(reference[["lat", "lon"]].to_numpy()), metric="haversine")
    rows = []
    for city, (column, side, bound, target) in STRAYS.items():
        members = located.loc[located.topology_city_id.eq(city)]
        stray = members.loc[members[column].lt(bound) if side == "<" else members[column].gt(bound)]
        kept = members.drop(stray.index)
        if stray.empty or kept.empty:
            raise ValueError(f"{city}: expected both a stray and a kept cluster")
        distance, index = tree.query(np.radians(stray[["lat", "lon"]].to_numpy()), k=1)
        for (_, row), km, i in zip(stray.iterrows(), distance[:, 0] * EARTH_KM, index[:, 0]):
            nearest = reference.iloc[i]
            if target == "nearest":
                if km > NEAREST_MAX_KM:
                    raise ValueError(f"{city}: {row.anchorage_id} is {km:.1f} km from any correctly mapped anchorage")
                new = {field: nearest[field] for field in CITY_FIELDS}
                rule = "nearest_correctly_mapped_anchorage"
            else:
                iso3, name = target.split(":", 1)
                new = {"country_iso3": iso3, "city_name": name, "raw_city_id": target,
                       "analytical_city_id": target, "topology_city_id": target}
                rule = "split_same_name_port"
            rows.append({"anchorage_id": row.anchorage_id, "gfw_label": row.label, "gfw_sublabel": row.sublabel,
                         "lat": row.lat, "lon": row.lon, "old_topology_city_id": city, "rule": rule,
                         **{f"new_{field}": value for field, value in new.items()},
                         "nearest_correct_anchorage": nearest.anchorage_id,
                         "nearest_correct_city": nearest.topology_city_id, "nearest_km": round(float(km), 3)})
    table = pd.DataFrame(rows).sort_values(["old_topology_city_id", "anchorage_id"]).reset_index(drop=True)
    if table.anchorage_id.duplicated().any():
        raise ValueError("an anchorage is corrected twice")
    return table


def apply(mapping_path: Path, corrections: pd.DataFrame, target: Path) -> pd.DataFrame:
    """Write a corrected copy of the mapping; only the listed rows' city fields change."""
    mapping = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
    selected = mapping.anchorage_id.isin(corrections.anchorage_id)
    if selected.sum() != len(corrections):
        raise ValueError("a corrected anchorage is missing from the mapping")
    by_id = corrections.set_index("anchorage_id")
    ids = mapping.loc[selected, "anchorage_id"]
    if not mapping.loc[selected, "topology_city_id"].eq(ids.map(by_id.old_topology_city_id)).all():
        raise ValueError("a corrected anchorage no longer holds its recorded old city")
    for field in CITY_FIELDS:
        mapping.loc[selected, field] = ids.map(by_id[f"new_{field}"])
    target.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(target, index=False)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    table = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.groupby(["old_topology_city_id", "new_topology_city_id"]).agg(
        anchorages=("anchorage_id", "size"), max_km=("nearest_km", "max")).to_string())


if __name__ == "__main__":
    main()
