#!/usr/bin/env python3
"""Attach frozen port coordinates to the adopted 2025 voyage endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[4]
PANEL = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release/10_u1/10_voyage_exposure_panel.parquet"
CITY = CODING / "data/frozen_inputs/e001_2025_pretrim_golfito_corrected/city_snap_table_2025.csv"
CITY_MANIFEST = CODING / "pipeline/00_environment_and_config/config/current_inputs_2025.json"
PRODUCT_MANIFEST = CODING / "data/processed/products/companion_tracks/2025/adoption_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    city_manifest = json.loads(CITY_MANIFEST.read_text())
    recorded = city_manifest["city_reference"]["sha256"]
    actual_city_hash = sha256(CITY)
    if actual_city_hash != recorded:
        raise ValueError("frozen city reference differs from its accepted manifest")

    voyages = pd.read_parquet(PANEL, columns=[
        "candidate_id", "origin_city", "dest_city", "departure_time", "arrival_time",
    ])
    if voyages.candidate_id.isna().any() or voyages.candidate_id.duplicated().any():
        raise ValueError("U1 voyage panel is not one row per candidate_id")
    if (voyages.arrival_time <= voyages.departure_time).any():
        raise ValueError("nonpositive voyage interval")
    cities = pd.read_csv(CITY, usecols=["city_id", "port_lat", "port_lon"])
    if cities.city_id.isna().any() or cities.city_id.duplicated().any():
        raise ValueError("frozen city reference has ambiguous city_id")
    for role, key in (("departure", "origin_city"), ("arrival", "dest_city")):
        mapping = cities.rename(columns={
            "city_id": key, "port_lat": f"{role}_port_lat", "port_lon": f"{role}_port_lon",
        })
        voyages = voyages.merge(mapping, on=key, how="left", validate="many_to_one", sort=False)
    if len(voyages) != 1_354_168 or voyages.candidate_id.duplicated().any():
        raise ValueError("port endpoint join changed the adopted voyage grain")
    coords = voyages[[
        "departure_port_lat", "departure_port_lon", "arrival_port_lat", "arrival_port_lon",
    ]].to_numpy(dtype=float)
    valid = np.isfinite(coords).all(axis=1)
    valid &= (np.abs(coords[:, [0, 2]]) <= 90).all(axis=1)
    valid &= (np.abs(coords[:, [1, 3]]) <= 180).all(axis=1)
    voyages["port_endpoints_connected"] = valid
    voyages["port_coordinate_source"] = np.where(valid, CITY.name, pd.NA)
    voyages.to_parquet(out / "port_endpoints_by_candidate.parquet", index=False)
    report = {
        "candidate_rows": int(len(voyages)),
        "unique_candidate_ids": int(voyages.candidate_id.nunique()),
        "connected": int(valid.sum()),
        "unconnected": int((~valid).sum()),
        "departure_city_unmatched": int(voyages.departure_port_lat.isna().sum()),
        "arrival_city_unmatched": int(voyages.arrival_port_lat.isna().sum()),
        "sources": {
            "times_and_candidate_id": {"path": str(PANEL), "sha256": sha256(PANEL),
                                       "adopted_product_manifest": str(PRODUCT_MANIFEST),
                                       "adopted_product_manifest_sha256": sha256(PRODUCT_MANIFEST)},
            "port_coordinates": {"path": str(CITY), "sha256": actual_city_hash,
                                 "manifest": str(CITY_MANIFEST), "manifest_sha256": sha256(CITY_MANIFEST)},
        },
        "output": str(out / "port_endpoints_by_candidate.parquet"),
        "output_sha256": sha256(out / "port_endpoints_by_candidate.parquet"),
    }
    (out / "port_endpoints_closure.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out), ensure_ascii=False, indent=2))
