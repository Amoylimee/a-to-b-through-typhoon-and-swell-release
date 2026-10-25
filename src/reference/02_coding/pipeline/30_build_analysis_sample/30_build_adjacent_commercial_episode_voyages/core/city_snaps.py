#!/usr/bin/env python3
"""Build independent 2021 production city coordinates and searoute snaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import searoute as sr


CODING_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = PIPELINE_ROOT / "config/frozen/city_snap_2021.json"
DEFAULT_ARTIFACT_DIR = DATA_ROOT / "main/support/frozen_2021"
DEFAULT_REPORT_DIR = DATA_ROOT / "main/validation/city_snaps"
CALL_COLUMNS = [
    "anchorage_id",
    "topology_city_id",
    "country_iso3",
    "city_name",
    "is_commercial_endpoint",
    "n_events",
]
PAIR_COLUMNS = [
    "pair_id",
    "origin_city",
    "dest_city",
]
ANCHORAGE_COLUMNS = ["s2id", "lat", "lon"]


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
        newline="",
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
        newline="",
    ) as handle:
        frame.to_csv(
            handle,
            index=False,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=CODING_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def clean_id(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .fillna("")
    )


def validate_frozen_file(path_string: str, expected_sha256: str) -> Path:
    path = CODING_ROOT / path_string
    if not path.is_file():
        raise FileNotFoundError(f"Frozen input is missing: {path_string}")
    actual_sha256 = sha256sum(path)
    if actual_sha256 != expected_sha256:
        raise AssertionError(
            f"Frozen SHA-256 mismatch for {path_string}: {actual_sha256}"
        )
    return path


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    branch = git_value("branch", "--show-current")
    if branch != config["approved_branch"]:
        raise AssertionError(
            f"Expected branch {config['approved_branch']}; found {branch}"
        )
    if config["analysis_year"] != 2021:
        raise AssertionError("C0.3 is restricted to the 2021 development year")
    if sr.__version__ != config["searoute"]["version"]:
        raise AssertionError(
            f"Expected searoute {config['searoute']['version']}; "
            f"found {sr.__version__}"
        )
    package_root = Path(inspect.getfile(sr)).resolve().parent
    marnet_path = package_root / "data/marnet_dict.py"
    if sha256sum(marnet_path) != config["searoute"]["marnet_sha256"]:
        raise AssertionError("Frozen searoute maritime-network hash changed")

    frozen_inputs: list[dict[str, Any]] = []
    for group_name in ("c0a_inputs", "lineage_inputs"):
        for input_name, specification in config[group_name].items():
            path = validate_frozen_file(
                specification["path"], specification["sha256"]
            )
            frozen_inputs.append(
                {
                    "group": group_name,
                    "input": input_name,
                    "path": specification["path"],
                    "byte_size": path.stat().st_size,
                    "sha256": specification["sha256"],
                }
            )
    p8x = config["p8x_audit_input"]
    p8x_path = validate_frozen_file(p8x["path"], p8x["sha256"])
    frozen_inputs.append(
        {
            "group": "audit_only",
            "input": "p8x_city_coordinates",
            "path": p8x["path"],
            "byte_size": p8x_path.stat().st_size,
            "sha256": p8x["sha256"],
        }
    )
    for index, specification in enumerate(
        config["coordinate_source_precedence"]
    ):
        path = validate_frozen_file(
            specification["path"], specification["sha256"]
        )
        columns = pd.read_csv(path, nrows=0).columns
        missing = sorted(set(ANCHORAGE_COLUMNS).difference(columns))
        if missing:
            raise AssertionError(
                f"Coordinate source {path} is missing columns: {missing}"
            )
        frozen_inputs.append(
            {
                "group": "production_coordinate_source",
                "input": f"precedence_{index + 1}",
                "path": specification["path"],
                "byte_size": path.stat().st_size,
                "sha256": specification["sha256"],
            }
        )
    return {
        "branch": branch,
        "base_commit": git_value("rev-parse", "HEAD"),
        "frozen_inputs": frozen_inputs,
        "marnet_path": str(marnet_path),
        "marnet_sha256": config["searoute"]["marnet_sha256"],
    }


def load_coordinate_lookup(
    source_specs: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    source_summaries = []
    for precedence, specification in enumerate(source_specs, start=1):
        path = CODING_ROOT / specification["path"]
        frame = pd.read_csv(
            path,
            usecols=ANCHORAGE_COLUMNS,
            dtype={"s2id": "string"},
        )
        frame["s2id"] = clean_id(frame["s2id"])
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        frame["coordinate_source"] = path.name
        frame["coordinate_source_precedence"] = precedence
        invalid = (
            frame["s2id"].eq("")
            | frame["lat"].isna()
            | frame["lon"].isna()
            | ~frame["lat"].between(-90.0, 90.0)
            | ~frame["lon"].between(-180.0, 180.0)
        )
        frames.append(frame)
        source_summaries.append(
            {
                "coordinate_source": path.name,
                "precedence": precedence,
                "source_rows": len(frame),
                "unique_s2id": frame["s2id"].nunique(),
                "invalid_coordinate_rows": int(invalid.sum()),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    invalid = (
        combined["s2id"].eq("")
        | combined["lat"].isna()
        | combined["lon"].isna()
        | ~combined["lat"].between(-90.0, 90.0)
        | ~combined["lon"].between(-180.0, 180.0)
    )
    invalid_rows = combined.loc[invalid].copy()
    combined = combined.loc[~invalid].copy()
    lookup = (
        combined.sort_values(
            ["coordinate_source_precedence", "s2id"],
            kind="mergesort",
        )
        .drop_duplicates("s2id", keep="first")
        .reset_index(drop=True)
    )
    if lookup["s2id"].duplicated().any():
        raise AssertionError("Coordinate lookup contains duplicate s2id")
    return (
        lookup,
        pd.DataFrame(source_summaries),
        invalid_rows.reset_index(drop=True),
    )


def build_pair_city_universe(
    pair_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = sorted(set(PAIR_COLUMNS).difference(pair_manifest.columns))
    if required:
        raise AssertionError(f"Pair manifest is missing columns: {required}")
    if pair_manifest["pair_id"].duplicated().any():
        raise AssertionError("Pair manifest contains duplicate pair_id")
    origins = (
        pair_manifest.groupby("origin_city", sort=True)
        .size()
        .rename("origin_pair_count")
    )
    destinations = (
        pair_manifest.groupby("dest_city", sort=True)
        .size()
        .rename("dest_pair_count")
    )
    city_ids = sorted(
        set(pair_manifest["origin_city"]).union(pair_manifest["dest_city"])
    )
    cities = pd.DataFrame({"city_id": city_ids})
    split = cities["city_id"].str.split(":", n=1, expand=True)
    if split.shape[1] != 2 or split.isna().any().any():
        raise AssertionError("Production city IDs must use ISO3:CITY syntax")
    cities["iso3"] = split[0]
    cities["city_name"] = split[1]
    cities = cities.join(origins, on="city_id").join(destinations, on="city_id")
    cities[["origin_pair_count", "dest_pair_count"]] = cities[
        ["origin_pair_count", "dest_pair_count"]
    ].fillna(0).astype(int)
    cities["total_pair_endpoint_roles"] = (
        cities["origin_pair_count"] + cities["dest_pair_count"]
    )
    return cities, pair_manifest[PAIR_COLUMNS].copy()


def circular_weighted_mean_deg(
    values: pd.Series, weights: pd.Series
) -> float:
    angles = np.radians(values.to_numpy(dtype=float))
    weight_values = weights.to_numpy(dtype=float)
    x = np.average(np.cos(angles), weights=weight_values)
    y = np.average(np.sin(angles), weights=weight_values)
    result = math.degrees(math.atan2(y, x))
    return ((result + 180.0) % 360.0) - 180.0


def circular_span_deg(values: pd.Series) -> float:
    normalized = np.sort(np.mod(values.to_numpy(dtype=float), 360.0))
    if len(normalized) <= 1:
        return 0.0
    gaps = np.diff(np.r_[normalized, normalized[0] + 360.0])
    return float(360.0 - gaps.max())


def haversine_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(
        ((lon2 - lon1 + 180.0) % 360.0) - 180.0
    )
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def weighted_coordinate(group: pd.DataFrame) -> tuple[float, float]:
    weights = group["coordinate_weight_records"].astype(float)
    return (
        float(np.average(group["lat"], weights=weights)),
        circular_weighted_mean_deg(group["lon"], weights),
    )


def build_city_coordinates(
    calls: pd.DataFrame,
    cities: pd.DataFrame,
    lookup: pd.DataFrame,
    spread_threshold_deg: float,
    coordinate_role_overrides: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    missing_columns = sorted(set(CALL_COLUMNS).difference(calls.columns))
    if missing_columns:
        raise AssertionError(
            f"Ordered call table is missing columns: {missing_columns}"
        )
    if calls["n_events"].isna().any() or (calls["n_events"] <= 0).any():
        raise AssertionError("Ordered calls contain invalid n_events weights")

    production_city_ids = set(cities["city_id"])
    commercial = calls.loc[
        calls["is_commercial_endpoint"]
        & calls["topology_city_id"].isin(production_city_ids)
    ].copy()
    commercial["anchorage_id"] = clean_id(commercial["anchorage_id"])
    call_city_ids = set(commercial["topology_city_id"])
    if call_city_ids != production_city_ids:
        missing = sorted(production_city_ids.difference(call_city_ids))
        extra = sorted(call_city_ids.difference(production_city_ids))
        raise AssertionError(
            f"Pair/call city mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    raw_identity_variants = (
        commercial[
            ["topology_city_id", "country_iso3", "city_name"]
        ]
        .drop_duplicates()
        .groupby("topology_city_id", sort=True)
        .size()
    )
    anchorage_weights = (
        commercial.groupby(
            ["topology_city_id", "anchorage_id"],
            as_index=False,
            sort=True,
        )
        .agg(
            source_call_count=("n_events", "size"),
            coordinate_weight_records=("n_events", "sum"),
        )
        .merge(
            lookup,
            left_on="anchorage_id",
            right_on="s2id",
            how="left",
            validate="many_to_one",
        )
    )
    missing_coordinates = anchorage_weights.loc[
        anchorage_weights[["lat", "lon"]].isna().any(axis=1)
    ].copy()
    if not missing_coordinates.empty:
        examples = missing_coordinates.head().to_dict("records")
        raise AssertionError(
            f"Production anchorages lack coordinates: {examples}"
        )

    overrides = coordinate_role_overrides or []
    override_city_ids = [str(item["city_id"]) for item in overrides]
    if len(override_city_ids) != len(set(override_city_ids)):
        raise AssertionError("Coordinate-role overrides contain duplicate city_id")
    unknown_override_cities = sorted(
        set(override_city_ids).difference(production_city_ids)
    )
    if unknown_override_cities:
        raise AssertionError(
            "Coordinate-role overrides reference non-production cities: "
            f"{unknown_override_cities}"
        )
    override_by_city = {
        str(item["city_id"]): item
        for item in overrides
    }
    anchorage_weights["coordinate_role"] = "shared_port_and_routing"

    aggregate_rows: list[dict[str, Any]] = []
    for city_id, group in anchorage_weights.groupby(
        "topology_city_id", sort=True
    ):
        weights = group["coordinate_weight_records"].astype(float)
        port_group = group
        routing_group = group
        role_mode = "shared_port_and_routing"
        gateway_override = False
        gateway_reason = ""
        gateway_evidence = ""
        override = override_by_city.get(city_id)
        if override is not None:
            if override.get("mode") != "separate_port_and_gateway":
                raise AssertionError(
                    f"Unsupported coordinate-role mode for {city_id}: "
                    f"{override.get('mode')}"
                )
            group_ids = set(group["anchorage_id"])
            port_ids = {
                str(value).strip()
                for value in override["port_anchorage_ids"]
            }
            gateway_ids = {
                str(value).strip()
                for value in override["routing_gateway_anchorage_ids"]
            }
            if not port_ids or not gateway_ids:
                raise AssertionError(
                    f"Coordinate-role override has an empty role for {city_id}"
                )
            if port_ids.intersection(gateway_ids):
                raise AssertionError(
                    f"Coordinate-role override roles overlap for {city_id}"
                )
            if port_ids.union(gateway_ids) != group_ids:
                missing = sorted(group_ids.difference(port_ids, gateway_ids))
                extra = sorted(
                    port_ids.union(gateway_ids).difference(group_ids)
                )
                raise AssertionError(
                    f"Coordinate-role override does not partition {city_id}: "
                    f"missing={missing}, extra={extra}"
                )
            port_group = group.loc[group["anchorage_id"].isin(port_ids)]
            routing_group = group.loc[
                group["anchorage_id"].isin(gateway_ids)
            ]
            role_mode = "separate_port_and_gateway"
            gateway_override = True
            gateway_reason = str(override["reason"])
            gateway_evidence = str(override["evidence"])
            anchorage_weights.loc[
                (anchorage_weights["topology_city_id"] == city_id)
                & anchorage_weights["anchorage_id"].isin(port_ids),
                "coordinate_role",
            ] = "physical_port"
            anchorage_weights.loc[
                (anchorage_weights["topology_city_id"] == city_id)
                & anchorage_weights["anchorage_id"].isin(gateway_ids),
                "coordinate_role",
            ] = "routing_gateway"

        port_lat, port_lon = weighted_coordinate(port_group)
        routing_lat, routing_lon = weighted_coordinate(routing_group)
        lat_spread = float(group["lat"].max() - group["lat"].min())
        lon_spread = circular_span_deg(group["lon"])
        source_counts = group["coordinate_source"].value_counts()
        aggregate_rows.append(
            {
                "city_id": city_id,
                "original_lat": port_lat,
                "original_lon": port_lon,
                "port_lat": port_lat,
                "port_lon": port_lon,
                "routing_lat": routing_lat,
                "routing_lon": routing_lon,
                "coordinate_role_mode": role_mode,
                "routing_gateway_override": gateway_override,
                "routing_gateway_reason": gateway_reason,
                "routing_gateway_evidence": gateway_evidence,
                "port_coordinate_weight_records": int(
                    port_group["coordinate_weight_records"].sum()
                ),
                "routing_coordinate_weight_records": int(
                    routing_group["coordinate_weight_records"].sum()
                ),
                "n_anchorages": int(group["anchorage_id"].nunique()),
                "source_call_count": int(group["source_call_count"].sum()),
                "coordinate_weight_records": int(weights.sum()),
                "lat_spread_deg": lat_spread,
                "lon_circular_spread_deg": lon_spread,
                "coordinate_spread_deg": max(lat_spread, lon_spread),
                "spread_gt_2deg": max(lat_spread, lon_spread)
                > spread_threshold_deg,
                "n_anchorages_from_20201104": int(
                    source_counts.get(
                        "named_anchorages_v2_20201104.csv", 0
                    )
                ),
                "n_anchorages_from_20221206": int(
                    source_counts.get(
                        "named_anchorages_v2_20221206.csv", 0
                    )
                ),
                "raw_identity_variant_count": int(
                    raw_identity_variants.get(city_id, 0)
                ),
            }
        )
    aggregates = pd.DataFrame(aggregate_rows)
    output = cities.merge(
        aggregates,
        on="city_id",
        how="left",
        validate="one_to_one",
    )
    if output[
        [
            "original_lat",
            "original_lon",
            "routing_lat",
            "routing_lon",
            "coordinate_weight_records",
        ]
    ].isna().any().any():
        raise AssertionError("At least one production city lacks coordinates")

    points = gpd.GeoDataFrame(
        output[["city_id"]].copy(),
        geometry=gpd.points_from_xy(
            output["port_lon"], output["port_lat"]
        ),
        crs="EPSG:4326",
    )
    if points.crs is None or points.crs.to_epsg() != 4326:
        raise AssertionError("Production point CRS is not EPSG:4326")
    if (~points.geometry.is_valid).any() or points.geometry.is_empty.any():
        raise AssertionError("Production city coordinates contain invalid points")

    diagnostics = {
        "commercial_call_count": int(len(commercial)),
        "commercial_anchorage_count": int(
            commercial["anchorage_id"].nunique()
        ),
        "pair_city_count": int(len(cities)),
        "call_city_count": int(commercial["topology_city_id"].nunique()),
        "raw_identity_multivariant_city_count": int(
            (raw_identity_variants > 1).sum()
        ),
        "missing_anchorage_coordinate_count": int(
            len(missing_coordinates)
        ),
        "coordinate_role_override_count": int(
            output["routing_gateway_override"].sum()
        ),
        "coordinate_crs": "EPSG:4326",
        "distance_policy": (
            "No planar distance in geographic degrees; snap distance uses "
            "WGS84 longitude/latitude haversine"
        ),
    }
    lineage_columns = [
        "topology_city_id",
        "anchorage_id",
        "lat",
        "lon",
        "source_call_count",
        "coordinate_weight_records",
        "coordinate_source",
        "coordinate_source_precedence",
        "coordinate_role",
    ]
    lineage = anchorage_weights[lineage_columns].sort_values(
        ["topology_city_id", "anchorage_id"],
        kind="mergesort",
    )
    return output, lineage.reset_index(drop=True), diagnostics


def add_network_snaps(
    cities: pd.DataFrame,
    threshold_km: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    marnet = sr.setup_M(backend="networkx")
    records = []
    for row in cities.sort_values("city_id").itertuples(index=False):
        snapped_lon, snapped_lat = marnet.kdtree.query(
            (row.routing_lon, row.routing_lat)
        )
        snapped_lon = float(snapped_lon)
        snapped_lat = float(snapped_lat)
        snap_distance = haversine_km(
            row.routing_lon,
            row.routing_lat,
            snapped_lon,
            snapped_lat,
        )
        record = row._asdict()
        record.update(
            {
                "network_lon": snapped_lon,
                "network_lat": snapped_lat,
                "network_node_id": f"{snapped_lon:.6f},{snapped_lat:.6f}",
                "snap_distance_km": snap_distance,
                "snap_distance_gt_50km": snap_distance > threshold_km,
                "coordinate_crs": "EPSG:4326",
                "snap_distance_method": (
                    "WGS84 haversine; mean Earth radius 6371.0088 km"
                ),
                "routing_tool": "searoute",
                "routing_tool_version": sr.__version__,
            }
        )
        records.append(record)
    output = pd.DataFrame(records)
    if len(output) != cities["city_id"].nunique():
        raise AssertionError("Snap output does not contain one row per city")
    if output["city_id"].duplicated().any():
        raise AssertionError("Snap output contains duplicate city_id")
    if output[
        ["network_lon", "network_lat", "snap_distance_km"]
    ].isna().any().any():
        raise AssertionError("Snap output contains missing network fields")
    diagnostics = {
        "marnet_node_count": int(marnet.number_of_nodes()),
        "marnet_edge_count": int(marnet.number_of_edges()),
        "snap_distance_gt_50km_count": int(
            output["snap_distance_gt_50km"].sum()
        ),
        "median_snap_distance_km": float(
            output["snap_distance_km"].median()
        ),
        "p95_snap_distance_km": float(
            output["snap_distance_km"].quantile(0.95)
        ),
        "p99_snap_distance_km": float(
            output["snap_distance_km"].quantile(0.99)
        ),
        "max_snap_distance_km": float(
            output["snap_distance_km"].max()
        ),
    }
    return output, diagnostics


def build_p8x_comparison(
    production: pd.DataFrame,
    p8x_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "city_id",
        "lat",
        "lon",
        "network_lat",
        "network_lon",
        "snap_distance_km",
        "routing_tool_version",
        "gateway_override",
    ]
    p8x = pd.read_csv(p8x_path, usecols=columns)
    if p8x["city_id"].duplicated().any():
        raise AssertionError("P8x audit table contains duplicate city_id")
    renamed = p8x.rename(
        columns={
            column: f"p8x_{column}"
            for column in columns
            if column != "city_id"
        }
    )
    comparison = production[
        [
            "city_id",
            "original_lat",
            "original_lon",
            "routing_lat",
            "routing_lon",
            "routing_gateway_override",
            "network_lat",
            "network_lon",
            "snap_distance_km",
        ]
    ].merge(renamed, on="city_id", how="left", validate="one_to_one")
    comparison["p8x_present"] = comparison["p8x_lat"].notna()
    comparison["production_coordinate_source"] = np.where(
        comparison["routing_gateway_override"],
        (
            "2021 ordered commercial calls + legacy anchorage coordinates; "
            "approved port/gateway role separation"
        ),
        "2021 ordered commercial calls + legacy anchorage coordinates",
    )
    comparison["p8x_used_in_production"] = False

    present = comparison["p8x_present"]
    comparison["coordinate_difference_km"] = np.nan
    comparison["routing_coordinate_difference_km"] = np.nan
    comparison["network_node_difference_km"] = np.nan
    comparison.loc[present, "coordinate_difference_km"] = comparison.loc[
        present
    ].apply(
        lambda row: haversine_km(
            row["original_lon"],
            row["original_lat"],
            row["p8x_lon"],
            row["p8x_lat"],
        ),
        axis=1,
    )
    comparison.loc[present, "routing_coordinate_difference_km"] = (
        comparison.loc[present].apply(
            lambda row: haversine_km(
                row["routing_lon"],
                row["routing_lat"],
                row["p8x_lon"],
                row["p8x_lat"],
            ),
            axis=1,
        )
    )
    comparison.loc[present, "network_node_difference_km"] = comparison.loc[
        present
    ].apply(
        lambda row: haversine_km(
            row["network_lon"],
            row["network_lat"],
            row["p8x_network_lon"],
            row["p8x_network_lat"],
        ),
        axis=1,
    )
    comparison["network_node_exact_match"] = (
        present
        & comparison["network_lon"].eq(comparison["p8x_network_lon"])
        & comparison["network_lat"].eq(comparison["p8x_network_lat"])
    )
    gateway_override = (
        comparison["p8x_gateway_override"]
        .astype("boolean")
        .fillna(False)
    )
    if comparison["p8x_used_in_production"].any():
        raise AssertionError("P8x audit values entered production fields")
    summary = {
        "production_city_count": int(len(comparison)),
        "p8x_overlap_city_count": int(present.sum()),
        "p8x_missing_city_count": int((~present).sum()),
        "network_node_exact_match_count": int(
            comparison["network_node_exact_match"].sum()
        ),
        "p8x_gateway_override_city_count": int(
            (present & gateway_override).sum()
        ),
        "p8x_gateway_override_node_mismatch_count": int(
            (
                present
                & gateway_override
                & ~comparison["network_node_exact_match"]
            ).sum()
        ),
        "p8x_non_gateway_node_mismatch_count": int(
            (
                present
                & ~gateway_override
                & ~comparison["network_node_exact_match"]
            ).sum()
        ),
        "median_coordinate_difference_km": float(
            comparison.loc[present, "coordinate_difference_km"].median()
        ),
        "p95_coordinate_difference_km": float(
            comparison.loc[present, "coordinate_difference_km"].quantile(0.95)
        ),
        "max_coordinate_difference_km": float(
            comparison.loc[present, "coordinate_difference_km"].max()
        ),
        "median_routing_coordinate_difference_km": float(
            comparison.loc[
                present, "routing_coordinate_difference_km"
            ].median()
        ),
        "p95_routing_coordinate_difference_km": float(
            comparison.loc[
                present, "routing_coordinate_difference_km"
            ].quantile(0.95)
        ),
        "max_routing_coordinate_difference_km": float(
            comparison.loc[
                present, "routing_coordinate_difference_km"
            ].max()
        ),
        "p95_network_node_difference_km": float(
            comparison.loc[
                present, "network_node_difference_km"
            ].quantile(0.95)
        ),
        "max_network_node_difference_km": float(
            comparison.loc[
                present, "network_node_difference_km"
            ].max()
        ),
    }
    return comparison, summary


def render_audit_markdown(summary: dict[str, Any], top: pd.DataFrame) -> str:
    lines = [
        "# C0.3 2021 Production City-Snap Audit",
        "",
        "Status: PASS",
        "",
        "## Production Result",
        "",
        f"- Production cities: {summary['production_city_count']:,}",
        f"- Directed production pairs retained: {summary['pair_count']:,}",
        f"- Commercial calls used: {summary['commercial_call_count']:,}",
        f"- Production anchorages used: {summary['commercial_anchorage_count']:,}",
        f"- Missing production city coordinates: {summary['missing_city_coordinate_count']:,}",
        f"- Missing network snaps: {summary['missing_network_snap_count']:,}",
        f"- Invalid source-coordinate rows isolated: {summary['invalid_source_coordinate_row_count']:,}",
        f"- Snap distance over 50 km: {summary['snap_distance_gt_50km_count']:,}",
        f"- Coordinate spread over 2 degrees: {summary['spread_gt_2deg_count']:,}",
        f"- Approved port/gateway role separations: {summary['coordinate_role_override_count']:,}",
        "",
        "All cities and pairs remain in the production universe. The 50 km",
        "threshold is diagnostic only and no row was excluded.",
        "",
        "## Distance Method",
        "",
        "- Coordinates use EPSG:4326 longitude/latitude.",
        "- Physical-port and routing coordinates are weighted by retained 2021",
        "  OD endpoint source records (`n_events`) represented by each ordered",
        "  commercial call.",
        "- The routing coordinate equals the physical-port coordinate unless a",
        "  documented coordinate-role separation is frozen in the config.",
        "- Longitude uses a circular weighted mean.",
        "- Network snapping uses the frozen searoute 1.6.0 maritime KD-tree.",
        "- Network snapping starts from the routing coordinate, not necessarily",
        "  the physical-port coordinate.",
        "- Snap distance uses WGS84 longitude/latitude haversine distance, not",
        "  planar degree distance.",
        "",
        "## Snap Distribution",
        "",
        f"- Median: {summary['median_snap_distance_km']:.2f} km",
        f"- p95: {summary['p95_snap_distance_km']:.2f} km",
        f"- p99: {summary['p99_snap_distance_km']:.2f} km",
        f"- Maximum: {summary['max_snap_distance_km']:.2f} km",
        "",
        "## P8x Audit-Only Comparison",
        "",
        f"- Overlap cities: {summary['p8x_overlap_city_count']:,}",
        f"- Cities absent from P8x: {summary['p8x_missing_city_count']:,}",
        f"- Exact snapped-node matches: {summary['network_node_exact_match_count']:,}",
        f"- P8x gateway overrides: {summary['p8x_gateway_override_city_count']:,}",
        f"- Node mismatches with a P8x gateway override: {summary['p8x_gateway_override_node_mismatch_count']:,}",
        f"- Node mismatches without a P8x gateway override: {summary['p8x_non_gateway_node_mismatch_count']:,}",
        f"- Median physical-port coordinate difference: {summary['median_coordinate_difference_km']:.2f} km",
        f"- p95 physical-port coordinate difference: {summary['p95_coordinate_difference_km']:.2f} km",
        f"- Median routing-coordinate difference: {summary['median_routing_coordinate_difference_km']:.2f} km",
        f"- p95 routing-coordinate difference: {summary['p95_routing_coordinate_difference_km']:.2f} km",
        f"- p95 snapped-node difference: {summary['p95_network_node_difference_km']:.2f} km",
        "",
        "P8x coordinates and snaps were used only after production construction",
        "for comparison. No P8x value was copied into a production field.",
        "",
        "## Largest Snap Distances",
        "",
        "| City | Snap km | Pair endpoint roles | Anchorages | Spread deg |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.city_id} | {row.snap_distance_km:.2f} | "
            f"{row.total_pair_endpoint_roles:,} | {row.n_anchorages:,} | "
            f"{row.coordinate_spread_deg:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Stop Point",
            "",
            "C0.3 is complete. The 30-pair benchmark and full production routing",
            "have not started and require the next explicit authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    config: dict[str, Any],
    artifact_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    contract = validate_contract(config)
    lookup, source_summary, invalid_source_coordinates = load_coordinate_lookup(
        config["coordinate_source_precedence"]
    )
    calls_spec = config["c0a_inputs"]["ordered_port_calls"]
    pairs_spec = config["c0a_inputs"]["passage_pair_manifest"]
    calls = pd.read_feather(
        CODING_ROOT / calls_spec["path"],
        columns=CALL_COLUMNS,
    )
    pairs = pd.read_csv(
        CODING_ROOT / pairs_spec["path"],
        usecols=PAIR_COLUMNS,
    )
    cities, pair_rows = build_pair_city_universe(pairs)
    city_coordinates, lineage, coordinate_diagnostics = (
        build_city_coordinates(
            calls,
            cities,
            lookup,
            float(config["coordinate_spread_review_threshold_deg"]),
            config.get("coordinate_role_overrides", []),
        )
    )
    production, snap_diagnostics = add_network_snaps(
        city_coordinates,
        float(config["snap_review_threshold_km"]),
    )
    p8x_comparison, p8x_summary = build_p8x_comparison(
        production,
        CODING_ROOT / config["p8x_audit_input"]["path"],
    )

    over_50 = production.loc[
        production["snap_distance_gt_50km"]
    ].sort_values(
        ["snap_distance_km", "city_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    spread_review = production.loc[
        production["spread_gt_2deg"]
    ].sort_values(
        ["coordinate_spread_deg", "city_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    if len(production) != len(cities):
        raise AssertionError("Production city set changed during C0.3")
    production_pairs = set(pair_rows["pair_id"])
    if len(production_pairs) != len(pair_rows):
        raise AssertionError("Production pair set is not unique")

    summary = {
        "status": "pass",
        "analysis_year": 2021,
        "production_city_count": int(len(production)),
        "pair_count": int(len(pair_rows)),
        "missing_city_coordinate_count": int(
            production[["original_lat", "original_lon"]].isna().any(axis=1).sum()
        ),
        "missing_network_snap_count": int(
            production[["network_lat", "network_lon"]].isna().any(axis=1).sum()
        ),
        "invalid_source_coordinate_row_count": int(
            len(invalid_source_coordinates)
        ),
        "invalid_source_coordinate_s2id_count": int(
            invalid_source_coordinates["s2id"].nunique()
        ),
        "spread_gt_2deg_count": int(len(spread_review)),
        "snap_review_threshold_km": float(
            config["snap_review_threshold_km"]
        ),
        "coordinate_spread_review_threshold_deg": float(
            config["coordinate_spread_review_threshold_deg"]
        ),
        "coordinate_weight": config["coordinate_weight"],
        "coordinate_source_precedence": config[
            "coordinate_source_precedence"
        ],
        "production_pair_set_sha256": object_sha256(
            sorted(production_pairs)
        ),
        "production_city_set_sha256": object_sha256(
            sorted(production["city_id"])
        ),
        "base_commit": contract["base_commit"],
        "runner_sha256": sha256sum(Path(__file__).resolve()),
        "config_sha256": object_sha256(config),
        **coordinate_diagnostics,
        **snap_diagnostics,
        **p8x_summary,
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    production_path = artifact_dir / "city_snap_table_2021.csv"
    lineage_path = artifact_dir / "city_coordinate_anchorage_lineage_2021.csv"
    write_csv_atomic(production_path, production)
    write_csv_atomic(lineage_path, lineage)
    write_csv_atomic(
        report_dir / "city_snap_over_50km.csv",
        over_50,
    )
    write_csv_atomic(
        report_dir / "city_coordinate_spread_over_2deg.csv",
        spread_review,
    )
    write_csv_atomic(
        report_dir / "p8x_coordinate_comparison.csv",
        p8x_comparison,
    )
    write_csv_atomic(
        report_dir / "city_coordinate_source_summary.csv",
        source_summary,
    )
    write_csv_atomic(
        report_dir / "city_coordinate_source_invalid_rows.csv",
        invalid_source_coordinates,
    )
    write_text_atomic(
        report_dir / "city_snap_audit.md",
        render_audit_markdown(
            summary,
            over_50.head(20),
        ),
    )
    write_json_atomic(
        report_dir / "city_snap_summary.json",
        summary,
    )

    output_paths = [
        production_path,
        lineage_path,
        report_dir / "city_snap_over_50km.csv",
        report_dir / "city_coordinate_spread_over_2deg.csv",
        report_dir / "p8x_coordinate_comparison.csv",
        report_dir / "city_coordinate_source_summary.csv",
        report_dir / "city_coordinate_source_invalid_rows.csv",
        report_dir / "city_snap_audit.md",
        report_dir / "city_snap_summary.json",
    ]
    output_manifest = {
        "status": "pass",
        "outputs": {
            str(path.relative_to(CODING_ROOT)): {
                "byte_size": path.stat().st_size,
                "sha256": sha256sum(path),
            }
            for path in output_paths
        },
    }
    write_json_atomic(
        artifact_dir / "city_snap_output_manifest.json",
        output_manifest,
    )
    return {**summary, "output_manifest": output_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build independent 2021 analytical-city coordinates and snaps."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    config_path = (
        args.config
        if args.config.is_absolute()
        else CODING_ROOT / args.config
    )
    artifact_dir = (
        args.artifact_dir
        if args.artifact_dir.is_absolute()
        else CODING_ROOT / args.artifact_dir
    )
    report_dir = (
        args.report_dir
        if args.report_dir.is_absolute()
        else CODING_ROOT / args.report_dir
    )
    result = run(load_json(config_path), artifact_dir, report_dir)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "production_city_count",
                    "pair_count",
                    "snap_distance_gt_50km_count",
                    "spread_gt_2deg_count",
                    "p8x_overlap_city_count",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
