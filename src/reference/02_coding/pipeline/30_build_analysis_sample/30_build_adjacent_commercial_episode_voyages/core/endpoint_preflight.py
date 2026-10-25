"""Audit 2025 C5 input coverage before building any candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


CODING_ROOT = Path(__file__).resolve().parents[4]
C5_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = C5_ROOT / "config/frozen/endpoint_validation_2025.json"
DEFAULT_REPORT_DIR = DATA_ROOT / "main/validation/endpoint_2025"
SENTINEL_ANCHORAGES = {"10000001", "NO_PREVIOUS_DATA", ""}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def normalize_token(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def resolve_od_schema(path: Path, aliases: dict[str, list[str]]) -> dict[str, str]:
    header = list(pd.read_csv(path, nrows=0).columns)
    resolved: dict[str, str] = {}
    for canonical, choices in aliases.items():
        matches = [column for column in choices if column in header]
        if len(matches) != 1:
            raise ValueError(
                f"{path.name}: expected one alias for {canonical}; found {matches}"
            )
        resolved[canonical] = matches[0]
    return resolved


def load_mapping(path: Path) -> pd.DataFrame:
    columns = ["anchorage_id", "topology_city_id", "endpoint_eligible"]
    mapping = pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    mapping["anchorage_id"] = normalize_token(mapping["anchorage_id"])
    mapping["topology_city_id"] = normalize_token(mapping["topology_city_id"])
    mapping["endpoint_eligible_bool"] = mapping[
        "endpoint_eligible"
    ].str.lower().isin({"true", "1", "yes"})
    if mapping["anchorage_id"].duplicated().any():
        raise ValueError("V3 mapping contains duplicate anchorage IDs")
    return mapping


def audit_od_files(
    paths: list[Path],
    analysis_mmsi: set[str],
    mapping_lookup: dict[str, str],
    all_mapped_anchorages: set[str],
    aliases: dict[str, list[str]],
    chunk_rows: int,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rows: list[dict[str, Any]] = []
    retained_vessels: set[str] = set()
    endpoint_anchorages: set[str] = set()
    non_endpoint_anchorages: set[str] = set()
    absent_mapping_anchorages: set[str] = set()
    mapped_cities: set[str] = set()
    direct_city_pairs: set[str] = set()
    mapped_city_anchorages: set[tuple[str, str]] = set()

    for path in paths:
        schema = resolve_od_schema(path, aliases)
        raw_rows = 0
        retained_rows = 0
        retained_file_vessels: set[str] = set()
        file_anchorages: set[str] = set()
        file_non_endpoint: set[str] = set()
        file_absent_mapping: set[str] = set()
        reader = pd.read_csv(
            path,
            usecols=list(schema.values()),
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_rows,
            low_memory=False,
        )
        rename = {source: target for target, source in schema.items()}
        for chunk in reader:
            raw_rows += len(chunk)
            chunk = chunk.rename(columns=rename)
            for column in ("mmsi", "origin_anchorage", "dest_anchorage"):
                chunk[column] = normalize_token(chunk[column])
            selected = chunk.loc[chunk["mmsi"].isin(analysis_mmsi)].copy()
            if selected.empty:
                continue
            retained_rows += len(selected)
            retained_file_vessels.update(selected["mmsi"])
            for column in ("origin_anchorage", "dest_anchorage"):
                values = set(selected[column]) - SENTINEL_ANCHORAGES
                file_anchorages.update(values)
            origin_city = selected["origin_anchorage"].map(mapping_lookup)
            dest_city = selected["dest_anchorage"].map(mapping_lookup)
            valid_pair = origin_city.notna() & dest_city.notna()
            mapped_cities.update(origin_city.dropna())
            mapped_cities.update(dest_city.dropna())
            mapped_city_anchorages.update(
                zip(
                    origin_city.loc[origin_city.notna()],
                    selected.loc[origin_city.notna(), "origin_anchorage"],
                    strict=True,
                )
            )
            mapped_city_anchorages.update(
                zip(
                    dest_city.loc[dest_city.notna()],
                    selected.loc[dest_city.notna(), "dest_anchorage"],
                    strict=True,
                )
            )
            direct_city_pairs.update(
                origin_city.loc[valid_pair] + "__" + dest_city.loc[valid_pair]
            )
        file_absent_mapping.update(file_anchorages - all_mapped_anchorages)
        file_non_endpoint.update(
            (file_anchorages & all_mapped_anchorages) - set(mapping_lookup)
        )
        retained_vessels.update(retained_file_vessels)
        endpoint_anchorages.update(file_anchorages)
        non_endpoint_anchorages.update(file_non_endpoint)
        absent_mapping_anchorages.update(file_absent_mapping)
        rows.append(
            {
                "source_file": path.name,
                "raw_rows": raw_rows,
                "analysis_vessel_rows": retained_rows,
                "analysis_vessels": len(retained_file_vessels),
                "endpoint_anchorages": len(file_anchorages),
                "non_endpoint_anchorages": len(file_non_endpoint),
                "absent_mapping_anchorages": len(file_absent_mapping),
                "mmsi_column": schema["mmsi"],
                "origin_anchorage_column": schema["origin_anchorage"],
                "dest_anchorage_column": schema["dest_anchorage"],
            }
        )
    sets = {
        "retained_vessels": retained_vessels,
        "endpoint_anchorages": endpoint_anchorages,
        "non_endpoint_anchorages": non_endpoint_anchorages,
        "absent_mapping_anchorages": absent_mapping_anchorages,
        "mapped_cities": mapped_cities,
        "direct_city_pairs": direct_city_pairs,
        "mapped_city_anchorages": mapped_city_anchorages,
    }
    return pd.DataFrame(rows), sets


def coordinate_anchorage_ids(paths: list[Path]) -> set[str]:
    available: set[str] = set()
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=["s2id", "lat", "lon"],
            dtype={"s2id": str},
            low_memory=False,
        )
        frame["s2id"] = normalize_token(frame["s2id"])
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        valid = (
            frame["s2id"].ne("")
            & frame["lat"].between(-90.0, 90.0)
            & frame["lon"].between(-180.0, 180.0)
        )
        available.update(frame.loc[valid, "s2id"])
    return available


def build_summary(
    config: dict[str, Any],
    file_audit: pd.DataFrame,
    coverage: dict[str, set[str]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    city_snap = pd.read_csv(
        CODING_ROOT / config["city_snap_2021"],
        usecols=["city_id"],
        dtype=str,
    )
    known_cities = set(city_snap["city_id"])
    yardstick = pd.read_feather(
        CODING_ROOT / config["yardstick_2021"],
        columns=["pair_id"],
    )
    yardstick_pairs = set(yardstick["pair_id"])
    p4 = pd.read_feather(
        CODING_ROOT / config["p4_city_shiptype_exposure"],
        columns=[
            "city_pair",
            "n_2021_valid",
            "BaselineSuezDependence_g",
            "n_2025_valid",
        ],
    )
    p4_product_pairs = set(p4["city_pair"])
    p4_baseline_pairs = set(
        p4.loc[
            p4["n_2021_valid"].gt(0)
            & p4["BaselineSuezDependence_g"].notna(),
            "city_pair",
        ]
    )
    p4_2025_pairs = set(p4.loc[p4["n_2025_valid"].gt(0), "city_pair"])
    direct_pairs_all = coverage["direct_city_pairs"]
    same_city_pairs = {
        pair
        for pair in direct_pairs_all
        if pair.split("__", 1)[0] == pair.split("__", 1)[1]
    }
    direct_pairs = direct_pairs_all - same_city_pairs
    support_classes = {
        "supported_by_2021_c0_and_p4": direct_pairs
        & yardstick_pairs
        & p4_baseline_pairs,
        "in_2021_c0_but_no_p4_baseline": (direct_pairs & yardstick_pairs)
        - p4_baseline_pairs,
        "not_2021_c0_but_has_p4_baseline": (direct_pairs - yardstick_pairs)
        & p4_baseline_pairs,
        "no_2021_c0_or_p4_baseline": direct_pairs
        - yardstick_pairs
        - p4_baseline_pairs,
    }
    support_summary = pd.DataFrame(
        [
            {
                "support_class": name,
                "direct_intercity_pairs": len(pairs),
                "pairs_with_p4_2025_valid": len(pairs & p4_2025_pairs),
                "pairs_without_p4_2025_valid": len(pairs - p4_2025_pairs),
            }
            for name, pairs in support_classes.items()
        ]
    )
    missing_cities = coverage["mapped_cities"] - known_cities
    coordinate_ids = coordinate_anchorage_ids(
        [CODING_ROOT / value for value in config["coordinate_sources"]]
    )
    city_anchorages = pd.DataFrame(
        sorted(coverage["mapped_city_anchorages"]),
        columns=["city_id", "anchorage_id"],
    )
    missing_city_coverage = (
        city_anchorages.loc[city_anchorages["city_id"].isin(missing_cities)]
        .assign(
            coordinate_available=lambda frame: frame["anchorage_id"].isin(
                coordinate_ids
            )
        )
        .groupby("city_id", as_index=False, sort=True)
        .agg(
            endpoint_anchorages=("anchorage_id", "nunique"),
            coordinate_anchorages=("coordinate_available", "sum"),
        )
    )
    missing_city_coverage["coordinate_ready"] = (
        missing_city_coverage["coordinate_anchorages"] > 0
    )
    summary = {
        "analysis_year": 2025,
        "operational_parameter_year": 2021,
        "od_files": len(file_audit),
        "raw_od_rows": int(file_audit["raw_rows"].sum()),
        "analysis_vessel_od_rows": int(file_audit["analysis_vessel_rows"].sum()),
        "analysis_vessels_observed": len(coverage["retained_vessels"]),
        "endpoint_anchorages": len(coverage["endpoint_anchorages"]),
        "non_endpoint_anchorages": len(coverage["non_endpoint_anchorages"]),
        "absent_mapping_anchorages": len(
            coverage["absent_mapping_anchorages"]
        ),
        "mapped_cities": len(coverage["mapped_cities"]),
        "mapped_cities_missing_2021_snap": len(missing_cities),
        "missing_snap_cities_coordinate_ready": int(
            missing_city_coverage["coordinate_ready"].sum()
        ),
        "missing_snap_cities_without_coordinates": int(
            (~missing_city_coverage["coordinate_ready"]).sum()
        ),
        "direct_city_pairs_including_same_city": len(direct_pairs_all),
        "same_city_direct_pairs": len(same_city_pairs),
        "direct_intercity_pairs": len(direct_pairs),
        "intercity_pairs_in_2021_yardstick": len(
            direct_pairs & yardstick_pairs
        ),
        "intercity_pairs_missing_2021_yardstick": len(
            direct_pairs - yardstick_pairs
        ),
        "intercity_pairs_in_p4_product": len(
            direct_pairs & p4_product_pairs
        ),
        "intercity_pairs_with_p4_2021_baseline": len(
            direct_pairs & p4_baseline_pairs
        ),
        "intercity_pairs_without_p4_2021_baseline": len(
            direct_pairs - p4_baseline_pairs
        ),
        "intercity_pairs_supported_by_2021_c0_and_p4": len(
            support_classes["supported_by_2021_c0_and_p4"]
        ),
        "intercity_pairs_in_2021_c0_but_no_p4_baseline": len(
            support_classes["in_2021_c0_but_no_p4_baseline"]
        ),
        "intercity_pairs_not_2021_c0_but_has_p4_baseline": len(
            support_classes["not_2021_c0_but_has_p4_baseline"]
        ),
        "intercity_pairs_no_2021_c0_or_p4_baseline": len(
            support_classes["no_2021_c0_or_p4_baseline"]
        ),
        "candidate_pair_coverage_pending": True,
        "shadow_parameter_role": "shadow_diagnostic_only",
        "2025_values_can_replace_2021_rules": False,
    }
    return summary, missing_city_coverage, support_summary


def run(
    config_path: Path,
    report_dir: Path,
    *,
    skip_content_hash: bool,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["analysis_year"]) != 2025:
        raise ValueError("C5 preflight is restricted to 2025")
    od_paths = [CODING_ROOT / value for value in config["od_files"]]
    if len(od_paths) != 12 or any(not path.exists() for path in od_paths):
        raise ValueError("C5 requires exactly 12 existing 2025 OD files")
    if any("/2022/" in str(path) or "/2023/" in str(path) or "/2024/" in str(path) for path in od_paths):
        raise ValueError("C5 must not read 2022-2024 OD data")

    vessel_path = CODING_ROOT / config["analysis_vessel_specs_2025"]
    vessels = pd.read_csv(
        vessel_path,
        usecols=["mmsi_clean"],
        dtype=str,
        keep_default_na=False,
    )
    vessels["mmsi_clean"] = normalize_token(vessels["mmsi_clean"])
    analysis_mmsi = set(vessels["mmsi_clean"])
    if len(analysis_mmsi) != int(config["expected_analysis_vessels"]):
        raise ValueError(
            f"Expected {config['expected_analysis_vessels']} vessels; "
            f"found {len(analysis_mmsi)}"
        )

    mapping_path = CODING_ROOT / config["anchorage_mapping"]
    mapping = load_mapping(mapping_path)
    eligible_mapping = mapping.loc[
        mapping["endpoint_eligible_bool"] & mapping["topology_city_id"].ne("")
    ]
    mapping_lookup = dict(
        zip(
            eligible_mapping["anchorage_id"],
            eligible_mapping["topology_city_id"],
            strict=True,
        )
    )
    file_audit, coverage = audit_od_files(
        od_paths,
        analysis_mmsi,
        mapping_lookup,
        set(mapping["anchorage_id"]),
        config["od_column_aliases"],
        int(config["chunk_rows"]),
    )
    summary, missing_city_coverage, support_summary = build_summary(
        config, file_audit, coverage
    )

    input_paths = [
        *od_paths,
        vessel_path,
        mapping_path,
        CODING_ROOT / config["city_snap_2021"],
        CODING_ROOT / config["yardstick_2021"],
        CODING_ROOT / config["p4_city_shiptype_exposure"],
        *[CODING_ROOT / value for value in config["coordinate_sources"]],
    ]
    manifest = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(CODING_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": "" if skip_content_hash else sha256_file(path),
                "content_hash_skipped": bool(skip_content_hash),
            }
            for path in input_paths
        ]
    )
    write_csv_atomic(report_dir / "c5_0_od_file_audit_2025.csv", file_audit)
    write_csv_atomic(
        report_dir / "c5_0_non_endpoint_anchorages_2025.csv",
        pd.DataFrame(
            {"anchorage_id": sorted(coverage["non_endpoint_anchorages"])}
        ),
    )
    write_csv_atomic(
        report_dir / "c5_0_anchorages_absent_mapping_2025.csv",
        pd.DataFrame(
            {"anchorage_id": sorted(coverage["absent_mapping_anchorages"])}
        ),
    )
    write_csv_atomic(
        report_dir / "c5_0_missing_city_coordinate_coverage.csv",
        missing_city_coverage,
    )
    write_csv_atomic(
        report_dir / "c5_0_direct_pair_support_summary.csv",
        support_summary,
    )
    write_csv_atomic(report_dir / "c5_0_input_manifest.csv", manifest)
    write_json_atomic(report_dir / "c5_0_preflight_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--skip-content-hash",
        action="store_true",
        help="Skip full-file hashes for a development coverage probe.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config_path = (
        arguments.config
        if arguments.config.is_absolute()
        else CODING_ROOT / arguments.config
    )
    report_dir = (
        arguments.report_dir
        if arguments.report_dir.is_absolute()
        else CODING_ROOT / arguments.report_dir
    )
    summary = run(
        config_path,
        report_dir,
        skip_content_hash=arguments.skip_content_hash,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
