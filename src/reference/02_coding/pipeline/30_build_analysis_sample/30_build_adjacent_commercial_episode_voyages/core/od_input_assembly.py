"""Assemble and guard the five-year E030 OD input seam.

The accepted adapter expects twelve monthly CSV files per departure year.  The
E030 stage-30/10 repair instead emits one harmonized Feather file per year, and
the five files live in three different roots.  This module turns those exact
files into one isolated, manifest-backed input root without touching accepted
products.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd


CODING_ROOT = Path(__file__).resolve().parents[4]
EXPECTED_YEARS = (2021, 2022, 2023, 2024, 2025)
BUFFERED_YEARS = frozenset((2021, 2022, 2023, 2024))
ACCEPTED_ADAPTED_OD_ROOT = CODING_ROOT / "data/processed/products/our_od"
VESSEL_SPEC_TEMPLATE = (
    CODING_ROOT
    / "data/processed/products/vessel_universe/analysis_vessel_specs_{year}.csv"
)
DEFAULT_SOURCES = {
    2021: CODING_ROOT
    / ".runs/2026-08-24_0313_E030_cross-year-stage30-four-years/port_calls/2021/"
    "harmonized_od_voyages_2021.feather",
    2022: CODING_ROOT
    / ".runs/2026-08-24_0251_E030_cross-year-stage30-2022/port_calls/2022/"
    "harmonized_od_voyages_2022.feather",
    2023: CODING_ROOT
    / ".runs/2026-08-24_0313_E030_cross-year-stage30-four-years/port_calls/2023/"
    "harmonized_od_voyages_2023.feather",
    2024: CODING_ROOT
    / ".runs/2026-08-24_0313_E030_cross-year-stage30-four-years/port_calls/2024/"
    "harmonized_od_voyages_2024.feather",
    2025: CODING_ROOT
    / "data/processed/products/port_calls_and_legs/2025/"
    "harmonized_od_voyages_2025.feather",
}

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
ADAPTED_COLUMNS = ["mmsi", *RENAME]
SOURCE_COLUMNS = [
    "vessel_uid",
    "internal_voyage_id",
    "trip_start",
    "trip_end",
    "origin_anchorage_id",
    "dest_anchorage_id",
    "origin_visit_id_internal",
    "dest_visit_id_internal",
    "origin_analytical_city_id",
    "dest_analytical_city_id",
]
DOUBLE_PREFIX_PATTERN = r"^\d{4}:\d{4}:"


class AssemblyError(ValueError):
    """Raised when the five-year OD seam violates its manifest contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".csv", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def create_source_link(link: Path, source: Path) -> None:
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise AssemblyError(f"source link points to the wrong file: {link}")
        return
    if link.exists():
        raise AssemblyError(f"source link path is occupied: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp-link")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source.resolve())
    temporary.replace(link)


def _read_source(path: Path, year: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not path.is_file():
        raise AssemblyError(f"missing {year} OD source: {path}")
    frame = pd.read_feather(path, columns=SOURCE_COLUMNS)
    if frame["internal_voyage_id"].isna().any():
        raise AssemblyError(f"{year} OD source contains a blank voyage ID")
    if frame["internal_voyage_id"].astype(str).duplicated().any():
        raise AssemblyError(f"{year} OD source duplicates internal_voyage_id")
    starts = pd.to_datetime(frame["trip_start"], errors="coerce", utc=True)
    ends = pd.to_datetime(frame["trip_end"], errors="coerce", utc=True)
    if starts.isna().any() or ends.isna().any():
        raise AssemblyError(f"{year} OD source contains an invalid timestamp")
    if starts.dt.year.ne(year).any():
        raise AssemblyError(f"{year} OD source contains a departure outside its cohort")
    if ends.le(starts).any():
        raise AssemblyError(f"{year} OD source contains non-positive voyage duration")
    for column in ("origin_visit_id_internal", "dest_visit_id_internal"):
        values = frame[column].astype("string")
        if values.str.match(DOUBLE_PREFIX_PATTERN, na=False).any():
            raise AssemblyError(f"{year} OD source contains a double-prefixed {column}")
    return frame, starts, ends


def _adapt_source(
    frame: pd.DataFrame,
    starts: pd.Series,
    ends: pd.Series,
    year: int,
    vessel_spec_root: Path,
) -> pd.DataFrame:
    specs_path = vessel_spec_root / f"analysis_vessel_specs_{year}.csv"
    specs = pd.read_csv(
        specs_path,
        usecols=["vessel_uid_clean", "mmsi_clean"],
        dtype=str,
    ).dropna()
    if specs["vessel_uid_clean"].duplicated().any():
        raise AssemblyError(f"{specs_path} duplicates vessel_uid_clean")
    source = frame.copy()
    source["vessel_uid"] = source["vessel_uid"].astype(str)
    merged = source.merge(
        specs.rename(
            columns={"vessel_uid_clean": "vessel_uid", "mmsi_clean": "mmsi"}
        ),
        on="vessel_uid",
        how="left",
        validate="many_to_one",
    )
    missing = int(merged["mmsi"].isna().sum())
    if missing:
        raise AssemblyError(f"{year} has {missing:,} OD rows without an accepted MMSI")
    adapted = pd.DataFrame({"mmsi": merged["mmsi"].astype(str)})
    for target, source_column in RENAME.items():
        adapted[target] = merged[source_column].astype(str)
    adapted["trip_start"] = starts.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    adapted["trip_end"] = ends.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return adapted[ADAPTED_COLUMNS]


def _read_accepted_adapted(year: int, accepted_root: Path) -> pd.DataFrame:
    paths = sorted((accepted_root / str(year)).glob(f"voyages_{year}??.csv"))
    if len(paths) != 12:
        raise AssemblyError(f"accepted adapted OD has {len(paths)} files for {year}")
    return pd.concat(
        [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths],
        ignore_index=True,
    )[ADAPTED_COLUMNS]


def _assert_accepted_rows_unchanged(
    adapted: pd.DataFrame,
    year: int,
    accepted_root: Path,
) -> tuple[int, int]:
    accepted = _read_accepted_adapted(year, accepted_root)
    if accepted["trip_id"].duplicated().any():
        raise AssemblyError(f"accepted adapted OD duplicates trip_id in {year}")
    if adapted["trip_id"].duplicated().any():
        raise AssemblyError(f"assembled adapted OD duplicates trip_id in {year}")
    comparison = accepted.merge(
        adapted,
        on="trip_id",
        how="left",
        suffixes=("_accepted", "_assembled"),
        validate="one_to_one",
        indicator=True,
    )
    if comparison["_merge"].ne("both").any():
        raise AssemblyError(f"{year} assembly loses an accepted adapted OD row")
    for column in [value for value in ADAPTED_COLUMNS if value != "trip_id"]:
        if comparison[f"{column}_accepted"].ne(
            comparison[f"{column}_assembled"]
        ).any():
            raise AssemblyError(f"{year} accepted adapted column changed: {column}")
    return int(len(accepted)), int(len(adapted) - len(accepted))


def _monthly_records(year_root: Path, year: int) -> list[dict[str, Any]]:
    records = []
    for month in range(1, 13):
        path = year_root / f"voyages_{year}{month:02d}.csv"
        rows = max(sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)
        records.append(
            {
                "month": month,
                "path": str(path.resolve()),
                "rows": rows,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return records


def assemble_od_input_root(
    output_root: Path,
    sources: Mapping[int, Path] = DEFAULT_SOURCES,
    *,
    vessel_spec_root: Path | None = None,
    accepted_adapted_root: Path = ACCEPTED_ADAPTED_OD_ROOT,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    accepted_products = (CODING_ROOT / "data/processed/products").resolve()
    if output_root == accepted_products or accepted_products in output_root.parents:
        raise AssemblyError("OD assembly must not write inside accepted products")
    if tuple(sorted(sources)) != EXPECTED_YEARS:
        raise AssemblyError(
            f"OD sources must contain exactly {list(EXPECTED_YEARS)}; got {sorted(sources)}"
        )
    vessel_spec_root = vessel_spec_root or (
        CODING_ROOT / "data/processed/products/vessel_universe"
    )
    manifest_path = output_root / "assembly_manifest.json"
    if manifest_path.exists():
        return verify_od_input_root(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    year_records: list[dict[str, Any]] = []
    bridge_frames: list[pd.DataFrame] = []
    all_ids: list[pd.Series] = []
    for year in EXPECTED_YEARS:
        source = Path(sources[year]).resolve()
        frame, starts, ends = _read_source(source, year)
        adapted = _adapt_source(frame, starts, ends, year, vessel_spec_root)
        accepted_rows, rows_added = _assert_accepted_rows_unchanged(
            adapted, year, accepted_adapted_root
        )
        bridge = starts.dt.year.eq(year) & ends.dt.year.eq(year + 1)
        bridge_count = int(bridge.sum())
        if year in BUFFERED_YEARS and bridge_count < 1_000:
            raise AssemblyError(f"{year} buffered source has only {bridge_count} bridges")
        if year == 2025 and bridge_count:
            raise AssemblyError("2025 calendar-truncated source unexpectedly crosses into 2026")

        year_root = output_root / str(year)
        source_link = year_root / f"harmonized_od_voyages_{year}.feather"
        create_source_link(source_link, source)
        for month in range(1, 13):
            selected = adapted.loc[starts.dt.month.eq(month)]
            write_csv_atomic(year_root / f"voyages_{year}{month:02d}.csv", selected)
        monthly = _monthly_records(year_root, year)
        if sum(item["rows"] for item in monthly) != len(frame):
            raise AssemblyError(f"{year} monthly adapted rows do not balance")

        bridge_frame = pd.DataFrame(
            {
                "analysis_year": year,
                "trip_id": frame.loc[bridge, "internal_voyage_id"].astype(str),
                "vessel_uid": frame.loc[bridge, "vessel_uid"].astype(str),
                "mmsi": adapted.loc[bridge, "mmsi"].astype(str),
                "trip_start": starts.loc[bridge],
                "trip_end": ends.loc[bridge],
                "origin_visit_id": frame.loc[
                    bridge, "origin_visit_id_internal"
                ].astype("string"),
                "dest_visit_id": frame.loc[
                    bridge, "dest_visit_id_internal"
                ].astype("string"),
                "origin_city": frame.loc[
                    bridge, "origin_analytical_city_id"
                ].astype("string"),
                "dest_city": frame.loc[
                    bridge, "dest_analytical_city_id"
                ].astype("string"),
            }
        )
        bridge_frames.append(bridge_frame)
        all_ids.append(frame["internal_voyage_id"].astype(str))
        year_records.append(
            {
                "year": year,
                "cohort_mode": "buffered" if year in BUFFERED_YEARS else "calendar",
                "calendar_truncated": year == 2025,
                "source_path": str(source),
                "source_size_bytes": int(source.stat().st_size),
                "source_sha256": sha256_file(source),
                "source_rows": int(len(frame)),
                "bridge_voyages": bridge_count,
                "accepted_rows_reproduced": accepted_rows,
                "rows_added_to_accepted_calendar_source": rows_added,
                "accepted_rows_all_adapter_columns_equal": True,
                "assembled_source_link": str(source_link),
                "monthly_files": monthly,
            }
        )

    combined_ids = pd.concat(all_ids, ignore_index=True)
    if combined_ids.duplicated().any():
        raise AssemblyError("internal_voyage_id duplicates across annual OD sources")
    bridges = pd.concat(bridge_frames, ignore_index=True)
    if bridges["trip_id"].duplicated().any():
        raise AssemblyError("bridge trip_id duplicates across years")
    bridge_registry = output_root / "bridge_voyages.feather"
    bridges.to_feather(bridge_registry, compression="zstd", compression_level=6)
    payload = {
        "status": "pass",
        "manifest_version": "e030_stage30_20_od_assembly_v1",
        "expected_years": list(EXPECTED_YEARS),
        "accepted_products_modified": False,
        "years": year_records,
        "total_source_rows": int(sum(item["source_rows"] for item in year_records)),
        "total_bridge_voyages": int(len(bridges)),
        "unique_internal_voyage_ids": int(len(combined_ids)),
        "bridge_registry": {
            "path": str(bridge_registry.resolve()),
            "rows": int(len(bridges)),
            "size_bytes": int(bridge_registry.stat().st_size),
            "sha256": sha256_file(bridge_registry),
        },
    }
    write_json_atomic(manifest_path, payload)
    return verify_od_input_root(output_root)


def verify_od_input_root(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / "assembly_manifest.json"
    if not manifest_path.is_file():
        raise AssemblyError(f"missing OD assembly manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("years", [])
    years = [int(item["year"]) for item in records]
    if len(years) != len(set(years)):
        raise AssemblyError("OD assembly manifest contains a duplicate year")
    if tuple(sorted(years)) != EXPECTED_YEARS:
        raise AssemblyError(
            f"OD assembly manifest years differ: {years} != {list(EXPECTED_YEARS)}"
        )
    all_ids = []
    bridge_total = 0
    source_total = 0
    for item in sorted(records, key=lambda value: int(value["year"])):
        year = int(item["year"])
        source = Path(item["source_path"])
        if not source.is_file():
            raise AssemblyError(f"manifest source is missing for {year}: {source}")
        if int(source.stat().st_size) != int(item["source_size_bytes"]):
            raise AssemblyError(f"manifest source size changed for {year}")
        if sha256_file(source) != item["source_sha256"]:
            raise AssemblyError(f"manifest source SHA-256 changed for {year}")
        frame, starts, ends = _read_source(source, year)
        if len(frame) != int(item["source_rows"]):
            raise AssemblyError(f"manifest source row count changed for {year}")
        bridge_count = int((starts.dt.year.eq(year) & ends.dt.year.eq(year + 1)).sum())
        if bridge_count != int(item["bridge_voyages"]):
            raise AssemblyError(f"manifest bridge count changed for {year}")
        expected_mode = "buffered" if year in BUFFERED_YEARS else "calendar"
        if item.get("cohort_mode") != expected_mode:
            raise AssemblyError(f"manifest cohort mode is wrong for {year}")
        if bool(item.get("calendar_truncated")) != (year == 2025):
            raise AssemblyError(f"manifest calendar boundary flag is wrong for {year}")
        source_link = output_root / str(year) / f"harmonized_od_voyages_{year}.feather"
        if not source_link.is_symlink() or source_link.resolve() != source.resolve():
            raise AssemblyError(f"assembled source link is wrong for {year}")
        monthly = item.get("monthly_files", [])
        if [int(value["month"]) for value in monthly] != list(range(1, 13)):
            raise AssemblyError(f"monthly assembly is incomplete or duplicated for {year}")
        monthly_rows = 0
        for value in monthly:
            path = Path(value["path"])
            if not path.is_file():
                raise AssemblyError(f"assembled monthly file is missing: {path}")
            if int(path.stat().st_size) != int(value["size_bytes"]):
                raise AssemblyError(f"assembled monthly size changed: {path}")
            if sha256_file(path) != value["sha256"]:
                raise AssemblyError(f"assembled monthly SHA-256 changed: {path}")
            rows = max(sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)
            if rows != int(value["rows"]):
                raise AssemblyError(f"assembled monthly row count changed: {path}")
            monthly_rows += rows
        if monthly_rows != len(frame):
            raise AssemblyError(f"assembled monthly rows do not balance for {year}")
        all_ids.append(frame["internal_voyage_id"].astype(str))
        source_total += len(frame)
        bridge_total += bridge_count

    combined_ids = pd.concat(all_ids, ignore_index=True)
    if combined_ids.duplicated().any():
        raise AssemblyError("internal_voyage_id duplicates across assembly years")
    bridge_record = manifest.get("bridge_registry", {})
    bridge_path = Path(bridge_record.get("path", ""))
    if not bridge_path.is_file():
        raise AssemblyError("bridge registry is missing")
    if sha256_file(bridge_path) != bridge_record.get("sha256"):
        raise AssemblyError("bridge registry SHA-256 changed")
    bridge_rows = len(pd.read_feather(bridge_path, columns=["trip_id"]))
    if bridge_rows != int(bridge_record.get("rows", -1)) or bridge_rows != bridge_total:
        raise AssemblyError("bridge registry row count does not balance")
    if source_total != int(manifest.get("total_source_rows", -1)):
        raise AssemblyError("five-year source row total changed")
    if bridge_total != int(manifest.get("total_bridge_voyages", -1)):
        raise AssemblyError("five-year bridge total changed")
    if len(combined_ids) != int(manifest.get("unique_internal_voyage_ids", -1)):
        raise AssemblyError("five-year unique voyage count changed")
    return manifest
