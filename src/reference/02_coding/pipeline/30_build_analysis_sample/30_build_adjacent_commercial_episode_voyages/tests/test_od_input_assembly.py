"""Unit checks for the guarded five-year stage-30/20 OD seam."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import od_input_assembly as assembly


def source_frame(year: int) -> pd.DataFrame:
    records = [
        {
            "vessel_uid": "100",
            "internal_voyage_id": f"{year}_100_0001",
            "trip_start": pd.Timestamp(f"{year}-01-02 00:00:00"),
            "trip_end": pd.Timestamp(f"{year}-01-03 00:00:00"),
            "origin_anchorage_id": "aaaaaaaa",
            "dest_anchorage_id": "bbbbbbbb",
            "origin_visit_id_internal": "100_v00001",
            "dest_visit_id_internal": "100_v00002",
            "origin_analytical_city_id": "AAA:ONE",
            "dest_analytical_city_id": "BBB:TWO",
        }
    ]
    if year in assembly.BUFFERED_YEARS:
        for index in range(1_000):
            records.append(
                {
                    "vessel_uid": "100",
                    "internal_voyage_id": f"{year}_100_b{index:04d}",
                    "trip_start": pd.Timestamp(f"{year}-12-31 00:00:00")
                    + pd.Timedelta(seconds=index),
                    "trip_end": pd.Timestamp(f"{year + 1}-01-02 00:00:00")
                    + pd.Timedelta(seconds=index),
                    "origin_anchorage_id": "aaaaaaaa",
                    "dest_anchorage_id": "bbbbbbbb",
                    "origin_visit_id_internal": f"{year}:100_v{index:05d}",
                    "dest_visit_id_internal": f"{year + 1}:100_v{index + 1:05d}",
                    "origin_analytical_city_id": "AAA:ONE",
                    "dest_analytical_city_id": "BBB:TWO",
                }
            )
    return pd.DataFrame(records)


def build_fixture(tmp_path: Path) -> tuple[dict[int, Path], Path, Path]:
    sources = {}
    specs_root = tmp_path / "specs"
    accepted_root = tmp_path / "accepted"
    specs_root.mkdir(parents=True)
    for year in assembly.EXPECTED_YEARS:
        frame = source_frame(year)
        source = tmp_path / f"source_{year}.feather"
        frame.to_feather(source)
        sources[year] = source
        pd.DataFrame(
            {"vessel_uid_clean": ["100"], "mmsi_clean": ["123456789"]}
        ).to_csv(specs_root / f"analysis_vessel_specs_{year}.csv", index=False)
        old = frame.iloc[[0]].copy()
        starts = pd.to_datetime(old["trip_start"], utc=True)
        ends = pd.to_datetime(old["trip_end"], utc=True)
        adapted = assembly._adapt_source(old, starts, ends, year, specs_root)
        year_root = accepted_root / str(year)
        year_root.mkdir(parents=True)
        for month in range(1, 13):
            adapted.loc[starts.dt.month.eq(month)].to_csv(
                year_root / f"voyages_{year}{month:02d}.csv", index=False
            )
    return sources, specs_root, accepted_root


def assemble_fixture(tmp_path: Path) -> tuple[Path, dict[int, Path]]:
    sources, specs_root, accepted_root = build_fixture(tmp_path)
    root = tmp_path / "assembly"
    manifest = assembly.assemble_od_input_root(
        root,
        sources,
        vessel_spec_root=specs_root,
        accepted_adapted_root=accepted_root,
    )
    assert manifest["total_source_rows"] == 4_005
    assert manifest["total_bridge_voyages"] == 4_000
    return root, sources


def test_assembly_and_guard_pass_with_five_unique_years(tmp_path: Path) -> None:
    root, _ = assemble_fixture(tmp_path)
    verified = assembly.verify_od_input_root(root)
    assert verified["status"] == "pass"
    assert [item["year"] for item in verified["years"]] == list(
        assembly.EXPECTED_YEARS
    )
    assert verified["years"][-1]["calendar_truncated"] is True
    assert (root / "2021/harmonized_od_voyages_2021.feather").is_symlink()
    assert len(list((root / "2021").glob("voyages_2021??.csv"))) == 12


def test_guard_rejects_changed_source_hash(tmp_path: Path) -> None:
    root, sources = assemble_fixture(tmp_path)
    changed = pd.read_feather(sources[2023])
    changed.loc[0, "dest_anchorage_id"] = "cccccccc"
    changed.to_feather(sources[2023])
    with pytest.raises(assembly.AssemblyError, match="size changed|SHA-256 changed"):
        assembly.verify_od_input_root(root)


def test_guard_rejects_duplicate_or_missing_year(tmp_path: Path) -> None:
    root, _ = assemble_fixture(tmp_path)
    manifest_path = root / "assembly_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["years"][-1]["year"] = 2024
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(assembly.AssemblyError, match="duplicate year"):
        assembly.verify_od_input_root(root)

    sources, specs_root, accepted_root = build_fixture(tmp_path / "missing")
    sources.pop(2022)
    with pytest.raises(assembly.AssemblyError, match="exactly"):
        assembly.assemble_od_input_root(
            tmp_path / "missing_assembly",
            sources,
            vessel_spec_root=specs_root,
            accepted_adapted_root=accepted_root,
        )
