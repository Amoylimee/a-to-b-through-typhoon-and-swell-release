from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "cross_year_voyages.py"
SPEC = importlib.util.spec_from_file_location("cross_year_voyages", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DRIVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "20_build_cross_year_buffered_od_voyages.py"
)
DRIVER_SPEC = importlib.util.spec_from_file_location(
    "cross_year_voyage_driver", DRIVER_PATH
)
assert DRIVER_SPEC is not None and DRIVER_SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(DRIVER_SPEC)
DRIVER_SPEC.loader.exec_module(DRIVER)


def _visit(
    year: int,
    ship: str,
    sequence: int,
    entry: str,
    exit_: str | None,
    anchorage: str,
    confidence: str = "high",
) -> dict[str, object]:
    return {
        "year": year,
        "vessel_uid": ship,
        "registry_imo_clean": f"imo-{ship}",
        "visit_id_internal": f"{ship}_v{sequence:05d}",
        "anchorage_id": anchorage,
        "anchorage_iso3": "CHN" if anchorage == "A" else "NLD",
        "anchorage_label": f"label-{anchorage}",
        "anchorage_sublabel": f"sub-{anchorage}",
        "analytical_city_id": f"city-{anchorage}",
        "city_name": f"City {anchorage}",
        "entry_time": pd.Timestamp(entry),
        "exit_time": pd.Timestamp(exit_) if exit_ is not None else pd.NaT,
        "visit_confidence": confidence,
    }


def _baseline() -> pd.DataFrame:
    row = {
        "year": 2022,
        "vessel_uid": "ship-1",
        "registry_imo_clean": "imo-ship-1",
        "internal_voyage_id": "2022_ship-1_0001",
        "voyage_sequence": 1,
        "trip_start": pd.Timestamp("2022-12-29 06:00:00"),
        "trip_end": pd.Timestamp("2022-12-30 08:00:00"),
        "duration_hours": 26.0,
        "duration_days": 26.0 / 24.0,
        "origin_visit_id_internal": "ship-1_v00001",
        "dest_visit_id_internal": "ship-1_v00002",
        "origin_anchorage_id": "A",
        "dest_anchorage_id": "B",
        "origin_iso3": "CHN",
        "dest_iso3": "NLD",
        "origin_label": "label-A",
        "dest_label": "label-B",
        "origin_sublabel": "sub-A",
        "dest_sublabel": "sub-B",
        "origin_analytical_city_id": "city-A",
        "dest_analytical_city_id": "city-B",
        "origin_city_name": "City A",
        "dest_city_name": "City B",
        "od_pair": "A__B",
        "city_pair": "city-A__city-B",
        "country_pair": "CHN__NLD",
        "is_international": True,
        "shiptype_group": "container",
        "size_bin": "large",
        "source_pipeline_generation": "internal_harmonized",
    }
    return pd.DataFrame([row], columns=MODULE.VOYAGE_OUTPUT_COLUMNS)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current = pd.DataFrame(
        [
            _visit(2022, "ship-1", 1, "2022-12-28", "2022-12-29 06:00", "A"),
            _visit(2022, "ship-1", 2, "2022-12-30 08:00", "2022-12-31 18:00", "B"),
            # A NaT row would sort last in the raw visit table, but it is low
            # confidence and therefore cannot affect voyage numbering.
            _visit(2022, "ship-1", 3, "2022-12-31 20:00", None, "B", "low"),
        ]
    )
    following = pd.DataFrame(
        [
            _visit(2023, "ship-1", 1, "2023-01-08 06:00", "2023-01-08 12:00", "A"),
            _visit(2023, "ship-1", 2, "2023-01-09", "2023-01-09 06:00", "B"),
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "vessel_uid": "ship-1",
                "registry_imo_clean": "imo-ship-1",
                "shiptype_group": "container",
                "size_bin": "large",
            }
        ]
    )
    return current, following, metadata


def test_buffer_appends_one_bridge_and_preserves_old_row_exactly() -> None:
    current, following, metadata = _inputs()
    baseline = _baseline()
    output, bridges, diagnostics = MODULE.build_buffered_voyages(
        current, following, baseline, metadata, 2022
    )

    assert diagnostics["existing_voyage_identity"]["status"] == "pass"
    assert len(output) == 2
    pd.testing.assert_frame_equal(
        output.iloc[[0]].reset_index(drop=True), baseline, check_exact=True
    )
    bridge = bridges.iloc[0]
    assert bridge["internal_voyage_id"] == "2022_ship-1_0002"
    assert bridge["trip_start"] == pd.Timestamp("2022-12-31 18:00:00")
    assert bridge["trip_end"] == pd.Timestamp("2023-01-08 06:00:00")
    assert bridge["origin_visit_id_internal"] == "2022:ship-1_v00002"
    assert bridge["dest_visit_id_internal"] == "2023:ship-1_v00001"


def test_existing_timestamp_drift_stops_before_output_assembly() -> None:
    current, following, metadata = _inputs()
    baseline = _baseline()
    baseline.loc[0, "trip_end"] = pd.Timestamp("2022-12-30 09:00:00")
    with pytest.raises(ValueError, match="timestamp stability failed"):
        MODULE.build_buffered_voyages(
            current, following, baseline, metadata, 2022
        )


def test_existing_id_drift_stops_before_output_assembly() -> None:
    current, following, metadata = _inputs()
    baseline = _baseline()
    baseline.loc[0, "internal_voyage_id"] = "2022_ship-1_0999"
    with pytest.raises(ValueError, match="ID stability failed"):
        MODULE.build_buffered_voyages(
            current, following, baseline, metadata, 2022
        )


def test_pairing_retains_departures_in_requested_year_only() -> None:
    current, following, _ = _inputs()
    pairs = MODULE.enumerate_departure_year_pairs(current, following, 2022)
    assert pairs["exit_time"].dt.year.eq(2022).all()
    assert set(pairs["_dest__visit_source_year"].astype(int)) == {2022, 2023}
    assert len(pairs) == 2


def test_visit_namespace_is_idempotent_and_preserves_blanks() -> None:
    assert MODULE.namespace_internal_visit_id("abc", 2022) == "2022:abc"
    assert MODULE.namespace_internal_visit_id("2023:abc", 2022) == "2023:abc"
    assert MODULE.namespace_internal_visit_id("", 2022) == ""
    assert pd.isna(MODULE.namespace_internal_visit_id(pd.NA, 2022))


def test_driver_refuses_source_substitution_and_accepted_output(
    tmp_path: Path,
) -> None:
    source, output = DRIVER.validate_roots(
        DRIVER.ACCEPTED_PORT_CALL_ROOT, tmp_path / "isolated"
    )
    assert source == DRIVER.ACCEPTED_PORT_CALL_ROOT.resolve()
    assert output == (tmp_path / "isolated").resolve()
    with pytest.raises(ValueError, match="source substitution refused"):
        DRIVER.validate_roots(tmp_path / "not-accepted", tmp_path / "isolated")
    with pytest.raises(ValueError, match="accepted products"):
        DRIVER.validate_roots(
            DRIVER.ACCEPTED_PORT_CALL_ROOT,
            DRIVER.CODING_ROOT / "data/processed/products/probe",
        )


def test_driver_marks_2025_as_unbuffered_outer_boundary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no accepted 2026 visits"):
        DRIVER.build_one_year(
            2025,
            DRIVER.ACCEPTED_PORT_CALL_ROOT,
            tmp_path,
        )


def test_year_worker_cap_and_sequential_failure_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="between 1 and 12"):
        DRIVER.run_years(
            (2021,),
            DRIVER.ACCEPTED_PORT_CALL_ROOT,
            tmp_path,
            13,
        )

    def fake_build(year: int, source_root: Path, output_root: Path) -> dict[str, object]:
        if year == 2023:
            raise RuntimeError("synthetic annual failure")
        return {"departure_year": year, "status": "pass"}

    monkeypatch.setattr(DRIVER, "build_one_year", fake_build)
    reports, failures, workers = DRIVER.run_years(
        (2021, 2023),
        DRIVER.ACCEPTED_PORT_CALL_ROOT,
        tmp_path,
        1,
    )
    assert workers == 1
    assert reports == [{"departure_year": 2021, "status": "pass"}]
    assert failures == [
        {"year": 2023, "error": "RuntimeError('synthetic annual failure')"}
    ]
