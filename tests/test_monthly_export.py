"""Exercise the public CLI with synthetic monthly aggregates and actual kernels."""
from pathlib import Path
import csv
import importlib.util
import json
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_export", ROOT / "src/build_public_table.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
m, projection = exporter.load_methods()


def monthly_frame(*, missing_wave=False):
    # A 10 h voyage: 2 h observed at 10 m, 8 h missing with wave bounds at 0 m.
    # The matched mean (10 m) lies outside the all-time range (2 m).
    matched, integral = (0, 0) if missing_wave else (2, 20)
    panel = pd.DataFrame([{
        "candidate_id": "synthetic", m.SHIP: "synthetic-vessel",
        "origin_city": "CHN:SyntheticA", "dest_city": "JPN:SyntheticB",
        "origin_country": "CHN", "dest_country": "JPN", "shiptype_group": "container",
        m.MONTH: "2025-07", "voyage_hours": 10, "track_hours": matched,
        "unresolved_track_hours": 10 - matched, "wave_matched_hours": matched,
        "wave_unmatched_hours": 10 - matched, "wave_integral_m_hours": integral,
        "wave_ge4m_matched_hours": matched, "tc300_positive_snapshot_hours": 0,
        "tc500_positive_snapshot_hours": 0, "main_wave_counted_hours": 10 - matched,
        "main_wave_min_integral": 0, "main_wave_max_integral": 0,
        "main_unbounded_hours": 0, "main_tc300_possible_hours": 0,
        "main_tc500_possible_hours": 0,
    }])
    return m.aggregate(panel, m.COUNTRY_KEY)


@pytest.mark.parametrize("level,fields", [("country", 41), ("city", 43)])
@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_cli_current_schema_keeps_single_voyage_and_unclipped_mean(tmp_path, level, fields, extension):
    frame = monthly_frame().assign(origin_city_code="CHN_c0001", dest_city_code="JPN_c0001",
                                   origin_city="CHN:SyntheticA", secret_source_id="not-public")
    source, target = tmp_path / ("input." + extension), tmp_path / "public.csv"
    if extension == "csv":
        frame.to_csv(source, index=False)
    else:
        frame.to_parquet(source, index=False)
    run = subprocess.run([sys.executable, str(ROOT / "src/build_public_table.py"),
                          "--input", str(source), "--output", str(target), "--level", level],
                         check=True, text=True, capture_output=True)
    report = json.loads(run.stdout)
    with target.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert len(reader.fieldnames) == report["columns"] == fields
    assert len(rows) == report["rows"] == report["records_with_1_voyage"] == 1
    assert not (set(projection.REMOVED) & set(reader.fieldnames))
    assert "origin_city" not in reader.fieldnames and "secret_source_id" not in reader.fieldnames
    row = rows[0]
    assert row["voyages"] == "1" and row["departure_month"] == "2025-07"
    assert row["mean_swh_matched_m"] == "10.000"
    assert row["mean_swh_constrained_m"] == row["mean_swh_max_possible_m"] == "2.000"
    assert row["share_wave_hours_ge4m"] == "1.000"
    assert row["single_ship"] == "True"


def test_missing_wave_stays_blank_while_range_survives(tmp_path):
    source, target = tmp_path / "input.parquet", tmp_path / "public.csv"
    monthly_frame(missing_wave=True).to_parquet(source, index=False)
    exporter.export_monthly(source, target, "country")
    with target.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["mean_swh_matched_m"] == row["share_wave_hours_ge4m"] == ""
    assert row["mean_swh_min_possible_m"] == "0.000"
    assert row["wave_mean_missing_reason"] == "no_matched_wave_hours_reachable_bounds_available"


@pytest.mark.parametrize("defect", ["duplicate", "reversed_bounds", "city_name"])
def test_invalid_input_does_not_leave_output(tmp_path, defect):
    frame = monthly_frame()
    level = "country"
    if defect == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    elif defect == "reversed_bounds":
        frame["mean_swh_min_possible_m"] = 5
    else:
        frame = frame.assign(origin_city_code="CHN:SyntheticA", dest_city_code="JPN_c0001")
        level = "city"
    source, target = tmp_path / "input.parquet", tmp_path / "public.csv"
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError):
        exporter.export_monthly(source, target, level)
    assert not target.exists()


def test_existing_output_and_input_are_preserved(tmp_path):
    source, target = tmp_path / "input.parquet", tmp_path / "public.csv"
    monthly_frame().to_parquet(source, index=False)
    original = source.read_bytes()
    target.write_text("existing user data")
    with pytest.raises(FileExistsError):
        exporter.export_monthly(source, target, "country")
    with pytest.raises(ValueError, match="separate"):
        exporter.export_monthly(source, source, "country")
    assert source.read_bytes() == original and target.read_text() == "existing user data"


@pytest.mark.parametrize("region", ["HKG", "TWN", "MAC"])
@pytest.mark.parametrize("side", ["origin", "dest"])
def test_unmerged_region_requires_voyage_regrouping_before_export(tmp_path, region, side):
    frame = monthly_frame()
    frame[side + "_country"] = region
    source, target = tmp_path / "input.parquet", tmp_path / "public.csv"
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match="Regroup"):
        exporter.export_monthly(source, target, "country")
    assert not target.exists()
