from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weather_core.tropical_cyclone import (
    CycloneTrackIndex,
    aggregate_candidate_tc_segments,
    load_ibtracs_snapshot,
)
from weather_core.geo import EARTH_RADIUS_KM


def _write_snapshot(path: Path, rows: list[dict[str, object]]) -> Path:
    columns = [
        "SID",
        "ISO_TIME",
        "NATURE",
        "LAT",
        "LON",
        "USA_WIND",
        "WMO_WIND",
        "TRACK_TYPE",
    ]
    with path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(columns) + "\n")
        stream.write(" , , ,degrees_north,degrees_east,kts,kts, \n")
        pd.DataFrame(rows, columns=columns).to_csv(stream, index=False, header=False)
    return path


def _fixes(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["fix_time_utc"] = pd.to_datetime(frame["fix_time_utc"], utc=True)
    defaults = {
        "nature": "TS",
        "track_type": "main",
        "center_valid": True,
        "wind_kt": np.nan,
        "wind_source": pd.NA,
        "is_provisional_nr": False,
        "retained_phase": 0,
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    if "_source_row" not in frame:
        frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
    if "_sid_source_order" not in frame:
        frame["_sid_source_order"] = frame.groupby("SID", sort=False).cumcount()
    return frame


def test_loader_filters_lifecycle_preserves_phase_and_wind_priority(tmp_path):
    snapshot = _write_snapshot(
        tmp_path / "ibtracs.csv",
        [
            {
                "SID": "A",
                "ISO_TIME": "2025-01-01 00:00:00",
                "NATURE": "TS",
                "LAT": 10,
                "LON": 179,
                "USA_WIND": 50,
                "WMO_WIND": 40,
                "TRACK_TYPE": "main",
            },
            {
                "SID": "A",
                "ISO_TIME": "2025-01-01 03:00:00",
                "NATURE": "DS",
                "LAT": 10,
                "LON": 180,
                "USA_WIND": 55,
                "WMO_WIND": 45,
                "TRACK_TYPE": "main",
            },
            {
                "SID": "A",
                "ISO_TIME": "2025-01-01 06:00:00",
                "NATURE": "TS",
                "LAT": 10,
                "LON": 181,
                "USA_WIND": "",
                "WMO_WIND": 60,
                "TRACK_TYPE": "main",
            },
            {
                "SID": "P",
                "ISO_TIME": "2025-02-01 00:00:00",
                "NATURE": "NR",
                "LAT": 5,
                "LON": 257.4,
                "USA_WIND": "",
                "WMO_WIND": 35,
                "TRACK_TYPE": "PROVISIONAL",
            },
            {
                "SID": "M",
                "ISO_TIME": "2025-02-01 00:00:00",
                "NATURE": "NR",
                "LAT": 5,
                "LON": 20,
                "USA_WIND": 30,
                "WMO_WIND": 20,
                "TRACK_TYPE": "main",
            },
            {
                "SID": "S",
                "ISO_TIME": "2025-03-01 00:00:00",
                "NATURE": "TS",
                "LAT": 5,
                "LON": 20,
                "USA_WIND": 30,
                "WMO_WIND": 20,
                "TRACK_TYPE": "spur-merge",
            },
        ],
    )
    fixes, audit = load_ibtracs_snapshot(snapshot, expected_sha256=None)

    assert fixes["SID"].tolist() == ["A", "A", "P"]
    assert fixes.loc[fixes["SID"].eq("A"), "retained_phase"].tolist() == [0, 1]
    assert fixes["wind_kt"].tolist() == [50.0, 60.0, 35.0]
    assert fixes["wind_source"].tolist() == ["USA_WIND", "WMO_WIND", "WMO_WIND"]
    assert fixes.loc[fixes["SID"].eq("P"), "is_provisional_nr"].item()
    assert fixes.loc[fixes["SID"].eq("P"), "center_lon"].item() == pytest.approx(-102.6)
    assert audit["provisional_nr_rows"] == 1
    assert audit["year_rows"]["2025"]["provisional_nr_rows"] == 1
    assert audit["year_rows"]["2025"]["track_type_counts"] == {
        "main": 2,
        "PROVISIONAL": 1,
        "US-PROVISIONAL": 0,
    }
    assert audit["window_track_type_counts"] == {
        "PROVISIONAL": 1,
        "main": 4,
        "spur-merge": 1,
    }


def test_loader_rejects_duplicate_sid_time(tmp_path):
    row = {
        "SID": "A",
        "ISO_TIME": "2025-01-01 00:00:00",
        "NATURE": "TS",
        "LAT": 10,
        "LON": 20,
        "USA_WIND": 50,
        "WMO_WIND": 40,
        "TRACK_TYPE": "main",
    }
    snapshot = _write_snapshot(tmp_path / "duplicate.csv", [row, row])
    with pytest.raises(ValueError, match=r"SID \+ ISO_TIME"):
        load_ibtracs_snapshot(snapshot, expected_sha256=None)


def test_loader_requires_explicit_hash_argument_and_rejects_mismatch(tmp_path):
    snapshot = _write_snapshot(
        tmp_path / "hash.csv",
        [
            {
                "SID": "A",
                "ISO_TIME": "2025-01-01 00:00:00",
                "NATURE": "TS",
                "LAT": 10,
                "LON": 20,
                "USA_WIND": 50,
                "WMO_WIND": 40,
                "TRACK_TYPE": "main",
            }
        ],
    )
    with pytest.raises(TypeError, match="expected_sha256"):
        load_ibtracs_snapshot(snapshot)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_ibtracs_snapshot(snapshot, expected_sha256="0" * 64)


def test_dateline_interpolation_and_exact_three_hour_boundary():
    index = CycloneTrackIndex(
        _fixes(
            [
                {
                    "SID": "A",
                    "fix_time_utc": "2025-01-01 00:00:00Z",
                    "center_lat": 10.0,
                    "center_lon": 179.0,
                    "wind_kt": 40.0,
                },
                {
                    "SID": "A",
                    "fix_time_utc": "2025-01-01 03:00:00Z",
                    "center_lat": 12.0,
                    "center_lon": -179.0,
                    "wind_kt": 70.0,
                },
            ]
        ),
        maximum_fix_gap_hours=3.0,
    )
    result = index.positions_at_times(
        ["2025-01-01 00:00:00Z", "2025-01-01 01:30:00Z", "2025-01-01 03:00:00Z"]
    )
    assert result["position_source"].tolist() == ["exact_fix", "interpolated", "exact_fix"]
    assert result["storm_lat"].tolist() == pytest.approx([10.0, 11.0, 12.0])
    assert result["storm_lon"].tolist() == pytest.approx([179.0, -180.0, -179.0])
    assert result["storm_wind_kt"].tolist() == pytest.approx([40.0, 55.0, 70.0])


def test_gap_above_three_hours_and_phase_break_are_not_bridged():
    index = CycloneTrackIndex(
        _fixes(
            [
                {
                    "SID": "GAP",
                    "fix_time_utc": "2025-01-01 00:00:00Z",
                    "center_lat": 0.0,
                    "center_lon": 0.0,
                },
                {
                    "SID": "GAP",
                    "fix_time_utc": "2025-01-01 03:00:01Z",
                    "center_lat": 1.0,
                    "center_lon": 1.0,
                },
                {
                    "SID": "PHASE",
                    "fix_time_utc": "2025-01-01 00:00:00Z",
                    "center_lat": 0.0,
                    "center_lon": 0.0,
                    "retained_phase": 0,
                },
                {
                    "SID": "PHASE",
                    "fix_time_utc": "2025-01-01 03:00:00Z",
                    "center_lat": 1.0,
                    "center_lon": 1.0,
                    "retained_phase": 1,
                },
            ]
        )
    )
    interior = index.positions_at_times(["2025-01-01 01:00:00Z"])
    assert interior.empty
    exact = index.positions_at_times(["2025-01-01 00:00:00Z"])
    assert set(exact["SID"]) == {"GAP", "PHASE"}


def test_wind_interpolation_requires_both_endpoints_but_exact_fix_survives():
    index = CycloneTrackIndex(
        _fixes(
            [
                {
                    "SID": "A",
                    "fix_time_utc": "2025-01-01 00:00:00Z",
                    "center_lat": 0.0,
                    "center_lon": 0.0,
                    "wind_kt": 50.0,
                },
                {
                    "SID": "A",
                    "fix_time_utc": "2025-01-01 03:00:00Z",
                    "center_lat": 0.0,
                    "center_lon": 1.0,
                    "wind_kt": np.nan,
                },
            ]
        )
    )
    result = index.positions_at_times(
        ["2025-01-01 00:00:00Z", "2025-01-01 01:00:00Z", "2025-01-01 03:00:00Z"]
    )
    assert result.loc[0, "storm_wind_kt"] == 50.0
    assert np.isnan(result.loc[1, "storm_wind_kt"])
    assert np.isnan(result.loc[2, "storm_wind_kt"])


def test_hourly_interpolation_is_half_open_and_requires_exact_hour_bounds():
    index = CycloneTrackIndex(
        _fixes(
            [
                {
                    "SID": "A",
                    "fix_time_utc": "2025-01-01 00:00:00Z",
                    "center_lat": 0.0,
                    "center_lon": 0.0,
                },
                {
                    "SID": "A",
                    "fix_time_utc": "2025-01-01 03:00:00Z",
                    "center_lat": 0.0,
                    "center_lon": 3.0,
                },
            ]
        )
    )
    hourly = index.interpolate_hourly_fixes(
        "2025-01-01 00:00:00Z", "2025-01-01 03:00:00Z"
    )
    assert hourly["target_time_utc"].dt.hour.tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="exact UTC hours"):
        index.interpolate_hourly_fixes(
            "2025-01-01 00:00:01Z", "2025-01-01 03:00:00Z"
        )


def test_position_match_multi_storm_counts_time_once_and_uses_min_and_max():
    fixes = _fixes(
        [
            {
                "SID": "A",
                "fix_time_utc": "2025-01-01 00:00:00Z",
                "center_lat": 0.0,
                "center_lon": 0.0,
                "wind_kt": np.nan,
            },
            {
                "SID": "B",
                "fix_time_utc": "2025-01-01 00:00:00Z",
                "center_lat": 0.0,
                "center_lon": 1.0,
                "wind_kt": 80.0,
            },
            {
                "SID": "C",
                "fix_time_utc": "2025-01-01 00:00:00Z",
                "center_lat": 0.0,
                "center_lon": 4.0,
                "wind_kt": 90.0,
            },
        ]
    )
    matched = CycloneTrackIndex(fixes).match_positions(
        pd.DataFrame(
            {
                "position_time_utc": ["2025-01-01 00:00:00Z"],
                "latitude": [0.0],
                "longitude": [0.0],
            },
            index=pd.Index([99], name="named_source_index"),
        )
    )
    row = matched.iloc[0]
    assert row["_input_index"] == 99
    assert row["tc_min_center_distance_km"] == pytest.approx(0.0, abs=1e-5)
    assert row["tc_within_300km"]
    assert row["tc_within_500km"]
    assert row["tc_max_storm_wind_kt_500km"] == 90.0
    assert row["tc_sids_500km"] == ("A", "B", "C")


def test_position_match_no_storm_missing_wind_and_invalid_position_semantics():
    index = CycloneTrackIndex(
        _fixes(
            [
                {
                    "SID": "A",
                    "fix_time_utc": "2025-01-01 00:00:00Z",
                    "center_lat": 0.0,
                    "center_lon": 0.0,
                    "wind_kt": np.nan,
                }
            ]
        )
    )
    matched = index.match_positions(
        pd.DataFrame(
            {
                "position_time_utc": [
                    "2025-01-01 00:00:00Z",
                    "2025-01-02 00:00:00Z",
                    "2025-01-01 00:00:00Z",
                ],
                "latitude": [0.0, 0.0, 0.0],
                "longitude": [0.0, 0.0, 181.0],
            }
        )
    )
    assert matched["tc_match_status"].tolist() == [
        "valid_storm",
        "valid_no_storm",
        "invalid_position",
    ]
    assert np.isnan(matched.loc[0, "tc_max_storm_wind_kt_500km"])
    assert matched.loc[0, "tc_sids_500km"] == ("A",)
    assert not matched.loc[1, "tc_storm_present"]
    assert np.isnan(matched.loc[1, "tc_min_center_distance_km"])
    assert not matched.loc[2, "position_valid"]


def test_radius_boundaries_are_inclusive_but_materially_above_is_outside():
    lon_300 = np.degrees(300.0 / EARTH_RADIUS_KM)
    lon_above_300 = np.degrees((300.0 + 1e-5) / EARTH_RADIUS_KM)
    lon_500 = np.degrees(500.0 / EARTH_RADIUS_KM)
    lon_above_500 = np.degrees((500.0 + 1e-5) / EARTH_RADIUS_KM)
    index = CycloneTrackIndex(
        _fixes(
            [
                {
                    "SID": sid,
                    "fix_time_utc": "2025-01-01 00:00:00Z",
                    "center_lat": 0.0,
                    "center_lon": longitude,
                }
                for sid, longitude in [
                    ("at300", lon_300),
                    ("over300", lon_above_300),
                    ("at500", lon_500),
                    ("over500", lon_above_500),
                ]
            ]
        )
    )
    matched = index.match_positions(
        pd.DataFrame(
            {
                "position_time_utc": ["2025-01-01 00:00:00Z"],
                "latitude": [0.0],
                "longitude": [0.0],
            }
        )
    ).iloc[0]
    assert matched["tc_within_300km"]
    assert matched["tc_within_500km"]
    assert matched["tc_sids_500km"] == ("at300", "at500", "over300")


def test_candidate_aggregation_multi_sid_weight_once_and_distinct():
    segments = pd.DataFrame(
        {
            "candidate_id": ["v", "v", "v"],
            "segment_weight_hours": [0.4, 0.6, 1.0],
            "position_valid": [True, True, True],
            "tc_min_center_distance_km": [100.0, 200.0, 900.0],
            "tc_within_500km": [True, True, False],
            "tc_within_300km": [True, True, False],
            "tc_max_storm_wind_kt_500km": [70.0, 90.0, np.nan],
            "tc_sids_500km": [("A", "B"), ("B", "C"), tuple()],
            "swh_m": [5.0, 3.0, np.nan],
        }
    )
    result = aggregate_candidate_tc_segments(segments).iloc[0]
    assert result["weather_track_hours"] == pytest.approx(2.0)
    assert result["tropical_cyclone_exposure_hours_500km"] == pytest.approx(1.0)
    assert result["tropical_cyclone_exposure_hours_300km"] == pytest.approx(1.0)
    assert result["tropical_cyclone_min_center_distance_km"] == pytest.approx(100.0)
    assert result["tropical_cyclone_max_storm_wind_kt_500km"] == pytest.approx(90.0)
    assert result["tropical_cyclone_count_500km"] == 3
    assert result["high_wave_tc_overlap_hours_ge4m_500km"] == pytest.approx(0.4)


def test_candidate_zero_and_na_semantics():
    segments = pd.DataFrame(
        {
            "candidate_id": ["zero_track", "no_storm", "missing_wave"],
            "segment_weight_hours": [1.0, 1.0, 1.0],
            "position_valid": [False, True, True],
            "tc_min_center_distance_km": [np.nan, np.nan, 100.0],
            "tc_within_500km": [False, False, True],
            "tc_within_300km": [False, False, True],
            "tc_max_storm_wind_kt_500km": [np.nan, np.nan, np.nan],
            "tc_sids_500km": [tuple(), tuple(), ("A",)],
            "swh_m": [np.nan, np.nan, np.nan],
        }
    )
    result = aggregate_candidate_tc_segments(segments).set_index("candidate_id")
    zero = result.loc["zero_track"]
    assert zero["weather_track_hours"] == 0.0
    assert pd.isna(zero["tropical_cyclone_count_500km"])
    assert pd.isna(zero["tropical_cyclone_exposure_hours_500km"])

    no_storm = result.loc["no_storm"]
    assert no_storm["tropical_cyclone_exposure_hours_500km"] == 0.0
    assert no_storm["tropical_cyclone_count_500km"] == 0
    assert pd.isna(no_storm["tropical_cyclone_min_center_distance_km"])
    assert pd.isna(no_storm["tropical_cyclone_max_storm_wind_kt_500km"])
    assert no_storm["high_wave_tc_overlap_hours_ge4m_500km"] == 0.0

    missing_wave = result.loc["missing_wave"]
    assert missing_wave["tropical_cyclone_exposure_hours_500km"] == 1.0
    assert missing_wave["tropical_cyclone_count_500km"] == 1
    assert pd.isna(missing_wave["tropical_cyclone_max_storm_wind_kt_500km"])
    assert pd.isna(missing_wave["high_wave_tc_overlap_hours_ge4m_500km"])


def test_empty_batches_have_stable_schema():
    index = CycloneTrackIndex(
        _fixes(
            [
                {
                    "SID": "A",
                    "fix_time_utc": "2025-01-01 00:00:00Z",
                    "center_lat": 0.0,
                    "center_lon": 0.0,
                }
            ]
        )
    )
    positions = index.positions_at_times([])
    matched = index.match_positions(
        pd.DataFrame(columns=["position_time_utc", "latitude", "longitude"])
    )
    assert positions.empty
    assert matched.empty
    assert "tc_match_status" in matched


def test_december_last_hours_keep_next_year_support_fix(tmp_path):
    path=_write_snapshot(tmp_path/'boundary.csv',[
        {'SID':'A','ISO_TIME':'2025-12-31 21:00:00','NATURE':'TS','LAT':0.,'LON':0.,'USA_WIND':40,'WMO_WIND':'','TRACK_TYPE':'main'},
        {'SID':'A','ISO_TIME':'2026-01-01 00:00:00','NATURE':'TS','LAT':0.,'LON':3.,'USA_WIND':40,'WMO_WIND':'','TRACK_TYPE':'main'},
    ])
    fixes,_=load_ibtracs_snapshot(path,expected_sha256=None,time_end_exclusive='2026-01-02T00:00:00Z')
    points=pd.DataFrame({'position_time_utc':pd.to_datetime(['2025-12-31 23:00'],utc=True),'latitude':[0.],'longitude':[2.]})
    matched=CycloneTrackIndex(fixes).match_positions(points)
    assert matched.tc_storm_present.iloc[0]
    assert matched.tc_min_center_distance_km.iloc[0]<1e-8
    truncated,_=load_ibtracs_snapshot(path,expected_sha256=None)
    assert not CycloneTrackIndex(truncated).match_positions(points).tc_storm_present.iloc[0]
