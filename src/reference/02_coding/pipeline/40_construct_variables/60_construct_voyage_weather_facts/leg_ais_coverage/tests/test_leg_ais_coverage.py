"""Guard the coverage arithmetic behind the E031 AIS-coverage diagnostic.

The whole diagnostic rests on one function deciding which hours a weather match
can see.  Its edge cases are exactly where a silent error would look plausible:
a gap sitting on the four-hour limit, a window with no fix at all, and the two
endpoint gaps that a naive adjacent-difference implementation forgets.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "10_measure_leg_ais_coverage.py"

_spec = importlib.util.spec_from_file_location("e031_coverage", MODULE_PATH)
coverage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coverage)

NS_PER_HOUR = 3_600 * 1_000_000_000


def hours(value: float) -> np.int64:
    return np.int64(round(value * NS_PER_HOUR))


def evaluate(point_hours, start_hour, end_hour):
    utc = np.array([hours(h) for h in point_hours], dtype=np.int64)
    dark, counts, longest, endpoint = coverage.coverage_for_vessel(
        utc, np.array([hours(start_hour)]), np.array([hours(end_hour)])
    )
    return dark[0], int(counts[0]), longest[0], endpoint[0]


def test_a_dense_track_is_fully_measurable():
    dark, counts, longest, _ = evaluate([0, 1, 2, 3, 4, 5], 0, 5)
    assert dark == 0.0
    assert counts == 6
    assert longest == 1.0


def test_an_interior_hole_costs_exactly_its_own_hours():
    dark, _, longest, endpoint = evaluate([0, 1, 11, 12], 0, 12)
    assert dark == pytest.approx(10.0)
    assert longest == pytest.approx(10.0)
    assert endpoint == 0.0, "an interior hole is not an endpoint hole"


def test_a_gap_exactly_on_the_limit_still_interpolates():
    """The contract breaks a segment when the gap is *strictly* greater than 4 h."""

    dark, _, longest, _ = evaluate([0, 4, 5], 0, 5)
    assert dark == 0.0
    assert longest == pytest.approx(4.0)


def test_a_gap_just_past_the_limit_is_dark():
    dark, _, _, _ = evaluate([0, 4.001, 5], 0, 5)
    assert dark == pytest.approx(4.001)


def test_the_gap_before_the_first_fix_counts():
    dark, _, _, endpoint = evaluate([9, 10, 11], 0, 11)
    assert dark == pytest.approx(9.0)
    assert endpoint == pytest.approx(9.0)


def test_the_gap_after_the_last_fix_counts():
    dark, _, _, endpoint = evaluate([0, 1, 2], 0, 9)
    assert dark == pytest.approx(7.0)
    assert endpoint == pytest.approx(7.0)


def test_both_endpoint_gaps_count_together():
    dark, _, _, endpoint = evaluate([9, 10], 0, 20)
    assert dark == pytest.approx(19.0)
    assert endpoint == pytest.approx(19.0)


def test_a_window_with_no_fix_is_entirely_dark():
    dark, counts, longest, _ = evaluate([100], 0, 20)
    assert counts == 0
    assert dark == pytest.approx(20.0)
    assert longest == pytest.approx(20.0)


def test_a_short_window_with_no_fix_is_still_interpolable():
    """A three-hour window spans less than the limit, so it is not dark."""

    dark, counts, _, _ = evaluate([100], 0, 3)
    assert counts == 0
    assert dark == 0.0


def test_a_fix_on_the_boundary_counts_as_observed():
    _, counts, _, _ = evaluate([0, 5], 0, 5)
    assert counts == 2, "the window is closed on both ends"


def test_windows_are_evaluated_independently():
    utc = np.array([hours(h) for h in [0, 1, 2, 30, 31]], dtype=np.int64)
    starts = np.array([hours(0), hours(30)])
    ends = np.array([hours(2), hours(31)])
    dark, counts, _, _ = coverage.coverage_for_vessel(utc, starts, ends)
    assert list(dark) == [0.0, 0.0]
    assert list(counts) == [3, 2]


def test_the_gap_limit_matches_the_accepted_contract():
    """A drifted constant would silently change every number this run reports."""

    assert coverage.MAXIMUM_INTERPOLATION_GAP_HOURS == 4.0
    assert coverage.GAP_LIMIT_NS == np.int64(4 * NS_PER_HOUR)


# ---------------------------------------------------------------------------
# identity join ordering
# ---------------------------------------------------------------------------


def _bucket(tmp_path, rows):
    """Write one vessel-bucket fragment in the shuffled layout."""

    import pyarrow as pa
    import pyarrow.feather as feather

    bucket = tmp_path / "bucket_0000"
    bucket.mkdir()
    frame = pd.DataFrame(rows)
    feather.write_feather(
        pa.Table.from_pandas(frame, preserve_index=False),
        bucket / "2024_shard_0000.feather",
    )
    return bucket


def _point(mmsi, api, when, lat, lon, ordinal):
    return {
        "vessel_uid": api,
        "mmsi": mmsi,
        "UTC": pd.Timestamp(when, tz="UTC"),
        "lat": lat,
        "lon": lon,
        "source_year": np.int16(2024),
        "source_shard": "shard_0000",
        "source_row_ordinal": np.int64(ordinal),
        "vessel_bucket": np.int16(0),
    }


def test_a_shared_mmsi_is_excluded_and_counted_not_fatal(tmp_path):
    """The identity join must run before deduplication.

    Two vessels share MMSI 111111111 and both report at 03:00.  Deduplication
    treats a differing ``vessel_uid`` as a disagreement in its own right, so
    running it first raises ``same-vessel duplicate UTC AIS observations
    disagree`` and the whole bucket dies.  The accepted stage orders these the
    other way round: a relevant MMSI carrying the wrong source identity is
    excluded and counted, never fatal and never relabelled, as required by
    CONTRACT section 5.

    The real vessel therefore keeps all three of its fixes.
    """

    rows = [
        _point("111111111", "777", "2024-01-01 00:00", 10.0, 20.0, 0),
        _point("111111111", "999", "2024-01-01 03:00", 10.5, 20.5, 1),
        _point("111111111", "777", "2024-01-01 03:00", 10.5, 20.5, 2),
        _point("111111111", "777", "2024-01-01 06:00", 11.0, 21.0, 3),
    ]
    bucket = _bucket(tmp_path, rows)

    candidates = pd.DataFrame(
        {
            "candidate_id": ["c0"],
            "mmsi": pd.array(["111111111"], dtype="string"),
            "vessel_uid_clean": pd.array(["777"], dtype="string"),
            "departure_time": [pd.Timestamp("2024-01-01 00:00", tz="UTC")],
            "arrival_time": [pd.Timestamp("2024-01-01 06:00", tz="UTC")],
            "departure_year": [2024],
            "observed_sea_hours": [6.0],
            "shiptype_group": ["tanker"],
            "origin_city": ["A"],
            "dest_city": ["B"],
        }
    )

    out = tmp_path / "per_candidate"
    out.mkdir()
    audit = coverage.evaluate_bucket(bucket, candidates, out)

    result = pd.read_parquet(out / "bucket_0000.parquet")
    assert len(result) == 1
    row = result.iloc[0]

    # The real vessel reported at 00:00, 03:00 and 06:00, so nothing is dark.
    assert row["ais_points_in_window"] == 3
    assert row["dark_hours_over_4h"] == pytest.approx(0.0)
    assert row["coverage_ratio"] == pytest.approx(1.0)
    assert audit["identity_join_mismatch_rows"] == 1

    # Pin the reason this ordering exists: the reverse order is fatal here.
    from ais_identity import deduplicate_vessel_ais

    with pytest.raises(ValueError, match="duplicate UTC AIS observations disagree"):
        deduplicate_vessel_ais(pd.DataFrame(rows))


def test_an_unrelated_mmsi_never_contributes_coverage(tmp_path):
    rows = [
        _point("222222222", "888", "2024-01-01 01:00", 5.0, 5.0, 0),
        _point("222222222", "888", "2024-01-01 05:00", 5.5, 5.5, 1),
    ]
    bucket = _bucket(tmp_path, rows)
    candidates = pd.DataFrame(
        {
            "candidate_id": ["c0"],
            "mmsi": pd.array(["111111111"], dtype="string"),
            "vessel_uid_clean": pd.array(["777"], dtype="string"),
            "departure_time": [pd.Timestamp("2024-01-01 00:00", tz="UTC")],
            "arrival_time": [pd.Timestamp("2024-01-01 12:00", tz="UTC")],
            "departure_year": [2024],
            "observed_sea_hours": [12.0],
            "shiptype_group": ["tanker"],
            "origin_city": ["A"],
            "dest_city": ["B"],
        }
    )
    out = tmp_path / "per_candidate"
    out.mkdir()
    audit = coverage.evaluate_bucket(bucket, candidates, out)

    row = pd.read_parquet(out / "bucket_0000.parquet").iloc[0]
    assert row["ais_points_in_window"] == 0
    assert row["dark_hours_over_4h"] == pytest.approx(12.0)
    assert audit["identity_join_empty"] == 1
