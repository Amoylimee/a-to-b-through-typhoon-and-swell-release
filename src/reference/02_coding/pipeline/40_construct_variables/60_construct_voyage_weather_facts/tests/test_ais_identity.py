from __future__ import annotations

import pandas as pd

from ais_identity import (
    VESSEL_BUCKET_HASH_VERSION,
    prepare_ais_batch,
    stable_vessel_bucket,
    validate_ais_physical_identity,
)


def _points(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    raw = pd.DataFrame(
        rows, columns=["vessel_uid", "mmsi", "UTC", "lat", "lon"]
    ).assign(UTC=lambda frame: pd.to_datetime(frame["UTC"], utc=True))
    prepared, audit = prepare_ais_batch(
        raw,
        source_year=2021,
        source_shard="synthetic.feather",
        source_row_offset=0,
        bucket_count=16,
    )
    assert audit["valid_rows"] == len(rows)
    return prepared


def _legs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("1", "123456789", "p0", "2021-01-01T00:00Z", "2021-01-01T01:00Z"),
            ("1", "123456789", "p1", "2021-01-01T01:00Z", "2021-01-01T02:00Z"),
        ],
        columns=[
            "vessel_uid_clean",
            "mmsi",
            "physical_leg_id",
            "departure_time",
            "arrival_time",
        ],
    ).assign(
        departure_time=lambda frame: pd.to_datetime(frame["departure_time"], utc=True),
        arrival_time=lambda frame: pd.to_datetime(frame["arrival_time"], utc=True),
    )


def test_stable_bucket_is_source_independent() -> None:
    first = stable_vessel_bucket("123456789", 16)
    second = stable_vessel_bucket("123456789", 16)
    assert first == second
    assert VESSEL_BUCKET_HASH_VERSION == "sha256-mmsi-utf8-first8-big-mod-v1"


def test_identity_join_drops_wrong_api_without_relabelling() -> None:
    points = _points(
        [
            ("9", "123456789", "2021-01-01T00:00Z", 1.0, 2.0),
            ("1", "123456789", "2021-01-01T00:30Z", 2.0, 3.0),
            ("8", "987654321", "2021-01-01T00:30Z", 4.0, 5.0),
        ]
    )
    matched, audit = validate_ais_physical_identity(points, _legs())
    assert matched["vessel_uid"].tolist() == ["1"]
    assert audit["api_identity_mismatch_rows"] == 1
    assert audit["irrelevant_vessel_rows_skipped"] == 1
