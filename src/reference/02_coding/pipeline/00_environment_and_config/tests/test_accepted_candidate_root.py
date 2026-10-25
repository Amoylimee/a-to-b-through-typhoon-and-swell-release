from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from accepted_candidate_root import read_candidate_root, sha256_file


COLUMNS = ("candidate_id", "mmsi", "vessel_uid_clean", "departure_time")


def _root(tmp_path: Path, *, row_count=3, missing_row_count=False) -> Path:
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    frames = [
        pd.DataFrame(
            {
                "candidate_id": ["c1", "c2"],
                "mmsi": [123, 456],
                "vessel_uid_clean": ["s1", "s2"],
                "departure_time": pd.to_datetime(
                    ["2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"], utc=True
                ),
            }
        ),
        pd.DataFrame(
            {
                "candidate_id": ["c3"],
                "mmsi": [789],
                "vessel_uid_clean": ["s3"],
                "departure_time": pd.to_datetime(["2025-03-01T00:00:00Z"], utc=True),
            }
        ),
    ]
    records = []
    for index, frame in enumerate(frames):
        path = parts_dir / f"part-{index:03d}.feather"
        feather.write_feather(frame, path)
        records.append(
            {
                "path": str(path),
                "rows": len(frame),
                "columns": list(frame.columns),
                "sha256": sha256_file(path),
            }
        )
    year = {
        "part_count": len(records),
        "parts": records,
        "candidate_enumeration_mode": "adjacent_commercial_episodes",
    }
    if not missing_row_count:
        year["row_count"] = row_count
    manifest = {
        "status": "pass",
        "years": {"2025": year},
    }
    path = tmp_path / "assembly_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_reads_verified_candidate_rows_and_derives_utc_departure_year(tmp_path):
    manifest = _root(tmp_path)

    result = read_candidate_root(
        manifest,
        COLUMNS,
        expected_manifest_sha256=sha256_file(manifest),
        expected_rows=3,
        expected_parts=2,
    )

    assert result["candidate_id"].tolist() == ["c1", "c2", "c3"]
    assert result["departure_year"].tolist() == [2025, 2025, 2025]
    assert result["candidate_id"].is_unique


@pytest.mark.parametrize("invalid_count", [None, 0, True, "3"])
def test_manifest_requires_present_positive_integer_row_count(tmp_path, invalid_count):
    manifest = _root(tmp_path, row_count=invalid_count)

    with pytest.raises(ValueError, match="passing 2025 assembly"):
        read_candidate_root(manifest, COLUMNS)


def test_manifest_requires_row_count_key(tmp_path):
    manifest = _root(tmp_path, missing_row_count=True)

    with pytest.raises(ValueError, match="passing 2025 assembly"):
        read_candidate_root(manifest, COLUMNS)


def test_manifest_part_rows_must_sum_to_year_row_count(tmp_path):
    manifest = _root(tmp_path, row_count=4)

    with pytest.raises(ValueError, match="do not sum"):
        read_candidate_root(manifest, COLUMNS)
