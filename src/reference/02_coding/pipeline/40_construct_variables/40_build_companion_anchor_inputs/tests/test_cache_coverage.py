from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
from shapely.geometry import box


STAGE = Path(__file__).parents[1]
sys.path.insert(0, str(STAGE))
MODULE_PATH = STAGE / "20_rematch_existing_presence.py"
SPEC = importlib.util.spec_from_file_location("rematch_presence", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def target():
    return pd.DataFrame(
        [
            {
                "target_hour_id": "123|2025-01-01T02:00:00+00:00",
                "gap_id": "g1",
                "candidate_id": "c1",
                "mmsi": "123",
                "hour_utc": pd.Timestamp("2025-01-01T02:00:00Z"),
                "source_gap_start_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
                "source_gap_end_utc": pd.Timestamp("2025-01-01T05:00:00Z"),
                "lat_a": 0.0,
                "lon_a": 0.0,
                "lat_b": 0.1,
                "lon_b": 0.1,
                "effective_speed_kn": 25.0,
                "bounded": True,
            }
        ]
    )


def test_query_geometry_must_cover_whole_motion_envelope():
    frame = target()
    full = MODULE.classify_cache_coverage(frame, box(-5, -5, 5, 5))
    partial = MODULE.classify_cache_coverage(frame, box(0, 0, 0.2, 0.2))
    none = MODULE.classify_cache_coverage(frame, box(20, 20, 21, 21))
    assert full.loc[0, "cache_coverage"] == "full"
    assert partial.loc[0, "cache_coverage"] == "partial"
    assert none.loc[0, "cache_coverage"] == "none"
