import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "10_build_pre_anchor_quality_review.py"
SPEC = importlib.util.spec_from_file_location("pre_anchor_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gap_boundaries_and_time_closure():
    hour = MODULE.NS_PER_HOUR
    times = np.array([0, 4 * hour, 9 * hour, 34 * hour], dtype=np.int64)
    lat = np.array([0.0, 1.0, 2.0, 3.0])
    lon = np.array([170.0, 175.0, -179.0, -170.0])
    metrics, gaps = MODULE.classify_window(
        "candidate", -hour, 36 * hour, times, lat, lon
    )
    assert metrics["ais_supported_le4h_hours"] == 4.0
    assert metrics["gfw_candidate_4_24h_hours"] == 5.0
    assert metrics["unresolved_internal_gt24h_hours"] == 25.0
    assert metrics["endpoint_or_no_ais_unknown_hours"] == 3.0
    assert [gap["gap_class"] for gap in gaps] == [
        "endpoint_or_no_ais_unknown",
        "endpoint_or_no_ais_unknown",
        "gfw_candidate_4_24h",
        "unresolved_internal_gt24h",
    ]


def test_no_ais_is_unknown_and_not_gfw_eligible():
    hour = MODULE.NS_PER_HOUR
    metrics, gaps = MODULE.classify_window(
        "empty",
        0,
        10 * hour,
        np.empty(0, dtype=np.int64),
        np.empty(0),
        np.empty(0),
    )
    assert metrics["endpoint_or_no_ais_unknown_hours"] == 10.0
    assert metrics["gfw_candidate_4_24h_hours"] == 0.0
    assert len(gaps) == 1
    assert gaps[0]["gfw_eligible"] is False
