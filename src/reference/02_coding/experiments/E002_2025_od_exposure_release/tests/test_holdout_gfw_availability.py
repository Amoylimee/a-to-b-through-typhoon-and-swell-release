import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "20_quantify_uncertainty"
    / "20_interpolation_holdout"
    / "30_audit_gfw_availability.py"
)
SPEC = importlib.util.spec_from_file_location("e002_gfw_availability", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_only_direct_gfw_anchor_inside_open_holdout_window_counts_as_available():
    sample = pd.Series(
        {
            "sample_id": "s1",
            "candidate_id": "c1",
            "sample_set": "main",
            "shiptype_group": "bulk_carrier",
            "nominal_gap_hours": 6,
            "gap_start_utc": "2025-01-01T00:00:00Z",
            "gap_end_utc": "2025-01-01T01:00:00Z",
        }
    )
    gfw_points = pd.DataFrame(
        {
            "utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",  # endpoint: excluded
                    "2025-01-01T00:20:00Z",  # observed AIS: not GFW availability
                    "2025-01-01T00:30:00Z",  # direct GFW anchor
                    "2025-01-01T00:40:00Z",  # supported track row, not direct anchor
                    "2025-01-01T01:00:00Z",  # endpoint: excluded
                ],
                utc=True,
            ),
            "track_evidence": [
                "gfw_anchor",
                "observed_ais",
                "gfw_anchor",
                "gfw_supported_interpolation",
                "gfw_anchor",
            ],
        }
    )

    result = AUDIT.summarize_window_gfw_records(sample, gfw_points)

    assert result["gfw_anchor_rows_in_window"] == 1
    assert result["gfw_anchor_hours_in_window"] == 1
    assert result["gfw_supported_rows_in_window"] == 1
    assert result["gfw_anchor_available"] is True


def test_gp2_hour_window_must_overlap_holdout_and_be_admissible():
    sample = pd.Series(
        {
            "gap_start_utc": "2025-01-01T00:10:00Z",
            "gap_end_utc": "2025-01-01T00:50:00Z",
        }
    )
    targets = pd.DataFrame(
        {
            "hour_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:30:00Z",
                    "2025-01-01T01:00:00Z",
                ],
                utc=True,
            ),
            "admissible_unique_anchor": [True, False, True],
        }
    )

    result = AUDIT.count_gp2_targets_in_window(sample, targets)

    assert result == {
        "gp2_target_hour_windows": 2,
        "gp2_admissible_anchor_hours": 1,
    }
