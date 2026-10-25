import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "20_quantify_uncertainty"
    / "20_interpolation_holdout"
    / "60_summarize_cyclone_agreement.py"
)
SPEC = importlib.util.spec_from_file_location("e002_cyclone_agreement", SCRIPT)
AGREEMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGREEMENT)


def test_near_positions_and_identical_window_shares():
    windows = pd.DataFrame({"sample_id": ["a", "b"], "nominal_gap_hours": [4, 4]})
    points = pd.DataFrame({
        "sample_id": ["a", "a", "b", "b"],
        # window a: one position flips at 300 km; window b: far from any cyclone
        "tc_within_300km_truth": [True, False, False, False],
        "tc_within_300km_prediction": [False, False, False, False],
        "tc_within_500km_truth": [True, True, False, False],
        "tc_within_500km_prediction": [True, True, False, False],
    })
    row = AGREEMENT.summarize(points, windows).iloc[0]
    assert row.windows_complete == 2
    assert row.windows_identical_share_300km_pct == 50.0
    assert row.windows_identical_share_500km_pct == 100.0
    assert row.near_positions_300km == 1
    assert row.near_positions_same_class_300km_pct == 0.0
    assert row.near_positions_500km == 2
    assert row.near_positions_same_class_500km_pct == 100.0
    assert row.all_positions_class_changed_300km_pct == 25.0
