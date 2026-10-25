from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "20_quantify_uncertainty/20_interpolation_holdout/10_prepare_holdout_samples.py"
)
SPEC = importlib.util.spec_from_file_location("e002_holdout_prep", SCRIPT)
assert SPEC and SPEC.loader
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def test_dense_window_has_one_hour_context_on_both_sides() -> None:
    times = np.arange(43, dtype=np.int64) * 600_000_000_000
    left, right = PREP.eligible_windows(times, 4)
    assert len(left) > 0
    assert np.all(times[left] - times[0] >= 3_600_000_000_000)
    assert np.all(times[-1] - times[right] >= 3_600_000_000_000)
    assert np.all(np.diff(times)[np.minimum(left + 1, len(times) - 1)] <= 1_800_000_000_000)
    assert np.all(np.abs((times[right] - times[left]) - 4 * 3_600_000_000_000) <= 600_000_000_000)


def test_any_observation_gap_over_thirty_minutes_splits_dense_runs() -> None:
    times = np.r_[
        np.arange(0, 25, dtype=np.int64) * 600_000_000_000,
        (25 * 600 + 3600) * 1_000_000_000,
        (26 * 600 + 3600) * 1_000_000_000,
        (27 * 600 + 3600) * 1_000_000_000,
        (28 * 600 + 3600) * 1_000_000_000,
    ]
    left, right = PREP.eligible_windows(times, 1)
    assert len(left) > 0
    assert np.all(right < 25)


def test_twenty_four_hour_holdout_stays_inside_supported_route_band() -> None:
    times = np.arange(27 * 6 + 1, dtype=np.int64) * 600_000_000_000
    left, right = PREP.eligible_windows(times, 24)
    assert len(left) > 0
    gaps = times[right] - times[left]
    assert np.all(gaps >= 4 * 3_600_000_000_000)
    assert np.all(gaps <= 24 * 3_600_000_000_000)
