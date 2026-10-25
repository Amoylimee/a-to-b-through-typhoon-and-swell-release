from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "20_quantify_uncertainty/20_interpolation_holdout/20_run_holdout.py"
)
SPEC = importlib.util.spec_from_file_location("e002_holdout_runner", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_short_production_interpolation_cannot_see_heldout_observations() -> None:
    timestamps = pd.date_range("2025-01-01 00:00:00Z", periods=19, freq="10min")
    observed = pd.DataFrame({
        "candidate_id": "candidate-1",
        "mmsi": "123456789",
        "vessel_uid_clean": "ship-1",
        "utc": timestamps,
        "lat": 10 + np.arange(len(timestamps)) / 100,
        "lon": 20 + np.arange(len(timestamps)) / 100,
        "observation_time_utc": timestamps,
        "track_evidence": "observed_ais",
    })
    sample = pd.Series({
        "sample_id": "H000001",
        "candidate_id": "candidate-1",
        "sample_set": "gap_length_x_shiptype",
        "shiptype_group": "container",
        "nominal_gap_hours": 1,
        "actual_gap_hours": 1.0,
        "gap_start_utc": "2025-01-01T01:00:00+00:00",
        "gap_end_utc": "2025-01-01T02:00:00+00:00",
        "source_shard": "shard_0000",
    })
    candidate = pd.Series({
        "candidate_id": "candidate-1",
        "mmsi": "123456789",
        "vessel_uid_clean": "ship-1",
        "layer": "L1",
    })

    paired, report = RUNNER._prediction_for_sample(sample, candidate, observed, None)

    assert report["heldout_truth_points"] == 5
    assert report["paired_prediction_points"] == 5
    assert report["heldout_points_leaked"] == 0
    np.testing.assert_allclose(paired.lat_truth, paired.lat_pred, atol=1e-12)
    np.testing.assert_allclose(paired.lon_truth, paired.lon_pred, atol=1e-12)


def test_long_route_reconstruction_receives_only_window_endpoints(monkeypatch) -> None:
    timestamps = pd.date_range("2025-01-01 00:00:00Z", periods=73, freq="10min")
    observed = pd.DataFrame({
        "candidate_id": "candidate-2",
        "mmsi": "987654321",
        "vessel_uid_clean": "ship-2",
        "utc": timestamps,
        "lat": 5 + np.arange(len(timestamps)) / 100,
        "lon": 30 + np.arange(len(timestamps)) / 100,
        "observation_time_utc": timestamps,
        "track_evidence": "observed_ais",
    })
    sample = pd.Series({
        "sample_id": "H000002",
        "candidate_id": "candidate-2",
        "sample_set": "gap_length_x_shiptype",
        "shiptype_group": "container",
        "nominal_gap_hours": 6,
        "actual_gap_hours": 6.0,
        "gap_start_utc": "2025-01-01T01:00:00+00:00",
        "gap_end_utc": "2025-01-01T07:00:00+00:00",
        "source_shard": "shard_0000",
    })
    candidate = pd.Series({
        "candidate_id": "candidate-2",
        "mmsi": "987654321",
        "vessel_uid_clean": "ship-2",
        "layer": "L1",
    })
    captured = {}

    def fake_reconstruct_gap(router, gap, anchors):
        captured["gap"] = gap
        captured["anchors"] = anchors
        track = pd.DataFrame({
            "utc": pd.date_range(gap.gap_start, gap.gap_end, freq="10min"),
            "lat": 5.0,
            "lon": 30.0,
            "track_evidence": "ais_interpolated_le4h",
        })
        return track, {"gap_outcome": "pilot"}, pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(RUNNER, "reconstruct_gap", fake_reconstruct_gap)
    paired, report = RUNNER._prediction_for_sample(sample, candidate, observed, object())

    gap = captured["gap"]
    assert gap.gap_start == pd.Timestamp(sample.gap_start_utc)
    assert gap.gap_end == pd.Timestamp(sample.gap_end_utc)
    assert not hasattr(gap, "hidden_points")
    assert captured["anchors"].empty
    assert report["heldout_points_leaked"] == 0
    assert report["heldout_truth_points"] == 35
    assert paired.lat_pred.notna().sum() == 35
