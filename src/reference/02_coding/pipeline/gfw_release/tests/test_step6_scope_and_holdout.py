"""Check authorization drift and that real held-out observations cannot leak."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import step6_core as core
holdout = core.module("e006_holdout_test", HERE / "230_run_gfw_holdouts.py")


@pytest.mark.parametrize("field,value", [("year", 2024), ("seed", 2),
    ("minimum_voyages", 6), ("bootstrap_replicates", 100),
    ("step7_authorized", True), ("holdout_max_observation_gap_seconds", 1800)])
def test_reject_changed_authorization(tmp_path, monkeypatch, field, value):
    decision = json.loads(core.DECISION.read_text())
    decision[field] = value
    p = tmp_path / "decision.json"; p.write_text(json.dumps(decision))
    monkeypatch.setattr(core, "DECISION", p)
    with pytest.raises(ValueError): core.binding()


def fixture():
    times = pd.date_range("2025-06-01T00:30:00Z", periods=5, freq="h")
    observed = pd.DataFrame({"UTC": times, "UTC_10min": times, "lat": [0., 0., 10., 0., 0.],
                             "lon": [150.]*5})
    sample = SimpleNamespace(gap_start_utc=times[1], gap_end_utc=times[3], truth_observation_count=1,
       sample_id="test", candidate_id="voyage", source_shard="shard_0000", sample_set="gap_length_x_shiptype",
       nominal_gap_hours=2, shiptype_group="container")
    candidate = SimpleNamespace(vessel_uid_clean="1", mmsi="2", origin_city="A", dest_city="B", shiptype_group="container")
    return observed, sample, candidate


def test_prediction_input_excludes_heldout_truth():
    observed, sample, candidate = fixture()
    def production(scoped, visible, route, shard):
        assert not visible.UTC.eq(observed.UTC.iloc[2]).any()
        assert not visible.lat.eq(10).any()
        return pd.DataFrame({"utc": [observed.UTC.iloc[2]], "lat": [0.], "lon": [150.], "track_evidence": ["gfw_supported_interpolation"]}), {}, []
    pairs, outcome = holdout.predict_window(sample, observed, candidate, SimpleNamespace(build_track=production), None)
    assert pairs.lat_truth.iloc[0] == 10 and pairs.lat_pred.iloc[0] == 0
    assert outcome["heldout_points_leaked"] == 0


def test_a_visible_alias_cannot_reintroduce_hidden_grid_bin():
    observed, sample, candidate = fixture()
    observed.loc[0, "UTC_10min"] = observed.UTC.iloc[2]
    with pytest.raises(ValueError, match="leaks"):
        holdout.predict_window(sample, observed, candidate, None, None)


def test_hourly_eligible_windows_have_truth_and_context():
    kernel = core.module("e006_hourly_sample_test", core.E002 / "20_quantify_uncertainty/20_interpolation_holdout/10_prepare_holdout_samples.py")
    kernel.DENSE_GAP_SECONDS = 3600
    times = pd.date_range("2025-06-01T00:30:00Z", periods=10, freq="h").astype("int64").to_numpy()
    l, r = kernel.eligible_windows(times, 2)
    assert len(l) and ((r-l) == 2).all() and (l > 0).all() and (r < 9).all()
    l, r = kernel.eligible_windows(times, 1)
    assert len(l) == 0


def test_cyclone_bootstrap_uses_full_time_and_keeps_few_ship_intervals_empty():
    import numpy as np
    bootstrap = core.module("e006_bootstrap_test", HERE / "240_bootstrap_gfw_release.py")
    seed = core.module("e006_wave_seed_test", core.E002 / "30_release/70_clip_full_voyage_wave.py")
    values = np.array([[2.,4.,1.,2.,20.]]*3)
    result = bootstrap.bootstrap_group(("A","B","container"), values, "base", seed)
    assert result["tc300_at_least_pct"] == 5
    assert result["tc500_at_least_pct_ci95_low"] == 10
    assert result["share_wave_hours_ge4m_ci95_high"] == .5
    result = bootstrap.bootstrap_group(("A","B","container"), values[:2], "base", seed)
    assert np.isnan(result["tc500_at_least_pct_ci95_low"])


def test_wave_bootstrap_uses_matched_mean_without_bounds():
    import numpy as np
    bootstrap = core.module("e006_wave_bootstrap_test", HERE / "240_bootstrap_gfw_release.py")
    seed = core.module("e006_wave_estimand_test", core.E002 / "30_release/70_clip_full_voyage_wave.py")
    # E007 revision: 4 m h over 1 matched hour is 4 m, never clipped into a possible range.
    result = bootstrap.bootstrap_group(("A","B","container"), np.array([[4.,1.]]*3), "wave", seed)
    assert result["mean_swh_matched_m"] == 4
    assert result["mean_swh_matched_m_ci95_low"] == 4


def test_wave_bootstrap_maps_to_release_ci_fields():
    assembly = core.module("e006_release_schema_test", HERE / "250_assemble_gfw_release.py")
    private = pd.DataFrame({"mean_swh_matched_m": [1.2],
                            "mean_swh_matched_m_ci95_low": [.8],
                            "mean_swh_matched_m_ci95_high": [1.5]})
    published = assembly.release_bootstrap_fields(private)
    assert published.columns.tolist() == ["mean_swh_matched_m", "mean_swh_matched_ci95_low",
                                          "mean_swh_matched_ci95_high"]
    assert set(published).issubset(assembly.RELEASE_COLUMNS)
    assert published.iloc[0].tolist() == [1.2, .8, 1.5]
