"""E007 rule revision: GFW fix time tolerance and the matched-hour wave estimand."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import step6_core as core
bootstrap = core.module("e006_bootstrap_test", HERE / "240_bootstrap_gfw_release.py")
seed_kernel = core.module("e006_seed_test", core.E002 / "30_release/70_clip_full_voyage_wave.py")


@pytest.mark.parametrize("field,value", [("fix_time_tolerance_hours", 0), ("fix_time_tolerance_hours", 1),
    ("wave_point_estimand", "constrained"), ("port_time_tolerance_hours", 0.5)])
def test_reject_changed_revision(tmp_path, monkeypatch, field, value):
    rule = json.loads(core.REVISION.read_text())
    rule[field] = value
    p = tmp_path / "revision.json"; p.write_text(json.dumps(rule))
    monkeypatch.setattr(core, "REVISION", p)
    with pytest.raises(ValueError): core.revision()


def test_fix_ends_widen_outward_and_keep_positions():
    half_hour = core.NS_HOUR // 2
    left, right = (1_000, 10.0, 20.0), (9_000, 11.0, 21.0)
    assert core.widen_fix_ends(left, right, half_hour) == ((1_000 - half_hour, 10.0, 20.0), (9_000 + half_hour, 11.0, 21.0))
    assert core.widen_fix_ends(None, right, half_hour) == (None, (9_000 + half_hour, 11.0, 21.0))
    assert core.widen_fix_ends(left, None, 0) == (left, None)


def test_wave_bootstrap_resamples_the_matched_mean_without_clipping():
    # wave integral (m h) and matched hours per ship
    values = np.array([[10.0, 10.0], [30.0, 10.0], [8.0, 4.0], [0.0, 0.0]])
    key = ("A", "B", "container")
    row = bootstrap.bootstrap_group(key, values, "wave", seed_kernel)
    assert row["mean_swh_matched_m"] == pytest.approx(48 / 24)
    draws = np.random.default_rng(seed_kernel.stable_seed(key)).integers(0, 4, size=(1000, 4))
    total = values[draws].sum(axis=1)
    means = np.divide(total[:, 0], total[:, 1], out=np.full(1000, np.nan), where=total[:, 1] > 0)
    finite = means[np.isfinite(means)]
    assert row["mean_swh_matched_m_valid_bootstrap_draws"] == len(finite)
    expected = np.quantile(finite, [.025, .975], method="linear")
    assert [row["mean_swh_matched_m_ci95_low"], row["mean_swh_matched_m_ci95_high"]] == pytest.approx(expected)
