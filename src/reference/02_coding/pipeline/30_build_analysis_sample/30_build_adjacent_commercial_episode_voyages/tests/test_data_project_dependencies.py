"""Annual sample construction must work without legacy Suez exposure inputs."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from core import global_qtrim


def test_optional_exposure_does_not_change_retention(tmp_path):
    rows = pd.DataFrame({"candidate_id": ["a", "b", "c", "d"],
                         "pair_id": ["p"] * 4, "shiptype_group": ["Container"] * 4,
                         "normalized_time_ratio": [.5, 1., 2., np.nan],
                         "temporal_scope_eligible": [True, True, True, False], "d_yardstick": [10.] * 4})
    source = tmp_path / "pretrim.feather"
    rows.to_feather(source)
    exposure = pd.DataFrame({"pair_id": ["p"], "shiptype_group": ["Container"], "BaselineSuezDependence_g": [.2]})
    saved_root = global_qtrim.CODING_ROOT
    global_qtrim.CODING_ROOT = tmp_path
    try:
        for name, descriptors in (("without", None), ("legacy", exposure)):
            result = global_qtrim.process_part(source, tmp_path / (name + ".feather"), tmp_path / (name + ".json"),
                                              descriptors, .5, 1., "unsupported", "bound-test")
            if descriptors is None:
                assert "exposure_supported" not in result
    finally:
        global_qtrim.CODING_ROOT = saved_root
    plain = pd.read_feather(tmp_path / "without.feather")
    old = pd.read_feather(tmp_path / "legacy.feather")
    assert plain.candidate_id.tolist() == old.candidate_id.tolist() == ["a", "b"]
    assert "BaselineSuezDependence_g" not in plain and "suez_dependence_bin" not in plain


def test_default_dependencies_are_owned_and_exclude_legacy_exposure():
    path = STAGE / "10_build_adjacent_commercial_episode_voyages.py"
    spec = importlib.util.spec_from_file_location("data_builder_dependencies", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert all("E015" not in name and "route_shares" not in name for name in builder.FROZEN_DEPENDENCIES)
    assert all((builder.CODING_ROOT / name).is_file() for name in builder.FROZEN_DEPENDENCIES)
    config = builder.downstream_config(2024, builder.CODING_ROOT / "pilot/yardstick.feather", 3,
                                       "adjacent_commercial_episodes")
    assert "p4_city_shiptype_exposure" not in config
