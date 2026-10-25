"""Reject wrong scope/changed input before GFW weather can create any output."""
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e006_weather_gate", HERE / "180_run_gfw_weather.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_user_authorization_binds_the_actual_gfw_baseline():
    result = runner.require_authorized_inputs()
    assert len(result["results"]) == 64
    assert result["rules"]["gap_max_hours"] == 48
    assert "E006-gfw-ten-minute-tracks" in result["baseline"]


@pytest.mark.parametrize("field,value", [
    ("year", 2024), ("operations", ["weather_matching", "publication"]),
    ("new_network_requests_approved", True), ("products_modified", True),
    ("step6_authorized", True), ("adoption_authorized", True),
])
def test_wrong_scope_cannot_borrow_the_step5_approval(tmp_path, field, value):
    decision = json.loads(runner.DECISION.read_text())
    decision[field] = value
    path = tmp_path / "wrong_scope.json"
    path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="authorization differs"):
        runner.require_authorized_inputs(path)


def test_changed_weather_configuration_is_rejected(tmp_path):
    config = json.loads(runner.CONFIG.read_text())
    config["maximum_wet_cell_snap_km"] = 1000
    path = tmp_path / "wrong_config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="authorized input changed"):
        runner.require_authorized_inputs(config_path=path)


def test_reference_ais_authorization_cannot_be_reused(tmp_path):
    path = tmp_path / "legacy_wx1.json"
    path.write_text(json.dumps({"decision": "approved_2025_weather_candidate"}))
    with pytest.raises(ValueError, match="authorization differs"):
        runner.require_authorized_inputs(path)
