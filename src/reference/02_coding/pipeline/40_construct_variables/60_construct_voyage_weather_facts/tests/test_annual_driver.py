"""Current 253-shard weather driver guards; legacy 64-bucket entry remains sealed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


STAGE = Path(__file__).resolve().parents[1]
CODING = STAGE.parents[2]
BASELINE = CODING / ".runs/2026-09-21_2043_E001-2025-contract-baseline-full"
DECISION = CODING / "data/review_gates/WX1_user_decision.json"
SCRIPT = STAGE / "60_run_ten_minute_annual_weather.py"
spec = importlib.util.spec_from_file_location("weather_annual_2025", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def args(tmp_path, **changes):
    values = {
        "baseline_run": BASELINE,
        "config": STAGE / "config/2025_ten_minute_weather.json",
        "decision": DECISION,
        "output": CODING / ".runs/2026-09-23_0152_E001-weather-test-dry-run",
        "workers": 4,
        "execute": False,
        "resume": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_current_snapshot_dry_run_does_not_write(tmp_path):
    a = args(tmp_path)
    assert not a.output.exists()
    plan = module.run(a)
    assert plan["shards"] == 253 and plan["execute"] is False
    assert plan["products_modified"] == 0 and not a.output.exists()


def test_wrong_baseline_is_rejected(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "validation.json").write_text(json.dumps({"status": "pass", "endpoint_policy": "candidate"}))
    with pytest.raises(ValueError, match="contract-compliant baseline"):
        module.run(args(tmp_path, baseline_run=baseline))


def test_wrong_weather_decision_is_rejected(tmp_path):
    decision = tmp_path / "wrong.json"
    value = json.loads(DECISION.read_text())
    value["decision"] = "not_approved"
    decision.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="not approved"):
        module.run(args(tmp_path, decision=decision))


def test_worker_limit_is_enforced(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 12"):
        module.run(args(tmp_path, workers=13))


def test_changed_fingerprint_cannot_resume(tmp_path):
    # Use an existing, different run root; no files are created by this test.
    output = CODING / ".runs/2026-09-21_2055_E001-2025-weather-full"
    with pytest.raises(ValueError, match="resume fingerprint differs"):
        module.run(args(tmp_path, output=output, execute=True, resume=True))
