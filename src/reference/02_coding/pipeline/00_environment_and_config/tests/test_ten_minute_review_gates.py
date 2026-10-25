"""Real gate failures: wrong year, changed scope, denied CP2 and stale receipts."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ten_minute_review_gates import require_scope_clearance, require_weather_clearance, require_anchor_source, sha256
from review_gates import GateNotOpen


@pytest.fixture
def approved(tmp_path):
    coding = tmp_path / "02_coding"
    directory = coding / "data/review_gates"
    directory.mkdir(parents=True)
    package = coding / "outputs/review"
    package.mkdir(parents=True)
    scope = package / "candidate_scope.parquet"
    pd.DataFrame({"candidate_id": ["2024:a"], "departure_time": ["2024-07-01T00:00:00Z"]}).to_parquet(scope)
    gap = package / "gaps.parquet"
    pd.DataFrame({"gap_id": ["a"]}).to_parquet(gap)
    record = {"path": str(scope), "sha256": sha256(scope)}
    snapshot = "a" * 64
    for gate in ("CP1", "CP2"):
        decision = {"gate": gate, "decision": "approved_with_anchors" if gate == "CP1" else "approved",
                    "anchors_adopted": True, "decided_by": "user", "decided_utc": "2026-10-01T00:00:00+00:00",
                    "reviewed_package": "02_coding/outputs/review", "input_snapshot_sha256": snapshot,
                    "verbatim_user_words": "approve this fixture scope",
                    "approved_scope": {"year": 2024, "candidate_scope_sha256": sha256(scope),
                                       "gfw_gap_scope_sha256": sha256(gap),
                                       "operations": ["anchor_response_rematch"] if gate == "CP1" else ["companion_reconstruction"]}}
        (directory / f"{gate}_user_decision.json").write_text(json.dumps(decision))
    (package / "cp1_review_manifest.json").write_text(json.dumps({
        "input_snapshot_sha256": snapshot, "scope_binding": {
            "candidate_scope": record, "proposed_gfw_gap_scope": {"path": str(gap), "sha256": sha256(gap)}}}))
    return coding, record


def update(coding, gate, field, value):
    path = coding / "data/review_gates" / f"{gate}_user_decision.json"
    d = json.loads(path.read_text())
    if field == "year": d["approved_scope"][field] = value
    else: d[field] = value
    path.write_text(json.dumps(d))


def test_approved_actual_scope_passes(approved):
    coding, scope = approved
    assert require_scope_clearance(coding, scope, 2024, "companion_reconstruction")["year"] == 2024


def test_old_year_approval_cannot_clear_new_year(approved):
    coding, scope = approved
    update(coding, "CP1", "year", 2025)
    with pytest.raises(GateNotOpen, match="approved year"):
        require_scope_clearance(coding, scope, 2024, "companion_reconstruction")


def test_changed_candidate_bytes_are_rejected(approved):
    coding, scope = approved
    with Path(scope["path"]).open("ab") as f: f.write(b"changed")
    with pytest.raises(GateNotOpen, match="bytes changed"):
        require_scope_clearance(coding, scope, 2024, "companion_reconstruction")


def test_approval_must_match_actual_departure_year(approved):
    coding, scope = approved
    # Metadata alone cannot relabel a year's actual voyage rows.
    for gate in ("CP1", "CP2"): update(coding, gate, "year", 2025)
    with pytest.raises(GateNotOpen, match="Actual departure years"):
        require_scope_clearance(coding, scope, 2025, "companion_reconstruction")


def test_cp2_denial_stops_reconstruction(approved):
    coding, scope = approved
    update(coding, "CP2", "decision", "rejected")
    with pytest.raises(GateNotOpen):
        require_scope_clearance(coding, scope, 2024, "companion_reconstruction")


def test_legacy_weather_baseline_without_receipt_is_rejected(approved):
    coding, _ = approved
    baseline = coding / ".runs/baseline"
    baseline.mkdir(parents=True)
    (baseline / "run_state.json").write_text("{}")
    with pytest.raises(GateNotOpen, match="no CP1/CP2 clearance receipt"):
        require_weather_clearance(coding, baseline, 2024)


def test_cp2_changed_base_snapshot_is_rejected(approved):
    coding, scope = approved
    update(coding, "CP2", "input_snapshot_sha256", "b" * 64)
    with pytest.raises(GateNotOpen, match="different base-table snapshots"):
        require_scope_clearance(coding, scope, 2024, "companion_reconstruction")


def test_weather_cannot_reuse_a_changed_decision(approved):
    coding, scope = approved
    clearance = require_scope_clearance(coding, scope, 2024, "companion_reconstruction")
    baseline = coding / ".runs/baseline"
    baseline.mkdir(parents=True)
    (baseline / "run_state.json").write_text(json.dumps({"review_clearance": clearance}))
    update(coding, "CP2", "verbatim_user_words", "different decision wording")
    with pytest.raises(GateNotOpen, match="decision changed"):
        require_weather_clearance(coding, baseline, 2024)


def test_weather_needs_its_own_year_approval(approved):
    coding, scope = approved
    clearance = require_scope_clearance(coding, scope, 2024, "companion_reconstruction")
    baseline = coding / ".runs/baseline"
    baseline.mkdir(parents=True)
    (baseline / "run_state.json").write_text(json.dumps({"review_clearance": clearance}))
    decision = coding / "data/review_gates/WX1_user_decision.json"
    decision.write_text(json.dumps({"decision": "approved_2025_weather_candidate",
                                   "approved_scope": {"year": 2025, "operations": ["weather_matching"]}}))
    with pytest.raises(GateNotOpen, match="this weather year"):
        require_weather_clearance(coding, baseline, 2024)


def test_actual_supplement_must_match_cp2_anchor_snapshot(approved):
    coding, scope = approved
    clearance = require_scope_clearance(coding, scope, 2024, "companion_reconstruction")
    path = coding / "data/review_gates/CP2_user_decision.json"
    cp2 = json.loads(path.read_text())
    cp2["approved_scope"]["anchor_rematch_manifest_sha256"] = "c" * 64
    path.write_text(json.dumps(cp2))
    base = coding / "base_state.json"
    actual = {"state": "completed", "input_snapshot_sha256": clearance["input_snapshot_sha256"],
              "candidate_scope_sha256": scope["sha256"], "anchor_rematch_manifest_sha256": "c" * 64}
    base.write_text(json.dumps(actual))
    record = {"path": str(base), "sha256": sha256(base)}
    require_anchor_source(coding, record, clearance)
    actual["anchor_rematch_manifest_sha256"] = "d" * 64
    base.write_text(json.dumps(actual))
    record["sha256"] = sha256(base)
    with pytest.raises(GateNotOpen, match="different base, candidate or anchor snapshot"):
        require_anchor_source(coding, record, clearance)
