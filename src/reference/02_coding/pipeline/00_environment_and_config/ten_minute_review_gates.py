"""Check CP1/CP2 against the actual annual scope before ten-minute execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from review_gates import GateNotOpen, _validate_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_scope_clearance(coding: Path, scope: dict, year: int, operation: str) -> dict:
    """Fail before creating output when year, snapshot or candidate scope differs.

    CP1 approves anchor preparation; CP2 subsequently approves reconstruction.
    Weather also needs its own WX1 decision, checked by the weather entry.
    No decision is generated or expanded by this function.
    """
    if operation not in ("companion_reconstruction", "weather_matching"):
        raise ValueError(operation)
    directory = coding / "data/review_gates"
    decisions = {}
    for gate in ("CP1", "CP2"):
        path = directory / f"{gate}_user_decision.json"
        if not path.is_file():
            raise GateNotOpen(f"{gate}: no user decision for the requested {year} scope")
        decision = json.loads(path.read_text())
        _validate_record(gate, decision, path)
        if decision["approved_scope"]["year"] != year:
            raise GateNotOpen(f"{gate}: approved year differs from requested {year}")
        decisions[gate] = decision
    cp1, cp2 = decisions["CP1"], decisions["CP2"]
    if cp1["decision"] != "approved_with_anchors" or cp1.get("anchors_adopted") is not True:
        raise GateNotOpen("CP1 does not approve anchor-based reconstruction")
    if not set(cp1["approved_scope"]["operations"]) & {"anchor_response_rematch", "companion_reconstruction"}:
        raise GateNotOpen("CP1 does not approve the anchor route")
    if cp2["decision"] != "approved" or "companion_reconstruction" not in cp2["approved_scope"]["operations"]:
        raise GateNotOpen("CP2 does not approve reconstruction")
    scope_path = Path(scope["path"])
    scope_hash = sha256(scope_path)
    if scope_hash != scope["sha256"]:
        raise GateNotOpen("Actual candidate scope bytes changed")
    for gate, decision in decisions.items():
        if decision["approved_scope"]["candidate_scope_sha256"] != scope_hash:
            raise GateNotOpen(f"{gate}: actual candidate scope was not approved")
    if cp1["input_snapshot_sha256"] != cp2["input_snapshot_sha256"]:
        raise GateNotOpen("CP1 and CP2 refer to different base-table snapshots")
    package = coding.parent / cp1["reviewed_package"]
    manifest_path = package / "cp1_review_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["input_snapshot_sha256"] != cp1["input_snapshot_sha256"]:
        raise GateNotOpen("Reviewed base-table snapshot differs from the decision")
    bound = manifest["scope_binding"]
    for key, hash_key in (("candidate_scope", "candidate_scope_sha256"),
                          ("proposed_gfw_gap_scope", "gfw_gap_scope_sha256")):
        if sha256(Path(bound[key]["path"])) != cp1["approved_scope"][hash_key]:
            raise GateNotOpen(f"Reviewed {key} bytes changed")
    columns = pd.read_parquet(scope_path, columns=["candidate_id", "departure_time"])
    if columns.candidate_id.astype(str).duplicated().any():
        raise GateNotOpen("Candidate scope has duplicate IDs")
    years = pd.to_datetime(columns.departure_time, utc=True).dt.year
    if not years.eq(year).all():
        raise GateNotOpen("Actual departure years differ from the approved year")
    return {"year": year, "scope": scope, "input_snapshot_sha256": cp1["input_snapshot_sha256"],
            "operation": operation, "decisions": {
                gate: {"path": str(directory / f"{gate}_user_decision.json"),
                       "sha256": sha256(directory / f"{gate}_user_decision.json")}
                for gate in decisions}}


def require_prepared_clearance(coding: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    clearance = require_scope_clearance(coding, manifest["sources"]["candidate_scope"],
                                        manifest["year"], "companion_reconstruction")
    base = manifest["sources"]["base_run_state"]
    require_anchor_source(coding, base, clearance)
    clearance["anchor_source_state"] = base
    return clearance


def require_anchor_source(coding: Path, base: dict, clearance: dict) -> None:
    """The actual 4--24h supplement must use the CP2-approved anchor snapshot."""
    path = Path(base["path"])
    if sha256(path) != base["sha256"]:
        raise GateNotOpen("Actual anchor supplement source state changed")
    state = json.loads(path.read_text())
    cp2 = json.loads((coding / "data/review_gates/CP2_user_decision.json").read_text())
    expected = {"input_snapshot_sha256": clearance["input_snapshot_sha256"],
                "candidate_scope_sha256": clearance["scope"]["sha256"],
                "anchor_rematch_manifest_sha256": cp2["approved_scope"].get("anchor_rematch_manifest_sha256")}
    if state.get("state") != "completed":
        raise GateNotOpen("Anchor supplement source is not completed")
    if not expected["anchor_rematch_manifest_sha256"]:
        raise GateNotOpen("CP2 has no approved anchor snapshot binding")
    if any(state.get(key) != value for key, value in expected.items()):
        raise GateNotOpen("Actual supplement uses a different base, candidate or anchor snapshot from CP2")


def require_weather_clearance(coding: Path, baseline: Path, year: int,
                              decision_path: Path | None = None,
                              prepared_manifest: Path | None = None) -> dict:
    state = json.loads((baseline / "run_state.json").read_text())
    binding = state.get("review_clearance")
    if not binding:
        actual_manifest = prepared_manifest or state.get("prepared_manifest_path")
        if not actual_manifest:
            raise GateNotOpen("Baseline has no CP1/CP2 clearance receipt; supply --prepared-input to verify its actual inputs")
        actual_manifest = Path(actual_manifest)
        if sha256(actual_manifest) != state.get("prepared_manifest_sha256"):
            raise GateNotOpen("Legacy baseline was built from a different prepared input")
        binding = require_prepared_clearance(coding, actual_manifest)
    clearance = require_scope_clearance(coding, binding["scope"], year, "weather_matching")
    if binding.get("anchor_source_state"):
        require_anchor_source(coding, binding["anchor_source_state"], clearance)
    if binding["input_snapshot_sha256"] != clearance["input_snapshot_sha256"]:
        raise GateNotOpen("Baseline snapshot differs from current clearance")
    for gate, item in binding["decisions"].items():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise GateNotOpen(f"Baseline {gate} decision changed")
    path = decision_path or coding / "data/review_gates/WX1_user_decision.json"
    decision = json.loads(path.read_text())
    scope = decision.get("approved_scope", {})
    if decision.get("decision") != f"approved_{year}_weather_candidate":
        raise GateNotOpen("WX1 does not approve this weather year")
    if scope.get("year") != year or scope.get("operations") != ["weather_matching"]:
        raise GateNotOpen("WX1 has the wrong operation or year")
    if scope.get("new_network_requests_approved") is not False or scope.get("products_modified") is not False:
        raise GateNotOpen("WX1 does not preserve offline candidate execution")
    clearance["weather_decision"] = {"path": str(path), "sha256": sha256(path)}
    return clearance
