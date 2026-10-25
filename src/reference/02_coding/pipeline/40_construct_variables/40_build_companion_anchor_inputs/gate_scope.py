"""Bind every offline GFW action to the user-approved CP1 package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
GATE_CODE = PIPELINE / "00_environment_and_config"
if str(GATE_CODE) not in sys.path:
    sys.path.insert(0, str(GATE_CODE))

from review_gates import _validate_record  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_approved_scope(
    project: Path,
    review_package: Path,
    operation: str,
    *,
    decision_path: Path | None = None,
    expected_year: int = 2025,
) -> tuple[dict, dict]:
    decision_path = (decision_path if decision_path is not None else
                     project / "02_coding/data/review_gates/CP1_user_decision.json")
    decision = json.loads(decision_path.read_text())
    _validate_record("CP1", decision, decision_path)
    if decision["decision"] != "approved_with_anchors" or not decision["anchors_adopted"]:
        raise SystemExit("CP1 does not approve anchors")
    if operation not in decision["approved_scope"]["operations"]:
        raise SystemExit(f"CP1 does not approve {operation}")

    review_package = review_package.resolve()
    expected_package = (project / decision["reviewed_package"]).resolve()
    if review_package != expected_package:
        raise SystemExit("requested review package differs from the user-approved package")
    manifest_path = review_package / "cp1_review_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Older accepted packages predate analysis_year; their only calendar is 2025.
    actual_year = manifest.get("analysis_year", manifest.get("year", 2025))
    if decision["approved_scope"]["year"] != expected_year or actual_year != expected_year:
        raise SystemExit("CP1 decision, reviewed package and requested year differ")
    if manifest["status"] != "awaiting_user_cp1_review":
        raise SystemExit("CP1 package has an unexpected status")
    if manifest["input_snapshot_sha256"] != decision["input_snapshot_sha256"]:
        raise SystemExit("CP1 input snapshot differs from the user decision")
    candidate_path = Path(manifest["scope_binding"]["candidate_scope"]["path"])
    gap_path = Path(manifest["scope_binding"]["proposed_gfw_gap_scope"]["path"])
    if sha256(candidate_path) != decision["approved_scope"]["candidate_scope_sha256"]:
        raise SystemExit("candidate scope differs from the user-approved scope")
    if sha256(gap_path) != decision["approved_scope"]["gfw_gap_scope_sha256"]:
        raise SystemExit("GFW gap scope differs from the user-approved scope")
    if decision["approved_scope"].get("new_api_requests_approved") is not False:
        raise SystemExit("this offline stage requires new_api_requests_approved=false")
    return decision, manifest
