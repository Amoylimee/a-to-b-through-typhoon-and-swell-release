"""User review gates CP1 and CP2, implemented as hard run boundaries.

CONTRACT sections 4 and 5 make two decisions the user's alone:

  CP1  after the pre-anchor base-table review package: whether the base table is
       acceptable, whether GFW anchors may be used at all, and in what scope.
  CP2  after anchors have been fetched, matched and filtered: whether they may drive
       track reconstruction and interpolation.

Neither an automatic check, nor a passing test suite, nor a reviewer's approval can
stand in for the user's decision. A decision is bound to the input snapshot and the
candidate scope that was actually reviewed, and to the operations the user approved.
A cached response, a small probe or a budget probe does not bypass a gate.

What this module enforces
-------------------------

1. Record completeness. Every required field must be present, of the right type and
   non-empty. A partially written decision file is not a decision.
2. The real input snapshot of the run that will actually happen. The facts come from
   stage_contract.resolve_run_facts(), which parses the same command line the upstream
   entry parses, so a run that selects another config with --config is measured as the
   config it will really load. There is no "snapshot unknown" path.
3. The approved scope, compared element by element with the run: the operation must
   appear in approved_scope.operations, the departure year must equal approved_scope.year,
   and the candidate set must hash to approved_scope.candidate_scope_sha256. A field that
   is merely present in the record proves nothing.
4. Whether anchors were adopted. CP1 records anchors_adopted explicitly and it must
   agree with the decision value. When the user accepted the base table but declined
   anchors, no anchor operation is authorised by that decision.
5. Weather on the anchor route needs CP2 as well. Weather matching consumes the
   companion product, so when CP1 adopted anchors the weather stage requires CP2 too.
   When CP1 declined anchors, weather runs on the approved no-anchor scope under CP1
   alone.
6. No bypass by running the migrated upstream entry directly. The sealed upstream
   entries re-derive the run's own facts from their real command line and re-evaluate
   every gate from scratch before doing anything. Comparing clearance tokens alone is
   not enough, because a token is derived from a decision file and can therefore be
   produced from a refused one; the seal only accepts a run that passes the gates now.

Decision file layout
--------------------

data/review_gates/CP1_user_decision.json::

    {
      "gate": "CP1",
      "decision": "approved_with_anchors" | "approved_without_anchors" | "rework",
      "anchors_adopted": true | false,
      "decided_by": "user",
      "decided_utc": "2026-..-..T..:..:..+00:00",
      "reviewed_package": "02_coding/outputs/<timestamped review package>",
      "input_snapshot_sha256": "<64 hex, the reviewed input manifest>",
      "approved_scope": {
        "operations": ["weather_matching", ...],
        "year": 2025,
        "candidate_scope_sha256": "<64 hex>",
        "notes": "<free text, optional>"
      },
      "verbatim_user_words": "<the user's own words>"
    }

CP2 is the same shape with decision in {"approved", "rework"} and no anchors_adopted
field required (reaching CP2 already implies anchors).

Workers do not write these files from their own judgement. The file records a decision
the user actually made; the verbatim words are kept so the scope can be audited later.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from stage_contract import RunFacts, RunFactsUnresolved, resolve_run_facts

HEX64 = re.compile(r"^[0-9a-f]{64}$")
CLEARANCE_ENV = "E001_GATE_CLEARANCE"

GATES: dict[str, dict] = {
    "CP1": {
        "title": "pre-anchor base-table quality review",
        "decides": ("whether the base table is acceptable, whether GFW anchors are used "
                    "at all, and in what scope"),
        "blocks": ("GFW anchor target construction, raw-response rematching or binding, "
                   "any anchor API or quota request, companion reconstruction and the "
                   "weather stage that depends on it"),
        "approved_values": ("approved_with_anchors", "approved_without_anchors"),
        "rejected_values": ("rework",),
        "requires_anchors_flag": True,
    },
    "CP2": {
        "title": "anchor quality review",
        "decides": "whether the accepted anchors may drive reconstruction and interpolation",
        "blocks": "anchor-driven companion track reconstruction and interpolation",
        "approved_values": ("approved",),
        "rejected_values": ("rework",),
        "requires_anchors_flag": False,
    },
}

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "gate": str,
    "decision": str,
    "decided_by": str,
    "decided_utc": str,
    "reviewed_package": str,
    "input_snapshot_sha256": str,
    "approved_scope": dict,
    "verbatim_user_words": str,
}
REQUIRED_SCOPE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "operations": list,
    "year": int,
    "candidate_scope_sha256": str,
}

# Which gates each operation needs. "anchor" marks operations that only an
# anchors-adopted CP1 can authorise.
OPERATIONS: dict[str, dict] = {
    "anchor_target_build": {"gates": ("CP1",), "anchor": True},
    "anchor_response_rematch": {"gates": ("CP1",), "anchor": True},
    "anchor_acquisition_request": {"gates": ("CP1",), "anchor": True},
    "companion_reconstruction": {"gates": ("CP1", "CP2"), "anchor": True},
    "weather_matching": {"gates": ("CP1",), "anchor": False,
                         "cp2_when_anchors_adopted": True},
}


class GateNotOpen(SystemExit):
    """Raised when a gated stage must not run. Inherits SystemExit so it stops a run."""


@dataclass
class Clearance:
    operation: str
    facts: RunFacts
    decisions: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)

    @property
    def input_snapshot_sha256(self) -> str:
        return self.facts.input_snapshot_sha256

    def env(self) -> dict[str, str]:
        return {CLEARANCE_ENV: json.dumps(
            {"operation": self.operation, "facts": self.facts.as_dict(),
             "tokens": self.tokens}, sort_keys=True)}

    @property
    def anchors_adopted(self) -> bool:
        return bool(self.decisions.get("CP1", {}).get("anchors_adopted", False))


def gate_path(data_root: Path | str, gate: str) -> Path:
    return Path(data_root) / "review_gates" / f"{gate}_user_decision.json"


def _decision_token(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def _refuse(message: str) -> None:
    raise GateNotOpen("\n" + message.rstrip() + "\n")


def _validate_record(gate: str, decision: dict, path: Path) -> None:
    spec = GATES[gate]
    missing = [name for name in REQUIRED_FIELDS if name not in decision]
    if missing:
        _refuse(f"{gate} decision record is incomplete: missing {sorted(missing)}\n"
                f"  file {path}\n"
                f"A partially written decision file is not a decision.")

    for name, expected in REQUIRED_FIELDS.items():
        value = decision[name]
        if not isinstance(value, expected):
            _refuse(f"{gate} decision field {name!r} has type "
                    f"{type(value).__name__}, expected {expected.__name__}\n  file {path}")
        if isinstance(value, str) and not value.strip():
            _refuse(f"{gate} decision field {name!r} is blank\n  file {path}\n"
                    f"An empty field records nothing.")

    if decision["gate"] != gate:
        _refuse(f"decision file records gate {decision['gate']!r} but was read as {gate}\n"
                f"  file {path}")

    if decision["decided_by"] != "user":
        _refuse(f"{gate} decision was not recorded as the user's "
                f"(decided_by={decision['decided_by']!r}).\n"
                f"Only the user opens this gate. A reviewer's or an agent's approval "
                f"does not substitute for it.")

    try:
        datetime.fromisoformat(decision["decided_utc"])
    except ValueError:
        _refuse(f"{gate} decided_utc is not an ISO 8601 timestamp: "
                f"{decision['decided_utc']!r}\n  file {path}")

    if not HEX64.match(decision["input_snapshot_sha256"]):
        _refuse(f"{gate} input_snapshot_sha256 is not a SHA-256 hex digest: "
                f"{decision['input_snapshot_sha256']!r}\n  file {path}")

    scope = decision["approved_scope"]
    scope_missing = [name for name in REQUIRED_SCOPE_FIELDS if name not in scope]
    if scope_missing:
        _refuse(f"{gate} approved_scope is incomplete: missing {sorted(scope_missing)}\n"
                f"  file {path}\nAn approval without a stated scope authorises nothing.")
    for name, expected in REQUIRED_SCOPE_FIELDS.items():
        if not isinstance(scope[name], expected):
            _refuse(f"{gate} approved_scope.{name} has type "
                    f"{type(scope[name]).__name__}, expected {expected.__name__}\n"
                    f"  file {path}")
    if not scope["operations"]:
        _refuse(f"{gate} approved_scope.operations is empty\n  file {path}\n"
                f"An approval that names no operation authorises nothing.")
    unknown = sorted(set(scope["operations"]) - set(OPERATIONS))
    if unknown:
        _refuse(f"{gate} approved_scope.operations names unknown operations {unknown}\n"
                f"  file {path}\n  known operations: {sorted(OPERATIONS)}")
    if not HEX64.match(scope["candidate_scope_sha256"]):
        _refuse(f"{gate} approved_scope.candidate_scope_sha256 is not a SHA-256 hex "
                f"digest: {scope['candidate_scope_sha256']!r}\n  file {path}")

    if spec["requires_anchors_flag"]:
        if "anchors_adopted" not in decision:
            _refuse(f"{gate} decision record is incomplete: missing 'anchors_adopted'\n"
                    f"  file {path}\nCP1 must state explicitly whether anchors are used.")
        if not isinstance(decision["anchors_adopted"], bool):
            _refuse(f"{gate} anchors_adopted must be a boolean, got "
                    f"{type(decision['anchors_adopted']).__name__}\n  file {path}")
        expected_flag = decision["decision"] == "approved_with_anchors"
        if decision["decision"] in spec["approved_values"] and \
                decision["anchors_adopted"] != expected_flag:
            _refuse(f"{gate} is inconsistent: decision={decision['decision']!r} but "
                    f"anchors_adopted={decision['anchors_adopted']}\n  file {path}\n"
                    f"The two must agree before anything runs.")


def read_decision(data_root: Path | str, gate: str) -> dict | None:
    path = gate_path(data_root, gate)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        _refuse(f"{gate} decision file is not valid JSON: {error}\n  file {path}")


def require_gate(data_root: Path | str, gate: str, facts: RunFacts) -> dict:
    """Return the user's decision for `gate`, or stop the run.

    Every element of the approved scope is compared with the run that is actually about
    to happen: the input snapshot, the operation, the departure year and the candidate
    set. A field that is merely present in the record proves nothing.
    """
    if gate not in GATES:
        raise ValueError(f"unknown gate {gate}")
    operation = facts.operation
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation {operation}")
    if not HEX64.match(facts.input_snapshot_sha256 or ""):
        _refuse(f"{gate} was asked to clear {operation!r} without a real input snapshot "
                f"hash (got {facts.input_snapshot_sha256!r}).\n"
                f"A gate cannot be cleared against an unknown input.")

    spec = GATES[gate]
    path = gate_path(data_root, gate)
    decision = read_decision(data_root, gate)

    if decision is None:
        _refuse(f"{gate} is not open. This stage is a hard stop.\n"
                f"  gate      {gate}: {spec['title']}\n"
                f"  decides   {spec['decides']}\n"
                f"  blocks    {spec['blocks']}\n"
                f"  requested {operation}\n"
                f"  expected  {path}\n"
                f"The user has not made this decision. Automatic checks, passing tests "
                f"and a reviewer's approval do not substitute for it, and no cached "
                f"response, probe or budget estimate may bypass it.")

    _validate_record(gate, decision, path)

    if decision["decision"] not in spec["approved_values"]:
        _refuse(f"{gate} is recorded as {decision['decision']!r}, which does not "
                f"authorise {operation!r}.\n"
                f"  approved_scope {decision['approved_scope']}\n"
                f"  file {path}")

    if decision["input_snapshot_sha256"] != facts.input_snapshot_sha256:
        _refuse(f"{gate} was decided against a different input snapshot.\n"
                f"  decided on   {decision['input_snapshot_sha256']}\n"
                f"  about to use {facts.input_snapshot_sha256}\n"
                f"  config       {facts.config_path}\n"
                f"A decision is bound to the base table and candidate range that was "
                f"reviewed; it does not carry over to a changed one.")

    scope = decision["approved_scope"]
    if operation not in scope["operations"]:
        _refuse(f"{gate} does not authorise {operation!r}.\n"
                f"  approved operations {sorted(scope['operations'])}\n"
                f"  file {path}\n"
                f"Approving one operation does not approve another, and approving a "
                f"probe does not approve a full run.")

    if scope["year"] != facts.year:
        _refuse(f"{gate} approved year {scope['year']} but this run processes "
                f"{facts.year}.\n"
                f"  config {facts.config_path}\n"
                f"  file   {path}\n"
                f"An approval for one year does not authorise another.")

    if scope["candidate_scope_sha256"] != facts.candidate_scope_sha256:
        _refuse(f"{gate} approved a different candidate set.\n"
                f"  approved     {scope['candidate_scope_sha256']}\n"
                f"  about to use {facts.candidate_scope_sha256}\n"
                f"  candidate set {facts.candidate_scope_path}\n"
                f"The decision was made about the voyages the user reviewed, not these.")

    if OPERATIONS[operation]["anchor"] and gate == "CP1" and not decision["anchors_adopted"]:
        _refuse(f"CP1 accepted the base table but declined anchors "
                f"(decision={decision['decision']!r}), so {operation!r} is not "
                f"authorised.\n  file {path}\n"
                f"Accepting the data quality is not the same as authorising anchors.")

    return decision


def require_operation(data_root: Path | str, facts: RunFacts) -> Clearance:
    """Clear every gate `facts.operation` needs, then return a Clearance.

    Weather matching is the case that needs more than a static list: it always needs
    CP1, and it additionally needs CP2 whenever CP1 adopted anchors, because the track
    it consumes is then anchor-driven.
    """
    operation = facts.operation
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation {operation}")
    spec = OPERATIONS[operation]
    clearance = Clearance(operation=operation, facts=facts)

    needed = list(spec["gates"])
    for gate in needed:
        clearance.decisions[gate] = require_gate(data_root, gate, facts)

    if spec.get("cp2_when_anchors_adopted") and clearance.anchors_adopted and "CP2" not in needed:
        clearance.decisions["CP2"] = require_gate(data_root, "CP2", facts)

    for gate in clearance.decisions:
        clearance.tokens[gate] = _decision_token(gate_path(data_root, gate))
    return clearance


def seal_check(caller_file: str, operation: str, data_root: Path | str,
               argv: list[str] | None = None) -> None:
    """Refuse to run a migrated upstream entry that was not cleared for this exact run.

    Comparing clearance tokens is not enough: the tokens are derived from the decision
    files, so anyone who can read a "rework" CP1 can also produce its token. The seal
    therefore re-derives the run's own facts from the command line this process is
    actually running with, and re-evaluates every gate from scratch. A refused decision,
    a missing CP2, another config, another year or another candidate set all stop the
    run here even when the environment says otherwise.
    """
    import sys

    raw = os.environ.get(CLEARANCE_ENV)
    if not raw:
        _refuse(f"{Path(caller_file).name} is a sealed upstream entry and was started "
                f"without clearance.\n"
                f"  operation {operation}\n"
                f"Run the gated stage entry instead; it clears "
                f"{OPERATIONS[operation]['gates']} first. Running the upstream module "
                f"directly is the bypass path this seal closes.")
    try:
        clearance = json.loads(raw)
    except json.JSONDecodeError:
        _refuse(f"{CLEARANCE_ENV} is not valid JSON; refusing to run "
                f"{Path(caller_file).name}.")

    if clearance.get("operation") != operation:
        _refuse(f"clearance is for {clearance.get('operation')!r}, not {operation!r}; "
                f"refusing to run {Path(caller_file).name}.")

    try:
        facts = resolve_run_facts(operation, list(sys.argv[1:] if argv is None else argv))
    except RunFactsUnresolved as error:
        _refuse(f"refusing to run {Path(caller_file).name}: {error}")

    # Re-evaluate the gates against the run that is actually happening. This is what
    # closes the bypass, not the token comparison below.
    fresh = require_operation(data_root, facts)

    recorded = clearance.get("facts") or {}
    for key in ("input_snapshot_sha256", "year", "candidate_scope_sha256", "config_path"):
        if key in recorded and recorded[key] != facts.as_dict()[key]:
            _refuse(f"the clearance was granted for a different run "
                    f"({key}: {recorded[key]!r} vs {facts.as_dict()[key]!r}); "
                    f"refusing to run {Path(caller_file).name}.")

    tokens = clearance.get("tokens") or {}
    if not tokens:
        _refuse(f"clearance carries no gate tokens; refusing to run "
                f"{Path(caller_file).name}.")
    if tokens != fresh.tokens:
        _refuse(f"the clearance does not match the decisions now on disk "
                f"(cleared {sorted(tokens)}, required {sorted(fresh.tokens)}); "
                f"refusing to run {Path(caller_file).name}.")


def status(data_root: Path | str) -> dict:
    out = {}
    for gate in GATES:
        path = gate_path(data_root, gate)
        if not path.exists():
            out[gate] = {"decision": "not_recorded"}
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            out[gate] = {"decision": "unreadable"}
            continue
        out[gate] = {"decision": record.get("decision", "malformed"),
                     "anchors_adopted": record.get("anchors_adopted"),
                     "approved_operations": (record.get("approved_scope") or {}).get("operations"),
                     "input_snapshot_sha256": record.get("input_snapshot_sha256")}
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="show the CP1/CP2 gate status")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    for gate, record in status(args.data_root).items():
        print(f"{gate}  {record.get('decision')}  "
              f"anchors_adopted={record.get('anchors_adopted')}  "
              f"operations={record.get('approved_operations')}")
