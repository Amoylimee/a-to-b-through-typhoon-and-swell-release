"""Rejection tests for the CP1/CP2 review gates.

Every test here asserts that something is *refused*. A gate that only works when the
paperwork is right is not a gate; what matters is that an incomplete record, a stale
snapshot, a swapped config, a scope that names another year or another candidate set,
an operation outside the approved scope, a missing CP2 on the anchor route, or a direct
run of a sealed upstream entry all stop the run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import review_gates as rg  # noqa: E402
from input_snapshot import sha256_file, snapshot_of  # noqa: E402
from stage_contract import RunFacts, RunFactsUnresolved, resolve_run_facts  # noqa: E402

SNAPSHOT = "a" * 64
OTHER_SNAPSHOT = "b" * 64
SCOPE_HASH = "c" * 64
OTHER_SCOPE_HASH = "d" * 64


def facts(operation="weather_matching", snapshot=SNAPSHOT, year=2025,
          scope=SCOPE_HASH, config="/nowhere/config.json") -> RunFacts:
    return RunFacts(operation=operation, config_path=Path(config),
                    inputs=(Path(config),), input_snapshot_sha256=snapshot,
                    year=year, candidate_scope_sha256=scope,
                    candidate_scope_path=Path("/nowhere/voyages.parquet"))


def cp1_record(**overrides) -> dict:
    record = {
        "gate": "CP1",
        "decision": "approved_with_anchors",
        "anchors_adopted": True,
        "decided_by": "user",
        "decided_utc": "2026-09-08T02:00:00+00:00",
        "reviewed_package": "02_coding/outputs/2026-09-08_0200_CP1_review",
        "input_snapshot_sha256": SNAPSHOT,
        "approved_scope": {
            "operations": ["anchor_target_build", "companion_reconstruction",
                           "weather_matching"],
            "year": 2025,
            "candidate_scope_sha256": SCOPE_HASH,
        },
        "verbatim_user_words": "底表可以，锚点按这个范围做",
    }
    record.update(overrides)
    return record


def cp2_record(**overrides) -> dict:
    record = {
        "gate": "CP2",
        "decision": "approved",
        "decided_by": "user",
        "decided_utc": "2026-09-08T03:00:00+00:00",
        "reviewed_package": "02_coding/outputs/2026-09-08_0300_CP2_review",
        "input_snapshot_sha256": SNAPSHOT,
        "approved_scope": {
            "operations": ["companion_reconstruction", "weather_matching"],
            "year": 2025,
            "candidate_scope_sha256": SCOPE_HASH,
        },
        "verbatim_user_words": "锚点没问题，可以重建",
    }
    record.update(overrides)
    return record


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "review_gates").mkdir(parents=True)
    return tmp_path


def write(data_root: Path, gate: str, record: dict | str) -> None:
    path = rg.gate_path(data_root, gate)
    path.write_text(record if isinstance(record, str)
                    else json.dumps(record, ensure_ascii=False))


# --- no decision at all -----------------------------------------------------

def test_missing_decision_file_refuses(data_root):
    with pytest.raises(rg.GateNotOpen, match="CP1 is not open"):
        rg.require_gate(data_root, "CP1", facts())


def test_unreadable_decision_file_refuses(data_root):
    write(data_root, "CP1", "{not json")
    with pytest.raises(rg.GateNotOpen, match="not valid JSON"):
        rg.require_gate(data_root, "CP1", facts())


# --- record completeness ----------------------------------------------------

@pytest.mark.parametrize("field", sorted(rg.REQUIRED_FIELDS))
def test_每个必填字段缺失都被拒(data_root, field):
    record = cp1_record()
    record.pop(field)
    write(data_root, "CP1", record)
    with pytest.raises(rg.GateNotOpen, match="incomplete|missing"):
        rg.require_gate(data_root, "CP1", facts())


def test_missing_anchors_adopted_refuses(data_root):
    record = cp1_record()
    record.pop("anchors_adopted")
    write(data_root, "CP1", record)
    with pytest.raises(rg.GateNotOpen, match="anchors_adopted"):
        rg.require_gate(data_root, "CP1", facts())


@pytest.mark.parametrize("field", ["reviewed_package", "verbatim_user_words",
                                   "input_snapshot_sha256"])
def test_blank_field_refuses(data_root, field):
    write(data_root, "CP1", cp1_record(**{field: "   "}))
    with pytest.raises(rg.GateNotOpen, match="blank"):
        rg.require_gate(data_root, "CP1", facts())


def test_scope_without_operations_refuses(data_root):
    write(data_root, "CP1", cp1_record(approved_scope={
        "operations": [], "year": 2025, "candidate_scope_sha256": SCOPE_HASH}))
    with pytest.raises(rg.GateNotOpen, match="authorises nothing"):
        rg.require_gate(data_root, "CP1", facts())


def test_scope_missing_candidate_hash_refuses(data_root):
    write(data_root, "CP1", cp1_record(approved_scope={
        "operations": ["weather_matching"], "year": 2025}))
    with pytest.raises(rg.GateNotOpen, match="approved_scope is incomplete"):
        rg.require_gate(data_root, "CP1", facts())


def test_unknown_operation_in_scope_refuses(data_root):
    write(data_root, "CP1", cp1_record(approved_scope={
        "operations": ["weather_matching", "do_whatever"], "year": 2025,
        "candidate_scope_sha256": SCOPE_HASH}))
    with pytest.raises(rg.GateNotOpen, match="unknown operations"):
        rg.require_gate(data_root, "CP1", facts())


def test_bad_timestamp_refuses(data_root):
    write(data_root, "CP1", cp1_record(decided_utc="last tuesday"))
    with pytest.raises(rg.GateNotOpen, match="ISO 8601"):
        rg.require_gate(data_root, "CP1", facts())


def test_non_user_decision_refuses(data_root):
    write(data_root, "CP1", cp1_record(decided_by="supreme_agent"))
    with pytest.raises(rg.GateNotOpen, match="Only the user opens this gate"):
        rg.require_gate(data_root, "CP1", facts())


def test_rework_decision_refuses(data_root):
    write(data_root, "CP1", cp1_record(decision="rework", anchors_adopted=False))
    with pytest.raises(rg.GateNotOpen, match="does not authorise"):
        rg.require_gate(data_root, "CP1", facts())


def test_anchors_flag_inconsistent_with_decision_refuses(data_root):
    write(data_root, "CP1", cp1_record(decision="approved_without_anchors",
                                       anchors_adopted=True))
    with pytest.raises(rg.GateNotOpen, match="inconsistent"):
        rg.require_gate(data_root, "CP1", facts())


# --- the real input snapshot ------------------------------------------------

def test_wrong_input_snapshot_refuses(data_root):
    write(data_root, "CP1", cp1_record())
    with pytest.raises(rg.GateNotOpen, match="different input snapshot"):
        rg.require_gate(data_root, "CP1", facts(snapshot=OTHER_SNAPSHOT))


@pytest.mark.parametrize("bogus", [None, "", "not-a-hash", "AB" * 32])
def test_caller_without_a_real_snapshot_refuses(data_root, bogus):
    write(data_root, "CP1", cp1_record())
    with pytest.raises(rg.GateNotOpen, match="without a real input snapshot"):
        rg.require_gate(data_root, "CP1", facts(snapshot=bogus))


def test_snapshot_over_missing_input_refuses(tmp_path):
    present = tmp_path / "present.json"
    present.write_text("{}")
    with pytest.raises(FileNotFoundError, match="declared inputs are missing"):
        snapshot_of([present, tmp_path / "absent.json"])


def test_snapshot_changes_when_an_input_changes(tmp_path):
    one = tmp_path / "one.txt"
    one.write_text("a")
    first, _ = snapshot_of([one])
    one.write_text("b")
    second, _ = snapshot_of([one])
    assert first != second


# --- approved scope: operation, year, candidate set -------------------------

def test_operation_outside_approved_scope_refuses(data_root):
    write(data_root, "CP1", cp1_record(approved_scope={
        "operations": ["weather_matching"], "year": 2025,
        "candidate_scope_sha256": SCOPE_HASH}))
    with pytest.raises(rg.GateNotOpen, match="does not authorise 'anchor_target_build'"):
        rg.require_gate(data_root, "CP1", facts(operation="anchor_target_build"))


def test_anchor_operation_refused_when_anchors_declined(data_root):
    write(data_root, "CP1", cp1_record(
        decision="approved_without_anchors", anchors_adopted=False,
        approved_scope={"operations": ["anchor_target_build", "weather_matching"],
                        "year": 2025, "candidate_scope_sha256": SCOPE_HASH}))
    with pytest.raises(rg.GateNotOpen, match="declined anchors"):
        rg.require_gate(data_root, "CP1", facts(operation="anchor_target_build"))


def test_批准另一年份不能放行本年份(data_root):
    """A decision that names 2024 must not clear a 2025 run."""
    write(data_root, "CP1", cp1_record(approved_scope={
        "operations": ["weather_matching"], "year": 2024,
        "candidate_scope_sha256": SCOPE_HASH}))
    with pytest.raises(rg.GateNotOpen,
                       match="approved year 2024 but this run processes 2025"):
        rg.require_gate(data_root, "CP1", facts(year=2025))


def test_年份对但候选集合不对也被拒(data_root):
    write(data_root, "CP1", cp1_record())
    with pytest.raises(rg.GateNotOpen, match="approved a different candidate set"):
        rg.require_gate(data_root, "CP1", facts(scope=OTHER_SCOPE_HASH))


def test_年份与候选集合都对才通过(data_root):
    write(data_root, "CP1", cp1_record(
        decision="approved_without_anchors", anchors_adopted=False,
        approved_scope={"operations": ["weather_matching"], "year": 2025,
                        "candidate_scope_sha256": SCOPE_HASH}))
    decision = rg.require_gate(data_root, "CP1", facts())
    assert decision["approved_scope"]["year"] == 2025


# --- CP2 on the anchor route ------------------------------------------------

def test_weather_needs_cp2_when_anchors_adopted(data_root):
    write(data_root, "CP1", cp1_record())
    with pytest.raises(rg.GateNotOpen, match="CP2 is not open"):
        rg.require_operation(data_root, facts())


def test_weather_runs_under_cp1_alone_when_anchors_declined(data_root):
    write(data_root, "CP1", cp1_record(
        decision="approved_without_anchors", anchors_adopted=False,
        approved_scope={"operations": ["weather_matching"], "year": 2025,
                        "candidate_scope_sha256": SCOPE_HASH}))
    clearance = rg.require_operation(data_root, facts())
    assert set(clearance.decisions) == {"CP1"}
    assert clearance.anchors_adopted is False


def test_weather_on_the_anchor_route_needs_both_gates(data_root):
    write(data_root, "CP1", cp1_record())
    write(data_root, "CP2", cp2_record())
    clearance = rg.require_operation(data_root, facts())
    assert set(clearance.decisions) == {"CP1", "CP2"}


def test_reconstruction_without_cp2_refuses(data_root):
    write(data_root, "CP1", cp1_record())
    with pytest.raises(rg.GateNotOpen, match="CP2 is not open"):
        rg.require_operation(data_root, facts(operation="companion_reconstruction"))


def test_reconstruction_with_stale_cp2_snapshot_refuses(data_root):
    write(data_root, "CP1", cp1_record())
    write(data_root, "CP2", cp2_record(input_snapshot_sha256=OTHER_SNAPSHOT))
    with pytest.raises(rg.GateNotOpen, match="different input snapshot"):
        rg.require_operation(data_root, facts(operation="companion_reconstruction"))


def test_reconstruction_with_cp2_naming_another_year_refuses(data_root):
    write(data_root, "CP1", cp1_record())
    write(data_root, "CP2", cp2_record(approved_scope={
        "operations": ["companion_reconstruction"], "year": 2024,
        "candidate_scope_sha256": SCOPE_HASH}))
    with pytest.raises(rg.GateNotOpen, match="approved year 2024"):
        rg.require_operation(data_root, facts(operation="companion_reconstruction"))


def test_reconstruction_with_both_gates_clears(data_root):
    write(data_root, "CP1", cp1_record())
    write(data_root, "CP2", cp2_record())
    clearance = rg.require_operation(data_root, facts(operation="companion_reconstruction"))
    assert set(clearance.decisions) == {"CP1", "CP2"}
    assert set(clearance.tokens) == {"CP1", "CP2"}


# --- the run's own facts come from the run's own command line ---------------

def build_weather_stage(tmp_path: Path, year: int = 2025, voyages_body: str = "v1",
                        buckets: int = 3) -> Path:
    """A complete, self-contained weather stage config tree under tmp_path.

    It includes the two per-bucket directories the weather computation really reads:
    the companion tracks it samples and the AIS buckets it opens.
    """
    project = tmp_path / "project"
    (project / "voyages").mkdir(parents=True)
    (project / "voyages" / "candidate_voyages.parquet").write_text(voyages_body)
    for bucket in range(buckets):
        companion = project / "companion" / "run" / "buckets" / f"bucket_{bucket:04d}"
        companion.mkdir(parents=True)
        (companion / "companion_track.parquet").write_text(f"track{bucket}")
        ais = project / "ais_buckets" / f"bucket_{bucket:04d}"
        ais.mkdir(parents=True)
        (ais / "2025_shard_0000.feather").write_text(f"ais{bucket}")
    (project / "companion" / "run" / "voyage_coverage_base.parquet").write_text("base")
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("ocean_mask.npy", "ibtracs.csv", "coverage_audit.json"):
        (assets / name).write_text(name)
    waves = tmp_path / "waves"
    waves.mkdir()
    for month in range(1, 13):
        (waves / f"wave_{month:02d}.nc").write_text(f"wave{month}")
    config = tmp_path / f"weather_{year}.json"
    config.write_text(json.dumps({
        "source_project": str(project),
        "voyages": "voyages/candidate_voyages.parquet",
        "companion_product": "companion",
        "ais_bucket_root": "ais_buckets",
        "year": year,
        "wave_directory": str(waves),
        "wave_filename_template": "wave_{month:02d}.nc",
        "ocean_mask_path": str(assets / "ocean_mask.npy"),
        "ibtracs_path": str(assets / "ibtracs.csv"),
        "ibtracs_calendar_coverage_audit": str(assets / "coverage_audit.json"),
    }))
    return config


def test_facts_follow_the_config_the_run_selects(tmp_path):
    """Approving one config must not clear a run that selects another with --config."""
    approved = build_weather_stage(tmp_path / "a", year=2025)
    other = build_weather_stage(tmp_path / "b", year=2025, voyages_body="different")

    approved_facts = resolve_run_facts("weather_matching", ["--config", str(approved)])
    other_facts = resolve_run_facts("weather_matching", ["--config", str(other)])

    assert approved_facts.config_path == approved.resolve()
    assert other_facts.config_path == other.resolve()
    assert approved_facts.input_snapshot_sha256 != other_facts.input_snapshot_sha256
    assert approved_facts.candidate_scope_sha256 != other_facts.candidate_scope_sha256


def test_批准一个配置不能放行另一个配置(data_root, tmp_path):
    approved = build_weather_stage(tmp_path / "a")
    other = build_weather_stage(tmp_path / "b", voyages_body="different")
    approved_facts = resolve_run_facts("weather_matching", ["--config", str(approved)])
    other_facts = resolve_run_facts("weather_matching", ["--config", str(other)])

    write(data_root, "CP1", cp1_record(
        decision="approved_without_anchors", anchors_adopted=False,
        input_snapshot_sha256=approved_facts.input_snapshot_sha256,
        approved_scope={"operations": ["weather_matching"],
                        "year": approved_facts.year,
                        "candidate_scope_sha256": approved_facts.candidate_scope_sha256}))

    rg.require_operation(data_root, approved_facts)  # the approved run clears
    with pytest.raises(rg.GateNotOpen,
                       match="different input snapshot|different candidate set"):
        rg.require_operation(data_root, other_facts)


def test_facts_follow_the_year_in_the_selected_config(tmp_path):
    twenty_four = build_weather_stage(tmp_path / "y24", year=2024)
    resolved = resolve_run_facts("weather_matching", ["--config", str(twenty_four)])
    assert resolved.year == 2024


def test_run_with_no_config_and_no_default_is_unresolvable():
    with pytest.raises(RunFactsUnresolved, match="unknown"):
        resolve_run_facts("companion_reconstruction", [])


def test_candidate_set_missing_is_unresolvable(tmp_path):
    """The voyage table is both a declared input and the candidate set, so its absence
    is refused at the snapshot step; either way the run cannot be described."""
    config = build_weather_stage(tmp_path)
    project = json.loads(config.read_text())["source_project"]
    Path(project, "voyages", "candidate_voyages.parquet").unlink()
    with pytest.raises(RunFactsUnresolved, match="missing|candidate set"):
        resolve_run_facts("weather_matching", ["--config", str(config)])


def test_missing_wave_input_is_unresolvable(tmp_path):
    config = build_weather_stage(tmp_path)
    Path(json.loads(config.read_text())["wave_directory"], "wave_07.nc").unlink()
    with pytest.raises(RunFactsUnresolved, match="missing"):
        resolve_run_facts("weather_matching", ["--config", str(config)])


def test_candidate_scope_hash_is_the_voyage_table(tmp_path):
    config = build_weather_stage(tmp_path)
    resolved = resolve_run_facts("weather_matching", ["--config", str(config)])
    assert resolved.candidate_scope_sha256 == sha256_file(resolved.candidate_scope_path)


# --- the data the run really consumes is in the snapshot --------------------

def approved_for(data_root: Path, run_facts) -> None:
    write(data_root, "CP1", cp1_record(
        decision="approved_without_anchors", anchors_adopted=False,
        input_snapshot_sha256=run_facts.input_snapshot_sha256,
        approved_scope={"operations": ["weather_matching"], "year": run_facts.year,
                        "candidate_scope_sha256": run_facts.candidate_scope_sha256}))


def test_快照覆盖天气真正消费的伴随轨迹与AIS桶(tmp_path):
    config = build_weather_stage(tmp_path, buckets=3)
    resolved = resolve_run_facts("weather_matching", ["--config", str(config)])
    names = [str(p) for p in resolved.inputs]
    assert any("companion/run/buckets/bucket_0000/companion_track.parquet" in n
               for n in names), "the companion tracks the stage samples are not declared"
    assert any("ais_buckets/bucket_0000/2025_shard_0000.feather" in n
               for n in names), "the AIS buckets the stage reads are not declared"
    assert any("voyage_coverage_base.parquet" in n for n in names)


def test_替换一个AIS桶文件会让旧批准失效(data_root, tmp_path):
    config = build_weather_stage(tmp_path)
    before = resolve_run_facts("weather_matching", ["--config", str(config)])
    approved_for(data_root, before)
    rg.require_operation(data_root, before)  # the approved batch clears

    project = Path(json.loads(config.read_text())["source_project"])
    (project / "ais_buckets" / "bucket_0001" / "2025_shard_0000.feather").write_text("swapped")

    after = resolve_run_facts("weather_matching", ["--config", str(config)])
    assert after.input_snapshot_sha256 != before.input_snapshot_sha256
    with pytest.raises(rg.GateNotOpen, match="different input snapshot"):
        rg.require_operation(data_root, after)


def test_删除一个伴随轨迹文件会让旧批准失效(data_root, tmp_path):
    config = build_weather_stage(tmp_path)
    before = resolve_run_facts("weather_matching", ["--config", str(config)])
    approved_for(data_root, before)

    project = Path(json.loads(config.read_text())["source_project"])
    (project / "companion" / "run" / "buckets" / "bucket_0002"
     / "companion_track.parquet").unlink()

    after = resolve_run_facts("weather_matching", ["--config", str(config)])
    assert after.input_snapshot_sha256 != before.input_snapshot_sha256
    with pytest.raises(rg.GateNotOpen, match="different input snapshot"):
        rg.require_operation(data_root, after)


def test_新增一个AIS桶文件会让旧批准失效(data_root, tmp_path):
    config = build_weather_stage(tmp_path)
    before = resolve_run_facts("weather_matching", ["--config", str(config)])
    approved_for(data_root, before)

    project = Path(json.loads(config.read_text())["source_project"])
    (project / "ais_buckets" / "bucket_0001" / "2025_shard_0001.feather").write_text("extra")

    after = resolve_run_facts("weather_matching", ["--config", str(config)])
    assert after.input_snapshot_sha256 != before.input_snapshot_sha256
    with pytest.raises(rg.GateNotOpen, match="different input snapshot"):
        rg.require_operation(data_root, after)


def test_新增一整个桶目录也会让旧批准失效(data_root, tmp_path):
    config = build_weather_stage(tmp_path)
    before = resolve_run_facts("weather_matching", ["--config", str(config)])
    approved_for(data_root, before)

    project = Path(json.loads(config.read_text())["source_project"])
    extra = project / "ais_buckets" / "bucket_0009"
    extra.mkdir(parents=True)
    (extra / "2025_shard_0000.feather").write_text("new bucket")

    with pytest.raises(rg.GateNotOpen, match="different input snapshot"):
        rg.require_operation(
            data_root, resolve_run_facts("weather_matching", ["--config", str(config)]))


def test_删除整个桶目录会让旧批准失效(data_root, tmp_path):
    """Dropping one bucket is not an unresolvable run, it is a different one."""
    import shutil

    config = build_weather_stage(tmp_path)
    before = resolve_run_facts("weather_matching", ["--config", str(config)])
    approved_for(data_root, before)

    shutil.rmtree(Path(json.loads(config.read_text())["source_project"])
                  / "ais_buckets" / "bucket_0000")
    after = resolve_run_facts("weather_matching", ["--config", str(config)])
    assert len(after.inputs) < len(before.inputs)
    with pytest.raises(rg.GateNotOpen, match="different input snapshot"):
        rg.require_operation(data_root, after)


def test_消费目录整个缺失或全空时拒绝产生事实(tmp_path):
    import shutil

    config = build_weather_stage(tmp_path)
    project = Path(json.loads(config.read_text())["source_project"])
    for bucket in project.glob("ais_buckets/bucket_*"):
        shutil.rmtree(bucket)
    with pytest.raises(RunFactsUnresolved, match="is empty"):
        resolve_run_facts("weather_matching", ["--config", str(config)])

    shutil.rmtree(project / "ais_buckets")
    with pytest.raises(RunFactsUnresolved, match="does not exist"):
        resolve_run_facts("weather_matching", ["--config", str(config)])


def test_替换消费数据后seal也拒(data_root, tmp_path, monkeypatch):
    config = build_weather_stage(tmp_path)
    before = resolve_run_facts("weather_matching", ["--config", str(config)])
    approved_for(data_root, before)
    clearance = rg.require_operation(data_root, before)
    monkeypatch.setenv(rg.CLEARANCE_ENV, clearance.env()[rg.CLEARANCE_ENV])

    project = Path(json.loads(config.read_text())["source_project"])
    (project / "companion" / "run" / "buckets" / "bucket_0000"
     / "companion_track.parquet").write_text("swapped after clearance")

    with pytest.raises(rg.GateNotOpen, match="different input snapshot"):
        rg.seal_check("_upstream_x.py", "weather_matching", data_root,
                      ["--config", str(config)])


def test_companion缺少上游必需参数时拒绝产生事实():
    with pytest.raises(RunFactsUnresolved, match="requires"):
        resolve_run_facts("companion_reconstruction", ["--config", "/nowhere/c.json"])


# --- the seal on the migrated upstream entries ------------------------------

def cleared_env(data_root: Path, tmp_path: Path):
    """A genuine clearance for a real, resolvable weather run."""
    config = build_weather_stage(tmp_path)
    run_facts = resolve_run_facts("weather_matching", ["--config", str(config)])
    write(data_root, "CP1", cp1_record(
        decision="approved_without_anchors", anchors_adopted=False,
        input_snapshot_sha256=run_facts.input_snapshot_sha256,
        approved_scope={"operations": ["weather_matching"], "year": run_facts.year,
                        "candidate_scope_sha256": run_facts.candidate_scope_sha256}))
    clearance = rg.require_operation(data_root, run_facts)
    return clearance, ["--config", str(config)]


def test_seal_refuses_without_clearance(data_root, monkeypatch):
    monkeypatch.delenv(rg.CLEARANCE_ENV, raising=False)
    with pytest.raises(rg.GateNotOpen, match="without clearance"):
        rg.seal_check("_upstream_x.py", "weather_matching", data_root, [])


def test_seal_accepts_a_genuine_clearance(data_root, tmp_path, monkeypatch):
    clearance, argv = cleared_env(data_root, tmp_path)
    monkeypatch.setenv(rg.CLEARANCE_ENV, clearance.env()[rg.CLEARANCE_ENV])
    rg.seal_check("_upstream_x.py", "weather_matching", data_root, argv)


def test_seal_refuses_clearance_for_another_operation(data_root, tmp_path, monkeypatch):
    clearance, argv = cleared_env(data_root, tmp_path)
    monkeypatch.setenv(rg.CLEARANCE_ENV, clearance.env()[rg.CLEARANCE_ENV])
    with pytest.raises(rg.GateNotOpen, match="clearance is for"):
        rg.seal_check("_upstream_x.py", "companion_reconstruction", data_root, argv)


def test_seal_refuses_when_the_decision_changed_after_clearance(data_root, tmp_path,
                                                                monkeypatch):
    clearance, argv = cleared_env(data_root, tmp_path)
    monkeypatch.setenv(rg.CLEARANCE_ENV, clearance.env()[rg.CLEARANCE_ENV])
    record = json.loads(rg.gate_path(data_root, "CP1").read_text())
    record["decision"] = "rework"
    record["anchors_adopted"] = False
    write(data_root, "CP1", record)
    with pytest.raises(rg.GateNotOpen, match="does not authorise"):
        rg.seal_check("_upstream_x.py", "weather_matching", data_root, argv)


def test_seal_refuses_a_hand_written_empty_clearance(data_root, tmp_path, monkeypatch):
    _, argv = cleared_env(data_root, tmp_path)
    monkeypatch.setenv(rg.CLEARANCE_ENV, json.dumps(
        {"operation": "weather_matching", "tokens": {}}))
    with pytest.raises(rg.GateNotOpen, match="no gate tokens"):
        rg.seal_check("_upstream_x.py", "weather_matching", data_root, argv)


def test_seal_refuses_a_forged_token(data_root, tmp_path, monkeypatch):
    _, argv = cleared_env(data_root, tmp_path)
    monkeypatch.setenv(rg.CLEARANCE_ENV, json.dumps(
        {"operation": "weather_matching", "tokens": {"CP1": "0" * 32}}))
    with pytest.raises(rg.GateNotOpen, match="does not match the decisions now on disk"):
        rg.seal_check("_upstream_x.py", "weather_matching", data_root, argv)


def test_返工的CP1配上真实令牌仍然过不了seal(data_root, tmp_path, monkeypatch):
    """The hole Supreme found: a token is derived from the decision file, so a refused
    decision can still produce a genuine-looking token. The seal must re-evaluate."""
    clearance, argv = cleared_env(data_root, tmp_path)
    record = json.loads(rg.gate_path(data_root, "CP1").read_text())
    record["decision"] = "rework"
    record["anchors_adopted"] = False
    write(data_root, "CP1", record)
    genuine_token = rg._decision_token(rg.gate_path(data_root, "CP1"))
    monkeypatch.setenv(rg.CLEARANCE_ENV, json.dumps(
        {"operation": "weather_matching",
         "facts": clearance.facts.as_dict(),
         "tokens": {"CP1": genuine_token}}))
    with pytest.raises(rg.GateNotOpen, match="does not authorise"):
        rg.seal_check("_upstream_x.py", "weather_matching", data_root, argv)


def test_采用锚点但缺CP2时seal也拒(data_root, tmp_path, monkeypatch):
    config = build_weather_stage(tmp_path)
    run_facts = resolve_run_facts("weather_matching", ["--config", str(config)])
    write(data_root, "CP1", cp1_record(
        input_snapshot_sha256=run_facts.input_snapshot_sha256,
        approved_scope={"operations": ["weather_matching"], "year": run_facts.year,
                        "candidate_scope_sha256": run_facts.candidate_scope_sha256}))
    token = rg._decision_token(rg.gate_path(data_root, "CP1"))
    monkeypatch.setenv(rg.CLEARANCE_ENV, json.dumps(
        {"operation": "weather_matching", "tokens": {"CP1": token}}))
    with pytest.raises(rg.GateNotOpen, match="CP2 is not open"):
        rg.seal_check("_upstream_x.py", "weather_matching", data_root,
                      ["--config", str(config)])


def test_seal_refuses_a_clearance_granted_for_another_config(data_root, tmp_path,
                                                             monkeypatch):
    clearance, _ = cleared_env(data_root, tmp_path / "approved")
    other = build_weather_stage(tmp_path / "other", voyages_body="different")
    monkeypatch.setenv(rg.CLEARANCE_ENV, clearance.env()[rg.CLEARANCE_ENV])
    with pytest.raises(rg.GateNotOpen,
                       match="different input snapshot|different candidate set|different run"):
        rg.seal_check("_upstream_x.py", "weather_matching", data_root,
                      ["--config", str(other)])


# --- the seal migration remains auditable as historical provenance ----------

def test_sealed_files_still_parse_and_seal_follows_future_imports():
    """A seal placed before `from __future__` is a SyntaxError.

    That would break the legitimate gated path, not just the bypass, so the placement
    rule is asserted rather than assumed.
    """
    import ast

    repo_pipeline = HERE.parents[1]
    patches = json.loads((repo_pipeline / "GATE_PATCHES.json").read_text())["patches"]
    active = [p for p in patches if not p.get("retired")]
    for record in active:
        path = repo_pipeline / record["sealed_file"]
        source = path.read_text()
        tree = ast.parse(source)  # fails loudly if the seal broke the file
        future_lines = [n.end_lineno for n in tree.body
                        if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
        seal_line = next(i for i, line in enumerate(source.splitlines(), start=1)
                         if "E001 review-gate seal" in line)
        assert all(seal_line > end for end in future_lines), (
            f"{path.name}: the seal must come after every __future__ import")


def test_gate_patches_match_the_migration_provenance():
    """Every sealed file must still be traceable to the pinned source commit."""
    repo_pipeline = HERE.parents[1]
    patches = json.loads((repo_pipeline / "GATE_PATCHES.json").read_text())["patches"]
    provenance = json.loads((repo_pipeline / "CODE_PROVENANCE.json").read_text())["entries"]
    by_name = {Path(e["target_path"]).name: e for e in provenance}

    assert patches, "no seal is recorded"
    for patch in patches:
        sealed = repo_pipeline / patch["sealed_file"]
        if patch.get("retired"):
            assert not sealed.exists()
            assert not (repo_pipeline / patch["gated_entry"]).exists()
            assert patch.get("retirement_reason")
            continue
        assert sealed.is_file()
        assert rg.hashlib.sha256(sealed.read_bytes()).hexdigest() == patch["sealed_sha256"]
        migrated_name = Path(patch["gated_entry"]).name
        assert by_name[migrated_name]["sha256"] == patch["original_sha256"], (
            f"{patch['sealed_file']}: recorded original hash does not match the "
            f"migration provenance for {migrated_name}")
        assert by_name[migrated_name]["bytes"] == patch["original_bytes"]
