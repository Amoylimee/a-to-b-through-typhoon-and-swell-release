from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "10_assemble_candidate_root.py"
SPEC = importlib.util.spec_from_file_location("e001_candidate_root", MODULE)
assert SPEC and SPEC.loader
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(root: Path, *, formal: bool = True, duplicate: bool = False) -> Path:
    source = root / "source"
    source.mkdir()
    parts = []
    for part_id in range(16):
        candidate_id = "duplicate" if duplicate and part_id in (0, 1) else f"c-{part_id}"
        frame = pd.DataFrame({
            "candidate_id": [candidate_id], "vessel_uid_clean": [f"api-{part_id}"],
            "mmsi": [f"m-{part_id}"], "imo": [f"i-{part_id}"],
            "origin_city": ["AAA:A"], "dest_city": ["BBB:B"], "pair_id": ["AAA:A__BBB:B"],
            "departure_time": pd.to_datetime(["2025-01-01T00:00:00Z"]),
            "arrival_time": pd.to_datetime(["2025-01-02T00:00:00Z"]),
            "shiptype_group": ["bulk"], "retain_after_c3_final": [True],
            "normalized_time_ratio": [0.01],
        })
        if formal:
            frame = frame.assign(
                retain_after_global_qtrim=True, global_qtrimmed=False,
                global_normalized_time_ratio_q01=0.001,
                global_normalized_time_ratio_q99=0.1,
                BaselineSuezDependence_g=0.0, suez_dependence_bin="none",
            )
        path = source / f"part-{part_id:03d}.feather"
        frame.to_feather(path)
        parts.append({"part_id": part_id, "path": str(path), "bytes": path.stat().st_size, "sha256": sha(path), "rows": 1})
    version = root / "input_version_manifest.json"
    version.write_text("{}\n", encoding="utf-8")
    pool_manifest = root / "verified_pool_manifest.json"
    pool_manifest.write_text(json.dumps({
        "producer_stage": "30c_build_adjacent_commercial_episode_voyages",
        "run_mode": "formal_annual",
        "analysis_year": 2025,
        "candidate_enumeration_mode": "adjacent_commercial_episodes",
        "annual_pool_complete": True,
        "completed_layers": ["L1", "L2", "L3"],
    }) + "\n", encoding="utf-8")
    verification = root / "input_verification.json"
    verification.write_text(json.dumps({
        "status": "pass",
        "scope": "complete_2025_L1_L2_L3_pretrim",
        "checks": {"parts": True, "identity": True},
    }) + "\n", encoding="utf-8")
    payload = {
        "status": "pass", "analysis_year": 2025,
        "candidate_enumeration_mode": "adjacent_commercial_episodes",
        "parts": parts, "summary": {"final_retained" if formal else "pretrim_rows": 16},
        "qtrim": {
            "status": "pass" if formal else "not_run",
            "fitted_on_complete_annual_pool": formal,
            "threshold_scope": "2025_pooled_no_grouping" if formal else None,
            "q_lower": 0.01 if formal else None,
            "q_upper": 0.99 if formal else None,
            "equality_retained": True if formal else None,
        },
    }
    if formal:
        payload.update(stage="30d_refit_and_apply_global_qtrim", annual_pool_complete=True,
                       run_mode="formal_annual",
                       completed_layers=["L1", "L2", "L3"],
                       upstream_input_version={"path": str(version), "sha256": sha(version)},
                       input_manifest={"path": str(pool_manifest), "sha256": sha(pool_manifest)},
                       input_verification={"path": str(verification), "sha256": sha(verification)})
        for part in parts:
            part["upstream_input_version_sha256"] = sha(version)
    else:
        payload.update(assembly_purpose="pretrim_structure_drill", annual_pool_complete=False,
                       completed_layers=["L1", "L2"])
    manifest = root / "source_manifest.json"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return manifest


class CandidateRootTests(unittest.TestCase):
    def test_formal_manifest_assembles_hash_bound_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(root)
            result = assembly.assemble_candidate_root(root / "candidate", manifest, workers=2)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["downstream_ready"])
            self.assertEqual(result["years"]["2025"]["part_count"], 16)
            self.assertEqual(result["years"]["2025"]["next_year_arrival_rows"], 0)
            self.assertFalse(result["years"]["2025"]["has_next_year_arrival"])
            self.assertTrue((root / "candidate/2025/part-000.feather").is_symlink())

    def test_formal_rejects_untrimmed_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(root, formal=False)
            with self.assertRaisesRegex(ValueError, "stage=30d"):
                assembly.assemble_candidate_root(root / "candidate", manifest)

    def test_duplicate_candidate_id_is_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(root, duplicate=True)
            with self.assertRaisesRegex(ValueError, "Duplicate candidate_id"):
                assembly.assemble_candidate_root(root / "candidate", manifest, workers=2)

    def test_missing_source_part_is_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(root)
            (root / "source/part-007.feather").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "source part is absent"):
                assembly.assemble_candidate_root(root / "candidate", manifest)

    def test_mixed_upstream_input_versions_are_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(root)
            payload = json.loads(manifest.read_text())
            payload["parts"][7]["upstream_input_version_sha256"] = "f" * 64
            manifest.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "mix or omit upstream input versions"):
                assembly.assemble_candidate_root(root / "candidate", manifest)

    def test_upstream_version_hash_mismatch_is_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(root)
            payload = json.loads(manifest.read_text())
            payload["upstream_input_version"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "input version"):
                assembly.assemble_candidate_root(root / "candidate", manifest)

    def test_drill_is_visibly_not_downstream_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(root, formal=False)
            result = assembly.assemble_candidate_root(root / "candidate", manifest, drill=True, workers=2)
            self.assertEqual(result["status"], "drill_only")
            self.assertFalse(result["downstream_ready"])
            self.assertEqual(result["years"]["2025"]["qtrim_status"], "not_run")


if __name__ == "__main__":
    unittest.main()
