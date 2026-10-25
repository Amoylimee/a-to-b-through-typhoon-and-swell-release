from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "30_assemble_candidate_root.py"
SPEC = importlib.util.spec_from_file_location("candidate_root_assembly", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)


def make_source(root: Path, year: int, next_year_arrival: bool) -> None:
    source = root / f"adjacent_commercial_episode_voyages_{year}"
    source.mkdir(parents=True)
    arrivals = [f"{year}-12-31T23:00:00Z", f"{year + 1 if next_year_arrival else year}-01-01T01:00:00Z"]
    frame = pd.DataFrame(
        {
            "candidate_id": [f"{year}-a", f"{year}-b"],
            "arrival_time": pd.to_datetime(arrivals, utc=True),
        }
    )
    frame.to_feather(source / "part-000.feather")
    (root / "manifests").mkdir(exist_ok=True)
    (root / "manifests" / f"final_sample_manifest_{year}.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "analysis_year": year,
                "candidate_enumeration_mode": "adjacent_commercial_episodes",
                "summary": {"final_retained": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )


class CandidateRootAssemblyTests(unittest.TestCase):
    def test_assembles_explicit_calendar_and_cross_year_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calendar = root / "calendar"
            cross_year = root / "cross_year"
            make_source(calendar, 2025, False)
            for year in (2021, 2022, 2023, 2024):
                make_source(cross_year, year, True)
            candidate_root = root / "candidate_root"

            result = assembly.assemble_candidate_root(
                candidate_root, calendar, cross_year
            )

            self.assertEqual(result["years"]["2022"]["cohort"], "cross_year")
            self.assertTrue(result["years"]["2022"]["has_next_year_arrival"])
            self.assertEqual(result["years"]["2021"]["cohort"], "cross_year")
            self.assertTrue((candidate_root / "2022" / "part-000.feather").is_symlink())
            self.assertTrue(
                (candidate_root / "2022" / "final_sample_manifest_2022.json").is_symlink()
            )
            self.assertTrue((candidate_root / "assembly_manifest.json").is_file())

    def test_rejects_non_adjacent_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calendar = root / "calendar"
            cross_year = root / "cross_year"
            make_source(calendar, 2025, False)
            for year in (2021, 2022, 2023, 2024):
                make_source(cross_year, year, True)
            manifest = cross_year / "manifests/final_sample_manifest_2021.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["candidate_enumeration_mode"] = "fanout_first_passage"
            manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mode disagrees"):
                assembly.assemble_candidate_root(
                    root / "candidate_root", calendar, cross_year
                )
