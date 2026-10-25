#!/usr/bin/env python3
"""Record T014's source lineage and full-run closure before independent review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CODING = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hash_.update(block)
    return hash_.hexdigest()


def build(out: Path, run: Path, tolerant_review: Path, gfw_run: Path,
          internal: Path) -> dict:
    full = json.loads((run / "30_census_full/full_run_status.json").read_text())
    census = json.loads((tolerant_review / "06_closure.json").read_text())
    clip = json.loads((out / "08_wave_clipping_summary.json").read_text())
    gfw = json.loads((out / "04_gfw_exposure_summary.json").read_text())
    port = json.loads((out / "03_port_tolerance_calibration.json").read_text())
    resources = json.loads((out / "09_resource_preflight.json").read_text())
    if full["shards_expected"] != 253 or full["failures"]:
        raise ValueError("T014 census incomplete or failed")
    if full["port_space_tolerance_km"] != 0 or full["port_time_tolerance_h"] != 1:
        raise ValueError("selected tolerance differs from approved conditional rule")
    if full["unbounded_conflict_voyages"] != 5:
        raise ValueError("port conflict exclusion count changed")
    if census["voyages"] != 1_354_168 or census["release_combinations"] != 46_981:
        raise ValueError("T014 census/release grain changed")
    if clip["wave_point_outside_possible_range"] != 0:
        raise ValueError("clipped wave point outside possible range")
    if gfw["hours"] != 2_293_663:
        raise ValueError("GFW exposure population changed")
    if port["status"] != "threshold_unmet" or port["conditional_selected_minimum_effective_radius"] != {"space_km": 0, "time_h": 1}:
        raise ValueError("conditional calibration status changed")
    if resources["estimated_full_wall_hours"] >= 24:
        raise ValueError("resource probe exceeded Contract stop threshold")
    scripts = [
        HERE.parent / "20_quantify_uncertainty/10_missing_positions/50_calibrate_port_tolerance.py",
        HERE.parent / "20_quantify_uncertainty/10_missing_positions/60_compute_tolerant_bounds.py",
        HERE.parent / "20_quantify_uncertainty/20_interpolation_holdout/50_run_gfw_crosscheck.py",
        HERE.parent / "20_quantify_uncertainty/20_interpolation_holdout/60_compare_gfw_exposure.py",
        HERE / "70_clip_full_voyage_wave.py",
        HERE / "80_build_tolerant_release.py",
        HERE / "90_assemble_t014_release.py",
        HERE / "92_write_t014_closure.py",
        HERE / "95_verify_t014_release.py",
        HERE / "100_document_t014_review.py",
    ]
    report = {"status": "closed_internal_draft_pending_supreme_review",
              "branch": "experiment/E002-2025-od-exposure-release",
              "base_head": "fb6bab423d3fdb96463cdcf7de65b48a5820615a",
              "t013_source_commit": "7d31437",
              "selected_main_rule": {"space_km": 0, "time_h": 1,
                                     "calibration_denominator_hours": 920,
                                     "reachable_hours": 911,
                                     "five_conflict_voyages_unbounded": True,
                                     "original_969_hour_target_met": False},
              "full_census": full, "census_closure": census,
              "p1_wave_clipping": clip, "p2_calibration": port,
              "p3_gfw_exposure": {"hours": gfw["hours"],
                                  "valid_wave_hours": gfw["swh_absolute_difference_m"]["valid_hours"],
                                  "tc300_flip_hours": gfw["tc300"]["flip_hours"],
                                  "tc500_flip_hours": gfw["tc500"]["flip_hours"]},
              "resource_preflight": resources,
              "sources_sha256": {
                  "t013_release": digest(CODING / "outputs/2026-09-24_2048_E002-T013-full-voyage-share-review/01_release_table_voyages_ge5.csv"),
                  "t012_zero_voyages": digest(CODING / ".runs/2026-09-24_1500_E002-T012-exposure-units/50_voyage_exposure_intervals_internal.parquet"),
                  "port_endpoints": digest(CODING / ".runs/2026-09-24_1500_E002-T012-exposure-units/10_c1/port_endpoints_by_candidate.parquet"),
                  "port_tolerance_hour_audit": digest(run.parent / "port_tolerance_hour_audit.parquet"),
                  "gfw_local_raw_summary": digest(CODING / "data/legacy_inputs/gfw_presence_responses_2025/summary.json"),
                  "gfw_climate_derived_manifest": digest(gfw_run / "climate_derived_source_manifest.json"),
                  "gfw_anchor_exclusion_manifest": digest(CODING / ".runs/2026-09-13_1930_E001-2025-merged-anchor-reconstruction/cache_rematch/rematch_manifest.json"),
                  "t014_full_run_status": digest(run / "30_census_full/full_run_status.json"),
                  "t014_tolerant_voyages": digest(run / "50_voyage_exposure_intervals_internal.parquet"),
                  "t014_wave_bootstrap": digest(internal / "wave_clipping_by_combination.parquet"),
                  "release": digest(out / "01_release_table_voyages_ge5.csv")},
              "scripts_sha256": {str(path.relative_to(CODING)): digest(path) for path in scripts}}
    (out / "06_closure.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {"status": report["status"], "census_shards": full["shards_expected"],
            "release_combinations": census["release_combinations"],
            "gfw_hours": gfw["hours"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tolerant-review", type=Path, required=True)
    parser.add_argument("--gfw-run", type=Path, required=True)
    parser.add_argument("--internal", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.run, args.tolerant_review,
                           args.gfw_run, args.internal), indent=2))
