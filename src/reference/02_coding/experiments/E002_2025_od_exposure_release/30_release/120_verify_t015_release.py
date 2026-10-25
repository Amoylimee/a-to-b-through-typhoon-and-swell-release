#!/usr/bin/env python3
"""Close T015 against private full precision and verify display-only rounding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
T014 = CODING / "outputs/2026-09-24_2218_E002-T014-reader-first-review"
KEY = ("origin_city", "dest_city", "shiptype_group")
EXEMPT = set(KEY) | {"voyages", "ships", "voyages_with_tc300", "voyages_with_tc500",
                     "wave_mean_missing_reason", "single_ship",
                     "sampling_interval_unavailable_fewer_than_3_ships"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def builder():
    spec = importlib.util.spec_from_file_location("e002_t015_build", HERE / "110_polish_t015_release.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("T015 builder unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(out: Path, private: Path) -> dict:
    build = builder()
    full_path = private / "01_release_full_precision.csv"
    rounded_path = out / "01_release_table_voyages_ge5.csv"
    source_path = T014 / "01_release_table_voyages_ge5.csv"
    if digest(full_path) != digest(source_path):
        raise ValueError("private full precision differs from frozen T014 release")
    old_closure = json.loads((T014 / "07_independent_closure.json").read_text())
    if old_closure["status"] != "pass" or old_closure["sources"]["release_sha256"] != digest(full_path):
        raise ValueError("private copy is not the previously closed full-precision release")
    unchanged = {name: digest(T014 / name) == digest(out / name) for name in (
        "02_below_threshold_summary.csv", "03_port_tolerance_calibration.json",
        "03_port_tolerance_reachability.csv", "03b_port_tolerance_without_conflict_voyages.csv",
        "04_gfw_exposure_by_month.csv", "04_gfw_exposure_summary.json",
        "05_voyage_rule_sensitivity.csv", "05a_wave_population_guarantee.csv",
        "08_wave_clipping_summary.json", "09_resource_preflight.json")}
    if not all(unchanged.values()):
        raise ValueError("T014 evidence changed during display-only T015")
    rows = 0
    example = None
    with full_path.open(newline="") as full_stream, rounded_path.open(newline="") as rounded_stream:
        full = csv.DictReader(full_stream)
        rounded = csv.DictReader(rounded_stream)
        if full.fieldnames != rounded.fieldnames or full.fieldnames is None or len(full.fieldnames) != 43:
            raise ValueError("release column order/schema changed")
        unknown = set(full.fieldnames) - EXEMPT - {name for name in full.fieldnames if build.precision(name) is not None}
        if unknown:
            raise ValueError(f"fields lack an explicit rounding rule: {sorted(unknown)}")
        for original, published in itertools.zip_longest(full, rounded):
            rows += 1
            if original is None or published is None:
                raise ValueError("full-precision and rounded release row counts differ")
            if tuple(original[name] for name in KEY) != tuple(published[name] for name in KEY):
                raise ValueError(f"release group key changed at row {rows}")
            for field in full.fieldnames:
                places = build.precision(field)
                expected = build.fixed(original[field], places) if places is not None else original[field]
                if published[field] != expected:
                    raise ValueError(f"release changed beyond display rounding: row {rows}, {field}")
            if tuple(original[name] for name in KEY) == build.EXAMPLE_KEY:
                example = published
    if rows != 46_981 or example is None:
        raise ValueError("release row count or real example changed")
    full_frame = pd.read_csv(full_path)
    below = pd.read_csv(out / "02_below_threshold_summary.csv").iloc[0]
    if int(full_frame.voyages.sum()) != 1_034_602 or int(full_frame.voyages.sum() + below.voyages) != 1_354_168:
        raise ValueError("full-precision release population fails to close")
    if full_frame.mean_swh_full_voyage_m.lt(full_frame.mean_swh_min_possible_m - 1e-9).any() or full_frame.mean_swh_full_voyage_m.gt(full_frame.mean_swh_max_possible_m + 1e-9).any():
        raise ValueError("full-precision wave point falls outside its existing bounds")
    if full_frame.loc[full_frame.ships.lt(3), [name for name in full_frame if "ci95" in name]].notna().any().any():
        raise ValueError("sampling interval exists on fewer than three vessels")
    sensitivity = pd.read_csv(out / "05_voyage_rule_sensitivity.csv")
    for radius in (300, 500):
        stem = f"tc{radius}"
        lower = full_frame[f"{stem}_at_least_pct"]
        upper = full_frame[f"{stem}_at_most_pct"]
        result = sensitivity.loc[sensitivity.population.eq("all")
                                 & sensitivity.port_rule.eq("conditional_0km_1h")
                                 & sensitivity.metric.eq(f"tc{radius}_width_pp")]
        if lower.gt(upper + 1e-9).any() or len(result) != 1 or float(result.p90.iloc[0]) != 0.0:
            raise ValueError(f"full-precision TC{radius} headline is unsupported")
        if not np.allclose(full_frame[f"{stem}_hours_at_most"],
                           full_frame[f"{stem}_hours_at_least"]
                           + full_frame[f"{stem}_bounded_possible_hours"] + full_frame.unbounded_hours,
                           rtol=0, atol=1e-6):
            raise ValueError(f"existing TC{radius} hour decomposition fails")
        if not np.allclose(lower, 100 * full_frame[f"{stem}_hours_at_least"] / full_frame.voyage_hours,
                           rtol=0, atol=1e-8) or not np.allclose(
                               upper, 100 * full_frame[f"{stem}_hours_at_most"] / full_frame.voyage_hours,
                               rtol=0, atol=1e-8):
            raise ValueError(f"existing TC{radius} percentage definition fails")
    wave_guarantee = pd.read_csv(out / "05a_wave_population_guarantee.csv")
    wave = wave_guarantee.loc[wave_guarantee.population.eq("all")
                              & wave_guarantee.port_rule.eq("conditional_0km_1h")]
    if len(wave) != 1 or not np.isclose(wave.p90.iloc[0], .3804803156479746, atol=1e-12):
        raise ValueError("full-population wave guarantee changed")
    port = pd.read_csv(out / "03_port_tolerance_reachability.csv")
    conditional = pd.read_csv(out / "03b_port_tolerance_without_conflict_voyages.csv")
    if len(port) != 12 or len(conditional) != 12 or not port.gfw_hours.eq(969).all() or not conditional.gfw_hours.eq(920).all():
        raise ValueError("port tolerance populations changed")
    if int(conditional.loc[conditional.space_km.eq(0) & conditional.time_h.eq(1), "reachable_hours"].iloc[0]) != 911:
        raise ValueError("conditional port rule changed")
    port_audit = pd.read_parquet(
        CODING / ".runs/2026-09-24_2218_E002-T014-reader-first/port_tolerance_hour_audit.parquet",
        columns=["candidate_id", "reachable_50km_2h"])
    voyage_match = port_audit.groupby("candidate_id").reachable_50km_2h.all()
    if len(voyage_match) != 117 or int(voyage_match.sum()) != 112:
        raise ValueError("117/112 port-position cross-check changed")
    t014_closure = json.loads((T014 / "06_closure.json").read_text())
    old_no_position = t014_closure["census_closure"]["old_unbounded_voyages"]
    affected_pct = 100 * old_no_position * int((~voyage_match).sum()) / len(voyage_match) / 1_354_168
    if old_no_position != 64_660 or not np.isclose(affected_pct, .20405502590873978, atol=1e-12):
        raise ValueError("approximate port-conflict extrapolation changed")
    holdout = pd.read_csv(out / "04_u3_holdout_precision.csv")
    if len(holdout) != 2 or holdout.gap_hours.tolist() != [4, 24]:
        raise ValueError("T015 holdout summary shape changed")
    if holdout.complete_windows.tolist() != [4601, 4402] or holdout.position_points.tolist() != [95780, 542664]:
        raise ValueError("T015 holdout denominators changed")
    if not np.allclose(holdout.position_error_p90_km, [9.85337241234453, 68.71808872556], atol=1e-9):
        raise ValueError("T015 position-error summary changed")
    if not np.allclose(holdout.wave_mean_absolute_error_p90_m, [.03562148905471161, .19119062316530255], atol=1e-12):
        raise ValueError("T015 wave-error summary changed")
    gfw = json.loads((out / "04_gfw_exposure_summary.json").read_text())
    if gfw["hours"] != 2_293_663 or not 7 < gfw["distance_km"]["p90"] < 7.2:
        raise ValueError("GFW comparison changed")
    for markdown in ("00_README.md", "03_FIELD_GUIDE.md"):
        text = (out / markdown).read_text()
        first = text.split("## ", 1)[0]
        if not text.startswith("# 数据可靠性一览 / Data reliability at a glance"):
            raise ValueError(f"{markdown} does not start with reader overview")
        if first.count("| 缺失位置 / Missing positions |") != 1 or first.count("| 抽样 / Sampling |") != 1:
            raise ValueError(f"{markdown} reliability overview incomplete")
        if "0.000" in first or "99%门槛" in first or "99% 门槛" in first:
            raise ValueError(f"{markdown} retains prohibited overview wording")
        overview_rows = [line for line in first.splitlines() if line.startswith("| ")
                         and " / " in line and not line.startswith("| 项目")]
        if len(overview_rows) != 4 or any(line.split("|")[2].count("。") > 2 for line in overview_rows):
            raise ValueError(f"{markdown} reliability rows exceed the two-sentence limit")
        if "同为 AIS 来源" not in text or "紧邻已有锚点" not in text or "没有 2025 年卫星数据" not in text:
            raise ValueError(f"{markdown} lacks GFW/SAR limitations")
    guide = (out / "03_FIELD_GUIDE.md").read_text()
    if sum(line.startswith("| `") for line in guide.splitlines()) != 43:
        raise ValueError("bilingual field guide lacks a release column")
    for field in full_frame.columns:
        if f"| `{field}` |" not in guide:
            raise ValueError(f"bilingual field guide lacks {field}")
    for field in ("tc500_at_least_pct", "tc500_at_most_pct", "mean_swh_full_voyage_m"):
        if f"| `{field}` |" not in (out / "00_README.md").read_text() or example[field] not in guide:
            raise ValueError("rounded example is inconsistent across files")
    forbidden = {"candidate_id", "mmsi", "vessel_uid_clean", "lat", "lon", "latitude", "longitude"}
    for csv_path in out.glob("*.csv"):
        with csv_path.open(newline="") as stream:
            header = next(csv.reader(stream))
        if forbidden.intersection(header):
            raise ValueError(f"restricted column in {csv_path.name}")
    shutil.copy2(T014 / "07_independent_closure.json", out / "07_independent_closure.json")
    report = {
        "status": "pass_display_only_pending_supreme_review",
        "branch": "experiment/E002-2025-od-exposure-release",
        "source_head": "079d2e3",
        "t014_source_commit": "8033ff7",
        "release_combinations": rows,
        "release_voyages": int(full_frame.voyages.sum()),
        "all_voyages_with_omitted": int(full_frame.voyages.sum() + below.voyages),
        "full_precision_source_closure": old_closure["status"],
        "full_precision_old_rule_voyages_closed": old_closure["old_rule_voyages_independently_closed"],
        "port_position_crosscheck_voyages": len(voyage_match),
        "port_position_crosscheck_matched_voyages": int(voyage_match.sum()),
        "port_potentially_affected_pct_rough_extrapolation": affected_pct,
        "port_extrapolation_basis": "(5 / 117) * 64660 / 1354168 * 100; not an individually verified error rate",
        "t014_evidence_files_unchanged": unchanged,
        "rounding": {"percentage_decimals": 3, "metre_decimals": 3,
                     "hour_decimals": 1, "coverage_and_ratio_decimals": 3,
                     "all_46981_rows_checked_field_by_field": True},
        "headline_full_precision": {"tc300_width_p90_pp": 0.0, "tc500_width_p90_pp": 0.0,
                                    "wave_width_p90_m": float(wave.p90.iloc[0])},
        "source_sha256": {"t014_release": digest(source_path),
                          "private_full_precision": digest(full_path),
                          "rounded_release": digest(rounded_path),
                          "t014_independent_closure": digest(T014 / "07_independent_closure.json"),
                          "t015_builder": digest(HERE / "110_polish_t015_release.py"),
                          "t015_verifier": digest(Path(__file__))},
    }
    (out / "06_closure.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.out, args.private)
    print(json.dumps({name: result[name] for name in (
        "status", "release_combinations", "release_voyages", "full_precision_source_closure",
    )}, indent=2))
