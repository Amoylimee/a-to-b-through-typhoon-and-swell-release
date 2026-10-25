#!/usr/bin/env python3
"""Apply the PI-approved final C3.1 route-circuity decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CODING_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = PIPELINE_ROOT / "config/frozen/route_decision_2021.json"
DEFAULT_ARTIFACT_DIR = DATA_ROOT / "main/provenance/route_decision_2021"
DEFAULT_REPORT_DIR = DATA_ROOT / "main/validation/route_decision_2021"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_feather_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".feather",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.reset_index(drop=True).to_feather(temporary)
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def apply_final_decision(
    frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    output = frame[
        [
            "candidate_id",
            "pair_id",
            "route_sequence_d_base_km",
            "direct_d_base_km",
            "route_circuity_ratio",
            "route_circuity_status",
        ]
    ].copy()
    exact = output["route_circuity_status"].eq("complete_positive")
    lower_bound = output["route_circuity_status"].eq(
        "same_network_node_transition"
    )
    unresolved = output["route_circuity_status"].eq(
        "direct_same_network_node"
    )
    if not (exact | lower_bound | unresolved).all():
        unknown = output.loc[
            ~(exact | lower_bound | unresolved),
            "route_circuity_status",
        ].unique()
        raise AssertionError(f"Unexpected route status: {unknown}")

    output["route_circuity_ratio_final"] = output["route_circuity_ratio"]
    output.loc[lower_bound, "route_circuity_ratio_final"] = (
        output.loc[lower_bound, "route_sequence_d_base_km"]
        / output.loc[lower_bound, "direct_d_base_km"]
    )
    output["route_circuity_decision_basis"] = np.select(
        [exact, lower_bound, unresolved],
        [
            "exact_positive_distance",
            "conservative_same_node_lower_bound",
            "unresolved_direct_same_node",
        ],
    )
    output["exclude_route_circuity_gt_2_final"] = (
        ~unresolved & output["route_circuity_ratio_final"].gt(threshold)
    )
    output["retain_after_route_circuity"] = ~output[
        "exclude_route_circuity_gt_2_final"
    ]
    if output.loc[unresolved, "route_circuity_ratio_final"].notna().any():
        raise AssertionError("Direct same-node candidates received a ratio")
    return output


def run(
    config_path: Path,
    artifact_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    if int(config["analysis_year"]) != 2021:
        raise AssertionError("Final route-circuity decision is 2021 only")
    if float(config["main_threshold"]) != 2.0:
        raise AssertionError("The approved main threshold must remain 2.0")
    manifest_path = CODING_ROOT / config["route_circuity_output_manifest"]
    if sha256sum(manifest_path) != config[
        "route_circuity_output_manifest_sha256"
    ]:
        raise AssertionError("The frozen route-circuity output manifest changed")

    source_parts = sorted(
        (CODING_ROOT / config["route_circuity_part_folder"]).glob(
            "part-*.feather"
        )
    )
    if len(source_parts) != int(config["expected_candidate_parts"]):
        raise AssertionError("The route-circuity part count changed")

    totals = {
        "candidate_rows": 0,
        "excluded": 0,
        "retained": 0,
        "direct_same_node_unresolved": 0,
    }
    sensitivity_counts = {
        str(value): 0 for value in config["sensitivity_thresholds"]
    }
    outputs = []
    for source_path in source_parts:
        frame = pd.read_feather(source_path)
        output = apply_final_decision(
            frame,
            float(config["main_threshold"]),
        )
        target_path = artifact_dir / "parts" / source_path.name
        write_feather_atomic(target_path, output)
        outputs.append(target_path)
        totals["candidate_rows"] += len(output)
        totals["excluded"] += int(
            output["exclude_route_circuity_gt_2_final"].sum()
        )
        totals["retained"] += int(
            output["retain_after_route_circuity"].sum()
        )
        totals["direct_same_node_unresolved"] += int(
            output["route_circuity_decision_basis"].eq(
                "unresolved_direct_same_node"
            ).sum()
        )
        for value in config["sensitivity_thresholds"]:
            sensitivity_counts[str(value)] += int(
                output["route_circuity_ratio_final"].gt(float(value)).sum()
            )

    expected = {
        "candidate_rows": int(config["expected_candidate_rows"]),
        "excluded": int(config["expected_main_excluded"]),
        "retained": int(config["expected_main_retained"]),
        "direct_same_node_unresolved": int(
            config["expected_direct_same_node_unresolved"]
        ),
    }
    if totals != expected:
        raise AssertionError(
            f"Final route-circuity totals changed: {totals} != {expected}"
        )
    summary = {
        "status": "pass",
        "analysis_year": 2021,
        "main_threshold": 2.0,
        "candidate_rows": totals["candidate_rows"],
        "excluded_candidates": totals["excluded"],
        "excluded_share": totals["excluded"] / totals["candidate_rows"],
        "retained_candidates": totals["retained"],
        "retained_share": totals["retained"] / totals["candidate_rows"],
        "direct_same_node_unresolved_retained": totals[
            "direct_same_node_unresolved"
        ],
        "sensitivity_excluded_counts": sensitivity_counts,
        "same_node_transition_rule": config["same_node_transition_rule"],
        "direct_same_node_rule": config["direct_same_node_rule"],
    }
    summary_path = report_dir / "c3_1_route_circuity_final_decision_summary.json"
    audit_path = report_dir / "c3_1_route_circuity_final_decision.md"
    write_json_atomic(summary_path, summary)
    lines = [
        "# C3.1 Final Route-Circuity Decision",
        "",
        "## PI Decision",
        "",
        "- Main global threshold: strictly above 2.0.",
        "- Positive-endpoint candidates with same-node intermediate transitions",
        "  use the conservative zero-segment lower-bound ratio.",
        "- Direct same-node endpoint candidates remain retained and unresolved.",
        "",
        "## 2021 Result",
        "",
        f"- Input C2-retained candidates: {totals['candidate_rows']:,}",
        f"- Excluded: {totals['excluded']:,} ({summary['excluded_share']:.3%})",
        f"- Retained: {totals['retained']:,} ({summary['retained_share']:.3%})",
        (
            f"- Retained direct same-node unresolved: "
            f"{totals['direct_same_node_unresolved']:,}"
        ),
        "",
        "These are 2021 OD candidate counts. The threshold was informed by the",
        "2021 AIS pilot; the remaining full AIS alignment has not yet run.",
    ]
    write_text_atomic(audit_path, "\n".join(lines) + "\n")
    manifest = {
        "status": "pass",
        "outputs": {
            str(path.relative_to(CODING_ROOT)): {
                "byte_size": path.stat().st_size,
                "sha256": sha256sum(path),
            }
            for path in [*outputs, summary_path, audit_path]
        },
    }
    write_json_atomic(artifact_dir / "output_manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the approved final 2021 route-circuity decision."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    summary = run(
        config_path=(
            arguments.config
            if arguments.config.is_absolute()
            else CODING_ROOT / arguments.config
        ),
        artifact_dir=(
            arguments.artifact_dir
            if arguments.artifact_dir.is_absolute()
            else CODING_ROOT / arguments.artifact_dir
        ),
        report_dir=(
            arguments.report_dir
            if arguments.report_dir.is_absolute()
            else CODING_ROOT / arguments.report_dir
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
