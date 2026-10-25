#!/usr/bin/env python3
"""Apply the frozen final 2021 C3 candidate-quality specification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .extreme_duration import (
    CODING_ROOT,
    C3_ROOT,
    load_json,
    sha256_file,
    write_feather_atomic,
    write_json_atomic,
    write_text_atomic,
)


DEFAULT_CONFIG = C3_ROOT / "config/c3_final_specification_2021.json"
DEFAULT_ARTIFACT_DIR = (
    C3_ROOT / "artifacts/c3_5_final_specification_2021"
)
DEFAULT_REPORT_DIR = C3_ROOT / "reports"
DEFAULT_RUN_DIR = (
    C3_ROOT / "runs/c3_2021/c3_5_final_specification_2021"
)


def apply_same_node_route_backstop(
    pre_backstop_decisions: pd.DataFrame,
    route_decisions: pd.DataFrame,
    *,
    threshold_km: float,
) -> pd.DataFrame:
    if threshold_km <= 0:
        raise ValueError("Same-node route backstop must be positive")
    if pre_backstop_decisions["candidate_id"].duplicated().any():
        raise ValueError("Pre-backstop decisions contain duplicate IDs")
    if route_decisions["candidate_id"].duplicated().any():
        raise ValueError("Route decisions contain duplicate IDs")
    route_columns = [
        "candidate_id",
        "route_sequence_d_base_km",
        "direct_d_base_km",
        "route_circuity_decision_basis",
        "retain_after_route_circuity",
    ]
    route = route_decisions[route_columns].rename(
        columns={
            "route_sequence_d_base_km": (
                "route_sequence_d_base_km_route_decision"
            ),
            "route_circuity_decision_basis": (
                "route_circuity_decision_basis_route_decision"
            ),
            "retain_after_route_circuity": (
                "retain_after_route_circuity_route_decision"
            ),
        }
    )
    output = pre_backstop_decisions.merge(
        route,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if output["direct_d_base_km"].isna().any():
        raise ValueError("At least one candidate lacks a direct distance")
    if output["route_sequence_d_base_km_route_decision"].isna().any():
        raise ValueError("At least one candidate lacks a route distance")
    if not np.allclose(
        output["route_sequence_d_base_km"].to_numpy(float),
        output["route_sequence_d_base_km_route_decision"].to_numpy(float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=False,
    ):
        raise ValueError("Route-sequence distance changed between decisions")
    if not output["route_circuity_decision_basis"].eq(
        output["route_circuity_decision_basis_route_decision"]
    ).all():
        raise ValueError("Route decision basis changed between decisions")
    if not output["retain_after_route_circuity"].eq(
        output["retain_after_route_circuity_route_decision"]
    ).all():
        raise ValueError("Route-retained status changed between decisions")
    if not output["retain_after_route_circuity"].all():
        raise ValueError("Pre-backstop input must contain route-retained rows")

    output = output.drop(
        columns=[
            "route_sequence_d_base_km_route_decision",
            "route_circuity_decision_basis_route_decision",
            "retain_after_route_circuity_route_decision",
        ]
    )
    output = output.rename(
        columns={
            "retain_after_c3_final": (
                "retain_after_c3_before_same_node_backstop"
            )
        }
    )
    direct_same_node = output["direct_d_base_km"].eq(0.0)
    over_threshold = output["route_sequence_d_base_km"].gt(
        float(threshold_km)
    )
    prior_retained = output[
        "retain_after_c3_before_same_node_backstop"
    ].astype(bool)
    backstop_exclusion = prior_retained & direct_same_node & over_threshold
    output["direct_endpoint_same_network_node"] = direct_same_node
    output["same_node_route_backstop_km"] = float(threshold_km)
    output["exclude_same_node_route_backstop"] = backstop_exclusion
    output["same_node_route_backstop_decision_reason"] = np.select(
        [
            backstop_exclusion,
            prior_retained & direct_same_node,
            ~prior_retained,
        ],
        [
            "direct_same_node_route_over_2000_km",
            "retained_direct_same_node_route_at_or_below_2000_km",
            "not_evaluated_prior_c3_exclusion",
        ],
        default="retained_non_same_node",
    )
    output["retain_after_c3_final"] = (
        prior_retained & ~backstop_exclusion
    )
    output["c3_final_decision_reason"] = np.select(
        [
            output["exclude_extreme_duration_speed_final"],
            backstop_exclusion,
        ],
        [
            output["extreme_duration_speed_decision_reason"],
            "direct_same_node_route_over_2000_km",
        ],
        default="retained",
    )
    return output.reset_index(drop=True)


def part_complete(
    output_path: Path,
    sidecar_path: Path,
    contract_hash: str,
    pre_backstop_sha256: str,
    route_decision_sha256: str,
) -> bool:
    if not output_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        sidecar = load_json(sidecar_path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        sidecar.get("contract_hash") == contract_hash
        and sidecar.get("pre_backstop_sha256") == pre_backstop_sha256
        and sidecar.get("route_decision_sha256") == route_decision_sha256
        and sidecar.get("output_sha256") == sha256_file(output_path)
    )


def render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# C3 Final 2021 Candidate-Quality Specification",
            "",
            "## Final Same-Node Backstop",
            "",
            "- The rule applies only when direct endpoint baseline distance is 0 km.",
            "- Exclude only when route-sequence baseline distance is strictly above 2,000 km.",
            "- The 2,000 km boundary itself is retained.",
            "- Shorter same-node local and regional multi-port service remains retained.",
            "",
            "## Final Accounting",
            "",
            (
                f"- Route-circuity-retained candidates: "
                f"{summary['route_retained_candidates']:,}."
            ),
            (
                f"- Prior extreme duration-speed exclusions: "
                f"{summary['prior_extreme_exclusions']:,}."
            ),
            (
                f"- Same-node route-backstop exclusions: "
                f"{summary['same_node_route_backstop_exclusions']:,}."
            ),
            (
                f"- Total C3 exclusions after route circuity: "
                f"{summary['total_c3_exclusions_after_route_circuity']:,}."
            ),
            f"- Final retained candidates: {summary['final_retained_candidates']:,}.",
            "",
            "This output is the final frozen 2021 C3 candidate decision.",
        ]
    ) + "\n"


def run(
    config_path: Path,
    artifact_dir: Path,
    report_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    if int(config["analysis_year"]) != 2021:
        raise ValueError("The final C3 specification is restricted to 2021")
    threshold_km = float(config["same_node_route_backstop_km"])
    if threshold_km != 2000.0:
        raise ValueError("The frozen same-node route backstop must be 2,000 km")
    pre_manifest_path = (
        CODING_ROOT / config["pre_backstop_decision_manifest"]
    )
    route_manifest_path = CODING_ROOT / config["route_decision_manifest"]
    if sha256_file(pre_manifest_path) != config[
        "pre_backstop_decision_manifest_sha256"
    ]:
        raise ValueError("Pre-backstop decision manifest hash mismatch")
    if sha256_file(route_manifest_path) != config[
        "route_decision_manifest_sha256"
    ]:
        raise ValueError("Route-decision manifest hash mismatch")
    pre_manifest = load_json(pre_manifest_path)
    route_manifest = load_json(route_manifest_path)
    if len(pre_manifest["parts"]) != int(config["expected_parts"]):
        raise ValueError("Unexpected pre-backstop part count")

    script_hash = sha256_file(Path(__file__).resolve())
    contract_hash = hashlib.sha256(
        json.dumps(
            {"config": config, "script_sha256": script_hash},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    route_hashes = {
        Path(path).name: metadata["sha256"]
        for path, metadata in route_manifest["outputs"].items()
        if "/parts/" in path
    }
    output_parts: list[dict[str, Any]] = []
    pending_tasks = 0
    for item in sorted(
        pre_manifest["parts"],
        key=lambda value: value["task_id"],
    ):
        task_id = int(item["task_id"])
        pre_path = (
            CODING_ROOT
            / config["pre_backstop_decision_part_folder"]
            / f"part-{task_id:03d}.feather"
        )
        route_path = (
            CODING_ROOT
            / config["route_decision_part_folder"]
            / f"part-{task_id:03d}.feather"
        )
        pre_sha256 = item["sha256"]
        route_sha256 = route_hashes.get(route_path.name)
        if route_sha256 is None:
            raise ValueError(f"Missing route-decision hash for task {task_id}")
        if sha256_file(pre_path) != pre_sha256:
            raise ValueError(f"Pre-backstop part hash mismatch: {task_id}")
        if sha256_file(route_path) != route_sha256:
            raise ValueError(f"Route-decision part hash mismatch: {task_id}")

        output_path = artifact_dir / "parts" / f"part-{task_id:03d}.feather"
        sidecar_path = run_dir / "parts" / f"part-{task_id:03d}.json"
        if not part_complete(
            output_path,
            sidecar_path,
            contract_hash,
            pre_sha256,
            route_sha256,
        ):
            pending_tasks += 1
            pre_decisions = pd.read_feather(pre_path)
            route_decisions = pd.read_feather(route_path)
            output = apply_same_node_route_backstop(
                pre_decisions,
                route_decisions,
                threshold_km=threshold_km,
            )
            write_feather_atomic(output_path, output)
            sidecar = {
                "task_id": task_id,
                "contract_hash": contract_hash,
                "pre_backstop_sha256": pre_sha256,
                "route_decision_sha256": route_sha256,
                "output_sha256": sha256_file(output_path),
                "route_retained_candidates": int(len(output)),
                "pre_backstop_retained_candidates": int(
                    output[
                        "retain_after_c3_before_same_node_backstop"
                    ].sum()
                ),
                "prior_extreme_exclusions": int(
                    output["exclude_extreme_duration_speed_final"].sum()
                ),
                "same_node_route_backstop_exclusions": int(
                    output["exclude_same_node_route_backstop"].sum()
                ),
                "total_c3_exclusions_after_route_circuity": int(
                    (~output["retain_after_c3_final"]).sum()
                ),
                "final_retained_candidates": int(
                    output["retain_after_c3_final"].sum()
                ),
            }
            write_json_atomic(sidecar_path, sidecar)
        sidecar = load_json(sidecar_path)
        output_parts.append(
            {
                "task_id": task_id,
                "path": str(output_path.relative_to(CODING_ROOT)),
                "rows": sidecar["route_retained_candidates"],
                "sha256": sidecar["output_sha256"],
                "pre_backstop_retained_candidates": sidecar[
                    "pre_backstop_retained_candidates"
                ],
                "prior_extreme_exclusions": sidecar[
                    "prior_extreme_exclusions"
                ],
                "same_node_route_backstop_exclusions": sidecar[
                    "same_node_route_backstop_exclusions"
                ],
                "total_c3_exclusions_after_route_circuity": sidecar[
                    "total_c3_exclusions_after_route_circuity"
                ],
                "final_retained_candidates": sidecar[
                    "final_retained_candidates"
                ],
            }
        )

    summary = {
        "status": "pass",
        "analysis_year": 2021,
        "same_node_route_backstop_km": threshold_km,
        "route_retained_candidates": int(
            sum(item["rows"] for item in output_parts)
        ),
        "pre_backstop_retained_candidates": int(
            sum(
                item["pre_backstop_retained_candidates"]
                for item in output_parts
            )
        ),
        "prior_extreme_exclusions": int(
            sum(item["prior_extreme_exclusions"] for item in output_parts)
        ),
        "same_node_route_backstop_exclusions": int(
            sum(
                item["same_node_route_backstop_exclusions"]
                for item in output_parts
            )
        ),
        "total_c3_exclusions_after_route_circuity": int(
            sum(
                item["total_c3_exclusions_after_route_circuity"]
                for item in output_parts
            )
        ),
        "final_retained_candidates": int(
            sum(item["final_retained_candidates"] for item in output_parts)
        ),
    }
    expected = {
        "route_retained_candidates": int(
            config["expected_route_retained_candidates"]
        ),
        "pre_backstop_retained_candidates": int(
            config["expected_pre_backstop_retained_candidates"]
        ),
        "prior_extreme_exclusions": int(
            config["expected_prior_extreme_exclusions"]
        ),
        "same_node_route_backstop_exclusions": int(
            config["expected_same_node_route_backstop_exclusions"]
        ),
        "total_c3_exclusions_after_route_circuity": int(
            config["expected_total_c3_exclusions_after_route_circuity"]
        ),
        "final_retained_candidates": int(
            config["expected_final_retained_candidates"]
        ),
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(
                f"Final C3 total changed for {key}: "
                f"{summary[key]} != {value}"
            )
    if (
        summary["prior_extreme_exclusions"]
        + summary["same_node_route_backstop_exclusions"]
        != summary["total_c3_exclusions_after_route_circuity"]
    ):
        raise ValueError("Final C3 exclusion reasons do not sum to total")

    summary_path = report_dir / "c3_5_final_specification_summary.json"
    report_path = report_dir / "c3_5_final_specification_2021.md"
    write_json_atomic(summary_path, summary)
    write_text_atomic(report_path, render_report(summary))
    manifest = {
        "status": "pass",
        "analysis_year": 2021,
        "contract_hash": contract_hash,
        "pre_backstop_decision_manifest_sha256": config[
            "pre_backstop_decision_manifest_sha256"
        ],
        "route_decision_manifest_sha256": config[
            "route_decision_manifest_sha256"
        ],
        "parts": output_parts,
        "summary": summary,
        "reports": {
            str(summary_path.relative_to(CODING_ROOT)): sha256_file(
                summary_path
            ),
            str(report_path.relative_to(CODING_ROOT)): sha256_file(
                report_path
            ),
        },
    }
    write_json_atomic(artifact_dir / "output_manifest.json", manifest)
    write_json_atomic(
        run_dir / "run_summary_latest.json",
        {
            "completed_tasks": len(output_parts),
            "pending_tasks_this_run": pending_tasks,
            "final_retained_candidates": summary[
                "final_retained_candidates"
            ],
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        args.config,
        args.artifact_dir,
        args.report_dir,
        args.run_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
