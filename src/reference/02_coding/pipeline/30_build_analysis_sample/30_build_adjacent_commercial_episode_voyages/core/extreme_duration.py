#!/usr/bin/env python3
"""Apply the PI-approved extreme duration and implied-speed decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


CODING_ROOT = Path(__file__).resolve().parents[4]
C3_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = (
    C3_ROOT / "config/frozen/extreme_duration_2021.json"
)
DEFAULT_ARTIFACT_DIR = (
    DATA_ROOT / "main/provenance/extreme_duration_2021"
)
DEFAULT_REPORT_DIR = DATA_ROOT / "main/validation/extreme_duration_2021"
DEFAULT_RUN_DIR = (
    CODING_ROOT
    / ".runs/build_multicall_first_passage_voyages/"
    "extreme_duration_2021"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def write_feather_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".feather",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.reset_index(drop=True).to_feather(
            temporary,
            compression="zstd",
            compression_level=6,
        )
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def apply_extreme_duration_speed_decision(
    candidates: pd.DataFrame,
    route_decisions: pd.DataFrame,
    *,
    duration_threshold_days: float,
    speed_threshold_knots: float,
    exclude_zero_route: bool,
) -> pd.DataFrame:
    if len(candidates) != len(route_decisions):
        raise ValueError("Candidate and route-decision part lengths differ")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("Candidate part contains duplicate IDs")
    if route_decisions["candidate_id"].duplicated().any():
        raise ValueError("Route-decision part contains duplicate IDs")
    route_columns = [
        "candidate_id",
        "route_sequence_d_base_km",
        "route_circuity_decision_basis",
        "retain_after_route_circuity",
    ]
    merged = candidates.merge(
        route_decisions[route_columns],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if merged["retain_after_route_circuity"].isna().any():
        raise ValueError("At least one candidate lacks a route decision")
    output = merged.loc[merged["retain_after_route_circuity"]].copy()
    if output["route_sequence_d_base_km"].isna().any():
        raise ValueError("A route-retained candidate lacks route distance")
    if output["route_sequence_d_base_km"].lt(0).any():
        raise ValueError("Route-sequence distance cannot be negative")
    if output["observed_sea_hours"].le(0).any():
        raise ValueError("Observed sea time must be positive")

    output["observed_sea_days"] = output["observed_sea_hours"] / 24.0
    positive_route = output["route_sequence_d_base_km"].gt(0)
    zero_route = output["route_sequence_d_base_km"].eq(0)
    implied_speed = np.full(len(output), np.nan, dtype=np.float64)
    np.divide(
        output["route_sequence_d_base_km"].to_numpy(float),
        1.852 * output["observed_sea_hours"].to_numpy(float),
        out=implied_speed,
        where=positive_route.to_numpy(),
    )
    output["implied_route_sequence_speed_knots"] = implied_speed
    long_duration = output["observed_sea_days"].gt(
        float(duration_threshold_days)
    )
    below_speed_threshold = output[
        "implied_route_sequence_speed_knots"
    ].lt(float(speed_threshold_knots)) & ~np.isclose(
        output["implied_route_sequence_speed_knots"],
        float(speed_threshold_knots),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=False,
    )
    positive_slow_long = (
        long_duration
        & positive_route
        & below_speed_threshold
    )
    zero_long = long_duration & zero_route & bool(exclude_zero_route)
    output["extreme_duration_speed_decision_reason"] = np.select(
        [positive_slow_long, zero_long],
        [
            "positive_route_below_1_kn_over_30d",
            "zero_route_over_30d",
        ],
        default="retained",
    )
    output["exclude_extreme_duration_speed_final"] = (
        positive_slow_long | zero_long
    )
    output["retain_after_c3_final"] = ~output[
        "exclude_extreme_duration_speed_final"
    ]
    return output.reset_index(drop=True)


def part_complete(
    output_path: Path,
    sidecar_path: Path,
    contract_hash: str,
    candidate_sha256: str,
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
        and sidecar.get("candidate_sha256") == candidate_sha256
        and sidecar.get("route_decision_sha256") == route_decision_sha256
        and sidecar.get("output_sha256") == sha256_file(output_path)
    )


def render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# C3.3 Final Extreme Duration-Speed Decision",
            "",
            "## PI Decision",
            "",
            "- Apply the rule only when observed sea time is strictly above 30 days.",
            "- For positive route distance, exclude only when implied speed is below 1 knot.",
            "- For zero route distance, exclude all cases strictly above 30 days.",
            "- The registered 6, 8, and 10 knot references remain diagnostic only.",
            "",
            "## 2021 Result",
            "",
            (
                f"- Route-circuity-retained input: "
                f"{summary['route_retained_candidates']:,}."
            ),
            (
                f"- Positive-route slow and long exclusions: "
                f"{summary['positive_route_slow_long_excluded']:,}."
            ),
            (
                f"- Zero-route over-30-day exclusions: "
                f"{summary['zero_route_long_excluded']:,}."
            ),
            (
                f"- Total excluded: {summary['total_excluded']:,} "
                f"({summary['excluded_share']:.3%})."
            ),
            f"- Final retained: {summary['final_retained']:,}.",
            "",
            "This is an OD-only extreme false-link rule. It does not require AIS",
            "distance and can therefore be reproduced for OD-only years after the",
            "route-sequence baseline is constructed.",
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
        raise ValueError("The final extreme decision is restricted to 2021")
    if float(config["duration_threshold_days"]) != 30.0:
        raise ValueError("The approved duration threshold must remain 30 days")
    if float(config["positive_route_speed_threshold_knots"]) != 1.0:
        raise ValueError("The approved speed threshold must remain 1 knot")
    if not bool(config["exclude_zero_route_above_duration_threshold"]):
        raise ValueError("The PI approved excluding long zero-route cases")

    c2_manifest_path = CODING_ROOT / config["c2_output_manifest"]
    route_manifest_path = CODING_ROOT / config["route_decision_manifest"]
    if sha256_file(c2_manifest_path) != config["c2_output_manifest_sha256"]:
        raise ValueError("C2 output manifest hash mismatch")
    if sha256_file(route_manifest_path) != config[
        "route_decision_manifest_sha256"
    ]:
        raise ValueError("Route-decision manifest hash mismatch")
    c2_manifest = load_json(c2_manifest_path)
    route_manifest = load_json(route_manifest_path)
    if len(c2_manifest["parts"]) != int(config["expected_parts"]):
        raise ValueError("Unexpected C2 part count")

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
    for item in sorted(c2_manifest["parts"], key=lambda value: value["task_id"]):
        task_id = int(item["task_id"])
        candidate_path = CODING_ROOT / item["retained_path"]
        route_path = (
            CODING_ROOT
            / config["route_decision_part_folder"]
            / f"part-{task_id:03d}.feather"
        )
        candidate_sha256 = item["retained_sha256"]
        route_sha256 = route_hashes.get(route_path.name)
        if route_sha256 is None:
            raise ValueError(f"Missing route-decision hash for task {task_id}")
        if sha256_file(candidate_path) != candidate_sha256:
            raise ValueError(f"C2 candidate part hash mismatch: {task_id}")
        if sha256_file(route_path) != route_sha256:
            raise ValueError(f"Route-decision part hash mismatch: {task_id}")

        output_path = artifact_dir / "parts" / f"part-{task_id:03d}.feather"
        sidecar_path = run_dir / "parts" / f"part-{task_id:03d}.json"
        if not part_complete(
            output_path,
            sidecar_path,
            contract_hash,
            candidate_sha256,
            route_sha256,
        ):
            pending_tasks += 1
            candidates = pd.read_feather(
                candidate_path,
                columns=[
                    "candidate_id",
                    "pair_id",
                    "vessel_uid_clean",
                    "origin_city",
                    "dest_city",
                    "observed_sea_hours",
                ],
            )
            route_decisions = pd.read_feather(route_path)
            output = apply_extreme_duration_speed_decision(
                candidates,
                route_decisions,
                duration_threshold_days=float(
                    config["duration_threshold_days"]
                ),
                speed_threshold_knots=float(
                    config["positive_route_speed_threshold_knots"]
                ),
                exclude_zero_route=bool(
                    config["exclude_zero_route_above_duration_threshold"]
                ),
            )
            write_feather_atomic(output_path, output)
            sidecar = {
                "task_id": task_id,
                "contract_hash": contract_hash,
                "candidate_sha256": candidate_sha256,
                "route_decision_sha256": route_sha256,
                "output_sha256": sha256_file(output_path),
                "route_retained_candidates": int(len(output)),
                "positive_route_slow_long_excluded": int(
                    output["extreme_duration_speed_decision_reason"].eq(
                        "positive_route_below_1_kn_over_30d"
                    ).sum()
                ),
                "zero_route_long_excluded": int(
                    output["extreme_duration_speed_decision_reason"].eq(
                        "zero_route_over_30d"
                    ).sum()
                ),
                "total_excluded": int(
                    output["exclude_extreme_duration_speed_final"].sum()
                ),
                "final_retained": int(output["retain_after_c3_final"].sum()),
            }
            write_json_atomic(sidecar_path, sidecar)
        sidecar = load_json(sidecar_path)
        output_parts.append(
            {
                "task_id": task_id,
                "path": str(output_path.relative_to(CODING_ROOT)),
                "rows": sidecar["route_retained_candidates"],
                "sha256": sidecar["output_sha256"],
                "positive_route_slow_long_excluded": sidecar[
                    "positive_route_slow_long_excluded"
                ],
                "zero_route_long_excluded": sidecar[
                    "zero_route_long_excluded"
                ],
                "total_excluded": sidecar["total_excluded"],
                "final_retained": sidecar["final_retained"],
            }
        )

    summary = {
        "status": "pass",
        "analysis_year": 2021,
        "duration_threshold_days": 30.0,
        "positive_route_speed_threshold_knots": 1.0,
        "exclude_zero_route_above_duration_threshold": True,
        "route_retained_candidates": int(
            sum(item["rows"] for item in output_parts)
        ),
        "positive_route_slow_long_excluded": int(
            sum(
                item["positive_route_slow_long_excluded"]
                for item in output_parts
            )
        ),
        "zero_route_long_excluded": int(
            sum(item["zero_route_long_excluded"] for item in output_parts)
        ),
        "total_excluded": int(
            sum(item["total_excluded"] for item in output_parts)
        ),
        "final_retained": int(
            sum(item["final_retained"] for item in output_parts)
        ),
    }
    summary["excluded_share"] = (
        summary["total_excluded"] / summary["route_retained_candidates"]
    )
    expected = {
        "route_retained_candidates": int(
            config["expected_route_retained_candidates"]
        ),
        "positive_route_slow_long_excluded": int(
            config["expected_positive_route_slow_long_excluded"]
        ),
        "zero_route_long_excluded": int(
            config["expected_zero_route_long_excluded"]
        ),
        "total_excluded": int(config["expected_total_excluded"]),
        "final_retained": int(config["expected_final_retained"]),
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(
                f"Final decision total changed for {key}: "
                f"{summary[key]} != {value}"
            )
    if (
        summary["positive_route_slow_long_excluded"]
        + summary["zero_route_long_excluded"]
        != summary["total_excluded"]
    ):
        raise ValueError("Final exclusion reasons do not sum to the total")

    summary_path = report_dir / "c3_3_extreme_duration_speed_summary.json"
    report_path = report_dir / "c3_3_extreme_duration_speed_final_decision.md"
    write_json_atomic(summary_path, summary)
    write_text_atomic(report_path, render_report(summary))
    manifest = {
        "status": "pass",
        "analysis_year": 2021,
        "contract_hash": contract_hash,
        "c2_output_manifest_sha256": config["c2_output_manifest_sha256"],
        "route_decision_manifest_sha256": config[
            "route_decision_manifest_sha256"
        ],
        "parts": output_parts,
        "summary": summary,
        "reports": {
            str(summary_path.relative_to(CODING_ROOT)): sha256_file(
                summary_path
            ),
            str(report_path.relative_to(CODING_ROOT)): sha256_file(report_path),
        },
    }
    write_json_atomic(artifact_dir / "output_manifest.json", manifest)
    write_json_atomic(
        run_dir / "run_summary_latest.json",
        {
            "completed_tasks": len(output_parts),
            "pending_tasks_this_run": pending_tasks,
            "final_retained": summary["final_retained"],
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
