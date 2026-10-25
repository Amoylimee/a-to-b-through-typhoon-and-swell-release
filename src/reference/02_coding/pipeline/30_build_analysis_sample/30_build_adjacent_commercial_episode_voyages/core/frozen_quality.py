#!/usr/bin/env python3
"""Apply the frozen 2021 C3 candidate-quality rules to C5 2025."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import city_snaps as c0_snap
from . import extreme_duration as c3_extreme
from . import final_quality as c3_final
from . import pair_routes as c0_route
from . import route_circuity as c3_route
from . import route_decision as c3_decision

CODING_ROOT = Path(__file__).resolve().parents[4]
C5_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = C5_ROOT / "config/frozen/frozen_quality_2025.json"
DEFAULT_ARTIFACT_DIR = DATA_ROOT / "main/provenance/frozen_quality_2025"
DEFAULT_RUN_DIR = (
    CODING_ROOT
    / ".runs/build_multicall_first_passage_voyages/"
    "frozen_quality_2025"
)

_DISTANCE: dict[str, float] = {}
_SAME_NODE: set[str] = set()
_DIRECT_DISTANCE: dict[str, float] = {}
_DIRECT_SAME_NODE: set[str] = set()
_SEPARATOR = ">"
_CIRCUITY_THRESHOLD = 2.0
_SENSITIVITY_THRESHOLDS: tuple[float, ...] = ()
_DURATION_THRESHOLD_DAYS = 30.0
_SPEED_THRESHOLD_KNOTS = 1.0
_EXCLUDE_ZERO_ROUTE = True
_SAME_NODE_BACKSTOP_KM = 2000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=CODING_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verified_path(config: dict[str, Any], path_key: str) -> Path:
    path = CODING_ROOT / config[path_key]
    if sha256_file(path) != config[f"{path_key}_sha256"]:
        raise ValueError(f"Frozen input hash mismatch: {path}")
    return path


def verified_output(
    manifest: dict[str, Any],
    output_name: str,
) -> Path:
    record = manifest["outputs"][output_name]
    path = CODING_ROOT / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Frozen output hash mismatch: {output_name}")
    return path


def verify_contract(
    config: dict[str, Any],
    config_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    if git_value("branch", "--show-current") != config["approved_branch"]:
        raise ValueError("Unexpected branch")
    exact_rules = {
        "route_circuity_threshold": 2.0,
        "duration_threshold_days": 30.0,
        "positive_route_speed_threshold_knots": 1.0,
        "same_node_route_backstop_km": 2000.0,
    }
    for key, expected in exact_rules.items():
        if float(config[key]) != expected:
            raise ValueError(f"Frozen C3 value changed: {key}")
    if not config["exclude_zero_route_above_duration_threshold"]:
        raise ValueError("Frozen zero-route duration rule changed")
    for key in (
        "apply_cumulative_w",
        "apply_standalone_speed_gate",
        "apply_physical_cap",
    ):
        if config[key]:
            raise ValueError(f"Prohibited C3/C4 gate enabled: {key}")
    for key in (
        "c5_1f_output_manifest",
        "c5_1d_output_manifest",
        "yardstick_table",
        "c5_1_output_manifest",
        "c5_1c_output_manifest",
        "frozen_city_snap_config",
        "frozen_city_snap_runner",
        "frozen_route_config",
        "frozen_route_runner",
        "frozen_2021_node_routes",
        "c5_new_node_routes",
        "frozen_route_circuity_runner",
        "frozen_route_decision_runner",
        "frozen_extreme_duration_runner",
        "frozen_final_specification_runner",
    ):
        verified_path(config, key)
    c2_manifest_path = CODING_ROOT / config["c5_1f_output_manifest"]
    c2_manifest = json.loads(c2_manifest_path.read_text(encoding="utf-8"))
    yardstick_path = CODING_ROOT / config["yardstick_table"]
    provenance = {
        "analysis_year": 2025,
        "operational_parameter_year": 2021,
        "base_commit": git_value("rev-parse", "HEAD"),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "c5_1f_output_manifest_sha256": config[
            "c5_1f_output_manifest_sha256"
        ],
        "c5_1d_output_manifest_sha256": config[
            "c5_1d_output_manifest_sha256"
        ],
        "yardstick_table_sha256": config["yardstick_table_sha256"],
        "c5_1_output_manifest_sha256": config[
            "c5_1_output_manifest_sha256"
        ],
        "c5_1c_output_manifest_sha256": config[
            "c5_1c_output_manifest_sha256"
        ],
        "frozen_city_snap_config_sha256": config[
            "frozen_city_snap_config_sha256"
        ],
        "frozen_city_snap_runner_sha256": config[
            "frozen_city_snap_runner_sha256"
        ],
        "frozen_route_config_sha256": config[
            "frozen_route_config_sha256"
        ],
        "frozen_route_runner_sha256": config[
            "frozen_route_runner_sha256"
        ],
        "frozen_2021_node_routes_sha256": config[
            "frozen_2021_node_routes_sha256"
        ],
        "c5_new_node_routes_sha256": config[
            "c5_new_node_routes_sha256"
        ],
        "frozen_route_circuity_runner_sha256": config[
            "frozen_route_circuity_runner_sha256"
        ],
        "frozen_route_decision_runner_sha256": config[
            "frozen_route_decision_runner_sha256"
        ],
        "frozen_extreme_duration_runner_sha256": config[
            "frozen_extreme_duration_runner_sha256"
        ],
        "frozen_final_specification_runner_sha256": config[
            "frozen_final_specification_runner_sha256"
        ],
        "rules": {
            key: config[key]
            for key in (
                "sequence_separator",
                "route_circuity_threshold",
                "sensitivity_thresholds",
                "duration_threshold_days",
                "positive_route_speed_threshold_knots",
                "exclude_zero_route_above_duration_threshold",
                "same_node_route_backstop_km",
                "apply_cumulative_w",
                "apply_standalone_speed_gate",
                "apply_physical_cap",
            )
        },
    }
    return c2_manifest, yardstick_path, provenance


def enumerate_transition_usage(
    c2_manifest: dict[str, Any],
    separator: str,
) -> pd.DataFrame:
    usage: Counter[str] = Counter()
    for record in c2_manifest["parts"]:
        frame = pd.read_feather(
            CODING_ROOT / record["retained_path"],
            columns=[
                "origin_city",
                "dest_city",
                "intermediate_city_sequence",
            ],
        )
        for row in frame.itertuples(index=False):
            cities = c3_route.candidate_city_chain(
                row.origin_city,
                row.intermediate_city_sequence or "",
                row.dest_city,
                separator,
            )
            usage.update(c3_route.transition_pair_ids(cities))
    records = []
    for pair_id, occurrences in sorted(usage.items()):
        origin_city, dest_city = pair_id.split("__", 1)
        records.append(
            {
                "pair_id": pair_id,
                "origin_city": origin_city,
                "dest_city": dest_city,
                "candidate_transition_occurrences": int(occurrences),
            }
        )
    return pd.DataFrame(records)


def build_supplementary_snaps(
    config: dict[str, Any],
    required_cities: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c5_1c = json.loads(
        (CODING_ROOT / config["c5_1c_output_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    existing = pd.read_feather(
        verified_output(c5_1c, "combined_snap")
    )
    missing = sorted(required_cities - set(existing["city_id"]))
    if not missing:
        return existing, existing.iloc[0:0].copy()
    c5_1 = json.loads(
        (CODING_ROOT / config["c5_1_output_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    calls = pd.read_feather(
        verified_output(c5_1, "calls"),
        columns=c0_snap.CALL_COLUMNS,
    )
    snap_config = json.loads(
        (CODING_ROOT / config["frozen_city_snap_config"]).read_text(
            encoding="utf-8"
        )
    )
    c0_snap.validate_contract(snap_config)
    lookup, _, _ = c0_snap.load_coordinate_lookup(
        snap_config["coordinate_source_precedence"]
    )
    city_rows = [
        {
            "city_id": city_id,
            "country_iso3": city_id.split(":", 1)[0],
            "city_name": city_id.split(":", 1)[1],
        }
        for city_id in missing
    ]
    coordinates, _, _ = c0_snap.build_city_coordinates(
        calls,
        pd.DataFrame(city_rows),
        lookup,
        float(snap_config["coordinate_spread_review_threshold_deg"]),
        [],
    )
    supplementary, _ = c0_snap.add_network_snaps(
        coordinates,
        float(snap_config["snap_review_threshold_km"]),
    )
    combined = (
        pd.concat([existing, supplementary], ignore_index=True)
        .sort_values("city_id", kind="stable")
        .reset_index(drop=True)
    )
    if not required_cities.issubset(set(combined["city_id"])):
        raise ValueError("A transition city still lacks a network snap")
    return combined, supplementary


def route_unresolved_nodes(
    unresolved: pd.DataFrame,
    config: dict[str, Any],
    artifact_dir: Path,
) -> pd.DataFrame:
    if unresolved.empty:
        return unresolved.assign(d_base=pd.Series(dtype="float64"))
    route_config = json.loads(
        (CODING_ROOT / config["frozen_route_config"]).read_text(
            encoding="utf-8"
        )
    )
    task_root = artifact_dir / "transition_route_tasks"
    inputs = task_root / "inputs"
    outputs = task_root / "outputs"
    failures = task_root / "failures"
    for path in (inputs, outputs, failures):
        path.mkdir(parents=True, exist_ok=True)
    origin_nodes = sorted(unresolved["origin_network_node_id"].unique())
    chunk_size = int(config["transition_route_origin_nodes_per_task"])
    tasks = []
    for task_id, start in enumerate(range(0, len(origin_nodes), chunk_size)):
        selected = set(origin_nodes[start : start + chunk_size])
        part = unresolved.loc[
            unresolved["origin_network_node_id"].isin(selected)
        ].reset_index(drop=True)
        input_path = inputs / f"part-{task_id:03d}.feather"
        output_path = outputs / f"part-{task_id:03d}.feather"
        failure_path = failures / f"part-{task_id:03d}.json"
        if not input_path.exists():
            part.to_feather(input_path, compression="zstd")
        tasks.append(
            {
                "task_id": f"transition-{task_id:03d}",
                "input_path": str(input_path),
                "output_path": str(output_path),
                "failure_path": str(failure_path),
                "origin_count": len(selected),
                "node_pair_count": len(part),
            }
        )
    pending = [task for task in tasks if not Path(task["output_path"]).is_file()]
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(int(config["worker_count"]), len(pending), 16),
            initializer=c0_route.initialize_route_worker,
            initargs=({"base": route_config["regimes"]["base"]},),
        ) as executor:
            futures = {
                executor.submit(c0_route.run_route_task, task): task
                for task in pending
            }
            failures_found = []
            for future in as_completed(futures):
                result = future.result()
                if result["status"] != "success":
                    failures_found.append(result)
            if failures_found:
                raise RuntimeError(
                    f"{len(failures_found)} transition route tasks failed"
                )
    routed = pd.concat(
        [pd.read_feather(task["output_path"]) for task in tasks],
        ignore_index=True,
    )
    if routed["d_base"].isna().any():
        raise ValueError("A required baseline transition route is infeasible")
    return routed


def build_transition_support(
    config: dict[str, Any],
    c2_manifest: dict[str, Any],
    yardstick_path: Path,
    artifact_dir: Path,
) -> tuple[Path, dict[str, int]]:
    usage = enumerate_transition_usage(
        c2_manifest, str(config["sequence_separator"])
    )
    direct = pd.read_feather(
        yardstick_path,
        columns=[
            "pair_id",
            "origin_city",
            "dest_city",
            "d_base",
            "same_network_node",
            "origin_network_node_id",
            "dest_network_node_id",
        ],
    )
    missing = usage.loc[~usage["pair_id"].isin(set(direct["pair_id"]))].copy()
    required_cities = set(missing["origin_city"]).union(missing["dest_city"])
    snaps, supplementary = build_supplementary_snaps(
        config, required_cities
    )
    snap_columns = [
        "city_id",
        "network_node_id",
        "network_lon",
        "network_lat",
    ]
    origin = snaps[snap_columns].rename(
        columns={
            "city_id": "origin_city",
            "network_node_id": "origin_network_node_id",
            "network_lon": "origin_network_lon",
            "network_lat": "origin_network_lat",
        }
    )
    destination = snaps[snap_columns].rename(
        columns={
            "city_id": "dest_city",
            "network_node_id": "dest_network_node_id",
            "network_lon": "dest_network_lon",
            "network_lat": "dest_network_lat",
        }
    )
    extension = (
        missing.merge(origin, on="origin_city", validate="many_to_one")
        .merge(destination, on="dest_city", validate="many_to_one")
    )
    extension["node_pair_id"] = (
        extension["origin_network_node_id"]
        + "__"
        + extension["dest_network_node_id"]
    )
    extension["same_network_node"] = extension[
        "origin_network_node_id"
    ].eq(extension["dest_network_node_id"])
    node_columns = ["node_pair_id", "d_base"]
    node_sources = [
        direct.assign(
            node_pair_id=(
                direct["origin_network_node_id"]
                + "__"
                + direct["dest_network_node_id"]
            )
        )[node_columns],
        pd.read_feather(
            CODING_ROOT / config["frozen_2021_node_routes"],
            columns=node_columns,
        ),
        pd.read_feather(
            CODING_ROOT / config["c5_new_node_routes"],
            columns=node_columns,
        ),
    ]
    known_nodes = pd.concat(node_sources, ignore_index=True)
    if (
        known_nodes.groupby("node_pair_id")["d_base"].nunique(dropna=False) > 1
    ).any():
        raise ValueError("Known node-pair distances disagree")
    known_nodes = known_nodes.drop_duplicates("node_pair_id")
    extension = extension.merge(
        known_nodes,
        on="node_pair_id",
        how="left",
        validate="many_to_one",
    )
    extension.loc[extension["same_network_node"], "d_base"] = 0.0
    unresolved = (
        extension.loc[
            extension["d_base"].isna(),
            [
                "node_pair_id",
                "origin_network_node_id",
                "origin_network_lon",
                "origin_network_lat",
                "dest_network_node_id",
                "dest_network_lon",
                "dest_network_lat",
            ],
        ]
        .drop_duplicates("node_pair_id")
        .reset_index(drop=True)
    )
    routed = route_unresolved_nodes(
        unresolved, config, artifact_dir
    )
    if not routed.empty:
        extension = extension.merge(
            routed[["node_pair_id", "d_base"]].rename(
                columns={"d_base": "new_d_base"}
            ),
            on="node_pair_id",
            how="left",
            validate="many_to_one",
        )
        extension["d_base"] = extension["d_base"].fillna(
            extension["new_d_base"]
        )
        extension = extension.drop(columns="new_d_base")
    if extension["d_base"].isna().any():
        raise ValueError("A required transition distance remains missing")
    transition = usage.merge(
        pd.concat(
            [
                direct[
                    [
                        "pair_id",
                        "d_base",
                        "same_network_node",
                    ]
                ],
                extension[
                    [
                        "pair_id",
                        "d_base",
                        "same_network_node",
                    ]
                ],
            ],
            ignore_index=True,
        ),
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    if transition["d_base"].isna().any():
        raise ValueError("Transition support is incomplete")
    usage_path = artifact_dir / "transition_usage_2025.feather"
    extension_path = artifact_dir / "transition_yardstick_extension_2025.feather"
    supplementary_path = artifact_dir / "supplementary_city_snaps_2025.csv"
    transition_path = artifact_dir / "transition_yardstick_2025.feather"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    usage.to_feather(usage_path, compression="zstd")
    extension.to_feather(extension_path, compression="zstd")
    supplementary.to_csv(supplementary_path, index=False)
    transition.to_feather(transition_path, compression="zstd")
    summary = {
        "required_transition_pair_count": int(len(usage)),
        "direct_yardstick_transition_pair_count": int(
            len(usage) - len(missing)
        ),
        "extended_transition_pair_count": int(len(missing)),
        "supplementary_city_snap_count": int(len(supplementary)),
        "new_transition_node_route_count": int(len(unresolved)),
    }
    return transition_path, summary


def initialize_worker(
    transition_yardstick_path: str,
    direct_yardstick_path: str,
    config: dict[str, Any],
) -> None:
    global _DISTANCE
    global _SAME_NODE
    global _DIRECT_DISTANCE
    global _DIRECT_SAME_NODE
    global _SEPARATOR
    global _CIRCUITY_THRESHOLD
    global _SENSITIVITY_THRESHOLDS
    global _DURATION_THRESHOLD_DAYS
    global _SPEED_THRESHOLD_KNOTS
    global _EXCLUDE_ZERO_ROUTE
    global _SAME_NODE_BACKSTOP_KM

    yardstick = pd.read_feather(
        transition_yardstick_path,
        columns=["pair_id", "d_base", "same_network_node"],
    )
    if yardstick["pair_id"].duplicated().any():
        raise ValueError("C5 yardstick duplicates a pair")
    if yardstick["d_base"].isna().any() or yardstick["d_base"].lt(0).any():
        raise ValueError("C5 yardstick has an invalid baseline distance")
    _DISTANCE = dict(zip(yardstick["pair_id"], yardstick["d_base"]))
    _SAME_NODE = set(
        yardstick.loc[yardstick["same_network_node"], "pair_id"]
    )
    direct = pd.read_feather(
        direct_yardstick_path,
        columns=["pair_id", "d_base", "same_network_node"],
    )
    if direct["pair_id"].duplicated().any():
        raise ValueError("C5 direct yardstick duplicates a pair")
    _DIRECT_DISTANCE = dict(zip(direct["pair_id"], direct["d_base"]))
    _DIRECT_SAME_NODE = set(
        direct.loc[direct["same_network_node"], "pair_id"]
    )
    _SEPARATOR = str(config["sequence_separator"])
    _CIRCUITY_THRESHOLD = float(config["route_circuity_threshold"])
    _SENSITIVITY_THRESHOLDS = tuple(
        float(value) for value in config["sensitivity_thresholds"]
    )
    _DURATION_THRESHOLD_DAYS = float(config["duration_threshold_days"])
    _SPEED_THRESHOLD_KNOTS = float(
        config["positive_route_speed_threshold_knots"]
    )
    _EXCLUDE_ZERO_ROUTE = bool(
        config["exclude_zero_route_above_duration_threshold"]
    )
    _SAME_NODE_BACKSTOP_KM = float(
        config["same_node_route_backstop_km"]
    )


def build_route_decisions(candidates: pd.DataFrame) -> pd.DataFrame:
    route_distance = np.full(len(candidates), np.nan, dtype=np.float64)
    circuity = np.full(len(candidates), np.nan, dtype=np.float64)
    statuses: list[str] = []
    for position, row in enumerate(
        candidates[
            [
                "pair_id",
                "origin_city",
                "dest_city",
                "intermediate_city_sequence",
                "n_intermediate_city_episodes",
            ]
        ].itertuples(index=False)
    ):
        cities = c3_route.candidate_city_chain(
            row.origin_city,
            row.intermediate_city_sequence or "",
            row.dest_city,
            _SEPARATOR,
        )
        pair_ids = c3_route.transition_pair_ids(cities)
        distances = [_DISTANCE.get(pair_id) for pair_id in pair_ids]
        same_node = [pair_id in _SAME_NODE for pair_id in pair_ids]
        distance, ratio, status = c3_route.classify_route_circuity(
            _DIRECT_DISTANCE.get(row.pair_id),
            row.pair_id in _DIRECT_SAME_NODE,
            distances,
            same_node,
            int(row.n_intermediate_city_episodes) + 1,
        )
        route_distance[position] = distance
        circuity[position] = ratio
        statuses.append(status)
    route = candidates[
        [
            "candidate_id",
            "pair_id",
            "n_observed_sea_legs",
            "n_intermediate_city_episodes",
        ]
    ].copy()
    route["route_sequence_d_base_km"] = route_distance
    route["direct_d_base_km"] = route["pair_id"].map(_DIRECT_DISTANCE)
    route["route_circuity_ratio"] = circuity
    route["route_circuity_status"] = statuses
    complete = route["route_circuity_status"].eq("complete_positive")
    route["exclude_route_circuity_gt_2"] = (
        complete
        & route["route_circuity_ratio"].gt(_CIRCUITY_THRESHOLD)
    )
    for threshold in _SENSITIVITY_THRESHOLDS:
        label = str(threshold).replace(".", "_")
        route[f"exclude_route_circuity_gt_{label}"] = (
            complete & route["route_circuity_ratio"].gt(threshold)
        )
    return c3_decision.apply_final_decision(route, _CIRCUITY_THRESHOLD)


def process_part(
    task_id: int,
    candidate_path_text: str,
    route_path_text: str,
    final_path_text: str,
    retained_path_text: str,
    sidecar_path_text: str,
    contract_hash: str,
    candidate_source_hash: str,
) -> dict[str, Any]:
    candidate_path = Path(candidate_path_text)
    route_path = Path(route_path_text)
    final_path = Path(final_path_text)
    retained_path = Path(retained_path_text)
    sidecar_path = Path(sidecar_path_text)
    candidates = pd.read_feather(candidate_path)
    route = build_route_decisions(candidates)
    pre_backstop = c3_extreme.apply_extreme_duration_speed_decision(
        candidates,
        route,
        duration_threshold_days=_DURATION_THRESHOLD_DAYS,
        speed_threshold_knots=_SPEED_THRESHOLD_KNOTS,
        exclude_zero_route=_EXCLUDE_ZERO_ROUTE,
    )
    final = c3_final.apply_same_node_route_backstop(
        pre_backstop,
        route,
        threshold_km=_SAME_NODE_BACKSTOP_KM,
    )
    retained = final.loc[final["retain_after_c3_final"]].reset_index(
        drop=True
    )
    for path in (
        route_path,
        final_path,
        retained_path,
        sidecar_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    route.to_feather(route_path, compression="zstd", compression_level=6)
    final.to_feather(final_path, compression="zstd", compression_level=6)
    retained.to_feather(
        retained_path, compression="zstd", compression_level=6
    )
    status_counts = Counter(route["route_circuity_decision_basis"])
    payload = {
        "task_id": task_id,
        "contract_hash": contract_hash,
        "candidate_source_sha256": candidate_source_hash,
        "input_candidates": int(len(candidates)),
        "route_circuity_exclusions": int(
            route["exclude_route_circuity_gt_2_final"].sum()
        ),
        "route_retained_candidates": int(
            route["retain_after_route_circuity"].sum()
        ),
        "positive_route_slow_long_exclusions": int(
            pre_backstop["extreme_duration_speed_decision_reason"]
            .eq("positive_route_below_1_kn_over_30d")
            .sum()
        ),
        "zero_route_long_exclusions": int(
            pre_backstop["extreme_duration_speed_decision_reason"]
            .eq("zero_route_over_30d")
            .sum()
        ),
        "same_node_backstop_exclusions": int(
            final["exclude_same_node_route_backstop"].sum()
        ),
        "final_retained_candidates": int(len(retained)),
        "decision_basis_counts": {
            str(key): int(value)
            for key, value in sorted(status_counts.items())
        },
        "route_sha256": sha256_file(route_path),
        "final_sha256": sha256_file(final_path),
        "retained_sha256": sha256_file(retained_path),
    }
    write_json(sidecar_path, payload)
    return payload


def part_is_complete(
    sidecar_path: Path,
    route_path: Path,
    final_path: Path,
    retained_path: Path,
    contract_hash: str,
    candidate_source_hash: str,
) -> bool:
    if not all(
        path.is_file()
        for path in (sidecar_path, route_path, final_path, retained_path)
    ):
        return False
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("contract_hash") == contract_hash
        and payload.get("candidate_source_sha256") == candidate_source_hash
        and payload.get("route_sha256") == sha256_file(route_path)
        and payload.get("final_sha256") == sha256_file(final_path)
        and payload.get("retained_sha256") == sha256_file(retained_path)
    )


def aggregate_counts(
    parts: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    result: Counter[str] = Counter()
    for part in parts:
        result.update(part[key])
    return dict(sorted(result.items()))


def run(
    config_path: Path,
    artifact_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    artifact_dir = artifact_dir.resolve()
    run_dir = run_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    c2_manifest, yardstick_path, provenance = verify_contract(
        config, config_path
    )
    contract_hash = hashlib.sha256(
        json.dumps(
            provenance, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    contract_path = artifact_dir / "execution_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != provenance:
            raise ValueError("C5.1g execution contract changed")
    else:
        write_json(contract_path, provenance)
    transition_path, transition_summary = build_transition_support(
        config,
        c2_manifest,
        yardstick_path,
        artifact_dir,
    )
    initialize_worker(
        str(transition_path),
        str(yardstick_path),
        config,
    )

    route_dir = artifact_dir / "route_decisions_2025"
    final_dir = artifact_dir / "c3_decisions_2025"
    retained_dir = artifact_dir / "candidate_passages_c3_retained_2025"
    sidecar_dir = run_dir / "parts"
    complete: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    if len(c2_manifest["parts"]) != int(config["task_count"]):
        raise ValueError("C5.1f part count changed")
    for record in c2_manifest["parts"]:
        task_id = int(record["task_id"])
        candidate_path = CODING_ROOT / record["retained_path"]
        candidate_hash = sha256_file(candidate_path)
        if candidate_hash != record["retained_sha256"]:
            raise ValueError("C5.1f retained-part hash mismatch")
        route_path = route_dir / f"part-{task_id:03d}.feather"
        final_path = final_dir / f"part-{task_id:03d}.feather"
        retained_path = retained_dir / f"part-{task_id:03d}.feather"
        sidecar_path = sidecar_dir / f"part-{task_id:03d}.json"
        if part_is_complete(
            sidecar_path,
            route_path,
            final_path,
            retained_path,
            contract_hash,
            candidate_hash,
        ):
            complete.append(
                json.loads(sidecar_path.read_text(encoding="utf-8"))
            )
        else:
            pending.append(
                (
                    task_id,
                    str(candidate_path),
                    str(route_path),
                    str(final_path),
                    str(retained_path),
                    str(sidecar_path),
                    contract_hash,
                    candidate_hash,
                )
            )
    if pending:
        context = mp.get_context("fork")
        workers = min(int(config["worker_count"]), len(pending), 16)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            futures = {
                executor.submit(process_part, *task): task[0]
                for task in pending
            }
            for future in as_completed(futures):
                payload = future.result()
                complete.append(payload)
                print(
                    f"[C5.1g] task {payload['task_id']:03d}: "
                    f"{payload['final_retained_candidates']:,} retained",
                    flush=True,
                )
    parts = sorted(complete, key=lambda row: row["task_id"])
    input_candidates = sum(row["input_candidates"] for row in parts)
    route_exclusions = sum(
        row["route_circuity_exclusions"] for row in parts
    )
    route_retained = sum(
        row["route_retained_candidates"] for row in parts
    )
    positive_slow_long = sum(
        row["positive_route_slow_long_exclusions"] for row in parts
    )
    zero_long = sum(row["zero_route_long_exclusions"] for row in parts)
    same_node_backstop = sum(
        row["same_node_backstop_exclusions"] for row in parts
    )
    final_retained = sum(
        row["final_retained_candidates"] for row in parts
    )
    if input_candidates != int(config["expected_c2_retained_candidates"]):
        raise ValueError("C5.1g input candidate count changed")
    if route_exclusions + route_retained != input_candidates:
        raise ValueError("C5.1g route decision partition failed")
    if (
        route_retained
        - positive_slow_long
        - zero_long
        - same_node_backstop
        != final_retained
    ):
        raise ValueError("C5.1g final decision partition failed")
    summary = {
        "status": "pass",
        "analysis_year": 2025,
        "operational_parameter_year": 2021,
        "input_c2_retained_candidates": input_candidates,
        "route_circuity_threshold": float(
            config["route_circuity_threshold"]
        ),
        "route_circuity_exclusions": route_exclusions,
        "route_retained_candidates": route_retained,
        "route_decision_basis_counts": aggregate_counts(
            parts, "decision_basis_counts"
        ),
        "duration_threshold_days": float(
            config["duration_threshold_days"]
        ),
        "positive_route_speed_threshold_knots": float(
            config["positive_route_speed_threshold_knots"]
        ),
        "positive_route_slow_long_exclusions": positive_slow_long,
        "zero_route_long_exclusions": zero_long,
        "same_node_route_backstop_km": float(
            config["same_node_route_backstop_km"]
        ),
        "same_node_backstop_exclusions": same_node_backstop,
        "total_c3_exclusions": (
            route_exclusions
            + positive_slow_long
            + zero_long
            + same_node_backstop
        ),
        "final_c3_retained_candidates": final_retained,
        "apply_cumulative_w": False,
        "apply_standalone_speed_gate": False,
        "apply_physical_cap": False,
        "task_count": len(parts),
        **transition_summary,
    }
    summary_path = artifact_dir / "c5_1g_frozen_c3_summary.json"
    write_json(summary_path, summary)
    output_paths = [
        summary_path,
        artifact_dir / "transition_usage_2025.feather",
        artifact_dir / "transition_yardstick_extension_2025.feather",
        artifact_dir / "supplementary_city_snaps_2025.csv",
        artifact_dir / "transition_yardstick_2025.feather",
    ]
    part_manifest = []
    for row in parts:
        task_id = row["task_id"]
        records: dict[str, Any] = {"task_id": task_id}
        for label, path, rows in (
            (
                "route",
                route_dir / f"part-{task_id:03d}.feather",
                row["input_candidates"],
            ),
            (
                "decision",
                final_dir / f"part-{task_id:03d}.feather",
                row["route_retained_candidates"],
            ),
            (
                "retained",
                retained_dir / f"part-{task_id:03d}.feather",
                row["final_retained_candidates"],
            ),
        ):
            output_paths.append(path)
            records[f"{label}_path"] = str(path.relative_to(CODING_ROOT))
            records[f"{label}_rows"] = rows
            records[f"{label}_sha256"] = sha256_file(path)
        part_manifest.append(records)
    manifest = {
        "status": "pass",
        "contract_hash": contract_hash,
        "summary": summary,
        "parts": part_manifest,
        "outputs": {
            str(path.relative_to(CODING_ROOT)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        },
    }
    write_json(artifact_dir / "c5_1g_output_manifest.json", manifest)
    write_json(
        run_dir / "STATUS.json",
        {
            **summary,
            "resumed_skipped_task_count": len(parts) - len(pending),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    arguments = parser.parse_args()
    result = run(
        arguments.config, arguments.artifact_dir, arguments.run_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
