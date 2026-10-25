#!/usr/bin/env python3
"""Build the complete 2021 route-circuity table for C2-retained candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import tempfile
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
DEFAULT_CONFIG = PIPELINE_ROOT / "config/frozen/route_circuity_2021.json"
DEFAULT_ARTIFACT_DIR = DATA_ROOT / "main/provenance/route_circuity_2021"
DEFAULT_REPORT_DIR = DATA_ROOT / "main/validation/route_circuity_2021"
DEFAULT_RUN_DIR = (
    CODING_ROOT
    / ".runs/build_multicall_first_passage_voyages/"
    "route_circuity_2021"
)

_TRANSITION_DISTANCE: dict[str, float] = {}
_TRANSITION_SAME_NODE: set[str] = set()
_DIRECT_DISTANCE: dict[str, float] = {}
_DIRECT_SAME_NODE: set[str] = set()
_SEPARATOR = ">"
_MAIN_THRESHOLD = 2.0
_SENSITIVITY_THRESHOLDS: tuple[float, ...] = ()


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
        newline="",
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


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
        newline="",
    ) as handle:
        frame.to_csv(
            handle,
            index=False,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


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


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(CODING_ROOT))
    except ValueError:
        return str(resolved)


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=CODING_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def candidate_city_chain(
    origin_city: str,
    intermediate_city_sequence: str,
    dest_city: str,
    separator: str,
) -> list[str]:
    intermediate = (
        intermediate_city_sequence.split(separator)
        if intermediate_city_sequence
        else []
    )
    return [origin_city, *intermediate, dest_city]


def transition_pair_ids(cities: list[str]) -> list[str]:
    return [
        f"{origin}__{destination}"
        for origin, destination in zip(cities[:-1], cities[1:])
    ]


def classify_route_circuity(
    direct_distance: float | None,
    direct_same_node: bool,
    transition_distances: list[float | None],
    transition_same_node: list[bool],
    expected_leg_count: int,
) -> tuple[float, float, str]:
    if len(transition_distances) != expected_leg_count:
        return math.nan, math.nan, "invalid_sequence"
    if direct_distance is None or not math.isfinite(direct_distance):
        return math.nan, math.nan, "missing_direct_route"
    if any(value is None or not math.isfinite(value) for value in transition_distances):
        return math.nan, math.nan, "missing_transition_route"

    route_distance = float(sum(float(value) for value in transition_distances))
    if direct_same_node or direct_distance == 0.0:
        return route_distance, math.nan, "direct_same_network_node"
    if any(transition_same_node) or any(
        float(value) == 0.0 for value in transition_distances
    ):
        return route_distance, math.nan, "same_network_node_transition"
    if direct_distance < 0.0 or route_distance < 0.0:
        return route_distance, math.nan, "invalid_negative_distance"
    return route_distance, route_distance / direct_distance, "complete_positive"


def validate_contract(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if int(config["analysis_year"]) != 2021:
        raise AssertionError("C3 route circuity is restricted to 2021")
    branch = git_value("branch", "--show-current")
    if branch != config["approved_branch"]:
        raise AssertionError(
            f"Expected branch {config['approved_branch']}; found {branch}"
        )
    workers = int(config["max_workers"])
    if workers < 1 or workers > 16:
        raise AssertionError("max_workers must remain between 1 and 16")
    if float(config["main_threshold"]) != 2.0:
        raise AssertionError("The frozen global-scale main threshold must be 2.0")

    inputs = {}
    for key in [
        "c2_output_manifest",
        "yardstick_table",
        "city_snap_table",
        "c0_node_routes",
        "c0_city_snap_config",
        "c0_city_snap_runner",
        "c0_route_config",
        "c0_route_runner",
    ]:
        path = CODING_ROOT / config[key]
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {path}")
        actual = sha256sum(path)
        expected = config[f"{key}_sha256"]
        if actual != expected:
            raise AssertionError(
                f"Frozen SHA-256 mismatch for {config[key]}: {actual}"
            )
        inputs[key] = {
            "path": config[key],
            "sha256": actual,
            "byte_size": path.stat().st_size,
        }
    return {
        "analysis_year": 2021,
        "branch": branch,
        "base_commit": git_value("rev-parse", "HEAD"),
        "config_path": display_path(config_path),
        "config_sha256": sha256sum(config_path),
        "runner_sha256": sha256sum(Path(__file__).resolve()),
        "inputs": inputs,
    }


def retained_candidate_parts(config: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = load_json(CODING_ROOT / config["c2_output_manifest"])
    parts = sorted(manifest["parts"], key=lambda item: int(item["task_id"]))
    if len(parts) != int(config["expected_candidate_parts"]):
        raise AssertionError("The retained candidate part count changed")
    total_rows = sum(int(item["retained_rows"]) for item in parts)
    if total_rows != int(config["expected_candidate_count"]):
        raise AssertionError("The retained candidate row count changed")
    for item in parts:
        path = CODING_ROOT / item["retained_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256sum(path) != item["retained_sha256"]:
            raise AssertionError(f"Retained candidate hash changed: {path}")
    return parts


def load_source_module(path: Path, module_name: str) -> Any:
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load source module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def enumerate_transition_usage(
    config: dict[str, Any],
    parts: list[dict[str, Any]],
) -> pd.DataFrame:
    usage: Counter[str] = Counter()
    separator = str(config["sequence_separator"])
    for item in parts:
        frame = pd.read_feather(
            CODING_ROOT / item["retained_path"],
            columns=[
                "origin_city",
                "dest_city",
                "intermediate_city_sequence",
                "n_intermediate_city_episodes",
            ],
        )
        for row in frame.itertuples(index=False):
            cities = candidate_city_chain(
                row.origin_city,
                row.intermediate_city_sequence or "",
                row.dest_city,
                separator,
            )
            pair_ids = transition_pair_ids(cities)
            expected = int(row.n_intermediate_city_episodes) + 1
            if len(pair_ids) != expected:
                raise AssertionError(
                    "Candidate city sequence does not match its intermediate "
                    "city-episode count"
                )
            usage.update(pair_ids)

    expected_occurrences = int(
        config["expected_candidate_transition_occurrences"]
    )
    if sum(usage.values()) != expected_occurrences:
        raise AssertionError(
            f"Expected {expected_occurrences} candidate transitions; "
            f"found {sum(usage.values())}"
        )
    expected_unique = int(config["expected_unique_transition_pairs"])
    if len(usage) != expected_unique:
        raise AssertionError(
            f"Expected {expected_unique} unique transitions; found {len(usage)}"
        )

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


def build_complete_city_snaps(
    config: dict[str, Any],
    usage: pd.DataFrame,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    snaps = pd.read_csv(CODING_ROOT / config["city_snap_table"])
    required_cities = sorted(
        set(usage["origin_city"]).union(usage["dest_city"])
    )
    if len(required_cities) != int(config["expected_required_cities"]):
        raise AssertionError("The required transition-city count changed")
    missing = sorted(set(required_cities).difference(snaps["city_id"]))
    if len(missing) != int(config["expected_supplementary_city_snaps"]):
        raise AssertionError(
            f"Expected {config['expected_supplementary_city_snaps']} "
            f"supplementary city snaps; found {len(missing)}"
        )

    supplementary = pd.DataFrame()
    lineage = pd.DataFrame()
    if missing:
        snap_module = load_source_module(
            CODING_ROOT / config["c0_city_snap_runner"],
            "c0_city_snap_auxiliary",
        )
        snap_config = load_json(CODING_ROOT / config["c0_city_snap_config"])
        snap_module.validate_contract(snap_config)
        coordinate_lookup, _, _ = snap_module.load_coordinate_lookup(
            snap_config["coordinate_source_precedence"]
        )
        calls = pd.read_feather(
            CODING_ROOT
            / snap_config["c0a_inputs"]["ordered_port_calls"]["path"],
            columns=snap_module.CALL_COLUMNS,
        )
        city_rows = []
        for city_id in missing:
            country_iso3, city_name = city_id.split(":", 1)
            city_rows.append(
                {
                    "city_id": city_id,
                    "country_iso3": country_iso3,
                    "city_name": city_name,
                }
            )
        coordinates, lineage, _ = snap_module.build_city_coordinates(
            calls,
            pd.DataFrame(city_rows),
            coordinate_lookup,
            float(snap_config["coordinate_spread_review_threshold_deg"]),
            [],
        )
        supplementary, _ = snap_module.add_network_snaps(
            coordinates,
            float(snap_config["snap_review_threshold_km"]),
        )
        write_csv_atomic(
            artifact_dir / "supplementary_city_snap_table_2021.csv",
            supplementary,
        )
        write_csv_atomic(
            artifact_dir / "supplementary_city_snap_lineage_2021.csv",
            lineage,
        )
        snaps = pd.concat([snaps, supplementary], ignore_index=True)

    if snaps["city_id"].duplicated().any():
        raise AssertionError("Combined city snap table has duplicate city IDs")
    if not set(required_cities).issubset(snaps["city_id"]):
        raise AssertionError("At least one required city lacks a network snap")
    return snaps, {
        "required_city_count": len(required_cities),
        "supplementary_city_snap_count": len(missing),
    }


def build_transition_table(
    config: dict[str, Any],
    parts: list[dict[str, Any]],
    artifact_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    usage = enumerate_transition_usage(config, parts)
    snaps, snap_summary = build_complete_city_snaps(
        config,
        usage,
        artifact_dir,
    )

    snap_columns = [
        "city_id",
        "network_node_id",
        "network_lon",
        "network_lat",
        "snap_distance_km",
        "routing_gateway_override",
    ]
    origin = snaps[snap_columns].rename(
        columns={
            "city_id": "origin_city",
            "network_node_id": "origin_network_node_id",
            "network_lon": "origin_network_lon",
            "network_lat": "origin_network_lat",
            "snap_distance_km": "snap_dist_o",
            "routing_gateway_override": "origin_routing_gateway_override",
        }
    )
    destination = snaps[snap_columns].rename(
        columns={
            "city_id": "dest_city",
            "network_node_id": "dest_network_node_id",
            "network_lon": "dest_network_lon",
            "network_lat": "dest_network_lat",
            "snap_distance_km": "snap_dist_d",
            "routing_gateway_override": "dest_routing_gateway_override",
        }
    )
    selected = usage.merge(
        origin,
        on="origin_city",
        how="left",
        validate="many_to_one",
    ).merge(
        destination,
        on="dest_city",
        how="left",
        validate="many_to_one",
    )
    route_fields = [
        "origin_network_node_id",
        "origin_network_lon",
        "origin_network_lat",
        "dest_network_node_id",
        "dest_network_lon",
        "dest_network_lat",
    ]
    if selected[route_fields].isna().any().any():
        raise AssertionError("At least one transition endpoint lacks a city snap")
    selected["node_pair_id"] = (
        selected["origin_network_node_id"]
        + "__"
        + selected["dest_network_node_id"]
    )
    selected["same_network_node"] = selected[
        "origin_network_node_id"
    ].eq(selected["dest_network_node_id"])

    yardstick = pd.read_feather(
        CODING_ROOT / config["yardstick_table"],
        columns=["pair_id", "d_base"],
    )
    selected = selected.merge(
        yardstick,
        on="pair_id",
        how="left",
        validate="one_to_one",
        indicator="city_pair_merge",
    )
    has_city_pair_route = selected["city_pair_merge"].eq("both")
    selected["route_source"] = np.where(
        has_city_pair_route,
        "c0_city_pair_route",
        "",
    )

    c0_node_routes = pd.read_feather(
        CODING_ROOT / config["c0_node_routes"],
        columns=["node_pair_id", "d_base"],
    ).rename(columns={"d_base": "c0_node_d_base"})
    selected = selected.merge(
        c0_node_routes,
        on="node_pair_id",
        how="left",
        validate="many_to_one",
    )
    use_c0_node = ~has_city_pair_route & selected["c0_node_d_base"].notna()
    selected.loc[use_c0_node, "d_base"] = selected.loc[
        use_c0_node, "c0_node_d_base"
    ]
    selected.loc[use_c0_node, "route_source"] = "c0_node_pair_route"

    needs_new_route = selected["d_base"].isna()
    new_node_pairs = (
        selected.loc[
            needs_new_route,
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
        .sort_values(
            ["origin_network_node_id", "dest_network_node_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if len(new_node_pairs) != int(
        config["expected_new_auxiliary_node_pair_routes"]
    ):
        raise AssertionError("The new auxiliary node-pair count changed")

    if not new_node_pairs.empty:
        route_module = load_source_module(
            CODING_ROOT / config["c0_route_runner"],
            "c0_route_auxiliary",
        )
        route_config = load_json(CODING_ROOT / config["c0_route_config"])
        route_module.validate_contract(
            route_config,
            CODING_ROOT / config["c0_route_config"],
        )
        route_dir = artifact_dir / "new_auxiliary_node_routes"
        input_path = route_dir / "node_pair_input.feather"
        output_path = route_dir / "node_pair_output.feather"
        failure_path = route_dir / "failure.json"
        write_feather_atomic(input_path, new_node_pairs)
        if not output_path.is_file():
            route_module.initialize_route_worker(
                {"base": route_config["regimes"]["base"]}
            )
            result = route_module.run_route_task(
                {
                    "task_id": "new_auxiliary_baseline_node_pairs",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "failure_path": str(failure_path),
                    "origin_count": int(
                        new_node_pairs["origin_network_node_id"].nunique()
                    ),
                    "node_pair_count": len(new_node_pairs),
                }
            )
            if result["status"] != "success":
                raise RuntimeError(
                    f"Auxiliary baseline routing failed: {result}"
                )
        auxiliary_routes = pd.read_feather(
            output_path,
            columns=["node_pair_id", "d_base"],
        ).rename(columns={"d_base": "auxiliary_d_base"})
        selected = selected.merge(
            auxiliary_routes,
            on="node_pair_id",
            how="left",
            validate="many_to_one",
        )
        use_auxiliary = selected["d_base"].isna() & selected[
            "auxiliary_d_base"
        ].notna()
        selected.loc[use_auxiliary, "d_base"] = selected.loc[
            use_auxiliary, "auxiliary_d_base"
        ]
        selected.loc[
            use_auxiliary, "route_source"
        ] = "new_auxiliary_node_route"
    else:
        use_auxiliary = pd.Series(False, index=selected.index)

    if selected["d_base"].isna().any():
        missing = selected.loc[selected["d_base"].isna(), "pair_id"].head()
        raise AssertionError(
            f"Required transition routes remain unresolved: {missing.tolist()}"
        )
    if selected["d_base"].lt(0).any():
        raise AssertionError("A required transition has negative baseline distance")
    selected["route_status"] = np.where(
        selected["same_network_node"],
        "same_network_node",
        "positive_baseline_route",
    )
    if (
        selected["same_network_node"] & selected["d_base"].ne(0.0)
    ).any():
        raise AssertionError("A same-node transition has nonzero distance")
    if (
        ~selected["same_network_node"] & selected["d_base"].le(0.0)
    ).any():
        raise AssertionError("A different-node transition has nonpositive distance")

    summary = {
        **snap_summary,
        "candidate_transition_occurrences": int(
            selected["candidate_transition_occurrences"].sum()
        ),
        "unique_transition_pairs": int(len(selected)),
        "c0_city_pair_routes_reused": int(has_city_pair_route.sum()),
        "c0_node_pair_routes_reused": int(use_c0_node.sum()),
        "new_auxiliary_city_pair_routes": int(use_auxiliary.sum()),
        "new_auxiliary_node_pair_routes": int(len(new_node_pairs)),
        "positive_transition_pairs": int(
            selected["route_status"].eq("positive_baseline_route").sum()
        ),
        "same_network_node_transition_pairs": int(
            selected["route_status"].eq("same_network_node").sum()
        ),
        "unresolved_transition_pairs": int(selected["d_base"].isna().sum()),
    }
    expected_keys = [
        "c0_city_pair_routes_reused",
        "c0_node_pair_routes_reused",
        "new_auxiliary_city_pair_routes",
        "new_auxiliary_node_pair_routes",
    ]
    for key in expected_keys:
        if summary[key] != int(config[f"expected_{key}"]):
            raise AssertionError(
                f"{key} changed: expected={config[f'expected_{key}']}, "
                f"observed={summary[key]}"
            )

    drop_columns = [
        "city_pair_merge",
        "c0_node_d_base",
        "auxiliary_d_base",
    ]
    selected = selected.drop(
        columns=[column for column in drop_columns if column in selected]
    ).sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    return selected, summary


def initialize_candidate_worker(
    transition_table_path: str,
    yardstick_path: str,
    separator: str,
    main_threshold: float,
    sensitivity_thresholds: list[float],
) -> None:
    global _TRANSITION_DISTANCE
    global _TRANSITION_SAME_NODE
    global _DIRECT_DISTANCE
    global _DIRECT_SAME_NODE
    global _SEPARATOR
    global _MAIN_THRESHOLD
    global _SENSITIVITY_THRESHOLDS

    transitions = pd.read_feather(transition_table_path)
    yardstick = pd.read_feather(
        yardstick_path,
        columns=["pair_id", "d_base", "same_network_node"],
    )
    _TRANSITION_DISTANCE = dict(
        zip(transitions["pair_id"], transitions["d_base"])
    )
    _TRANSITION_SAME_NODE = set(
        transitions.loc[transitions["same_network_node"], "pair_id"]
    )
    _DIRECT_DISTANCE = dict(zip(yardstick["pair_id"], yardstick["d_base"]))
    _DIRECT_SAME_NODE = set(
        yardstick.loc[yardstick["same_network_node"], "pair_id"]
    )
    _SEPARATOR = separator
    _MAIN_THRESHOLD = float(main_threshold)
    _SENSITIVITY_THRESHOLDS = tuple(
        float(value) for value in sensitivity_thresholds
    )


def process_candidate_part(task: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(task["input_path"])
    output_path = Path(task["output_path"])
    summary_path = Path(task["summary_path"])
    failure_path = Path(task["failure_path"])
    try:
        frame = pd.read_feather(
            input_path,
            columns=[
                "candidate_id",
                "pair_id",
                "origin_city",
                "dest_city",
                "intermediate_city_sequence",
                "n_observed_sea_legs",
                "n_intermediate_city_episodes",
            ],
        )
        route_distances = np.full(len(frame), np.nan, dtype=float)
        circuity = np.full(len(frame), np.nan, dtype=float)
        statuses: list[str] = []
        transition_usage: Counter[str] = Counter()

        for position, row in enumerate(frame.itertuples(index=False)):
            cities = candidate_city_chain(
                row.origin_city,
                row.intermediate_city_sequence or "",
                row.dest_city,
                _SEPARATOR,
            )
            pair_ids = transition_pair_ids(cities)
            transition_usage.update(pair_ids)
            distances = [_TRANSITION_DISTANCE.get(pair_id) for pair_id in pair_ids]
            same_node = [
                pair_id in _TRANSITION_SAME_NODE for pair_id in pair_ids
            ]
            route_distance, ratio, status = classify_route_circuity(
                _DIRECT_DISTANCE.get(row.pair_id),
                row.pair_id in _DIRECT_SAME_NODE,
                distances,
                same_node,
                int(row.n_intermediate_city_episodes) + 1,
            )
            route_distances[position] = route_distance
            circuity[position] = ratio
            statuses.append(status)

        output = frame[
            [
                "candidate_id",
                "pair_id",
                "n_observed_sea_legs",
                "n_intermediate_city_episodes",
            ]
        ].copy()
        output["route_sequence_d_base_km"] = route_distances
        output["direct_d_base_km"] = output["pair_id"].map(_DIRECT_DISTANCE)
        output["route_circuity_ratio"] = circuity
        output["route_circuity_status"] = statuses
        complete = output["route_circuity_status"].eq("complete_positive")
        output["exclude_route_circuity_gt_2"] = (
            complete & output["route_circuity_ratio"].gt(_MAIN_THRESHOLD)
        )
        for threshold in _SENSITIVITY_THRESHOLDS:
            label = str(threshold).replace(".", "_")
            output[f"exclude_route_circuity_gt_{label}"] = (
                complete & output["route_circuity_ratio"].gt(threshold)
            )
        write_feather_atomic(output_path, output)

        status_counts = output["route_circuity_status"].value_counts().to_dict()
        summary = {
            "task_id": task["task_id"],
            "status": "success",
            "candidate_rows": int(len(output)),
            "complete_positive_rows": int(complete.sum()),
            "excluded_main_rows": int(
                output["exclude_route_circuity_gt_2"].sum()
            ),
            "status_counts": {
                str(key): int(value) for key, value in status_counts.items()
            },
            "transition_usage": dict(sorted(transition_usage.items())),
            "output_path": display_path(output_path),
            "output_sha256": sha256sum(output_path),
        }
        write_json_atomic(summary_path, summary)
        if failure_path.exists():
            failure_path.unlink()
        return summary
    except Exception as error:
        failure = {
            "task_id": task["task_id"],
            "status": "failed",
            "expected_output_path": str(output_path),
            "error_class": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(failure_path, failure)
        return failure


def build_tasks(
    parts: list[dict[str, Any]],
    artifact_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records = []
    tasks = []
    for item in parts:
        task_id = f"part-{int(item['task_id']):03d}"
        task = {
            "task_id": task_id,
            "input_path": str((CODING_ROOT / item["retained_path"]).resolve()),
            "output_path": str(
                (
                    artifact_dir
                    / "candidate_route_circuity_2021"
                    / f"{task_id}.feather"
                ).resolve()
            ),
            "summary_path": str(
                (artifact_dir / "part_summaries" / f"{task_id}.json").resolve()
            ),
            "failure_path": str(
                (artifact_dir / "failures" / f"{task_id}.json").resolve()
            ),
            "expected_rows": int(item["retained_rows"]),
        }
        tasks.append(task)
        records.append(
            {
                "task_id": task_id,
                "expected_rows": task["expected_rows"],
                "input_path": display_path(Path(task["input_path"])),
                "expected_output_path": display_path(Path(task["output_path"])),
                "expected_summary_path": display_path(Path(task["summary_path"])),
            }
        )
    return pd.DataFrame(records), tasks


def aggregate_summaries(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    transition_table: pd.DataFrame,
    transition_summary: dict[str, int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    summaries = [load_json(Path(task["summary_path"])) for task in tasks]
    candidate_rows = sum(int(item["candidate_rows"]) for item in summaries)
    if candidate_rows != int(config["expected_candidate_count"]):
        raise AssertionError("Aggregated candidate row count changed")

    status_counts: Counter[str] = Counter()
    transition_usage: Counter[str] = Counter()
    for item in summaries:
        status_counts.update(item["status_counts"])
        transition_usage.update(item["transition_usage"])
    missing_usage = set(transition_usage).difference(transition_table["pair_id"])
    unused_table_pairs = set(transition_table["pair_id"]).difference(transition_usage)
    if missing_usage:
        raise AssertionError(
            f"Candidate sequences use missing transitions: {sorted(missing_usage)[:5]}"
        )
    if unused_table_pairs:
        raise AssertionError(
            f"Direct transition table contains unused pairs: {sorted(unused_table_pairs)[:5]}"
        )

    threshold_rows = []
    thresholds = [
        float(config["main_threshold"]),
        *[float(value) for value in config["sensitivity_thresholds"]],
    ]
    for threshold in sorted(set(thresholds)):
        excluded = 0
        complete = 0
        for task in tasks:
            output = pd.read_feather(
                Path(task["output_path"]),
                columns=["route_circuity_ratio", "route_circuity_status"],
            )
            valid = output["route_circuity_status"].eq("complete_positive")
            complete += int(valid.sum())
            excluded += int(
                (valid & output["route_circuity_ratio"].gt(threshold)).sum()
            )
        threshold_rows.append(
            {
                "threshold": threshold,
                "complete_positive_candidates": complete,
                "excluded_candidates": excluded,
                "excluded_share_of_complete": excluded / complete,
                "retained_candidates": candidate_rows - excluded,
                "unresolved_candidates": candidate_rows - complete,
            }
        )
    sensitivity = pd.DataFrame(threshold_rows).sort_values("threshold")
    summary = {
        "status": "pass",
        "analysis_year": 2021,
        "candidate_rows": candidate_rows,
        "candidate_parts": len(tasks),
        **transition_summary,
        "candidate_transition_occurrences": int(sum(transition_usage.values())),
        "status_counts": dict(sorted(status_counts.items())),
        "complete_positive_candidates": int(
            status_counts["complete_positive"]
        ),
        "unresolved_candidates": int(
            candidate_rows - status_counts["complete_positive"]
        ),
        "main_threshold": float(config["main_threshold"]),
        "main_excluded_candidates": int(
            sensitivity.loc[
                sensitivity["threshold"].eq(float(config["main_threshold"])),
                "excluded_candidates",
            ].iloc[0]
        ),
        "hard_boundaries": {
            "years_read": [2021],
            "ais_read": False,
            "new_auxiliary_baseline_node_routes": int(
                transition_summary["new_auxiliary_node_pair_routes"]
            ),
            "unresolved_auto_excluded": False,
            "c4_run": False,
        },
    }
    return summary, sensitivity


def write_audit_report(path: Path, summary: dict[str, Any]) -> None:
    status_lines = [
        f"- `{key}`: {value:,}"
        for key, value in summary["status_counts"].items()
    ]
    lines = [
        "# C3.1 Full Route-Circuity Audit",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- C2-retained candidates: {summary['candidate_rows']:,}",
        f"- Required directed transitions: {summary['unique_transition_pairs']:,}",
        (
            f"- Transition distances reused from C0 city-pair routes: "
            f"{summary['c0_city_pair_routes_reused']:,}"
        ),
        (
            f"- Additional transitions reused from C0 node-pair routes: "
            f"{summary['c0_node_pair_routes_reused']:,}"
        ),
        (
            f"- Transition city pairs covered by new auxiliary node routes: "
            f"{summary['new_auxiliary_city_pair_routes']:,}"
        ),
        (
            f"- New auxiliary baseline node routes calculated: "
            f"{summary['new_auxiliary_node_pair_routes']:,}"
        ),
        (
            f"- Supplementary city snaps built under the C0 contract: "
            f"{summary['supplementary_city_snap_count']:,}"
        ),
        (
            f"- Required transition pairs unresolved after routing: "
            f"{summary['unresolved_transition_pairs']:,}"
        ),
        "",
        "## Candidate Status",
        "",
        *status_lines,
        "",
        "## Main Global-Scale Rule",
        "",
        (
            f"- Threshold: route_circuity_ratio > "
            f"{summary['main_threshold']:.1f}"
        ),
        (
            f"- Complete positive candidates excluded: "
            f"{summary['main_excluded_candidates']:,}"
        ),
        (
            f"- Unresolved candidates retained without automatic exclusion: "
            f"{summary['unresolved_candidates']:,}"
        ),
        "",
        "## Interpretation",
        "",
        "- Most transition distances reuse a frozen C0 city-pair or node-pair route.",
        "- Only node pairs absent from C0 are newly routed, under the identical",
        "  baseline SeaRoute contract and frozen city-coordinate method.",
        "- Zero-distance same-network-node cases remain explicitly unresolved.",
        "",
        "C3.1 route circuity ends here. AIS parameter fitting remains a separate step.",
    ]
    write_text_atomic(path, "\n".join(lines) + "\n")


def run(
    config_path: Path,
    artifact_dir: Path,
    report_dir: Path,
    run_dir: Path,
    max_workers_override: int | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    artifact_dir = artifact_dir.resolve()
    report_dir = report_dir.resolve()
    run_dir = run_dir.resolve()
    config = load_json(config_path)
    provenance = validate_contract(config, config_path)
    parts = retained_candidate_parts(config)
    max_workers = (
        int(max_workers_override)
        if max_workers_override is not None
        else int(config["max_workers"])
    )
    if max_workers < 1 or max_workers > 16:
        raise AssertionError("Runtime max_workers must remain between 1 and 16")

    contract = {
        "analysis_year": 2021,
        "config_sha256": provenance["config_sha256"],
        "runner_sha256": provenance["runner_sha256"],
        "inputs": provenance["inputs"],
        "main_threshold": float(config["main_threshold"]),
        "sensitivity_thresholds": config["sensitivity_thresholds"],
    }
    contract_path = artifact_dir / "execution_contract.json"
    if contract_path.exists():
        if load_json(contract_path) != contract:
            raise AssertionError(
                "Existing artifacts use a different execution contract"
            )
    else:
        write_json_atomic(contract_path, contract)

    transition_table, transition_summary = build_transition_table(
        config,
        parts,
        artifact_dir,
    )
    transition_path = (
        artifact_dir / "auxiliary_baseline_transition_routes_2021.feather"
    )
    write_feather_atomic(transition_path, transition_table)

    task_manifest, tasks = build_tasks(parts, artifact_dir)
    write_csv_atomic(run_dir / "full_task_list.csv", task_manifest)
    write_json_atomic(
        run_dir / "run_configuration.json",
        {
            **contract,
            "artifact_dir": display_path(artifact_dir),
            "report_dir": display_path(report_dir),
            "run_dir": display_path(run_dir),
            "max_workers": max_workers,
            **transition_summary,
        },
    )
    pending = [
        task
        for task in tasks
        if not (
            Path(task["output_path"]).is_file()
            and Path(task["summary_path"]).is_file()
        )
    ]
    skipped = [task for task in tasks if task not in pending]
    write_csv_atomic(
        run_dir / "pending_task_list.csv",
        task_manifest.loc[task_manifest["task_id"].isin(
            [task["task_id"] for task in pending]
        )],
    )
    write_csv_atomic(
        run_dir / "skipped_existing_output_task_list.csv",
        task_manifest.loc[task_manifest["task_id"].isin(
            [task["task_id"] for task in skipped]
        )],
    )

    results = []
    if pending:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initialize_candidate_worker,
            initargs=(
                str(transition_path),
                str((CODING_ROOT / config["yardstick_table"]).resolve()),
                config["sequence_separator"],
                float(config["main_threshold"]),
                config["sensitivity_thresholds"],
            ),
        ) as executor:
            futures = {
                executor.submit(process_candidate_part, task): task
                for task in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[C3.1] completed {completed}/{len(pending)} "
                    f"{result['task_id']} status={result['status']}",
                    flush=True,
                )
    failures = [result for result in results if result["status"] != "success"]
    write_json_atomic(run_dir / "failed_task_list.json", failures)
    if failures:
        raise RuntimeError(f"C3.1 stopped with {len(failures)} failed tasks")
    if any(
        not Path(task["output_path"]).is_file()
        or not Path(task["summary_path"]).is_file()
        for task in tasks
    ):
        raise AssertionError("At least one expected task output is missing")

    summary, sensitivity = aggregate_summaries(
        config,
        tasks,
        transition_table,
        transition_summary,
    )
    summary.update(
        {
            "branch": provenance["branch"],
            "base_commit": provenance["base_commit"],
            "config_sha256": provenance["config_sha256"],
            "runner_sha256": provenance["runner_sha256"],
            "max_workers": max_workers,
            "resumed_skipped_task_count": len(skipped),
            "transition_table_path": display_path(transition_path),
            "transition_table_sha256": sha256sum(transition_path),
        }
    )
    summary_path = report_dir / "c3_1_route_circuity_full_summary.json"
    sensitivity_path = (
        report_dir / "c3_1_route_circuity_full_threshold_sensitivity.csv"
    )
    audit_path = report_dir / "c3_1_route_circuity_full_audit.md"
    write_json_atomic(summary_path, summary)
    write_csv_atomic(sensitivity_path, sensitivity)
    write_audit_report(audit_path, summary)

    outputs = [transition_path, summary_path, sensitivity_path, audit_path]
    for task in tasks:
        outputs.extend(
            [Path(task["output_path"]), Path(task["summary_path"])]
        )
    manifest = {
        "status": "pass",
        "outputs": {
            display_path(path): {
                "byte_size": path.stat().st_size,
                "sha256": sha256sum(path),
            }
            for path in outputs
        },
    }
    write_json_atomic(artifact_dir / "c3_1_output_manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build baseline transition distances and route circuity for every "
            "C2-retained 2021 candidate."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-workers", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = run(
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
        run_dir=(
            arguments.run_dir
            if arguments.run_dir.is_absolute()
            else CODING_ROOT / arguments.run_dir
        ),
        max_workers_override=arguments.max_workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
