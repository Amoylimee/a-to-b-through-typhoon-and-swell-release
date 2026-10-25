#!/usr/bin/env python3
"""Build the C0.4 full 2021 pair-by-regime route yardstick table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import searoute as sr
from searoute.classes.passages import Passage
from searoute.utils import distance_length


CODING_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = PIPELINE_ROOT / "config/frozen/pair_routes_2021.json"
DEFAULT_ARTIFACT_DIR = DATA_ROOT / "main/support/frozen_2021"
DEFAULT_REPORT_DIR = DATA_ROOT / "main/validation/pair_routes"
DEFAULT_RUN_DIR = (
    CODING_ROOT
    / ".runs/build_multicall_first_passage_voyages/"
    "pair_routes_2021"
)

_WORKER_GRAPH: Any = None
_WORKER_VIEWS: dict[str, Any] = {}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=CODING_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(CODING_ROOT))
    except ValueError:
        return str(resolved)


def validate_contract(
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    if config["analysis_year"] != 2021:
        raise AssertionError("C0.4 is restricted to the 2021 development year")
    branch = git_value("branch", "--show-current")
    if branch != config["approved_branch"]:
        raise AssertionError(
            f"Expected branch {config['approved_branch']}; found {branch}"
        )
    if sr.__version__ != config["searoute"]["version"]:
        raise AssertionError(
            f"Expected searoute {config['searoute']['version']}; "
            f"found {sr.__version__}"
        )
    package_root = Path(inspect.getfile(sr)).resolve().parent
    marnet_path = package_root / "data/marnet_dict.py"
    marnet_hash = sha256sum(marnet_path)
    if marnet_hash != config["searoute"]["marnet_sha256"]:
        raise AssertionError("Frozen searoute maritime-network hash changed")

    valid_tokens = {
        token for token in Passage.valid_passages() if isinstance(token, str)
    }
    for regime, restrictions in config["regimes"].items():
        invalid = sorted(set(restrictions).difference(valid_tokens))
        if invalid:
            raise AssertionError(
                f"Invalid restriction tokens for {regime}: {invalid}"
            )

    frozen_inputs = []
    for name, specification in config["frozen_inputs"].items():
        path_string = specification["path"]
        if "/2025/" in path_string or "/ais" in path_string.lower():
            raise AssertionError(
                f"Unauthorized 2025 or AIS input entered C0.4: {path_string}"
            )
        path = CODING_ROOT / path_string
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {path_string}")
        actual = sha256sum(path)
        if actual != specification["sha256"]:
            raise AssertionError(
                f"Frozen SHA-256 mismatch for {path_string}: {actual}"
            )
        frozen_inputs.append(
            {
                "name": name,
                "path": path_string,
                "byte_size": path.stat().st_size,
                "sha256": actual,
            }
        )

    workers = int(config["processing"]["max_workers"])
    if workers < 1 or workers > 16:
        raise AssertionError("C0.4 max_workers must remain between 1 and 16")
    task_size = int(config["processing"]["origin_nodes_per_task"])
    if task_size < 1:
        raise AssertionError("origin_nodes_per_task must be positive")

    return {
        "branch": branch,
        "base_commit": git_value("rev-parse", "HEAD"),
        "config_path": display_path(config_path),
        "config_sha256": sha256sum(config_path),
        "runner_sha256": sha256sum(Path(__file__).resolve()),
        "frozen_inputs": frozen_inputs,
        "marnet_path": str(marnet_path),
        "marnet_sha256": marnet_hash,
        "searoute_version": sr.__version__,
    }


def build_pair_and_node_manifests(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    pair_path = CODING_ROOT / config["frozen_inputs"]["pair_manifest"]["path"]
    pairs = pd.read_csv(
        pair_path,
        usecols=[
            "pair_id",
            "origin_city",
            "dest_city",
            "candidate_count_2021",
            "vessel_count_2021",
        ],
    )
    if pairs["pair_id"].duplicated().any():
        raise AssertionError("Frozen pair manifest contains duplicate pair_id")
    expected_pair_ids = pairs["origin_city"] + "__" + pairs["dest_city"]
    if not pairs["pair_id"].eq(expected_pair_ids).all():
        raise AssertionError("pair_id does not match its endpoint city fields")

    city_path = CODING_ROOT / config["frozen_inputs"]["city_snap_table"]["path"]
    cities = pd.read_csv(
        city_path,
        usecols=[
            "city_id",
            "network_node_id",
            "network_lon",
            "network_lat",
            "snap_distance_km",
            "routing_gateway_override",
        ],
    )
    if cities["city_id"].duplicated().any():
        raise AssertionError("City snap table contains duplicate city_id")
    if cities[
        ["network_node_id", "network_lon", "network_lat", "snap_distance_km"]
    ].isna().any().any():
        raise AssertionError("City snap table contains missing route fields")

    origin = cities.rename(
        columns={
            "city_id": "origin_city",
            "network_node_id": "origin_network_node_id",
            "network_lon": "origin_network_lon",
            "network_lat": "origin_network_lat",
            "snap_distance_km": "snap_dist_o",
            "routing_gateway_override": "origin_routing_gateway_override",
        }
    )
    destination = cities.rename(
        columns={
            "city_id": "dest_city",
            "network_node_id": "dest_network_node_id",
            "network_lon": "dest_network_lon",
            "network_lat": "dest_network_lat",
            "snap_distance_km": "snap_dist_d",
            "routing_gateway_override": "dest_routing_gateway_override",
        }
    )
    pairs = (
        pairs.merge(origin, on="origin_city", how="left", validate="many_to_one")
        .merge(
            destination,
            on="dest_city",
            how="left",
            validate="many_to_one",
        )
        .sort_values("pair_id", kind="mergesort")
        .reset_index(drop=True)
    )
    route_fields = [
        "origin_network_node_id",
        "origin_network_lon",
        "origin_network_lat",
        "dest_network_node_id",
        "dest_network_lon",
        "dest_network_lat",
    ]
    if pairs[route_fields].isna().any().any():
        raise AssertionError("At least one pair endpoint lacks a city snap")

    pairs["node_pair_id"] = (
        pairs["origin_network_node_id"]
        + "__"
        + pairs["dest_network_node_id"]
    )
    pairs["same_network_node"] = pairs["origin_network_node_id"].eq(
        pairs["dest_network_node_id"]
    )
    node_pairs = (
        pairs[
            [
                "node_pair_id",
                "origin_network_node_id",
                "origin_network_lon",
                "origin_network_lat",
                "dest_network_node_id",
                "dest_network_lon",
                "dest_network_lat",
            ]
        ]
        .drop_duplicates("node_pair_id")
        .sort_values(
            ["origin_network_node_id", "dest_network_node_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if node_pairs["node_pair_id"].duplicated().any():
        raise AssertionError("Node-pair manifest contains duplicate keys")

    observed_counts = {
        "pair_count": int(len(pairs)),
        "city_count": int(len(cities)),
        "unique_network_node_count": int(cities["network_node_id"].nunique()),
        "unique_origin_network_node_count": int(
            pairs["origin_network_node_id"].nunique()
        ),
        "unique_node_pair_count": int(len(node_pairs)),
        "same_network_node_pair_count": int(
            pairs["same_network_node"].sum()
        ),
    }
    expected_counts = {
        key: int(value)
        for key, value in config["expected_counts"].items()
    }
    if observed_counts != expected_counts:
        raise AssertionError(
            "Frozen C0.4 counts changed: "
            f"observed={observed_counts}, expected={expected_counts}"
        )
    return pairs, node_pairs, observed_counts


def build_task_manifest(
    node_pairs: pd.DataFrame,
    artifact_dir: Path,
    origin_nodes_per_task: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    origins = sorted(node_pairs["origin_network_node_id"].unique().tolist())
    task_inputs = artifact_dir / "task_inputs"
    task_outputs = artifact_dir / "chunks"
    task_failures = artifact_dir / "failures"
    records = []
    tasks = []
    for offset in range(0, len(origins), origin_nodes_per_task):
        task_number = offset // origin_nodes_per_task
        task_id = f"origin_nodes_{task_number:05d}"
        selected_origins = origins[offset : offset + origin_nodes_per_task]
        task_frame = node_pairs.loc[
            node_pairs["origin_network_node_id"].isin(selected_origins)
        ].copy()
        input_path = task_inputs / f"{task_id}.feather"
        output_path = task_outputs / f"{task_id}.feather"
        failure_path = task_failures / f"{task_id}.json"
        write_feather_atomic(input_path, task_frame)
        task = {
            "task_id": task_id,
            "input_path": str(input_path.resolve()),
            "output_path": str(output_path.resolve()),
            "failure_path": str(failure_path.resolve()),
            "origin_count": len(selected_origins),
            "node_pair_count": len(task_frame),
        }
        tasks.append(task)
        records.append(
            {
                "task_id": task_id,
                "origin_count": len(selected_origins),
                "node_pair_count": len(task_frame),
                "first_origin_network_node_id": selected_origins[0],
                "last_origin_network_node_id": selected_origins[-1],
                "input_path": display_path(input_path),
                "expected_output_path": display_path(output_path),
            }
        )
    return pd.DataFrame(records), tasks


def initialize_route_worker(regimes: dict[str, list[str]]) -> None:
    global _WORKER_GRAPH, _WORKER_VIEWS
    _WORKER_GRAPH, _ = sr.get_graphs(backend="networkx")
    views = {}
    for regime, restrictions in regimes.items():
        blocked = frozenset(restrictions)

        def edge_allowed(
            origin: tuple[float, float],
            destination: tuple[float, float],
            blocked_passages: frozenset[str] = blocked,
        ) -> bool:
            return (
                _WORKER_GRAPH[origin][destination].get("passage")
                not in blocked_passages
            )

        views[regime] = nx.subgraph_view(
            _WORKER_GRAPH,
            filter_edge=edge_allowed,
        )
    _WORKER_VIEWS = views


def exact_path_distance_km(path: list[tuple[float, float]]) -> float:
    if len(path) <= 1:
        return 0.0
    return float(distance_length(path, units="km"))


def run_route_task(task: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(task["input_path"])
    output_path = Path(task["output_path"])
    failure_path = Path(task["failure_path"])
    try:
        frame = pd.read_feather(input_path)
        regimes = list(_WORKER_VIEWS)
        for regime in regimes:
            frame[f"d_{regime}"] = np.nan
            frame[f"feasible_{regime}"] = False

        for _, source_group in frame.groupby(
            "origin_network_node_id",
            sort=False,
        ):
            source = (
                float(source_group["origin_network_lon"].iloc[0]),
                float(source_group["origin_network_lat"].iloc[0]),
            )
            row_indices = source_group.index.to_numpy()
            destinations = [
                (float(lon), float(lat))
                for lon, lat in source_group[
                    ["dest_network_lon", "dest_network_lat"]
                ].itertuples(index=False, name=None)
            ]
            for regime, graph_view in _WORKER_VIEWS.items():
                _, paths = nx.single_source_dijkstra(
                    graph_view,
                    source=source,
                    weight="weight",
                )
                distances = []
                feasible = []
                for destination in destinations:
                    path = paths.get(destination)
                    if path is None:
                        distances.append(math.nan)
                        feasible.append(False)
                    else:
                        distances.append(exact_path_distance_km(path))
                        feasible.append(True)
                frame.loc[row_indices, f"d_{regime}"] = distances
                frame.loc[row_indices, f"feasible_{regime}"] = feasible

        write_feather_atomic(output_path, frame)
        if failure_path.exists():
            failure_path.unlink()
        return {
            "task_id": task["task_id"],
            "status": "success",
            "origin_count": int(task["origin_count"]),
            "node_pair_count": int(len(frame)),
            "output_path": str(output_path),
            "output_sha256": sha256sum(output_path),
        }
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


def add_yardstick_fields(
    node_routes: pd.DataFrame,
    regimes: list[str],
) -> pd.DataFrame:
    output = node_routes.copy()
    distance_columns = [f"d_{regime}" for regime in regimes]
    distances = output[distance_columns].to_numpy(dtype=float)
    feasible_any = np.isfinite(distances).any(axis=1)
    yardstick = np.full(len(output), np.nan, dtype=float)
    yardstick[feasible_any] = np.nanmax(
        distances[feasible_any],
        axis=1,
    )
    output["d_yardstick"] = yardstick
    regime_used = []
    flags = []
    for values, maximum in zip(distances, yardstick):
        feasible = np.isfinite(values)
        flags.append(
            json.dumps(
                {
                    regime: bool(is_feasible)
                    for regime, is_feasible in zip(regimes, feasible)
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        if not math.isfinite(maximum):
            regime_used.append("")
            continue
        tied = [
            regime
            for regime, value in zip(regimes, values)
            if math.isfinite(value)
            and math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-9)
        ]
        regime_used.append(";".join(tied))
    output["regime_used"] = regime_used
    output["feasible_flags"] = flags
    output["same_network_node"] = output["origin_network_node_id"].eq(
        output["dest_network_node_id"]
    )
    return output


def build_yardstick_table(
    pairs: pd.DataFrame,
    node_routes: pd.DataFrame,
) -> pd.DataFrame:
    route_columns = [
        "node_pair_id",
        "d_base",
        "d_canal",
        "d_redsea",
        "d_yardstick",
        "regime_used",
        "feasible_flags",
    ]
    output = pairs.merge(
        node_routes[route_columns],
        on="node_pair_id",
        how="left",
        validate="many_to_one",
    )
    columns = [
        "pair_id",
        "origin_city",
        "dest_city",
        "candidate_count_2021",
        "vessel_count_2021",
        "d_base",
        "d_canal",
        "d_redsea",
        "d_yardstick",
        "regime_used",
        "snap_dist_o",
        "snap_dist_d",
        "feasible_flags",
        "same_network_node",
        "origin_network_node_id",
        "dest_network_node_id",
        "origin_routing_gateway_override",
        "dest_routing_gateway_override",
    ]
    return output[columns].sort_values(
        "pair_id",
        kind="mergesort",
    ).reset_index(drop=True)


def validate_yardstick_table(
    config: dict[str, Any],
    yardstick: pd.DataFrame,
) -> dict[str, int]:
    expected = int(config["expected_counts"]["pair_count"])
    if len(yardstick) != expected:
        raise AssertionError(
            f"Expected {expected} yardstick rows; found {len(yardstick)}"
        )
    if yardstick["pair_id"].duplicated().any():
        raise AssertionError("Yardstick table contains duplicate pair_id")
    distances = yardstick[["d_base", "d_canal", "d_redsea"]]
    calculated = distances.max(axis=1, skipna=True)
    mismatch = ~np.isclose(
        yardstick["d_yardstick"].to_numpy(dtype=float),
        calculated.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
        equal_nan=True,
    )
    if mismatch.any():
        raise AssertionError("d_yardstick is not the regime outer maximum")
    if (distances < 0).any().any():
        raise AssertionError("Negative route distance found")
    zero_distance = distances.eq(0).any(axis=1)
    if (zero_distance & ~yardstick["same_network_node"]).any():
        raise AssertionError("Zero route distance found across different nodes")
    same_count = int(yardstick["same_network_node"].sum())
    expected_same = int(
        config["expected_counts"]["same_network_node_pair_count"]
    )
    if same_count != expected_same:
        raise AssertionError(
            f"Expected {expected_same} same-node pairs; found {same_count}"
        )
    return {
        "pair_count": int(len(yardstick)),
        "fully_feasible_pair_count": int(distances.notna().all(axis=1).sum()),
        "partly_feasible_pair_count": int(
            (distances.notna().any(axis=1) & ~distances.notna().all(axis=1)).sum()
        ),
        "fully_infeasible_pair_count": int(distances.isna().all(axis=1).sum()),
        "same_network_node_pair_count": same_count,
        "same_network_node_candidate_count": int(
            yardstick.loc[
                yardstick["same_network_node"],
                "candidate_count_2021",
            ].sum()
        ),
    }


def compare_with_c0_2_benchmark(
    config: dict[str, Any],
    yardstick: pd.DataFrame,
) -> pd.DataFrame:
    path = (
        CODING_ROOT
        / config["frozen_inputs"]["benchmark_yardstick"]["path"]
    )
    benchmark = pd.read_csv(
        path,
        usecols=[
            "pair_id",
            "d_base_km",
            "d_canal_km",
            "d_redsea_km",
            "d_yardstick_km",
        ],
    ).rename(
        columns={
            "d_base_km": "benchmark_d_base",
            "d_canal_km": "benchmark_d_canal",
            "d_redsea_km": "benchmark_d_redsea",
            "d_yardstick_km": "benchmark_d_yardstick",
        }
    )
    selected = benchmark.merge(
        yardstick[
            [
                "pair_id",
                "d_base",
                "d_canal",
                "d_redsea",
                "d_yardstick",
            ]
        ],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    comparisons = []
    for row in selected.itertuples(index=False):
        values = row._asdict()
        record = {"pair_id": row.pair_id}
        passed = True
        for regime in ["base", "canal", "redsea", "yardstick"]:
            expected = float(values[f"benchmark_d_{regime}"])
            actual = float(values[f"d_{regime}"])
            difference = abs(actual - expected)
            record[f"benchmark_d_{regime}"] = expected
            record[f"full_d_{regime}"] = actual
            record[f"absolute_difference_{regime}"] = difference
            passed = passed and difference <= 1e-9
        record["status"] = "pass" if passed else "fail"
        comparisons.append(record)
    output = pd.DataFrame(comparisons).sort_values(
        "pair_id",
        kind="mergesort",
    )
    if len(output) != 30 or not output["status"].eq("pass").all():
        raise AssertionError("C0.4 does not exactly reproduce C0.2 benchmark")
    return output.reset_index(drop=True)


def write_audit_report(
    path: Path,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# C0.4 Full-Pair Route Audit",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- 2021 directed city pairs: {summary['pair_count']:,}",
        f"- Unique network-node pairs routed: {summary['unique_node_pair_count']:,}",
        f"- Fully feasible under all three regimes: {summary['fully_feasible_pair_count']:,}",
        f"- Partly feasible: {summary['partly_feasible_pair_count']:,}",
        f"- Fully infeasible: {summary['fully_infeasible_pair_count']:,}",
        f"- C0.2 benchmark rows reproduced exactly: {summary['benchmark_exact_match_count']}/30",
        "",
        "## Same-Node Diagnostic",
        "",
        (
            f"- {summary['same_network_node_pair_count']:,} different city pairs "
            "snap to the same maritime-network node."
        ),
        (
            f"- They represent {summary['same_network_node_candidate_count']:,} "
            "2021 candidate passages."
        ),
        "- Their network distance is retained as 0.0 km and explicitly flagged.",
        "- No connector distance or substitute short-route distance was invented.",
        "",
        "## Boundaries",
        "",
        "- Inputs are restricted to frozen 2021 C0A and C0.3 products.",
        "- No AIS or 2025 data were read.",
        "- No C0.5 comparison or C1 candidate filtering was run.",
        "",
        "C0.4 ends here. Continue only after review of this output.",
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
    max_workers = (
        int(max_workers_override)
        if max_workers_override is not None
        else int(config["processing"]["max_workers"])
    )
    if max_workers < 1 or max_workers > 16:
        raise AssertionError("Runtime max_workers must remain between 1 and 16")

    contract = {
        "analysis_year": 2021,
        "config_sha256": provenance["config_sha256"],
        "runner_sha256": provenance["runner_sha256"],
        "frozen_inputs": provenance["frozen_inputs"],
        "marnet_sha256": provenance["marnet_sha256"],
        "searoute_version": provenance["searoute_version"],
        "regimes": config["regimes"],
    }
    contract_path = artifact_dir / "execution_contract.json"
    if contract_path.exists():
        existing_contract = load_json(contract_path)
        if existing_contract != contract:
            raise AssertionError(
                "Existing C0.4 artifact directory has a different execution "
                "contract; use a new artifact directory"
            )
    else:
        write_json_atomic(contract_path, contract)

    pairs, node_pairs, counts = build_pair_and_node_manifests(config)
    node_pair_path = artifact_dir / "node_pair_manifest_2021.feather"
    write_feather_atomic(node_pair_path, node_pairs)
    task_manifest, tasks = build_task_manifest(
        node_pairs,
        artifact_dir,
        int(config["processing"]["origin_nodes_per_task"]),
    )
    write_csv_atomic(run_dir / "full_task_list.csv", task_manifest)
    write_json_atomic(
        run_dir / "run_configuration.json",
        {
            **contract,
            "artifact_dir": display_path(artifact_dir),
            "report_dir": display_path(report_dir),
            "run_dir": display_path(run_dir),
            "max_workers": max_workers,
            "origin_nodes_per_task": int(
                config["processing"]["origin_nodes_per_task"]
            ),
            "task_count": len(tasks),
        },
    )

    pending = [
        task for task in tasks if not Path(task["output_path"]).is_file()
    ]
    skipped = [
        task for task in tasks if Path(task["output_path"]).is_file()
    ]
    manifest_columns = [
        "task_id",
        "origin_count",
        "node_pair_count",
        "output_path",
    ]
    write_csv_atomic(
        run_dir / "pending_task_list.csv",
        pd.DataFrame(pending, columns=manifest_columns),
    )
    write_csv_atomic(
        run_dir / "skipped_existing_output_task_list.csv",
        pd.DataFrame(skipped, columns=manifest_columns),
    )

    task_results = []
    if pending:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initialize_route_worker,
            initargs=(config["regimes"],),
        ) as executor:
            futures = {
                executor.submit(run_route_task, task): task
                for task in pending
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                task_results.append(result)
                completed += 1
                print(
                    f"[C0.4] completed {completed}/{len(pending)} "
                    f"{result['task_id']} status={result['status']}",
                    flush=True,
                )
    failures = [
        result for result in task_results if result["status"] != "success"
    ]
    write_json_atomic(run_dir / "failed_task_list.json", failures)
    if failures:
        raise RuntimeError(
            f"C0.4 stopped with {len(failures)} failed route tasks"
        )

    missing_outputs = [
        task["task_id"]
        for task in tasks
        if not Path(task["output_path"]).is_file()
    ]
    if missing_outputs:
        raise AssertionError(
            f"Expected task outputs are missing: {missing_outputs[:10]}"
        )

    chunks = [
        pd.read_feather(Path(task["output_path"]))
        for task in sorted(tasks, key=lambda item: item["task_id"])
    ]
    node_routes = pd.concat(chunks, ignore_index=True)
    node_routes = node_routes.sort_values(
        ["origin_network_node_id", "dest_network_node_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(node_routes) != counts["unique_node_pair_count"]:
        raise AssertionError("Aggregated node-route row count changed")
    if node_routes["node_pair_id"].duplicated().any():
        raise AssertionError("Aggregated node routes contain duplicate keys")
    node_routes = add_yardstick_fields(
        node_routes,
        list(config["regimes"]),
    )
    node_routes_path = artifact_dir / "node_pair_routes_2021.feather"
    write_feather_atomic(node_routes_path, node_routes)

    yardstick = build_yardstick_table(pairs, node_routes)
    qa = validate_yardstick_table(config, yardstick)
    yardstick_path = artifact_dir / "yardstick_table_2021.feather"
    write_feather_atomic(yardstick_path, yardstick)
    benchmark_check = compare_with_c0_2_benchmark(config, yardstick)
    benchmark_path = report_dir / "c0_4_benchmark_replay_check.csv"
    write_csv_atomic(benchmark_path, benchmark_check)

    regime_counts = (
        yardstick["regime_used"]
        .value_counts(dropna=False)
        .rename_axis("regime_used")
        .reset_index(name="pair_count")
        .sort_values("regime_used", kind="mergesort")
    )
    regime_counts_path = report_dir / "c0_4_regime_counts.csv"
    write_csv_atomic(regime_counts_path, regime_counts)

    same_node_sample = (
        yardstick.loc[yardstick["same_network_node"]]
        .sort_values(
            ["candidate_count_2021", "pair_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(200)
    )
    same_node_path = report_dir / "c0_4_same_network_node_top200.csv"
    write_csv_atomic(same_node_path, same_node_sample)

    summary = {
        "status": "pass",
        "analysis_year": 2021,
        "branch": provenance["branch"],
        "base_commit": provenance["base_commit"],
        "config_sha256": provenance["config_sha256"],
        "runner_sha256": provenance["runner_sha256"],
        "searoute_version": provenance["searoute_version"],
        "marnet_sha256": provenance["marnet_sha256"],
        "pair_count": qa["pair_count"],
        "city_count": counts["city_count"],
        "unique_network_node_count": counts["unique_network_node_count"],
        "unique_origin_network_node_count": counts[
            "unique_origin_network_node_count"
        ],
        "unique_node_pair_count": counts["unique_node_pair_count"],
        "fully_feasible_pair_count": qa["fully_feasible_pair_count"],
        "partly_feasible_pair_count": qa["partly_feasible_pair_count"],
        "fully_infeasible_pair_count": qa["fully_infeasible_pair_count"],
        "same_network_node_pair_count": qa[
            "same_network_node_pair_count"
        ],
        "same_network_node_candidate_count": qa[
            "same_network_node_candidate_count"
        ],
        "benchmark_exact_match_count": int(
            benchmark_check["status"].eq("pass").sum()
        ),
        "task_count": len(tasks),
        "max_workers": max_workers,
        "resumed_skipped_task_count": len(skipped),
        "node_pair_routes_path": display_path(node_routes_path),
        "node_pair_routes_sha256": sha256sum(node_routes_path),
        "yardstick_table_path": display_path(yardstick_path),
        "yardstick_table_sha256": sha256sum(yardstick_path),
        "pair_id_set_sha256": object_sha256(
            sorted(yardstick["pair_id"].tolist())
        ),
        "hard_boundaries": {
            "years_read": [2021],
            "ais_read": False,
            "c0_5_run": False,
            "c1_run": False,
        },
    }
    summary_path = report_dir / "c0_4_full_route_summary.json"
    write_json_atomic(summary_path, summary)
    audit_path = report_dir / "c0_4_full_route_audit.md"
    write_audit_report(audit_path, summary)

    output_paths = [
        node_pair_path,
        node_routes_path,
        yardstick_path,
        benchmark_path,
        regime_counts_path,
        same_node_path,
        summary_path,
        audit_path,
    ]
    output_manifest = {
        "status": "pass",
        "outputs": {
            display_path(path): {
                "byte_size": path.stat().st_size,
                "sha256": sha256sum(path),
            }
            for path in output_paths
        },
    }
    write_json_atomic(
        artifact_dir / "c0_4_output_manifest.json",
        output_manifest,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Route every frozen 2021 C0A city pair under the three C0 regimes."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-workers", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = run(
        config_path=CODING_ROOT / arguments.config
        if not arguments.config.is_absolute()
        else arguments.config,
        artifact_dir=CODING_ROOT / arguments.artifact_dir
        if not arguments.artifact_dir.is_absolute()
        else arguments.artifact_dir,
        report_dir=CODING_ROOT / arguments.report_dir
        if not arguments.report_dir.is_absolute()
        else arguments.report_dir,
        run_dir=CODING_ROOT / arguments.run_dir
        if not arguments.run_dir.is_absolute()
        else arguments.run_dir,
        max_workers_override=arguments.max_workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
