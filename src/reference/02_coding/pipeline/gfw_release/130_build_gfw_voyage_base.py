#!/usr/bin/env python3
"""E006 GFW version, step 3b: raw legs -> 30b -> 30c -> annual q-trim -> candidate root.

Follows E005's annual voyage base (`experiments/E005_2024_voyage_base/20_build_voyage_pretrim_
pilot.py` and `40_run_annual_voyage_base.py`, on that branch) for the 2025 GFW population:

1. 30b: the unchanged OD adapter turns the step-3a raw legs into the monthly `our_od` files.
2. 30c: the unchanged builder makes city episodes, adjacent-commercial-episode candidates,
   city snaps, yardsticks, c2 (idle gap), c3 (route circuity, duration/speed) and c4 pretrim.
3. 30d: pooled q01/q99 refitted on this population's own eligible candidates (T030 c).
4. 30e: candidate root of the retained voyages, then independent checks.

City reference (T032): every city coordinate is recomputed from this run's GFW port calls.
The builder normally reuses the frozen 2025 city table and computes only cities missing from
it; here it is given an empty table with the same columns, so every required city goes
through the same `build_city_coordinates` + `add_network_snaps` path. The two reference AIS-weighted
city tables in the frozen stage-30 support are not linked into the input view at all; only
the geometry-only node-pair route caches are (reusable per T032). Manual geography is kept:

- Puerto Ordaz keeps its configured port/gateway role split. Each anchorage keeps the role the
  reviewed override gives it; the override's lists are narrowed to the anchorages that actually
  carry GFW-version calls, because the frozen function requires the lists to partition exactly
  the called anchorages. An unlisted called anchorage or an empty role stops the run.
- Golfito keeps the approved same-coast network node (E001 `5e0670a`): when its recomputed
  coordinate snaps to the reviewed wrong-coast node, the node is replaced exactly as E001's
  `93_replay_golfito_coast_override.py` did. A different snap stops the run for review.

Anchorage-to-city mapping (T034): with `--mapping-corrections`, the reviewed list from
`125_build_city_mapping_corrections.py` is applied to a copy of the frozen mapping inside the
run (four cities whose anchorages included another port's); the frozen file is not modified.

30d and 30e: the pipeline entries accept only the reference AIS 2025 L1/L2/L3 pool (16 parts, with a
historical retention list) or the E005 2024 pool. This independent 64-part GFW population runs
through their unchanged functions instead -- `core.global_qtrim.fit_global_thresholds`, the 30d
`apply_thresholds`, and the 30e `inspect_part` / `link_or_verify` -- and writes manifests in the
same layout. No network use; outputs stay in `.runs/`.

    python 130_build_gfw_voyage_base.py --port-root <step-3a root> --out <new .runs root> [--workers 32] \
        [--mapping-corrections config/t034_anchorage_city_corrections.csv]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
PIPELINE = CODING / "pipeline/30_build_analysis_sample"
STAGE = PIPELINE / "30_build_adjacent_commercial_episode_voyages"
ADAPTER = PIPELINE / "20_adapt_od_for_voyage_construction/10_adapt_od_for_first_passage.py"
QTRIM = PIPELINE / "40_refit_and_apply_global_qtrim/10_refit_and_apply_global_qtrim.py"
ASSEMBLY = PIPELINE / "50_assemble_candidate_root/10_assemble_candidate_root.py"
SUPPORT = CODING / "data/frozen_inputs/stage30_support_2025"
MAPPING = SUPPORT / "data/legacy_inputs/anchorage_to_analytical_city_candidate_v3.csv"
NAMED = CODING / "data/frozen_inputs/named_anchorages"
ROSTER = CODING / "data/frozen_inputs/roster_2025/p2/extended_sample_mmsi_2025.csv"
CITY_COLUMNS_FROM = CODING / "data/frozen_inputs/e001_2025_pretrim_golfito_corrected/city_snap_table_2025.csv"
GOLFITO_EVIDENCE = CODING / "outputs/2026-09-08_1636_E001_remote-city-mapping-review/golfito_coast_evidence.json"
EXPECTED = {
    ROSTER: "32526296cde85a63489de621f29880a48b78686391f0b3eef44f3e64c8d9c078",
    CITY_COLUMNS_FROM: "af17cbe3e6aedaf1125d89ed50f731dc5b2bf87022bbce2db8213d9938af55c1",
    GOLFITO_EVIDENCE: "e8cd7d9420610a70b199c88be8200f3de3d92fe7e63201d13023d8dad9c69927",
}
# Geometry-only route caches between searoute network nodes (T032: reusable).
ROUTE_CACHES = ("support/frozen_2021/node_pair_routes_2021.feather",
                "support/endpoint_2025/new_node_pair_routes_2025.feather")
MAIN = "data/processed/build_multicall_first_passage_voyages/main"
MODE = "adjacent_commercial_episodes"
YEAR = 2025
PART_COUNT = 64
LAYER = "GFW2025"
POPULATION = "E006_gfw_presence_2025"
ROSTER_VESSELS = 36_000
GFW_VESSELS = 34_884
GOLFITO = {"city_id": "CRI:GOLFITO", "old_node": (-82.059631, 8.959689), "new_node": (-82.8, 7.4167),
           "new_node_id": "-82.800000,7.416700", "reason": "approved_same_coast_existing_node_override",
           "evidence": "65bff71/6cce249; frozen GSHHG local-component audit"}
# Network node of the reviewed Puerto Ordaz pilot gateway (role split approved 2026-07-26).
REVIEWED_GATEWAY_NODES = {"VEN:PUERTO ORDAZ": (-60.320435, 8.966471)}
PROTECTED_FILTERS = {"route_circuity_threshold": 2., "duration_threshold_days": 30.,
                     "positive_route_speed_threshold_knots": 1., "same_node_route_backstop_km": 2000.,
                     "exclude_zero_route_above_duration_threshold": True}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise ValueError(f"bound input changed: {path}")
    return path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalized(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def code_records() -> list[dict]:
    paths = [*STAGE.rglob("*.py"), ADAPTER, QTRIM, ASSEMBLY, Path(__file__),
             HERE / "120_run_gfw_port_calls.py", HERE / "125_build_city_mapping_corrections.py"]
    return [record(p) for p in sorted(set(paths)) if "__pycache__" not in p.parts and "tests" not in p.parts]


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


# --- Inputs -----------------------------------------------------------------------------


def port_inputs(port_root: Path) -> tuple[dict, list[dict]]:
    state = json.loads((port_root / "execution.json").read_text())
    if (state["status"] != "pass" or not state["complete_year"] or state["failures"]
            or len(state["results"]) != 64 or not all(state.get("closure", {}).values())):
        raise ValueError("step 3a is not the complete, closed 64-shard port-call run")
    for item in state["results"]:
        for output in item["outputs"]:
            verify(output)
        verify(item["vessel_input_disposition"])
    verify(state["specs"])
    return state, state["results"]


def build_view(root: Path, port_state: dict, results: list[dict], corrections: Path | None) -> dict:
    """Specs, raw legs, 30b monthly OD and the reference links the builder reads."""
    view = root / "input_view"
    disposition = pd.concat([pd.read_csv(verify(r["vessel_input_disposition"]), dtype={"vessel_uid": str})
                             for r in results], ignore_index=True)
    ids = set(disposition.vessel_uid)
    if len(ids) != GFW_VESSELS or disposition.vessel_uid.duplicated().any():
        raise ValueError("cleaned GFW vessel denominator differs from step 2")
    specs = pd.read_csv(verify(port_state["specs"]), dtype=str)
    specs = specs.loc[specs.vessel_uid_clean.isin(ids)].copy()
    if len(specs) != len(ids):
        raise ValueError("analysis specs do not cover every cleaned GFW vessel")
    specs_dir = view / "data/processed/products/vessel_universe"
    specs_dir.mkdir(parents=True)
    specs.to_csv(specs_dir / f"analysis_vessel_specs_{YEAR}.csv", index=False)
    pd.DataFrame({"vesselUid": sorted(ids)}).to_csv(specs_dir / "observed_fleet.csv", index=False)
    legs = pd.concat([pd.read_feather(r["voyages_output_path"]) for r in sorted(results, key=lambda r: r["shard"])],
                     ignore_index=True)
    if legs.internal_voyage_id.duplicated().any() or not set(legs.vessel_uid.astype(str)) <= ids:
        raise ValueError("raw legs duplicate across shards or name an unknown vessel")
    raw = view / f"raw_port_calls/{YEAR}"
    raw.mkdir(parents=True)
    legs.to_feather(raw / f"harmonized_od_voyages_{YEAR}.feather")

    # Reference links: route geometry caches and the analytical-city mapping only.
    manifest = json.loads((SUPPORT / "manifest.json").read_text())
    linked = []
    for item in manifest["files"]:
        source = Path(item["target"]["path"])
        relative = source.relative_to(SUPPORT)
        wanted = relative.name == "anchorage_to_analytical_city_candidate_v3.csv" or any(
            str(relative) == f"{MAIN}/{cache}" for cache in ROUTE_CACHES)
        if not wanted:
            continue
        verify(item["target"])
        destination = view / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve())
        linked.append(record(source))
    if len(linked) != 3:
        raise ValueError("expected the mapping and two route caches in the frozen support")
    for name in ("named_anchorages_v2_20201104.csv", "named_anchorages_v2_20221206.csv"):
        destination = view / "data/named_anchorages" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to((NAMED / name).resolve())
        linked.append(record(NAMED / name))
    alias = view / "pipeline/30_build_analysis_sample/30_build_adjacent_commercial_episode_voyages"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(STAGE, target_is_directory=True)
    empty = view / "support_inputs/city_reference_empty.csv"
    empty.parent.mkdir(parents=True)
    pd.read_csv(CITY_COLUMNS_FROM, nrows=0).to_csv(empty, index=False)
    mapping = {"mapping": record(MAPPING), "mapping_corrections": None, "mapping_corrected_anchorages": 0}
    if corrections is not None:
        # T034: the reviewed anchorage-to-city corrections, applied to a copy of the frozen mapping.
        fix = load_module("e006_mapping_fix", HERE / "125_build_city_mapping_corrections.py")
        table = pd.read_csv(corrections, dtype=str, keep_default_na=False)
        corrected = view / "support_inputs/anchorage_to_analytical_city_corrected.csv"
        fix.apply(MAPPING, table, corrected)
        mapping = {"mapping": record(corrected), "mapping_corrections": record(corrections),
                   "mapping_corrected_anchorages": len(table),
                   "mapping_corrected_cities": sorted(set(table.old_topology_city_id) | set(table.new_topology_city_id))}

    adapter = load_module("e006_adapter", ADAPTER)
    adapter.SPECS_TEMPLATE = specs_dir / "analysis_vessel_specs_{year}.csv"
    saved = sys.argv
    try:
        sys.argv = [str(ADAPTER), "--years", str(YEAR), "--od-root", str(raw.parent), "--out", str(view / "our_od")]
        adapter.main()
    finally:
        sys.argv = saved
    monthly = pd.concat([pd.read_csv(p, dtype=str) for p in sorted((view / f"our_od/{YEAR}").glob("*.csv"))],
                        ignore_index=True)
    if len(monthly) != len(legs) or monthly.trip_id.duplicated().any():
        raise ValueError("30b lost or multiplied raw legs")
    joined = monthly.merge(legs, left_on="trip_id", right_on="internal_voyage_id", validate="one_to_one")
    for left, right in (("vessel_id", "vessel_uid"), ("trip_start_anchorage_s2id", "origin_anchorage_id"),
                        ("trip_end_anchorage_s2id", "dest_anchorage_id"),
                        ("trip_start_visit_id", "origin_visit_id_internal"), ("trip_end_visit_id", "dest_visit_id_internal")):
        if not normalized(joined[left]).eq(normalized(joined[right])).all():
            raise ValueError(f"30b changed {left}")
    for column in ("trip_start", "trip_end"):
        if not pd.to_datetime(joined[column + "_x"], utc=True).eq(pd.to_datetime(joined[column + "_y"], utc=True)).all():
            raise ValueError(f"30b changed {column}")
    by_api = specs.set_index("vessel_uid_clean").mmsi_clean
    if not normalized(monthly.mmsi).eq(normalized(monthly.vessel_id).map(by_api)).all():
        raise ValueError("30b canonical identity join differs")
    return {"view": str(view), "vessels": len(ids), "raw_legs": len(legs), "monthly_rows": len(monthly),
            "joins_one_to_one": True, "reference_links": linked, "empty_city_reference": record(empty),
            "cleaned_rows": int(disposition.cleaned_rows.sum()), **mapping}


# --- City reference recompute ---------------------------------------------------------


class CityRecompute:
    """Wrap the two frozen city functions so every city is recomputed with manual geography kept."""

    def __init__(self, c0_snap):
        self.c0_snap = c0_snap
        self.build_city_coordinates = c0_snap.build_city_coordinates
        self.add_network_snaps = c0_snap.add_network_snaps
        self.role_overrides: list[dict] = []
        self.node_overrides: list[dict] = []
        self.gateway_checks: dict[str, tuple[float, float]] = {}
        c0_snap.build_city_coordinates = self.coordinates
        c0_snap.add_network_snaps = self.snaps

    def coordinates(self, calls, cities, lookup, spread_threshold_deg, coordinate_role_overrides=None):
        """Apply each reviewed port/gateway split to the anchorages that carry GFW-version calls.

        A called anchorage missing from the reviewed lists (they hold the 2021-called cells)
        takes the role of its nearest listed anchorage, if one lies within 25 km. When no
        gateway anchorage is called, the routing point is the equal-weight mean of the reviewed
        gateway anchorages' catalog positions: there are no GFW-version calls to weight it by,
        and reference AIS era weights are not used (T032).
        """
        clean = self.c0_snap.clean_id
        positions = lookup.set_index("s2id")[["lat", "lon"]]
        adapted, without_gateway_calls = [], []
        for item in coordinate_role_overrides or []:
            city = calls.loc[calls["is_commercial_endpoint"].astype(bool)
                             & calls["topology_city_id"].eq(item["city_id"]), "anchorage_id"]
            present = set(clean(city))
            listed_port = set(clean(pd.Series(item["port_anchorage_ids"])))
            listed_gateway = set(clean(pd.Series(item["routing_gateway_anchorage_ids"])))
            assigned = {}
            for anchorage in sorted(present - listed_port - listed_gateway):
                here = positions.loc[anchorage]
                distances = {other: haversine_km(here.lon, here.lat, positions.loc[other, "lon"], positions.loc[other, "lat"])
                             for other in listed_port | listed_gateway}
                nearest = min(distances, key=distances.get)
                if distances[nearest] > 25:
                    raise ValueError(f"{item['city_id']}: called anchorage {anchorage} is {distances[nearest]:.1f} km "
                                     "from every reviewed anchorage")
                assigned[anchorage] = {"role": "port" if nearest in listed_port else "gateway",
                                       "nearest_listed": nearest, "distance_km": round(distances[nearest], 3)}
            port = sorted((listed_port & present) | {a for a, v in assigned.items() if v["role"] == "port"})
            gateway = sorted((listed_gateway & present) | {a for a, v in assigned.items() if v["role"] == "gateway"})
            entry = {"city_id": item["city_id"], "called_anchorages": sorted(present), "port": port, "gateway": gateway,
                     "unlisted_called_assigned_by_nearest_listed": assigned,
                     "listed_port_not_called": sorted(listed_port - present),
                     "listed_gateway_not_called": sorted(listed_gateway - present)}
            self.role_overrides.append(entry)
            if not port:
                raise ValueError(f"coordinate-role override has no called port anchorage: {entry}")
            if gateway:
                entry["routing_point"] = "weighted by GFW-version calls at the called gateway anchorages"
                adapted.append({**item, "port_anchorage_ids": port, "routing_gateway_anchorage_ids": gateway})
            else:
                without_gateway_calls.append((item, entry, sorted(listed_gateway)))
        output, lineage, diagnostics = self.build_city_coordinates(calls, cities, lookup, spread_threshold_deg, adapted)
        for item, entry, gateway in without_gateway_calls:
            if not set(gateway) <= set(positions.index):
                raise ValueError(f"{item['city_id']}: a reviewed gateway anchorage has no catalog position")
            cells = positions.loc[gateway].reset_index().assign(coordinate_weight_records=1)
            lat, lon = self.c0_snap.weighted_coordinate(cells)
            selected = output.city_id.eq(item["city_id"])
            if selected.sum() != 1 or output.loc[selected, "routing_gateway_override"].any():
                raise ValueError(f"{item['city_id']}: unexpected coordinate row")
            output.loc[selected, ["routing_lat", "routing_lon", "coordinate_role_mode", "routing_gateway_override",
                                  "routing_gateway_reason", "routing_gateway_evidence",
                                  "routing_coordinate_weight_records"]] = [
                lat, lon, "separate_port_and_gateway", True,
                f"{item['reason']} E006: no GFW-version gateway calls; routing point is the equal-weight mean "
                "of the reviewed gateway anchorages.", str(item["evidence"]), 0]
            entry["routing_point"] = {"rule": "equal-weight mean of reviewed gateway anchorage positions",
                                      "anchorages": gateway, "routing_lat": lat, "routing_lon": lon}
            self.gateway_checks[item["city_id"]] = (lon, lat)
        diagnostics["coordinate_role_override_count"] = int(output["routing_gateway_override"].astype(bool).sum())
        return output, lineage, diagnostics

    def snaps(self, cities, threshold_km):
        output, diagnostics = self.add_network_snaps(cities, threshold_km)
        for city_id in set(self.gateway_checks) & set(output.city_id):
            row = output.loc[output.city_id.eq(city_id)].iloc[0]
            if not np.allclose([row.network_lon, row.network_lat], REVIEWED_GATEWAY_NODES[city_id], rtol=0, atol=1e-9):
                raise ValueError(f"{city_id}: gateway routing point snapped to {(row.network_lon, row.network_lat)}, "
                                 f"not the reviewed node {REVIEWED_GATEWAY_NODES[city_id]}")
        selected = output.city_id.eq(GOLFITO["city_id"])
        if selected.any():
            row = output.loc[selected].iloc[0]
            old_lon, old_lat = GOLFITO["old_node"]
            if not np.allclose([row.network_lon, row.network_lat], [old_lon, old_lat], rtol=0, atol=1e-9):
                raise ValueError(f"Golfito snapped to an unreviewed node: {(row.network_lon, row.network_lat)}")
            new_lon, new_lat = GOLFITO["new_node"]
            snap = haversine_km(float(row.routing_lon), float(row.routing_lat), new_lon, new_lat)
            before = {k: row[k] for k in ("network_lon", "network_lat", "network_node_id", "snap_distance_km",
                                          "routing_gateway_override")}
            output.loc[selected, ["network_lon", "network_lat", "network_node_id", "snap_distance_km",
                                  "snap_distance_gt_50km", "routing_gateway_override", "routing_gateway_reason",
                                  "routing_gateway_evidence"]] = [
                new_lon, new_lat, GOLFITO["new_node_id"], snap, snap > threshold_km, True,
                GOLFITO["reason"], GOLFITO["evidence"]]
            self.node_overrides.append({"city_id": GOLFITO["city_id"], "before": before,
                                        "after": {"network_lon": new_lon, "network_lat": new_lat,
                                                  "network_node_id": GOLFITO["new_node_id"], "snap_distance_km": snap},
                                        "routing_lon": float(row.routing_lon), "routing_lat": float(row.routing_lat)})
            diagnostics = {**diagnostics, "golfito_node_override": True,
                           "snap_distance_gt_50km_count": int(output.snap_distance_gt_50km.astype(bool).sum())}
        return output, diagnostics


def configure_builder(view: Path, empty_reference: Path, mapping: Path):
    builder = load_module("e006_builder", STAGE / "10_build_adjacent_commercial_episode_voyages.py")
    builder.CODING_ROOT = view
    builder.OD_ROOT = view / "our_od"
    builder.FROZEN_SUPPORT_ROOT = view / MAIN
    builder.CITY_REFERENCE_PATH = empty_reference
    builder.MAPPING_PATH = mapping
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(STAGE) and hasattr(module, "CODING_ROOT"):
            module.CODING_ROOT = view
    return builder, CityRecompute(builder.c0_snap)


# --- 30c ------------------------------------------------------------------------------


def run_30c(root: Path, view: Path, empty_reference: Path, mapping: Path, workers: int, state: dict, save) -> dict:
    builder, cities = configure_builder(view, empty_reference, mapping)
    result = view / "results"
    result.mkdir(exist_ok=True)

    def step(name):
        state.update(current_step=name)
        save()
        print(f"{now()} {name}", flush=True)

    step("30c_city_episodes")
    episode = builder.build_year_episodes(YEAR, result, 500_000, MODE)
    episodes, raw_legs, call_meta = builder.load_candidate_inputs(episode)
    step("30c_candidate_occurrences")
    parts = builder.run_candidate_tasks(YEAR, result, episodes, raw_legs, call_meta, workers, PART_COUNT, None,
                                        builder.episode_manifest_identity(episode), MODE)
    candidates = builder.validate_candidate_outputs(YEAR, result, episode, parts, PART_COUNT, MODE)
    if candidates["status"] != "pass":
        raise ValueError("candidate enumeration failed")
    step("30c_city_reference_recompute")
    city = builder.build_combined_city_snaps((YEAR,), result, {YEAR: episode})
    city_summary = json.loads((result / "support/city_snap_summary.json").read_text())
    if city_summary["reused_frozen_city_count"] != 0 or city_summary["extension_city_count"] != city_summary["required_city_count"]:
        raise ValueError(f"a city coordinate was not recomputed: {city_summary}")
    step("30c_yardstick_and_c2")
    yardstick = builder.build_year_yardstick(YEAR, result, episode, city, workers)
    c2 = builder.materialize_c2(YEAR, result, PART_COUNT, MODE)
    transition = builder.build_transition_yardstick(YEAR, result, c2, city, yardstick, workers, (YEAR,))
    filtering = builder.downstream_config(YEAR, yardstick, workers, MODE)
    filtering["candidate_part_count"] = PART_COUNT
    if any(filtering[key] != value for key, value in PROTECTED_FILTERS.items()) or "p4_city_shiptype_exposure" in filtering:
        raise ValueError("quality settings or dependency boundary changed")
    step("30c_c3_and_pretrim")
    artifact = result / f"intermediate/{YEAR}/filters"
    c3 = builder.rebuild.run_c3(YEAR, filtering, c2, transition, artifact)
    pretrim = builder.rebuild.prepare_c4_parts(YEAR, filtering, artifact)
    rows = pd.concat([pd.read_feather(p) for p in pretrim], ignore_index=True)
    if rows.candidate_id.duplicated().any():
        raise ValueError("duplicate pretrim candidates")
    final_city = pd.read_csv(city)
    city_summary = json.loads((result / "support/city_snap_summary.json").read_text())
    recomputed = city_summary["extension_city_count"] + city_summary.get("transition_extension_city_count", 0)
    if len(final_city) != recomputed or final_city.city_id.duplicated().any():
        raise ValueError("final city table holds a city this run did not compute")
    if final_city.city_id.eq(GOLFITO["city_id"]).any() and not cities.node_overrides:
        raise ValueError("Golfito is present without its approved node override")
    overrides = {"coordinate_role_overrides": cities.role_overrides, "network_node_overrides": cities.node_overrides}
    write_json(root / "city_reference_overrides.json", overrides)
    return {"status": "complete_pretrim_candidate_only", "episode_summary": episode["diagnostics"],
            "candidate_summary": candidates["diagnostics"], "c2_summary": c2["summary"], "c3_summary": c3["summary"],
            "pretrim_rows": len(rows), "pretrim_vessels": int(rows.vessel_uid_clean.nunique()),
            "pretrim": [record(p) for p in pretrim], "city_snapshot": record(city),
            "city_summary": city_summary, "city_overrides": overrides,
            "yardstick": record(yardstick), "transition_yardstick": record(transition), "filter_config": filtering}


# --- 30d / 30e through the unchanged pipeline functions ---------------------------------


def run_qtrim(root: Path, voyage: dict) -> dict:
    stage = load_module("e006_qtrim_stage", QTRIM)
    paths = [verify(item) for item in voyage["pretrim"]]
    if [p.name for p in paths] != [f"part-{i:03d}.feather" for i in range(PART_COUNT)]:
        raise ValueError("pretrim parts are not the 64 contiguous parts")
    thresholds = stage.qtrim.fit_global_thresholds(paths, 0.01, 0.99, YEAR)
    lower, upper = float(thresholds["normalized_time_ratio_q01"]), float(thresholds["normalized_time_ratio_q99"])
    out = root / "qtrim"
    final_dir = out / f"final_sample_{YEAR}"
    final_dir.mkdir(parents=True)
    parts, decisions, totals = [], [], {"input_rows": 0, "eligible_rows": 0, "global_qtrimmed": 0,
                                         "temporal_scope_excluded": 0, "final_retained": 0}
    for i, path in enumerate(paths):
        frame = pd.read_feather(path)
        applied = stage.apply_thresholds(frame, lower, upper, year=YEAR)
        eligible = (applied.temporal_scope_eligible.astype(bool) & np.isfinite(applied.normalized_time_ratio)
                    & applied.d_yardstick.gt(0))
        retained = applied.loc[applied.retain_after_global_qtrim].reset_index(drop=True)
        target = final_dir / path.name
        retained.to_feather(target, compression="zstd", compression_level=6)
        parts.append({"part_id": i, **record(target), "rows": len(retained), "input_rows": len(frame),
                      "source": record(path)})
        decisions.append(applied[[*stage.IDENTITY_COLUMNS, "temporal_scope_eligible", "normalized_time_ratio",
                                  "d_yardstick", "global_qtrimmed", "retain_after_global_qtrim"]])
        totals["input_rows"] += len(frame)
        totals["eligible_rows"] += int(eligible.sum())
        totals["global_qtrimmed"] += int(applied.global_qtrimmed.sum())
        totals["temporal_scope_excluded"] += int((~applied.temporal_scope_eligible.astype(bool)).sum())
        totals["final_retained"] += len(retained)
    if totals["eligible_rows"] != int(thresholds["eligible_rows"]):
        raise ValueError("eligibility count disagrees with the threshold implementation")
    decision_frame = pd.concat(decisions, ignore_index=True)
    (out / "global_qtrim_refit").mkdir()
    decision_frame.to_parquet(out / f"global_qtrim_refit/candidate_retain_decisions_{YEAR}.parquet", index=False)
    manifest = {
        "status": "pass", "stage": "30d_refit_and_apply_global_qtrim",
        "run_mode": "formal_annual", "analysis_year": YEAR, "population": POPULATION,
        "candidate_enumeration_mode": MODE, "annual_pool_complete": True, "completed_layers": [LAYER],
        "generated_at": now(), "state": "candidate run artifact under .runs/, not adopted",
        "implementation": "E006 adapter over unchanged core.global_qtrim.fit_global_thresholds and 30d apply_thresholds",
        "pretrim_source": voyage["pretrim"],
        "qtrim": {"status": "pass", "fitted_on_complete_annual_pool": True,
                  "threshold_scope": f"{YEAR}_pooled_no_grouping", "q_lower": 0.01, "q_upper": 0.99,
                  "equality_retained": True, "normalized_time_ratio_q01": lower, "normalized_time_ratio_q99": upper,
                  "threshold_implementation": record(STAGE / "core/global_qtrim.py"),
                  "apply_implementation": record(QTRIM), "not_reused_from_reference_ais_version": True},
        "summary": {**totals, "unique_candidate_id": int(decision_frame.candidate_id.nunique()),
                    "unique_vessel_uid": int(decision_frame.vessel_uid_clean.nunique())},
        "parts": parts,
    }
    if sum(p["rows"] for p in parts) != totals["final_retained"]:
        raise ValueError("retained part counts do not close")
    write_json(out / f"manifests/final_sample_manifest_{YEAR}.json", manifest)
    return manifest


def run_assembly(root: Path, qmanifest_path: Path, workers: int) -> dict:
    assembly = load_module("e006_assembly", ASSEMBLY)
    manifest = json.loads(qmanifest_path.read_text())
    candidate_root = root / "candidate_root"
    year_dir = candidate_root / str(YEAR)
    year_dir.mkdir(parents=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(min(workers, PART_COUNT)) as pool:
        inspected = list(pool.map(lambda item: assembly.inspect_part(verify(item), assembly.FORMAL_COLUMNS, True, YEAR),
                                  manifest["parts"]))
    identities = pd.concat([item.pop("identities") for item in inspected], ignore_index=True)
    if identities.candidate_id.duplicated().any():
        raise ValueError("duplicate candidate_id across retained parts")
    if sum(item["rows"] for item in inspected) != manifest["summary"]["final_retained"]:
        raise ValueError("candidate root rows differ from the q-trim manifest")
    for i, item in enumerate(inspected):
        assembly.link_or_verify(Path(item["path"]), year_dir / f"part-{i:03d}.feather")
    assembly.link_or_verify(qmanifest_path, year_dir / f"final_sample_manifest_{YEAR}.json")
    entry = {"cohort": "calendar", "calendar_truncated": True, "candidate_enumeration_mode": MODE,
             "population": POPULATION, "source_path": str(Path(inspected[0]["path"]).parent),
             "source_manifest_path": str(qmanifest_path), "source_manifest_sha256": sha(qmanifest_path),
             "link_path": str(year_dir), "row_count": sum(item["rows"] for item in inspected),
             "part_count": len(inspected), "parts": inspected,
             "next_year_arrival_rows": sum(item["next_year_arrival_rows"] for item in inspected),
             "has_next_year_arrival": any(item["next_year_arrival_rows"] for item in inspected),
             "qtrim_status": manifest["qtrim"]["status"], "annual_pool_complete": True}
    if entry["next_year_arrival_rows"]:
        raise ValueError("a GFW-version voyage arrives after 2025")
    payload = {"status": "pass", "stage": "30e_assemble_candidate_root", "analysis_years": [YEAR],
               "assembly_mode": "formal_annual_independent_population", "population": POPULATION,
               "implementation": "E006 adapter over unchanged 30e inspect_part and link_or_verify",
               "candidate_enumeration_mode": MODE, "candidate_root": str(candidate_root), "years": {str(YEAR): entry},
               "source_30d_manifest_sha256": sha(qmanifest_path), "downstream_ready": True, "adopted": False,
               "downstream_contract": {"year_directory": f"{YEAR}/", "part_pattern": "part-*.feather",
                                       "required_part_count": len(inspected),
                                       "source_manifest_link": f"{YEAR}/final_sample_manifest_{YEAR}.json"}}
    write_json(candidate_root / "assembly_manifest.json", payload)
    return payload


# --- Independent checks ---------------------------------------------------------------


def validate(root: Path, port_state: dict, prep: dict, voyage: dict) -> dict:
    q = json.loads((root / f"qtrim/manifests/final_sample_manifest_{YEAR}.json").read_text())
    assembly = json.loads((root / "candidate_root/assembly_manifest.json").read_text())
    pre = pd.concat([pd.read_feather(verify(item), columns=["candidate_id", "vessel_uid_clean", "temporal_scope_eligible",
                                                             "normalized_time_ratio", "d_yardstick"])
                     for item in voyage["pretrim"]], ignore_index=True)
    retained = pd.concat([pd.read_feather(verify(item)) for item in q["parts"]], ignore_index=True)
    eligible = pre.temporal_scope_eligible.astype(bool) & np.isfinite(pre.normalized_time_ratio) & pre.d_yardstick.gt(0)
    independent = np.quantile(pre.loc[eligible, "normalized_time_ratio"].to_numpy(), [0.01, 0.99], method="linear")
    lower, upper = q["qtrim"]["normalized_time_ratio_q01"], q["qtrim"]["normalized_time_ratio_q99"]
    excluded = eligible & (pre.normalized_time_ratio.lt(lower) | pre.normalized_time_ratio.gt(upper))
    expected = set(pre.loc[pre.temporal_scope_eligible.astype(bool) & ~excluded, "candidate_id"])
    specs = pd.read_csv(verify(port_state["specs"]), dtype=str).set_index("vessel_uid_clean")
    identities = retained.drop_duplicates("vessel_uid_clean").set_index("vessel_uid_clean")
    departure, arrival = pd.to_datetime(retained.departure_time, utc=True), pd.to_datetime(retained.arrival_time, utc=True)
    checks = {
        "complete_64_port_shards": len(port_state["results"]) == 64 and not port_state["failures"],
        "cleaned_rows_close": prep["cleaned_rows"] == 255_216_589,
        "cleaned_vessels_close": prep["vessels"] == GFW_VESSELS,
        "independent_numpy_quantiles": bool(np.allclose(independent, [lower, upper], rtol=1e-12, atol=0)),
        "qtrim_eligible_denominator": int(eligible.sum()) == q["summary"]["eligible_rows"],
        "qtrim_exclusion_denominator": int(excluded.sum()) == q["summary"]["global_qtrimmed"],
        "retained_ids_exactly_expected": not retained.candidate_id.duplicated().any() and set(retained.candidate_id) == expected,
        "canonical_mmsi_unchanged": normalized(identities.mmsi).eq(normalized(specs.loc[identities.index, "mmsi_clean"])).all(),
        "canonical_imo_unchanged": normalized(identities.imo).eq(normalized(specs.loc[identities.index, "registry_imo_clean"])).all(),
        "calendar_2025_endpoints": bool((departure.dt.year.eq(YEAR) & arrival.dt.year.eq(YEAR) & arrival.gt(departure)).all()),
        "assembly_rows_and_parts": assembly["status"] == "pass" and assembly["years"][str(YEAR)]["row_count"] == len(retained)
                                   and assembly["years"][str(YEAR)]["part_count"] == PART_COUNT,
        "no_reference_ais_city_reference": voyage["city_summary"]["reused_frozen_city_count"] == 0,
    }
    if prep["mapping_corrections"] is not None:
        corrected = pd.read_csv(verify(voyage["city_snapshot"]))
        corrected = corrected.loc[corrected.city_id.isin(prep["mapping_corrected_cities"])]
        checks["corrected_cities_compact"] = not corrected.spread_gt_2deg.astype(bool).any()
    for i, part in enumerate(q["parts"]):
        link = root / f"candidate_root/{YEAR}/part-{i:03d}.feather"
        if not link.is_symlink() or link.resolve() != Path(part["path"]).resolve():
            raise ValueError(f"candidate-root link differs: {link}")
    checks["all_64_source_links"] = True

    # Every roster vessel keeps a disposition, including those GFW never saw.
    roster = pd.read_csv(ROSTER, dtype=str)
    vessels = pd.DataFrame({"vessel_uid": normalized(roster.vessel_uid), "mmsi": normalized(roster.mmsi),
                            "ship_type_sample": roster.ship_type_sample})
    observed = pd.concat([pd.read_csv(verify(r["vessel_input_disposition"]), dtype={"vessel_uid": str})
                          for r in port_state["results"]], ignore_index=True)
    vessels = vessels.merge(observed, on="vessel_uid", how="left", validate="one_to_one")
    vessels["pretrim_voyages"] = vessels.vessel_uid.map(pre.groupby("vessel_uid_clean").size()).fillna(0).astype(int)
    vessels["retained_voyages"] = vessels.vessel_uid.map(retained.groupby("vessel_uid_clean").size()).fillna(0).astype(int)
    vessels["voyage_stage_disposition"] = np.select(
        [vessels.cleaned_rows.isna(), vessels.raw_legs.eq(0), vessels.pretrim_voyages.eq(0), vessels.retained_voyages.eq(0)],
        ["no_gfw_positions", "no_high_confidence_raw_leg", "no_pretrim_candidate", "no_retained_after_annual_qtrim"],
        default="has_retained_voyage")
    vessels.to_csv(root / "annual_vessel_disposition.csv", index=False)
    checks["all_36000_roster_vessels_preserved"] = (len(vessels) == ROSTER_VESSELS
                                                    and int(vessels.cleaned_rows.isna().sum()) == ROSTER_VESSELS - GFW_VESSELS)
    checks["vessel_candidate_counts_close"] = (int(vessels.pretrim_voyages.sum()) == len(pre)
                                               and int(vessels.retained_voyages.sum()) == len(retained))
    if not all(checks.values()):
        raise ValueError(f"independent checks failed: {checks}")
    cities = pd.read_csv(verify(voyage["city_snapshot"]))
    result = {"status": "passed", "year": YEAR, "population": POPULATION, "checks": {k: bool(v) for k, v in checks.items()},
              "roster_vessels": ROSTER_VESSELS, "gfw_vessels": GFW_VESSELS, "raw_legs": prep["raw_legs"],
              "initial_candidates": voyage["candidate_summary"].get("candidate_rows"),
              "c2": voyage["c2_summary"], "c3": voyage["c3_summary"], "pretrim_rows": len(pre),
              "qtrim": {k: q["qtrim"][k] for k in ("normalized_time_ratio_q01", "normalized_time_ratio_q99")},
              "qtrim_summary": q["summary"], "retained_rows": len(retained),
              "retained_vessels": int(retained.vessel_uid_clean.nunique()),
              "vessel_disposition": vessels.voyage_stage_disposition.value_counts().to_dict(),
              "cities_recomputed": len(cities), "city_snap_over_50km": int(cities.snap_distance_gt_50km.astype(bool).sum()),
              "city_overrides": voyage["city_overrides"],
              "mapping_corrections": prep["mapping_corrections"],
              "mapping_corrected_anchorages": prep["mapping_corrected_anchorages"], "adopted": False,
              "stops_before": "step 4 (ten-minute tracks)"}
    write_json(root / "completion_validation.json", result)
    return result


# --- Driver ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--mapping-corrections", type=Path,
                        help="T034 anchorage-to-city corrections from 125_build_city_mapping_corrections.py")
    args = parser.parse_args()
    if args.mapping_corrections is not None:
        args.mapping_corrections = args.mapping_corrections.resolve()
    port_root, root = args.port_root.resolve(), args.out.resolve()
    if not root.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"nested numerical threads must be capped: {name}")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ValueError(f"frozen input changed: {path}")
    evidence = json.loads(GOLFITO_EVIDENCE.read_text())
    if (not np.allclose(evidence["old_node"], GOLFITO["old_node"], rtol=0, atol=1e-12)
            or not np.allclose(evidence["diagnostic_node"], GOLFITO["new_node"], rtol=0, atol=1e-12)):
        raise ValueError("Golfito constants differ from the reviewed evidence")
    port_state, results = port_inputs(port_root)
    root.mkdir(parents=True, exist_ok=False)
    state = {"step": "E006 GFW version step 3b: 30b -> 30c -> 30d -> 30e", "status": "running", "year": YEAR,
             "population": POPULATION, "workers": args.workers, "host": socket.gethostname(), "pid": os.getpid(),
             "started_at": now(), "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             "port_input": record(port_root / "execution.json"), "code": code_records(),
             "city_reference_policy": "all cities recomputed from GFW-version calls (T032); manual geography kept",
             "adopted": False, "network_requests": 0, "steps": {}}
    save = lambda: write_json(root / "execution.json", state)  # noqa: E731
    save()
    started = time.monotonic()

    def step(name, function):
        state.update(current_step=name)
        save()
        print(f"{now()} START {name}", flush=True)
        tick = time.monotonic()
        value = function()
        state["steps"][name] = {"status": "pass", "wall_seconds": round(time.monotonic() - tick, 1)}
        save()
        print(f"{now()} DONE {name}", flush=True)
        return value

    try:
        prep = step("30b_input_view", lambda: build_view(root, port_state, results, args.mapping_corrections))
        state["input_view"] = prep
        voyage = step("30c_build", lambda: run_30c(root, Path(prep["view"]), Path(prep["empty_city_reference"]["path"]),
                                                   Path(verify(prep["mapping"])),
                                                   args.workers, state, save))
        write_json(root / "voyages_execution.json", voyage)
        state["voyages"] = record(root / "voyages_execution.json")
        q = step("30d_annual_qtrim", lambda: run_qtrim(root, voyage))
        step("30e_candidate_root", lambda: run_assembly(root, root / f"qtrim/manifests/final_sample_manifest_{YEAR}.json",
                                                        args.workers))
        summary = step("independent_validation", lambda: validate(root, port_state, prep, voyage))
        for item in state["code"]:
            verify(item)
        state.update(status="complete_through_30e_candidate_only", current_step="stopped_before_step_4",
                     validation=record(root / "completion_validation.json"), summary=summary,
                     qtrim={k: q["qtrim"][k] for k in ("normalized_time_ratio_q01", "normalized_time_ratio_q99")})
    except BaseException as error:
        state.update(status="failed", error=repr(error)[:4000])
        raise
    finally:
        state.update(wall_seconds=round(time.monotonic() - started, 1), ended_at=now(),
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save()


if __name__ == "__main__":
    main()
