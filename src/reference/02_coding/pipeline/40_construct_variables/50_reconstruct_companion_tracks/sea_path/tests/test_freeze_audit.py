"""Tests for the independent E032 sea-path freeze-audit helpers."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

import sea_path.freeze_audit as freeze_audit
from sea_path.freeze_audit import (
    CORRIDORS,
    GridSpec,
    audit_route_csv,
    audit_route_directory,
    canonical_edge,
    grid_index,
    measure_corridors,
)
from sea_path.ocean_mask import maritime_channel_edges


EXPECTED_CORRIDOR_COUNTS = {
    "suez": (705, 609),
    "panama": (287, 248),
    "malacca": (2_267, 1_993),
    "red_sea": (4_246, 3_928),
    "bab_el_mandeb": (464, 405),
    "cape": (10_604, 10_472),
    "open_indian": (30_000, 30_000),
    "open_north_pacific": (60_000, 60_000),
}


def test_independent_auditor_does_not_import_routing():
    source = inspect.getsource(freeze_audit)
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "sea_path.routing" not in imported_modules
    assert "_segment_is_ocean" not in source


def test_grid_index_wraps_the_antimeridian():
    grid = GridSpec(n_lat=3, n_lon=4, resolution_deg=1.0, lat_min=-1.0, lon_min=-1.5)
    assert grid_index(0.0, 2.5, grid=grid) == (1, 0)
    assert grid_index(0.0, -1.5, grid=grid) == (1, 0)


def test_route_audit_separates_declared_from_undeclared_land_contact(tmp_path):
    grid = GridSpec(n_lat=3, n_lon=4, resolution_deg=1.0, lat_min=0.0, lon_min=0.0)
    ocean = np.ones((3, 4), dtype=bool)
    ocean[0, 1] = False
    route_csv = tmp_path / "route.csv"
    route_csv.write_text(
        "latitude,longitude\n0.0,0.0\n0.0,2.0\n",
        encoding="utf-8",
    )
    edge = canonical_edge((0, 0), (0, 2))

    declared_summary, _ = audit_route_csv(
        ocean,
        route_csv,
        frozenset({edge}),
        grid=grid,
    )
    undeclared_summary, _ = audit_route_csv(
        ocean,
        route_csv,
        frozenset(),
        grid=grid,
    )

    assert declared_summary["declared_land_touching_segment_count"] == 1
    assert declared_summary["undeclared_land_touching_segment_count"] == 0
    assert undeclared_summary["undeclared_land_touching_segment_count"] == 1


def test_frozen_corridor_counts_reproduce(ocean_mask):
    results = measure_corridors(ocean_mask, CORRIDORS, workers=8)
    actual = {
        row["corridor_id"]: (
            row["denominator_ocean_containing_0p1deg_cells"],
            row["centre_navigable_cells"],
        )
        for row in results
    }
    assert actual == EXPECTED_CORRIDOR_COUNTS


def test_independent_real_route_audit_is_bounded_to_declared_edges(ocean_mask):
    routes_dir = Path(__file__).resolve().parent / "fixtures/navigation_routes"
    assert routes_dir.is_dir(), "Required committed navigation-route fixtures are missing"

    summaries, segments = audit_route_directory(
        ocean_mask,
        routes_dir,
        maritime_channel_edges(),
        workers=8,
    )

    assert len(summaries) == 8
    assert len(segments) == 7
    assert sum(row["undeclared_land_touching_segment_count"] for row in summaries) == 0
    assert all(row["declared_channel_edge"] for row in segments)
