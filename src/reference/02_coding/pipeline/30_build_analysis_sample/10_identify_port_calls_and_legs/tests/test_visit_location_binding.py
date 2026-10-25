from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "10_build_port_calls_and_legs.py"
SPEC = importlib.util.spec_from_file_location("build_port_calls_and_legs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _times(n: int) -> np.ndarray:
    start = np.datetime64("2023-01-01T00:00:00", "ns")
    return start + np.arange(n) * np.timedelta64(10, "m")


def test_visit_cannot_borrow_a_stop_from_another_location() -> None:
    """A visit opened at A must not inherit a later stop at location B."""
    visits = MODULE.build_vessel_visits(
        vessel_uid="ship-1",
        registry_imo_clean="imo-1",
        times=_times(4),
        speeds=np.array([8.0, 8.0, 0.1, 8.0]),
        latitudes=np.array([0.0, 0.0, 0.0, 0.0]),
        longitudes=np.array([0.0, 0.10, 0.10, 0.15]),
        nearest_idx=np.array([0, 1, 1, 1]),
        nearest_dist_km=np.array([1.0, 1.0, 1.0, 5.0]),
        anchorage_latitudes=np.array([0.0, 0.0]),
        anchorage_longitudes=np.array([0.0, 0.10]),
    )

    assert visits[0]["anchorage_idx"] == 0
    assert visits[0]["has_stop"] is False
    assert visits[0]["exit_time"] == _times(4)[1]


def test_nearest_cell_can_change_while_opening_cell_remains_near() -> None:
    """Nearest-cell changes alone must not close a visit inside its 4 km radius."""
    visits = MODULE.build_vessel_visits(
        vessel_uid="ship-1",
        registry_imo_clean="imo-1",
        times=_times(4),
        speeds=np.array([8.0, 0.1, 0.1, 8.0]),
        latitudes=np.array([0.0, 0.0, 0.0, 0.0]),
        longitudes=np.array([0.0, 0.015, 0.020, 0.050]),
        nearest_idx=np.array([0, 1, 1, 2]),
        nearest_dist_km=np.array([1.0, 1.0, 1.0, 1.0]),
        anchorage_latitudes=np.array([0.0, 0.0, 0.0]),
        anchorage_longitudes=np.array([0.0, 0.020, 0.050]),
    )

    assert visits[0]["anchorage_idx"] == 0
    assert visits[0]["has_stop"] is True
    assert visits[0]["exit_time"] == _times(4)[3]
    assert visits[0]["n_ais_points_in_visit"] == 4
