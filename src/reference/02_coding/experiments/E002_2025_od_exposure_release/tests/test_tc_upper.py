from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "20_quantify_uncertainty/10_missing_positions/20_compute_tc_upper.py"
)
SPEC = importlib.util.spec_from_file_location("e002_tc_upper", SCRIPT)
assert SPEC and SPEC.loader
TC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TC)


def test_single_endpoint_circle_tangency_is_included() -> None:
    endpoint = TC._unit(0.0, 0.0)
    storm = TC._unit(0.0, 1.0)
    distance = float(TC.LEGACY.great_circle_km(0.0, 0.0, 0.0, 1.0))
    assert TC.spherical_caps_intersect([
        TC._cap(endpoint, distance - 20.0),
        TC._cap(storm, 20.0),
    ])
    assert not TC.spherical_caps_intersect([
        TC._cap(endpoint, distance - 20.1),
        TC._cap(storm, 20.0),
    ])


def test_two_sided_reachable_intersection_and_disjoint_case() -> None:
    left = TC._unit(0.0, -1.0)
    right = TC._unit(0.0, 1.0)
    storm_at_midpoint = TC._unit(0.0, 0.0)
    far_storm = TC._unit(0.0, 3.0)
    caps = [TC._cap(left, 130.0), TC._cap(right, 130.0)]
    assert TC.spherical_caps_intersect([*caps, TC._cap(storm_at_midpoint, 0.0)])
    assert not TC.spherical_caps_intersect([*caps, TC._cap(far_storm, 20.0)])


def test_pairwise_overlapping_caps_need_not_have_a_common_point() -> None:
    a = TC._unit(0.0, 0.0)
    b = TC._unit(0.0, 0.8993)
    c = TC._unit(0.779, 0.44965)
    assert not TC.spherical_caps_intersect([
        TC._cap(a, 54.0), TC._cap(b, 54.0), TC._cap(c, 54.0)
    ])
    assert TC.spherical_caps_intersect([
        TC._cap(a, 60.0), TC._cap(b, 60.0), TC._cap(c, 60.0)
    ])


def test_user_necessary_condition_discards_a_center_too_far_from_endpoint() -> None:
    result = TC.necessary_center_filter(
        storm_lat=np.array([0.0]),
        storm_lon=np.array([10.0]),
        event_ns=3_600_000_000_000,
        left_time_ns=0,
        left_lat=0.0,
        left_lon=0.0,
        right_time_ns=-1,
        right_lat=np.nan,
        right_lon=np.nan,
        radius_km=500.0,
    )
    assert result.tolist() == [False]


def test_two_sided_necessary_condition_requires_both_endpoints() -> None:
    result = TC.necessary_center_filter(
        storm_lat=np.array([0.0]),
        storm_lon=np.array([0.0]),
        event_ns=3_600_000_000_000,
        left_time_ns=0,
        left_lat=0.0,
        left_lon=0.0,
        right_time_ns=7_200_000_000_000,
        right_lat=0.0,
        right_lon=20.0,
        radius_km=500.0,
    )
    assert result.tolist() == [False]


def test_every_combination_gets_a_finite_conservative_upper_and_unbounded_hours() -> None:
    u1 = pd.DataFrame({
        "origin_city": ["AAA:A", "CCC:C"],
        "dest_city": ["BBB:B", "DDD:D"],
        "shiptype_group": ["container", "tanker"],
        "tc300_hours_observed": [1.0, 0.0],
        "tc500_hours_observed": [2.0, 0.0],
    })
    combos, report = TC._merge_combo_stats(u1, {
        "tc300_possible_by_combo": {
            ("AAA:A", "BBB:B", "container"): {
                "segments": 2, "unknown_weight_hours": 0.5,
            },
        },
        "tc500_possible_by_combo": {
            ("AAA:A", "BBB:B", "container"): {
                "segments": 3, "unknown_weight_hours": 1.0,
            },
        },
        "unbounded_by_combo": [{
            "origin_city": "AAA:A",
            "dest_city": "BBB:B",
            "shiptype_group": "container",
            "unbounded_voyages": 1,
            "unbounded_unknown_hours": 4.0,
        }],
    })
    assert len(combos) == 2
    ab = combos.loc[combos.origin_city.eq("AAA:A")].iloc[0]
    assert ab.tc300_hours_lower == 1.0
    assert ab.tc300_hours_upper == 5.5
    assert ab.tc500_hours_upper == 7.0
    assert ab.unbounded_voyages == 1
    assert ab.unbounded_unknown_hours == 4.0
    assert combos.tc300_hours_lower.le(combos.tc300_hours_upper).all()
    assert combos.tc500_hours_lower.le(combos.tc500_hours_upper).all()
    assert report["checks"]["tc500_lower_exceeds_upper"] == 0
