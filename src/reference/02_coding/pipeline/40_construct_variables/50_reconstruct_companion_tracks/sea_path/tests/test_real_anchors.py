"""Tests for verified real-anchor ingestion and per-leg reconstruction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sea_path.real_anchors import (
    ANCHOR_ADMISSIBLE,
    ANCHOR_MOTION_CONFLICT,
    ANCHOR_NO_PRESENCE,
    ANCHOR_SOURCE_REAL,
    CONSUMER_HOUR_BOUNDARY,
    CONSUMER_HOUR_INTERIOR,
    AnchorWindow,
    GapTask,
    classify_consumer_hour_scope,
    make_gap_tasks,
    reconstruct_real_anchor_task,
    solve_interval_anchor_schedule,
    verified_anchor_audit_path,
)
from sea_path.routing import ROUTE_ANCHOR_NOT_NAVIGABLE, ROUTE_RESOLVED


HOUR = 3600.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completed_manifest_path_is_followed_into_offline_recovery(tmp_path: Path):
    sidecar = tmp_path / "blocks/2024-04-23/06_12/offline_geometry_overreturn_recovery"
    sidecar.mkdir(parents=True)
    audit = sidecar / "anchor_hour_audit.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(audit, index=False)
    manifest = sidecar / "completed_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "products_modified": 0,
                "response": {"anchor_audit_path": str(audit)},
            }
        ),
        encoding="utf-8",
    )
    block = {
        "completed_manifest": str(manifest),
        "completed_manifest_sha256": sha256(manifest),
    }
    resolved_manifest, resolved_audit = verified_anchor_audit_path(tmp_path, block)
    assert resolved_manifest == manifest
    assert resolved_audit == audit


def test_completed_manifest_cannot_escape_run_root(tmp_path: Path):
    outside = tmp_path.parent / "outside-manifest.json"
    outside.write_text("{}", encoding="utf-8")
    block = {
        "completed_manifest": str(outside),
        "completed_manifest_sha256": sha256(outside),
    }
    with pytest.raises(ValueError, match="outside its run root"):
        verified_anchor_audit_path(tmp_path, block)


def _gap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gap_id": ["gap_a"],
            "candidate_id": ["candidate_a"],
            "mmsi": ["123456789"],
            "vessel_uid_clean": ["ship_a"],
            "route": ["suez"],
            "gap_type": ["internal_bounded"],
            "bounded": [True],
            "source_gap_start_utc": [pd.Timestamp("2024-05-01T00:00:00Z")],
            "source_gap_end_utc": [pd.Timestamp("2024-05-01T12:00:00Z")],
            "gap_start_utc": [pd.Timestamp("2024-05-01T00:00:00Z")],
            "gap_end_utc": [pd.Timestamp("2024-05-01T12:00:00Z")],
            "dark_hours_in_probe": [12.0],
            "full_source_gap_hours": [12.0],
            "lat_a": [20.0],
            "lon_a": [38.0],
            "lat_b": [21.2],
            "lon_b": [37.9],
            "reconstruction_scope": ["gfw_anchor_candidate"],
            "phase_overlap_hours": [12.0],
        }
    )


def _hour_frame() -> pd.DataFrame:
    hours = pd.to_datetime(
        [
            "2024-05-01T05:00:00Z",
            "2024-05-01T06:00:00Z",
            "2024-05-01T07:00:00Z",
        ],
        utc=True,
    )
    statuses = [ANCHOR_NO_PRESENCE, ANCHOR_ADMISSIBLE, ANCHOR_MOTION_CONFLICT]
    return pd.DataFrame(
        {
            "target_hour_id": [f"123456789|{x.isoformat()}" for x in hours],
            "gap_id": "gap_a",
            "candidate_id": "candidate_a",
            "mmsi": "123456789",
            "route": "suez",
            "hour_utc": hours,
            "source_gap_start_utc": pd.Timestamp("2024-05-01T00:00:00Z"),
            "source_gap_end_utc": pd.Timestamp("2024-05-01T12:00:00Z"),
            "lat_a": 20.0,
            "lon_a": 38.0,
            "lat_b": 21.2,
            "lon_b": 37.9,
            "admissible_unique_anchor": [False, True, False],
            "anchor_status": statuses,
            "anchor_lat": [np.nan, 20.6, np.nan],
            "anchor_lon": [np.nan, 37.95, np.nan],
        }
    )


def test_tasks_keep_rejections_separate_from_admissible_anchor():
    tasks, metadata = make_gap_tasks(_gap_frame(), _hour_frame())
    assert len(tasks) == 1
    task = tasks[0]
    assert len(task.anchors) == 1
    assert task.anchors[0].source == ANCHOR_SOURCE_REAL
    assert dict(task.upstream_status_counts) == {
        "admissible_unique_anchor": 1,
        "exact_identity_time_motion_conflict": 1,
        "multiple_feasible_grid_centres": 0,
        "no_exact_identity_time_presence": 1,
    }
    assert metadata.iloc[0].to_dict() == {
        "gap_id": "gap_a",
        "acquisition_query_hours": 3,
        "query_target_hours": 3,
        "boundary_query_hours": 0,
        "acquisition_admissible_anchor_hours": 1,
        "admissible_anchor_hours": 1,
        "usable_anchor_windows": 1,
        "boundary_admissible_anchor_hours": 0,
        "tasked": True,
    }


def test_observed_boundary_hour_stays_in_acquisition_but_not_consumer_scope():
    hours = _hour_frame()
    hours.loc[0, "hour_utc"] = hours.loc[0, "source_gap_start_utc"]
    classified = classify_consumer_hour_scope(hours)
    assert list(classified["consumer_hour_scope"]) == [
        CONSUMER_HOUR_BOUNDARY,
        CONSUMER_HOUR_INTERIOR,
        CONSUMER_HOUR_INTERIOR,
    ]
    tasks, metadata = make_gap_tasks(_gap_frame(), classified)
    assert len(tasks) == 1
    assert len(tasks[0].target_times) == 2
    assert len(tasks[0].anchors) == 1
    assert metadata.iloc[0]["acquisition_query_hours"] == 3
    assert metadata.iloc[0]["query_target_hours"] == 2
    assert metadata.iloc[0]["boundary_query_hours"] == 1


def test_boundary_hour_anchor_constrains_route_without_becoming_output_hour():
    hours = _hour_frame()
    hours.loc[0, "hour_utc"] = hours.loc[0, "source_gap_start_utc"]
    hours.loc[0, "admissible_unique_anchor"] = True
    hours.loc[0, "anchor_status"] = ANCHOR_ADMISSIBLE
    hours.loc[0, "anchor_lat"] = 20.1
    hours.loc[0, "anchor_lon"] = 38.0
    tasks, metadata = make_gap_tasks(
        _gap_frame(), classify_consumer_hour_scope(hours)
    )
    assert tasks[0].target_times[0] == pd.Timestamp("2024-05-01T06:00:00Z").timestamp()
    assert tasks[0].anchors[0].hour_start_s == pd.Timestamp(
        "2024-05-01T00:00:00Z"
    ).timestamp()
    assert metadata.iloc[0]["boundary_admissible_anchor_hours"] == 1
    assert metadata.iloc[0]["usable_anchor_windows"] == 2


def test_real_anchor_task_emits_only_requested_phase_hours(router):
    task = make_gap_tasks(_gap_frame(), _hour_frame())[0][0]
    out = reconstruct_real_anchor_task(router, task)
    assert out["gap_outcome"] == ROUTE_RESOLVED
    assert out["emitted_target_hours"] == 3
    assert list(pd.to_datetime(out["times"], unit="s", utc=True)) == list(
        _hour_frame()["hour_utc"]
    )
    assert out["schedule_status"] == ROUTE_RESOLVED
    assert out["scheduled_anchor_windows"] == 1
    scheduled = out["anchor_records"][0]["scheduled_epoch_s"]
    assert pd.Timestamp("2024-05-01T06:00:00Z").timestamp() <= scheduled
    assert scheduled < pd.Timestamp("2024-05-01T07:00:00Z").timestamp()
    assert out["anchors_emitted"] + out["interpolated_positions"] == 3
    assert all(
        record["implied_speed_knots"] <= 25.0 + 1e-9
        for record in out["leg_records"]
        if np.isfinite(record["implied_speed_knots"])
    )


def test_land_anchor_is_not_silently_replaced_by_unanchored_route(router):
    task = GapTask(
        gap_id="land_anchor",
        candidate_id="candidate",
        mmsi="123456789",
        route="suez",
        t_start=0.0,
        lat_start=20.0,
        lon_start=38.0,
        t_end=12 * HOUR,
        lat_end=31.6,
        lon_end=32.35,
        target_times=tuple(x * HOUR for x in range(1, 12)),
        anchors=(
            AnchorWindow(
                target_hour_id="123456789|1970-01-01T06:00:00+00:00",
                hour_start_s=6 * HOUR,
                hour_end_s=7 * HOUR,
                lat=29.2,
                lon=32.6,
                source=ANCHOR_SOURCE_REAL,
            ),
        ),
        consumer_admissible_anchor_hours=1,
        upstream_status_counts=((ANCHOR_ADMISSIBLE, 1),),
    )
    out = reconstruct_real_anchor_task(router, task)
    assert out["gap_outcome"] == ROUTE_ANCHOR_NOT_NAVIGABLE
    assert out["leg_statuses"] == (
        ROUTE_ANCHOR_NOT_NAVIGABLE,
        ROUTE_ANCHOR_NOT_NAVIGABLE,
    )
    assert out["emitted_target_hours"] == 0
    assert out["missing_target_hours"] == 11
    assert len(out["positions"]) == 0


def _window(hour: float) -> AnchorWindow:
    return AnchorWindow(
        target_hour_id=f"anchor-{hour}",
        hour_start_s=hour * HOUR,
        hour_end_s=(hour + 1) * HOUR,
        lat=20.0,
        lon=38.0,
        source=ANCHOR_SOURCE_REAL,
    )


def test_schedule_moves_first_anchor_inside_hour_when_bin_start_is_too_early():
    schedule = solve_interval_anchor_schedule(
        9.75 * HOUR,
        12.0 * HOUR,
        [_window(10.0)],
        [0.5 * HOUR, 0.5 * HOUR],
    )
    assert schedule.status == ROUTE_RESOLVED
    assert 10.25 * HOUR <= schedule.scheduled_times[0] < 11.0 * HOUR
    assert schedule.scheduled_times[0] != 10.0 * HOUR


def test_schedule_satisfies_adjacent_leg_minimum_times_globally():
    schedule = solve_interval_anchor_schedule(
        9.0 * HOUR,
        13.0 * HOUR,
        [_window(10.0), _window(11.0)],
        [0.75 * HOUR, 1.25 * HOUR, 0.75 * HOUR],
    )
    assert schedule.status == ROUTE_RESOLVED
    nodes = np.concatenate(([9.0 * HOUR], schedule.scheduled_times, [13.0 * HOUR]))
    assert np.all(np.diff(nodes) >= np.asarray([0.75, 1.25, 0.75]) * HOUR)


def test_schedule_reports_the_first_leg_that_makes_hour_window_impossible():
    schedule = solve_interval_anchor_schedule(
        9.75 * HOUR,
        12.0 * HOUR,
        [_window(10.0)],
        [1.3 * HOUR, 0.1 * HOUR],
    )
    assert schedule.status == "speed_infeasible"
    assert schedule.failure_leg_index == 0
    assert np.isnan(schedule.scheduled_times).all()


def test_schedule_rejects_overlapping_or_out_of_gap_windows():
    with pytest.raises(ValueError, match="outside"):
        solve_interval_anchor_schedule(
            10.5 * HOUR,
            13.0 * HOUR,
            [_window(10.0)],
            [0.0, 0.0],
        )


def test_schedule_tolerates_only_epoch_scale_floating_roundoff():
    base = pd.Timestamp("2024-05-01T00:00:00Z").timestamp()
    rng = np.random.default_rng(20260902)
    anchors = [
        AnchorWindow(
            target_hour_id=f"anchor-{hour}",
            hour_start_s=base + hour * HOUR,
            hour_end_s=base + (hour + 1) * HOUR,
            lat=20.0,
            lon=38.0,
            source=ANCHOR_SOURCE_REAL,
        )
        for hour in range(1, 6)
    ]
    for _ in range(100):
        witness = np.asarray(
            [base + 300.0]
            + [base + hour * HOUR + float(rng.uniform(60.0, 3540.0)) for hour in range(1, 6)]
            + [base + 7 * HOUR - 300.0]
        )
        minimum = np.diff(witness) * rng.uniform(0.7, 0.999999, size=6)
        schedule = solve_interval_anchor_schedule(
            witness[0], witness[-1], anchors, minimum
        )
        assert schedule.status == ROUTE_RESOLVED
        nodes = np.concatenate(([witness[0]], schedule.scheduled_times, [witness[-1]]))
        assert (np.diff(nodes) + 1e-5 >= minimum).all()
