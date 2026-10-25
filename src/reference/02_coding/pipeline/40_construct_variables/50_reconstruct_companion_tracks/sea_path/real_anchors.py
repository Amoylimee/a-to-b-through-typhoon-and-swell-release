"""Verified ingestion and routing tasks for real phase-one GFW anchors.

The phase-one acquisition has one exceptional block whose accepted audit lives
under an offline-recovery sidecar.  Consumers must therefore follow the
completed-manifest paths recorded in ``status.json`` rather than globbing the
usual block layout.  This module also keeps rejected Presence evidence separate
from admissible anchors and carries every route-leg status into the output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .companion_track import (
    GAP_RECONSTRUCTED,
    POSITION_ANCHOR,
    POSITION_INTERPOLATED,
    classify_gap,
)
from .routing import (
    DEFAULT_MAX_SPEED_KNOTS,
    DEFAULT_ROUTE_TIMEOUT_S,
    KM_PER_NAUTICAL_MILE,
    ROUTE_RESOLVED,
    ROUTE_SPEED_INFEASIBLE,
    SeaPathRouter,
    interpolate_along_path,
)


ANCHOR_SOURCE_REAL = "gfw_presence_v4_phase_one_admissible_grid_centre"

ANCHOR_ADMISSIBLE = "admissible_unique_anchor"
ANCHOR_MOTION_CONFLICT = "exact_identity_time_motion_conflict"
ANCHOR_MULTIPLE_CENTRES = "multiple_feasible_grid_centres"
ANCHOR_NO_PRESENCE = "no_exact_identity_time_presence"
ANCHOR_STATUSES = (
    ANCHOR_ADMISSIBLE,
    ANCHOR_MOTION_CONFLICT,
    ANCHOR_MULTIPLE_CENTRES,
    ANCHOR_NO_PRESENCE,
)

EXPECTED_PHASE_ONE_TOTALS = {
    "target_hours": 4_141_648,
    "gfw_query_target_hours": 4_139_832,
    "unbracketed_target_hours": 1_816,
    "admissible_unique_anchor_hours": 3_459_782,
    "motion_conflict_hours": 11_626,
    "multiple_feasible_grid_centre_hours": 11,
}

GAP_OUTCOME_PARTIAL = "partial"
GAP_OUTCOME_MULTIPLE_FAILURES = "multiple_failures"
CONSUMER_HOUR_INTERIOR = "interior_target_hour"
CONSUMER_HOUR_BOUNDARY = "observed_boundary_hour_excluded"

REQUIRED_ANCHOR_COLUMNS = (
    "target_hour_id",
    "gap_id",
    "candidate_id",
    "mmsi",
    "route",
    "hour_utc",
    "source_gap_start_utc",
    "source_gap_end_utc",
    "lat_a",
    "lon_a",
    "lat_b",
    "lon_b",
    "admissible_unique_anchor",
    "anchor_status",
    "anchor_lat",
    "anchor_lon",
)


class GapTask(NamedTuple):
    """One bounded 4--24 hour source gap and its phase-one evidence."""

    gap_id: str
    candidate_id: str
    mmsi: str
    route: str
    t_start: float
    lat_start: float
    lon_start: float
    t_end: float
    lat_end: float
    lon_end: float
    target_times: tuple[float, ...]
    anchors: tuple["AnchorWindow", ...]
    consumer_admissible_anchor_hours: int
    upstream_status_counts: tuple[tuple[str, int], ...]


class AnchorWindow(NamedTuple):
    """One GFW grid centre known only to occur within its UTC hour."""

    target_hour_id: str
    hour_start_s: float
    hour_end_s: float
    lat: float
    lon: float
    source: str


class AnchorSchedule(NamedTuple):
    """Globally feasible anchor times, or the leg proving infeasibility."""

    status: str
    scheduled_times: np.ndarray
    earliest_times: np.ndarray
    latest_times: np.ndarray
    failure_leg_index: int | None


def solve_interval_anchor_schedule(
    t_start: float,
    t_end: float,
    anchors: Sequence[AnchorWindow],
    minimum_leg_seconds: Sequence[float],
) -> AnchorSchedule:
    """Schedule hourly evidence without pretending its bin label is a fix time.

    Each route leg contributes a minimum travel time at the frozen speed cap.
    A forward pass finds the earliest globally feasible time for every anchor;
    a backward pass finds the latest.  Their component-wise midpoint remains a
    feasible schedule because both bounding schedules satisfy every adjacent
    minimum-time inequality.  This centres each anchor in its *global feasible
    envelope*, rather than assigning an arbitrary minute within the hour.
    """

    if not np.isfinite([t_start, t_end]).all() or t_end <= t_start:
        raise ValueError("gap boundaries must be finite and strictly ordered")
    ordered = tuple(anchors)
    minimum = np.asarray(minimum_leg_seconds, dtype=float)
    if minimum.shape != (len(ordered) + 1,):
        raise ValueError("minimum leg-time count must equal anchor count plus one")
    if not np.isfinite(minimum).all() or (minimum < 0.0).any():
        raise ValueError("minimum leg times must be finite and non-negative")
    # Epoch seconds are around 1.7e9 in 2024, so arithmetic has sub-microsecond
    # quantization.  Add a scale-derived numerical guard to positive travel
    # times; this makes the saved schedule strictly no faster than 25 knots.
    epoch_roundoff_s = max(abs(t_start), abs(t_end), 1.0) * np.finfo(float).eps * 32.0
    guarded_minimum = minimum + np.where(minimum > 0.0, epoch_roundoff_s, 0.0)

    starts = np.asarray([anchor.hour_start_s for anchor in ordered], dtype=float)
    ends = np.asarray([anchor.hour_end_s for anchor in ordered], dtype=float)
    if len(ordered):
        if not np.isfinite(starts).all() or not np.isfinite(ends).all():
            raise ValueError("anchor windows must be finite")
        if (ends <= starts).any():
            raise ValueError("anchor windows must have positive width")
        if (starts < t_start).any() or (ends > t_end).any():
            raise ValueError("anchor window lies outside the source gap")
        if len(ordered) > 1 and (starts[1:] < ends[:-1]).any():
            raise ValueError("anchor windows must be ordered and non-overlapping")

    # Presence bins are half-open [hour_start, hour_end).  nextafter gives the
    # largest representable epoch second still inside each bin, avoiding an
    # invented epsilon and preventing a schedule from slipping into the next.
    upper = np.nextafter(ends, -np.inf)
    earliest = np.full(len(ordered), np.nan, dtype=float)
    previous = float(t_start)
    for index, lower in enumerate(starts):
        value = max(float(lower), previous + float(guarded_minimum[index]))
        if value > float(upper[index]):
            return AnchorSchedule(
                ROUTE_SPEED_INFEASIBLE,
                np.full(len(ordered), np.nan, dtype=float),
                earliest,
                np.full(len(ordered), np.nan, dtype=float),
                index,
            )
        earliest[index] = value
        previous = value
    if previous + float(guarded_minimum[-1]) > t_end:
        return AnchorSchedule(
            ROUTE_SPEED_INFEASIBLE,
            np.full(len(ordered), np.nan, dtype=float),
            earliest,
            np.full(len(ordered), np.nan, dtype=float),
            len(ordered),
        )

    latest = np.full(len(ordered), np.nan, dtype=float)
    following = float(t_end)
    for index in range(len(ordered) - 1, -1, -1):
        value = min(float(upper[index]), following - float(guarded_minimum[index + 1]))
        if value < float(starts[index]):
            raise AssertionError("forward-feasible schedule failed its backward pass")
        latest[index] = value
        following = value
    if len(ordered) and (earliest > latest).any():
        raise AssertionError("earliest anchor time exceeds latest anchor time")

    scheduled = (earliest + latest) / 2.0
    node_times = np.concatenate(([t_start], scheduled, [t_end]))
    shortfall = minimum - np.diff(node_times)
    if (shortfall > 0.0).any():
        raise AssertionError(
            "constructed anchor schedule violates the speed ceiling: "
            f"maximum shortfall {float(shortfall.max())} seconds after "
            f"a {epoch_roundoff_s}-second numerical guard"
        )
    if len(ordered) and (
        (scheduled < starts).any() or (scheduled >= ends).any()
    ):
        raise AssertionError("constructed anchor time lies outside its evidence hour")
    return AnchorSchedule(
        ROUTE_RESOLVED,
        scheduled,
        earliest,
        latest,
        None,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"phase-one manifest points outside its run root: {resolved}") from exc
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_anchor_audit_path(
    run_root: Path,
    block_status: dict[str, Any],
) -> tuple[Path, Path]:
    """Resolve one block's accepted manifest and audit, including sidecars."""

    run_root = run_root.resolve()
    manifest_path = _inside(Path(block_status["completed_manifest"]), run_root)
    manifest_hash = file_sha256(manifest_path)
    if manifest_hash != block_status.get("completed_manifest_sha256"):
        raise ValueError(f"completed-manifest hash changed: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("products_modified") != 0:
        raise ValueError(f"completed manifest is not accepted read-only evidence: {manifest_path}")
    anchor_path = _inside(Path(manifest["response"]["anchor_audit_path"]), run_root)
    if not anchor_path.is_file():
        raise FileNotFoundError(anchor_path)
    return manifest_path, anchor_path


def load_verified_phase_one_anchors(
    run_root: Path,
    *,
    routes: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load all 440 accepted anchor audits through their recorded manifests.

    The returned frame is optionally route-filtered, while the audit always
    records closure over the full authorized acquisition.
    """

    run_root = run_root.resolve()
    status_path = run_root / "status.json"
    summary_path = run_root / "summary.json"
    run_manifest_path = run_root / "run_manifest.json"
    for path in (status_path, summary_path, run_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    status = _load_json(status_path)
    summary = _load_json(summary_path)
    run_manifest = _load_json(run_manifest_path)
    required_state = {
        "state": "completed",
        "authorized_blocks": 440,
        "completed_blocks": 440,
        "failed_blocks": 0,
        "pending_blocks": 0,
        "products_modified": 0,
        "phase_two_authorized": False,
    }
    for key, expected in required_state.items():
        if status.get(key) != expected:
            raise ValueError(
                f"phase-one status mismatch for {key}: "
                f"expected {expected!r}, got {status.get(key)!r}"
            )
    if summary.get("state") != "completed" or summary.get("products_modified") != 0:
        raise ValueError("phase-one summary is not a completed read-only run")
    plan = summary.get("plan") or {}
    if plan.get("expected_spatial_reports") != 494:
        raise ValueError("phase-one expected spatial-report count changed")
    if plan.get("spatial_reports") != 494:
        raise ValueError("phase-one spatial reports do not close to 494")
    for key in ("target_hours", "gfw_query_target_hours", "unbracketed_target_hours"):
        if int(plan.get(key, -1)) != EXPECTED_PHASE_ONE_TOTALS[key]:
            raise ValueError(f"phase-one frozen total changed for {key}")
    # This manifest is the immutable pre-run plan gate.  Completion belongs to
    # status.json and summary.json; rewriting the frozen manifest at closeout
    # would erase the identity the runner was launched against.
    if (
        run_manifest.get("state") != "plan_gates_passed"
        or run_manifest.get("products_modified") != 0
    ):
        raise ValueError("phase-one run manifest is not the frozen read-only plan gate")

    blocks = status.get("blocks") or []
    if len(blocks) != 440:
        raise ValueError(f"phase-one status contains {len(blocks)} blocks, expected 440")
    ordinals = [int(item["block"]["ordinal"]) for item in blocks]
    if ordinals != list(range(440)):
        raise ValueError("phase-one blocks are not the exact ordered ordinals 0..439")

    frames: list[pd.DataFrame] = []
    file_records: list[dict[str, Any]] = []
    totals = {
        "query_target_hours": 0,
        "admissible_unique_anchor_hours": 0,
        "motion_conflict_hours": 0,
        "multiple_feasible_grid_centre_hours": 0,
    }
    for item in blocks:
        if item.get("status") != "completed":
            raise ValueError(f"phase-one block {item['block']['ordinal']} is not completed")
        manifest_path, anchor_path = verified_anchor_audit_path(run_root, item)
        table = pq.read_table(anchor_path, columns=list(REQUIRED_ANCHOR_COLUMNS))
        frame = table.to_pandas()
        expected_rows = int(item["gfw_query_target_hours"])
        if len(frame) != expected_rows:
            raise ValueError(
                f"anchor audit row mismatch for block {item['block']['ordinal']}: "
                f"{len(frame)} != {expected_rows}"
            )
        counts = frame["anchor_status"].value_counts().to_dict()
        expected_counts = {
            ANCHOR_ADMISSIBLE: int(item["admissible_unique_anchor_hours"]),
            ANCHOR_MOTION_CONFLICT: int(item["motion_conflict_hours"]),
            ANCHOR_MULTIPLE_CENTRES: int(item["multiple_feasible_grid_centre_hours"]),
            ANCHOR_NO_PRESENCE: int(item["absent_presence_hours"])
            - int(item["unbracketed_target_hours"]),
        }
        if {key: int(counts.get(key, 0)) for key in ANCHOR_STATUSES} != expected_counts:
            raise ValueError(f"anchor status counts changed in block {item['block']['ordinal']}")
        file_records.append(
            {
                "ordinal": int(item["block"]["ordinal"]),
                "path": str(anchor_path.relative_to(run_root)),
                "sha256": file_sha256(anchor_path),
                "rows": int(len(frame)),
            }
        )
        totals["query_target_hours"] += expected_rows
        totals["admissible_unique_anchor_hours"] += expected_counts[ANCHOR_ADMISSIBLE]
        totals["motion_conflict_hours"] += expected_counts[ANCHOR_MOTION_CONFLICT]
        totals["multiple_feasible_grid_centre_hours"] += expected_counts[
            ANCHOR_MULTIPLE_CENTRES
        ]
        frames.append(frame)

    hours = pd.concat(frames, ignore_index=True)
    if hours["target_hour_id"].duplicated().any():
        raise ValueError("target_hour_id is not globally unique across phase one")
    if hours.duplicated(["mmsi", "hour_utc"]).any():
        raise ValueError("MMSI/hour is not globally unique across phase one")
    if len(hours) != int(plan.get("gfw_query_target_hours", -1)):
        raise ValueError("phase-one anchor audits do not close to the plan denominator")
    if totals["admissible_unique_anchor_hours"] != int(
        hours["admissible_unique_anchor"].sum()
    ):
        raise AssertionError("admissible-anchor total disagrees with the concatenated audits")
    for key in (
        "admissible_unique_anchor_hours",
        "motion_conflict_hours",
        "multiple_feasible_grid_centre_hours",
    ):
        if totals[key] != EXPECTED_PHASE_ONE_TOTALS[key]:
            raise ValueError(f"phase-one accepted anchor total changed for {key}")

    digest_payload = "".join(
        f"{r['ordinal']}|{r['path']}|{r['sha256']}|{r['rows']}\n" for r in file_records
    ).encode("utf-8")
    input_digest = hashlib.sha256(digest_payload).hexdigest()
    audit: dict[str, Any] = {
        "run_root": str(run_root),
        "status_sha256": file_sha256(status_path),
        "summary_sha256": file_sha256(summary_path),
        "run_manifest_sha256": file_sha256(run_manifest_path),
        "source_target_manifest_sha256": run_manifest[
            "source_target_manifest_sha256"
        ],
        "authorized_blocks": 440,
        "anchor_audit_files": 440,
        "successful_spatial_reports": 494,
        "phase_start_utc": blocks[0]["block"]["start"],
        "phase_end_utc": blocks[-1]["block"]["end"],
        "planned_target_hours": int(plan["target_hours"]),
        "query_target_hours": int(plan["gfw_query_target_hours"]),
        "unbracketed_target_hours": int(plan["unbracketed_target_hours"]),
        "admissible_unique_anchor_hours": totals[
            "admissible_unique_anchor_hours"
        ],
        "motion_conflict_hours": totals["motion_conflict_hours"],
        "multiple_feasible_grid_centre_hours": totals[
            "multiple_feasible_grid_centre_hours"
        ],
        "anchor_audit_set_sha256": input_digest,
        "offline_recovery_audits_followed": int(
            sum("offline_geometry_overreturn_recovery" in r["path"] for r in file_records)
        ),
    }
    if routes is not None:
        wanted = set(routes)
        hours = hours.loc[hours["route"].isin(wanted)].reset_index(drop=True)
        audit["routes_loaded"] = sorted(wanted)
        audit["route_filtered_query_hours"] = int(len(hours))
        audit["route_filtered_admissible_anchor_hours"] = int(
            hours["admissible_unique_anchor"].sum()
        )
    return hours, audit


def load_phase_one_gap_scope(
    target_root: Path,
    *,
    expected_manifest_sha256: str,
    phase_start: pd.Timestamp,
    phase_end: pd.Timestamp,
    routes: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load target gaps that overlap the authorized phase-one interval."""

    target_root = target_root.resolve()
    manifest_path = target_root / "anchor_target_manifest.json"
    actual_manifest_sha256 = file_sha256(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("frozen anchor-target manifest hash changed")
    paths = sorted(target_root.glob("gaps/bucket_*.parquet"))
    if len(paths) != 64:
        raise ValueError(f"anchor target contains {len(paths)} gap shards, expected 64")
    columns = [
        "gap_id",
        "candidate_id",
        "mmsi",
        "vessel_uid_clean",
        "route",
        "gap_type",
        "bounded",
        "source_gap_start_utc",
        "source_gap_end_utc",
        "gap_start_utc",
        "gap_end_utc",
        "dark_hours_in_probe",
        "full_source_gap_hours",
        "lat_a",
        "lon_a",
        "lat_b",
        "lon_b",
        "reconstruction_scope",
    ]
    parts = [pq.read_table(path, columns=columns).to_pandas() for path in paths]
    gaps = pd.concat(parts, ignore_index=True)
    if gaps["gap_id"].duplicated().any():
        raise ValueError("gap_id is not globally unique in the frozen anchor target")
    for name in (
        "source_gap_start_utc",
        "source_gap_end_utc",
        "gap_start_utc",
        "gap_end_utc",
    ):
        gaps[name] = pd.to_datetime(gaps[name], utc=True)
    phase_start = pd.Timestamp(phase_start)
    phase_end = pd.Timestamp(phase_end)
    if phase_start.tzinfo is None:
        phase_start = phase_start.tz_localize("UTC")
    else:
        phase_start = phase_start.tz_convert("UTC")
    if phase_end.tzinfo is None:
        phase_end = phase_end.tz_localize("UTC")
    else:
        phase_end = phase_end.tz_convert("UTC")
    if phase_end <= phase_start:
        raise ValueError("phase-one end must follow its start")
    wanted = set(routes)
    gaps = gaps.loc[
        gaps["route"].isin(wanted)
        & gaps["gap_end_utc"].gt(phase_start)
        & gaps["gap_start_utc"].lt(phase_end)
    ].copy()
    overlap_start = gaps["gap_start_utc"].where(
        gaps["gap_start_utc"].gt(phase_start), phase_start
    )
    overlap_end = gaps["gap_end_utc"].where(
        gaps["gap_end_utc"].lt(phase_end), phase_end
    )
    gaps["phase_overlap_hours"] = (
        overlap_end - overlap_start
    ).dt.total_seconds() / 3600.0
    if gaps["phase_overlap_hours"].le(0).any():
        raise AssertionError("selected phase-one gap has no positive overlap")
    shard_records = [
        {
            "path": str(path.relative_to(target_root)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]
    digest_payload = "".join(
        f"{r['path']}|{r['sha256']}|{r['bytes']}\n" for r in shard_records
    ).encode("utf-8")
    audit = {
        "target_root": str(target_root),
        "anchor_target_manifest_sha256": actual_manifest_sha256,
        "gap_shards": 64,
        "gap_shard_set_sha256": hashlib.sha256(digest_payload).hexdigest(),
        "phase_start_utc": phase_start.isoformat(),
        "phase_end_utc": phase_end.isoformat(),
        "routes_loaded": sorted(wanted),
        "phase_overlap_gap_rows": int(len(gaps)),
        "phase_overlap_hours": float(gaps["phase_overlap_hours"].sum()),
    }
    return gaps.reset_index(drop=True), audit


def classify_consumer_hour_scope(anchor_hours: pd.DataFrame) -> pd.DataFrame:
    """Separate hourly output instants from interval-censored anchor evidence.

    A Presence label can equal an observed source-gap start because it denotes
    the full following hour.  That hour remains a usable route constraint, but
    the already-observed boundary instant is excluded from the companion output
    grid.  Every acquired bin must be fully contained in its source gap.
    """

    required = {"hour_utc", "source_gap_start_utc", "source_gap_end_utc"}
    missing = sorted(required.difference(anchor_hours.columns))
    if missing:
        raise ValueError(f"anchor-hour table is missing boundary columns: {missing}")
    result = anchor_hours.copy()
    for name in required:
        result[name] = pd.to_datetime(result[name], utc=True)
    hour_end = result["hour_utc"] + pd.Timedelta(hours=1)
    window_usable = result["hour_utc"].ge(result["source_gap_start_utc"]) & hour_end.le(
        result["source_gap_end_utc"]
    )
    if (~window_usable).any():
        raise ValueError("an acquisition evidence hour is not fully inside its source gap")
    interior = result["hour_utc"].gt(result["source_gap_start_utc"]) & result[
        "hour_utc"
    ].lt(result["source_gap_end_utc"])
    boundary = result["hour_utc"].eq(result["source_gap_start_utc"])
    if (~interior & ~boundary).any():
        raise ValueError("an acquisition hour has unknown consumer scope")
    result["consumer_target_hour"] = interior
    result["anchor_window_usable"] = window_usable
    result["consumer_hour_scope"] = np.where(
        interior, CONSUMER_HOUR_INTERIOR, CONSUMER_HOUR_BOUNDARY
    )
    return result


def make_gap_tasks(
    gaps: pd.DataFrame,
    anchor_hours: pd.DataFrame,
) -> tuple[list[GapTask], pd.DataFrame]:
    """Join verified hourly evidence to bounded target gaps and make tasks."""

    if anchor_hours["target_hour_id"].duplicated().any():
        raise ValueError("target_hour_id must be unique before task construction")
    known = set(gaps["gap_id"].astype(str))
    unknown = sorted(set(anchor_hours["gap_id"].astype(str)).difference(known))
    if unknown:
        raise ValueError(f"phase-one anchors refer to {len(unknown)} unknown target gaps")
    eligible = gaps.loc[gaps["reconstruction_scope"].eq("gfw_anchor_candidate")].copy()
    by_gap = {str(key): block for key, block in anchor_hours.groupby("gap_id", sort=False)}
    tasks: list[GapTask] = []
    metadata: list[dict[str, Any]] = []
    for row in eligible.itertuples(index=False):
        evidence = by_gap.get(str(row.gap_id))
        if evidence is None:
            metadata.append(
                {
                    "gap_id": str(row.gap_id),
                    "acquisition_query_hours": 0,
                    "query_target_hours": 0,
                    "boundary_query_hours": 0,
                    "acquisition_admissible_anchor_hours": 0,
                    "admissible_anchor_hours": 0,
                    "usable_anchor_windows": 0,
                    "boundary_admissible_anchor_hours": 0,
                    "tasked": False,
                }
            )
            continue
        ordered_all = evidence.sort_values("hour_utc", kind="stable")
        if "consumer_target_hour" not in ordered_all:
            ordered_all = classify_consumer_hour_scope(ordered_all)
        if ordered_all["candidate_id"].astype(str).nunique() != 1:
            raise ValueError(f"gap {row.gap_id} carries multiple candidate IDs")
        if str(ordered_all["candidate_id"].iloc[0]) != str(row.candidate_id):
            raise ValueError(f"gap {row.gap_id} candidate ID changed")
        if not ordered_all["route"].eq(row.route).all():
            raise ValueError(f"gap {row.gap_id} route changed")
        if not ordered_all["source_gap_start_utc"].eq(row.source_gap_start_utc).all():
            raise ValueError(f"gap {row.gap_id} start boundary changed")
        if not ordered_all["source_gap_end_utc"].eq(row.source_gap_end_utc).all():
            raise ValueError(f"gap {row.gap_id} end boundary changed")
        ordered = ordered_all.loc[ordered_all["consumer_target_hour"]]
        boundary = ordered_all.loc[~ordered_all["consumer_target_hour"]]
        if ordered.empty:
            metadata.append(
                {
                    "gap_id": str(row.gap_id),
                    "acquisition_query_hours": int(len(ordered_all)),
                    "query_target_hours": 0,
                    "boundary_query_hours": int(len(boundary)),
                    "acquisition_admissible_anchor_hours": int(
                        ordered_all["admissible_unique_anchor"].sum()
                    ),
                    "admissible_anchor_hours": 0,
                    "usable_anchor_windows": 0,
                    "boundary_admissible_anchor_hours": int(
                        boundary["admissible_unique_anchor"].sum()
                    ),
                    "tasked": False,
                }
            )
            continue
        admitted = ordered_all.loc[
            ordered_all["anchor_window_usable"]
            & ordered_all["admissible_unique_anchor"]
        ]
        consumer_admitted = ordered.loc[ordered["admissible_unique_anchor"]]
        anchors = tuple(
            AnchorWindow(
                target_hour_id=str(target_hour_id),
                hour_start_s=float(stamp.timestamp()),
                hour_end_s=float(stamp.timestamp() + 3600.0),
                lat=float(lat),
                lon=float(lon),
                source=ANCHOR_SOURCE_REAL,
            )
            for target_hour_id, stamp, lat, lon in admitted[
                ["target_hour_id", "hour_utc", "anchor_lat", "anchor_lon"]
            ].itertuples(index=False, name=None)
        )
        status_counts = ordered["anchor_status"].value_counts().to_dict()
        task = GapTask(
            gap_id=str(row.gap_id),
            candidate_id=str(row.candidate_id),
            mmsi=str(row.mmsi),
            route=str(row.route),
            t_start=float(row.source_gap_start_utc.timestamp()),
            lat_start=float(row.lat_a),
            lon_start=float(row.lon_a),
            t_end=float(row.source_gap_end_utc.timestamp()),
            lat_end=float(row.lat_b),
            lon_end=float(row.lon_b),
            target_times=tuple(float(x.timestamp()) for x in ordered["hour_utc"]),
            anchors=anchors,
            consumer_admissible_anchor_hours=int(len(consumer_admitted)),
            upstream_status_counts=tuple(
                (status, int(status_counts.get(status, 0))) for status in ANCHOR_STATUSES
            ),
        )
        tasks.append(task)
        metadata.append(
            {
                "gap_id": str(row.gap_id),
                "acquisition_query_hours": int(len(ordered_all)),
                "query_target_hours": int(len(ordered)),
                "boundary_query_hours": int(len(boundary)),
                "acquisition_admissible_anchor_hours": int(
                    ordered_all["admissible_unique_anchor"].sum()
                ),
                "admissible_anchor_hours": int(len(consumer_admitted)),
                "usable_anchor_windows": int(len(anchors)),
                "boundary_admissible_anchor_hours": int(
                    boundary["admissible_unique_anchor"].sum()
                ),
                "tasked": True,
            }
        )
    return tasks, pd.DataFrame(metadata)


def reconstruct_real_anchor_task(router: SeaPathRouter, task: GapTask) -> dict[str, Any]:
    """Route and globally schedule one real-anchor task under the frozen cap."""

    duration_h = (task.t_end - task.t_start) / 3600.0
    if classify_gap(duration_h) != GAP_RECONSTRUCTED:
        raise ValueError("a real-anchor task lies outside the 4--24 hour policy band")
    ordered = tuple(sorted(task.anchors, key=lambda anchor: anchor.hour_start_s))
    nodes = (
        [(task.t_start, task.lat_start, task.lon_start, "observed_ais_start")]
        + [
            (np.nan, anchor.lat, anchor.lon, "gfw_interval_anchor")
            for anchor in ordered
        ]
        + [(task.t_end, task.lat_end, task.lon_end, "observed_ais_end")]
    )
    geometry = []
    minimum_leg_seconds: list[float] = []
    for left, right in zip(nodes[:-1], nodes[1:]):
        _, lat_a, lon_a, _ = left
        _, lat_b, lon_b, _ = right
        resolution = router.resolve_path(
            lat_a,
            lon_a,
            lat_b,
            lon_b,
            snap=False,
            timeout_s=DEFAULT_ROUTE_TIMEOUT_S,
            duration_hours=None,
            max_speed_knots=DEFAULT_MAX_SPEED_KNOTS,
        )
        geometry.append(resolution)
        distance_km = (
            float(resolution.path.total_km)
            if resolution.status == ROUTE_RESOLVED and resolution.path is not None
            else float(router.haversine_km(lat_a, lon_a, lat_b, lon_b))
        )
        minimum_leg_seconds.append(
            distance_km
            / (DEFAULT_MAX_SPEED_KNOTS * KM_PER_NAUTICAL_MILE)
            * 3600.0
        )

    schedule = solve_interval_anchor_schedule(
        task.t_start,
        task.t_end,
        ordered,
        minimum_leg_seconds,
    )
    statuses = [resolution.status for resolution in geometry]
    geometry_failed = any(status != ROUTE_RESOLVED for status in statuses)
    if schedule.status == ROUTE_SPEED_INFEASIBLE:
        failure_leg = schedule.failure_leg_index
        if (
            not geometry_failed
            and failure_leg is not None
            and statuses[failure_leg] == ROUTE_RESOLVED
        ):
            statuses[failure_leg] = ROUTE_SPEED_INFEASIBLE
        node_times = np.full(len(nodes), np.nan, dtype=float)
        times = np.empty(0, dtype=float)
        positions = np.empty((0, 2), dtype=float)
        leg_index = np.empty(0, dtype=int)
        position_source = np.empty(0, dtype=object)
        reconstructed_h = 0.0
        unresolved_h = duration_h
    else:
        node_times = np.concatenate(
            ([task.t_start], schedule.scheduled_times, [task.t_end])
        )
        target = np.asarray(task.target_times, dtype=float)
        target_set = set(target.tolist())
        time_parts: list[np.ndarray] = []
        position_parts: list[np.ndarray] = []
        leg_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        reconstructed_h = 0.0
        unresolved_h = 0.0
        for leg, resolution in enumerate(geometry):
            ta = float(node_times[leg])
            tb = float(node_times[leg + 1])
            leg_hours = (tb - ta) / 3600.0
            if resolution.status != ROUTE_RESOLVED or resolution.path is None:
                unresolved_h += leg_hours
                continue
            implied_knots = resolution.path.total_km / leg_hours / KM_PER_NAUTICAL_MILE
            if implied_knots > DEFAULT_MAX_SPEED_KNOTS + 1e-9:
                raise AssertionError("scheduled resolved leg exceeds the frozen speed cap")
            reconstructed_h += leg_hours
            if leg > 0 and ta in target_set:
                _, lat_a, lon_a, _ = nodes[leg]
                time_parts.append(np.asarray([ta], dtype=float))
                position_parts.append(np.asarray([[lat_a, lon_a]], dtype=float))
                leg_parts.append(np.asarray([leg], dtype=int))
                source_parts.append(np.asarray([POSITION_ANCHOR], dtype=object))
            targets = target[(target > ta) & (target < tb)]
            if len(targets):
                time_parts.append(targets)
                position_parts.append(
                    interpolate_along_path(resolution.path, ta, tb, targets)
                )
                leg_parts.append(np.full(len(targets), leg, dtype=int))
                source_parts.append(
                    np.full(len(targets), POSITION_INTERPOLATED, dtype=object)
                )
        times = np.concatenate(time_parts) if time_parts else np.empty(0, dtype=float)
        positions = (
            np.concatenate(position_parts)
            if position_parts
            else np.empty((0, 2), dtype=float)
        )
        leg_index = (
            np.concatenate(leg_parts) if leg_parts else np.empty(0, dtype=int)
        )
        position_source = (
            np.concatenate(source_parts)
            if source_parts
            else np.empty(0, dtype=object)
        )
        order = np.argsort(times, kind="mergesort")
        times = times[order]
        positions = positions[order]
        leg_index = leg_index[order]
        position_source = position_source[order]

    target = np.asarray(task.target_times, dtype=float)
    if len(np.unique(times)) != len(times):
        raise AssertionError(f"gap {task.gap_id} emitted duplicate target hours")
    if not set(times.tolist()).issubset(set(target.tolist())):
        raise AssertionError(f"gap {task.gap_id} emitted an out-of-scope hour")
    statuses = tuple(statuses)
    failed = tuple(status for status in statuses if status != ROUTE_RESOLVED)
    if schedule.status == ROUTE_SPEED_INFEASIBLE and not geometry_failed:
        outcome = ROUTE_SPEED_INFEASIBLE
    elif not failed:
        outcome = ROUTE_RESOLVED
    elif ROUTE_RESOLVED in statuses:
        outcome = GAP_OUTCOME_PARTIAL
    elif len(set(failed)) == 1:
        outcome = failed[0]
    else:
        outcome = GAP_OUTCOME_MULTIPLE_FAILURES
    status_counts = {status: int(statuses.count(status)) for status in set(statuses)}
    upstream = dict(task.upstream_status_counts)
    anchor_records = []
    for index, anchor in enumerate(ordered):
        anchor_records.append(
            {
                "gap_id": task.gap_id,
                "candidate_id": task.candidate_id,
                "mmsi": task.mmsi,
                "route": task.route,
                "anchor_index": index,
                "target_hour_id": anchor.target_hour_id,
                "hour_start_epoch_s": anchor.hour_start_s,
                "hour_end_epoch_s": anchor.hour_end_s,
                "scheduled_epoch_s": (
                    float(schedule.scheduled_times[index])
                    if schedule.status == ROUTE_RESOLVED
                    else np.nan
                ),
                "earliest_epoch_s": (
                    float(schedule.earliest_times[index])
                    if schedule.status == ROUTE_RESOLVED
                    else np.nan
                ),
                "latest_epoch_s": (
                    float(schedule.latest_times[index])
                    if schedule.status == ROUTE_RESOLVED
                    else np.nan
                ),
                "lat": anchor.lat,
                "lon": anchor.lon,
                "anchor_source": anchor.source,
                "schedule_status": schedule.status,
            }
        )
    leg_records = []
    for leg, resolution in enumerate(geometry):
        path_distance_km = (
            float(resolution.path.total_km)
            if resolution.status == ROUTE_RESOLVED and resolution.path is not None
            else np.nan
        )
        scheduled_start = float(node_times[leg])
        scheduled_end = float(node_times[leg + 1])
        scheduled_hours = (scheduled_end - scheduled_start) / 3600.0
        implied_knots = (
            path_distance_km / scheduled_hours / KM_PER_NAUTICAL_MILE
            if np.isfinite(path_distance_km) and np.isfinite(scheduled_hours)
            else np.nan
        )
        leg_records.append(
            {
                "gap_id": task.gap_id,
                "candidate_id": task.candidate_id,
                "mmsi": task.mmsi,
                "route": task.route,
                "leg_index": leg,
                "start_node_type": nodes[leg][3],
                "end_node_type": nodes[leg + 1][3],
                "geometry_status": resolution.status,
                "leg_status": statuses[leg],
                "path_distance_km": path_distance_km,
                "minimum_travel_seconds": float(minimum_leg_seconds[leg]),
                "minimum_time_basis": (
                    "sea_path" if np.isfinite(path_distance_km) else "great_circle_lower_bound"
                ),
                "scheduled_start_epoch_s": scheduled_start,
                "scheduled_end_epoch_s": scheduled_end,
                "scheduled_duration_hours": scheduled_hours,
                "implied_speed_knots": implied_knots,
                "max_speed_knots": DEFAULT_MAX_SPEED_KNOTS,
                "schedule_status": schedule.status,
            }
        )
    return {
        "gap_id": task.gap_id,
        "candidate_id": task.candidate_id,
        "mmsi": task.mmsi,
        "route": task.route,
        "gap_outcome": outcome,
        "leg_statuses": statuses,
        "leg_status_counts": status_counts,
        "n_legs": int(len(statuses)),
        "schedule_status": schedule.status,
        "schedule_failure_leg_index": schedule.failure_leg_index,
        "query_target_hours": int(len(target)),
        "admissible_anchor_hours": int(task.consumer_admissible_anchor_hours),
        "usable_anchor_windows": int(len(task.anchors)),
        "scheduled_anchor_windows": (
            int(len(task.anchors)) if schedule.status == ROUTE_RESOLVED else 0
        ),
        "emitted_target_hours": int(len(times)),
        "missing_target_hours": int(len(target) - len(times)),
        "anchors_emitted": int(np.sum(position_source == "anchor")),
        "interpolated_positions": int(np.sum(position_source == "interpolated")),
        "upstream_status_counts": upstream,
        "anchor_records": anchor_records,
        "leg_records": leg_records,
        "times": times,
        "positions": positions,
        "leg_index": leg_index,
        "position_source": position_source,
        "reconstructed_hours": reconstructed_h,
        "unresolved_hours": unresolved_h,
    }
