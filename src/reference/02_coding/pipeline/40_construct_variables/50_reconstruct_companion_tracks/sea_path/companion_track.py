"""Build an experiment-only companion track over 4-24 hour source-AIS gaps.

The layer reconstructs positions on an hourly UTC cadence inside gaps strictly
over four and at most 24 hours, by interpolating along the frozen sea path
between the fixes that bound the gap.  Gaps at or below four hours keep the
accepted short-gap treatment and are not touched here; gaps over 24 hours are
never reconstructed and keep their whole duration as ``unresolved_long_gap``.

Anchors are a parameter, not an assumption.  GFW phase-one acquisition is still
running, so the anchor source is presently empty and every gap is one leg.  When
admissible anchors exist they enter as ordered intermediate waypoints and each
gap becomes several legs; nothing else about the construction changes.  A track
built without anchors is the unanchored lower bound of the contract policy, and
:data:`ANCHOR_SOURCE_EMPTY` is recorded on every row so it can never be mistaken
for the anchored reconstruction.

Every emitted position is marked reconstructed.  An observed and a reconstructed
position must never be indistinguishable downstream.
"""

from __future__ import annotations

import math
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from .routing import (
    DEFAULT_MAX_SPEED_KNOTS,
    DEFAULT_ROUTE_TIMEOUT_S,
    ROUTE_RESOLVED,
    SeaPathRouter,
    interpolate_along_path,
)

SHORT_GAP_HOURS = 4.0
LONG_GAP_HOURS = 24.0
SECONDS_PER_HOUR = 3600.0

#: Recorded on every row while GFW anchors are unavailable.
ANCHOR_SOURCE_EMPTY = "none_gfw_phase_one_incomplete"

GAP_SHORT = "short_gap_accepted_treatment"
GAP_LONG_UNRESOLVED = "unresolved_long_gap"
GAP_RECONSTRUCTED = "reconstructed"


class Anchor(NamedTuple):
    """An admissible intermediate position inside a gap."""

    epoch_s: float
    lat: float
    lon: float
    source: str


#: How a single emitted position was obtained.  An anchor is approximate
#: provider evidence, not an interpolation and not an observed AIS fix, so it is
#: kept distinguishable from both.
POSITION_INTERPOLATED = "interpolated"
POSITION_ANCHOR = "anchor"


class GapReconstruction(NamedTuple):
    """One gap's outcome: its class, per-leg statuses, and emitted positions."""

    gap_class: str
    leg_statuses: tuple[str, ...]
    times: np.ndarray          # epoch seconds, hourly UTC
    positions: np.ndarray      # (N, 2) lat/lon
    leg_index: np.ndarray      # which leg each position came from
    position_source: np.ndarray  # POSITION_INTERPOLATED or POSITION_ANCHOR
    reconstructed_hours: float
    unresolved_hours: float


def classify_gap(gap_hours: float) -> str:
    """Duration decides treatment, and it decides it before anything else runs."""
    if not math.isfinite(gap_hours) or gap_hours <= 0.0:
        raise ValueError(f"gap_hours must be positive and finite: {gap_hours}")
    if gap_hours <= SHORT_GAP_HOURS:
        return GAP_SHORT
    if gap_hours > LONG_GAP_HOURS:
        return GAP_LONG_UNRESOLVED
    return GAP_RECONSTRUCTED


def hourly_utc_grid(t_start: float, t_end: float) -> np.ndarray:
    """Whole UTC hours strictly inside a gap.

    The bounding fixes are observed, so the grid excludes them: emitting a
    reconstructed position on top of an observed one would overwrite evidence
    with an estimate.
    """
    first = math.floor(t_start / SECONDS_PER_HOUR) + 1
    last = math.ceil(t_end / SECONDS_PER_HOUR) - 1
    if last < first:
        return np.empty(0, dtype=float)
    return np.arange(first, last + 1, dtype=float) * SECONDS_PER_HOUR


def reconstruct_gap(
    router: SeaPathRouter,
    t_start: float,
    lat_start: float,
    lon_start: float,
    t_end: float,
    lat_end: float,
    lon_end: float,
    *,
    anchors: Sequence[Anchor] = (),
    timeout_s: float | None = DEFAULT_ROUTE_TIMEOUT_S,
    max_speed_knots: float = DEFAULT_MAX_SPEED_KNOTS,
) -> GapReconstruction:
    """Reconstruct one gap, leg by leg, or record why it could not be.

    A leg that trips either guard contributes no positions and leaves its hours
    unresolved, while the gap's other legs still stand.  Partial reconstruction
    is reported honestly rather than being all-or-nothing in either direction.
    """
    duration_h = (t_end - t_start) / SECONDS_PER_HOUR
    gap_class = classify_gap(duration_h)
    empty = np.empty(0, dtype=float)
    if gap_class != GAP_RECONSTRUCTED:
        return GapReconstruction(
            gap_class=gap_class,
            leg_statuses=(),
            times=empty,
            positions=np.empty((0, 2), dtype=float),
            leg_index=np.empty(0, dtype=int),
            position_source=np.empty(0, dtype=object),
            reconstructed_hours=0.0,
            unresolved_hours=duration_h if gap_class == GAP_LONG_UNRESOLVED else 0.0,
        )

    ordered = sorted(anchors, key=lambda a: a.epoch_s)
    for anchor in ordered:
        if not t_start < anchor.epoch_s < t_end:
            raise ValueError(
                f"anchor at {anchor.epoch_s} lies outside the gap "
                f"[{t_start}, {t_end}]"
            )
    nodes = (
        [(t_start, lat_start, lon_start)]
        + [(a.epoch_s, a.lat, a.lon) for a in ordered]
        + [(t_end, lat_end, lon_end)]
    )

    grid = hourly_utc_grid(t_start, t_end)
    statuses: list[str] = []
    time_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    reconstructed_h = 0.0
    unresolved_h = 0.0

    # GFW Presence matches on the same UTC hour, so an anchor's timestamp
    # normally falls exactly on this grid.  Legs are half-open on both sides, so
    # without this the shared hour would be dropped from both neighbours --
    # discarding precisely the hours carrying the strongest evidence.
    anchor_at = {a.epoch_s: a for a in ordered}
    grid_set = set(grid.tolist())

    for leg, ((ta, la, lo), (tb, lb, lob)) in enumerate(zip(nodes[:-1], nodes[1:])):
        leg_hours = (tb - ta) / SECONDS_PER_HOUR
        resolution = router.resolve_path(
            la, lo, lb, lob,
            timeout_s=timeout_s,
            duration_hours=leg_hours,
            max_speed_knots=max_speed_knots,
        )
        statuses.append(resolution.status)
        if resolution.status != ROUTE_RESOLVED or resolution.path is None:
            unresolved_h += leg_hours
            continue

        reconstructed_h += leg_hours

        # The anchor that opens this leg, but only when it lands on the hourly
        # grid.  An off-grid anchor still shapes the route as a node; emitting a
        # row at its own timestamp would silently break the hourly cadence that
        # the contract specifies and that downstream weather matching joins on.
        if leg > 0 and ta in anchor_at and ta in grid_set:
            time_parts.append(np.array([ta], dtype=float))
            position_parts.append(np.array([[la, lo]], dtype=float))
            index_parts.append(np.array([leg], dtype=int))
            source_parts.append(np.array([POSITION_ANCHOR], dtype=object))

        targets = grid[(grid > ta) & (grid < tb)]
        if not len(targets):
            continue
        positions = interpolate_along_path(resolution.path, ta, tb, targets)
        time_parts.append(targets)
        position_parts.append(positions)
        index_parts.append(np.full(len(targets), leg, dtype=int))
        source_parts.append(np.full(len(targets), POSITION_INTERPOLATED, dtype=object))

    times = np.concatenate(time_parts) if time_parts else empty
    positions = (
        np.concatenate(position_parts) if position_parts
        else np.empty((0, 2), dtype=float)
    )
    legs = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=int)
    sources = (
        np.concatenate(source_parts) if source_parts else np.empty(0, dtype=object)
    )
    order = np.argsort(times, kind="mergesort")
    return GapReconstruction(
        gap_class=gap_class,
        leg_statuses=tuple(statuses),
        times=times[order],
        positions=positions[order],
        leg_index=legs[order],
        position_source=sources[order],
        reconstructed_hours=reconstructed_h,
        unresolved_hours=unresolved_h,
    )
