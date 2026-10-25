"""Source-AIS gap extraction shared by the coverage bound and the track build.

Cleaned AIS trajectories are cut wherever the source gap exceeds four hours, so
the boundary between consecutive trajectories of one vessel *is* a gap, and the
trajectory intervals are the vessel's observed time.  Both are returned together
because observed time must be measured, never inferred by subtracting known gaps
from total hours -- a voyage running past the end of the year has neither a
trajectory nor a gap record there, and that residual differs by route.
"""

from __future__ import annotations

import pandas as pd

#: Cleaned AIS is already split at this threshold, so anything longer appears as
#: a boundary between trajectories rather than as a within-trajectory interval.
SHORT_GAP_H = 4.0

AIS_GAP_COLUMNS = ["UTC", "vessel_uid", "lat", "lon", "traj_id"]


def gaps_from_shard(task: tuple[str, set]) -> pd.DataFrame:
    """Every >4 h source gap for the target vessels in one shard.

    Cleaned AIS trajectories are cut wherever the source gap exceeds four hours,
    so the boundary between consecutive trajectories of one vessel *is* a gap.
    Sorting by (vessel, time) makes each trajectory contiguous, so the first and
    last row of every trajectory fall out of a single shift comparison; a
    groupby here costs minutes per shard instead of milliseconds.
    """
    shard_path, wanted = task
    import pyarrow.feather as feather

    frame = feather.read_table(
        shard_path, columns=["UTC", "vessel_uid", "lat", "lon", "traj_id"],
    ).to_pandas()
    frame = frame[frame["vessel_uid"].isin(wanted)]
    if len(frame) < 2:
        return pd.DataFrame(), pd.DataFrame()

    frame = frame.sort_values(["vessel_uid", "UTC"], kind="mergesort")
    traj = frame["traj_id"]
    last = frame[traj.ne(traj.shift(-1))]
    first = frame[traj.ne(traj.shift(1))]
    if len(last) < 2:
        return pd.DataFrame(), pd.DataFrame()

    ship_before = last["vessel_uid"].to_numpy()[:-1]
    ship_after = first["vessel_uid"].to_numpy()[1:]
    same_vessel = ship_before == ship_after

    out = pd.DataFrame({
        "vessel_uid": ship_before,
        "gap_start": last["UTC"].to_numpy()[:-1],
        "gap_end": first["UTC"].to_numpy()[1:],
        "lat_start": last["lat"].to_numpy()[:-1],
        "lon_start": last["lon"].to_numpy()[:-1],
        "lat_end": first["lat"].to_numpy()[1:],
        "lon_end": first["lon"].to_numpy()[1:],
    })[same_vessel].reset_index(drop=True)
    if not out.empty:
        out["gap_hours"] = (out["gap_end"] - out["gap_start"]).dt.total_seconds() / 3600.0
        out = out[out["gap_hours"] > SHORT_GAP_H].reset_index(drop=True)

    # Observed time is measured here, never inferred by subtraction.  A voyage
    # running past the end of this year's AIS has no trajectory and no gap
    # record either, so subtracting gaps from total hours would score that time
    # as observed -- and Cape voyages are longer than Suez ones, which makes
    # the error route-correlated in exactly the comparison this bound protects.
    intervals = pd.DataFrame({
        "vessel_uid": first["vessel_uid"].to_numpy(),
        "traj_start": first["UTC"].to_numpy(),
        "traj_end": last["UTC"].to_numpy(),
    })
    return out, intervals
