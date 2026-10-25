"""Offline classification of exact GFW Presence cells as usable anchors.

An anchor is admitted only when an exact MMSI and UTC-hour match exists, the
reported grid centre is reachable from both source-AIS boundary fixes during
that hour, and exactly one distinct feasible grid centre survives.  This
module never queries GFW and never interpolates a trajectory.

Identity multiplicity is a fifth verdict.  ``group-by=MMSI`` sums presence
across every provider entity sharing one MMSI, so a cell reporting more than
one hour inside a single UTC hour reports an identity that is not unique.  The
grid centre may still be unique while *which vessel produced it* is not, and
the anchor rule rests on identity, not only on geometry.  Such an hour is
therefore recorded and never admitted.  It is not a hard failure: this module
is the first place that knows the target keys, and a non-target row must never
be able to stop an acquisition block.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .presence_overlap import cell_reachable_during_hour


REQUIRED_DARK_COLUMNS = {
    "target_hour_id",
    "gap_id",
    "candidate_id",
    "mmsi",
    "hour_utc",
    "source_gap_start_utc",
    "source_gap_end_utc",
    "lat_a",
    "lon_a",
    "lat_b",
    "lon_b",
    "effective_speed_kn",
}
REQUIRED_PRESENCE_COLUMNS = {"hour_utc", "mmsi", "hours", "lat", "lon"}


def classify_admissible_anchors(
    dark_hours: pd.DataFrame,
    presence: pd.DataFrame,
    *,
    cell_degrees: float,
    tolerance_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return hour verdicts, exact evidence cells, and an input/output audit."""

    missing_dark = sorted(REQUIRED_DARK_COLUMNS.difference(dark_hours.columns))
    missing_presence = sorted(REQUIRED_PRESENCE_COLUMNS.difference(presence.columns))
    if missing_dark:
        raise ValueError(f"dark-hour table is missing columns: {missing_dark}")
    if missing_presence:
        raise ValueError(f"presence table is missing columns: {missing_presence}")
    if dark_hours["target_hour_id"].duplicated().any():
        raise ValueError("dark-hour denominator must be unique by target_hour_id")
    if dark_hours.duplicated(["mmsi", "hour_utc"]).any():
        raise ValueError("dark-hour denominator must be unique by MMSI/hour")

    dark = dark_hours.copy()
    observed = presence.loc[:, sorted(REQUIRED_PRESENCE_COLUMNS)].copy()
    for frame in (dark, observed):
        frame["mmsi"] = frame["mmsi"].astype("string")
        frame["hour_utc"] = pd.to_datetime(frame["hour_utc"], utc=True)
    for column in ("hours", "lat", "lon"):
        observed[column] = pd.to_numeric(observed[column], errors="raise")
    observed["hour_multiplicity"] = observed["hours"].ne(1)
    if not observed["lat"].between(-90, 90).all():
        raise ValueError("Presence latitude is outside [-90, 90]")
    if not observed["lon"].between(-180, 180).all():
        raise ValueError("Presence longitude is outside [-180, 180]")

    centre_key = ["mmsi", "hour_utc", "lat", "lon"]
    exact_duplicate_rows = int(observed.duplicated(centre_key).sum())
    # A stable sort on an all-False column preserves the original order exactly,
    # so this changes nothing unless a centre really does carry multiplicity --
    # in which case the multiplicity row must be the one that survives dedup.
    observed = observed.sort_values(
        "hour_multiplicity", ascending=False, kind="stable"
    )
    observed = observed.drop_duplicates(centre_key, keep="first").reset_index(drop=True)
    cells = observed.merge(
        dark,
        on=["mmsi", "hour_utc"],
        how="inner",
        validate="many_to_one",
    )
    if len(cells):
        cells["motion_feasible"] = [
            cell_reachable_during_hour(
                row._asdict(),
                float(row.lat),
                float(row.lon),
                cell_degrees=cell_degrees,
                tolerance_km=tolerance_km,
            )
            for row in cells.itertuples(index=False)
        ]
        # A multiplicity centre is geometrically feasible but identity-ambiguous,
        # so it is counted separately and never becomes an admissible centre.
        cells["admissible_centre"] = cells["motion_feasible"] & ~cells[
            "hour_multiplicity"
        ]
        counts = cells.groupby("target_hour_id", observed=True).agg(
            exact_grid_centres=("motion_feasible", "size"),
            feasible_grid_centres=("admissible_centre", "sum"),
            multiplicity_grid_centres=("hour_multiplicity", "sum"),
        )
    else:
        cells["motion_feasible"] = pd.Series(dtype=bool)
        cells["admissible_centre"] = pd.Series(dtype=bool)
        counts = pd.DataFrame(
            columns=[
                "exact_grid_centres",
                "feasible_grid_centres",
                "multiplicity_grid_centres",
            ]
        ).rename_axis("target_hour_id")

    hours = dark.merge(
        counts.reset_index(),
        on="target_hour_id",
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "exact_grid_centres",
        "feasible_grid_centres",
        "multiplicity_grid_centres",
    ]
    hours[count_columns] = hours[count_columns].fillna(0).astype("int64")
    hours["exact_identity_time_hit"] = hours["exact_grid_centres"].gt(0)
    hours["motion_feasible_hit"] = hours["feasible_grid_centres"].gt(0)
    # Multiplicity is tested first.  An hour touched by an ambiguous identity is
    # not admitted even when a second, clean centre would have satisfied the
    # geometry, because the rule is identity plus geometry, not geometry alone.
    hours["anchor_status"] = np.select(
        [
            hours["multiplicity_grid_centres"].gt(0),
            hours["feasible_grid_centres"].gt(1),
            hours["feasible_grid_centres"].eq(1),
            hours["exact_grid_centres"].gt(0),
        ],
        [
            "identity_multiplicity_conflict",
            "multiple_feasible_grid_centres",
            "admissible_unique_anchor",
            "exact_identity_time_motion_conflict",
        ],
        default="no_exact_identity_time_presence",
    )
    hours["admissible_unique_anchor"] = hours["anchor_status"].eq(
        "admissible_unique_anchor"
    )

    feasible = cells.loc[cells["admissible_centre"]].copy()
    unique_ids = set(
        hours.loc[hours["admissible_unique_anchor"], "target_hour_id"].astype(str)
    )
    anchors = feasible.loc[
        feasible["target_hour_id"].astype(str).isin(unique_ids),
        ["target_hour_id", "lat", "lon"],
    ].rename(columns={"lat": "anchor_lat", "lon": "anchor_lon"})
    if anchors["target_hour_id"].duplicated().any():
        raise AssertionError("an admitted hour does not have exactly one feasible centre")
    hours = hours.merge(anchors, on="target_hour_id", how="left", validate="one_to_one")
    if hours.loc[hours["admissible_unique_anchor"], ["anchor_lat", "anchor_lon"]].isna().any(
        axis=None
    ):
        raise AssertionError("an admitted anchor is missing its grid centre")
    if hours.loc[~hours["admissible_unique_anchor"], ["anchor_lat", "anchor_lon"]].notna().any(
        axis=None
    ):
        raise AssertionError("a rejected hour received an anchor coordinate")

    audit: dict[str, int] = {
        "target_hours": int(len(hours)),
        "presence_rows_input": int(len(presence)),
        "exact_duplicate_presence_centres_excluded": exact_duplicate_rows,
        "distinct_exact_identity_time_cells": int(len(cells)),
        "motion_feasible_cells": int(cells["motion_feasible"].sum()),
        "hour_multiplicity_cells": int(cells["hour_multiplicity"].sum()),
        "exact_identity_time_hours": int(hours["exact_identity_time_hit"].sum()),
        "motion_feasible_hours": int(hours["motion_feasible_hit"].sum()),
        "admissible_unique_anchor_hours": int(hours["admissible_unique_anchor"].sum()),
        "multiple_feasible_grid_centre_hours": int(
            hours["anchor_status"].eq("multiple_feasible_grid_centres").sum()
        ),
        "motion_conflict_hours": int(
            hours["anchor_status"].eq("exact_identity_time_motion_conflict").sum()
        ),
        "no_exact_identity_time_presence_hours": int(
            hours["anchor_status"].eq("no_exact_identity_time_presence").sum()
        ),
        "identity_multiplicity_hours": int(
            hours["anchor_status"].eq("identity_multiplicity_conflict").sum()
        ),
    }
    if sum(
        audit[key]
        for key in (
            "admissible_unique_anchor_hours",
            "identity_multiplicity_hours",
            "multiple_feasible_grid_centre_hours",
            "motion_conflict_hours",
            "no_exact_identity_time_presence_hours",
        )
    ) != audit["target_hours"]:
        raise AssertionError("anchor-status categories do not close")
    return hours, cells, audit
