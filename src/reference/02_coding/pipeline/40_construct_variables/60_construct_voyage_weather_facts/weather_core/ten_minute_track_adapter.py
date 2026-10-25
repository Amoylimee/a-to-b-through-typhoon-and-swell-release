"""Build centred-hour weather samples from the accepted ten-minute grid.

The ten-minute product is the only position source.  Full centred hours land
on exact ten-minute timestamps.  Partial first/last hours whose midpoint is
not present in the strict interior grid remain unresolved; this module never
extrapolates from a nearby grid point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .track_adapter import voyage_segments


SUMMARY_COLUMNS = {
    "candidate_id",
    "mmsi",
    "vessel_uid_clean",
    "departure_time",
    "arrival_time",
}
TRACK_COLUMNS = {
    "candidate_id",
    "mmsi",
    "vessel_uid_clean",
    "utc",
    "lat",
    "lon",
    "track_evidence",
    "position_source",
    "gap_id",
    "gap_class",
    "anchor_source",
}


def build_sampling_positions(voyages: pd.DataFrame, track: pd.DataFrame) -> pd.DataFrame:
    missing_summary = sorted(SUMMARY_COLUMNS - set(voyages.columns))
    missing_track = sorted(TRACK_COLUMNS - set(track.columns))
    if missing_summary:
        raise ValueError(f"voyage summary missing columns: {missing_summary}")
    if missing_track:
        raise ValueError(f"ten-minute track missing columns: {missing_track}")

    summary = voyages.copy()
    points = track.copy()
    for frame, columns in (
        (summary, ["candidate_id", "mmsi", "vessel_uid_clean"]),
        (points, ["candidate_id", "mmsi", "vessel_uid_clean"]),
    ):
        for column in columns:
            if frame[column].isna().any():
                raise ValueError(f"{column} contains null identity values")
            frame[column] = frame[column].astype("string")
    if summary.candidate_id.duplicated().any():
        raise ValueError("voyage summary contains duplicate candidate IDs")
    if points.duplicated(["candidate_id", "utc"]).any():
        raise ValueError("ten-minute track contains duplicate candidate/time keys")
    if not set(points.candidate_id).issubset(set(summary.candidate_id)):
        raise ValueError("ten-minute track contains a foreign candidate")

    identity = summary.set_index("candidate_id")[["mmsi", "vessel_uid_clean"]]
    expected = identity.reindex(points.candidate_id)
    if not np.array_equal(points.mmsi.to_numpy(object), expected.mmsi.to_numpy(object)):
        raise ValueError("ten-minute track MMSI disagrees with voyage summary")
    if not np.array_equal(
        points.vessel_uid_clean.to_numpy(object), expected.vessel_uid_clean.to_numpy(object)
    ):
        raise ValueError("ten-minute track API identity disagrees with voyage summary")

    pieces: list[pd.DataFrame] = []
    for voyage in summary.itertuples(index=False):
        segment = voyage_segments(voyage.departure_time, voyage.arrival_time)
        segment["candidate_id"] = str(voyage.candidate_id)
        segment["mmsi"] = str(voyage.mmsi)
        segment["vessel_uid_clean"] = str(voyage.vessel_uid_clean)
        pieces.append(segment)
    samples = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if samples.empty:
        raise ValueError("weather sampling scope is empty")

    selected = points[
        [
            "candidate_id",
            "utc",
            "lat",
            "lon",
            "track_evidence",
            "position_source",
            "gap_id",
            "gap_class",
            "anchor_source",
        ]
    ].rename(columns={"utc": "position_time_utc", "track_evidence": "ten_minute_track_evidence"})
    samples = samples.merge(
        selected,
        on=["candidate_id", "position_time_utc"],
        how="left",
        validate="one_to_one",
    )
    matched = samples.ten_minute_track_evidence.notna()
    samples["weather_position_method"] = np.where(
        matched, "exact_ten_minute_grid", "unresolved_partial_hour_midpoint"
    )
    samples["track_evidence"] = samples.ten_minute_track_evidence.fillna("unresolved")
    for column in ["lat", "lon"]:
        samples.loc[~matched, column] = np.nan
    for column in ["position_source", "gap_id", "gap_class", "anchor_source"]:
        samples.loc[~matched, column] = pd.NA

    resolved = samples.track_evidence.ne("unresolved")
    if not np.array_equal(
        resolved.to_numpy(dtype=bool),
        (samples.lat.notna() & samples.lon.notna()).to_numpy(dtype=bool),
    ):
        raise ValueError("weather sample evidence and coordinate missingness disagree")
    if samples.duplicated(["candidate_id", "hour_anchor_utc"]).any():
        raise ValueError("weather samples contain duplicate candidate/hour keys")
    duration = (
        pd.to_datetime(summary.arrival_time, utc=True)
        - pd.to_datetime(summary.departure_time, utc=True)
    ).dt.total_seconds().div(3600.0)
    actual = samples.groupby("candidate_id").segment_weight_hours.sum().reindex(summary.candidate_id)
    if not np.allclose(actual.to_numpy(), duration.to_numpy(), rtol=0, atol=1e-8):
        raise ValueError("weather sampling denominator does not close to voyage duration")
    return samples
