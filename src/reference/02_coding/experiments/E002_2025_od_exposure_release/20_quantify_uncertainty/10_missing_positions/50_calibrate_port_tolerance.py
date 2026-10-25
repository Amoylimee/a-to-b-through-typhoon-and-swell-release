#!/usr/bin/env python3
"""Calibrate T014 port endpoint tolerances against the frozen 969 GFW hours."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[4]
SOURCE = CODING / ".runs/2026-09-24_1720_E002-crossproject-gfw-holdout/zero_track_port_check_internal.parquet"
SPEED_KMH = 25 * 1.852


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrate(out: Path, internal: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    internal.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(SOURCE)
    if len(frame) != 969 or not frame.motion_feasible.eq(True).all():
        raise ValueError("frozen GFW calibration population changed")
    if frame.duplicated(["candidate_id", "hour_utc"]).any():
        raise ValueError("duplicate candidate-hour in calibration data")
    hour = pd.to_datetime(frame.hour_utc, utc=True)
    departure = pd.to_datetime(frame.departure_time, utc=True)
    arrival = pd.to_datetime(frame.arrival_time, utc=True)
    rows = []
    for space_km in (0, 10, 25, 50):
        for time_h in (0, 1, 2):
            shifted_departure = departure - pd.Timedelta(hours=time_h)
            shifted_arrival = arrival + pd.Timedelta(hours=time_h)
            from_origin = pd.to_timedelta(
                np.maximum(0, frame.origin_distance_km - space_km) / SPEED_KMH, unit="h")
            to_destination = pd.to_timedelta(
                np.maximum(0, frame.dest_distance_km - space_km) / SPEED_KMH, unit="h")
            earliest = pd.concat([hour, shifted_departure,
                                  shifted_departure + from_origin], axis=1).max(axis=1)
            latest = pd.concat([hour + pd.Timedelta(hours=1), shifted_arrival,
                                shifted_arrival - to_destination], axis=1).min(axis=1)
            reachable = earliest.le(latest)
            if space_km == 0 and time_h == 0 and not reachable.equals(frame.centre_reachable):
                raise ValueError("zero-tolerance calculation does not reproduce frozen audit")
            frame[f"reachable_{space_km}km_{time_h}h"] = reachable
            rows.append({"space_km": space_km, "time_h": time_h,
                         "gfw_hours": len(frame), "reachable_hours": int(reachable.sum()),
                         "unreachable_hours": int((~reachable).sum()),
                         "reachable_pct": 100 * float(reachable.mean()),
                         "meets_99pct": bool(reachable.mean() >= .99)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "03_port_tolerance_reachability.csv", index=False)
    conflicts = set(frame.loc[~frame.reachable_50km_2h, "candidate_id"])
    remaining = frame.loc[~frame.candidate_id.isin(conflicts)]
    conditional_rows = []
    for row in rows:
        col = f"reachable_{row['space_km']}km_{row['time_h']}h"
        conditional_rows.append({"space_km": row["space_km"], "time_h": row["time_h"],
                                 "gfw_hours": len(remaining),
                                 "reachable_hours": int(remaining[col].sum()),
                                 "reachable_pct": 100 * float(remaining[col].mean()),
                                 "meets_99pct": bool(remaining[col].mean() >= .99)})
    conditional = pd.DataFrame(conditional_rows)
    conditional["equivalent_radius_km_at_25kn"] = conditional.space_km + SPEED_KMH * conditional.time_h
    conditional.to_csv(out / "03b_port_tolerance_without_conflict_voyages.csv", index=False)
    # Identifiers and coordinates stay under .runs.
    frame.to_parquet(internal / "port_tolerance_hour_audit.parquet", index=False)
    eligible = table.loc[table.meets_99pct].sort_values(["space_km", "time_h"])
    selected = eligible.iloc[0][["space_km", "time_h"]].to_dict() if len(eligible) else None
    conditional_eligible = conditional.loc[conditional.meets_99pct].sort_values(
        ["equivalent_radius_km_at_25kn", "space_km", "time_h"])
    conditional_selected = (conditional_eligible.iloc[0][["space_km", "time_h"]].to_dict()
                            if len(conditional_eligible) else None)
    widest = frame.loc[~frame.reachable_50km_2h]
    report = {"source": str(SOURCE), "source_sha256": sha256(SOURCE),
              "speed_kn": 25, "calibration_hours": len(frame),
              "zero_tolerance_reachable_hours": int(table.iloc[0].reachable_hours),
              "threshold_pct": 99, "selected": selected,
              "maximum_grid_reachable_hours": int(table.iloc[-1].reachable_hours),
              "maximum_grid_reachable_pct": float(table.iloc[-1].reachable_pct),
              "maximum_grid_remaining_hours": len(widest),
              "maximum_grid_remaining_voyages": int(widest.candidate_id.nunique()),
              "whole_conflict_voyage_hours": len(frame) - len(remaining),
              "remaining_hours_if_conflict_voyages_set_unbounded": len(remaining),
              "conditional_0km_1h_reachable_hours": int(remaining.reachable_0km_1h.sum()),
              "conditional_0km_1h_reachable_pct": 100 * float(remaining.reachable_0km_1h.mean()),
              "conditional_selected_minimum_effective_radius": conditional_selected,
              "status": "threshold_unmet" if selected is None else "selected"}
    (out / "03_port_tolerance_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--internal", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.out, args.internal), ensure_ascii=False, indent=2))
