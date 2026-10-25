#!/usr/bin/env python3
"""Plot one sampled 25 kn reachable region against frozen IBTrACS circles."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("10_probe_reachability.py")
SPEC = importlib.util.spec_from_file_location("e002_reachability_probe", SCRIPT)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def _destination(lat: float, lon: float, radius_km: float) -> tuple[np.ndarray, np.ndarray]:
    bearings = np.linspace(0.0, 2.0 * np.pi, 361)
    angular = radius_km / PROBE.legacy.EARTH_RADIUS_KM
    phi1, lam1 = np.radians(lat), np.radians(lon)
    phi2 = np.arcsin(
        np.sin(phi1) * np.cos(angular)
        + np.cos(phi1) * np.sin(angular) * np.cos(bearings)
    )
    lam2 = lam1 + np.arctan2(
        np.sin(bearings) * np.sin(angular) * np.cos(phi1),
        np.cos(angular) - np.sin(phi1) * np.sin(phi2),
    )
    return np.degrees(phi2), ((np.degrees(lam2) + 180.0) % 360.0) - 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-parquet", type=Path, required=True)
    parser.add_argument("--bounds-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sample = pd.read_parquet(args.sample_parquet)
    bounds = pd.read_csv(args.bounds_csv)
    possible = bounds.loc[
        bounds.speed_case.eq("fixed_25_kn_primary")
        & bounds.geometry_status.eq("matched")
        & bounds.tc500_possible.astype("boolean").fillna(False).astype(bool)
        & bounds.sample_index.notna()
    ].copy()
    if possible.empty:
        raise SystemExit("no matched sampled unknown segment has a 25 kn reachable cell within 500 km of a storm")
    possible.sort_values(["reachable_grid_cells", "position_time_utc"], inplace=True)
    result_row = possible.iloc[0]
    sample_row = sample.loc[int(result_row.sample_index)]
    lat, lon, capped, endpoint_case = PROBE._reachable(sample_row, PROBE.SPEED_KN)
    if capped:
        raise SystemExit("selected example was cell-capped; refusing a misleading plot")

    year, month = pd.Timestamp(sample_row.hour_anchor_utc).year, pd.Timestamp(sample_row.hour_anchor_utc).month
    matcher = PROBE.Era5MonthlyMatcher(
        PROBE.resolve_monthly_era5_assets(year=year, month=month),
        maximum_wet_cell_snap_km=0.0,
    )
    try:
        PROBE.validate_month_time_grid(matcher, year, month)
        matcher.preload_swh()
        matched = matcher.match(pd.DataFrame({
            "hour_anchor_utc": [sample_row.hour_anchor_utc] * len(lat),
            "lat": lat,
            "lon": lon,
        }))
    finally:
        matcher.close()
    wet = matched.loc[matched.swh_m.notna()].copy()

    fixes, audit = PROBE.load_ibtracs_snapshot(
        PROBE.CONFIG["ibtracs_path"],
        expected_sha256=PROBE.CONFIG["ibtracs_sha256"],
    )
    cyclones = PROBE.CycloneTrackIndex(fixes, maximum_fix_gap_hours=3.0)
    storms = cyclones.positions_at_times([sample_row.position_time_utc])
    if storms.empty:
        raise SystemExit("selected sample lost its frozen storm-center match")

    center_lon = float(storms.iloc[0].storm_lon)
    wrap_lon = lambda values: center_lon + ((np.asarray(values) - center_lon + 180.0) % 360.0) - 180.0
    fig, ax = plt.subplots(figsize=(9.2, 6.4), constrained_layout=True)
    points = ax.scatter(
        wrap_lon(wet.lon), wet.lat, c=wet.swh_m, cmap="viridis", s=15,
        linewidths=0, alpha=0.8, label="reachable ERA5 cells",
    )
    for number, storm in enumerate(storms.itertuples(index=False), start=1):
        ax.scatter([wrap_lon([storm.storm_lon])[0]], [storm.storm_lat], marker="*", s=145, c="#c62828", zorder=5, label="IBTrACS centre" if number == 1 else None)
        for radius_km, color, style in ((300.0, "#c62828", "--"), (500.0, "#ef6c00", "-")):
            circle_lat, circle_lon = _destination(float(storm.storm_lat), float(storm.storm_lon), radius_km)
            ax.plot(wrap_lon(circle_lon), circle_lat, color=color, linestyle=style, linewidth=1.4, label=f"{int(radius_km)} km" if number == 1 else None)

    for side, label in (("left", "known endpoint A"), ("right", "known endpoint B")):
        timestamp = sample_row.get(f"{side}_time_ns")
        if pd.notna(timestamp):
            ax.scatter(
                [wrap_lon([sample_row[f"{side}_lon"]])[0]],
                [float(sample_row[f"{side}_lat"])],
                marker="D", s=55, edgecolors="black", linewidths=0.7,
                label=label,
            )
    ax.set_xlabel("Longitude (degrees; unwrapped around storm centre)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(
        "25 kn possible-location set and frozen storm circles\n"
        f"{pd.Timestamp(sample_row.position_time_utc).isoformat()} · "
        f"{len(wet):,} wet cells · {endpoint_case}"
    )
    fig.colorbar(points, ax=ax, label="Significant wave height (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    audit_out = {
        "sample_index": int(result_row.sample_index),
        "sample_month": str(sample_row.sample_month),
        "position_time_utc": pd.Timestamp(sample_row.position_time_utc).isoformat(),
        "hour_anchor_utc": pd.Timestamp(sample_row.hour_anchor_utc).isoformat(),
        "endpoint_case": endpoint_case,
        "fixed_primary_speed_cap_kn": PROBE.SPEED_KN,
        "reachable_grid_cells": int(len(lat)),
        "wet_cells_with_swh": int(len(wet)),
        "swh_min_m": float(wet.swh_m.min()),
        "swh_max_m": float(wet.swh_m.max()),
        "tc500_possible": True,
        "storm_centres": len(storms),
        "ibtracs_sha256": PROBE.CONFIG["ibtracs_sha256"],
        "ibtracs_audit": {key: value for key, value in audit.items() if key in {"retained_rows", "retained_sids", "provisional_nr_rows"}},
        "classification": "internal review illustration; not a publication figure or AIS trajectory",
        "figure": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(audit_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit_out, indent=2))


if __name__ == "__main__":
    main()
