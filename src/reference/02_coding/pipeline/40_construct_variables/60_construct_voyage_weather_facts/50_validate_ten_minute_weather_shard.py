#!/usr/bin/env python3
"""Independently validate one E001 2025 weather shard."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate(run: Path) -> dict[str, object]:
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError("weather shard is not completed")
    if status.get("network_requests") != 0 or status.get("products_modified") != 0:
        raise ValueError("weather shard changed external or accepted state")
    for record in status["outputs"].values():
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"weather output hash mismatch: {path}")

    source_track_path = Path(status["sources"]["track"]["path"])
    source_track = pd.read_parquet(
        source_track_path,
        columns=["candidate_id", "utc", "lat", "lon", "track_evidence"],
    ).rename(
        columns={
            "utc": "position_time_utc",
            "lat": "source_lat",
            "lon": "source_lon",
            "track_evidence": "source_track_evidence",
        }
    )
    source_track["candidate_id"] = source_track.candidate_id.astype("string")

    month_checks: list[dict[str, object]] = []
    keys: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for month_path in sorted(run.glob("weather_*.parquet")):
        frame = pd.read_parquet(month_path)
        frame["candidate_id"] = frame.candidate_id.astype("string")
        if frame.duplicated(["candidate_id", "hour_anchor_utc"]).any():
            raise ValueError("duplicate weather candidate/hour key")
        weight_ns = (frame.segment_end_utc - frame.segment_start_utc).astype("timedelta64[ns]").astype("int64")
        if not np.array_equal(weight_ns, frame.segment_weight_ns.to_numpy()):
            raise ValueError("weather segment nanosecond weight mismatch")
        if not np.allclose(weight_ns / 3.6e12, frame.segment_weight_hours, rtol=0, atol=0):
            raise ValueError("weather segment hour weight mismatch")
        if not frame.segment_weight_hours.between(0, 1, inclusive="right").all():
            raise ValueError("weather segment weight outside (0, 1]")
        if not frame.position_time_utc.eq(
            frame.segment_start_utc + (frame.segment_end_utc - frame.segment_start_utc) / 2
        ).all():
            raise ValueError("weather sample is not at its segment midpoint")
        missing = frame.lat.isna()
        if not missing.equals(frame.lon.isna()):
            raise ValueError("weather position latitude/longitude missingness differs")
        if not frame.loc[missing, "track_evidence"].eq("unresolved").all():
            raise ValueError("missing weather position is not labelled unresolved")
        if not frame.loc[missing, "swh_m"].isna().all():
            raise ValueError("missing weather position received wave exposure")
        for column in ["tc_within_300km", "tc_within_500km", "tc_storm_present"]:
            if not frame.loc[missing, column].isna().all():
                raise ValueError("missing weather position received cyclone exposure")

        exact = frame.weather_position_method.eq("exact_ten_minute_grid")
        compared = frame.loc[exact].merge(
            source_track,
            on=["candidate_id", "position_time_utc"],
            how="left",
            validate="one_to_one",
        )
        if len(compared) != int(exact.sum()) or compared.source_track_evidence.isna().any():
            raise ValueError("weather exact-grid position lacks a ten-minute source row")
        if not np.allclose(compared.lat, compared.source_lat, rtol=0, atol=0, equal_nan=True):
            raise ValueError("weather latitude differs from ten-minute source")
        if not np.allclose(compared.lon, compared.source_lon, rtol=0, atol=0, equal_nan=True):
            raise ValueError("weather longitude differs from ten-minute source")
        if not compared.track_evidence.astype("string").equals(compared.source_track_evidence.astype("string")):
            raise ValueError("weather evidence differs from ten-minute source")

        good = frame.loc[frame.swh_m.notna()]
        sample = good.sample(min(24, len(good)), random_state=20260921)
        if len(sample):
            source_path = Path(sample.wave_source_path.iloc[0])
            with xr.open_dataset(source_path) as dataset:
                for row in sample.itertuples(index=False):
                    value = float(
                        dataset.swh.sel(
                            valid_time=row.hour_anchor_utc.tz_localize(None),
                            latitude=row.matched_lat,
                            longitude=row.matched_lon,
                        ).values
                    )
                    if abs(value - row.swh_m) > 1e-10:
                        raise ValueError("independent ERA5 lookup differs")
            if (sample.wave_snap_distance_km > 117.94 + 1e-8).any():
                raise ValueError("ERA5 wet-cell snap exceeds 117.94 km")

        weights = frame.segment_weight_hours
        summary = pd.DataFrame(
            {
                "candidate_id": frame.candidate_id,
                "voyage_hours": weights,
                "track_hours": weights.where(frame.lat.notna(), 0),
                "wave_matched_hours": weights.where(frame.swh_m.notna(), 0),
                "wave_integral_m_hours": (weights * frame.swh_m).fillna(0),
            }
        )
        summaries.append(summary.groupby("candidate_id", as_index=False).sum())
        keys.append(frame[["candidate_id", "hour_anchor_utc"]])
        month_checks.append(
            {
                "month": month_path.stem.removeprefix("weather_"),
                "rows": int(len(frame)),
                "exact_ten_minute_positions": int(exact.sum()),
                "era5_cells_independently_verified": int(len(sample)),
            }
        )
    if not month_checks:
        raise ValueError("weather shard has no monthly outputs")
    if pd.concat(keys, ignore_index=True).duplicated().any():
        raise ValueError("duplicate candidate/hour key across weather months")

    recomputed = pd.concat(summaries).groupby("candidate_id", as_index=False).sum().set_index("candidate_id")
    delivered = pd.read_parquet(run / "voyage_weather_facts.parquet").set_index("candidate_id")
    delivered.index = delivered.index.astype("string")
    recomputed.index = recomputed.index.astype("string")
    if set(recomputed.index) != set(delivered.index):
        raise ValueError("weather summary candidate set differs")
    delivered = delivered.reindex(recomputed.index)
    for column in ["voyage_hours", "track_hours", "wave_matched_hours", "wave_integral_m_hours"]:
        if not np.allclose(recomputed[column], delivered[column], rtol=1e-12, atol=1e-8):
            raise ValueError(f"weather summary differs: {column}")
    return {
        "status": "pass",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run.resolve()),
        "shard": status["shard"],
        "mode": status["mode"],
        "candidates": int(len(delivered)),
        "rows": int(sum(item["rows"] for item in month_checks)),
        "voyage_hours": float(recomputed.voyage_hours.sum()),
        "track_hours": float(recomputed.track_hours.sum()),
        "wave_matched_hours": float(recomputed.wave_matched_hours.sum()),
        "months": month_checks,
        "checks": {
            "output_hashes": True,
            "time_denominators": True,
            "exact_ten_minute_position_provenance": True,
            "missing_position_weather_is_unknown": True,
            "independent_era5_cell_values": True,
            "summary_recalculation": True,
        },
        "products_modified": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.run.resolve())
    write_json(args.run.resolve() / "validation.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
