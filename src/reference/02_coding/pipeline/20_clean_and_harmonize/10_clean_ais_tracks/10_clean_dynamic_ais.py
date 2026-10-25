import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
import natsort
import numpy as np
import pandas as pd
import sinbue as sb
import iogenius as iog

from cleaning_common import (
    SHIP_ID_COL,
    TIMESTAMP_COL,
    LON_COL,
    LAT_COL,
    SPEED_COL,
    SUBPROJECT_ROOT,
    build_run_log_dir,
    drop_exact_duplicates,
    drop_invalid_basic_rows,
    enrich_with_specs,
    ensure_directories,
    get_subproject_path,
    haversine_series_nm,
    list_input_files,
    load_config,
    load_specs_table,
    normalize_dynamic_df,
    read_stats_json,
    require_config_value,
    resolve_duplicate_ship_timestamp_positions,
    shard_output_paths,
    shard_outputs_exist,
    write_json,
    write_text_lines,
)


SCRIPT_DIR = Path(__file__).resolve().parent
STEP_TAG = "10_clean_dynamic_ais"
LOG_DIR = SUBPROJECT_ROOT / ".runs" / "clean_ais_tracks" / STEP_TAG
RUN_LOG_ROOT_DIR = LOG_DIR.parent


def assign_traj_ids(df, gap_threshold_hours):
    if df.empty:
        return df.copy()

    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.sort_values(by=[SHIP_ID_COL, TIMESTAMP_COL, "_row_uid"], kind="stable").copy()
    gap_hours = (
        df.groupby(SHIP_ID_COL, sort=False)[TIMESTAMP_COL].diff().dt.total_seconds() / 3600.0
    )
    split_mask = gap_hours.gt(gap_threshold_hours).fillna(False)
    df["_segment_index"] = split_mask.groupby(df[SHIP_ID_COL], sort=False).cumsum().astype(int)
    df["traj_id"] = (
        df[SHIP_ID_COL].astype(str)
        + "_"
        + df["_segment_index"].map(lambda value: f"{int(value):04d}")
    )
    df = df.drop(columns=["_segment_index"])
    return df


def build_signal_loss_gap_log(df, gap_threshold_hours):
    if df.empty:
        return pd.DataFrame(
            columns=[
                SHIP_ID_COL,
                "traj_id_before",
                "traj_id_after",
                "gap_end_lon",
                "gap_end_lat",
                "gap_start_lon",
                "gap_start_lat",
                "gap_duration_hours",
                "gap_reason",
            ]
        )

    ordered = df.sort_values(by=[SHIP_ID_COL, TIMESTAMP_COL, "_row_uid"], kind="stable").copy()
    ordered["_prev_timestamp"] = ordered.groupby(SHIP_ID_COL, sort=False)[TIMESTAMP_COL].shift(1)
    ordered["_prev_lon"] = ordered.groupby(SHIP_ID_COL, sort=False)[LON_COL].shift(1)
    ordered["_prev_lat"] = ordered.groupby(SHIP_ID_COL, sort=False)[LAT_COL].shift(1)
    ordered["_prev_traj_id"] = ordered.groupby(SHIP_ID_COL, sort=False)["traj_id"].shift(1)
    ordered["_gap_hours"] = (
        (ordered[TIMESTAMP_COL] - ordered["_prev_timestamp"]).dt.total_seconds() / 3600.0
    )
    gap_df = ordered.loc[ordered["_gap_hours"].gt(gap_threshold_hours).fillna(False)].copy()
    if gap_df.empty:
        return pd.DataFrame(
            columns=[
                SHIP_ID_COL,
                "traj_id_before",
                "traj_id_after",
                "gap_end_lon",
                "gap_end_lat",
                "gap_start_lon",
                "gap_start_lat",
                "gap_duration_hours",
                "gap_reason",
            ]
        )
    return pd.DataFrame(
        {
            SHIP_ID_COL: gap_df[SHIP_ID_COL].values,
            "traj_id_before": gap_df["_prev_traj_id"].values,
            "traj_id_after": gap_df["traj_id"].values,
            "gap_end_lon": gap_df["_prev_lon"].values,
            "gap_end_lat": gap_df["_prev_lat"].values,
            "gap_start_lon": gap_df[LON_COL].values,
            "gap_start_lat": gap_df[LAT_COL].values,
            "gap_duration_hours": gap_df["_gap_hours"].values,
            "gap_reason": "signal_loss",
        }
    )


def add_calc_speed_knots(df):
    """Add calc_speed_knots using haversine distance between consecutive trajectory points."""
    if df.empty:
        df = df.copy()
        df["calc_speed_knots"] = pd.Series(dtype="float64")
        return df

    df = df.copy().sort_values(["traj_id", TIMESTAMP_COL, "_row_uid"], kind="stable")
    prev_lon = df.groupby("traj_id", sort=False)[LON_COL].shift(1)
    prev_lat = df.groupby("traj_id", sort=False)[LAT_COL].shift(1)
    delta_time_h = (
        df.groupby("traj_id", sort=False)[TIMESTAMP_COL].diff().dt.total_seconds() / 3600.0
    )
    dist_nm = haversine_series_nm(df[LON_COL], df[LAT_COL], prev_lon, prev_lat)
    calc_speed = pd.Series(dist_nm, index=df.index) / delta_time_h
    calc_speed[delta_time_h.fillna(0.0) <= 0.0] = np.nan
    df["calc_speed_knots"] = calc_speed
    return df


def build_drift_gap_rows(original_df, kept_df):
    if original_df.empty or kept_df.empty:
        return []

    rows = []
    kept_lookup = set(kept_df["_row_uid"].tolist())
    for traj_id, traj_df in original_df.groupby("traj_id", sort=False):
        traj_df = traj_df.sort_values(by=[TIMESTAMP_COL, "_row_uid"], kind="stable").copy()
        removed_mask = ~traj_df["_row_uid"].isin(kept_lookup)
        if not removed_mask.any():
            continue

        removed_positions = [idx for idx, value in enumerate(removed_mask.tolist()) if value]
        block_start = removed_positions[0]
        previous_position = removed_positions[0]
        blocks = []
        for position in removed_positions[1:]:
            if position == previous_position + 1:
                previous_position = position
                continue
            blocks.append((block_start, previous_position))
            block_start = position
            previous_position = position
        blocks.append((block_start, previous_position))

        for start_idx, end_idx in blocks:
            if start_idx == 0 or end_idx == len(traj_df) - 1:
                continue
            before_row = traj_df.iloc[start_idx - 1]
            after_row = traj_df.iloc[end_idx + 1]
            rows.append(
                {
                    SHIP_ID_COL: before_row[SHIP_ID_COL],
                    "traj_id_before": traj_id,
                    "traj_id_after": traj_id,
                    "gap_end_lon": before_row[LON_COL],
                    "gap_end_lat": before_row[LAT_COL],
                    "gap_start_lon": after_row[LON_COL],
                    "gap_start_lat": after_row[LAT_COL],
                    "gap_duration_hours": (
                        (after_row[TIMESTAMP_COL] - before_row[TIMESTAMP_COL]).total_seconds()
                        / 3600.0
                    ),
                    "gap_reason": "drift_removal",
                }
            )
    return rows


def split_ship_chunks(df, target_rows):
    if df.empty:
        return []

    ship_sizes_df = (
        df.groupby(SHIP_ID_COL, sort=False).size().reset_index(name="n_rows")
    )
    chunk_ship_ids = []
    current_ship_ids = []
    current_rows = 0

    for ship_id, n_rows in ship_sizes_df[[SHIP_ID_COL, "n_rows"]].itertuples(index=False):
        n_rows = int(n_rows)
        if current_ship_ids and current_rows + n_rows > target_rows:
            chunk_ship_ids.append(current_ship_ids)
            current_ship_ids = []
            current_rows = 0
        current_ship_ids.append(ship_id)
        current_rows += n_rows

    if current_ship_ids:
        chunk_ship_ids.append(current_ship_ids)

    return chunk_ship_ids


def get_chunk_temp_dir(output_dir, input_file):
    return Path(output_dir) / f"{Path(input_file).stem}_temp_chunks"


def get_chunk_temp_paths(output_dir, input_file, chunk_idx):
    temp_dir = get_chunk_temp_dir(output_dir, input_file)
    chunk_tag = f"chunk_{int(chunk_idx):03d}"
    return {
        "dir": temp_dir,
        "cleaned": temp_dir / f"{chunk_tag}_cleaned.feather",
        "gap_log": temp_dir / f"{chunk_tag}_gap_log.feather",
        "removed": temp_dir / f"{chunk_tag}_removed_points.feather",
        "stats": temp_dir / f"{chunk_tag}_stats.json",
    }


def chunk_outputs_exist(output_dir, input_file, chunk_idx):
    chunk_paths = get_chunk_temp_paths(output_dir, input_file, chunk_idx)
    required = [chunk_paths["cleaned"], chunk_paths["gap_log"], chunk_paths["removed"], chunk_paths["stats"]]
    return all(path.exists() for path in required)


def write_chunk_outputs(output_dir, input_file, chunk_idx, chunk_result):
    chunk_paths = get_chunk_temp_paths(output_dir, input_file, chunk_idx)
    ensure_directories(chunk_paths["dir"])
    chunk_result["cleaned"].to_feather(chunk_paths["cleaned"])
    chunk_result["gap_log"].to_feather(chunk_paths["gap_log"])
    chunk_result["removed"].to_feather(chunk_paths["removed"])
    write_json(chunk_paths["stats"], chunk_result["stats"])


def read_chunk_outputs(output_dir, input_file, chunk_idx):
    chunk_paths = get_chunk_temp_paths(output_dir, input_file, chunk_idx)
    return {
        "cleaned": pd.read_feather(chunk_paths["cleaned"]),
        "gap_log": pd.read_feather(chunk_paths["gap_log"]),
        "removed": pd.read_feather(chunk_paths["removed"]),
        "stats": read_stats_json(chunk_paths["stats"]),
    }


def run_outlier_cleaner(df):
    cleaned_frames = []
    removed_frames = []
    for traj_id, traj_df in df.groupby("traj_id", sort=False):
        traj_df = traj_df.sort_values(by=[TIMESTAMP_COL, "_row_uid"], kind="stable").copy()
        if len(traj_df) < 2:
            cleaned_frames.append(traj_df.copy())
            continue
        traj_gdf = gpd.GeoDataFrame(
            traj_df.copy(),
            geometry=gpd.points_from_xy(traj_df[LON_COL], traj_df[LAT_COL]),
            crs="EPSG:4326",
        ).set_index(TIMESTAMP_COL)
        traj = mpd.Trajectory(traj_gdf, traj_id=traj_id)
        vessel_max_ms = float(traj_df["max_speed_knots"].iloc[0]) * 0.5144
        cleaned_traj = mpd.OutlierCleaner(traj).clean(v_max=vessel_max_ms)
        cleaned_df = cleaned_traj.df.reset_index()
        cleaned_row_ids = set(cleaned_df["_row_uid"].tolist())
        removed_df = traj_df.loc[~traj_df["_row_uid"].isin(cleaned_row_ids)].copy()
        if not removed_df.empty:
            removed_df["removed_reason"] = "speed_outlier"
            removed_frames.append(removed_df)
        cleaned_frames.append(cleaned_df.drop(columns=["geometry"], errors="ignore"))

    cleaned_output_df = (
        pd.concat(cleaned_frames, ignore_index=True) if cleaned_frames else df.iloc[0:0].copy()
    )
    removed_output_df = (
        pd.concat(removed_frames, ignore_index=True) if removed_frames else df.iloc[0:0].copy()
    )
    if not cleaned_output_df.empty:
        cleaned_output_df[TIMESTAMP_COL] = pd.to_datetime(
            cleaned_output_df[TIMESTAMP_COL], errors="coerce"
        )
    if not removed_output_df.empty:
        removed_output_df[TIMESTAMP_COL] = pd.to_datetime(
            removed_output_df[TIMESTAMP_COL], errors="coerce"
        )
    return cleaned_output_df, removed_output_df


def process_cleaning_chunk(chunk_df, gap_threshold_hours, drift_multiplier):
    chunk_df = assign_traj_ids(chunk_df, gap_threshold_hours)
    signal_gap_df = build_signal_loss_gap_log(chunk_df, gap_threshold_hours)

    drift_input_df = add_calc_speed_knots(chunk_df)
    drift_threshold = drift_input_df["max_speed_knots"] * drift_multiplier
    drift_mask = pd.to_numeric(drift_input_df["calc_speed_knots"], errors="coerce").gt(drift_threshold)
    drift_removed_df = drift_input_df.loc[drift_mask.fillna(False)].copy()
    drift_removed_df["removed_reason"] = "drift"
    kept_after_drift_df = drift_input_df.loc[~drift_mask.fillna(False)].copy()

    drift_gap_df = pd.DataFrame(build_drift_gap_rows(drift_input_df, kept_after_drift_df))
    if kept_after_drift_df.empty:
        final_clean_df = kept_after_drift_df.copy()
        outlier_removed_df = kept_after_drift_df.iloc[0:0].copy()
    else:
        kept_after_drift_df = assign_traj_ids(kept_after_drift_df, gap_threshold_hours)
        outlier_input_df = add_calc_speed_knots(kept_after_drift_df)
        final_clean_df, outlier_removed_df = run_outlier_cleaner(outlier_input_df)
        final_clean_df = assign_traj_ids(final_clean_df, gap_threshold_hours)
        final_clean_df = add_calc_speed_knots(final_clean_df)

    removed_points_df = (
        pd.concat([drift_removed_df, outlier_removed_df], ignore_index=True)
        if not drift_removed_df.empty or not outlier_removed_df.empty
        else drift_input_df.iloc[0:0].copy()
    )
    gap_log_df = pd.concat([signal_gap_df, drift_gap_df], ignore_index=True, sort=False)

    stats_payload = {
        "input_rows": int(len(chunk_df)),
        "removed_drift": int(len(drift_removed_df)),
        "removed_speed_outlier": int(len(outlier_removed_df)),
        "retained_rows": int(len(final_clean_df)),
        "signal_loss_gap_rows": int(len(signal_gap_df)),
        "drift_gap_rows": int(len(drift_gap_df)),
        "gap_rows": int(len(gap_log_df)),
        "final_traj_segments": (
            int(final_clean_df["traj_id"].nunique()) if not final_clean_df.empty else 0
        ),
    }

    return {
        "cleaned": final_clean_df,
        "gap_log": gap_log_df,
        "removed": removed_points_df,
        "stats": stats_payload,
    }


def summarize_stats(stats_rows, output_path):
    if not stats_rows:
        Path(output_path).write_text("No shard stats were available.\n", encoding="utf-8")
        return

    stats_df = pd.DataFrame(stats_rows)
    report_lines = [
        "=== AIS Cleaning Summary ===",
        f"Input rows:                  {int(stats_df['input_rows'].sum()):,}",
        f"After basic sanity filter:   {int(stats_df['after_basic_filters'].sum()):,}",
        f"After exact duplicate drop:  {int(stats_df['after_exact_duplicate_drop'].sum()):,}",
        f"After timestamp resolve:     {int(stats_df['after_timestamp_resolution'].sum()):,}",
        f"Total removed as drift:      {int(stats_df['removed_drift'].sum()):,}",
        f"Total removed as outlier:    {int(stats_df['removed_speed_outlier'].sum()):,}",
        f"Total retained:              {int(stats_df['retained_rows'].sum()):,}",
        "",
        f"Trajectory segments (final): {int(stats_df['final_traj_segments'].sum()):,}",
        f"Total gaps logged:           {int(stats_df['gap_rows'].sum()):,}",
        f"  - Signal loss gaps:        {int(stats_df['signal_loss_gap_rows'].sum()):,}",
        f"  - Drift removal gaps:      {int(stats_df['drift_gap_rows'].sum()):,}",
        "",
        f"Gap duration mean (hours):   {stats_df['gap_duration_mean_hours'].mean():.6f}",
        f"Gap duration median(hours):  {stats_df['gap_duration_median_hours'].median():.6f}",
        f"Gap duration max (hours):    {stats_df['gap_duration_max_hours'].max():.6f}",
    ]
    Path(output_path).write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def clean_single_shard(file_in, output_dir, log_dir, specs_df, config):
    file_in = Path(file_in)
    output_dir = Path(output_dir)
    log_dir = Path(log_dir)
    gap_threshold_hours = float(require_config_value(config, "gap.threshold_hours"))
    drift_multiplier = float(config.get("drift", {}).get("speed_multiplier", 1.5))
    chunk_target_rows = int(config.get("chunking", {}).get("target_rows", 200000))

    output_paths = shard_output_paths(output_dir, file_in)
    with sb.PrintRedirector(log_dir / f"{file_in.stem}.log"):
        print(f"Processing {file_in}")
        raw_df = pd.read_feather(file_in)
        raw_df = normalize_dynamic_df(raw_df)
        raw_df["_row_uid"] = raw_df.index.astype("int64")
        input_rows = int(len(raw_df))

        basic_df, basic_counts = drop_invalid_basic_rows(raw_df)
        exact_dedup_df, exact_duplicates_removed = drop_exact_duplicates(basic_df)
        timestamp_resolved_df, timestamp_duplicates_removed = resolve_duplicate_ship_timestamp_positions(
            exact_dedup_df
        )
        timestamp_resolved_df = enrich_with_specs(timestamp_resolved_df, specs_df)
        ship_chunks = split_ship_chunks(timestamp_resolved_df, target_rows=chunk_target_rows)
        print(
            f"Prepared {len(ship_chunks)} ship-based chunk(s) "
            f"from {timestamp_resolved_df[SHIP_ID_COL].nunique()} ships "
            f"with target_rows={chunk_target_rows}"
        )

        cleaned_frames = []
        gap_frames = []
        removed_frames = []
        chunk_stats_rows = []
        temp_chunk_dir = get_chunk_temp_dir(output_dir, file_in)
        ensure_directories(temp_chunk_dir)
        for chunk_idx, ship_ids in enumerate(ship_chunks, start=1):
            if chunk_outputs_exist(output_dir, file_in, chunk_idx):
                print(f"Chunk {chunk_idx}/{len(ship_chunks)} already exists on disk; reusing temp outputs")
                chunk_result = read_chunk_outputs(output_dir, file_in, chunk_idx)
            else:
                chunk_df = timestamp_resolved_df.loc[
                    timestamp_resolved_df[SHIP_ID_COL].isin(ship_ids)
                ].copy()
                print(
                    f"Chunk {chunk_idx}/{len(ship_chunks)}: "
                    f"ships={len(ship_ids)} rows={len(chunk_df)}"
                )
                chunk_result = process_cleaning_chunk(
                    chunk_df,
                    gap_threshold_hours=gap_threshold_hours,
                    drift_multiplier=drift_multiplier,
                )
                write_chunk_outputs(output_dir, file_in, chunk_idx, chunk_result)
                print(
                    f"Chunk {chunk_idx}/{len(ship_chunks)} complete: "
                    f"retained={chunk_result['stats']['retained_rows']} "
                    f"removed_drift={chunk_result['stats']['removed_drift']} "
                    f"removed_outlier={chunk_result['stats']['removed_speed_outlier']}"
                )
            cleaned_frames.append(chunk_result["cleaned"])
            gap_frames.append(chunk_result["gap_log"])
            removed_frames.append(chunk_result["removed"])
            chunk_stats_rows.append(chunk_result["stats"])

        final_clean_df = (
            pd.concat(cleaned_frames, ignore_index=True)
            if cleaned_frames
            else timestamp_resolved_df.iloc[0:0].copy()
        )
        gap_log_df = (
            pd.concat(gap_frames, ignore_index=True, sort=False)
            if gap_frames
            else build_signal_loss_gap_log(timestamp_resolved_df.iloc[0:0], gap_threshold_hours)
        )
        removed_points_df = (
            pd.concat(removed_frames, ignore_index=True)
            if removed_frames
            else timestamp_resolved_df.iloc[0:0].copy()
        )

        final_clean_df = final_clean_df.sort_values(
            by=[SHIP_ID_COL, TIMESTAMP_COL, "_row_uid"], kind="stable"
        )
        removed_points_df = removed_points_df.sort_values(
            by=[SHIP_ID_COL, TIMESTAMP_COL, "_row_uid"], kind="stable"
        )
        final_clean_df = final_clean_df.drop(columns=["_row_uid"], errors="ignore")
        removed_points_df = removed_points_df.drop(columns=["_row_uid"], errors="ignore")

        final_clean_df.to_feather(output_paths["cleaned"])
        gap_log_df.to_feather(output_paths["gap_log"])
        removed_points_df.to_feather(output_paths["removed"])

        gap_duration_series = pd.to_numeric(
            gap_log_df.get("gap_duration_hours"), errors="coerce"
        )
        stats_payload = {
            "input_file": str(file_in),
            "input_rows": input_rows,
            "removed_null_ship_id": basic_counts["null_ship_id"],
            "removed_null_timestamp": basic_counts["null_timestamp"],
            "removed_invalid_lon": basic_counts["invalid_lon"],
            "removed_invalid_lat": basic_counts["invalid_lat"],
            "removed_negative_speed": basic_counts["negative_speed"],
            "after_basic_filters": int(len(basic_df)),
            "removed_exact_duplicates": int(exact_duplicates_removed),
            "after_exact_duplicate_drop": int(len(exact_dedup_df)),
            "removed_duplicate_ship_timestamp": int(timestamp_duplicates_removed),
            "after_timestamp_resolution": int(len(timestamp_resolved_df)),
            "n_ship_chunks": int(len(ship_chunks)),
            "chunk_target_rows": int(chunk_target_rows),
            "removed_drift": int(sum(row["removed_drift"] for row in chunk_stats_rows)),
            "removed_speed_outlier": int(sum(row["removed_speed_outlier"] for row in chunk_stats_rows)),
            "retained_rows": int(len(final_clean_df)),
            "signal_loss_gap_rows": int(sum(row["signal_loss_gap_rows"] for row in chunk_stats_rows)),
            "drift_gap_rows": int(sum(row["drift_gap_rows"] for row in chunk_stats_rows)),
            "gap_rows": int(len(gap_log_df)),
            "final_traj_segments": (
                int(final_clean_df["traj_id"].nunique()) if not final_clean_df.empty else 0
            ),
            "gap_duration_mean_hours": float(gap_duration_series.mean()) if not gap_duration_series.empty else 0.0,
            "gap_duration_median_hours": float(gap_duration_series.median()) if not gap_duration_series.empty else 0.0,
            "gap_duration_max_hours": float(gap_duration_series.max()) if not gap_duration_series.empty else 0.0,
            "gap_duration_hours_total_series": float(gap_duration_series.sum()) if not gap_duration_series.empty else 0.0,
            "gap_threshold_hours": gap_threshold_hours,
            "drift_multiplier": drift_multiplier,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(output_paths["stats"], stats_payload)
        shutil.rmtree(temp_chunk_dir, ignore_errors=True)

        print(f"Saved cleaned shard to {output_paths['cleaned']}")
        print(f"Saved gap log to {output_paths['gap_log']}")
        print(f"Saved removed-points audit to {output_paths['removed']}")
        print(f"Saved shard stats to {output_paths['stats']}")
        print(f"Removed temp chunk directory {temp_chunk_dir}")


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)

    config = load_config()
    gap_threshold_hours = require_config_value(config, "gap.threshold_hours")
    if gap_threshold_hours in [None, "", "~"]:
        raise ValueError(
            "config.yaml gap.threshold_hours must be set to a confirmed numeric value "
            "before running 10_clean_dynamic_ais.py"
        )

    iog.set_working_directory(str(SUBPROJECT_ROOT))
    output_root_dir = get_subproject_path(config["output_dir"])
    output_dir = output_root_dir / STEP_TAG
    ensure_directories(output_dir, LOG_DIR)

    input_dir = get_subproject_path(config["input_dir"])
    specs_path = get_subproject_path(config["specs_table_path"])
    max_workers = int(config.get("parallel", {}).get("max_workers", 4))
    chunk_target_rows = int(config.get("chunking", {}).get("target_rows", 200000))

    input_files = list_input_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No shard Feather files found under {input_dir}")

    specs_df = load_specs_table(specs_path)
    pending_input_files = []
    skipped_input_files = []
    for input_file in input_files:
        if shard_outputs_exist(output_dir, input_file):
            skipped_input_files.append(str(input_file))
        else:
            pending_input_files.append(input_file)

    config_fingerprint = hashlib.md5(
        f"{config['_config_path']}|{gap_threshold_hours}|{specs_path}|{max_workers}".encode("utf-8")
    ).hexdigest()[:8]
    run_dir = build_run_log_dir(RUN_LOG_ROOT_DIR, STEP_TAG, input_files, config_fingerprint)
    write_json(
        run_dir / "run_config.json",
        {
            "run_started_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": config["_config_path"],
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "log_dir": str(LOG_DIR),
            "specs_table_path": str(specs_path),
            "max_workers": max_workers,
            "gap_threshold_hours": float(gap_threshold_hours),
            "drift_multiplier": float(config.get("drift", {}).get("speed_multiplier", 1.5)),
            "chunk_target_rows": chunk_target_rows,
            "n_input_files": len(input_files),
            "n_pending_files": len(pending_input_files),
            "n_skipped_existing_outputs": len(skipped_input_files),
        },
    )
    write_text_lines(run_dir / "pending_tasks.txt", [str(path) for path in pending_input_files])
    write_text_lines(run_dir / "skipped_existing_outputs.txt", skipped_input_files)

    if pending_input_files:
        sb.process_files_parallel(
            clean_single_shard,
            input_files=natsort.natsorted(pending_input_files),
            output_dir=output_dir,
            log_dir=LOG_DIR,
            specs_df=specs_df,
            config=config,
            max_workers=max_workers,
            show_progress=True,
        )

    failed_tasks = []
    stats_rows = []
    for input_file in input_files:
        output_paths = shard_output_paths(output_dir, input_file)
        if not all(path.exists() for path in output_paths.values()):
            failed_tasks.append(str(input_file))
            continue
        stats_payload = read_stats_json(output_paths["stats"])
        if stats_payload is not None:
            stats_rows.append(stats_payload)

    write_text_lines(run_dir / "failed_tasks.txt", failed_tasks)
    summary_df = pd.DataFrame(stats_rows)
    if not summary_df.empty:
        summary_df.to_csv(run_dir / "summary_by_shard.csv", index=False)
    else:
        pd.DataFrame(columns=["input_file"]).to_csv(run_dir / "summary_by_shard.csv", index=False)

    summarize_stats(stats_rows, output_dir / "cleaning_report.txt")

    print(f"Input shard files: {len(input_files)}")
    print(f"Pending shard files this run: {len(pending_input_files)}")
    print(f"Skipped existing outputs this run: {len(skipped_input_files)}")
    print(f"Failed shard files this run: {len(failed_tasks)}")
    print(f"Run log dir: {run_dir}")


if __name__ == "__main__":
    main()
