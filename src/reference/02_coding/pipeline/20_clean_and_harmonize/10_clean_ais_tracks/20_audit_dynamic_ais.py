from datetime import datetime
from pathlib import Path

import natsort
import numpy as np
import pandas as pd
import sinbue as sb
import iogenius as iog

from cleaning_common import (
    COURSE_COL,
    HEADING_COL,
    LAT_COL,
    LON_COL,
    SHIP_ID_COL,
    SPEED_COL,
    SUBPROJECT_ROOT,
    TIMESTAMP_COL,
    build_run_log_dir,
    enrich_with_specs,
    ensure_directories,
    get_subproject_path,
    haversine_series_nm,
    list_input_files,
    list_cleaned_point_shard_files,
    load_config,
    load_specs_table,
    normalize_dynamic_df,
    read_stats_json,
    write_json,
    write_text_lines,
)


STEP_TAG = "20_audit_dynamic_ais"
INPUT_STEP_TAG = "10_clean_dynamic_ais"
LOG_DIR = SUBPROJECT_ROOT / ".runs" / "clean_ais_tracks" / STEP_TAG
RUN_LOG_ROOT_DIR = LOG_DIR.parent


def get_audit_config(config):
    audit_config = config.get("audit", {})
    return {
        "residual_speed_multiplier": float(
            audit_config.get("residual_speed_multiplier", 1.1)
        ),
        "residual_speed_min_excess_knots": float(
            audit_config.get("residual_speed_min_excess_knots", 2.0)
        ),
        "reported_sog_multiplier": float(audit_config.get("reported_sog_multiplier", 1.5)),
        "reported_sog_min_excess_knots": float(
            audit_config.get("reported_sog_min_excess_knots", 5.0)
        ),
        "flag_row_limit_per_shard": int(audit_config.get("flag_row_limit_per_shard", 200000)),
        "top_vessel_limit": int(audit_config.get("top_vessel_limit", 200)),
    }


def audit_output_paths(output_dir, input_file):
    input_file = Path(input_file)
    return {
        "flagged_rows": Path(output_dir) / f"{input_file.stem}_flagged_rows.feather",
        "trajectory_summary": Path(output_dir) / f"{input_file.stem}_trajectory_summary.csv",
        "stats": Path(output_dir) / f"{input_file.stem}_audit_stats.json",
    }


def audit_outputs_exist(output_dir, input_file):
    return all(path.exists() for path in audit_output_paths(output_dir, input_file).values())


def build_task(input_file, output_dir, task_result_dir):
    input_file = Path(input_file)
    paths = audit_output_paths(output_dir, input_file)
    return {
        "task_id": input_file.stem,
        "input_file": str(input_file),
        "flagged_rows_path": str(paths["flagged_rows"]),
        "trajectory_summary_path": str(paths["trajectory_summary"]),
        "stats_path": str(paths["stats"]),
        "task_result_path": str(Path(task_result_dir) / f"{input_file.stem}.json"),
    }


def append_issue(issue_frames, df, mask, issue_type, issue_detail):
    mask = mask.fillna(False)
    if not mask.any():
        return

    selected_columns = [
        column
        for column in [
            "_row_number",
            SHIP_ID_COL,
            "traj_id",
            TIMESTAMP_COL,
            LON_COL,
            LAT_COL,
            SPEED_COL,
            COURSE_COL,
            HEADING_COL,
            "max_speed_knots",
            "calc_speed_knots_reaudit",
            "_prev_timestamp",
            "_prev_lon",
            "_prev_lat",
            "_delta_hours",
            "_distance_nm",
        ]
        if column in df.columns
    ]
    issue_df = df.loc[mask, selected_columns].copy()
    issue_df.insert(0, "issue_type", issue_type)
    issue_df.insert(1, "issue_detail", issue_detail)
    issue_frames.append(issue_df)


def prepare_reaudit_metrics(df):
    df = df.copy()
    if "traj_id" not in df.columns:
        df["traj_id"] = ""

    df["_traj_key"] = df["traj_id"].fillna("").astype(str)
    missing_traj_mask = df["_traj_key"].isin(["", "nan", "None", "<NA>"])
    df.loc[missing_traj_mask, "_traj_key"] = (
        df.loc[missing_traj_mask, SHIP_ID_COL].astype(str) + "_missing_traj"
    )

    df = df.sort_values(by=[SHIP_ID_COL, "_traj_key", TIMESTAMP_COL, "_row_number"], kind="stable")
    grouped = df.groupby([SHIP_ID_COL, "_traj_key"], sort=False, dropna=False)
    df["_prev_timestamp"] = grouped[TIMESTAMP_COL].shift(1)
    df["_prev_lon"] = grouped[LON_COL].shift(1)
    df["_prev_lat"] = grouped[LAT_COL].shift(1)
    df["_delta_hours"] = (
        (df[TIMESTAMP_COL] - df["_prev_timestamp"]).dt.total_seconds() / 3600.0
    )
    df["_distance_nm"] = haversine_series_nm(
        df[LON_COL],
        df[LAT_COL],
        df["_prev_lon"],
        df["_prev_lat"],
    )
    df["calc_speed_knots_reaudit"] = df["_distance_nm"] / df["_delta_hours"]
    df.loc[df["_delta_hours"].fillna(0.0) <= 0.0, "calc_speed_knots_reaudit"] = np.nan
    return df


def build_trajectory_summary(df, residual_speed_mask, nonpositive_time_mask):
    work_df = df.copy()
    work_df["_residual_speed_flag"] = residual_speed_mask.fillna(False).astype(int)
    work_df["_nonpositive_time_flag"] = nonpositive_time_mask.fillna(False).astype(int)
    summary_df = (
        work_df.groupby([SHIP_ID_COL, "_traj_key"], sort=False, dropna=False)
        .agg(
            traj_id=("traj_id", "first"),
            n_rows=(TIMESTAMP_COL, "size"),
            start_time=(TIMESTAMP_COL, "min"),
            end_time=(TIMESTAMP_COL, "max"),
            max_calc_speed_knots=("calc_speed_knots_reaudit", "max"),
            max_reported_speed_knots=(SPEED_COL, "max"),
            n_residual_speed_flags=("_residual_speed_flag", "sum"),
            n_nonpositive_time_flags=("_nonpositive_time_flag", "sum"),
        )
        .reset_index()
        .rename(columns={"_traj_key": "trajectory_key"})
    )
    summary_df["duration_hours"] = (
        summary_df["end_time"] - summary_df["start_time"]
    ).dt.total_seconds() / 3600.0
    anomaly_mask = (
        summary_df["n_residual_speed_flags"].gt(0)
        | summary_df["n_nonpositive_time_flags"].gt(0)
    )
    return summary_df.loc[anomaly_mask].copy()


def audit_single_shard(task, log_dir, specs_df, audit_config):
    input_file = Path(task["input_file"])
    output_paths = {
        "flagged_rows": Path(task["flagged_rows_path"]),
        "trajectory_summary": Path(task["trajectory_summary_path"]),
        "stats": Path(task["stats_path"]),
    }
    task_result_path = Path(task["task_result_path"])
    ensure_directories(output_paths["flagged_rows"].parent, Path(log_dir), task_result_path.parent)

    try:
        with sb.PrintRedirector(Path(log_dir) / f"{input_file.stem}.log"):
            print(f"Auditing {input_file}")
            raw_df = pd.read_feather(input_file)
            input_rows = int(len(raw_df))
            df = normalize_dynamic_df(raw_df)
            df["_row_number"] = np.arange(len(df), dtype="int64")

            if "max_speed_knots" not in df.columns:
                df = enrich_with_specs(df, specs_df)
            df["max_speed_knots"] = pd.to_numeric(df["max_speed_knots"], errors="coerce")
            df["max_speed_knots"] = df["max_speed_knots"].fillna(30.0)

            df = prepare_reaudit_metrics(df)

            residual_speed_limit = np.maximum(
                df["max_speed_knots"] * audit_config["residual_speed_multiplier"],
                df["max_speed_knots"] + audit_config["residual_speed_min_excess_knots"],
            )
            reported_sog_limit = np.maximum(
                df["max_speed_knots"] * audit_config["reported_sog_multiplier"],
                df["max_speed_knots"] + audit_config["reported_sog_min_excess_knots"],
            )

            duplicate_ship_timestamp_mask = df.duplicated(
                subset=[SHIP_ID_COL, TIMESTAMP_COL], keep=False
            )
            nonpositive_time_mask = df["_delta_hours"].le(0.0) & df["_prev_timestamp"].notna()
            residual_speed_mask = df["calc_speed_knots_reaudit"].gt(residual_speed_limit)

            issue_frames = []
            append_issue(
                issue_frames,
                df,
                df[SHIP_ID_COL].eq(""),
                "null_ship_id",
                "Missing vessel_uid after normalization.",
            )
            append_issue(
                issue_frames,
                df,
                df[TIMESTAMP_COL].isna(),
                "null_timestamp",
                "Missing or unparsable UTC timestamp.",
            )
            append_issue(
                issue_frames,
                df,
                ~df[LON_COL].between(-180, 180, inclusive="both"),
                "invalid_lon",
                "Longitude outside [-180, 180].",
            )
            append_issue(
                issue_frames,
                df,
                ~df[LAT_COL].between(-90, 90, inclusive="both"),
                "invalid_lat",
                "Latitude outside [-90, 90].",
            )
            append_issue(
                issue_frames,
                df,
                df[SPEED_COL].lt(0),
                "negative_speed",
                "Reported SOG is negative.",
            )
            append_issue(
                issue_frames,
                df,
                duplicate_ship_timestamp_mask,
                "duplicate_ship_timestamp",
                "Multiple rows share the same vessel and UTC timestamp.",
            )
            append_issue(
                issue_frames,
                df,
                nonpositive_time_mask,
                "nonpositive_trajectory_time_delta",
                "Consecutive points within a trajectory are not strictly time-increasing.",
            )
            append_issue(
                issue_frames,
                df,
                residual_speed_mask,
                "residual_speed_jump",
                "Recomputed inter-point speed remains above vessel-specific audit limit.",
            )
            append_issue(
                issue_frames,
                df,
                df[SPEED_COL].gt(reported_sog_limit)
                & ~np.isclose(df[SPEED_COL], 102.3, atol=0.01),
                "reported_sog_over_limit",
                "Reported SOG is unusually high relative to vessel max-speed threshold and is not the AIS unavailable code 102.3.",
            )
            course_out_of_range_mask = (
                df[COURSE_COL].notna()
                & ~df[COURSE_COL].between(0, 360, inclusive="both")
                & df[COURSE_COL].ne(511)
            )
            append_issue(
                issue_frames,
                df,
                course_out_of_range_mask,
                "course_out_of_range",
                "Course is outside [0, 360] and is not the observed unavailable code 511.",
            )
            if HEADING_COL in df.columns:
                heading_out_of_range_mask = (
                    df[HEADING_COL].notna()
                    & ~df[HEADING_COL].between(0, 360, inclusive="both")
                    & df[HEADING_COL].ne(511)
                )
                append_issue(
                    issue_frames,
                    df,
                    heading_out_of_range_mask,
                    "heading_out_of_range",
                    "Heading is outside [0, 360] and is not the AIS unavailable code 511.",
                )
            append_issue(
                issue_frames,
                df,
                df["traj_id"].fillna("").astype(str).isin(["", "nan", "None", "<NA>"]),
                "missing_traj_id",
                "Cleaned row is missing traj_id.",
            )

            flagged_df = (
                pd.concat(issue_frames, ignore_index=True, sort=False)
                if issue_frames
                else pd.DataFrame(columns=["issue_type", "issue_detail"])
            )
            n_flagged_rows_total = int(len(flagged_df))
            flagged_truncated = False
            if len(flagged_df) > audit_config["flag_row_limit_per_shard"]:
                flagged_df = flagged_df.head(audit_config["flag_row_limit_per_shard"]).copy()
                flagged_truncated = True

            trajectory_summary_df = build_trajectory_summary(
                df,
                residual_speed_mask=residual_speed_mask,
                nonpositive_time_mask=nonpositive_time_mask,
            )

            flagged_df.to_feather(output_paths["flagged_rows"])
            trajectory_summary_df.to_csv(output_paths["trajectory_summary"], index=False)

            issue_counts = (
                flagged_df["issue_type"].value_counts().sort_index().to_dict()
                if not flagged_df.empty
                else {}
            )
            vessel_issue_counts = (
                flagged_df.groupby(SHIP_ID_COL).size().sort_values(ascending=False).head(
                    audit_config["top_vessel_limit"]
                )
                if SHIP_ID_COL in flagged_df.columns and not flagged_df.empty
                else pd.Series(dtype="int64")
            )
            stats_payload = {
                "input_file": str(input_file),
                "input_rows": input_rows,
                "n_flagged_issue_rows_total": n_flagged_rows_total,
                "n_flagged_issue_rows_written": int(len(flagged_df)),
                "flagged_rows_truncated": flagged_truncated,
                "n_unique_flagged_vessels": (
                    int(flagged_df[SHIP_ID_COL].nunique())
                    if SHIP_ID_COL in flagged_df.columns and not flagged_df.empty
                    else 0
                ),
                "n_anomalous_trajectories": int(len(trajectory_summary_df)),
                "issue_counts_written": {key: int(value) for key, value in issue_counts.items()},
                "top_flagged_vessels_written": {
                    str(key): int(value) for key, value in vessel_issue_counts.to_dict().items()
                },
                "audit_config": audit_config,
                "processed_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_json(output_paths["stats"], stats_payload)

            result_payload = {
                "task_id": task["task_id"],
                "status": "success",
                "input_file": str(input_file),
                "stats_path": str(output_paths["stats"]),
                "error": "",
            }
            write_json(task_result_path, result_payload)
            print(f"Flagged issue rows: {n_flagged_rows_total:,}")
            print(f"Anomalous trajectories: {len(trajectory_summary_df):,}")
            print(f"Saved audit outputs under {output_paths['flagged_rows'].parent}")
            return result_payload
    except Exception as exc:
        result_payload = {
            "task_id": task["task_id"],
            "status": "failed",
            "input_file": str(input_file),
            "stats_path": str(output_paths["stats"]),
            "error": str(exc),
        }
        write_json(task_result_path, result_payload)
        return result_payload


def summarize_audit(stats_rows, output_path):
    if not stats_rows:
        Path(output_path).write_text("No audit stats were available.\n", encoding="utf-8")
        return

    stats_df = pd.DataFrame(stats_rows)
    issue_totals = {}
    for issue_counts in stats_df["issue_counts_written"].dropna():
        for issue_type, count in issue_counts.items():
            issue_totals[issue_type] = issue_totals.get(issue_type, 0) + int(count)

    report_lines = [
        "=== AIS Second-Pass Audit Summary ===",
        f"Audited shard files:          {len(stats_rows):,}",
        f"Input rows audited:           {int(stats_df['input_rows'].sum()):,}",
        f"Flagged issue rows written:   {int(stats_df['n_flagged_issue_rows_written'].sum()):,}",
        f"Anomalous trajectories:       {int(stats_df['n_anomalous_trajectories'].sum()):,}",
        f"Shards with truncated flags:  {int(stats_df['flagged_rows_truncated'].sum()):,}",
        "",
        "Issue counts written:",
    ]
    if issue_totals:
        for issue_type, count in sorted(issue_totals.items()):
            report_lines.append(f"  - {issue_type}: {count:,}")
    else:
        report_lines.append("  - none")
    Path(output_path).write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)

    config = load_config()
    iog.set_working_directory(str(SUBPROJECT_ROOT))

    output_root_dir = get_subproject_path(config["output_dir"])
    input_dir = output_root_dir / INPUT_STEP_TAG
    output_dir = output_root_dir / STEP_TAG
    task_result_dir = LOG_DIR / "task_results"
    ensure_directories(output_dir, LOG_DIR, task_result_dir)

    input_files = list_cleaned_point_shard_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No cleaned shard Feather files found under {input_dir}")

    specs_path = get_subproject_path(config["specs_table_path"])
    specs_df = load_specs_table(specs_path)
    audit_config = get_audit_config(config)
    max_workers = int(config.get("parallel", {}).get("max_workers", 4))

    all_tasks = [build_task(input_file, output_dir, task_result_dir) for input_file in input_files]
    skipped_tasks = [
        task for task in all_tasks if audit_outputs_exist(output_dir, task["input_file"])
    ]
    pending_tasks = [
        task for task in all_tasks if not audit_outputs_exist(output_dir, task["input_file"])
    ]

    run_dir = build_run_log_dir(
        RUN_LOG_ROOT_DIR,
        STEP_TAG,
        input_files,
        f"audit_{audit_config['residual_speed_multiplier']}_{max_workers}",
    )
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
            "n_input_files": len(input_files),
            "n_pending_files": len(pending_tasks),
            "n_skipped_existing_outputs": len(skipped_tasks),
            "audit_config": audit_config,
        },
    )
    write_json(run_dir / "all_tasks.json", all_tasks)
    write_json(run_dir / "pending_tasks.json", pending_tasks)
    write_json(run_dir / "skipped_existing_output_tasks.json", skipped_tasks)

    if pending_tasks:
        sb.process_files_parallel(
            audit_single_shard,
            input_files=natsort.natsorted(pending_tasks, key=lambda task: task["task_id"]),
            log_dir=LOG_DIR,
            specs_df=specs_df,
            audit_config=audit_config,
            max_workers=max_workers,
            show_progress=True,
        )

    failed_tasks = []
    stats_rows = []
    for task in all_tasks:
        result_path = Path(task["task_result_path"])
        result_payload = read_stats_json(result_path)
        if result_payload is not None and result_payload.get("status") == "failed":
            failed_tasks.append(result_payload)
            continue

        stats_payload = read_stats_json(task["stats_path"])
        if stats_payload is None:
            failed_tasks.append(
                {
                    "task_id": task["task_id"],
                    "status": "failed",
                    "input_file": task["input_file"],
                    "error": "Missing audit stats output after run.",
                }
            )
            continue
        stats_rows.append(stats_payload)

    write_json(run_dir / "failed_tasks.json", failed_tasks)
    summary_df = pd.DataFrame(stats_rows)
    if not summary_df.empty:
        summary_df.to_csv(run_dir / "summary_by_shard.csv", index=False)
    else:
        pd.DataFrame(columns=["input_file"]).to_csv(run_dir / "summary_by_shard.csv", index=False)

    summarize_audit(stats_rows, output_dir / "second_pass_audit_report.txt")

    print(f"Input cleaned shard files: {len(input_files)}")
    print(f"Pending audit shard files this run: {len(pending_tasks)}")
    print(f"Skipped existing audit outputs this run: {len(skipped_tasks)}")
    print(f"Failed audit shard files this run: {len(failed_tasks)}")
    print(f"Run log dir: {run_dir}")


if __name__ == "__main__":
    main()
