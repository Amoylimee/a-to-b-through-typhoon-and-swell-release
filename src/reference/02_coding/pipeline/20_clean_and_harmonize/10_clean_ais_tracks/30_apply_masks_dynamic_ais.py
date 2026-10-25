from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re

import natsort
import numpy as np
import pandas as pd
import sinbue as sb
import iogenius as iog

from cleaning_common import (
    COURSE_COL,
    HEADING_COL,
    SHIP_ID_COL,
    SPEED_COL,
    SUBPROJECT_ROOT,
    TIMESTAMP_COL,
    build_run_log_dir,
    ensure_directories,
    get_subproject_path,
    load_config,
    list_cleaned_point_shard_files,
    read_stats_json,
    shard_output_paths,
    write_json,
)


STEP_TAG = "30_apply_masks_dynamic_ais"
INPUT_STEP_TAG = "10_clean_dynamic_ais"
AUDIT_STEP_TAG = "20_audit_dynamic_ais"
LOG_DIR = SUBPROJECT_ROOT / ".runs" / "clean_ais_tracks" / STEP_TAG
RUN_LOG_ROOT_DIR = LOG_DIR.parent
# The per-year analysis specs are emitted by 10_ingest_data/10_define_observed_fleet,
# not inherited: only 2021 and 2025 were ever published, and all five years must
# enforce the same observed fleet for the analysis filter below to mean anything.
ANALYSIS_SPECS_DIR = (
    SUBPROJECT_ROOT
    / "data"
    / "processed"
    / "products"
    / "vessel_universe"
)

RESIDUAL_SPEED_ISSUE = "residual_speed_jump"
REPORTED_SOG_ISSUE = "reported_sog_over_limit"
COURSE_ISSUE = "course_out_of_range"
HEADING_ISSUE = "heading_out_of_range"


def second_pass_output_paths(output_dir, input_file):
    input_file = Path(input_file)
    return {
        "cleaned": Path(output_dir) / input_file.name,
        "stats": Path(output_dir) / f"{input_file.stem}_stats.json",
    }


def second_pass_outputs_exist(output_dir, input_file):
    output_paths = second_pass_output_paths(output_dir, input_file)
    if not all(path.exists() for path in output_paths.values()):
        return False

    stats_payload = read_stats_json(output_paths["stats"])
    return bool(stats_payload and stats_payload.get("analysis_vessel_filter_enabled") is True)


def audit_flagged_rows_path(audit_dir, input_file):
    input_file = Path(input_file)
    return Path(audit_dir) / f"{input_file.stem}_flagged_rows.feather"


def build_task(input_file, output_dir, audit_dir, task_result_dir, analysis_specs_path):
    input_file = Path(input_file)
    output_paths = second_pass_output_paths(output_dir, input_file)
    return {
        "task_id": input_file.stem,
        "input_file": str(input_file),
        "audit_flagged_rows_path": str(audit_flagged_rows_path(audit_dir, input_file)),
        "output_path": str(output_paths["cleaned"]),
        "stats_path": str(output_paths["stats"]),
        "task_result_path": str(Path(task_result_dir) / f"{input_file.stem}.json"),
        "analysis_specs_path": str(analysis_specs_path),
    }


def infer_year_from_output_dir(output_dir):
    match = re.search(r"ais_cleaning_(\d{4})", str(output_dir))
    if match is None:
        raise ValueError(f"Could not infer AIS cleaning year from output_dir: {output_dir}")
    return int(match.group(1))


def analysis_specs_path(year):
    return ANALYSIS_SPECS_DIR / f"analysis_vessel_specs_{year}.csv"


def clean_int_series(series):
    text = series.fillna("").astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    cleaned = text.copy()
    numeric_mask = numeric.notna()
    cleaned.loc[numeric_mask] = numeric.loc[numeric_mask].astype("int64").astype(str)
    return cleaned


@lru_cache(maxsize=None)
def load_analysis_ship_ids(specs_path):
    specs_path = Path(specs_path)
    if not specs_path.exists():
        raise FileNotFoundError(
            f"Analysis vessel specs not found: {specs_path}. "
            "Resolve the approved analysis_vessel_specs link first."
        )

    header = pd.read_csv(specs_path, nrows=0)
    if "vessel_uid_clean" in header.columns:
        id_column = "vessel_uid_clean"
    elif SHIP_ID_COL in header.columns:
        id_column = SHIP_ID_COL
    else:
        raise ValueError(f"{specs_path} is missing vessel_uid/vessel_uid_clean.")

    specs = pd.read_csv(specs_path, usecols=[id_column], dtype=str)
    ship_ids = set(clean_int_series(specs[id_column]))
    ship_ids.discard("")
    if not ship_ids:
        raise ValueError(f"No analysis vessel_uid values found in {specs_path}")
    return ship_ids


def filter_to_analysis_vessels(df, specs_path):
    if SHIP_ID_COL not in df.columns:
        raise ValueError(f"Cleaned AIS shard missing required column: {SHIP_ID_COL}")

    analysis_ship_ids = load_analysis_ship_ids(str(specs_path))
    ship_ids = clean_int_series(df[SHIP_ID_COL])
    return df.loc[ship_ids.isin(analysis_ship_ids)].copy(), len(analysis_ship_ids)


def read_issue_row_numbers(flagged_rows_path, issue_type):
    flagged_rows_path = Path(flagged_rows_path)
    if not flagged_rows_path.exists():
        raise FileNotFoundError(f"Required audit flagged-rows file not found: {flagged_rows_path}")

    flagged_df = pd.read_feather(flagged_rows_path)
    if flagged_df.empty:
        return set()
    if "_row_number" not in flagged_df.columns or "issue_type" not in flagged_df.columns:
        raise ValueError(
            f"Audit flagged-rows file missing _row_number/issue_type columns: {flagged_rows_path}"
        )

    issue_rows = flagged_df.loc[flagged_df["issue_type"] == issue_type, "_row_number"]
    return set(pd.to_numeric(issue_rows, errors="coerce").dropna().astype("int64").tolist())


def build_mask_series(row_numbers, flagged_row_numbers):
    if not flagged_row_numbers:
        return pd.Series(False, index=row_numbers.index)
    return row_numbers.isin(flagged_row_numbers)


def apply_second_pass_masks(df, flagged_rows_path):
    df = df.copy()
    row_numbers = pd.Series(np.arange(len(df), dtype="int64"), index=df.index)

    residual_speed_rows = read_issue_row_numbers(flagged_rows_path, RESIDUAL_SPEED_ISSUE)
    reported_sog_rows = read_issue_row_numbers(flagged_rows_path, REPORTED_SOG_ISSUE)
    course_rows = read_issue_row_numbers(flagged_rows_path, COURSE_ISSUE)
    heading_rows = read_issue_row_numbers(flagged_rows_path, HEADING_ISSUE)

    df["is_residual_speed_jump"] = build_mask_series(row_numbers, residual_speed_rows)
    df["is_reported_sog_over_limit"] = build_mask_series(row_numbers, reported_sog_rows)
    df["is_course_out_of_range"] = build_mask_series(row_numbers, course_rows)
    df["is_heading_out_of_range"] = build_mask_series(row_numbers, heading_rows)
    df["has_second_pass_issue"] = (
        df["is_residual_speed_jump"]
        | df["is_reported_sog_over_limit"]
        | df["is_course_out_of_range"]
        | df["is_heading_out_of_range"]
    )

    if "calc_speed_knots" in df.columns:
        df["calc_speed_knots_masked"] = pd.to_numeric(
            df["calc_speed_knots"], errors="coerce"
        )
        df.loc[df["is_residual_speed_jump"], "calc_speed_knots_masked"] = np.nan

    if SPEED_COL in df.columns:
        df["speed_masked"] = pd.to_numeric(df[SPEED_COL], errors="coerce")
        df.loc[df["is_reported_sog_over_limit"], "speed_masked"] = np.nan

    if COURSE_COL in df.columns:
        df["course_masked"] = pd.to_numeric(df[COURSE_COL], errors="coerce")
        df.loc[df["is_course_out_of_range"], "course_masked"] = np.nan

    if HEADING_COL in df.columns:
        df["heading_masked"] = pd.to_numeric(df[HEADING_COL], errors="coerce")
        df.loc[df["is_heading_out_of_range"], "heading_masked"] = np.nan

    stats = {
        "input_rows": int(len(df)),
        "n_residual_speed_jump": int(df["is_residual_speed_jump"].sum()),
        "n_reported_sog_over_limit": int(df["is_reported_sog_over_limit"].sum()),
        "n_course_out_of_range": int(df["is_course_out_of_range"].sum()),
        "n_heading_out_of_range": int(df["is_heading_out_of_range"].sum()),
        "n_rows_with_second_pass_issue": int(df["has_second_pass_issue"].sum()),
    }
    return df, stats


def process_task(task, log_dir):
    input_file = Path(task["input_file"])
    output_path = Path(task["output_path"])
    stats_path = Path(task["stats_path"])
    task_result_path = Path(task["task_result_path"])
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    ensure_directories(output_path.parent, stats_path.parent, task_result_path.parent, Path(log_dir))

    try:
        with sb.PrintRedirector(Path(log_dir) / f"{task['task_id']}.log"):
            print(f"Applying second-pass masks to {input_file}")
            df = pd.read_feather(input_file)
            masked_df, stats_payload = apply_second_pass_masks(
                df,
                flagged_rows_path=task["audit_flagged_rows_path"],
            )
            filtered_df, n_analysis_ship_ids = filter_to_analysis_vessels(
                masked_df,
                specs_path=task["analysis_specs_path"],
            )
            stats_payload.update(
                {
                    "analysis_vessel_filter_enabled": True,
                    "analysis_specs_path": task["analysis_specs_path"],
                    "n_analysis_ship_ids": int(n_analysis_ship_ids),
                    "rows_before_analysis_vessel_filter": int(len(masked_df)),
                    "rows_after_analysis_vessel_filter": int(len(filtered_df)),
                    "rows_dropped_by_analysis_vessel_filter": int(
                        len(masked_df) - len(filtered_df)
                    ),
                    "n_unique_ship_ids_after_analysis_vessel_filter": int(
                        clean_int_series(filtered_df[SHIP_ID_COL]).nunique()
                        if not filtered_df.empty
                        else 0
                    ),
                }
            )
            filtered_df.to_feather(tmp_output_path)
            tmp_output_path.replace(output_path)

            stats_payload.update(
                {
                    "input_file": str(input_file),
                    "audit_flagged_rows_path": task["audit_flagged_rows_path"],
                    "output_path": str(output_path),
                    "processed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            write_json(stats_path, stats_payload)

            result_payload = {
                "task_id": task["task_id"],
                "status": "success",
                "input_file": str(input_file),
                "output_path": str(output_path),
                "stats_path": str(stats_path),
                "error": "",
            }
            write_json(task_result_path, result_payload)
            print(f"Rows with second-pass issue: {stats_payload['n_rows_with_second_pass_issue']:,}")
            print(f"Saved masked shard to {output_path}")
            return result_payload
    except Exception as exc:
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        result_payload = {
            "task_id": task["task_id"],
            "status": "failed",
            "input_file": str(input_file),
            "output_path": str(output_path),
            "stats_path": str(stats_path),
            "error": str(exc),
        }
        write_json(task_result_path, result_payload)
        return result_payload


def summarize_stats(stats_rows, output_path):
    if not stats_rows:
        Path(output_path).write_text("No second-pass mask stats were available.\n", encoding="utf-8")
        return

    stats_df = pd.DataFrame(stats_rows)
    report_lines = [
        "=== AIS Second-Pass Mask Summary ===",
        f"Processed shard files:           {len(stats_rows):,}",
        f"Input rows processed:            {int(stats_df['input_rows'].sum()):,}",
        f"Rows after analysis filter:      {int(stats_df['rows_after_analysis_vessel_filter'].sum()):,}",
        f"Rows dropped by analysis filter: {int(stats_df['rows_dropped_by_analysis_vessel_filter'].sum()):,}",
        f"Rows with any second-pass issue: {int(stats_df['n_rows_with_second_pass_issue'].sum()):,}",
        "",
        f"Residual speed jumps masked:     {int(stats_df['n_residual_speed_jump'].sum()):,}",
        f"Reported SOG values masked:      {int(stats_df['n_reported_sog_over_limit'].sum()):,}",
        f"Course values masked:            {int(stats_df['n_course_out_of_range'].sum()):,}",
        f"Heading values masked:           {int(stats_df['n_heading_out_of_range'].sum()):,}",
    ]
    Path(output_path).write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)

    config = load_config()
    iog.set_working_directory(str(SUBPROJECT_ROOT))

    output_root_dir = get_subproject_path(config["output_dir"])
    input_dir = output_root_dir / INPUT_STEP_TAG
    audit_dir = output_root_dir / AUDIT_STEP_TAG
    output_dir = output_root_dir / STEP_TAG
    task_result_dir = LOG_DIR / "task_results"
    ensure_directories(output_dir, LOG_DIR, task_result_dir)
    cleaning_year = infer_year_from_output_dir(output_dir)
    analysis_specs = analysis_specs_path(cleaning_year)
    if not analysis_specs.exists():
        raise FileNotFoundError(
            f"Analysis vessel specs not found: {analysis_specs}. "
            "Resolve the approved analysis_vessel_specs link first."
        )

    input_files = list_cleaned_point_shard_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No first-pass cleaned shard files found under {input_dir}")
    if not audit_dir.exists():
        raise FileNotFoundError(f"Second-pass audit directory not found: {audit_dir}")

    max_workers = int(config.get("parallel", {}).get("max_workers", 4))
    all_tasks = [
        build_task(input_file, output_dir, audit_dir, task_result_dir, analysis_specs)
        for input_file in input_files
    ]
    skipped_tasks = [
        task for task in all_tasks if second_pass_outputs_exist(output_dir, task["input_file"])
    ]
    pending_tasks = [
        task for task in all_tasks if not second_pass_outputs_exist(output_dir, task["input_file"])
    ]

    run_dir = build_run_log_dir(
        RUN_LOG_ROOT_DIR,
        STEP_TAG,
        input_files,
        f"mask_{max_workers}",
    )
    write_json(
        run_dir / "run_config.json",
        {
            "run_started_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": config["_config_path"],
            "input_dir": str(input_dir),
            "audit_dir": str(audit_dir),
            "output_dir": str(output_dir),
            "cleaning_year": cleaning_year,
            "analysis_specs_path": str(analysis_specs),
            "analysis_vessel_filter_enabled": True,
            "log_dir": str(LOG_DIR),
            "max_workers": max_workers,
            "n_input_files": len(input_files),
            "n_pending_files": len(pending_tasks),
            "n_skipped_existing_outputs": len(skipped_tasks),
        },
    )
    write_json(run_dir / "all_tasks.json", all_tasks)
    write_json(run_dir / "pending_tasks.json", pending_tasks)
    write_json(run_dir / "skipped_existing_output_tasks.json", skipped_tasks)

    if pending_tasks:
        sb.process_files_parallel(
            process_task,
            input_files=natsort.natsorted(pending_tasks, key=lambda task: task["task_id"]),
            log_dir=LOG_DIR,
            max_workers=max_workers,
            show_progress=True,
        )

    failed_tasks = []
    stats_rows = []
    for task in all_tasks:
        result_payload = read_stats_json(task["task_result_path"])
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
                    "error": "Missing second-pass mask stats output after run.",
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

    summarize_stats(stats_rows, output_dir / "second_pass_mask_report.txt")

    print(f"Input first-pass cleaned shard files: {len(input_files)}")
    print(f"Pending second-pass mask shard files this run: {len(pending_tasks)}")
    print(f"Skipped existing second-pass outputs this run: {len(skipped_tasks)}")
    print(f"Failed second-pass mask shard files this run: {len(failed_tasks)}")
    print(f"Run log dir: {run_dir}")


if __name__ == "__main__":
    main()
