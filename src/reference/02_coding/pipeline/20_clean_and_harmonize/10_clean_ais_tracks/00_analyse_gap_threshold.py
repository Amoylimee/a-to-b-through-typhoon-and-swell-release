import hashlib
from datetime import datetime
from pathlib import Path

import natsort
import pandas as pd
import iogenius as iog

from cleaning_common import (
    SHIP_ID_COL,
    TIMESTAMP_COL,
    SUBPROJECT_ROOT,
    build_run_log_dir,
    enrich_with_specs,
    ensure_directories,
    get_subproject_path,
    list_input_files,
    load_config,
    load_specs_table,
    normalize_dynamic_df,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent

STEP_TAG = "00_analyse_gap_threshold"
OUTPUT_DIRNAME = STEP_TAG
LOG_DIR = SUBPROJECT_ROOT / ".runs" / "clean_ais_tracks" / STEP_TAG


def summarise_gap_hours(gap_hours, quantiles):
    if gap_hours.empty:
        return {
            "n_gaps": 0,
            "mean_hours": None,
            "median_hours": None,
            "max_hours": None,
            **{f"p{int(quantile * 1000):03d}_hours": None for quantile in quantiles},
        }

    summary = {
        "n_gaps": int(gap_hours.shape[0]),
        "mean_hours": float(gap_hours.mean()),
        "median_hours": float(gap_hours.median()),
        "max_hours": float(gap_hours.max()),
    }
    for quantile in quantiles:
        summary[f"p{int(quantile * 1000):03d}_hours"] = float(gap_hours.quantile(quantile))
    return summary


def collect_gap_rows(input_files, specs_df):
    gap_rows = []
    per_shard_rows = []

    for input_file in input_files:
        df = pd.read_feather(input_file)
        df = normalize_dynamic_df(df)
        df = df.loc[(df[SHIP_ID_COL] != "") & df[TIMESTAMP_COL].notna()].copy()
        if df.empty:
            per_shard_rows.append(
                {"input_file": str(input_file), "n_rows_input": 0, "n_rows_usable": 0, "n_gaps": 0}
            )
            continue

        df = enrich_with_specs(df, specs_df)
        df = df.sort_values(by=[SHIP_ID_COL, TIMESTAMP_COL], kind="stable").reset_index(drop=True)
        df["gap_hours"] = (
            df.groupby(SHIP_ID_COL, sort=False)[TIMESTAMP_COL].diff().dt.total_seconds() / 3600.0
        )
        valid_gap_df = df.loc[df["gap_hours"].notna()].copy()
        valid_gap_df["input_file"] = str(input_file.relative_to(SUBPROJECT_ROOT))
        gap_rows.extend(
            valid_gap_df[[SHIP_ID_COL, "imo_ship_type", "gap_hours", "input_file"]].to_dict("records")
        )
        per_shard_rows.append(
            {
                "input_file": str(input_file.relative_to(SUBPROJECT_ROOT)),
                "n_rows_input": int(len(df)),
                "n_rows_usable": int(len(df)),
                "n_gaps": int(len(valid_gap_df)),
            }
        )

    return pd.DataFrame(gap_rows), pd.DataFrame(per_shard_rows)


def build_candidate_table(gap_hours, quantiles):
    candidate_rows = []
    if gap_hours.empty:
        return pd.DataFrame(
            columns=["candidate_source", "threshold_hours", "rounded_threshold_hours", "note"]
        )

    for quantile in quantiles:
        threshold = float(gap_hours.quantile(quantile))
        candidate_rows.append(
            {
                "candidate_source": f"quantile_{quantile:.3f}",
                "threshold_hours": threshold,
                "rounded_threshold_hours": round(threshold, 3),
                "note": "Copy one confirmed value into config.yaml gap.threshold_hours after review.",
            }
        )
    for fixed_hours in [1.0, 2.0, 3.0, 4.0, 6.0]:
        candidate_rows.append(
            {
                "candidate_source": "guide_reference_window",
                "threshold_hours": fixed_hours,
                "rounded_threshold_hours": round(fixed_hours, 3),
                "note": "Guide says typical ocean-going threshold candidates often land between 1 and 6 hours.",
            }
        )
    candidate_df = pd.DataFrame(candidate_rows)
    return candidate_df.drop_duplicates(subset=["rounded_threshold_hours"], keep="first").reset_index(
        drop=True
    )


def write_analysis_notes(path, gap_df, quantiles, sample_files, specs_path):
    overall = summarise_gap_hours(gap_df["gap_hours"], quantiles) if not gap_df.empty else {}
    lines = [
        "AIS gap-threshold analysis notes",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Specs table: {specs_path}",
        f"Sample shards analysed: {len(sample_files)}",
        "Sample shard files:",
    ]
    lines.extend([f"- {Path(file_path).name}" for file_path in sample_files])
    lines.append("")
    if gap_df.empty:
        lines.append("No inter-point gaps were available in the analysed sample.")
    else:
        lines.append("Overall gap summary (hours):")
        lines.append(f"- n_gaps: {overall['n_gaps']}")
        lines.append(f"- mean: {overall['mean_hours']:.6f}")
        lines.append(f"- median: {overall['median_hours']:.6f}")
        for quantile in quantiles:
            key = f"p{int(quantile * 1000):03d}_hours"
            lines.append(f"- {key}: {overall[key]:.6f}")
        lines.append(f"- max: {overall['max_hours']:.6f}")
        lines.append("")
        lines.append(
            "Action: inspect recommended_threshold_candidates.csv, choose a confirmed threshold, "
            "and then set gap.threshold_hours in config.yaml manually before running 10_clean_dynamic_ais.py."
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)

    config = load_config()
    iog.set_working_directory(str(SUBPROJECT_ROOT))
    output_root_dir = get_subproject_path(config["output_dir"])
    output_dir = output_root_dir / STEP_TAG
    ensure_directories(output_dir, LOG_DIR)

    input_dir = get_subproject_path(config["input_dir"])
    specs_path = get_subproject_path(config["specs_table_path"])
    sample_shards = int(config.get("gap", {}).get("sample_shards", 3))
    quantiles = [float(value) for value in config.get("report", {}).get("sample_gap_quantiles", [])]

    input_files = list_input_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No input shard Feather files found under {input_dir}")

    sample_files = natsort.natsorted(input_files[:sample_shards])
    specs_df = load_specs_table(specs_path)
    config_fingerprint = hashlib.md5(
        f"{sample_shards}|{quantiles}|{specs_path}".encode("utf-8")
    ).hexdigest()[:8]
    run_dir = build_run_log_dir(LOG_DIR.parent, STEP_TAG, sample_files, config_fingerprint)

    gap_df, per_shard_df = collect_gap_rows(sample_files, specs_df)

    if gap_df.empty:
        overall_summary_df = pd.DataFrame([summarise_gap_hours(pd.Series(dtype=float), quantiles)])
        by_ship_type_df = pd.DataFrame()
    else:
        overall_summary_df = pd.DataFrame([summarise_gap_hours(gap_df["gap_hours"], quantiles)])
        by_ship_type_rows = []
        for imo_ship_type, ship_type_df in gap_df.groupby("imo_ship_type", dropna=False):
            summary_row = {"imo_ship_type": imo_ship_type}
            summary_row.update(summarise_gap_hours(ship_type_df["gap_hours"], quantiles))
            by_ship_type_rows.append(summary_row)
        by_ship_type_df = pd.DataFrame(by_ship_type_rows)

    candidate_df = build_candidate_table(
        gap_df["gap_hours"] if not gap_df.empty else pd.Series(dtype=float),
        quantiles,
    )

    overall_path = output_dir / "gap_distribution_overall.csv"
    by_type_path = output_dir / "gap_distribution_by_ship_type.csv"
    candidate_path = output_dir / "recommended_threshold_candidates.csv"
    notes_path = output_dir / "analysis_notes.txt"
    shard_summary_path = output_dir / "sample_shard_summary.csv"

    overall_summary_df.to_csv(overall_path, index=False)
    by_ship_type_df.to_csv(by_type_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    per_shard_df.to_csv(shard_summary_path, index=False)
    write_analysis_notes(notes_path, gap_df, quantiles, sample_files, specs_path)

    write_json(
        run_dir / "run_config.json",
        {
            "run_started_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": config["_config_path"],
            "input_dir": str(input_dir),
            "specs_table_path": str(specs_path),
            "sample_shards": sample_shards,
            "sample_input_files": [str(path) for path in sample_files],
            "report_quantiles": quantiles,
        },
    )
    pd.DataFrame({"input_file": [str(path) for path in sample_files]}).to_csv(
        run_dir / "pending_tasks.txt", index=False, header=False
    )
    overall_summary_df.to_csv(run_dir / "gap_distribution_overall.csv", index=False)
    by_ship_type_df.to_csv(run_dir / "gap_distribution_by_ship_type.csv", index=False)
    candidate_df.to_csv(run_dir / "recommended_threshold_candidates.csv", index=False)
    per_shard_df.to_csv(run_dir / "sample_shard_summary.csv", index=False)

    print(f"Sample shards analysed: {len(sample_files)}")
    print(f"Overall gap summary: {overall_path}")
    print(f"Per ship-type gap summary: {by_type_path}")
    print(f"Threshold candidates: {candidate_path}")
    print(f"Analysis notes: {notes_path}")
    print(f"Run log dir: {run_dir}")


if __name__ == "__main__":
    main()
