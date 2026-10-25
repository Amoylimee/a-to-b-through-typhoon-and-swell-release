#!/usr/bin/env python3
"""Build PI-approved year-specific pooled q01/q99 endpoint samples."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

CODING_ROOT = Path(__file__).resolve().parents[4]
C5_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    CODING_ROOT
    / "data/processed/build_multicall_first_passage_voyages"
)
DEFAULT_CONFIG = C5_ROOT / "config/frozen/global_qtrim_2021_2025.json"
DEFAULT_ARTIFACT_DIR = DATA_ROOT / "main"
DEFAULT_REPORT_DIR = DATA_ROOT / "main/validation/endpoint_2021_2025"

COMPACT_COLUMNS = [
    "candidate_id",
    "imo",
    "pair_id",
    "shiptype_group",
    "departure_month",
    "total_voyage_time_days",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_file(config: dict[str, Any], key: str) -> Path:
    path = CODING_ROOT / config[key]
    if sha256_file(path) != config[f"{key}_sha256"]:
        raise ValueError(f"Input hash mismatch: {key}")
    return path


def verify_contract(config: dict[str, Any]) -> None:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=CODING_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != config["approved_branch"]:
        raise ValueError("Unexpected branch")
    if config["analysis_years"] != [2021, 2025]:
        raise ValueError("Only endpoint years are allowed")
    if config["threshold_scope"] != "year_specific_pooled":
        raise ValueError("Global qtrim must be fitted separately within each year")
    if config["grouping_columns"]:
        raise ValueError("Global qtrim cannot group by ship type or OD pair")
    if config["q_lower"] != 0.01 or config["q_upper"] != 0.99:
        raise ValueError("Global qtrim quantiles changed")
    if config["decision_mode"] != "pi_approved_candidate_main":
        raise ValueError("Global qtrim lacks the approved decision label")
    if config["allow_middle_years"]:
        raise ValueError("Middle years cannot be read")
    if config["allow_shiptype_grouping"] or config["allow_od_grouping"]:
        raise ValueError("Grouped qtrim is not permitted")
    for key in (
        "base_rebuild_config",
        "c3_manifest_2021",
        "c3_manifest_2025",
        "shiptype_final_manifest_2021",
        "shiptype_final_manifest_2025",
        "p4_city_shiptype_exposure",
    ):
        verify_file(config, key)


def assign_suez_bin(values: pd.Series, missing_label: str) -> pd.Series:
    bins = pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 0.0, 0.25, 0.5, 0.75, np.inf],
        labels=["zero", "low", "medium", "high", "very_high"],
        include_lowest=True,
        right=True,
    )
    return bins.astype("string").fillna(missing_label)


def pretrim_paths(year: int, config: dict[str, Any]) -> list[Path]:
    source_dir = CODING_ROOT / config["pretrim_dir_pattern"].format(year=year)
    paths = sorted(source_dir.glob("part-*.feather"))
    if len(paths) != config["expected_pretrim_parts_per_year"]:
        raise ValueError(f"Unexpected pretrim part count for {year}")
    return paths


def fit_global_thresholds(
    paths: list[Path],
    q_lower: float,
    q_upper: float,
    parameter_year: int,
) -> dict[str, Any]:
    compact = pd.concat(
        [
            pd.read_feather(
                path,
                columns=[
                    "normalized_time_ratio",
                    "temporal_scope_eligible",
                    "d_yardstick",
                ],
            )
            for path in paths
        ],
        ignore_index=True,
    )
    eligible = (
        compact["temporal_scope_eligible"].astype(bool)
        & np.isfinite(compact["normalized_time_ratio"])
        & compact["d_yardstick"].gt(0)
    )
    values = compact.loc[eligible, "normalized_time_ratio"]
    return {
        "parameter_year": parameter_year,
        "threshold_scope": "year_specific_pooled",
        "grouping_columns": [],
        "eligible_rows": int(len(values)),
        "normalized_time_ratio_q01": float(values.quantile(q_lower)),
        "normalized_time_ratio_q99": float(values.quantile(q_upper)),
        "equality_retained": True,
    }


def apply_global_qtrim(
    frame: pd.DataFrame,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    output = frame.copy()
    eligible = (
        output["temporal_scope_eligible"].astype(bool)
        & np.isfinite(output["normalized_time_ratio"])
        & output["d_yardstick"].gt(0)
    )
    output["global_normalized_time_ratio_q01"] = lower
    output["global_normalized_time_ratio_q99"] = upper
    output["global_trimmed_below_q01"] = eligible & output[
        "normalized_time_ratio"
    ].lt(lower)
    output["global_trimmed_above_q99"] = eligible & output[
        "normalized_time_ratio"
    ].gt(upper)
    output["global_qtrimmed"] = (
        output["global_trimmed_below_q01"]
        | output["global_trimmed_above_q99"]
    )
    output["retain_after_global_qtrim"] = (
        output["temporal_scope_eligible"].astype(bool)
        & ~output["global_qtrimmed"]
    )
    output["global_qtrim_version"] = "pooled_2021_q01_q99_no_grouping"
    return output


def part_is_complete(
    source: Path,
    output: Path,
    sidecar: Path,
    contract_hash: str,
) -> bool:
    if not output.is_file() or not sidecar.is_file():
        return False
    prior = load_json(sidecar)
    return bool(
        prior.get("contract_hash") == contract_hash
        and prior.get("source_sha256") == sha256_file(source)
        and prior.get("output_sha256") == sha256_file(output)
    )


def process_part(
    source: Path,
    output: Path,
    sidecar: Path,
    exposure: pd.DataFrame | None,
    lower: float,
    upper: float,
    missing_label: str,
    contract_hash: str,
) -> dict[str, Any]:
    if part_is_complete(source, output, sidecar, contract_hash):
        return load_json(sidecar)
    frame = apply_global_qtrim(pd.read_feather(source), lower, upper)
    # Legacy callers may attach exposure descriptors; annual data preparation
    # does not require or generate those unrelated research variables.
    if exposure is not None:
        frame = frame.merge(
            exposure, on=["pair_id", "shiptype_group"], how="left", validate="many_to_one"
        )
        frame["suez_dependence_bin"] = assign_suez_bin(
            frame["BaselineSuezDependence_g"], missing_label
        )
    retained = frame.loc[frame["retain_after_global_qtrim"]].reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    retained.to_feather(output, compression="zstd", compression_level=6)
    payload = {
        "contract_hash": contract_hash,
        "source_path": str(source.relative_to(CODING_ROOT)),
        "source_sha256": sha256_file(source),
        "input_rows": int(len(frame)),
        "temporal_scope_excluded": int(
            (~frame["temporal_scope_eligible"].astype(bool)).sum()
        ),
        "q01_trimmed": int(frame["global_trimmed_below_q01"].sum()),
        "q99_trimmed": int(frame["global_trimmed_above_q99"].sum()),
        "final_retained": int(len(retained)),
        "exposure_supported": int(
            retained["BaselineSuezDependence_g"].notna().sum() if exposure is not None else 0
        ),
        "positive_exposure": int(
            retained["BaselineSuezDependence_g"].gt(0).sum() if exposure is not None else 0
        ),
        "high_very_exposure": int(
            retained["suez_dependence_bin"].isin(["high", "very_high"]).sum() if exposure is not None else 0
        ),
        "output_path": str(output.relative_to(CODING_ROOT)),
        "output_sha256": sha256_file(output),
    }
    if exposure is None:
        for key in ("exposure_supported", "positive_exposure", "high_very_exposure"):
            payload.pop(key)
    if payload["final_retained"] != (
        payload["input_rows"]
        - payload["temporal_scope_excluded"]
        - payload["q01_trimmed"]
        - payload["q99_trimmed"]
    ):
        raise ValueError("Global qtrim attrition does not balance")
    write_json(sidecar, payload)
    return payload


def build_global_samples(
    config: dict[str, Any],
    thresholds_by_year: dict[int, dict[str, Any]],
    exposure: pd.DataFrame,
    artifact_dir: Path,
) -> dict[int, dict[str, Any]]:
    contract_hash = hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "thresholds_by_year": thresholds_by_year,
                "exposure_sha256": config["p4_city_shiptype_exposure_sha256"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifests = {}
    for year in config["analysis_years"]:
        thresholds = thresholds_by_year[year]
        results = []
        for source in pretrim_paths(year, config):
            output = artifact_dir / f"global_final_sample_{year}" / source.name
            sidecar = (
                artifact_dir
                / f"global_final_sidecars_{year}"
                / source.with_suffix(".json").name
            )
            results.append(
                process_part(
                    source,
                    output,
                    sidecar,
                    exposure,
                    thresholds["normalized_time_ratio_q01"],
                    thresholds["normalized_time_ratio_q99"],
                    config["missing_exposure_label"],
                    contract_hash,
                )
            )
        totals = Counter()
        for item in results:
            for key in (
                "input_rows",
                "temporal_scope_excluded",
                "q01_trimmed",
                "q99_trimmed",
                "final_retained",
                "exposure_supported",
                "positive_exposure",
                "high_very_exposure",
            ):
                totals[key] += item[key]
        manifest = {
            "status": "pass",
            "analysis_year": year,
            "contract_hash": contract_hash,
            "summary": dict(totals),
            "parts": results,
        }
        write_json(
            artifact_dir / f"global_final_sample_manifest_{year}.json",
            manifest,
        )
        manifests[year] = manifest
    return manifests


def load_exposure(config: dict[str, Any]) -> pd.DataFrame:
    exposure = pd.read_feather(
        CODING_ROOT / config["p4_city_shiptype_exposure"],
        columns=["city_pair", "shiptype_group", "BaselineSuezDependence_g"],
    ).rename(columns={"city_pair": "pair_id"})
    if exposure.duplicated(["pair_id", "shiptype_group"]).any():
        raise ValueError("Route exposure duplicates pair by ship type")
    return exposure


def load_compact_sample(
    spec: str,
    year: int,
    config: dict[str, Any],
    exposure: pd.DataFrame,
    artifact_dir: Path,
) -> pd.DataFrame:
    if spec == "global_qtrim":
        paths = sorted((artifact_dir / f"global_final_sample_{year}").glob("*.feather"))
        frame = pd.concat(
            [
                pd.read_feather(
                    path,
                    columns=[
                        *COMPACT_COLUMNS,
                        "BaselineSuezDependence_g",
                        "suez_dependence_bin",
                    ],
                )
                for path in paths
            ],
            ignore_index=True,
        )
    elif spec == "shiptype_qtrim":
        source_dir = CODING_ROOT / config["shiptype_final_dir_pattern"].format(
            year=year
        )
        frame = pd.concat(
            [
                pd.read_feather(
                    path,
                    columns=[
                        *COMPACT_COLUMNS,
                        "BaselineSuezDependence_g",
                        "suez_dependence_bin",
                    ],
                )
                for path in sorted(source_dir.glob("part-*.feather"))
            ],
            ignore_index=True,
        )
    elif spec == "no_qtrim":
        frame = pd.concat(
            [
                pd.read_feather(
                    path,
                    columns=[*COMPACT_COLUMNS, "temporal_scope_eligible"],
                )
                for path in pretrim_paths(year, config)
            ],
            ignore_index=True,
        )
        frame = frame.loc[frame["temporal_scope_eligible"].astype(bool)].drop(
            columns="temporal_scope_eligible"
        )
        frame = frame.merge(
            exposure,
            on=["pair_id", "shiptype_group"],
            how="left",
            validate="many_to_one",
        )
        frame["suez_dependence_bin"] = assign_suez_bin(
            frame["BaselineSuezDependence_g"],
            config["missing_exposure_label"],
        )
    else:
        raise ValueError(f"Unknown sample specification: {spec}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"Duplicate candidate IDs in {spec} {year}")
    return frame


def summarize_sample(spec: str, year: int, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "specification": spec,
        "year": year,
        "voyage_rows": int(len(frame)),
        "exposure_supported_rows": int(
            frame["BaselineSuezDependence_g"].notna().sum()
        ),
        "positive_suez_rows": int(frame["BaselineSuezDependence_g"].gt(0).sum()),
        "high_very_suez_rows": int(
            frame["suez_dependence_bin"].isin(["high", "very_high"]).sum()
        ),
    }


def summarize_panel(spec: str, samples: dict[int, pd.DataFrame]) -> list[dict[str, Any]]:
    panel_frames = []
    for year, frame in samples.items():
        supported = frame.loc[frame["BaselineSuezDependence_g"].notna()]
        panel = (
            supported.groupby(
                ["pair_id", "shiptype_group", "departure_month"],
                observed=True,
            )
            .agg(
                voyage_count=("candidate_id", "size"),
                vessel_count=("imo", "nunique"),
            )
            .reset_index()
        )
        panel["year"] = year
        expected = int(supported["candidate_id"].size)
        if int(panel["voyage_count"].sum()) != expected:
            raise ValueError(f"{spec} {year} monthly panel does not balance")
        panel_frames.append(panel)
    panel = pd.concat(panel_frames, ignore_index=True)
    groups_by_year = panel.groupby(["pair_id", "shiptype_group"])["year"].nunique()
    common = groups_by_year.loc[groups_by_year.eq(2)].index
    panel_index = pd.MultiIndex.from_frame(panel[["pair_id", "shiptype_group"]])
    common_panel = panel.loc[panel_index.isin(common)]
    month_counts = (
        common_panel.groupby(["pair_id", "shiptype_group", "year"])
        .size()
        .unstack("year", fill_value=0)
    )
    at_least_six = month_counts.loc[
        month_counts.get(2021, 0).ge(6) & month_counts.get(2025, 0).ge(6)
    ].index
    balanced = month_counts.loc[
        month_counts.get(2021, 0).eq(12) & month_counts.get(2025, 0).eq(12)
    ].index
    rows = []
    for label, index in (
        ("common_groups", common),
        ("at_least_6_months_each_year", at_least_six),
        ("balanced_12_months_each_year", balanced),
    ):
        selected = panel.loc[panel_index.isin(index)]
        rows.append(
            {
                "specification": spec,
                "sample": label,
                "pair_shiptype_groups": int(len(index)),
                "monthly_observations": int(len(selected)),
                "monthly_observations_ge_2_voyages": int(
                    selected["voyage_count"].ge(2).sum()
                ),
                "monthly_observations_ge_2_vessels": int(
                    selected["vessel_count"].ge(2).sum()
                ),
            }
        )
    return rows


def build_comparison_reports(
    config: dict[str, Any],
    manifests: dict[int, dict[str, Any]],
    thresholds_by_year: dict[int, dict[str, Any]],
    exposure: pd.DataFrame,
    artifact_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    specs = ["global_qtrim", "shiptype_qtrim", "no_qtrim"]
    samples = {
        spec: {
            year: load_compact_sample(
                spec, year, config, exposure, artifact_dir
            )
            for year in config["analysis_years"]
        }
        for spec in specs
    }
    sample_rows = [
        summarize_sample(spec, year, samples[spec][year])
        for spec in specs
        for year in config["analysis_years"]
    ]
    sample_comparison = pd.DataFrame(sample_rows)
    no_qtrim_counts = (
        sample_comparison.loc[
            sample_comparison["specification"].eq("no_qtrim"),
            ["year", "voyage_rows"],
        ]
        .rename(columns={"voyage_rows": "no_qtrim_voyage_rows"})
    )
    sample_comparison = sample_comparison.merge(
        no_qtrim_counts, on="year", validate="many_to_one"
    )
    sample_comparison["removed_vs_no_qtrim_rows"] = (
        sample_comparison["no_qtrim_voyage_rows"]
        - sample_comparison["voyage_rows"]
    )
    sample_comparison.to_csv(
        report_dir / "sample_comparison.csv", index=False
    )

    panel_rows = []
    for spec in specs:
        panel_rows.extend(summarize_panel(spec, samples[spec]))
    panel_comparison = pd.DataFrame(panel_rows)
    panel_comparison.to_csv(
        report_dir / "regression_panel_comparison.csv", index=False
    )

    base_config = load_json(CODING_ROOT / config["base_rebuild_config"])
    key_rows = []
    for pair_id, shiptype_group, label in base_config["key_city_pairs"]:
        for spec in specs:
            for year in config["analysis_years"]:
                selected = samples[spec][year].loc[
                    samples[spec][year]["pair_id"].eq(pair_id)
                    & samples[spec][year]["shiptype_group"].eq(shiptype_group)
                ]
                key_rows.append(
                    {
                        "city_pair": label,
                        "pair_id": pair_id,
                        "shiptype_group": shiptype_group,
                        "specification": spec,
                        "year": year,
                        "voyage_count": int(len(selected)),
                        "vessel_count": int(selected["imo"].nunique()),
                        "median_voyage_days": (
                            float(selected["total_voyage_time_days"].median())
                            if len(selected)
                            else np.nan
                        ),
                    }
                )
    key_times = pd.DataFrame(key_rows)
    if len(key_times) != len(base_config["key_city_pairs"]) * len(specs) * 2:
        raise ValueError("Key city-pair comparison is incomplete")
    if key_times["voyage_count"].eq(0).any():
        raise ValueError("A key city-pair cell lost endpoint-year support")
    key_times.to_csv(report_dir / "key_city_pair_comparison.csv", index=False)
    pivot = key_times.pivot_table(
        index=["city_pair", "year"],
        columns="specification",
        values="median_voyage_days",
    )
    max_global_shiptype = float(
        (pivot["global_qtrim"] - pivot["shiptype_qtrim"]).abs().max()
    )
    max_global_no_trim = float(
        (pivot["global_qtrim"] - pivot["no_qtrim"]).abs().max()
    )

    global_rows = sample_comparison.loc[
        sample_comparison["specification"].eq("global_qtrim")
    ].set_index("year")
    shiptype_rows = sample_comparison.loc[
        sample_comparison["specification"].eq("shiptype_qtrim")
    ].set_index("year")
    summary = {
        "status": "pass",
        "decision_mode": config["decision_mode"],
        "global_thresholds": {
            str(year): thresholds_by_year[year]
            for year in config["analysis_years"]
        },
        "global_final_sample": {
            str(year): manifests[year]["summary"]
            for year in config["analysis_years"]
        },
        "global_minus_shiptype_voyages": {
            str(year): int(
                global_rows.loc[year, "voyage_rows"]
                - shiptype_rows.loc[year, "voyage_rows"]
            )
            for year in config["analysis_years"]
        },
        "maximum_key_median_difference_global_vs_shiptype_days": (
            max_global_shiptype
        ),
        "maximum_key_median_difference_global_vs_no_trim_days": (
            max_global_no_trim
        ),
    }
    write_json(report_dir / "summary.json", summary)
    write_json(
        report_dir / "global_qtrim_thresholds.json",
        {
            str(year): thresholds_by_year[year]
            for year in config["analysis_years"]
        },
    )

    lines = [
        "# C5.6h Global q01/q99 Sample",
        "",
        "## Decision implemented",
        "",
        "The candidate main specification fits one pooled q01/q99 pair inside",
        "each year. It does not group by ship type or OD pair. Each year's",
        "voyages therefore use that year's own distribution, and equality is",
        "retained.",
        "",
    ]
    for year in config["analysis_years"]:
        thresholds = thresholds_by_year[year]
        lines.append(
            f"- {year}: q01={thresholds['normalized_time_ratio_q01']:.9f}; "
            f"q99={thresholds['normalized_time_ratio_q99']:.9f}; "
            f"eligible rows={thresholds['eligible_rows']:,}."
        )
    lines.extend(
        [
            "",
            "## Sample comparison",
            "",
            "| Specification | Year | Voyages | Removed vs no qtrim | Supported | Positive Suez |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sample_comparison.itertuples(index=False):
        lines.append(
            f"| {row.specification} | {row.year} | {row.voyage_rows:,} | "
            f"{row.removed_vs_no_qtrim_rows:,} | "
            f"{row.exposure_supported_rows:,} | {row.positive_suez_rows:,} |"
        )
    lines.extend(
        [
            "",
            "## Stability",
            "",
            f"- Maximum absolute key-cell median difference between global and "
            f"ship-type qtrim: {max_global_shiptype:.2f} days.",
            f"- Maximum absolute key-cell median difference between global qtrim "
            f"and no qtrim: {max_global_no_trim:.2f} days.",
            "- All 15 key city-pair-by-ship-type cells retain both endpoint years.",
            "",
            "The global trim remains a statistical tail filter, not a claim that",
            "every removed voyage is physically invalid. The ship-type qtrim and",
            "no-qtrim samples remain explicit robustness comparisons.",
            "",
        ]
    )
    (report_dir / "global_qtrim_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    config = load_json(args.config)
    verify_contract(config)
    artifact_dir = args.artifact_dir.resolve()
    report_dir = args.report_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    exposure = load_exposure(config)
    thresholds_by_year = {
        year: fit_global_thresholds(
            pretrim_paths(year, config),
            config["q_lower"],
            config["q_upper"],
            year,
        )
        for year in config["analysis_years"]
    }
    manifests = build_global_samples(
        config, thresholds_by_year, exposure, artifact_dir
    )
    summary = build_comparison_reports(
        config,
        manifests,
        thresholds_by_year,
        exposure,
        artifact_dir,
        report_dir,
    )
    outputs = sorted(
        [
            report_dir / "global_qtrim_thresholds.json",
            report_dir / "sample_comparison.csv",
            report_dir / "regression_panel_comparison.csv",
            report_dir / "key_city_pair_comparison.csv",
            report_dir / "summary.json",
            report_dir / "global_qtrim_report.md",
        ]
    )
    manifest = {
        "status": "pass",
        "decision_mode": config["decision_mode"],
        "summary": summary,
        "outputs": [
            {
                "path": str(path.relative_to(CODING_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    write_json(report_dir / "c5_6h_output_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
