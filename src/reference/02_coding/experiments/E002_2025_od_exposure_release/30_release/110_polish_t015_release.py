#!/usr/bin/env python3
"""Polish the closed T014 release for T015 without recomputing bounds or intervals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[3]
T014 = CODING / "outputs/2026-09-24_2218_E002-T014-reader-first-review"
T012 = CODING / "outputs/2026-09-24_1648_E002-T012-exposure-unit-review"
U3 = CODING / ".runs/2026-09-24_0135_E002-2025-od-exposure-release/20_u3_full"
WEATHER_CORE = CODING / "pipeline/40_construct_variables/60_construct_voyage_weather_facts"
EXAMPLE_KEY = ("CHN:SHANGHAI", "CHN:NINGBO", "container")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def precision(field: str) -> int | None:
    if field.startswith("share_wave_hours_ge4m"):
        return 3
    if "_hours" in field:
        return 1
    if field.startswith("mean_swh_full_voyage_ci95_"):
        return 3
    if field.endswith(("_pct", "_pct_ci95_low", "_pct_ci95_high", "_m")):
        return 3
    if "coverage" in field:
        return 3
    return None


def fixed(value: str, places: int) -> str:
    if value == "":
        return ""
    rendered = f"{float(value):.{places}f}"
    return rendered.replace("-0." + "0" * places, "0." + "0" * places)


def rounded_release(source: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as original, target.open("w", newline="") as published:
        reader = csv.DictReader(original)
        fields = reader.fieldnames
        if fields is None or len(fields) != 43:
            raise ValueError("T014 release schema changed")
        writer = csv.DictWriter(published, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        rows = 0
        example = None
        for row in reader:
            values = {field: fixed(row[field], places) if (places := precision(field)) is not None
                      else row[field] for field in fields}
            writer.writerow(values)
            rows += 1
            if tuple(row[key] for key in ("origin_city", "dest_city", "shiptype_group")) == EXAMPLE_KEY:
                example = values
    if rows != 46_981 or example is None:
        raise ValueError("T014 release row count or example changed")
    return {"rows": rows, "example": example, "fields": fields}


def holdout_digest() -> pd.DataFrame:
    sys.path.insert(0, str(WEATHER_CORE))
    from weather_core.geo import great_circle_km

    outcomes = pd.read_csv(U3 / "holdout_sample_outcomes.csv", usecols=[
        "sample_id", "nominal_gap_hours", "status", "paired_prediction_points"
    ])
    complete = outcomes.loc[outcomes.status.eq("predicted") & outcomes.nominal_gap_hours.isin([4, 24])]
    if complete.groupby("nominal_gap_hours").size().to_dict() != {4: 4601, 24: 4402}:
        raise ValueError("U3 complete holdout population changed")
    points = pd.read_parquet(U3 / "holdout_point_results.parquet", columns=[
        "sample_id", "nominal_gap_hours", "lat_truth", "lon_truth", "lat_pred", "lon_pred"
    ])
    points = points.loc[points.sample_id.isin(set(complete.sample_id))].copy()
    counts = points.groupby("sample_id").size()
    expected = complete.set_index("sample_id").paired_prediction_points
    if not counts.reindex(expected.index).eq(expected).all():
        raise ValueError("U3 paired point inventory changed")
    if points[["lat_truth", "lon_truth", "lat_pred", "lon_pred"]].isna().any().any():
        raise ValueError("complete U3 position holdout gained nulls")
    points["error_km"] = great_circle_km(
        points.lon_truth.to_numpy(), points.lat_truth.to_numpy(),
        points.lon_pred.to_numpy(), points.lat_pred.to_numpy())
    wave = pd.read_csv(T012 / "07_u3_exposure_error_quantiles.csv")
    records = []
    for hours, group in points.groupby("nominal_gap_hours", sort=True):
        source = wave.loc[wave.gap_hours.eq(hours) & wave.metric.eq("wave_mean_absolute_error_m")]
        if len(source) != 1 or int(source.windows_complete.iloc[0]) != int(complete.nominal_gap_hours.eq(hours).sum()):
            raise ValueError("U3 position/wave window populations changed")
        records.append({
            "gap_hours": int(hours), "complete_windows": int(group.sample_id.nunique()),
            "position_points": int(len(group)),
            "position_error_p90_km": float(group.error_km.quantile(.9)),
            "wave_windows_with_metric": int(source.windows_with_metric.iloc[0]),
            "wave_mean_absolute_error_p90_m": float(source.p90.iloc[0]),
        })
    return pd.DataFrame(records)


def copy_evidence(out: Path) -> None:
    files = ["02_below_threshold_summary.csv", "03_port_tolerance_calibration.json",
             "03_port_tolerance_reachability.csv", "03b_port_tolerance_without_conflict_voyages.csv",
             "04_gfw_exposure_by_month.csv", "04_gfw_exposure_summary.json",
             "05_voyage_rule_sensitivity.csv", "05a_wave_population_guarantee.csv",
             "08_wave_clipping_summary.json", "09_resource_preflight.json"]
    for name in files:
        shutil.copy2(T014 / name, out / name)


def overview(holdout: pd.DataFrame, release: pd.DataFrame, wave_population: pd.DataFrame,
             gfw: dict) -> list[tuple[str, str, str]]:
    h4 = holdout.loc[holdout.gap_hours.eq(4)].iloc[0]
    h24 = holdout.loc[holdout.gap_hours.eq(24)].iloc[0]
    wave = wave_population.loc[wave_population.population.eq("all")
                                & wave_population.port_rule.eq("conditional_0km_1h")].iloc[0]
    ci300 = (release.tc300_at_least_pct_ci95_high - release.tc300_at_least_pct_ci95_low).quantile(.9)
    ci500 = (release.tc500_at_least_pct_ci95_high - release.tc500_at_least_pct_ci95_low).quantile(.9)
    ciwave = (release.mean_swh_full_voyage_ci95_high - release.mean_swh_full_voyage_ci95_low).quantile(.9)
    if not (wave.p90 < 1 and ci300 > 0 and ci500 > 0 and ciwave > 0):
        raise ValueError("T015 headline inputs changed")
    return [
        ("缺失位置 / Missing positions",
         f"至少 90% 的航次，TC300 和 TC500 台风暴露上下限完全相同。90% 的航次，全程平均浪高的可能范围在 {wave.p90:.3f} m 以内。",
         f"For at least 90% of voyages, the TC300 and TC500 exposure limits coincide exactly. For 90% of voyages, the possible full-voyage mean wave-height range is within {wave.p90:.3f} m."),
        ("补出位置的精度 / Reconstructed positions",
         f"挖空真实 AIS 再重补，4 h 缺口的 90% 位置误差在 {h4.position_error_p90_km:.1f} km、平均浪高误差在 {h4.wave_mean_absolute_error_p90_m:.3f} m 以内；24 h 缺口分别为 {h24.position_error_p90_km:.1f} km、{h24.wave_mean_absolute_error_p90_m:.3f} m。GFW 对照中 90% 的位置相距约 7 km，浪高几乎没有差别、台风判定几乎不变；GFW 不是完全独立的真值。",
         f"After hiding real AIS positions and reconstructing them, 90% of 4 h gaps have position error within {h4.position_error_p90_km:.1f} km and mean wave error within {h4.wave_mean_absolute_error_p90_m:.3f} m; for 24 h gaps the figures are {h24.position_error_p90_km:.1f} km and {h24.wave_mean_absolute_error_p90_m:.3f} m. In the GFW comparison, 90% of positions are about 7 km apart, wave heights barely differ, and cyclone classifications almost never change; GFW is not a fully independent reference."),
        ("港口定界 / Port bounds",
         "117 个可用独立位置核对的无位置航次中，112 个吻合，其余 5 个的港口或时刻记录有误。可能受影响的约占全部航次的 0.2%。",
         "Of 117 positionless voyages checked against separate position records, 112 agree; the other five have conflicting port or time records. Voyages potentially affected account for about 0.2% of all voyages."),
        ("抽样 / Sampling",
         f"台风是稀有事件，主要不确定性来自抽样而非缺失位置：90% 的发布组合，TC300/TC500 的 95% 抽样区间宽度分别不超过 {ci300:.3f}/{ci500:.3f} 个百分点；看台风请看区间。90% 的有浪高区间组合，全程平均浪高的 95% 抽样区间宽度不超过 {ciwave:.3f} m。",
         f"Cyclones are rare, so sampling drives more uncertainty than missing positions: 90% of release groups have TC300/TC500 95% interval widths within {ci300:.3f}/{ci500:.3f} percentage points; read the interval for cyclone exposure. For 90% of groups with a wave interval, its 95% width is within {ciwave:.3f} m."),
    ]


def build_docs(out: Path, private: Path, example: dict, holdout: pd.DataFrame) -> None:
    full = pd.read_csv(private / "01_release_full_precision.csv")
    wave_population = pd.read_csv(out / "05a_wave_population_guarantee.csv")
    gfw = json.loads((out / "04_gfw_exposure_summary.json").read_text())
    rows = overview(holdout, full, wave_population, gfw)
    table = ["| 项目 / Item | 中文结论 | English conclusion |", "|---|---|---|"]
    table += [f"| {name} | {zh} | {en} |" for name, zh, en in rows]
    reason_zh = "可靠性依据是挖空真实 AIS 的留出检验、未作本轮锚点的 GFW 对照、港口位置核对，以及逐航次界限和按船抽样的闭合检查。"
    reason_en = "Reliability is supported by withheld AIS tests, GFW hours not used as current anchors, port-position checks, and closed voyage bounds and vessel resampling."
    example_fields = ["voyages", "ships", "voyage_hours", "position_coverage",
                      "tc300_at_least_pct", "tc300_at_most_pct", "tc300_at_least_pct_ci95_low",
                      "tc300_at_least_pct_ci95_high", "tc500_at_least_pct", "tc500_at_most_pct",
                      "tc500_at_least_pct_ci95_low", "tc500_at_least_pct_ci95_high",
                      "mean_swh_full_voyage_m", "mean_swh_min_possible_m", "mean_swh_max_possible_m",
                      "mean_swh_full_voyage_ci95_low", "mean_swh_full_voyage_ci95_high"]
    descriptions = {
        "voyages": "航次数 / voyages", "ships": "不同船数 / vessels",
        "voyage_hours": "离港至到港的合计小时 / full voyage hours",
        "position_coverage": "有位置小时占全程小时，0–1 / positioned-hour share, 0–1",
        "tc300_at_least_pct": "已确认的 TC300 全程占比 / confirmed TC300 share",
        "tc300_at_most_pct": "TC300 最坏情况下的全程占比 / TC300 worst-case share",
        "tc300_at_least_pct_ci95_low": "TC300 抽样区间下端 / sampling lower limit",
        "tc300_at_least_pct_ci95_high": "TC300 抽样区间上端 / sampling upper limit",
        "tc500_at_least_pct": "已确认的 TC500 全程占比 / confirmed TC500 share",
        "tc500_at_most_pct": "TC500 最坏情况下的全程占比 / TC500 worst-case share",
        "tc500_at_least_pct_ci95_low": "TC500 抽样区间下端 / sampling lower limit",
        "tc500_at_least_pct_ci95_high": "TC500 抽样区间上端 / sampling upper limit",
        "mean_swh_full_voyage_m": "截到可能范围内的浪高估计 / clipped wave estimate",
        "mean_swh_min_possible_m": "最低可能平均浪高 / possible minimum",
        "mean_swh_max_possible_m": "最高可能平均浪高 / possible maximum",
        "mean_swh_full_voyage_ci95_low": "浪高抽样区间下端 / sampling lower limit",
        "mean_swh_full_voyage_ci95_high": "浪高抽样区间上端 / sampling upper limit",
    }
    example_table = ["| 字段 / Field | 这行发布值 / Published value | 怎么读 / Meaning |",
                     "|---|---:|---|"]
    example_table += [f"| `{field}` | {example[field]} | {descriptions[field]} |" for field in example_fields]
    t014_guide = (T014 / "03_FIELD_GUIDE.md").read_text()
    tail = "## 一行真实数据与逐列说明" + t014_guide.split("## 一行真实数据与逐列说明", 1)[1]
    guide_lines = []
    replaced = set()
    for line in tail.splitlines():
        if line.startswith("| `"):
            cells = line.split("|")
            field = cells[1].strip().strip("`")
            if field in example:
                cells[4] = " " + (example[field] or "空 / blank") + " "
                line = "|".join(cells)
                replaced.add(field)
        if line.startswith("| `wave_unmatched_hours` "):
            line = line.replace("有位置但没有浪高匹配的时长；小时。",
                                "全程小时减浪高匹配小时；包括位置未知及有位置但未匹配浪高的时段；小时。")
            line = line.replace("Positioned hours without a wave-height match.",
                                "Full voyage hours minus wave-matched hours, including unknown-position time and positioned time without a wave match.")
        guide_lines.append(line)
    if replaced != set(example):
        raise ValueError("T015 field guide example does not cover all 43 release columns")
    guide = ["# 数据可靠性一览 / Data reliability at a glance", "", *table, "",
             reason_zh, "", reason_en, "",
             "台风百分比以全程小时为分母：‘至少’是有位置时确认遇到台风，‘最多’计入未知位置的最坏情况；读台风请同时看按船抽样的区间。",
             "", "Cyclone percentages divide by all voyage hours: at least is confirmed exposure and at most includes the location-unknown worst case; also read the vessel-sampling interval.",
             "", "GFW 对照有两点局限：两边同为 AIS 来源；留出的小时多紧邻已有锚点，所以比较偏容易。本机没有 2025 年卫星数据，因此没有做卫星检验。GFW 位置距离第 90 百分位是 7.15 km；两处位置常落在同一 ERA5 浪高格点，浪高几乎没有差别。",
             "", "The GFW check has two limits: both sides draw from AIS, and many held-out hours are close to existing anchors. No 2025 satellite data were available locally, so no satellite test was run. The GFW position-distance 90th percentile is 7.15 km; wave heights barely differ because both positions often occupy the same ERA5 wave cell.",
             "", "港口一行的约 0.2% 是潜在影响的粗略外推：把核对样本中的 5/117 端点冲突率用于原规则下 64,660 个无位置航次，再除以全部 1,354,168 个航次；它不是逐航次确认的错误比例。",
             "", "The roughly 0.2% potentially affected is a rough extrapolation: apply the 5/117 port-conflict rate to 64,660 previously positionless voyages, then divide by all 1,354,168 voyages. It is not a verified error rate for individual voyages.",
             "", "发布表的百分比、米和覆盖率保留三位小数，小时保留一位；`share_wave_hours_ge4m` 及其区间是 0–1 比例，也保留三位。每列先用全精度算好，再分别取整，因此发布值相加或相减可能出现末位差异；全精度版在 `.runs/`，闭合检查也用它。",
             "", "Percentages, metres, and coverage are displayed to three decimals; hours to one decimal. The ≥4 m wave shares remain 0–1 ratios, displayed to three decimals. Columns are rounded separately after calculation, so last-digit arithmetic may differ; closure uses the private full-precision copy.",
             "", *guide_lines, ""]
    (out / "03_FIELD_GUIDE.md").write_text("\n".join(guide))
    clip = json.loads((out / "08_wave_clipping_summary.json").read_text())
    calibration = json.loads((out / "03_port_tolerance_calibration.json").read_text())
    sensitivity = pd.read_csv(out / "05a_wave_population_guarantee.csv")
    wave = sensitivity.loc[sensitivity.population.eq("all")
                           & sensitivity.port_rule.eq("conditional_0km_1h")].iloc[0]
    readme = ["# 数据可靠性一览 / Data reliability at a glance", "", *table, "",
              reason_zh, "", reason_en, "",
              "## 一行真实发布数据 / One real release row", "",
              "`CHN:SHANGHAI` → `CHN:NINGBO`，`container`。这行来自取整后的发布表；[字段说明](03_FIELD_GUIDE.md)逐列给出定义、分母和单位。",
              "", *example_table, "",
              f"这行可读作：{example['voyages']} 个航次，TC500 全程暴露至少 {example['tc500_at_least_pct']}%、最多 {example['tc500_at_most_pct']}%；按船抽样的‘至少’占比 95% 区间为 {example['tc500_at_least_pct_ci95_low']}–{example['tc500_at_least_pct_ci95_high']}%。全程平均浪高估计 {example['mean_swh_full_voyage_m']} m，可能范围 {example['mean_swh_min_possible_m']}–{example['mean_swh_max_possible_m']} m，抽样区间 {example['mean_swh_full_voyage_ci95_low']}–{example['mean_swh_full_voyage_ci95_high']} m。",
              "", "## 口径与交付 / Definitions and delivery", "",
              "这是 E002 的 2025 年 OD × 船型聚合数据内部评审包。先读本页，再看 `03_FIELD_GUIDE.md` 和 `01_release_table_voyages_ge5.csv`。源分支 `experiment/E002-2025-od-exposure-release`，基于 `8033ff7` 的计算结果与 `079d2e3` 的 T015 决定；待 Supreme 审查和用户 UG2 验收。",
              "", "‘至少’台风占比 = 已确认台风小时 ÷ 全程小时；‘最多’还计入未知位置可能暴露的小时。浪高点估计是观测均值截到最低/最高可能平均值之内；缺失小时不当作 0 m。缺失位置的可能范围与按船抽样的 95% 区间回答不同问题。",
              "", f"发布表保留 {len(full):,} 个至少 5 航次的组合、{int(full.voyages.sum()):,} 个航次；未达门槛的组合与航次见 `02_below_threshold_summary.csv`。{int(clip['combinations'] - clip['with_observed_mean']):,} 个发布组合没有观测浪高均值，其中 {int((full.wave_mean_missing_reason == 'no_matched_or_reachable_wave_hours').sum()):,} 个连可定界浪高也没有，浪高栏为空。",
              "", "发布表只在展示时取整：百分比、米和覆盖率三位小数，小时一位小数；≥4 m 浪高占比仍按 0–1 比例显示三位小数。全精度版本保存在 `.runs/`，所有闭合检查基于全精度；取整后各列重新相加可能差一个末位。",
              "", "## 核对依据与限制 / Evidence and limits", "",
              f"4 h 和 24 h 留出结果分别来自 {int(holdout.loc[holdout.gap_hours.eq(4), 'complete_windows'].iloc[0]):,}、{int(holdout.loc[holdout.gap_hours.eq(24), 'complete_windows'].iloc[0]):,} 个完整窗口；位置误差按留出点计算，平均浪高误差按窗口计算。原始汇总见 `04_u3_holdout_precision.csv`。",
              "", f"GFW 位置距离的实际第 90 百分位是 {gfw['distance_km']['p90']:.2f} km；TC300/TC500 判定只在 {gfw['tc300']['flip_hours']:,}/{gfw['tc500']['flip_hours']:,} 个对照小时翻转，占 {gfw['tc300']['flip_pct_all']:.4f}%/{gfw['tc500']['flip_pct_all']:.4f}%。浪高在 90% 的双方有效小时几乎没有差别，因为常落在同一 ERA5 格点。GFW 与补轨同为 AIS 来源，且留出小时多紧邻已有锚点，所以它是一致性核对，不是完全独立的真值；本机没有 2025 年卫星数据，因此没有做卫星检验。",
              "", f"港口端点采用 25 kn、0 km/1 h 的条件性规则；5 个端点冲突航次保持无界。原始 969 个 GFW 校准小时中，最大容差组合可达 {calibration['maximum_grid_reachable_hours']} 小时；单列冲突航次后，其余 920 小时中 911 小时可达。首页约 0.2% 的潜在影响是把 5/117 的冲突比例粗略外推到原来 64,660 个无位置航次，再除以全部 1,354,168 个航次，并非逐航次确认的错误率。原始和条件性全部 12 组结果分别见 `03_port_tolerance_reachability.csv` 和 `03b_port_tolerance_without_conflict_voyages.csv`。",
              "", f"浪高点估计在 {clip['changed_combinations']:,}/{clip['with_observed_mean']:,} 个有观测组合中因截断而改变；发生改变者绝对改动的中位数为 {clip['absolute_change_p50_changed_m']:.4f} m。完整的新旧容差敏感性见 `05_voyage_rule_sensitivity.csv` 和 `05a_wave_population_guarantee.csv`；后者把无浪高数值界限的航次放在最差尾部，全部航次的浪高宽度第 90 百分位为 {wave.p90:.3f} m。",
              "", "`06_closure.json` 记录本轮对全精度来源、取整规则和现成数值关系的闭合；`07_independent_closure.json` 保留 T014 原独立核查。逐航次、坐标与全精度聚合数据只留在 `.runs/`。正式产品未改，未做 2024、未请求 API、未下载数据；未合并、未 push、未上传。", ""]
    (out / "00_README.md").write_text("\n".join(readme))


def main(out: Path, private: Path) -> None:
    if "outputs" not in out.parts or ".runs" not in private.parts:
        raise ValueError("public review must be under outputs and precision copy under .runs")
    out.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    source = T014 / "01_release_table_voyages_ge5.csv"
    shutil.copy2(source, private / "01_release_full_precision.csv")
    if digest(source) != digest(private / "01_release_full_precision.csv"):
        raise ValueError("private full-precision source copy differs")
    result = rounded_release(private / "01_release_full_precision.csv", out / "01_release_table_voyages_ge5.csv")
    copy_evidence(out)
    holdout = holdout_digest()
    holdout.to_csv(out / "04_u3_holdout_precision.csv", index=False)
    build_docs(out, private, result["example"], holdout)
    (private / "t015_source_manifest.json").write_text(json.dumps({
        "t014_release_sha256": digest(source),
        "u3_point_results_sha256": digest(U3 / "holdout_point_results.parquet"),
        "u3_outcomes_sha256": digest(U3 / "holdout_sample_outcomes.csv"),
        "t012_wave_quantiles_sha256": digest(T012 / "07_u3_exposure_error_quantiles.csv"),
        "rounded_release_sha256": digest(out / "01_release_table_voyages_ge5.csv"),
        "full_precision_copy": str(private / "01_release_full_precision.csv"),
    }, indent=2) + "\n")
    print(json.dumps({"release_rows": result["rows"],
                      "holdout": holdout.to_dict(orient="records"),
                      "rounding_only": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    main(args.out, args.private)
