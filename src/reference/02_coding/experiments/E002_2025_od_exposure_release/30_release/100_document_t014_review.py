#!/usr/bin/env python3
"""Write T014's reader-first bilingual review guide from closed evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODING = Path(__file__).resolve().parents[3]
T012 = CODING / "outputs/2026-09-24_1648_E002-T012-exposure-unit-review"
HERE = Path(__file__).resolve().parent


def old_definitions() -> dict:
    spec = importlib.util.spec_from_file_location("e002_t013_docs", HERE / "60_document_t013_review.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.field_definitions()


def value_text(value: object) -> str:
    if pd.isna(value):
        return "空 / blank"
    if isinstance(value, (bool, np.bool_)):
        return "是 / yes" if value else "否 / no"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def quantile(table: pd.DataFrame, population: str, metric: str, rule: str) -> pd.Series:
    item = table.loc[table.population.eq(population) & table.metric.eq(metric)
                     & table.rule.eq(rule)]
    if len(item) != 1:
        raise ValueError(f"missing quantile {population}/{metric}/{rule}")
    return item.iloc[0]


def build(out: Path, tolerant_review: Path, gfw_run: Path) -> None:
    closure = json.loads((out / "07_independent_closure.json").read_text())
    if closure["status"] != "pass":
        raise ValueError("independent closure must pass before documentation")
    release = pd.read_csv(out / "01_release_table_voyages_ge5.csv")
    below = pd.read_csv(out / "02_below_threshold_summary.csv").iloc[0]
    clip = json.loads((out / "08_wave_clipping_summary.json").read_text())
    gfw = json.loads((out / "04_gfw_exposure_summary.json").read_text())
    gfw_position = json.loads((gfw_run / "summary.json").read_text())
    calibration = json.loads((out / "03_port_tolerance_calibration.json").read_text())
    preflight = json.loads((out / "09_resource_preflight.json").read_text())
    tq = pd.read_csv(tolerant_review / "01_voyage_error_quantiles.csv")
    zq = pd.read_csv(T012 / "01_voyage_error_quantiles.csv")
    tg = pd.read_csv(tolerant_review / "01a_wave_population_guarantee.csv")
    zg = pd.read_csv(T012 / "01a_wave_population_guarantee.csv")
    u3 = pd.read_csv(T012 / "07_u3_exposure_error_quantiles.csv")
    u3_wave_24h = u3.loc[u3.gap_hours.eq(24) & u3.metric.eq("wave_mean_absolute_error_m")]
    if len(u3_wave_24h) != 1:
        raise ValueError("U3 24-hour holdout summary missing")
    u3_wave_p90 = float(u3_wave_24h.p90.iloc[0])
    u3_wave_n = int(u3_wave_24h.windows_with_metric.iloc[0])
    sensitivity = pd.concat([
        zq.loc[zq.rule.eq("new")].assign(port_rule="zero_tolerance"),
        tq.loc[tq.rule.eq("new")].assign(port_rule="conditional_0km_1h"),
    ], ignore_index=True)
    sensitivity.drop(columns="rule").to_csv(out / "05_voyage_rule_sensitivity.csv", index=False)
    wave_guarantee = pd.concat([
        zg.loc[zg.rule.eq("new")].assign(port_rule="zero_tolerance"),
        tg.loc[tg.rule.eq("new")].assign(port_rule="conditional_0km_1h"),
    ], ignore_index=True)
    wave_guarantee.drop(columns="rule").to_csv(out / "05a_wave_population_guarantee.csv", index=False)
    all300 = quantile(tq, "all", "tc300_width_pp", "new")
    all500 = quantile(tq, "all", "tc500_width_pp", "new")
    pub300 = quantile(tq, "release_voyages_ge5", "tc300_width_pp", "new")
    pub500 = quantile(tq, "release_voyages_ge5", "tc500_width_pp", "new")
    all_wave = quantile(tg, "all", "wave_width_m", "new")
    pub_wave = quantile(tg, "release_voyages_ge5", "wave_width_m", "new")
    zero_all300 = quantile(zq, "all", "tc300_width_pp", "new")
    zero_all500 = quantile(zq, "all", "tc500_width_pp", "new")
    zero_all_wave = quantile(zg, "all", "wave_width_m", "new")
    wave_p99_text = f"{all_wave.p99:.3f} m" if pd.notna(all_wave.p99) else "无有限保证 / no finite guarantee"
    ci_wave = release.mean_swh_full_voyage_ci95_high - release.mean_swh_full_voyage_ci95_low
    ci_300 = release.tc300_at_least_pct_ci95_high - release.tc300_at_least_pct_ci95_low
    ci_500 = release.tc500_at_least_pct_ci95_high - release.tc500_at_least_pct_ci95_low
    sample = release.loc[release.ships.ge(3) & release.voyages.ge(8)
                         & release.tc500_hours_at_least.gt(0)
                         & release.tc500_at_most_pct.gt(release.tc500_at_least_pct)
                         & release.mean_swh_full_voyage_m.notna()]
    if sample.empty:
        sample = release.loc[release.ships.ge(3) & release.tc500_hours_at_least.gt(0)
                             & release.mean_swh_full_voyage_m.notna()]
    if sample.empty:
        raise ValueError("no informative real release row for example")
    example = sample.sort_values(["voyages", "origin_city", "dest_city"],
                                 ascending=[False, True, True]).iloc[0]
    reason_cn = ("可靠性依据：有观测 AIS 的留出检验、未用作本轮合格锚点的 GFW 小时对照、"
                 "港口可达性校准，以及逐航次上下限与按船抽样的闭合检查相互印证。")
    reason_en = ("Reliability rests on held-out AIS positions, GFW hours excluded from the current "
                 "eligible-anchor audit, port reachability calibration, and closed voyage bounds and vessel resampling.")
    overview = [
        ("缺失位置 / Missing positions",
         f"90% 的全部航次，TC300/TC500 占比可能范围宽度分别不超过 {all300.p90:.3f}/{all500.p90:.3f} 个百分点，平均浪高范围宽度不超过 {all_wave.p90:.3f} m。",
         f"For 90% of all voyages, TC300/TC500 share ranges are within {all300.p90:.3f}/{all500.p90:.3f} percentage points, and mean wave range within {all_wave.p90:.3f} m."),
        ("插值精度 / Interpolation accuracy",
         f"在 {u3_wave_n:,} 个有浪高结果的 24 h AIS 留出窗口中，90% 的平均浪高绝对误差不超过 {u3_wave_p90:.3f} m；90% 的 {gfw['hours']:,} 个 GFW 对照小时，格点中心距补出位置不超过 {gfw['distance_km']['p90']:.2f} km；在 {gfw['swh_absolute_difference_m']['valid_hours']:,} 个双方有浪高的小时，90% 的浪高差不超过 {gfw['swh_absolute_difference_m']['p90']:.3f} m。",
         f"Among {u3_wave_n:,} 24-hour AIS holdouts with wave data, 90% of mean wave errors are within {u3_wave_p90:.3f} m; for 90% of {gfw['hours']:,} GFW hours, distance is within {gfw['distance_km']['p90']:.2f} km; among {gfw['swh_absolute_difference_m']['valid_hours']:,} paired-wave hours, 90% of wave differences are within {gfw['swh_absolute_difference_m']['p90']:.3f} m."),
        ("港口定界 / Port bounds",
         f"在单列的 5 个端点冲突航次之外，90% 的 GFW 校准小时在 0 km 空间、1 h 时间容差内可达；实际为 {calibration['conditional_0km_1h_reachable_hours']}/{calibration['remaining_hours_if_conflict_voyages_set_unbounded']}（{calibration['conditional_0km_1h_reachable_pct']:.2f}%）。",
         f"Outside five separately flagged port-conflict voyages, 90% of GFW calibration hours are reachable with 0 km spatial and 1 h temporal tolerance; the observed rate is {calibration['conditional_0km_1h_reachable_pct']:.2f}%."),
        ("抽样 / Sampling",
         f"在 {ci_wave.notna().sum():,} 个有浪高抽样区间的发布组合中，90% 的截断后全程浪高 95% 区间宽度不超过 {ci_wave.quantile(.9):.3f} m；在 {ci_300.notna().sum():,} 个有台风区间的组合中，90% 的 TC300/TC500“至少”占比区间宽度分别不超过 {ci_300.quantile(.9):.3f}/{ci_500.quantile(.9):.3f} 个百分点。",
         f"Among {ci_wave.notna().sum():,} release groups with wave intervals, 90% of clipped-wave 95% intervals are within {ci_wave.quantile(.9):.3f} m; among {ci_300.notna().sum():,} groups with TC intervals, 90% of TC300/TC500 at-least-share intervals are within {ci_300.quantile(.9):.3f}/{ci_500.quantile(.9):.3f} percentage points."),
    ]
    overview_lines = ["| 项目 / Item | 中文：90% 在多少以内 | English: 90% within |",
                      "|---|---|---|"]
    overview_lines += [f"| {name} | {cn} | {en} |" for name, cn, en in overview]
    definitions = old_definitions()
    for name in ("mean_swh_matched_m", "mean_swh_m_ci95_low", "mean_swh_m_ci95_high"):
        del definitions[name]
    definitions["mean_swh_full_voyage_m"] = (
        "全程平均显著波高估计", "Full-voyage mean significant wave height estimate",
        "观测浪高平均值截到最低/最高可能平均值之内；m。无观测浪高时为空。",
        "Observed wave mean clipped to the lowest/highest possible mean; m. Blank without observed wave data.")
    for end, zh, en in (("low", "下端", "lower"), ("high", "上端", "upper")):
        definitions[f"mean_swh_full_voyage_ci95_{end}"] = (
            f"全程平均浪高抽样区间{zh}", f"Full-voyage mean SWH 95% {en}",
            "组合内整艘船重抽 1,000 次，每次用同一批航次重算观测均值和可能范围后截断；m；船数少于 3 时为空。",
            "1,000 whole-vessel resamples; each draw recomputes and clips the observed mean within its paired possible range; m; blank below three vessels.")
    definitions["unbounded_hours"] = (
        "仍无空间界限小时", "Still-unbounded hours",
        "无法连接可靠港口端点的未知小时，含 5 个 GFW 证实端点冲突的航次；台风最坏上限计入这些小时。",
        "Unknown hours without reliable port bounds, including five GFW-confirmed port conflicts; included in conservative TC maxima.")
    definitions["wave_unknown_counted_hours"] = (
        "浪高界限计入的未知小时", "Unknown hours counted in wave bounds",
        "25 kn、0 km/1 h 端点容差下，可达区有湿 ERA5 格点的未知小时；与观测浪高小时同作可能范围分母。",
        "Unknown hours with wet ERA5 cells in the 25-knot, 0 km/1 h reachable area; added to matched hours for possible wave means.")
    definitions["mean_swh_min_possible_m"] = (
        "全程平均浪高最低可能值", "Lowest possible full-voyage mean SWH",
        "(观测浪高积分 + 可定界未知小时的最小浪高积分) ÷ (观测浪高小时 + 可定界未知小时)；m；无界小时不伪装为 0。",
        "Observed plus reachable minimum wave integral divided by observed plus boundable unknown hours; m; unbounded hours are never set to zero.")
    definitions["mean_swh_max_possible_m"] = (
        "全程平均浪高最高可能值", "Highest possible full-voyage mean SWH",
        "同一分母，未知小时使用可达最大浪高；m。", "Same denominator, with reachable maximum wave height; m.")
    definitions["mean_swh_possible_range_m"] = (
        "全程平均浪高可能范围宽度", "Width of possible full-voyage mean SWH",
        "最高可能值减最低可能值；m。", "Highest minus lowest possible mean; m.")
    definitions["wave_mean_missing_reason"] = (
        "浪高空值原因", "Reason wave estimate is blank",
        "无观测且无可定界浪高时，点估计与上下限均空；无观测但可定界时，点估计空而上下限有值。",
        "Without observed or boundable wave hours all estimates are blank; with only boundable hours the point is blank but limits are present.")
    if set(release.columns) != set(definitions):
        raise ValueError(f"bilingual field definitions mismatch: {set(release.columns)^set(definitions)}")
    guide = ["# 数据可靠性一览 / Data reliability at a glance", "",
             *overview_lines, "", reason_cn, "", reason_en, "",
             "台风百分比的分母都是全程小时：“至少”是已观测到的暴露，“最多”把未知位置的最坏情况计入。浪高估计是观测均值截到可能范围；不把未知小时算作 0 m。",
             "", "TC shares divide by full voyage hours: at least is observed exposure; at most adds the worst case for unknown positions. The wave estimate clips the observed mean to its possible range; unknown hours are not zero.",
             "", "## 一行真实数据与逐列说明 / One real row, every field explained", "",
             f"示例组合 / Example group: `{example.origin_city}` → `{example.dest_city}`, `{example.shiptype_group}`。下表每一列的示例值均来自同一真实发布行。",
             "", "| 字段 / Field | 中文 | English | 该行数值 / This row | 中文定义、分母与单位 | English definition, denominator, and unit |",
             "|---|---|---|---:|---|---|"]
    for field in release.columns:
        zh, en, zh_meaning, en_meaning = definitions[field]
        guide.append(f"| `{field}` | {zh} | {en} | {value_text(example[field])} | {zh_meaning} | {en_meaning} |")
    guide.extend(["", "## 空值与两种不确定性 / Missingness and two kinds of uncertainty", "",
                  f"{int(clip['combinations'] - clip['with_observed_mean']):,} 个发布组合没有观测浪高均值；其中 {int((release.wave_mean_missing_reason == 'no_matched_or_reachable_wave_hours').sum()):,} 个也没有可定界的浪高小时，因此浪高一栏全空。其余无观测而有可定界小时的组合只给最低/最高可能值。",
                  "", "没有有效浪高观测的整船重抽样本不产生点估计；每组仍尝试 1,000 次，只有有效重抽进入区间分位数。",
                  "", "No observed wave mean is available in those groups. Draws without observed wave data yield no point estimate; all 1,000 resamples are attempted, and valid draws determine the percentiles.",
                  "", "台风/浪高可能范围反映缺失位置；按船重抽的 95% 区间反映组合抽样波动。这两种区间不能互换。",
                  "", "TC/wave possible ranges reflect missing locations. Whole-vessel 95% intervals reflect variation from sampling ships. They answer different questions.", ""])
    (out / "03_FIELD_GUIDE.md").write_text("\n".join(guide))
    example_fields = ["voyages", "ships", "voyage_hours", "track_hours", "unresolved_track_hours",
                      "tc300_at_least_pct", "tc300_at_most_pct", "tc500_at_least_pct",
                      "tc500_at_most_pct", "mean_swh_full_voyage_m",
                      "mean_swh_min_possible_m", "mean_swh_max_possible_m",
                      "mean_swh_full_voyage_ci95_low", "mean_swh_full_voyage_ci95_high"]
    example_table = ["| 字段 | 发布值 | 怎么读 |", "|---|---:|---|"]
    simple = {
        "voyages": "这一组包含的航次数", "ships": "参与的不同船数",
        "voyage_hours": "离港至到港的合计小时", "track_hours": "其中有位置的小时",
        "unresolved_track_hours": "仍无位置的小时", "tc300_at_least_pct": "全程中确认在台风 300 km 内的最低占比（%）",
        "tc300_at_most_pct": "计入未知位置最坏情况后的最高占比（%）",
        "tc500_at_least_pct": "全程中确认在台风 500 km 内的最低占比（%）",
        "tc500_at_most_pct": "计入未知位置最坏情况后的最高占比（%）",
        "mean_swh_full_voyage_m": "截断到可能范围的全程浪高估计（m）",
        "mean_swh_min_possible_m": "缺失位置下的最低可能平均浪高（m）",
        "mean_swh_max_possible_m": "缺失位置下的最高可能平均浪高（m）",
        "mean_swh_full_voyage_ci95_low": "按船重抽的估计值 95% 区间下端（m）",
        "mean_swh_full_voyage_ci95_high": "按船重抽的估计值 95% 区间上端（m）",
    }
    example_table += [f"| `{field}` | {value_text(example[field])} | {simple[field]} |" for field in example_fields]
    readme = ["# E002 2025 台风与浪高暴露：T014 内部评审", "",
              "先读本文件，再查 `03_FIELD_GUIDE.md` 和 `01_release_table_voyages_ge5.csv`。源分支 `experiment/E002-2025-od-exposure-release`；本轮开始时 HEAD 为 `fb6bab4`，实现代码与本评审包随同本轮提交。",
              "",
              f"**至少 90% 的全部 {1_354_168:,} 个航次，距台风中心 300/500 km 内的全程暴露占比上下限分别相差不超过 {all300.p90:.3f}/{all500.p90:.3f} 个百分点；90% 的航次平均浪高可能范围宽度不超过 {all_wave.p90:.3f} m。** 浪高这一句把无数值界限的航次保守放在最差尾部。",
              "", "台风比例这样读：‘至少’是已观测到的台风小时占全程小时的百分比；‘最多’把未知位置可能遇台风的小时也计入，同样除以全程小时。",
              "", "## 数据可靠性一览 / Data reliability at a glance", "", *overview_lines,
              "", reason_cn, "", reason_en, "",
              "## 一行真实发布数据 / One real release row", "",
              f"`{example.origin_city}` → `{example.dest_city}`，船型 `{example.shiptype_group}`。下列数值来自发布表的同一行；[字段说明](03_FIELD_GUIDE.md)给出这行每一列的中英文含义和单位。",
              "", *example_table,
              "", f"这行可直接读作：{int(example.voyages):,} 个航次中，距台风 500 km 内的全程暴露至少 {example.tc500_at_least_pct:.3f}%、最多 {example.tc500_at_most_pct:.3f}%（相差 {example.tc500_at_most_pct-example.tc500_at_least_pct:.3f} 个百分点）；全程平均浪高估计 {example.mean_swh_full_voyage_m:.3f} m，缺失位置下的可能范围为 {example.mean_swh_min_possible_m:.3f}–{example.mean_swh_max_possible_m:.3f} m，按船抽样的 95% 区间为 {example.mean_swh_full_voyage_ci95_low:.3f}–{example.mean_swh_full_voyage_ci95_high:.3f} m。",
              "", "## 口径、闭合与限制", "",
              f"发布表有 {len(release):,} 个‘出发城市 × 到达城市 × 船型’组合、{int(release.voyages.sum()):,} 个航次，每组至少 5 个航次。未达门槛的 {int(below.combinations):,} 个组合、{int(below.voyages):,} 个航次见 `02_below_threshold_summary.csv`。逐航次与坐标只在 `.runs/`。",
              "", f"浪高列名为‘全程平均浪高’：点估计是观测均值截到最低/最高可能值之内，{clip['changed_combinations']:,}/{clip['with_observed_mean']:,} 个有观测组合发生截断；发生截断者改动中位数 {clip['absolute_change_p50_changed_m']:.4f} m、90 分位 {clip['absolute_change_p90_changed_m']:.4f} m。每次按船重抽都用同一批航次同时重算观测均值与可能范围，再截断；每组 1,000 次，seed `20260923`，少于 3 艘船不报区间。",
              "", f"港口校准原始 {calibration['calibration_hours']} 小时的 12 种容差没有一组达到 99%；即使 50 km、2 h 也只有 {calibration['maximum_grid_reachable_hours']}/{calibration['calibration_hours']}（{calibration['maximum_grid_reachable_pct']:.2f}%）。剩余 24 小时来自 5 个端点冲突航次；这些航次的全部 {calibration['whole_conflict_voyage_hours']} 个 GFW 校准小时单列，其未知时段保持无界。其余 920 小时选 0 km、1 h，911/920（99.02%）可达。`03_port_tolerance_reachability.csv` 保留原始 969 小时的全部 12 组结果，`03b_port_tolerance_without_conflict_voyages.csv` 给出条件性结果。",
              "", "最小组合按‘空间容差 + 25 kn × 时间容差’对应的可达半径扩张排序；0 km、1 h 相当于从每个港口端点扩张 46.3 km：出发时间最多提前 1 h，到达时间最多延后 1 h。5 个冲突航次中，2 个按端点直线距离与记录时长需超过 25 kn，1 个疑似同名城市的港口坐标歧义，另 2 个的 GFW 位置在离港后很早就接近目的港；因此不把它们强行纳入可达港口范围。逐航次证据留在 `.runs/`。",
              "", f"零容差到条件性主规则，全部航次 90 分位的 TC300/TC500 区间宽度由 {zero_all300.p90:.3f}/{zero_all500.p90:.3f} 变为 {all300.p90:.3f}/{all500.p90:.3f} 个百分点；平均浪高宽度由 {zero_all_wave.p90:.3f} 变为 {all_wave.p90:.3f} m。发布集对应 TC300/TC500 为 {pub300.p90:.3f}/{pub500.p90:.3f} 个百分点，浪高 {pub_wave.p90:.3f} m。完整 p50/p75/p90/p95/p99 及有效分母见 `05_voyage_rule_sensitivity.csv` 和 `05a_wave_population_guarantee.csv`。",
              "", "容差有时也会让台风上限降低：零容差下若港口端点要求不可能的速度，原规则把整段保守列为可能暴露；放宽到可计算的可达区后，可排除其中一些台风位置。",
              "", f"尾部也保留：全部航次 TC300/TC500 宽度 p95 为 {all300.p95:.3f}/{all500.p95:.3f} 个百分点，p99 为 {all300.p99:.3f}/{all500.p99:.3f} 个百分点；浪高宽度 p95 为 {all_wave.p95:.3f} m，p99 为 {wave_p99_text}。少数航次的区间仍较宽。",
              "", f"GFW 对照中浪高有效 {gfw['swh_absolute_difference_m']['valid_hours']:,}/{gfw['hours']:,} 小时，绝对差 p95 {gfw['swh_absolute_difference_m']['p95']:.3f} m、p99 {gfw['swh_absolute_difference_m']['p99']:.3f} m；TC300/TC500 判定总体翻转 {gfw['tc300']['flip_hours']:,}/{gfw['tc500']['flip_hours']:,} 小时，占全部对照小时 {gfw['tc300']['flip_pct_all']:.4f}%/{gfw['tc500']['flip_pct_all']:.4f}%，在至少一方有台风的小时中分别为 {gfw['tc300']['flip_pct_either_positive']:.2f}%/{gfw['tc500']['flip_pct_either_positive']:.2f}%。GFW 来自与补轨锚点相同的数据源，格点中心不等于精确船位，且比较使用小时内最近的十分钟点；这是外部留出诊断，不是完全独立的真值。",
              "", f"90 分位的 GFW 浪高差为 {gfw['swh_absolute_difference_m']['p90']:.3f} m，主要因为两处位置常落入同一 ERA5 浪高格点；位置距离的 90 分位为 {gfw['distance_km']['p90']:.2f} km。",
              "", f"GFW 对照先排除与本项目合格锚点审计重合的 {gfw_position['source_counts']['excluded_e001_admissible_hour_audit_overlap']:,} 个小时；在本项目航次面板内留下 {gfw_position['source_counts']['heldout_in_e001_panel']:,} 小时，其中 {gfw['hours']:,} 小时是完整的 GFW 支持插值。本仓原始响应副本的 summary 哈希与另一项目一致；只有本仓不存在的逐小时派生表从另一项目只读取得，路径与内容哈希留在 `.runs` 清单和 `06_closure.json`。",
              "", f"资源 probe 的保守全量估计 {preflight['estimated_full_wall_hours']:.1f} h，低于 24 h 停止线。`06_closure.json` 与 `07_independent_closure.json` 记录航次/组合/小时、台风小时拆分、波高截断、重抽种子、P2 校准与 GFW 对照闭合；原始规则仅作对照。",
              "", "仍无可靠港口端点、速度不可行或无湿 ERA5 格点的时段保持未知。浪高可能范围只对有观测或可定界的小时给数值，未知小时不补零；无界航次在全样本 90 分位保证中放最差尾部。按船抽样区间和缺失位置可能范围反映不同不确定性。IBTrACS 仍是冻结的 provisional 快照。",
              "", "这是 E002 分支内部草稿，待 Supreme 审查及用户 UG2 验收；正式产品未改。未发 API 请求、未下载新数据、未做 2024、未合并、未 push、未上传。", ""]
    (out / "00_README.md").write_text("\n".join(readme))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerant-review", type=Path, required=True)
    parser.add_argument("--gfw-run", type=Path, required=True)
    args = parser.parse_args()
    build(args.out, args.tolerant_review, args.gfw_run)
