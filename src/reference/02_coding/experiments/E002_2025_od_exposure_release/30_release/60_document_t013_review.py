#!/usr/bin/env python3
"""Write the T013 bilingual field guide and plain-language review summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def field_definitions() -> dict[str, tuple[str, str, str, str]]:
    fields = {
        "origin_city": ("出发城市", "Origin city", "有向航线的起点城市代码。", "Origin city code in the directed pair."),
        "dest_city": ("到达城市", "Destination city", "有向航线的终点城市代码。", "Destination city code in the directed pair."),
        "shiptype_group": ("船型组", "Ship type group", "正式产品的船型分类。", "Ship type category in the adopted product."),
        "voyages": ("航次数", "Voyages", "该组合的航次数；发布表仅保留至少 5 次。", "Number of voyages; the release includes groups with at least five."),
        "ships": ("船数", "Ships", "按稳定船舶标识去重；不公开标识本身。", "Distinct vessels by the stable vessel key; the key is not released."),
        "voyage_hours": ("全程小时", "Full voyage hours", "组合内航次从离港到到港的总时长；小时。", "Sum of departure-to-arrival duration; hours."),
        "track_hours": ("有位置小时", "Positioned hours", "成品轨迹有位置的时长；小时。", "Hours with a position in the adopted track product."),
        "position_coverage": ("位置覆盖率", "Position coverage", "有位置小时 ÷ 全程小时；0–1。", "Positioned hours divided by full voyage hours; 0–1."),
        "unresolved_track_hours": ("位置未知小时", "Unresolved position hours", "成品轨迹没有位置的时长；小时；不补零。", "Hours without a resolved position; never filled with zero."),
        "wave_matched_hours": ("浪高匹配小时", "Wave-matched hours", "已知位置并匹配 ERA5 总显著波高的时长；小时。", "Hours with a positioned ERA5 total significant-wave-height match."),
        "wave_unmatched_hours": ("浪高未匹配小时", "Wave-unmatched hours", "有位置但没有浪高匹配的时长；小时。", "Positioned hours without a wave-height match."),
        "wave_ge4m_matched_hours": ("浪高至少 4 m 小时", "Matched hours with wave height ≥4 m", "浪高匹配小时中显著波高至少 4 m 的时长。", "Matched hours with significant wave height at least 4 m."),
        "wave_coverage": ("浪高覆盖率", "Wave coverage", "浪高匹配小时 ÷ 全程小时；0–1。", "Wave-matched hours divided by full voyage hours; 0–1."),
        "mean_swh_matched_m": ("观测平均显著波高", "Observed mean significant wave height", "已匹配浪高的小时积分 ÷ 浪高匹配小时；m。没有匹配小时则为空。", "Wave-height hour integral divided by matched wave hours; m. Blank without matched hours."),
        "mean_swh_m_ci95_low": ("观测平均浪高抽样区间下端", "Observed mean SWH 95% lower", "按船整簇重抽的 95% 区间；m；少于 3 艘船或无匹配浪高时为空。", "Whole-vessel cluster bootstrap 95% lower limit; m; blank with fewer than three vessels or no matched wave data."),
        "mean_swh_m_ci95_high": ("观测平均浪高抽样区间上端", "Observed mean SWH 95% upper", "定义与下端相同；m。", "Upper limit for the same bootstrap interval; m."),
        "share_wave_hours_ge4m": ("至少 4 m 浪高占比", "Share of matched wave hours ≥4 m", "至少 4 m 的匹配小时 ÷ 全部浪高匹配小时；0–1。", "Matched hours at least 4 m divided by all wave-matched hours; 0–1."),
        "share_wave_hours_ge4m_ci95_low": ("至少 4 m 占比抽样区间下端", "Share ≥4 m 95% lower", "按船整簇重抽的 95% 区间；0–1；少于 3 艘船或无匹配浪高时为空。", "Whole-vessel cluster bootstrap 95% lower limit; 0–1; blank with fewer than three vessels or no matched wave data."),
        "share_wave_hours_ge4m_ci95_high": ("至少 4 m 占比抽样区间上端", "Share ≥4 m 95% upper", "定义与下端相同；0–1。", "Upper limit for the same bootstrap interval; 0–1."),
        "voyages_with_tc300": ("观测到 300 km 台风的航次数", "Voyages with observed TC300", "至少一个有位置小时落在台风 300 km 范围内的航次数。", "Voyages with at least one positioned hour within 300 km of a tropical cyclone."),
        "voyages_with_tc500": ("观测到 500 km 台风的航次数", "Voyages with observed TC500", "至少一个有位置小时落在台风 500 km 范围内的航次数。", "Voyages with at least one positioned hour within 500 km of a tropical cyclone."),
        "single_ship": ("单船标记", "Single-vessel flag", "该组合只有一艘船。", "True when the group contains one vessel."),
        "sampling_interval_unavailable_fewer_than_3_ships": ("抽样区间不可用", "Sampling interval unavailable", "船数少于 3 时为真；所有按船抽样区间为空。", "True with fewer than three vessels; all vessel-bootstrap intervals are blank."),
        "unbounded_hours": ("仍无空间界限小时", "Still-unbounded hours", "港口端点仍无法完整连接的未知小时；计入台风最坏情况上限。", "Unknown hours without complete port endpoints; included in the conservative TC upper bound."),
        "wave_unknown_counted_hours": ("浪高界限计入的未知小时", "Unknown hours counted in wave bounds", "25 kn 可达区有可用湿 ERA5 格点的未知小时；与匹配小时一同构成浪高界限分母。", "Unknown hours with eligible wet ERA5 cells in the 25-knot reachable area; added to matched hours in wave-bound denominators."),
        "mean_swh_min_possible_m": ("最低可能平均显著波高", "Lowest possible mean significant wave height", "(已匹配浪高积分 + 未知时段可达最小浪高积分) ÷ (匹配小时 + 计入的未知小时)；m。", "Matched wave integral plus reachable minimum integral, divided by matched plus counted unknown hours; m."),
        "mean_swh_max_possible_m": ("最高可能平均显著波高", "Highest possible mean significant wave height", "同一分母，未知时段改用可达最大浪高；m。", "Same denominator, using reachable maximum wave height during unknown hours; m."),
        "mean_swh_possible_range_m": ("平均浪高可能范围宽度", "Width of possible mean wave height", "最高可能值 − 最低可能值；m。", "Highest minus lowest possible mean; m."),
        "wave_mean_missing_reason": ("平均浪高缺失原因", "Reason wave mean is blank", "`no_matched_or_reachable_wave_hours`：匹配和可定界时长均为 0，平均值及上下限都空；`no_matched_wave_hours_reachable_bounds_available`：观测均值空，但有可达上下限。", "No matched or reachable wave hours: all wave estimates blank. No matched wave hours but reachable bounds available: observed mean blank while possible limits exist."),
    }
    for radius in (300, 500):
        name = f"tc{radius}"
        fields[f"{name}_hours_at_least"] = (f"{radius} km 台风至少小时", f"TC{radius} hours at least", "有位置时观测到的台风暴露小时；小时。", "Observed TC exposure in positioned hours; hours.")
        fields[f"{name}_bounded_possible_hours"] = (f"{radius} km 台风有界可能小时", f"TC{radius} bounded possible hours", "未知位置时可达范围可能遇到台风的小时；含速度不可行段的保守回退。", "Unknown hours possibly exposed within the reachable set, including conservative fallback for speed conflicts.")
        fields[f"{name}_hours_at_most"] = (f"{radius} km 台风最多小时", f"TC{radius} hours at most", "至少小时 + 有界可能小时 + 仍无空间界限小时；小时。", "At-least plus bounded-possible plus still-unbounded hours; hours.")
        fields[f"{name}_at_least_pct"] = (f"{radius} km 台风至少占比", f"TC{radius} exposure at least", "至少小时 ÷ 全程小时 ×100；数值为百分比（%）。", "At-least hours divided by full voyage hours times 100; percent (%).")
        fields[f"{name}_at_most_pct"] = (f"{radius} km 台风最多占比", f"TC{radius} exposure at most", "最多小时 ÷ 全程小时 ×100；未知位置时可能遇台风的最坏情况；%。", "At-most hours divided by full voyage hours times 100; worst case for location-unknown time; %.")
        fields[f"{name}_at_least_pct_ci95_low"] = (f"{radius} km 至少占比抽样区间下端", f"TC{radius} at-least share 95% lower", "组合内按船整簇重抽 1,000 次的 95% 下端；%；船数少于 3 为空。", "Lower 95% percentile limit from 1,000 whole-vessel resamples within the group; %; blank with fewer than three vessels.")
        fields[f"{name}_at_least_pct_ci95_high"] = (f"{radius} km 至少占比抽样区间上端", f"TC{radius} at-least share 95% upper", "同一抽样区间的上端；%。", "Upper limit of the same bootstrap interval; %.")
    return fields


def build(out: Path) -> None:
    report = json.loads((out / "06_closure.json").read_text())
    verified = json.loads((out / "07_independent_closure.json").read_text())
    if verified["status"] != "pass":
        raise ValueError("independent closure must pass before documentation")
    release = pd.read_csv(out / "01_release_table_voyages_ge5.csv", nrows=0)
    definitions = field_definitions()
    missing = [column for column in release.columns if column not in definitions]
    if missing:
        raise ValueError(f"fields missing bilingual definitions: {missing}")
    guide = ["# 发布表字段说明 / Release field guide", "",
             "台风占比字段以全程小时为分母，数值单位为 %（例如 2 表示 2%）；两列相减的宽度单位是百分点。浪高均值的分母为有浪高数据的小时。",
             "", "TC shares use full voyage hours as the denominator and are recorded as percentages (2 means 2%). The difference between two shares is measured in percentage points. Mean wave height uses hours with wave data, not full voyage hours.", "",
             "| Field | 中文 | English | 中文定义、分母与单位 | English definition, denominator, and unit |",
             "|---|---|---|---|---|"]
    for column in release.columns:
        chinese, english, cn_meaning, en_meaning = definitions[column]
        guide.append(f"| `{column}` | {chinese} | {english} | {cn_meaning} | {en_meaning} |")
    guide.extend(["", "**空值与两类区间 / Missingness and two kinds of intervals**", "",
                  "515 个组合既无已匹配浪高小时，也无可用于定界的未知小时；观测平均浪高及最低/最高可能值均为空。另有 408 个组合没有已匹配浪高小时，因此观测均值为空，但有可达范围内的浪高上下限。未知不补零。",
                  "", "In 515 groups, neither matched nor boundable unknown wave hours exist, so the observed mean and possible limits are blank. Another 408 groups have no matched wave hours, leaving the observed mean blank, while reachable wave bounds are available. Unknown values are never set to zero.", "",
                  "按船重抽的 95% 区间描述组合抽样波动；台风“至少/最多”与浪高“最低/最高”描述缺失位置造成的可能范围，两者含义不同。",
                  "", "Vessel-bootstrap 95% intervals describe sampling variation. TC at-least/at-most and wave lowest/highest limits describe the range caused by unknown positions.", ""])
    (out / "03_FIELD_GUIDE.md").write_text("\n".join(guide))

    q = pd.read_csv(out / "04_voyage_error_quantiles.csv")
    g = pd.read_csv(out / "05_wave_population_guarantee.csv")
    def quantile(population: str, metric: str) -> pd.Series:
        rows = q.loc[q.population.eq(population) & q.metric.eq(metric)]
        if len(rows) != 1:
            raise ValueError(f"missing quantile: {population}/{metric}")
        return rows.iloc[0]
    def guarantee(population: str) -> pd.Series:
        rows = g.loc[g.population.eq(population)]
        if len(rows) != 1:
            raise ValueError(f"missing wave guarantee: {population}")
        return rows.iloc[0]
    all_300 = quantile("all", "tc300_width_pp")
    all_500 = quantile("all", "tc500_width_pp")
    pub_300 = quantile("release_voyages_ge5", "tc300_width_pp")
    pub_500 = quantile("release_voyages_ge5", "tc500_width_pp")
    all_wave = guarantee("all")
    pub_wave = guarantee("release_voyages_ge5")
    readme = f"""# E002 2025 全程台风占比与浪高暴露：T013 内部评审

**至少 90% 的全部航次，300 km 和 500 km 台风暴露占比的“至少/最多”上下限重合；90% 的航次平均显著波高可能范围宽度在 {all_wave.p90:.3f} m 以内。** 这里的浪高结论把无法给出数值界限的航次放在最差尾部，按全部 {all_wave.voyages_total:,.0f} 个航次计算。

台风比例这样读：“至少”是有位置时已经观测到的台风小时占**全程小时**的百分比；“最多”把未知位置时可能遇到台风的小时也计入，同样除以全程小时。例如全程 100 小时，观测到 2 小时、未知时段另有 3 小时可能遇台风，就读作至少 2%、最多 5%。

**先读 `01_release_table_voyages_ge5.csv`，逐列中英定义见 `03_FIELD_GUIDE.md`。** 这是内部发布草稿，待 Supreme 审查及用户 UG2 验收；没有对外发布。来源分支为 `experiment/E002-2025-od-exposure-release`，本轮执行前 HEAD 为 `99cd9d0`，C1/C2 计算结果来自 `584126f`。

## 发布集与计算口径

- 每个有向“出发城市 × 到达城市 × 船型”至少 5 个航次：{report['release_combinations']:,} 个组合、{report['release_voyages']:,} 个航次。其余 {report['below_threshold_combinations']:,} 个组合、{report['below_threshold_voyages']:,} 个航次在 `02_below_threshold_summary.csv` 汇总，没有静默舍弃。
- 发布集逐航次 90 分位：300 km 台风占比范围宽 {pub_300.p90:.3f} 个百分点，500 km 为 {pub_500.p90:.3f} 个百分点，平均显著波高范围宽 {pub_wave.p90:.3f} m（按全部发布集航次，把无界限者放在最差尾部）。因此全年“至少 90% 台风上下限重合”的句子对应**全部航次**；发布集的 500 km 90 分位虽接近零，但不是严格零。
- 台风“至少/最多”均以全程小时为分母，分别保留对应小时数。发布表不再含“观测台风小时 ÷ 有位置小时”的旧率及其区间，也没有旧端点规则的列。旧规则对照仍在上一轮 T012 评审包。
- 新的台风“至少”95% 抽样区间按组合内整艘船重抽，1,000 次、seed `20260923`，使用与 U4 相同的每组合种子与百分位算法。平均浪高和至少 4 m 浪高比例的原 U4 抽样区间逐值保留。船数少于 3 的 {report['fewer_than_3_ships']:,} 个组合保留在表内，其抽样区间为空。
- 观测平均显著波高 = 已匹配浪高小时积分 ÷ 浪高匹配小时，不能除以全程小时，否则未知时段会被当成 0 m。最低/最高可能平均浪高另将可达未知小时及其 ERA5 显著波高上下限计入；这些是位置缺失范围，含义不同于按船抽样区间。
- {report['no_wave_data_combinations']:,} 个组合既没有匹配浪高小时，也没有可定界的未知浪高小时，因此浪高均值与最低/最高可能值为空；另有 {report['no_observed_wave_but_bounds_combinations']:,} 个组合观测均值为空，但有可达浪高上下限。`wave_mean_missing_reason` 逐行说明。未知不补零。

## 复用和闭合

- 复用 `584126f` 已完成的港口端点 C1、253/253 分片浪高 C2，以及 U1 航次与 U4 浪高/≥4 m 抽样结果；没有重跑这些阶段。T013 只重算台风“至少”比例的抽样区间、重组发布列并复核闭合。来源哈希见 `06_closure.json`。
- `07_independent_closure.json` 从 U1 全量组合表与 T012 新规则组合表独立核对发布集加未达门槛汇总的组合数、航次数及小时数；逐组合核对台风至少 ≤ 最多、全程分母和小时拆分；核对新抽样区间与 U4 的种子/次数，并独立重抽 {verified['bootstrap_combinations_independently_redrawn']} 个组合；同时核对少船空值和 515 个无浪高数据组合。
- `04_voyage_error_quantiles.csv`、`05_wave_population_guarantee.csv` 只列新端点规则。原规则、逐航次及 C1/C2 详细限制仍在 [T012 评审包](../2026-09-24_1648_E002-T012-exposure-unit-review/00_README.md)。本轮结果是暴露区间的宽度，不是“真实暴露落在区间内”的实测覆盖率。

正式 2025 产品未改。没有 API 请求、下载、2024 计算、合并、push 或上传。逐航次、按船抽样内部结果与含坐标数据只在 Git 忽略的 `02_coding/.runs/`；本目录没有船舶/航次 ID 或坐标。
"""
    (out / "00_README.md").write_text(readme)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
