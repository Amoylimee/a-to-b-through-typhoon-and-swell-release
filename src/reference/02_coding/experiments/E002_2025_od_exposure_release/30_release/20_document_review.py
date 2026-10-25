#!/usr/bin/env python3
"""Write the T012/T011 internal review guide from closed aggregate outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


CODING = Path(__file__).resolve().parents[3]
RUN = CODING / ".runs/2026-09-24_1500_E002-T012-exposure-units"


def _fields() -> dict[str, tuple[str, str, str]]:
    fields = {
        "origin_city": ("出发城市", "Origin city", "有向 OD 的出发城市代码；无分母"),
        "dest_city": ("到达城市", "Destination city", "有向 OD 的到达城市代码；无分母"),
        "shiptype_group": ("船型组", "Ship type group", "正式产品的船型分类；无分母"),
        "voyages": ("航次数", "Voyages", "该组合保留航次数；单位：次"),
        "ships": ("船数", "Ships", "按 vessel_uid_clean 去重；单位：艘"),
        "voyage_hours": ("航次总小时", "Total voyage hours", "全年该组合航次总时长；单位：小时"),
        "track_hours": ("有位置小时", "Positioned hours", "正式产品可定位时长；单位：小时"),
        "position_coverage": ("位置覆盖率", "Position coverage", "有位置小时 ÷ 航次总小时；0–1"),
        "unresolved_track_hours": ("未知位置小时", "Unresolved position hours", "正式产品未解析的位置时长；单位：小时；不补零"),
        "wave_matched_hours": ("浪高匹配小时", "Wave matched hours", "已知位置且匹配 ERA5 显著波高的时长；单位：小时"),
        "wave_unmatched_hours": ("浪高未匹配小时", "Wave unmatched hours", "有位置但浪高未匹配的时长；单位：小时"),
        "wave_ge4m_matched_hours": ("浪高至少 4 m 的匹配小时", "Matched hours with SWH at least 4 m", "在浪高匹配小时内计算；单位：小时"),
        "wave_coverage": ("浪高覆盖率", "Wave coverage", "浪高匹配小时 ÷ 航次总小时；0–1"),
        "mean_swh_matched_m": ("观测平均显著波高", "Observed mean significant wave height", "已匹配浪高×小时积分 ÷ 浪高匹配小时；单位：m"),
        "mean_swh_m_ci95_low": ("平均浪高抽样区间下端", "Mean SWH bootstrap 95% lower", "按船整簇 bootstrap 的 95% 区间；单位：m；船数少于 3 时空值"),
        "mean_swh_m_ci95_high": ("平均浪高抽样区间上端", "Mean SWH bootstrap 95% upper", "同上；单位：m"),
        "share_wave_hours_ge4m": ("至少 4 m 浪高比例", "Share of wave hours at least 4 m", "≥4 m 匹配小时 ÷ 浪高匹配小时；0–1"),
        "share_wave_hours_ge4m_ci95_low": ("至少 4 m 比例抽样区间下端", "SWH at least 4 m bootstrap 95% lower", "按船整簇 bootstrap；0–1；船数少于 3 时空值"),
        "share_wave_hours_ge4m_ci95_high": ("至少 4 m 比例抽样区间上端", "SWH at least 4 m bootstrap 95% upper", "同上；0–1"),
        "tc300_hours_observed": ("300 km 台风观测小时", "Observed TC300 hours", "正式产品有位置时可观测的 TC300 暴露；单位：小时"),
        "tc500_hours_observed": ("500 km 台风观测小时", "Observed TC500 hours", "正式产品有位置时可观测的 TC500 暴露；单位：小时"),
        "tc300_rate_per_position_hour": ("300 km 台风观测率", "Observed TC300 rate per positioned hour", "TC300 观测小时 ÷ 有位置小时；0–1；发布点估计"),
        "tc500_rate_per_position_hour": ("500 km 台风观测率", "Observed TC500 rate per positioned hour", "TC500 观测小时 ÷ 有位置小时；0–1；发布点估计"),
        "tc300_rate_per_position_hour_ci95_low": ("300 km 观测率抽样区间下端", "TC300 rate bootstrap 95% lower", "按船整簇 bootstrap；0–1；船数少于 3 时空值"),
        "tc300_rate_per_position_hour_ci95_high": ("300 km 观测率抽样区间上端", "TC300 rate bootstrap 95% upper", "同上；0–1"),
        "tc500_rate_per_position_hour_ci95_low": ("500 km 观测率抽样区间下端", "TC500 rate bootstrap 95% lower", "按船整簇 bootstrap；0–1；船数少于 3 时空值"),
        "tc500_rate_per_position_hour_ci95_high": ("500 km 观测率抽样区间上端", "TC500 rate bootstrap 95% upper", "同上；0–1"),
        "voyages_with_tc300": ("有 TC300 暴露的航次数", "Voyages with observed TC300 exposure", "按正式产品观测值大于零计数；单位：次"),
        "voyages_with_tc500": ("有 TC500 暴露的航次数", "Voyages with observed TC500 exposure", "按正式产品观测值大于零计数；单位：次"),
        "single_ship": ("单船标记", "Single ship flag", "该组合只有 1 艘船；布尔值"),
        "sampling_interval_unavailable_fewer_than_3_ships": ("少船区间不可用标记", "Sampling interval unavailable below three ships", "船数少于 3 时为真，四组 bootstrap 区间留空"),
        "new_unbounded_hours": ("新规则仍无界小时", "Unbounded hours under port rule", "港口端点不能完整连接的无中途位置航次时长；单位：小时；纳入台风保守上限"),
        "new_wave_counted_hours": ("新规则计入浪高界限的未知小时", "Unknown hours counted in new wave bounds", "25 kn 可达区含湿 ERA5 格点的未知时长；单位：小时"),
        "new_wave_lower_m": ("新规则平均浪高下限", "Mean SWH lower under port rule", "(已匹配浪高积分 + 可达最小浪高积分) ÷ (匹配小时 + 计入未知小时)；单位：m"),
        "new_wave_upper_m": ("新规则平均浪高上限", "Mean SWH upper under port rule", "同上，用可达最大浪高积分；单位：m"),
        "new_wave_width_m": ("新规则平均浪高误差宽度", "Mean SWH bound width under port rule", "上限 − 下限；单位：m"),
    }
    for radius in (300, 500):
        prefix = f"new_tc{radius}"
        fields[f"{prefix}_hours_lower"] = (f"{radius} km 台风小时下限", f"TC{radius} hour lower", "正式产品可观测台风小时；单位：小时")
        fields[f"{prefix}_bounded_possible_hours"] = (f"{radius} km 有界可能小时", f"TC{radius} bounded possible hours", "未知位置可能暴露小时；含港口端点速度不可行时的保守回退；单位：小时")
        fields[f"{prefix}_hours_upper"] = (f"{radius} km 台风小时上限", f"TC{radius} hour upper", "下限 + 有界可能小时 + 无界小时；单位：小时")
        fields[f"{prefix}_lower_pp"] = (f"{radius} km 台风占比下限", f"TC{radius} exposure share lower", "台风小时下限 ÷ 航次总小时 ×100；单位：百分点")
        fields[f"{prefix}_upper_pp"] = (f"{radius} km 台风占比上限", f"TC{radius} exposure share upper", "台风小时上限 ÷ 航次总小时 ×100；单位：百分点")
        fields[f"{prefix}_width_pp"] = (f"{radius} km 台风占比误差宽度", f"TC{radius} exposure share width", "上限 − 下限；单位：百分点")
    return fields


def build(out: Path) -> None:
    closure = json.loads((out / "06_closure.json").read_text())
    c1 = json.loads((RUN / "10_c1/port_endpoints_closure.json").read_text())
    preflight = json.loads((RUN / "20_c2_resource_preflight.json").read_text())
    cached = preflight["cached_upgrade"]
    live = preflight["live_calibration_after_first_batch"]
    run_status = json.loads((RUN / "30_census_full/full_run_status.json").read_text())
    quantiles = pd.read_csv(out / "01_voyage_error_quantiles.csv")
    guarantees = pd.read_csv(out / "01a_wave_population_guarantee.csv")
    outside = pd.read_csv(out / "02_tc_point_outside_bounds.csv")
    u3_path = RUN / "40_u3_conversion/u3_exposure_error_quantiles.csv"
    shutil.copyfile(u3_path, out / "07_u3_exposure_error_quantiles.csv")
    shutil.copyfile(RUN / "40_u3_conversion/u3_exposure_conversion_closure.json",
                    out / "08_u3_exposure_conversion_closure.json")
    shutil.copyfile(RUN / "10_c1/port_endpoints_closure.json", out / "09_c1_port_endpoint_provenance.json")
    shutil.copyfile(RUN / "20_c2_resource_preflight.json", out / "10_c2_resource_preflight.json")
    shutil.copyfile(RUN / "30_census_full/full_run_status.json", out / "13_c2_full_run_status.json")
    release = pd.read_csv(out / "04_release_table_draft_voyages_ge5.csv", nrows=0)
    fields = _fields()
    unknown = [column for column in release.columns if column not in fields]
    if unknown:
        raise ValueError(f"release fields lack bilingual explanation: {unknown}")
    field_rows = ["# 发布表逐列说明 / Release field guide", "",
                  "单位按列说明。比例字段为 0–1，台风暴露上下限字段为百分点。空值保留未知；不把未知填零。", "",
                  "| 字段 / Field | 中文 | English | 定义、分母与单位 |",
                  "|---|---|---|---|"]
    for column in release.columns:
        chinese, english, meaning = fields[column]
        field_rows.append(f"| `{column}` | {chinese} | {english} | {meaning} |")
    field_rows.extend(["", "同一行的按船抽样 95% 区间衡量组合抽样波动；25 kn 可达上下限衡量未知位置造成的暴露范围。两者不能相互替代。",
                       "", "平均浪高仅以已匹配小时及可达湿格点未知小时为分母。台风发布点估计仅以有位置小时为分母；台风上下限则以航次总小时为分母，因此点估计可能在该区间外。",
                       "", "旧规则对照和逐航次有效样本分位数见 `01_voyage_error_quantiles.csv`、`03_all_combinations_exposure_bounds.csv`；全部航次的浪高保守保证见 `01a_wave_population_guarantee.csv`；逐航次与含坐标结果只在 `.runs/`。", ""])
    (out / "11_FIELD_GUIDE.md").write_text("\n".join(field_rows))

    def q(population: str, rule: str, metric: str) -> pd.Series:
        row = quantiles.loc[(quantiles.population == population) & (quantiles.rule == rule)
                            & (quantiles.metric == metric)]
        if len(row) != 1:
            raise ValueError(f"missing quantile {population}/{rule}/{metric}")
        return row.iloc[0]

    all_tc300 = q("all", "new", "tc300_width_pp")
    all_tc500 = q("all", "new", "tc500_width_pp")
    all_wave = q("all", "new", "wave_width_m")
    pub_tc300 = q("release_voyages_ge5", "new", "tc300_width_pp")
    pub_tc500 = q("release_voyages_ge5", "new", "tc500_width_pp")
    pub_wave = q("release_voyages_ge5", "new", "wave_width_m")
    all_wave_guarantee = guarantees.loc[guarantees.population.eq("all") & guarantees.rule.eq("new")].iloc[0]
    pub_wave_guarantee = guarantees.loc[guarantees.population.eq("release_voyages_ge5") & guarantees.rule.eq("new")].iloc[0]
    old_tc300 = q("all", "old", "tc300_width_pp")
    old_tc500 = q("all", "old", "tc500_width_pp")
    old_wave = q("all", "old", "wave_width_m")
    new_out = outside.loc[outside.rule.eq("new") & outside.population.eq("all")]
    text = f"""# E002 2025 暴露单位不确定性与 OD × 船型发布表：内部评审

**90% 的全部航次，TC300 台风暴露占比误差在 {all_tc300.p90:.3f} 个百分点以内，TC500 在 {all_tc500.p90:.3f} 个百分点以内，平均显著波高误差在 {all_wave_guarantee.p90:.3f} m 以内。** 浪高把无可计算界限的 {all_wave.voyages_without_bound:,} 个航次保守放在最差尾部；其余 {all_wave.voyages_with_bound:,} 个才有数值界限。这里的“误差”是上限减下限，不是预测值减观测值。

台风 p90 为零表示至少 90% 的航次上下限相同；仍有较宽的尾部。全部航次的新规则 p95 为 TC300 {all_tc300.p95:.3f}、TC500 {all_tc500.p95:.3f} 个百分点，p99 为 TC300 {all_tc300.p99:.3f}、TC500 {all_tc500.p99:.3f} 个百分点。浪高若只看有界限航次，p90 为 {all_wave.p90:.3f} m；`01a_wave_population_guarantee.csv` 给出把无界限航次放在最差尾部后的全年分位数，若高分位无法保证有限误差则留空。

发布集（每个有向 OD × 船型至少 5 个航次）含 {closure['release_combinations']:,} 个组合、{closure['release_voyages']:,} 个航次。该集逐航次 p90 为 TC300 {pub_tc300.p90:.3f} 个百分点、TC500 {pub_tc500.p90:.3f} 个百分点、平均浪高 {pub_wave_guarantee.p90:.3f} m（无界限航次放在最差尾部；有界限 {pub_wave.voyages_with_bound:,}/{pub_wave.voyages_total:,}）。

**状态：内部草稿，待 Supreme 审查和用户 UG2 验收。** 来源分支 `experiment/E002-2025-od-exposure-release`，计算基线 `98c4d26`。正式产品未改；没有 API 请求、下载、2024 工作、合并、push 或上传。

## 方法与新旧规则

- C1：以正式航次 `candidate_id`、离港/到港 UTC 时刻和冻结 2025 城市参考表的港口坐标一对一连接。来源及 SHA-256 见 `09_c1_port_endpoint_provenance.json`。{c1['connected']:,}/{c1['candidate_rows']:,} 个航次两端均可连接；{c1['unconnected']:,} 个未完整连接。原来 {closure['old_unbounded_voyages']:,} 个途中无位置航次中，{closure['old_no_position_voyages_port_bounded']:,} 个获得两端港口定界，{closure['old_no_position_voyages_still_unbounded']:,} 个仍无界；另有 {closure['former_one_sided_segments_port_completed']:,} 个原单侧未知段补上对应港口端点。不能连接的不猜位置。速度上限 25 kn。
- C2：对全部 {closure['unknown_segments']:,} 个未知段计算可达区内湿 ERA5 格点的显著波高最小值和最大值。没有湿格点或两端速度不可行的段不进入浪高界限分母，分别计数。新规则无湿格点 {closure['new_wave_no_wet_segments']:,} 段、速度不可行 {closure['new_wave_speed_infeasible_segments']:,} 段；旧规则分别为 {closure['old_wave_no_wet_segments']:,}、{closure['old_wave_speed_infeasible_segments']:,} 段。初始 24-worker probe 预计 {preflight['linear_projection_hours']:.2f} 小时；大分片缓存 probe 与原结果逐值相同，改用 32 个 worker 后单分片外推 {cached['linear_projection_hours']:.2f} 小时。首批实际并发吞吐外推 {live['projected_resume_total_hours']:.2f} 小时，均低于 24 小时门槛。复用初轮 {cached['initial_full_shards_reused']} 个闭合分片；恢复阶段实际耗时 {run_status['elapsed_wall_seconds']/3600:.2f} 小时。见 `10_c2_resource_preflight.json`。
- C3：台风下限＝可观测台风小时 ÷ 航次总小时；上限＝可观测小时加未知可能小时后除以航次总小时。浪高下限/上限＝已匹配浪高小时积分加可达最小/最大浪高小时积分，再除以已匹配小时加计入未知小时。逐航次结果在 `.runs/`，组合对照见 `03_all_combinations_exposure_bounds.csv`。
- C4：`01_voyage_error_quantiles.csv` 给全部航次及发布集的新旧规则 p50/p75/p90/p95/p99、有效分母。旧规则全部航次 p90 为 TC300 {old_tc300.p90:.3f}、TC500 {old_tc500.p90:.3f} 个百分点，浪高 {old_wave.p90:.3f} m（浪高有效 {old_wave.voyages_with_bound:,}/{old_wave.voyages_total:,}）。`07_u3_exposure_error_quantiles.csv` 将既有完整留出窗口的逐点真值/预测换算为窗口平均浪高绝对误差和台风占比绝对误差，不重跑插值；544 个部分窗口单列。
- C5：253/253 分片、{closure['voyages']:,} 航次、{closure['combinations']:,} 个组合及未知段数/小时闭合；旧规则逐组合 TC 上限与已接受 U2 全量表一致。新旧逐航次、逐组合均检查下限 ≤ 上限；发布集加未达门槛汇总与 U1 全量闭合。详细计数见 `06_closure.json`；另从 `.runs/` 逐航次表独立复算 C4 全部分位数，结果见 `12_independent_recalculation.json`。

## 发布表与限制

先读 `04_release_table_draft_voyages_ge5.csv`，逐列中英解释见 `11_FIELD_GUIDE.md`。`05_below_threshold_summary.csv` 单列少于 5 个航次的 {closure['below_threshold_combinations']:,} 个组合和 {closure['below_threshold_voyages']:,} 个航次；没有静默舍弃。船数少于 3 的 {closure['fewer_than_3_ships_in_release']:,} 个发布组合保留，但按船抽样区间为空并标记。

台风发布点估计仍为“观测台风小时 ÷ 有位置小时”。它和以总航次小时作分母的上下限语义不同。新规则中，点估计落在区间外的 TC300 组合 {int(new_out.loc[new_out.radius_km.eq(300), 'combinations_outside'].iloc[0]):,} 个，TC500 {int(new_out.loc[new_out.radius_km.eq(500), 'combinations_outside'].iloc[0]):,} 个；对应航次占全年比例见 `02_tc_point_outside_bounds.csv`。此项只是诊断，不是闭合失败，也不把点估计改写为区间中心。

海浪变量为 ERA5 总显著波高 `swh`，不能称为涌浪。可达范围是 25 kn 与港口坐标所给的几何范围，不代表真实航迹；IBTrACS 沿用冻结的 provisional 快照。无湿格点、速度不可行和未连接港口保留为显式未知。港口端点的速度约束若与既有定位冲突，浪高该段不纳入分母，台风上限对此段作保守回退，避免错误收窄区间。区间宽度衡量缺失位置的可能暴露范围，按船 bootstrap 区间衡量抽样波动，两者不能互换。旧规则浪高界限对无中途位置且无匹配浪高的航次通常无定义，故新旧浪高 p90 必须连同有效航次数一起阅读。

评审文件均无航次/船舶 ID 或坐标。逐航次、逐窗口、含坐标证据只存于被 Git 忽略的 `02_coding/.runs/`。评审后由 Supreme 判断是否接受；UG2 和公开发布仍由用户决定。
"""
    (out / "00_README.md").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
