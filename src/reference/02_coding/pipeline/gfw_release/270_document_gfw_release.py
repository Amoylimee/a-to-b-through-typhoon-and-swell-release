#!/usr/bin/env python3
"""Write aggregate-only review files from independently validated GFW results."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from step6_core import *


def places(field):
    if field.startswith("share_wave_hours_ge4m"): return 3
    if "_hours" in field: return 1
    if field.startswith(("mean_swh_", "share_wave_hours_ge4m")) or "coverage" in field or "_pct" in field: return 3
    return None


def rounded_csv(frame, target):
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(frame.columns)
        for row in frame.itertuples(index=False, name=None):
            rendered = []
            for field, value in zip(frame.columns, row):
                p = places(field)
                if pd.isna(value): rendered.append("")
                elif p is not None:
                    text_value = f"{float(value):.{p}f}"
                    rendered.append(text_value.replace("-0."+"0"*p, "0."+"0"*p))
                else: rendered.append(value)
            writer.writerow(rendered)


def field_definitions():
    definitions = {
        "origin_city": ("城市键", "起点城市；有向城市对的起点"),
        "dest_city": ("城市键", "终点城市；反向航线另算一个组合"),
        "shiptype_group": ("类别", "船型组；本步骤沿用第 3 步标签，公开来源资格仍待核查"),
        "voyages": ("航次", "此组合 2025 年保留的航次数，发布候选门槛为 ≥5"),
        "ships": ("艘", "组合内不同船舶身份数，同一船多个航次只计一艘"),
        "voyage_hours": ("h", "全部航次离港至到港的时长之和；台风百分比的全程分母"),
        "track_hours": ("h", "小时窗中点有观测或重建位置的时长"),
        "position_coverage": ("0–1", "track_hours ÷ voyage_hours；这是覆盖率，不是位置准确率"),
        "unresolved_track_hours": ("h", "位置未知时长；与 track_hours 合计为 voyage_hours"),
        "wave_matched_hours": ("h", "已有位置且成功匹配 ERA5 有效显著波高的时长"),
        "wave_unmatched_hours": ("h", "voyage_hours − wave_matched_hours，含未知位置及有位置但未匹配浪高的时段"),
        "wave_ge4m_matched_hours": ("h", "已匹配浪高至少 4 m 的时长；不是所有未知时段的高浪上限"),
        "wave_coverage": ("0–1", "wave_matched_hours ÷ voyage_hours"),
        "mean_swh_matched_m": ("m", "主值：已匹配浪高小时的平均显著波高 = 浪高小时积分 ÷ wave_matched_hours；不截断、未知不补零"),
        "mean_swh_min_possible_m": ("m", "已知浪高积分与未知可达湿格最小浪高积分之和 ÷ 支持小时"),
        "mean_swh_max_possible_m": ("m", "已知浪高积分与未知可达湿格最大浪高积分之和 ÷ 支持小时"),
        "mean_swh_possible_range_m": ("m", "最高可能平均浪高 − 最低可能平均浪高；反映缺失位置的不确定性"),
        "wave_bound_hours": ("h", "浪高上下限覆盖的时长 = wave_matched_hours + wave_unknown_counted_hours"),
        "wave_bound_coverage": ("0–1", "wave_bound_hours ÷ voyage_hours；上下限只描述这部分时间"),
        "wave_unboundable_hours": ("h", "voyage_hours − wave_bound_hours：有位置但没匹配到 ERA5，或位置未知且可达区内没有有效湿格的时长"),
        "mean_swh_constrained_m": ("m", "次要：主值截到 [最低可能, 最高可能] 内的值；修订前的点估计，保留供对照"),
        "mean_swh_constrained_shift_m": ("m", "mean_swh_constrained_m − mean_swh_matched_m；0 表示主值本来就在范围内"),
        "wave_mean_was_constrained": ("布尔或空", "主值落在可能范围之外、被截过时为 True；没有主值时留空"),
        "wave_mean_missing_reason": ("文本或空", "无可观测均值时说明原因；有可观测均值时留空。空不代表波高为 0"),
        "share_wave_hours_ge4m": ("0–1", "wave_ge4m_matched_hours ÷ wave_matched_hours；分母为已匹配小时"),
        "voyages_with_tc300": ("航次", "至少有一个已知位置小时命中 300 km 台风圈的航次数"),
        "voyages_with_tc500": ("航次", "同上，半径 500 km；未确认不等于未知时段没有台风"),
        "single_ship": ("布尔", "ships = 1；保留该组合，但抽样区间不估"),
        "sampling_interval_unavailable_fewer_than_3_ships": ("布尔", "ships <3；四个指标的按船抽样区间留空"),
        "unbounded_hours": ("h", "缺少位置与可连接港口端点，无法定界的时长；在台风最多暴露中计入"),
        "wave_unknown_counted_hours": ("h", "位置未知但可达区有有效 ERA5 湿格，已纳入浪高上下限的时长"),
    }
    for metric, unit, meaning in (
        ("mean_swh_matched", "m", "已匹配小时平均浪高"),
        ("share_wave_hours_ge4m", "0–1", "已匹配小时中 ≥4 m 比例")):
        for side, word in (("low", "下端"),("high", "上端")):
            definitions[f"{metric}_ci95_{side}"] = (unit, f"{meaning}的按船整簇重抽 95% 区间{word}；不足 3 艘或无有效重抽值时留空")
    for radius in (300,500):
        p = f"tc{radius}"
        definitions.update({
            f"{p}_hours_at_least": ("h", f"已知位置距同时刻台风中心 ≤{radius} km 的确认暴露小时"),
            f"{p}_bounded_possible_hours": ("h", f"位置未知但有界，可达区与 {radius} km 圈相交的可能暴露小时"),
            f"{p}_hours_at_most": ("h", f"{p}_hours_at_least + {p}_bounded_possible_hours + unbounded_hours"),
            f"{p}_at_least_pct": ("%", f"100 × {p}_hours_at_least ÷ voyage_hours"),
            f"{p}_at_most_pct": ("%", f"100 × {p}_hours_at_most ÷ voyage_hours；未知时段按可达条件计入"),
            f"{p}_at_least_pct_ci95_low": ("%", "确认台风占比的按船抽样 95% 区间下端；不足 3 艘时留空"),
            f"{p}_at_least_pct_ci95_high": ("%", "确认台风占比的按船抽样 95% 区间上端；不足 3 艘时留空"),
        })
    return definitions


def threshold_table(full, grouped):
    rows = []
    for n in (5,10,20,30):
        part = full.loc[full.voyages.ge(n)]
        row = {"minimum_voyages": n, "retained_combinations": len(part), "retained_voyages": int(part.voyages.sum()),
               "retained_voyage_hours": float(part.voyage_hours.sum()),
               "voyage_share_of_all": float(part.voyages.sum()/grouped.voyages.sum()),
               "fewer_than_3_ships": int(part.ships.lt(3).sum())}
        for metric in ("mean_swh_matched", "share_wave_hours_ge4m", "tc300_at_least_pct", "tc500_at_least_pct"):
            widths = part[metric+"_ci95_high"]-part[metric+"_ci95_low"]
            row[metric+"_interval_groups"] = int(widths.notna().sum())
            row[metric+"_width_p50"] = float(widths.quantile(.5))
            row[metric+"_width_p90"] = float(widths.quantile(.9))
        rows.append(row)
    return pd.DataFrame(rows)


def document(root: Path, out: Path):
    source = binding(); root = inside_runs(root); out = out.resolve()
    if not out.is_relative_to((CODING/"outputs").resolve()): raise ValueError("review files belong in outputs")
    validation = json.loads((root/"validation.json").read_text())
    if validation["status"] != "pass" or not all(validation["checks"].values()): raise ValueError("independent validation required")
    full = pd.read_parquet(root/"assembly/release_full_precision.parquet")
    grouped = pd.read_parquet(root/"assembly/all_combinations.parquet")
    fields = field_definitions()
    if set(full) != set(fields): raise ValueError(f"incomplete field meanings: {set(full)^set(fields)}")
    out.mkdir(parents=True, exist_ok=False)
    rounded_csv(full, out/"01_release_table_voyages_ge5.csv")
    below = json.loads((root/"assembly/below_threshold_summary.json").read_text())
    pd.DataFrame([below]).to_csv(out/"02_below_threshold_summary.csv", index=False, lineterminator="\n")
    for filename in ("metrics_by_stratum.csv", "window_error_summary.csv"):
        shutil.copy2(root/"holdout"/filename, out/("04_holdout_"+filename))
    shutil.copy2(root/"assembly/missing_exposure_quantiles.csv", out/"05_missing_exposure_quantiles.csv")
    thresholds = threshold_table(full, grouped)
    thresholds.to_csv(out/"06_sampling_thresholds.csv", index=False, lineterminator="\n")
    points = pd.concat([pd.read_parquet(p, columns=["sample_set","nominal_gap_hours","sample_id","position_error_km"])
                       for p in sorted((root/"holdout/shards").glob("*/point_results.parquet"))],ignore_index=True)
    summaries = pd.read_csv(root/"holdout/window_error_summary.csv")
    digest = []
    for hours, block in points.loc[points.sample_set.eq("gap_length_x_shiptype")].groupby("nominal_gap_hours",sort=True):
        row = summaries.loc[summaries.sample_set.eq("gap_length_x_shiptype") & summaries.nominal_gap_hours.eq(hours)].iloc[0].to_dict()
        row.update(position_points=len(block), position_predicted_points=int(block.position_error_km.notna().sum()),
                   position_error_p90_km=float(block.position_error_km.quantile(.9)))
        digest.append(row)
    digest = pd.DataFrame(digest)
    digest.to_csv(out/"04_holdout_digest.csv",index=False,lineterminator="\n")
    quantiles = pd.read_csv(out/"05_missing_exposure_quantiles.csv")
    d = validation["details"]
    wave = quantiles.loc[quantiles.population.eq("all") & quantiles.rule.eq("main") & quantiles.metric.eq("wave_range_m")].iloc[0]
    tc = quantiles.loc[quantiles.population.eq("all") & quantiles.rule.eq("main") & quantiles.metric.str.startswith("tc")]
    h4 = digest.loc[digest.nominal_gap_hours.eq(4)].iloc[0]; h24 = digest.loc[digest.nominal_gap_hours.eq(24)].iloc[0]
    storm24 = summaries.loc[summaries.sample_set.eq("tc500km_additional") & summaries.nominal_gap_hours.eq(24)].iloc[0]
    ci = thresholds.iloc[0]
    wave_text = (f"可计算浪高界限的 {int(wave.available):,} 条航次中，至少 90% 的上下限完全相同"
                 if wave.p90 <= 1e-12 else
                 f"可计算浪高界限的 {int(wave.available):,} 条航次中，90% 的界限宽度约在 {wave.p90:.3f} m 以内")
    tc300_width = float(tc.loc[tc.metric.eq("tc300_range_pp"),"p90"].iloc[0])
    tc500_width = float(tc.loc[tc.metric.eq("tc500_range_pp"),"p90"].iloc[0])
    tc_text = ("至少 90% 的航次，台风 300／500 km 的上下限完全相同"
               if max(tc300_width,tc500_width) <= 1e-12 else
               f"台风 300／500 km 界限宽度的 p90 为 {tc300_width:.3f}／{tc500_width:.3f} 个百分点")
    summary_rows = [
        ("缺失位置", f"{d['totals']['unresolved_track_hours']:,.0f} 个小时位置未知。{wave_text}；{tc_text}。"),
        ("插值精度", f"共 23,000 个 GFW 留出窗口，{d['unresolved_holdout_windows']:,} 个未完整重建。预测成功的点中，4/24 h 缺口位置误差 p90 为 {h4.position_error_p90_km:.1f}/{h24.position_error_p90_km:.1f} km；可匹配窗口的平均浪高差绝对值 p90 为 {h4.window_wave_mean_difference_m_p90_abs:.3f}/{h24.window_wave_mean_difference_m_p90_abs:.3f} m。"),
        ("港口定界", f"{source['counts']['candidates']:,} 条航次均连接到本次 GFW 城市参考。完全无位置的 {d['no_position_voyages']:,} 条航次按 25 kn 与港口 ±1 h 检查，端点速度冲突为 {d['conflicting_main_port_endpoints']}；冲突航次保留，未知时段按无界处理并单列，不调高航速。主值 0 km、1 h，并保存 0 km、0 h 对照；尚无独立港口真值验证。"),
        ("抽样稳定性", f"{d['fewer_than_3_ships']:,} 个候选组合不足 3 艘船，区间留空。有可估区间的组合中，平均浪高区间宽度 p90 为 {ci.mean_swh_matched_width_p90:.3f} m；台风 300/500 km 为 {ci.tc300_at_least_pct_width_p90:.3f}/{ci.tc500_at_least_pct_width_p90:.3f} 个百分点。"),
    ]
    overview = ["| 角度 | 本次 GFW 结果 |", "|---|---|"]+[f"| {a} | {b} |" for a,b in summary_rows]
    provenance_info = provenance([Path(__file__), DECISION, REVISION])
    readme = ["# E006 第 6 步：GFW 聚合与不确定性评审", "",
        f"本步骤已完成计算与独立核验。全年 {d['candidates']:,} 条 GFW 航次形成 {d['all_combinations']:,} 个有向城市对 × 船型组合；按至少 5 航次门槛，发布候选表保留 **{d['release_combinations']:,} 个组合、{d['release_voyages']:,} 条航次**。另外 {d['below_combinations']:,} 个组合、{d['below_voyages']:,} 条航次只进入门槛下汇总。", "",
        "先读本文件，再看 [发布候选表](01_release_table_voyages_ge5.csv) 与 [字段说明](03_FIELD_GUIDE.md)。这是本地评审候选，尚未经授权 reviewer 正式采纳；没有公开上传、push 或合并 master。", "", *overview, "",
        "## 怎样读浪高、台风和区间", "",
        "缺失位置界限问的是：在 25 kn 航速上限和已知端点假设下，未知时段可能到哪里、可能遇到什么天气；GFW 点只记到整点，前后已知位置的时刻各放宽 30 分钟（E007 规则修订）。按船抽样区间问的是：换一组船后，该城市对的汇总是否稳定。两种区间回答不同问题，不能互相替代。", "",
        "台风‘至少/最多’以全程航次小时为分母。浪高未知不补零：主值 `mean_swh_matched_m` 只平均已匹配浪高的小时；`mean_swh_min_possible_m`／`mean_swh_max_possible_m` 再把有有效 ERA5 可达界限的未知小时算进来，给出这些时间合起来的平均浪高最低、最高可能值。",
        f"有位置但未匹配浪高的 {d['known_position_wave_unmatched_hours']:,.0f} h，以及位置未知且无有效湿格界限的 {d['wave_unknown_unsupported_hours']:,.0f} h，不进入浪高均值与上下限；因此这些界限不能约束全部航次时间的真实海浪。两类不支持时间均保留在覆盖字段及核验汇总中。", "",
        "`wave_bound_hours` = wave_matched_hours + wave_unknown_counted_hours，`wave_bound_coverage` = wave_bound_hours ÷ voyage_hours，剩下的 `wave_unboundable_hours` 不受上下限约束。主值落在可能范围之外时，`mean_swh_constrained_m` 给出截到范围内的值，`mean_swh_constrained_shift_m` 给出截动的幅度；它是修订前的点估计，只作对照。按船抽样区间针对主值；船数 <3 的组合照常保留，抽样区间为空。", "",
        f"每个可估组合尝试 1,000 次按船重抽；若某次没有可匹配浪高，该次浪高估计保持空，区间只取有效抽样的分位数。本次有 {d['wave_bootstrap_groups_fewer_than_1000_valid_draws']:,} 个可估浪高组合不足 1,000 次有效抽样，最少 {d['wave_bootstrap_min_valid_draws']} 次；不补零、不追加抽样。", "",
        "## 本次验证的边界", "",
        "留出真值来自连续逐小时的实际 GFW 观测点，不是生产插值点。先冻结样本清单，再隐藏窗口内全部观测；前后各 1 h 上下文保留，生产第 4 步寻路代码直接复用。1 h 窗口没有内部逐小时观测点，依用户批准标记不可估。各 45 个‘长度 × 船型’层 500 段，另有 500 个台风 500 km 内窗口。", "",
        "窗口天气差只比较同一组内部观测时刻：浪高只在真值和预测两处都有 ERA5 匹配的点上求均值，台风比例只在有有效预测的内部点上求值。窗口端点不计入误差。未重建和单侧浪高未匹配不会当作零误差，其数量在分层表中单列。", "",
        "误差是相对 GFW 小时位置的重建误差；GFW 时间按 hh:30 记录，空间分辨率 0.01°，不能视为独立物理定位真值。该检验不单独估计小时内取点时刻的不确定性。主表的覆盖率也不能当定位准确率；30 分记时使部分航次首尾恰好落在观测点上，会提高有位置比例。", "",
        f"台风附近不能用普通窗口的低翻转率概括。在额外台风样本的 {int(storm24.windows)} 个 24 h 窗口中，TC300／TC500 小时占比差绝对值的 p90 为 {storm24.window_tc300_share_difference_pp_p90_abs:.3f}／{storm24.window_tc500_share_difference_pp_p90_abs:.3f} 个百分点。这是刻意抽取的台风附近窗口误差，不是全年航次错误率。", "",
        "天气快照仍为 ERA5 2025 总显著波高 0.5°、湿格上限 117.94 km，以及 IBTrACS v04r01 2026-09-06（含 provisional，非最终 best track），台风支撑 gap ≤3 h。来源、时间、位置、全程分母沿用第 5 步冻结规则。", "",
        "## 文件与复核入口", "",
        "- `01_release_table_voyages_ge5.csv`：49 个字段；小时 1 位、米/百分比/比例 3 位小数，空值保持空。各列独立取整，末位可能不相加。",
        "- `02_below_threshold_summary.csv`：门槛下组合的汇总，不公开逐航次记录。",
        "- `04_holdout_*`：分层数量、位置误差、浪高差与台风翻转率；未重建计入数量，误差条件于有效预测或匹配。",
        "- `05_missing_exposure_quantiles.csv`：主值与零时间容差对照的逐航次界限宽度分布，分别报告全部航次与发布候选航次。",
        "- `06_sampling_thresholds.csv`：门槛 5/10/20/30 的保留数量、小时与抽样区间宽度；当前仍采用 5。",
        "- `07_validation_summary.json`：独立核验结论、年度闭合和支持时间；全精度证据保存在下列运行根。", "",
        f"运行根：`{root.relative_to(ROOT)}`。全精度表：`assembly/release_full_precision.parquet`；航次界限：`assembly/voyage_intervals.parquet`；完整核验：`validation.json`。", "",
        f"来源分支 `{provenance_info['source_branch']}`；构建代码提交 `{provenance_info['source_commit']}`。各组件自己的 source_commit 和代码哈希保存在运行 receipt 中，允许流水线不同组件在不同已保存提交上执行。", "",
        "规则修订（2026-10-09，用户决定）见 `pipeline/gfw_release/config/step6_rule_revision.json`，并已登记于 Contract §3。船型公开来源资格与公开许可尚须后续核查。", ""]
    (out/"00_README.md").write_text("\n".join(readme),encoding="utf-8")
    example = full.loc[full.origin_city.eq("CHN:SHANGHAI") & full.dest_city.eq("CHN:NINGBO") & full.shiptype_group.eq("container")]
    if example.empty: example = full.sort_values("voyages",ascending=False).head(1)
    example = example.iloc[0]
    guide = ["# 2025 有向城市对 × 船型：字段与空值", "",
        "一行是起点城市 → 终点城市、一个船型、2025 全年的聚合。往返方向分开，不是航次级或国家级数据。", "",
        "CI 是按船整簇有放回重抽 1,000 次的 2.5% 与 97.5% 分位数；seed 20260923。船数少于 3 时不估区间。浪高和高浪比例还可能因无可匹配小时而留空；`wave_mean_missing_reason` 给出浪高点估计缺失原因。空值不能当作 0。", "",
        f"1,000 次是尝试次数；某次抽样没有浪高匹配时，该次浪高估计为空，CI 使用有效抽样的分位数。本次 {d['wave_bootstrap_groups_fewer_than_1000_valid_draws']:,} 个可估浪高组合的有效次数不足 1,000，最少 {d['wave_bootstrap_min_valid_draws']} 次。", "",
        "台风百分比均以全程小时为分母。浪高支持分母和其局限见 [评审说明](00_README.md)。抽样区间描述不同船样本的变化，不要求落在原样本的缺失位置上下限内。", "",
        f"示例：`{example.origin_city}` → `{example.dest_city}`，`{example.shiptype_group}`；示例值来自本次表，以下展示精度同 CSV。", "",
        "| 字段 | 单位 | 含义与分母 | 示例 |", "|---|---|---|---|"]
    for field in full:
        unit, meaning = fields[field]; value = example[field]; p = places(field)
        display = "空" if pd.isna(value) else (f"{float(value):.{p}f}" if p is not None else str(value))
        guide.append(f"| `{field}` | {unit} | {meaning} | {display} |")
    guide.extend(["", "船型标签：`bulk_carrier` 散货船；`container` 集装箱船；`crude_oil_tanker` 原油船；"
                  "`gas_tanker` 气体运输船；`general_cargo` 普通货船；`product_chemical_tanker` 成品油／化学品船；"
                  "`reefer` 冷藏船；`ro_ro_vehicle` 滚装／车辆运输船；`other_retained` 其他按现行样本规则保留的船型。", "",
                  "浪高缺失原因：`no_matched_wave_hours_reachable_bounds_available` 表示没有观测匹配均值，仍有可达区上下限；"
                  "`no_matched_or_reachable_wave_hours` 表示观测与可达区都没有支持小时。此原因列为空时，浪高点估计已有可观测均值。"])
    guide.extend(["", "零台风确认小时表示已知位置上未命中，须同时查看未知小时和最多暴露；两个比例都为零且没有位置未知时，才表示该冻结快照下未发现暴露。", ""])
    (out/"03_FIELD_GUIDE.md").write_text("\n".join(guide),encoding="utf-8")
    summary = {"status":"pass", "passed_checks":len(validation["checks"]), "details":d,
               "private_validation":record(root/"validation.json"), "adopted":False, **provenance_info}
    write_json(out/"07_validation_summary.json", summary)
    reread = pd.read_csv(out/"01_release_table_voyages_ge5.csv")
    if list(reread) != list(full) or len(reread) != len(full): raise ValueError("review CSV shape changed")
    if not reread[KEY].equals(full[KEY].reset_index(drop=True)): raise ValueError("review CSV keys changed")
    for field in full.select_dtypes(include="number"):
        p = places(field); tolerance = .5*10**(-p)+1e-10 if p is not None else 0
        if not np.allclose(reread[field],full[field],rtol=0,atol=tolerance,equal_nan=True): raise ValueError("CSV rounding error: "+field)
    manifest = {"status":"documented_and_checked", "source_run":str(root), "source_validation":record(root/"validation.json"),
                "rows":len(full), "columns":len(full.columns), "aggregate_only":True, "adopted":False,
                "outputs":{p.name:record(p) for p in out.iterdir() if p.is_file()}, **provenance_info}
    write_json(out/"08_review_manifest.json",manifest)
    print(json.dumps({"status":manifest["status"],"review":str(out),"rows":len(full),"fields":len(full.columns)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();document(args.run,args.out)
