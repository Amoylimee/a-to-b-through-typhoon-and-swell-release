#!/usr/bin/env python3
"""Refresh coverage, threshold and city references on the corrected release.

Track composition uses weighted hourly weather segments, so its denominator is
the full voyage duration. It does not equate interior ten-minute row counts to
voyage hours. Reuse the existing month, threshold and city-reference functions.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa
HERE=Path(__file__).resolve().parent
CODING=HERE.parents[1]
sys.path.insert(0,str(CODING/'pipeline/gfw_release'))
import step6_core as core


def composition(shard,weather,out):
    pa.set_cpu_count(1)
    parts=[]
    for p in sorted((Path(weather)/'shards'/shard).glob('weather_*.parquet')):
        f=pd.read_parquet(p,columns=['candidate_id','segment_weight_hours','track_evidence','gap_class'])
        short=f.track_evidence.eq('observed_ais') | (f.track_evidence.eq('gfw_supported_interpolation')&f.gap_class.eq('gfw_sea_path_le4h'))
        long=f.track_evidence.eq('gfw_supported_interpolation')&f.gap_class.isin(['gfw_sea_path_4_24h','gfw_sea_path_24_48h'])
        unknown=f.track_evidence.eq('unresolved')
        if not (short.astype(int)+long.astype(int)+unknown.astype(int)).eq(1).all():
            raise ValueError('weather track categories are not exhaustive and exclusive')
        h=f.segment_weight_hours
        parts.append(pd.DataFrame({'candidate_id':f.candidate_id,'short_gap_supported_hours':h.where(short,0),
                                   'long_gap_4_48h_supported_hours':h.where(long,0),'unknown_hours':h.where(unknown,0)}).groupby('candidate_id').sum())
    f=pd.concat(parts).groupby(level=0).sum().reset_index()
    p=Path(out)/f'{shard}.parquet';f.to_parquet(p,index=False)
    return core.record(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','base','ports','weather','out','review']:
        p.add_argument('--'+name,type=lambda value:Path(value).resolve(),required=True)
    p.add_argument('--workers',type=int,default=12);args=p.parse_args()
    receipt=json.loads((args.run/'validation.json').read_text())
    if receipt['status']!='pass' or not all(receipt['checks'].values()):raise ValueError('validated release required')
    out=core.inside_runs(args.out);out.mkdir(parents=True,exist_ok=False);(out/'composition').mkdir()
    review=args.review.resolve()
    if not review.is_relative_to((CODING/'outputs').resolve()):raise ValueError('aggregate review only')
    review.mkdir(parents=True,exist_ok=False)
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(composition,s,args.weather,out/'composition'):s for s in core.SHARDS}
        for future in as_completed(jobs):
            try:
                results.append(future.result());print(json.dumps({'composition_shards':len(results),'shard':jobs[future]}),flush=True)
            except Exception:failures.append({'shard':jobs[future],'error':traceback.format_exc()})
    if failures:
        core.write_json(out/'validation.json',{'status':'failed','failures':failures});raise SystemExit(1)
    f=pd.read_parquet(args.run/'assembly/release_full_precision.parquet')
    v=pd.read_parquet(args.run/'assembly/voyage_intervals.parquet')
    comp=pd.concat([pd.read_parquet(x['path']) for x in results],ignore_index=True)
    if comp.candidate_id.duplicated().any() or set(comp.candidate_id)!=set(v.candidate_id):raise ValueError('composition candidate closure')
    # Preserve the independently calculated denominator beside the bound
    # panel's existing unknown_hours, then compare both before aggregation.
    comp=comp.rename(columns={'unknown_hours':'composition_unknown_hours'})
    v=v.merge(comp,on='candidate_id',validate='one_to_one')
    columns=['short_gap_supported_hours','long_gap_4_48h_supported_hours','composition_unknown_hours']
    if not np.allclose(v[columns].sum(axis=1),v.voyage_hours,rtol=0,atol=1e-8):raise ValueError('hour composition closure')
    for name in ['unknown_hours','unresolved_track_hours']:
        if not np.allclose(v.composition_unknown_hours,v[name],rtol=0,atol=1e-8):
            raise ValueError('unknown-hour composition differs: '+name)
    v=v.merge(f[core.KEY],on=core.KEY,how='inner',validate='many_to_one')
    months=core.module('segmented_month_coverage',HERE/'40_build_supplementary_fields.py').month_hours(v)
    month=months.groupby([*core.KEY,'month']).hours.sum().reset_index()
    covered=month.groupby(core.KEY).agg(active_months=('month','nunique'),month_hours=('hours','sum'),max_month_hours=('hours','max'))
    if not np.allclose(covered.month_hours.sort_index(),f.set_index(core.KEY).voyage_hours.sort_index(),rtol=0,atol=1e-7):raise ValueError('monthly full-time closure')
    covered['max_month_share_of_voyage_hours']=covered.max_month_hours/covered.month_hours
    covered['departure_months']=v.assign(month=v.departure_time.dt.month).groupby(core.KEY).month.nunique()
    covered=covered.drop(columns=['month_hours','max_month_hours'])
    comp=v.groupby(core.KEY)[columns].sum().rename(columns={'composition_unknown_hours':'unknown_hours'})
    total=comp.sum(axis=1)
    comp['share_hours_observed_or_between_gfw_fixes_le4h']=comp.short_gap_supported_hours/total
    comp['share_hours_supported_in_4_48h_gaps']=comp.long_gap_4_48h_supported_hours/total
    comp['share_hours_unknown']=comp.unknown_hours/total
    fields=f.set_index(core.KEY)[['voyages','voyage_hours','wave_bound_hours','wave_bound_coverage','wave_unboundable_hours','mean_swh_matched_m','mean_swh_constrained_m','mean_swh_constrained_shift_m','wave_mean_was_constrained']].join(covered).join(comp).reset_index()
    if len(fields)!=len(f):raise ValueError('supplement release closure')
    fields.to_csv(review/'record_coverage_fields.csv',index=False)
    months.groupby([*core.KEY,'month']).hours.sum().rename('voyage_hours').reset_index().to_csv(review/'record_monthly_hours.csv',index=False)
    # Invoke the existing functions with explicitly rebound corrected inputs.
    threshold=core.module('segmented_threshold_reference',HERE/'50_threshold_sensitivity.py')
    threshold.COMBINATIONS=args.run/'assembly/all_combinations.parquet'
    previous=sys.argv;sys.argv=['threshold','--out',str(out/'threshold')]
    try:threshold.main()
    finally:sys.argv=previous
    cities=core.module('segmented_city_reference',HERE/'45_build_city_reference.py')
    cities.BASE=args.base/'input_view';cities.CITY_TABLE=cities.BASE/'results/support/city_snap_table_2025.csv'
    cities.CROSSWALK=cities.BASE/'support_inputs/anchorage_to_analytical_city_corrected.csv'
    cities.VISITS=args.ports/'shards';cities.RELEASE=args.run/'assembly/release_full_precision.parquet'
    sys.argv=['cities','--out',str(out/'cities')]
    try:cities.main()
    finally:sys.argv=previous
    for folder in ['threshold','cities']:
        for source in (out/folder).glob('*.csv'):shutil.copy2(source,review/source.name)
    storm=CODING/'.runs/2026-10-08_2256_E007-supplementary-fields/ibtracs_2025_systems.csv'
    shutil.copy2(storm,review/storm.name)
    checks={'all64_composition_shards':len(results)==64,'full_hour_categories_close':True,'unknown_hours_match_release':True,
            'monthly_hours_close':True,'supplement_covers_every_release_record':True,'threshold_and_city_reference_refreshed':True,
            'frozen_storm_list_reused':True}
    summary={'status':'pass','checks':checks,'records':len(f),'released_voyages':len(v),
             'active_months_median':float(fields.active_months.median()),'active_months_p10':float(fields.active_months.quantile(.1)),
             'records_wave_bound_coverage_below90pct':int(fields.wave_bound_coverage.lt(.9).sum()),
             'code':[core.record(x) for x in [Path(__file__),HERE/'40_build_supplementary_fields.py',HERE/'45_build_city_reference.py',HERE/'50_threshold_sensitivity.py']],
             'release':core.record(args.run/'validation.json'),'storm_list_source':core.record(storm),
             'outputs':[core.record(x) for x in review.glob('*.csv')],'adopted':False,'network_requests':0}
    core.write_json(out/'validation.json',summary);core.write_json(review/'validation.json',summary)
    (review/'00_README.md').write_text('本目录的月份覆盖、轨迹来源时长、城市参考与门槛敏感性全部来自本轮修正后候选。先读上级00_README.md。轨迹比例使用天气小时窗的实际时长权重，三类小时合计为完整航次时长；短缺口指相邻GFW点间隔≤4h，长缺口为4–48h已解析部分，未解析小时单列。城市键沿用冻结分析划分，不等于商业泊位或行政市边界。IBTrACS清单复用未变的2025冻结源。字段单位、分母和空值含义见FIELDS.md；来源及检查见validation.json。\n')
    (review/'FIELDS.md').write_text('''# 补充字段说明

每行是有向分析城市对×船型，门槛≥5航次。`origin_city`、`dest_city`、`shiptype_group`与主表连接。

| 字段 | 单位与含义 |
|---|---|
| active_months | 1–12；有正航行时长的UTC月份数；跨月航次按实际时长拆分 |
| departure_months | 1–12；有出发航次的UTC月份数 |
| max_month_share_of_voyage_hours | 0–1；最多的一个月时长÷完整全年航次小时 |
| short_gap_supported_hours | h；观测标签或相邻GFW点间隔≤4h的已解析小时；不代表独立位置真值 |
| long_gap_4_48h_supported_hours | h；相邻GFW点间隔4–48h且解析成功的小时 |
| unknown_hours | h；未解析位置小时；与上述两类合计为voyage_hours |
| share_hours_observed_or_between_gfw_fixes_le4h | 短缺口支持小时÷voyage_hours |
| share_hours_supported_in_4_48h_gaps | 长缺口解析小时÷voyage_hours |
| share_hours_unknown | 未解析小时÷voyage_hours |
| wave_bound_hours / wave_bound_coverage | 浪高界限支持小时及其占voyage_hours的比例；空浪高不能补零 |
| wave_unboundable_hours | h；全程小时扣除界限支持小时 |
| mean_swh_matched_m | m；已匹配小时的主平均浪高；空表示无可估主值 |
| mean_swh_constrained_m / mean_swh_constrained_shift_m | m；旧截断定义作为次要对照及其相对主值的差 |

其他主表字段见上级release/字段说明。`record_monthly_hours.csv`中的month为UTC月份1–12，voyage_hours为该月时长。门槛表中的kept/dropped分别为达到/未达到相应minimum_voyages的组合；比例分母为本轮全部保留分析航次或小时。city_reference坐标为本轮GFW到访重算的参考位置（度）；city_anchorages中的到访数包括高/低置信访问，dock_flag沿用冻结标记，不能当每次到访实际靠泊。IBTrACS清单按2025内实际时间纳入，不按SEASON单独排除跨年系统。
''')
    print(json.dumps({k:summary[k] for k in ['status','records','released_voyages','active_months_median','records_wave_bound_coverage_below90pct']},indent=2))


if __name__=='__main__':main()
