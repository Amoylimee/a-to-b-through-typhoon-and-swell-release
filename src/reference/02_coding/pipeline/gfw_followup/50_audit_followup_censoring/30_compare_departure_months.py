"""Repeat the October 22 monthly comparison with explicit, immutable input paths."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('followup_censoring_comparison',HERE/'10_audit_followup_censoring.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def summarize(root):
    p=pd.read_parquet(root/'preparation/all_voyage_panel.parquet')
    p['departure_month']=p.departure_time.dt.strftime('%Y-%m')
    duration=(p.arrival_time-p.departure_time).dt.total_seconds()/3600
    if not np.array_equal(duration,p.voyage_hours):raise ValueError('duration and voyage hours differ')
    if p.candidate_id.duplicated().any():raise ValueError('duplicate voyage')
    s=p.groupby('departure_month').agg(voyages=('candidate_id','size'),distinct_ships=('vessel_uid_clean','nunique'),
        voyage_hours=('voyage_hours','sum'),mean_hours=('voyage_hours','mean'),median_hours=('voyage_hours','median'),
        p90_hours=('voyage_hours',lambda x:x.quantile(.9)),p99_hours=('voyage_hours',lambda x:x.quantile(.99)),
        track_hours=('track_hours','sum'),wave_matched_hours=('wave_matched_hours','sum'),
        wave_integral_m_hours=('wave_integral_m_hours','sum'),wave_ge4m_matched_hours=('wave_ge4m_matched_hours','sum'))
    s['days']=pd.to_datetime(s.index+'-01').days_in_month
    s['voyages_per_day']=s.voyages/s.days;s['voyage_hours_per_day']=s.voyage_hours/s.days
    s['position_coverage_pct']=s.track_hours/s.voyage_hours*100
    s['wave_coverage_pct']=s.wave_matched_hours/s.voyage_hours*100
    s['mean_swh_matched_m']=s.wave_integral_m_hours/s.wave_matched_hours
    s['high_wave_pct']=s.wave_ge4m_matched_hours/s.wave_matched_hours*100
    for d in [30,45,60]:
        g=p[p.voyage_hours.gt(d*24)].groupby('departure_month')
        s[f'over_{d}d_voyage_share']=g.size().reindex(s.index,fill_value=0)/s.voyages
        s[f'over_{d}d_hour_share']=g.voyage_hours.sum().reindex(s.index,fill_value=0)/s.voyage_hours
    table=pd.read_parquet(root/'private/country_full_precision.parquet')
    additive=['voyages','voyage_hours','track_hours','wave_matched_hours','wave_ge4m_matched_hours']
    totals=table.groupby('departure_month')[additive].sum()
    if not np.array_equal(totals.to_numpy(),s[additive].to_numpy()):raise ValueError('country and voyage monthly totals differ')
    for r in [300,500]:
        for end in ['at_least','at_most']:
            col=f'tc{r}_hours_{end}'
            s[f'tc{r}_{end}_pct']=table.groupby('departure_month')[col].sum()/s.voyage_hours*100
    return s


def run(release,baseline,out):
    out.mkdir(parents=True,exist_ok=False)
    before=summarize(baseline);after=summarize(release)
    before.to_csv(out/'baseline_monthly.csv');after.to_csv(out/'current_monthly.csv')
    paired=before.add_suffix('_baseline').join(after.add_suffix('_current'))
    for c in ['voyages','voyage_hours','voyages_per_day','voyage_hours_per_day']:
        paired[c+'_change']=after[c]-before[c]
        paired[c+'_change_pct']=(after[c]/before[c]-1)*100
    paired.to_csv(out/'monthly_comparison.csv')
    receipt={'status':'pass','baseline':str(baseline),'current':str(release),
             'checks':{'country_vs_voyage_monthly_exact':True,'duration_exact':True,'candidate_ids_unique':True},
             'source_sha256':{str(root/'preparation/all_voyage_panel.parquet'):audit.sha(root/'preparation/all_voyage_panel.parquet') for root in [baseline,release]},
             'interpretation':'Descriptive sums and distributions only; compare with follow-up censoring audit before judging completeness.'}
    (out/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
    lines=['# E009：月度对比复算','','本表按 UTC 出发月计入完整航程，使用逐航次全精度数据和按小时加权的覆盖率；与全精度国家表加总逐值一致。完整核算见 monthly_comparison.csv。月际量级相似不能证明完整性，应结合后续观察截断审计。','',
           '| 出发月 | 原日均小时 | 新日均小时 | 原 >30 天小时占比 | 新 >30 天小时占比 |','|---|---:|---:|---:|---:|']
    for month in ['2025-10','2025-11','2025-12']:
        a=before.loc[month];b=after.loc[month]
        lines.append(f'| {month} | {a.voyage_hours_per_day:,.2f} | {b.voyage_hours_per_day:,.2f} | {a.over_30d_hour_share:.4%} | {b.over_30d_hour_share:.4%} |')
    (out/'00_README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['release','baseline-release','out']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();run(a.release.resolve(),a.baseline_release.resolve(),a.out.resolve())
