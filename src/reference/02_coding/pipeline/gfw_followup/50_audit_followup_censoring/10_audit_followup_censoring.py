"""Descriptive follow-up censoring audit; never modifies or imputes voyages."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
CODING = HERE.parents[3]
SHIP = 'vessel_uid_clean'
REF = [f'2025-{m:02d}' for m in range(1, 9)]
BIN_LABELS = ['<2d', '2-5d', '5-10d', '10-20d', '20-30d', '>30d']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(panel, cutoff):
    p = panel.copy()
    for col in ['departure_time', 'arrival_time']:
        p[col] = pd.to_datetime(p[col], utc=True)
    if p.candidate_id.duplicated().any() or p[SHIP].isna().any():
        raise ValueError('duplicate voyage or absent ship identity')
    if not p.departure_time.dt.year.eq(2025).all() or not p.arrival_time.lt(cutoff).all():
        raise ValueError('panel outside 2025 departure / observation cutoff scope')
    duration = (p.arrival_time - p.departure_time).dt.total_seconds() / 3600
    if not np.array_equal(duration, p.voyage_hours) or duration.le(0).any():
        raise ValueError('voyage duration disagrees with timestamps')
    p = p.sort_values([SHIP, 'departure_time', 'arrival_time', 'candidate_id'], kind='stable')
    nxt = p.groupby(SHIP, sort=False)[['departure_time', 'origin_city']].shift(-1)
    stay = (nxt.departure_time - p.arrival_time).dt.total_seconds() / 3600
    available = nxt.origin_city.eq(p.dest_city) & stay.notna()
    if (available & stay.lt(0)).any():
        raise ValueError('matching consecutive voyages overlap')
    p['stay_proxy_available'] = available
    p['stay_proxy_hours'] = stay.where(available, 0.0)
    p['arrival_plus_stay'] = p.arrival_time + pd.to_timedelta(p.stay_proxy_hours, unit='h')
    p['departure_month'] = p.departure_time.dt.strftime('%Y-%m')
    return p


def monthly(p):
    rows = []
    for month, g in p.groupby('departure_month', sort=True):
        days = pd.Timestamp(month + '-01').days_in_month
        row = {'departure_month': month, 'voyages': len(g), 'voyage_hours': g.voyage_hours.sum(),
               'days': days, 'voyages_per_day': len(g)/days,
               'voyage_hours_per_day': g.voyage_hours.sum()/days,
               'mean_voyage_hours': g.voyage_hours.mean(),
               'stay_proxy_available_share': g.stay_proxy_available.mean()}
        for d in [30, 45, 60]:
            selected = g.voyage_hours.gt(d*24)
            row[f'over_{d}d_voyages'] = int(selected.sum())
            row[f'over_{d}d_voyage_share'] = selected.mean()
            row[f'over_{d}d_hours'] = g.loc[selected, 'voyage_hours'].sum()
            row[f'over_{d}d_hour_share'] = row[f'over_{d}d_hours']/row['voyage_hours']
        rows.append(row)
    return pd.DataFrame(rows)


def pseudo_censor(p):
    rows = []
    for month in REF:
        g = p.loc[p.departure_month.eq(month)]
        if not len(g):
            raise ValueError(f'reference month missing: {month}')
        for k in [2, 3, 4]:
            cutoff = pd.Timestamp(month + '-01', tz='UTC') + pd.DateOffset(months=k)
            for rule, col in [('arrival', 'arrival_time'), ('arrival_plus_stay', 'arrival_plus_stay')]:
                lost = g[col].ge(cutoff)
                rows.append({'reference_month':month, 'followup_months_k':k, 'rule':rule,
                             'pseudo_cutoff':cutoff.isoformat(), 'reference_voyages':len(g),
                             'reference_hours':g.voyage_hours.sum(), 'lost_voyages':int(lost.sum()),
                             'lost_hours':g.loc[lost,'voyage_hours'].sum(),
                             'voyage_loss_rate':lost.mean(),
                             'hour_loss_rate':g.loc[lost,'voyage_hours'].sum()/g.voyage_hours.sum()})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(['followup_months_k','rule'],as_index=False).agg(
        voyage_loss_mean=('voyage_loss_rate','mean'), voyage_loss_min=('voyage_loss_rate','min'),
        voyage_loss_max=('voyage_loss_rate','max'), hour_loss_mean=('hour_loss_rate','mean'),
        hour_loss_min=('hour_loss_rate','min'), hour_loss_max=('hour_loss_rate','max'))
    return detail, summary


def route_ratios(p, keys):
    reference = p[p.departure_month.isin(REF)]
    medians = reference.groupby(keys, observed=True).voyage_hours.median().rename('reference_median_hours').reset_index()
    # [0,2), [2,5), [5,10), [10,20), [20,30], (30,infinity), in days.
    d = medians.reference_median_hours/24
    medians['route_duration_bin'] = np.select([d.lt(2), d.lt(5), d.lt(10), d.lt(20), d.le(30)], BIN_LABELS[:-1], default=BIN_LABELS[-1])
    labeled = p.merge(medians, on=keys, how='left', validate='many_to_one')
    labeled['route_duration_bin'] = labeled.route_duration_bin.fillna('no_reference_route')
    counts = pd.crosstab(labeled.departure_month, labeled.route_duration_bin).reindex(
        index=sorted(p.departure_month.unique()), columns=BIN_LABELS+['no_reference_route'], fill_value=0)
    reference_days = sum(pd.Timestamp(m+'-01').days_in_month for m in REF)
    reference_daily = counts.loc[REF].sum()/reference_days
    rows=[]
    for month in counts.index:
        for label in counts.columns:
            n=int(counts.loc[month,label]); daily=n/pd.Timestamp(month+'-01').days_in_month
            denom=reference_daily[label]
            rows.append({'departure_month':month,'route_duration_bin':label,'voyages':n,
                         'voyages_per_day':daily,'reference_voyages_per_day':denom,
                         'daily_departure_ratio':daily/denom if denom else None})
    return pd.DataFrame(rows), medians


def audit(panel_path, cutoff, output, assert_baseline=False):
    output.mkdir(parents=True, exist_ok=False)
    source_sha = sha(panel_path)
    p = prepare(pd.read_parquet(panel_path), cutoff)
    m = monthly(p); detail, pseudo = pseudo_censor(p)
    stay=p.loc[p.stay_proxy_available,'stay_proxy_hours']
    proxy={'available_count':int(p.stay_proxy_available.sum()),'total_voyages':len(p),
           'available_share':float(p.stay_proxy_available.mean()),
           'quantiles_available_hours':{str(q):float(stay.quantile(q)) for q in [0,.25,.5,.75,.9,.95,.99,1]},
           'unavailable_assigned_hours':0,'same_ship_next_retained_voyage_same_city_required':True}
    m.to_csv(output/'monthly_metrics.csv',index=False)
    detail.to_csv(output/'pseudo_censoring_by_reference_month.csv',index=False)
    pseudo.to_csv(output/'pseudo_censoring_summary.csv',index=False)
    for name, keys in [('route',['origin_city','dest_city']),('route_shiptype',['origin_city','dest_city','shiptype_group'])]:
        ratios, medians=route_ratios(p,keys)
        ratios.to_csv(output/f'{name}_duration_bin_ratios.csv',index=False)
        medians.to_parquet(output/f'{name}_reference_medians_PRIVATE.parquet',index=False)
    type_count=p.groupby(['departure_month','shiptype_group']).size().rename('voyages').reset_index()
    type_ref=p[p.departure_month.isin(REF)].groupby('shiptype_group').size()/243
    type_count['voyages_per_day']=type_count.voyages/pd.to_datetime(type_count.departure_month+'-01').dt.days_in_month
    type_count['reference_voyages_per_day']=type_count.shiptype_group.map(type_ref)
    type_count['daily_departure_ratio']=type_count.voyages_per_day/type_count.reference_voyages_per_day
    type_count.to_csv(output/'shiptype_daily_ratios.csv',index=False)
    implied=[]
    for month in ['2025-10','2025-11','2025-12']:
        k=(cutoff.year-2025)*12+cutoff.month-int(month[-2:])
        observed=m.set_index('departure_month').loc[month]
        for _, row in pseudo[pseudo.followup_months_k.eq(k)].iterrows():
            implied.append({'departure_month':month,'followup_months_k':k,'rule':row.rule,
                            'observed_voyages':int(observed.voyages),'observed_hours':observed.voyage_hours,
                            'mean_voyage_loss_rate':row.voyage_loss_mean,'mean_hour_loss_rate':row.hour_loss_mean,
                            'diagnostic_implied_missing_voyages':observed.voyages*row.voyage_loss_mean/(1-row.voyage_loss_mean),
                            'diagnostic_implied_missing_hours':observed.voyage_hours*row.hour_loss_mean/(1-row.hour_loss_mean)})
    pd.DataFrame(implied).to_csv(output/'diagnostic_implied_gaps_NO_IMPUTATION.csv',index=False)
    ps=pseudo.set_index(['followup_months_k','rule']); dec=m.set_index('departure_month').loc['2025-12']
    baseline_values={
        'stay_available_share':(proxy['available_share'],.8629,5e-5),
        'stay_median_hours':(proxy['quantiles_available_hours']['0.5'],28.,1e-9),
        'k2_arrival_voyage_loss':(ps.loc[(2,'arrival'),'voyage_loss_mean'],.0103,5e-5),
        'k2_arrival_hour_loss':(ps.loc[(2,'arrival'),'hour_loss_mean'],.1025,5e-5),
        'k2_arrival_hour_loss_min':(ps.loc[(2,'arrival'),'hour_loss_min'],.0949,5e-5),
        'k2_arrival_hour_loss_max':(ps.loc[(2,'arrival'),'hour_loss_max'],.1266,5e-5),
        'k2_stay_voyage_loss':(ps.loc[(2,'arrival_plus_stay'),'voyage_loss_mean'],.0183,5e-5),
        'k2_stay_hour_loss':(ps.loc[(2,'arrival_plus_stay'),'hour_loss_mean'],.1326,5e-5),
        'k2_stay_hour_loss_min':(ps.loc[(2,'arrival_plus_stay'),'hour_loss_min'],.1235,5e-5),
        'k2_stay_hour_loss_max':(ps.loc[(2,'arrival_plus_stay'),'hour_loss_max'],.1608,5e-5),
        'k3_stay_hour_loss':(ps.loc[(3,'arrival_plus_stay'),'hour_loss_mean'],.0331,5e-5),
        'k4_stay_hour_loss':(ps.loc[(4,'arrival_plus_stay'),'hour_loss_mean'],.0138,5e-5),
        'dec_over30_hour_share':(dec.over_30d_hour_share,.1153,5e-5),
        'dec_hours_per_day':(dec.voyage_hours_per_day,399038.74,.005)}
    checks={key:{'actual':float(a),'expected':e,'tolerance':tol,'pass':bool(abs(a-e)<=tol)} for key,(a,e,tol) in baseline_values.items()} if assert_baseline else {}
    if assert_baseline and cutoff!=pd.Timestamp('2026-02-01',tz='UTC'):
        raise ValueError('January baseline check requires February 1 cutoff')
    passed=all(c['pass'] for c in checks.values())
    result={'status':'pass' if passed else 'failed','observation_cutoff_exclusive':cutoff.isoformat(),
            'source_panel':str(panel_path),'source_sha256':source_sha,'script_sha256':sha(HERE),
            'source_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
            'source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'stay_proxy':proxy,'january_baseline_checks':checks,'reference_months':REF,
            'diagnostic_only':True,'imputation_performed':False,
            'limits':'Reference-month analogues are diagnostic, not measured missing voyages or causal identification. Unavailable stay is zero. Route composition is not controlled beyond reported strata.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# E009 后续观察截断审计','',f"状态：{result['status']}；UTC 观察截止（不含）：{cutoff.isoformat()}。",
          '', '先读本文件及 summary.json；拟截断逐月值和均值在 pseudo_censoring_*.csv；实际月度分布见 monthly_metrics.csv。',
          '', '诊断缺口只按观测量 × 损失率 / (1 − 损失率) 换算，绝不写回航次、天气或月度表。参考 1–8 月的损失率不是年末真实缺失率的直接测量。',
          '',f"停留代理可得 {proxy['available_share']:.8f}，可得部分中位数 {stay.median():.2f} h；其余保守取零。",'',
          '| k | 规则 | 航次损失均值 | 小时损失均值 | 小时损失范围 |','|---:|---|---:|---:|---:|']
    for _, row in pseudo.iterrows():
        text.append(f'| {row.followup_months_k} | {row.rule} | {row.voyage_loss_mean:.6f} | {row.hour_loss_mean:.6f} | {row.hour_loss_min:.6f}–{row.hour_loss_max:.6f} |')
    text+=['','航线为有方向的城市对；另出城市对×船型分箱，以及船型单独的日均比率。参考日均量用 1–8 月总数除以 243 天，未见参考航线单列。分箱边界：<2、[2,5)、[5,10)、[10,20)、[20,30]、>30 天。',
           '', '同一船按离港、到港、候选 ID 稳定排序，仅紧接的下一条保留航次起点与当前终点城市相同才有停留代理。不搜索更远的替代航次。',
           '', '1 月基线检查：'+('全部通过。' if passed and assert_baseline else '未请求。' if not assert_baseline else '失败；后续获取与重建必须停止。'),
           '',f"来源分支 `{result['source_branch']}`，提交 `{result['source_head']}`。原输入与既有运行目录未改。"]
    (output/'00_README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'status':result['status'],'output':str(output),'stay_proxy':proxy,'checks':checks},indent=2))
    if not passed:
        raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--observation-cutoff',required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--assert-january-baseline',action='store_true')
    a=parser.parse_args()
    cutoff=pd.Timestamp(a.observation_cutoff)
    cutoff=cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
    if cutoff.day!=1 or cutoff.hour or cutoff.minute or cutoff.second:
        raise ValueError('observation cutoff must be UTC month start')
    output=a.output or CODING/'.runs'/f"{datetime.now(ZoneInfo('Asia/Hong_Kong')):%Y-%m-%d_%H%M%S}_E009-followup-censoring-audit"
    audit(a.panel.resolve(),cutoff,output.resolve(),a.assert_january_baseline)
