#!/usr/bin/env python3
"""Rebind the frozen 122 reference hours to corrected voyages without resampling.

Freeze membership by vessel and UTC time before looking at new weather or
bounds. Report every excluded/now-known hour. The old sampling strata are
historical labels, not weights or a representative design for the new sample.
Dense reference AIS observations remain validation-only; reuse the existing evaluator.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa

HERE=Path(__file__).resolve().parent
CODING=HERE.parents[1]
sys.path.insert(0,str(CODING/'pipeline/gfw_release'))
import step6_core as core
REAL=core.module('segmented_real_reference_evaluator',HERE/'140_validate_real_unknown_intervals.py')
FROZEN=CODING/'.runs/2026-10-11_1143_E007-real-unknown-interval-manifest/samples.parquet'
BEFORE=CODING/'.runs/2026-10-11_1238_E007-wave-support-release/preparation/voyage_panel.parquet'


def prepare(base,out):
    samples=pd.read_parquet(FROZEN)
    if len(samples)!=122 or samples.sample_id.duplicated().any():
        raise ValueError('original frozen reference manifest changed')
    old=pd.read_parquet(BEFORE,columns=['candidate_id',core.SHIP],
                        filters=[('candidate_id','in',samples.candidate_id.tolist())])
    samples=samples.merge(old,on='candidate_id',validate='many_to_one')
    if len(samples)!=122 or samples[core.SHIP].isna().any():raise ValueError('old identities do not close')
    current=pd.read_parquet(base/'retained_panel.parquet',
                           columns=['candidate_id',core.SHIP,*core.KEY,'source_shard','departure_time','arrival_time','weather_source'])
    by_ship={ship:g for ship,g in current.loc[current[core.SHIP].isin(samples[core.SHIP])].groupby(core.SHIP,sort=False)}
    rows=[]
    for item in samples.to_dict('records'):
        group=by_ship.get(item[core.SHIP],current.iloc[:0])
        event=pd.Timestamp(item['tau'],tz='UTC')
        selected=group.loc[group.departure_time.lt(event)&group.arrival_time.gt(event)]
        row={**item,'old_candidate_id':item['candidate_id'],'membership_matches':len(selected),
             'membership_disposition':'unique_retained_voyage' if len(selected)==1 else
                                      'no_retained_voyage' if len(selected)==0 else 'ambiguous_retained_voyages'}
        if len(selected)==1:
            new=selected.iloc[0]
            if new.shiptype_group!=item['shiptype_group'] or new.source_shard!=item['source_shard']:
                raise ValueError('frozen vessel label or shard changed')
            row.update({k:new[k] for k in ['candidate_id',*core.KEY,'source_shard','departure_time','arrival_time','weather_source']})
        rows.append(row)
    mapping=pd.DataFrame(rows)
    mapping.to_parquet(out/'membership_before_weather.parquet',index=False)
    mapping.loc[mapping.membership_disposition.eq('unique_retained_voyage')].to_parquet(out/'mapped_samples.parquet',index=False)
    report={'status':'frozen','original_samples':122,'membership_dispositions':mapping.membership_disposition.value_counts().to_dict(),
            'selection_uses_weather_or_bounds':False,'resampled':False,'independent_annual_accuracy':False,
            'source':[core.record(FROZEN),core.record(BEFORE),core.record(base/'retained_panel.parquet')],
            'outputs':[core.record(out/x) for x in ['membership_before_weather.parquet','mapped_samples.parquet']],
            'code':core.record(Path(__file__))}
    core.write_json(out/'manifest.json',report)
    print(json.dumps({k:report[k] for k in ['status','original_samples','membership_dispositions']}),flush=True)


def evaluate_worker(shard,samples,prepared,weather,out):
    pa.set_cpu_count(1)
    return REAL.evaluate_shard(shard,str(samples),str(prepared),str(weather),str(out),True)


def evaluate(args,out):
    frozen=json.loads((args.manifest/'manifest.json').read_text())
    if frozen['status']!='frozen':raise ValueError('membership must be frozen before evaluation')
    for item in [*frozen['source'],*frozen['outputs'],frozen['code']]:
        if core.record(Path(item['path']))!=item:raise ValueError('frozen membership input changed')
    source=core.binding()
    if source['weather_run'].resolve()!=args.weather.resolve():raise ValueError('wrong corrected weather source')
    prep=json.loads((args.prepared/'preparation.json').read_text())
    if prep['annual_weather']!=source['annual']:raise ValueError('preparation and corrected weather differ')
    mapping=pd.read_parquet(args.manifest/'membership_before_weather.parquet')
    mapped=pd.read_parquet(args.manifest/'mapped_samples.parquet')
    decisions=[]
    for shard,samples in mapped.groupby('source_shard',sort=True):
        observed=pd.concat([pd.read_parquet(p,columns=['candidate_id','position_time_utc','segment_weight_hours','track_evidence','lat','lon'],
                               filters=[('candidate_id','in',samples.candidate_id.tolist())])
                            for p in sorted((args.weather/'shards'/shard).glob('weather_*.parquet'))],ignore_index=True)
        observed['tau']=observed.position_time_utc.astype('int64')
        selected=samples.merge(observed,on=['candidate_id','tau'],how='left',validate='many_to_one',indicator=True)
        for item in selected.to_dict('records'):
            state='no_weather_hour' if item['_merge']!='both' else \
                  'partial_weather_hour' if item['segment_weight_hours']!=1 else \
                  'eligible_real_unknown' if item['track_evidence']=='unresolved' and pd.isna(item['lat']) and pd.isna(item['lon']) else \
                  'now_position_supported'
            decisions.append({'sample_id':item['sample_id'],'weather_disposition':state})
    dispositions=mapping.merge(pd.DataFrame(decisions),on='sample_id',how='left',validate='one_to_one')
    dispositions['disposition']=np.where(dispositions.membership_disposition.eq('unique_retained_voyage'),
                                         dispositions.weather_disposition,dispositions.membership_disposition)
    if dispositions.disposition.isna().any() or len(dispositions)!=122:raise ValueError('all original reference hours must be accounted for')
    dispositions.to_parquet(out/'sample_dispositions.parquet',index=False)
    eligible=mapped.loc[mapped.sample_id.isin(dispositions.loc[dispositions.disposition.eq('eligible_real_unknown'),'sample_id'])].copy()
    if eligible.empty:raise ValueError('no frozen reference hour remains an evaluable unknown')
    eligible.to_parquet(out/'samples.parquet',index=False)
    code=[core.record(x) for x in [Path(__file__),HERE/'140_validate_real_unknown_intervals.py',HERE/'interval_validation_core.py',
                                   HERE/'wave_support_bounds.py',HERE/'config/wave_support_revision.json']]
    state={'state':'running','original_samples':122,'eligible_samples':len(eligible),'workers':args.workers,
           'dispositions':dispositions.disposition.value_counts().to_dict(),'membership_manifest':core.record(args.manifest/'manifest.json'),
           'eligible_manifest':core.record(out/'samples.parquet'),'preparation':core.record(args.prepared/'preparation.json'),
           'corrected_weather_validation':source['validation'],'code':code,'completed':[],'failures':[],
           'old_strata_not_reweighted':True,'annual_accuracy_estimated':False,'network_requests':0,'adopted':False}
    core.write_json(out/'execution.json',state)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(evaluate_worker,s,out/'samples.parquet',args.prepared,args.weather,out):s
              for s in sorted(eligible.source_shard.unique())}
        for future in as_completed(jobs):
            try:
                future.result();state['completed'].append(jobs[future])
                print(json.dumps({'reference_shards_passed':len(state['completed']),'shard':jobs[future]}),flush=True)
            except Exception:state['failures'].append({'shard':jobs[future],'error':traceback.format_exc()})
            core.write_json(out/'execution.json',state)
    if state['failures']:raise RuntimeError(state['failures'])
    results=pd.concat([pd.read_parquet(p) for p in sorted((out/'shards').glob('*/sample_results.parquet'))],ignore_index=True)
    if results.sample_id.duplicated().any() or set(results.sample_id)!=set(eligible.sample_id):raise ValueError('reference-result identity closure')
    results.to_parquet(out/'sample_results.parquet',index=False)
    summaries=[]
    for population,block in [('all',results),*list(results.groupby('sample_set',sort=True))]:
        row={'population':population,'samples':len(block)}
        for metric in ['wave','tc300','tc500']:
            valid=block[f'{metric}_inside'].notna()
            row[f'{metric}_evaluable']=int(valid.sum())
            row[f'{metric}_inside']=int(block.loc[valid,f'{metric}_inside'].astype(bool).sum())
            row[f'{metric}_outside']=int(valid.sum())-row[f'{metric}_inside']
            row[f'{metric}_unevaluable']=int((~valid).sum())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'inclusion_summary.csv',index=False)
    dispositions.groupby(['sample_set','disposition'],dropna=False).size().rename('samples').reset_index().to_csv(out/'reference_disposition_summary.csv',index=False)
    for item in code:
        if core.record(Path(item['path']))!=item:raise ValueError('reference evaluator changed during execution')
    checks={'all_122_dispositions_retained':len(dispositions)==122,'membership_frozen_before_weather':True,
            'result_identity_closure':True,'reference_exact_utc_hour':bool(eligible.tau.mod(core.NS_HOUR).eq(0).all()),
            'known_or_partial_hours_not_mislabelled_unknown':True,'all_evaluator_shards_passed':True}
    overall=summaries[0]
    state.update(state='completed',status='pass',checks=checks,summary=summaries,
                 source_conditional_inclusion_pass=all(overall[f'{m}_outside']==0 for m in ['wave','tc300','tc500']),
                 outputs=[core.record(out/x) for x in ['sample_dispositions.parquet','samples.parquet','sample_results.parquet','inclusion_summary.csv','reference_disposition_summary.csv']])
    core.write_json(out/'validation.json',state);core.write_json(out/'execution.json',state)
    print(json.dumps({k:state[k] for k in ['status','dispositions','summary','source_conditional_inclusion_pass']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','evaluate'])
    p.add_argument('--base',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--prepared',type=Path)
    p.add_argument('--weather',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.workers<=8:raise ValueError('reference workers must be 1..8')
    if args.stage=='prepare' and args.base is None:raise ValueError('need corrected candidate base')
    if args.stage=='evaluate' and any(x is None for x in [args.manifest,args.prepared,args.weather]):raise ValueError('need frozen membership, preparation and weather')
    out=core.inside_runs(args.out);out.mkdir(parents=True,exist_ok=False)
    if args.stage=='prepare':prepare(args.base.resolve(),out)
    else:evaluate(args,out)


if __name__=='__main__':main()
