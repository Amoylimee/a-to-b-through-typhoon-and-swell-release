#!/usr/bin/env python3
"""Direct exposure-interval diagnostics using frozen production kernels and local references.

Prepare and freeze the manifest before evaluation. No interior position or
reference weather is supplied to the bound kernel. reference AIS remains validation-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CODING=HERE.parents[1]
E006=CODING/'pipeline/gfw_release'
sys.path.insert(0,str(E006))
import step6_core as core
from interval_validation_core import exposure_comparison

DESIGN=HERE/'config/exposure_interval_diagnostic.json'
PAIRS=CODING/'.runs/2026-10-13_2006_E007-one-cell-real-gap-weather'
CANDIDATES=CODING/'.runs/2026-10-13_2006_E007-one-cell-real-gap-comparison'
# Default to the adopted population; an explicit environment selects another frozen run.
PAIRS=Path(os.environ.get('E007_REAL_GAP_WEATHER',PAIRS))
CANDIDATES=Path(os.environ.get('E007_REAL_GAP_COMPARISON',CANDIDATES))
N=core.NS_HOUR


def rank(*parts):
    return hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()


def read_observed(shard, ids, voyage_set):
    state=json.loads((Path(voyage_set)/'shards'/shard/'status.json').read_text())
    track=state['sources']['track']
    if core.record(Path(track['path'])) != track:
        raise ValueError(f'{shard}: track checksum changed')
    f=pd.read_parquet(track['path'],columns=['candidate_id','observation_time_utc','lat','lon','source_provider'],
        filters=[('track_evidence','=','observed_ais'),('candidate_id','in',list(ids))])
    if not f.source_provider.fillna('').str.startswith('gfw').all() or not f.observation_time_utc.dt.minute.eq(30).all():
        raise ValueError('manifest reference is not observed GFW at hh:30')
    f=f.sort_values(['candidate_id','observation_time_utc']).drop_duplicates(['candidate_id','observation_time_utc'])
    return f,track


def long_pool(shard, pool, voyage_set, design):
    f,track=read_observed(shard,pool.candidate_id.unique(),voyage_set)
    by_id={str(cid): g for cid,g in f.groupby('candidate_id',observed=True,sort=False)}
    rows=[]
    for item in pool.to_dict('records'):
        obs=by_id.get(str(item['candidate_id']))
        if obs is None: continue
        times=obs.observation_time_utc.astype('int64').to_numpy()
        for duration in (36,48):
            # One-hour actual-observation context on each side. Endpoints and
            # every intervening observation must be exactly one hour apart.
            eligible=[]
            for left in range(1,len(times)-duration-1):
                right=left+duration
                if np.all(np.diff(times[left-1:right+2])==N):
                    if item['sample_set']=='tc500km_additional' and not (
                        times[left]<=pd.Timestamp(item['gap_start_utc']).value and
                        times[right]>=pd.Timestamp(item['gap_end_utc']).value): continue
                    eligible.append(left)
            if not eligible: continue
            h=rank(design['seed'],item['candidate_id'],duration,item['sample_set'],'window')
            left=eligible[int(h[:16],16)%len(eligible)]; right=left+duration
            row={k:item[k] for k in ['candidate_id','source_shard','shiptype_group','sample_set']}
            row.update(nominal_gap_hours=duration,actual_gap_hours=float(duration),
                gap_start_utc=pd.Timestamp(times[left],tz='UTC').isoformat(),
                gap_end_utc=pd.Timestamp(times[right],tz='UTC').isoformat(),
                truth_observation_count=duration-1)
            rows.append(row)
    return rows,track


def select_manifest(args,design):
    original=pd.read_parquet(args.samples)
    original['diagnostic_rank']=[rank(design['seed'],x) for x in original.sample_id]
    main=original[original.sample_set.eq('gap_length_x_shiptype')].sort_values('diagnostic_rank')
    selected=main.groupby(['nominal_gap_hours','shiptype_group'],sort=True).head(design['windows_per_main_stratum'])
    storm=original[original.sample_set.eq('tc500km_additional')].sort_values('diagnostic_rank')
    selected=pd.concat([selected,storm.groupby('nominal_gap_hours').head(design['storm_windows_per_duration'])])
    pool=pd.concat([main[main.nominal_gap_hours.eq(24)].groupby('shiptype_group').head(design['long_window_candidate_pool_per_shiptype']),
                    storm[storm.nominal_gap_hours.eq(24)].head(design['long_window_storm_candidate_pool'])])
    long_rows=[]; receipts=[]; failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs={executor.submit(long_pool,s,g,str(args.voyage_set),design):s for s,g in pool.groupby('source_shard')}
        for future in as_completed(jobs):
            try:
                rows,receipt=future.result(); long_rows.extend(rows); receipts.append(receipt)
            except Exception: failures.append({'shard':jobs[future],'traceback':traceback.format_exc()})
    if failures:
        core.write_json(args.out/'preparation_failures.json',failures); raise RuntimeError(failures)
    long=pd.DataFrame(long_rows)
    long['diagnostic_rank']=[rank(design['seed'],r['candidate_id'],r['nominal_gap_hours'],r['sample_set']) for r in long_rows]
    more_main=long[long.sample_set.eq('gap_length_x_shiptype')].sort_values('diagnostic_rank').groupby(['nominal_gap_hours','shiptype_group']).head(design['windows_per_main_stratum'])
    more_storm=long[long.sample_set.eq('tc500km_additional')].sort_values('diagnostic_rank').groupby('nominal_gap_hours').head(design['storm_windows_per_duration'])
    selected=pd.concat([selected,more_main,more_storm],ignore_index=True)
    selected['source_sample_id']=selected.get('sample_id')
    selected['sample_id']=[rank(design['seed'],r['candidate_id'],r['sample_set'],r['nominal_gap_hours'],r['gap_start_utc'])[:24] for r in selected.to_dict('records')]
    selected=selected.sort_values('sample_id').drop(columns='diagnostic_rank')
    if selected.sample_id.duplicated().any(): raise ValueError('duplicate frozen windows')
    selected.to_parquet(args.out/'samples.parquet',index=False)
    return {'selected_windows':len(selected),'long_pool_selected':len(long),
            'manifest':core.record(args.out/'samples.parquet'),'long_pool_track_receipts':receipts,
            'counts':selected.groupby(['sample_set','nominal_gap_hours','shiptype_group']).size().rename('windows').reset_index().to_dict('records')}


def evaluate_shard(shard, manifest, voyage_set, out, wave_support_revision=False):
    started=time.perf_counter(); folder=Path(out)/'shards'/shard
    samples=pd.read_parquet(manifest,filters=[('source_shard','=',shard)])
    ids=samples.candidate_id.unique().tolist()
    observed,track=read_observed(shard,ids,voyage_set)
    groups={str(cid):g.set_index('observation_time_utc') for cid,g in observed.groupby('candidate_id',observed=True,sort=False)}
    plans=[]; info={}
    for item in samples.to_dict('records'):
        start=pd.Timestamp(item['gap_start_utc']); end=pd.Timestamp(item['gap_end_utc'])
        raw=groups[str(item['candidate_id'])]
        between=raw.loc[(raw.index>=start)&(raw.index<=end)]
        times=between.index.astype('int64').to_numpy()
        if len(times)!=int(item['nominal_gap_hours'])+1 or not np.all(np.diff(times)==N):
            raise ValueError(f"{item['sample_id']}: actual GFW density changed")
        left=raw.loc[start]; right=raw.loc[end]
        ends=core.widen_fix_ends((start.value,float(left.lat),float(left.lon)),(end.value,float(right.lat),float(right.lon)),N//2)
        info[item['sample_id']]={**item,'left':ends[0],'right':ends[1]}
        for stamp in pd.date_range(start.floor('h'),end.ceil('h'),freq='h'):
            plans.append({'sample_id':item['sample_id'],'candidate_id':item['candidate_id'],
                          'hour':stamp.value,'is_hidden':start<stamp<end})
    frame=pd.DataFrame(plans)
    months=pd.to_datetime(frame.hour,utc=True).dt.strftime('%Y-%m').unique()
    cols=['candidate_id','position_time_utc','swh_m','lat','lon','tc_within_300km_snapshot','tc_within_500km_snapshot','segment_weight_hours']
    weather=pd.concat([pd.read_parquet(Path(voyage_set)/'shards'/shard/f'weather_{m}.parquet',columns=cols,
        filters=[('candidate_id','in',ids)]) for m in months],ignore_index=True)
    weather['hour']=weather.position_time_utc.astype('int64')
    if weather.duplicated(['candidate_id','hour']).any(): raise ValueError('duplicate production hourly weather')
    weather=weather.rename(columns={'swh_m':'gfw_swh','tc_within_300km_snapshot':'gfw_tc300','tc_within_500km_snapshot':'gfw_tc500'})
    frame=frame.merge(weather,on=['candidate_id','hour'],how='left',validate='many_to_one')
    if not frame.segment_weight_hours.eq(1).all():
        raise ValueError('dense-window full-hour support unavailable')
    # Dense original GFW observations do not guarantee a reconstructed UTC
    # hour: river/disconnected-water and speed failures remain unknown. Keep
    # these frozen samples, count missing reference support, and never read the
    # production False placeholder as a known no-cyclone hour.
    unavailable=frame.lat.isna()|frame.lon.isna()
    frame['gfw_position_unavailable']=unavailable
    for col in ['gfw_tc300','gfw_tc500']:
        frame[col]=frame[col].astype(float).mask(unavailable)
    # References are loaded only after the mask and bound-input plans are frozen.
    cmap=pd.read_parquet(CANDIDATES/'shards'/shard/'candidates.parquet',filters=[('candidate_id','in',ids)])
    pairs_path=PAIRS/'shards'/shard/'hourly_weather_pairs.parquet'
    pairs=pd.read_parquet(pairs_path,columns=['cand','hour','swh_true','tc300_true','tc500_true'],
        filters=[('cand','in',cmap.cand.tolist())]) if len(cmap) else pd.DataFrame(columns=['cand','hour','swh_true','tc300_true','tc500_true'])
    pairs=pairs.merge(cmap[['cand','candidate_id']],on='cand',how='inner')
    pairs=pairs.rename(columns={'swh_true':'reference_ais_swh','tc300_true':'reference_ais_tc300','tc500_true':'reference_ais_tc500'})
    frame=frame.merge(pairs[['candidate_id','hour','reference_ais_swh','reference_ais_tc300','reference_ais_tc500']],on=['candidate_id','hour'],how='left',validate='many_to_one')
    for col in ['reference_ais_tc300','reference_ais_tc500']: frame[col]=frame[col].map({True:1.,False:0.})
    kernels=core.module('e007_direct_interval_kernels',core.E002/'20_quantify_uncertainty/10_missing_positions/60_compute_tolerant_bounds.py')
    if wave_support_revision:
        from wave_support_bounds import install
        install(kernels)
    config=json.loads((core.WEATHER/'config/2025_ten_minute_weather.json').read_text())
    fixes,_=kernels.PROBE.load_ibtracs_snapshot(Path(config['ibtracs_path']),expected_sha256=config['ibtracs_sha256'],time_end_exclusive='2026-01-02T00:00:00Z')
    cyclone=kernels.PROBE.CycloneTrackIndex(fixes,maximum_fix_gap_hours=3)
    frame['wave_lo_m']=np.nan; frame['wave_hi_m']=np.nan; frame['wave_status']='known'
    frame['tc300_possible']=False; frame['tc500_possible']=False
    hidden=frame[frame.is_hidden]
    for month, block in hidden.groupby(pd.to_datetime(hidden.hour,utc=True).dt.strftime('%Y-%m')):
        year,mon=map(int,month.split('-'))
        with kernels.PROBE.Era5MonthlyMatcher(kernels.PROBE.resolve_monthly_era5_assets(year=year,month=mon),maximum_wet_cell_snap_km=0,cached_hours=2) as matcher:
            kernels.PROBE.validate_month_time_grid(matcher,year,mon); matcher.preload_swh()
            storms=cyclone.positions_at_times(pd.to_datetime(block.hour.unique(),utc=True))
            storm_map={pd.Timestamp(t).value:(g.storm_lat.to_numpy(float),g.storm_lon.to_numpy(float)) for t,g in storms.groupby('target_time_utc')}
            for idx,item in block.iterrows():
                ends=info[item.sample_id]
                geometry=kernels._geometry(int(item.hour),ends['left'],ends['right'])
                lo,hi,status=kernels._wave_bounds(geometry,pd.Timestamp(item.hour,tz='UTC'),matcher)
                frame.loc[idx,['wave_lo_m','wave_hi_m','wave_status']]=[lo,hi,status]
                for radius in (300,500):
                    frame.loc[idx,f'tc{radius}_possible']=kernels._tc_possible(geometry,storm_map.get(int(item.hour)),radius,old_rule=True)
    results=[]
    for sid, block in frame.groupby('sample_id',sort=False):
        metadata={k:v for k,v in info[sid].items() if k not in ('left','right')}
        for reference,end_to_end in [('gfw',False),('reference_ais',False),('reference_ais',True)]:
            results.append({**metadata,'gfw_hourly_position_unavailable':int(block.gfw_position_unavailable.sum()),
                            **exposure_comparison(block,reference,end_to_end)})
    folder.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(folder/'hour_results.parquet',index=False)
    pd.DataFrame(results).to_parquet(folder/'window_results.parquet',index=False)
    receipt={'state':'pass','shard':shard,'samples':len(samples),'hours':len(frame),'hidden_hours':int(frame.is_hidden.sum()),
             'wave_support_revision':wave_support_revision,
             'interior_reference_positions_supplied_to_bounds':0,'track':track,
             'reference_pairs':core.record(pairs_path),'seconds':time.perf_counter()-started,
             'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
             'outputs':[core.record(folder/x) for x in ['hour_results.parquet','window_results.parquet']]}
    core.write_json(folder/'status.json',receipt)
    return receipt


def summarize(out):
    results=pd.concat([pd.read_parquet(p) for p in sorted((out/'shards').glob('*/window_results.parquet'))],ignore_index=True)
    results.to_parquet(out/'window_results.parquet',index=False)
    summaries=[]
    for levels in [['reference','comparison'],['reference','comparison','sample_set','nominal_gap_hours'],['reference','comparison','sample_set','nominal_gap_hours','shiptype_group']]:
        for key, block in results.groupby(levels,dropna=False):
            row=dict(zip(levels,key)); row.update(windows=len(block),distinct_voyages=block.candidate_id.nunique())
            for metric in ['wave','tc300','tc500']:
                valid=block[f'{metric}_inside'].notna()
                row[f'{metric}_evaluable_windows']=int(valid.sum())
                row[f'{metric}_inside_windows']=int(block.loc[valid,f'{metric}_inside'].astype(bool).sum())
                row[f'{metric}_outside_windows']=int(valid.sum())-row[f'{metric}_inside_windows']
                row[f'{metric}_unevaluable_windows']=int((~valid).sum())
                row[f'{metric}_inclusion_fraction']=row[f'{metric}_inside_windows']/int(valid.sum()) if valid.any() else None
            for col in ['target_hours','hidden_hours','wave_unsupported_hours','wave_compared_hours','wave_reference_missing_on_supported_hours','tc300_compared_hours','tc300_reference_missing_hours']:
                row[col]=int(block[col].sum())
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'inclusion_summary.csv',index=False)
    widths=[]
    for key, block in results.groupby(['reference','comparison','sample_set','nominal_gap_hours']):
        row=dict(zip(['reference','comparison','sample_set','nominal_gap_hours'],key))
        row['windows']=len(block)
        for metric,unit in [('wave','m'),('tc300','pct'),('tc500','pct')]:
            valid=block[f'{metric}_inside'].notna()
            lower=f'{metric}_lower_{unit}'; upper=f'{metric}_upper_{unit}'
            width=block.loc[valid,upper]-block.loc[valid,lower]
            row[f'{metric}_evaluable']=int(valid.sum())
            for q in [0.5,0.9]:
                row[f'{metric}_width_p{int(q*100)}_{unit}']=float(width.quantile(q)) if len(width) else None
            if metric!='wave':
                row[f'{metric}_positive_reference_windows']=int(block.loc[valid,f'{metric}_reference_hours'].gt(0).sum())
        widths.append(row)
    pd.DataFrame(widths).to_csv(out/'interval_width_summary.csv',index=False)
    failures=results.loc[results.wave_inside.eq(False)|results.tc300_inside.eq(False)|results.tc500_inside.eq(False)]
    failures.to_csv(out/'outside_windows.csv',index=False)
    overall=[r for r in summaries if 'sample_set' not in r]
    return {'state':'pass','checks':{'three_comparisons_per_sample':len(results)==results.sample_id.nunique()*3,
            'unique_sample_comparisons':not results.duplicated(['sample_id','reference','comparison']).any(),
            'wave_lower_le_upper':bool((results.wave_lower_m.dropna()<=results.wave_upper_m.dropna()+1e-8).all()),
            'tc300_lower_le_upper':bool((results.tc300_lower_hours<=results.tc300_upper_hours+1e-8).all()),
            'tc500_lower_le_upper':bool((results.tc500_lower_hours<=results.tc500_upper_hours+1e-8).all())},
            'windows':results.sample_id.nunique(),'summary':overall,
            'interpretation':'Execution/aggregation checks may pass even when scientific intervals fail to include a reference. Report all failures; no parameter tuning.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','evaluate']); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--voyage-set',type=Path,required=True); p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12); p.add_argument('--shards',nargs='+')
    p.add_argument('--wave-support-revision',action='store_true',help='authorized E007 matcher-support bounds; original diagnostic unchanged by default')
    args=p.parse_args(); args.out=core.inside_runs(args.out)
    if not 1<=args.workers<=12: raise ValueError('diagnostic workers must be 1..12')
    design=json.loads(DESIGN.read_text()); binding=core.binding(); rule=core.revision()
    if design['speed_cap_knots']!=25 or design['fix_time_tolerance_hours']!=rule['fix_time_tolerance_hours']: raise ValueError('protected bound-rule drift')
    if args.voyage_set.resolve()!=binding['weather_run'].resolve(): raise ValueError('wrong option-C voyage source')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.wave_support_revision:
        from wave_support_bounds import binding as wave_binding
        wave_binding()
    state={'state':'running','stage':args.stage,'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'workers':args.workers,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        'design':core.record(DESIGN),'samples_input':core.record(args.samples),'real_gap_sources':{'pairs':str(PAIRS),'candidates':str(CANDIDATES)},'voyage_set_authorization':binding['authorization'],
        'code':[core.record(Path(__file__)),core.record(HERE/'interval_validation_core.py'),
                core.record(E006/'step6_core.py'),
                core.record(core.E002/'20_quantify_uncertainty/10_missing_positions/60_compute_tolerant_bounds.py'),
                core.record(core.E002/'20_quantify_uncertainty/10_missing_positions/10_probe_reachability.py'),
                core.record(core.E002/'20_quantify_uncertainty/10_missing_positions/20_compute_tc_upper.py'),
                core.record(core.WEATHER/'leg_ais_coverage/30_bound_dark_hour_wave_exposure.py'),
                core.record(core.WEATHER/'config/2025_ten_minute_weather.json')],
        'network_requests':0,'products_modified':0,'adopted':False,'failures':[],'completed':[]}
    state['wave_support_revision']=args.wave_support_revision
    if args.wave_support_revision:
        state['code'].extend([core.record(HERE/'wave_support_bounds.py'),core.record(HERE/'config/wave_support_revision.json')])
    core.write_json(args.out/f'{args.stage}_execution.json',state)
    try:
        if args.stage=='prepare': state.update(select_manifest(args,design))
        else:
            all_shards=sorted(pd.read_parquet(args.samples,columns=['source_shard']).source_shard.unique())
            shards=args.shards or all_shards
            if not set(shards).issubset(all_shards): raise ValueError('shard outside frozen sample manifest')
            state['shards']=shards
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                jobs={}
                for s in shards:
                    receipt=args.out/'shards'/s/'status.json'
                    if receipt.exists():
                        saved=json.loads(receipt.read_text())
                        if saved['state']=='pass' and saved.get('wave_support_revision',False)==args.wave_support_revision and all(core.record(Path(r['path']))==r for r in saved['outputs']):
                            state['completed'].append(s); continue
                        raise ValueError(f'{s}: existing output is not valid')
                    jobs[executor.submit(evaluate_shard,s,str(args.samples.resolve()),str(args.voyage_set.resolve()),str(args.out),args.wave_support_revision)]=s
                for future in as_completed(jobs):
                    try:
                        receipt=future.result(); state['completed'].append(jobs[future]); print(json.dumps({k:receipt[k] for k in ['shard','samples','seconds','peak_rss_gib']}),flush=True)
                    except Exception: state['failures'].append({'shard':jobs[future],'traceback':traceback.format_exc()})
                    core.write_json(args.out/'evaluate_execution.json',state)
            if state['failures']: raise RuntimeError(state['failures'])
            summary=summarize(args.out)
            if not all(summary['checks'].values()): raise ValueError(summary['checks'])
            core.write_json(args.out/'summary.json',summary); state['summary']=summary
        state['state']='pass'
    except Exception:
        state['state']='fail'; state['traceback']=traceback.format_exc(); raise
    finally:
        state['finished_utc']=pd.Timestamp.now(tz='UTC').isoformat()
        core.write_json(args.out/f'{args.stage}_execution.json',state)
    print(json.dumps({'state':state['state'],'stage':args.stage,'selected_windows':state.get('selected_windows'),'completed_shards':len(state['completed'])}),flush=True)


if __name__=='__main__': main()
