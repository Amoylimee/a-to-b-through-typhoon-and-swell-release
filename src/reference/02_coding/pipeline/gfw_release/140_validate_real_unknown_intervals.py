#!/usr/bin/env python3
"""Direct bounds versus exact-UTC-hour reference AIS observations inside real unknown GFW gaps.

Five voyages per shiptype x gap-length class plus up to 30 cyclone-near hours,
frozen before exposure-bound evaluation. One selected unknown hour per sample;
nearest known production hours are M, selected unknown hour is Q. This is an
internal diagnostic of conditional missing-position bounds, not annual accuracy.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; CODING=HERE.parents[1]
sys.path.insert(0,str(CODING/'pipeline/gfw_release'))
import step6_core as core
from port_constraints import port_ends_usable
from interval_validation_core import exposure_comparison
POINTS=CODING/'.runs/2026-10-08_2235_E007-real-gap-comparison'
N=core.NS_HOUR; SEED=20261011


def rank(*args): return hashlib.sha256('|'.join(map(str,args)).encode()).hexdigest()


def kernels_and_cyclone(wave_support_revision=False):
    kernels=core.module('e007_real_unknown_kernels',core.E002/'20_quantify_uncertainty/10_missing_positions/60_compute_tolerant_bounds.py')
    if wave_support_revision:
        from wave_support_bounds import install
        install(kernels)
    cfg=json.loads((core.WEATHER/'config/2025_ten_minute_weather.json').read_text())
    fixes,_=kernels.PROBE.load_ibtracs_snapshot(Path(cfg['ibtracs_path']),expected_sha256=cfg['ibtracs_sha256'],time_end_exclusive='2026-01-02T00:00:00Z')
    return kernels,kernels.PROBE.CycloneTrackIndex(fixes,maximum_fix_gap_hours=3),cfg


def build_pool(shard,prepared):
    p=POINTS/'shards'/shard/'points.parquet'
    f=pd.read_parquet(p,columns=['cand','tau','k_lat','k_lon','gap_hours'],filters=[('status','=','unresolved')])
    raw_count=len(f); f=f.loc[f.tau.mod(N).eq(0)].copy()
    exact_count=len(f)
    cmap=pd.read_parquet(POINTS/'shards'/shard/'candidates.parquet')
    keep=pd.read_parquet(Path(prepared)/'port_endpoints.parquet',columns=['candidate_id'],filters=[('source_shard','=',shard)])
    f=f.merge(cmap[cmap.candidate_id.isin(keep.candidate_id)],on='cand',validate='many_to_one')
    f=f.sort_values(['cand','tau']).drop_duplicates(['cand','tau'])
    f['source_shard']=shard
    f['gap_group']=np.select([f.gap_hours.le(24),f.gap_hours.le(48)],['up_to_24h','25_to_48h'],default='over_48h')
    f['priority']=[rank(SEED,cid,t,'real_hour') for cid,t in zip(f.candidate_id,f.tau)]
    kernels,cyclone,cfg=kernels_and_cyclone()
    pos=f.rename(columns={'tau':'position_time_utc','k_lat':'lat','k_lon':'lon'})
    pos['position_time_utc']=pd.to_datetime(pos.position_time_utc,utc=True)
    tc=cyclone.match_positions(pos,lat_col='lat',lon_col='lon')
    f['near_tc500']=tc.tc_within_500km.to_numpy(bool)
    f=f.sort_values('priority')
    main=f.drop_duplicates(['candidate_id','gap_group']).groupby(['shiptype_group','gap_group']).head(10).copy(); main['sample_set']='real_unknown_by_shiptype_gap'
    storm=f[f.near_tc500].drop_duplicates('candidate_id').head(60).copy(); storm['sample_set']='real_unknown_tc500_stress'
    return pd.concat([main,storm]),{'shard':shard,'unresolved_reference_points':raw_count,'exact_hour_points':exact_count,
            'eligible_exact_hours_in_current_voyages':len(f),'points_source':core.record(p)}


def evaluate_shard(shard,samples,prepared,voyage_set,out,wave_support_revision=False):
    started=time.perf_counter(); f=pd.read_parquet(samples,filters=[('source_shard','=',shard)])
    ids=f.candidate_id.unique().tolist()
    weather=pd.concat([pd.read_parquet(p,columns=['candidate_id','position_time_utc','hour_anchor_utc','lat','lon','swh_m',
            'segment_weight_hours','tc_within_300km_snapshot','tc_within_500km_snapshot'],filters=[('candidate_id','in',ids)])
            for p in sorted((Path(voyage_set)/'shards'/shard).glob('weather_*.parquet'))],ignore_index=True)
    weather['hour']=weather.position_time_utc.astype('int64')
    ports=pd.read_parquet(Path(prepared)/'port_endpoints.parquet',filters=[('source_shard','=',shard)]).set_index('candidate_id')
    kernels,cyclone,cfg=kernels_and_cyclone(wave_support_revision)
    targets=pd.DataFrame({'hour_anchor_utc':pd.to_datetime(f.tau,utc=True),'lat':f.k_lat.to_numpy(float),'lon':f.k_lon.to_numpy(float)})
    targets['swh_m']=np.nan
    for month,block in targets.groupby(targets.hour_anchor_utc.dt.strftime('%Y-%m')):
        year,mon=map(int,month.split('-'))
        with kernels.PROBE.Era5MonthlyMatcher(kernels.PROBE.resolve_monthly_era5_assets(year=year,month=mon),maximum_wet_cell_snap_km=cfg['maximum_wet_cell_snap_km']) as matcher:
            kernels.PROBE.validate_month_time_grid(matcher,year,mon); matcher.preload_swh()
            targets.loc[block.index,'swh_m']=matcher.match(block[['hour_anchor_utc','lat','lon']]).swh_m.to_numpy()
    true_tc=cyclone.match_positions(targets.assign(position_time_utc=targets.hour_anchor_utc),lat_col='lat',lon_col='lon')
    f['reference_swh_m']=targets.swh_m.to_numpy(); f['reference_tc300']=true_tc.tc_within_300km.to_numpy(bool); f['reference_tc500']=true_tc.tc_within_500km.to_numpy(bool)
    storms=cyclone.positions_at_times(pd.to_datetime(f.tau.unique(),utc=True))
    storm_map={pd.Timestamp(t).value:(g.storm_lat.to_numpy(float),g.storm_lon.to_numpy(float)) for t,g in storms.groupby('target_time_utc')}
    frames=[]; results=[]; geometry_info={}
    for item in f.to_dict('records'):
        block=weather[weather.candidate_id.eq(item['candidate_id'])].sort_values('hour'); event=int(item['tau'])
        row=block[block.hour.eq(event)]
        if len(row)!=1 or row.lat.notna().any() or not row.segment_weight_hours.eq(1).all():
            raise ValueError('exact-hour reference is not a full real unknown production hour')
        known=block[block.lat.notna()&block.lon.notna()]
        l=known[known.hour.lt(event)].tail(1); r=known[known.hour.gt(event)].head(1)
        left=(int(l.iloc[0].hour),float(l.iloc[0].lat),float(l.iloc[0].lon)) if len(l) else None
        right=(int(r.iloc[0].hour),float(r.iloc[0].lat),float(r.iloc[0].lon)) if len(r) else None
        left,right=core.widen_fix_ends(left,right,N//2)
        port=ports.loc[item['candidate_id']]; use_ports=port_ends_usable(port,len(known)>0)
        pleft=(int(port.departure_time.value)-N,float(port.departure_port_lat),float(port.departure_port_lon)) if use_ports and pd.notna(port.departure_port_lat) else None
        pright=(int(port.arrival_time.value)+N,float(port.arrival_port_lat),float(port.arrival_port_lon)) if use_ports and pd.notna(port.arrival_port_lat) else None
        ends=(left or pleft,right or pright)
        geometry_info[item['sample_id']]=(ends,left,right)
        frame=pd.concat([l,row,r],ignore_index=True).rename(columns={'swh_m':'gfw_swh','tc_within_300km_snapshot':'gfw_tc300','tc_within_500km_snapshot':'gfw_tc500'})
        frame['sample_id']=item['sample_id']; frame['is_hidden']=frame.hour.eq(event)
        frame['reference_ais_swh']=frame.gfw_swh; frame['reference_ais_tc300']=frame.gfw_tc300.astype(float); frame['reference_ais_tc500']=frame.gfw_tc500.astype(float)
        frame.loc[frame.is_hidden,['reference_ais_swh','reference_ais_tc300','reference_ais_tc500']]=[item['reference_swh_m'],float(item['reference_tc300']),float(item['reference_tc500'])]
        frame.loc[frame.is_hidden,['gfw_tc300','gfw_tc500']]=np.nan
        frame['wave_status']='known'; frame['wave_lo_m']=np.nan; frame['wave_hi_m']=np.nan; frame['tc300_possible']=False; frame['tc500_possible']=False
        frames.append(frame)
    hourly=pd.concat(frames,ignore_index=True)
    hidden=hourly[hourly.is_hidden]
    for month,block in hidden.groupby(pd.to_datetime(hidden.hour,utc=True).dt.strftime('%Y-%m')):
        year,mon=map(int,month.split('-'))
        with kernels.PROBE.Era5MonthlyMatcher(kernels.PROBE.resolve_monthly_era5_assets(year=year,month=mon),maximum_wet_cell_snap_km=0,cached_hours=2) as matcher:
            kernels.PROBE.validate_month_time_grid(matcher,year,mon); matcher.preload_swh()
            for idx,row in block.iterrows():
                ends,left,right=geometry_info[row.sample_id]
                geometry=kernels._geometry(int(row.hour),*ends) if any(x is not None for x in ends) else None
                lo,hi,status=kernels._wave_bounds(geometry,pd.Timestamp(row.hour,tz='UTC'),matcher)
                hourly.loc[idx,['wave_lo_m','wave_hi_m','wave_status']]=[lo,hi,status]
                for radius in (300,500): hourly.loc[idx,f'tc{radius}_possible']=kernels._tc_possible(geometry,storm_map.get(int(row.hour)),radius,old_rule=ends==(left,right))
    for item in f.to_dict('records'):
        frame=hourly[hourly.sample_id.eq(item['sample_id'])]
        x=exposure_comparison(frame,'reference_ais',False); q=exposure_comparison(frame[frame.is_hidden],'reference_ais',False)
        # Adding the same fixed M to each numerator cannot alter inclusion.
        for metric in ['wave','tc300','tc500']:
            if x[f'{metric}_inside']!=q[f'{metric}_inside']: raise ValueError('M+Q and Q inclusion disagree')
        results.append({**item,**x,'unknown_hour_wave_status':frame.loc[frame.is_hidden,'wave_status'].iloc[0],
                        'unknown_only_wave_lower_m':q['wave_lower_m'],'unknown_only_wave_upper_m':q['wave_upper_m'],
                        'unknown_only_wave_reference_m':q['wave_reference_m']})
    folder=Path(out)/'shards'/shard; folder.mkdir(parents=True,exist_ok=True)
    hourly.to_parquet(folder/'hour_results.parquet',index=False); pd.DataFrame(results).to_parquet(folder/'sample_results.parquet',index=False)
    report={'state':'pass','shard':shard,'samples':len(f),'seconds':time.perf_counter()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            'checks':{'reference_exact_hour':bool(f.tau.mod(N).eq(0).all()),'all_Q_real_unknown':True,'reference_positions_supplied_to_bounds':False,'M_plus_Q_inclusion_equals_Q':True},
            'outputs':[core.record(folder/x) for x in ['hour_results.parquet','sample_results.parquet']]}
    core.write_json(folder/'status.json',report); return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['prepare','evaluate'])
    p.add_argument('--out',type=Path,required=True); p.add_argument('--prepared',type=Path,required=True); p.add_argument('--voyage-set',type=Path,required=True); p.add_argument('--samples',type=Path)
    p.add_argument('--workers',type=int,default=12); p.add_argument('--shards',nargs='+')
    p.add_argument('--wave-support-revision',action='store_true')
    args=p.parse_args(); args.out=core.inside_runs(args.out); binding=core.binding(); core.revision()
    if args.voyage_set.resolve()!=binding['weather_run'].resolve() or not 1<=args.workers<=12: raise ValueError('binding/workers differ from scope')
    args.out.mkdir(parents=True,exist_ok=True)
    state={'state':'running','stage':args.stage,'workers':args.workers,'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'code':core.record(Path(__file__)),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        'seed':SEED,'main_per_shiptype_gap':5,'storm_target':30,'reference_time_tolerance_seconds':0,
        'authorization':binding['authorization'],'products_modified':0,'network_requests':0,'adopted':False,'completed':[],'failures':[]}
    core.write_json(args.out/f'{args.stage}_execution.json',state)
    state['wave_support_revision']=args.wave_support_revision
    if args.wave_support_revision:
        from wave_support_bounds import binding as wave_binding
        wave_binding();state['wave_support_code']=[core.record(HERE/'wave_support_bounds.py'),core.record(HERE/'config/wave_support_revision.json')]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            if args.stage=='prepare': jobs={pool.submit(build_pool,s,str(args.prepared)):s for s in (args.shards or core.SHARDS)}
            else:
                if args.samples is None: raise ValueError('need frozen manifest')
                state['sample_manifest']=core.record(args.samples)
                shards=args.shards or sorted(pd.read_parquet(args.samples,columns=['source_shard']).source_shard.unique())
                jobs={pool.submit(evaluate_shard,s,str(args.samples),str(args.prepared),str(args.voyage_set),str(args.out),args.wave_support_revision):s for s in shards}
            collected=[]
            for future in as_completed(jobs):
                try:
                    result=future.result(); collected.append(result); state['completed'].append(jobs[future]); print(json.dumps({'shard':jobs[future],'completed':len(state['completed'])}),flush=True)
                except Exception: state['failures'].append({'shard':jobs[future],'traceback':traceback.format_exc()})
                core.write_json(args.out/f'{args.stage}_execution.json',state)
        if state['failures']: raise RuntimeError(state['failures'])
        if args.stage=='prepare':
            f=pd.concat([x[0] for x in collected],ignore_index=True); f=f.sort_values('priority')
            main=f[f.sample_set.eq('real_unknown_by_shiptype_gap')].drop_duplicates(['candidate_id','gap_group']).groupby(['shiptype_group','gap_group']).head(5)
            storm=f[f.sample_set.eq('real_unknown_tc500_stress')].drop_duplicates('candidate_id').head(30)
            f=pd.concat([main,storm],ignore_index=True); f['sample_id']=[rank(SEED,r['candidate_id'],r['tau'],r['sample_set'])[:24] for r in f.to_dict('records')]
            f=f.sort_values('sample_id'); f.to_parquet(args.out/'samples.parquet',index=False)
            state.update(samples=len(f),pool_receipts=[x[1] for x in collected],manifest=core.record(args.out/'samples.parquet'))
        else:
            f=pd.concat([pd.read_parquet(p) for p in sorted((args.out/'shards').glob('*/sample_results.parquet'))],ignore_index=True)
            if f.sample_id.duplicated().any() or set(f.sample_id)!=set(pd.read_parquet(args.samples).query('source_shard in @shards').sample_id): raise ValueError('sample closure failed')
            f.to_parquet(args.out/'sample_results.parquet',index=False)
            summaries=[]
            for levels in [['sample_set'],['sample_set','gap_group'],['sample_set','gap_group','shiptype_group']]:
                for key,g in f.groupby(levels):
                    if not isinstance(key,tuple): key=(key,)
                    row=dict(zip(levels,key)); row['samples']=len(g)
                    for metric in ['wave','tc300','tc500']:
                        valid=g[f'{metric}_inside'].notna(); row[f'{metric}_evaluable']=int(valid.sum()); row[f'{metric}_inside']=int(g.loc[valid,f'{metric}_inside'].astype(bool).sum()); row[f'{metric}_outside']=row[f'{metric}_evaluable']-row[f'{metric}_inside']; row[f'{metric}_unevaluable']=int((~valid).sum())
                    summaries.append(row)
            pd.DataFrame(summaries).to_csv(args.out/'inclusion_summary.csv',index=False)
            core.write_json(args.out/'summary.json',{'state':'pass','samples':len(f),'summary':[r for r in summaries if 'gap_group' not in r],
                'support_status':f.unknown_hour_wave_status.value_counts().to_dict(),'reference_exact_UTC_hour':True,'interpretation':'Only selected real unknown hours and fixed known context; no independent end-to-end physical or annual OD claim.'})
            state['samples']=len(f)
        state['state']='pass'
    except Exception:
        state['state']='fail'; state['traceback']=traceback.format_exc(); raise
    finally:
        state['finished_utc']=pd.Timestamp.now(tz='UTC').isoformat(); core.write_json(args.out/f'{args.stage}_execution.json',state)
    print(json.dumps({'state':state['state'],'samples':state['samples']}),flush=True)


if __name__=='__main__': main()
