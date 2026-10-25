#!/usr/bin/env python3
"""Reuse bounds only when all inputs used by their geometry are identical.

Same physical voyage and verified composed weather are required. Each unknown
hour either has both fixed-position ends or uses unchanged city coordinates on
every missing side. All other unknown voyages are recomputed. Two reused voyages
per shard are independently replayed with the current kernels as a control.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa
HERE=Path(__file__).resolve().parent
E006=HERE.parents[1]/'pipeline/gfw_release'
sys.path.insert(0,str(E006))
import step6_core as core
from wave_support_bounds import install
OLD=HERE.parents[1]/'.runs/2026-10-11_1238_E007-wave-support-release'


def make_stage():
    stage=core.module('e007_equivalent_bound_stage',E006/'210_bound_gfw_missing_weather.py')
    original=stage.module
    def loader(name,path):
        value=original(name,path)
        if Path(path).name=='60_compute_tolerant_bounds.py':install(value)
        return value
    stage.module=loader
    return stage


def compare_bounds(a,b):
    a=a.set_index('candidate_id').sort_index();b=b.set_index('candidate_id').sort_index()
    if not a.index.equals(b.index):raise ValueError('replayed bound identity changed')
    cols=sorted(set(a.columns)|set(b.columns))
    a=a.reindex(columns=cols,fill_value=0).fillna(0);b=b.reindex(columns=cols,fill_value=0).fillna(0)
    if not np.allclose(a.to_numpy(float),b.to_numpy(float),rtol=0,atol=1e-8):raise ValueError('reused geometry control differs')


def classify_equivalence(plan,current,previous):
    """Which port sides are actually required by any unknown hour?"""
    plan=plan.copy()
    for column in ['first_known','last_known','first_unknown','last_unknown']:
        plan[column]=pd.to_datetime(plan[column],utc=True)
    plan['needs_departure_port']=plan.first_known.isna()|plan.first_unknown.lt(plan.first_known)
    plan['needs_arrival_port']=plan.last_known.isna()|plan.last_unknown.gt(plan.last_known)
    def identical(names):
        a,b=current[names],previous[names]
        return (a.eq(b)|(a.isna()&b.isna())).all(axis=1)
    plan['departure_ports_identical']=identical(['departure_port_lat','departure_port_lon'])
    plan['arrival_ports_identical']=identical(['arrival_port_lat','arrival_port_lon'])
    plan['reusable']=plan.weather_source.eq('reuse_verified_option_c') & \
        (~plan.needs_departure_port|plan.departure_ports_identical) & \
        (~plan.needs_arrival_port|plan.arrival_ports_identical) & identical(['port_endpoints_connected'])
    return plan


def worker(shard,prepared,base,out):
    pa.set_cpu_count(1);started=time.monotonic();out=Path(out)
    stage=make_stage();tolerance=core.revision()['fix_time_tolerance_hours']
    ports=pd.read_parquet(Path(prepared)/'port_endpoints.parquet',filters=[('source_shard','=',shard)]).set_index('candidate_id')
    old_ports=pd.read_parquet(OLD/'preparation/port_endpoints.parquet',filters=[('source_shard','=',shard)]).set_index('candidate_id')
    bridge=pd.read_parquet(Path(base)/'physical_voyage_bridge.parquet',filters=[('source_shard','=',shard)]).set_index('candidate_id')
    frame=core.read_weather(shard)
    unknown=frame.loc[frame.track_evidence.eq('unresolved')]
    known=frame.loc[frame.lat.notna()]
    u=unknown.groupby('candidate_id').agg(first_unknown=('position_time_utc','min'),last_unknown=('position_time_utc','max'),
                                        unknown_hours=('segment_weight_hours','sum'))
    k=known.groupby('candidate_id').agg(first_known=('position_time_utc','min'),last_known=('position_time_utc','max'))
    plan=u.join(k).join(bridge[['old_candidate_id','weather_source']])
    previous=old_ports.reindex(plan.old_candidate_id.to_numpy());previous.index=plan.index
    current=ports.reindex(plan.index)
    plan=classify_equivalence(plan,current,previous)
    # Physical matching already guarantees both UTC boundaries. The port
    # feasibility fallback therefore also has identical inputs for blind trips.
    if plan.old_candidate_id.loc[plan.reusable].isna().any():raise ValueError('missing physical identity')
    old_status=json.loads((OLD/'bounds'/shard/'status.json').read_text())
    old_record=old_status['output']
    if core.record(Path(old_record['path']))!=old_record:raise ValueError('source bound output changed')
    old=pd.read_parquet(old_record['path']).set_index('candidate_id')
    reused=plan.loc[plan.reusable]
    if not reused.old_candidate_id.isin(old.index).all():raise ValueError('reused unknown voyage lacks old bound')
    kept=old.loc[reused.old_candidate_id].copy();kept.index=reused.index;kept=kept.reset_index()
    changed=plan.index[~plan.reusable].astype(str).tolist()
    parts=[kept]
    if changed:
        stage.run_shard(shard,str(prepared),str(out/'recomputed'),None,tolerance,False,changed)
        parts.append(pd.read_parquet(out/'recomputed'/shard/'voyage_bounds.parquet'))
    # Select stable short examples to keep the control bounded; the complete
    # reuse decision above inspects every unknown hour's required endpoints.
    control=reused.sort_values('unknown_hours').head(2).index.astype(str).tolist()
    if control:
        stage.run_shard(shard,str(prepared),str(out/'reuse_controls'),None,tolerance,False,control)
        actual=pd.read_parquet(out/'reuse_controls'/shard/'voyage_bounds.parquet')
        compare_bounds(kept.loc[kept.candidate_id.isin(control)],actual)
    f=pd.concat(parts,ignore_index=True).fillna(0).sort_values('candidate_id').reset_index(drop=True)
    if f.candidate_id.duplicated().any() or set(f.candidate_id)!=set(plan.index):raise ValueError('unknown candidate closure')
    if not np.isclose(f.get('unknown_hours',pd.Series(dtype=float)).sum(),unknown.segment_weight_hours.sum(),rtol=0,atol=1e-6):
        raise ValueError('composed unknown-hour closure')
    folder=out/shard;folder.mkdir();f.to_parquet(folder/'voyage_bounds.parquet',index=False)
    plan.reset_index().to_parquet(folder/'geometry_input_equivalence.parquet',index=False)
    progress={'state':'completed','shard':shard,'probe_candidates':None,'hour_detail':False,'selected_candidate_ids':None,
              'fix_time_tolerance_hours':tolerance,'weather_rows':len(frame),'unknown_segments':len(unknown),
              'voyages_with_unknown':len(plan),'reused_unknown_voyages':len(reused),'recomputed_unknown_voyages':len(changed),
              'independently_replayed_reuses':len(control),'reuse_control_matches':True,
              'totals':{c:float(f[c].sum()) for c in f if c!='candidate_id'},
              'output':core.record(folder/'voyage_bounds.parquet'),'old_bound':old_record,
              'geometry_equivalence':core.record(folder/'geometry_input_equivalence.parquet'),
              'elapsed_seconds':time.monotonic()-started}
    core.write_json(folder/'status.json',progress);return progress


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','base','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--shards',nargs='+',default=core.SHARDS)
    p.add_argument('--reuse-components',type=Path,nargs='*',default=[])
    args=p.parse_args()
    shards=sorted(args.shards)
    if len(shards)!=len(set(shards)) or not set(shards)<=set(core.SHARDS):raise ValueError('invalid shard set')
    out=core.inside_runs(args.out);out.mkdir(parents=True,exist_ok=False)
    source=core.binding();rule=core.revision()
    old_validation=json.loads((OLD/'validation.json').read_text())
    if old_validation['status']!='pass' or not all(old_validation['checks'].values()):raise ValueError('validated source bounds required')
    assets=core.weather_runner.verify_weather_assets()
    paths=[Path(__file__),HERE/'230_bound_segmented_weather.py',HERE/'wave_support_bounds.py',HERE/'config/wave_support_revision.json',
           HERE/'config/port_segmentation_revision.json',E006/'210_bound_gfw_missing_weather.py',E006/'step6_core.py',E006/'port_constraints.py',
           core.E002/'20_quantify_uncertainty/10_missing_positions/60_compute_tolerant_bounds.py']
    code=[core.record(x) for x in paths]
    state={'state':'running','step':6,'component':'missing_weather_bounds','shards':shards,'workers':args.workers,
           'probe_candidates':None,'hour_detail':False,'fix_time_tolerance_hours':rule['fix_time_tolerance_hours'],
           'completed':[],'failures':[],'weather_assets':assets,'code':code,'authorization':source['authorization'],
           'preparation':core.record(args.prepared/'preparation.json'),'bridge':core.record(args.base/'physical_voyage_bridge.parquet'),
           'old_validated_release':core.record(OLD/'validation.json'),'wave_support_revision':True,'segmentation_revision':True,
           'scope':'All unknown hours assessed; recompute changed geometry and reuse only provably identical required inputs.',
           'adopted':False,'network_requests':0,'products_modified':0}
    start=time.monotonic();core.write_json(out/'status.json',state)
    results=[]
    for component in args.reuse_components:
        previous=json.loads((component/'status.json').read_text())
        if previous['state']!='completed' or previous['failures']:raise ValueError('component incomplete')
        for key in ['code','preparation','bridge','authorization','fix_time_tolerance_hours','old_validated_release']:
            if previous[key]!=state[key]:raise ValueError('bound component inputs differ')
        for shard in previous['completed']:
            if shard not in shards or shard in state['completed']:raise ValueError('component shard overlap')
            r=json.loads((component/shard/'status.json').read_text())
            for item in [r['output'],r['geometry_equivalence']]:
                if core.record(Path(item['path']))!=item:raise ValueError('bound component output changed')
            (out/shard).symlink_to((component/shard).resolve(),target_is_directory=True)
            results.append(r);state['completed'].append(shard)
    state['reused_components']=[core.record(x/'status.json') for x in args.reuse_components]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(worker,s,args.prepared,args.base,out):s for s in shards if s not in state['completed']}
        for future in as_completed(jobs):
            try:
                r=future.result();results.append(r);state['completed'].append(r['shard'])
                print(json.dumps({k:r[k] for k in ['shard','unknown_segments','reused_unknown_voyages','recomputed_unknown_voyages','elapsed_seconds']}),flush=True)
            except Exception:state['failures'].append({'shard':jobs[future],'traceback':traceback.format_exc()})
            core.write_json(out/'status.json',state)
    for item in code:
        if core.record(Path(item['path']))!=item:raise ValueError('bound implementation changed during run')
    state.update(state='failed' if state['failures'] else 'completed',elapsed_seconds=time.monotonic()-start,
                 reused_unknown_voyages=sum(r['reused_unknown_voyages'] for r in results),
                 recomputed_unknown_voyages=sum(r['recomputed_unknown_voyages'] for r in results),
                 independently_replayed_reuses=sum(r['independently_replayed_reuses'] for r in results))
    core.write_json(out/'status.json',state)
    if state['failures']:raise RuntimeError(state['failures'])


if __name__=='__main__':main()
