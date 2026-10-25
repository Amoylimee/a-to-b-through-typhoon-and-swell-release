#!/usr/bin/env python3
"""Reuse E006 30b/30c, then apply Option C and explicit segmentation guards."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa
HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
E006 = CODING / 'pipeline/gfw_release'
sys.path.insert(0, str(E006))
import step6_core as core
from port_segmentation import guard_candidates, distances
RULE = HERE / 'config/port_segmentation_revision.json'
CLEAN = CODING / '.runs/2026-10-05_2305_E006-gfw-cleaning'
OLD_PANEL = CODING / '.runs/2026-10-11_1238_E007-wave-support-release/preparation/voyage_panel.parquet'
NATURAL_KEY = ['vessel_uid_clean', 'origin_city', 'dest_city', 'departure_time', 'arrival_time']


def guard_worker(shard, panel, port_root, out):
    pa.set_cpu_count(1)
    f = pd.read_parquet(panel, filters=[('source_shard','=',shard)])
    visits = pd.read_parquet(Path(port_root)/'shards'/shard/'visit_quality.parquet')
    for c in ['entry_time','exit_time','first_seen_time','last_seen_time']:
        visits[c] = pd.to_datetime(visits[c], utc=True)
    p = CLEAN/'shards'/shard/'mask'/f'{shard}.feather'
    points = pd.read_feather(p, columns=['vessel_uid','UTC','lat','lon'])
    points['UTC'] = pd.to_datetime(points.UTC, utc=True)
    f = guard_candidates(f, visits, points)
    path = Path(out)/f'{shard}.parquet'
    f.to_parquet(path, index=False)
    return {'shard':shard,'rows':len(f),'eligible':int(f.segmentation_eligible.sum()),
            'source_points':core.record(p),'source_visits':core.record(Path(port_root)/'shards'/shard/'visit_quality.parquet'),
            'output':core.record(path)}


def finalize(root, port_root, voyage, workers, verify):
    f = pd.concat([pd.read_feather(verify(p)) for p in voyage['pretrim']], ignore_index=True)
    if f.candidate_id.duplicated().any(): raise ValueError('pretrim candidate IDs duplicate')
    for c in ['candidate_id','vessel_uid_clean','mmsi']: f[c] = f[c].astype(str)
    for c in ['departure_time','arrival_time']: f[c] = pd.to_datetime(f[c],utc=True)
    cities = pd.read_csv(voyage['city_snapshot']['path']).set_index('city_id')
    a, b = cities.reindex(f.origin_city), cities.reindex(f.dest_city)
    if a.port_lat.isna().any() or b.port_lat.isna().any(): raise ValueError('missing GFW city coordinate')
    hours = (f.arrival_time-f.departure_time).dt.total_seconds()/3600
    if not hours.gt(0).all(): raise ValueError('invalid voyage duration')
    f['straight_line_knots'] = distances(a.port_lat.to_numpy(),a.port_lon.to_numpy(),b.port_lat.to_numpy(),b.port_lon.to_numpy())/(hours+1)/1.852
    eligible_ratio = np.isfinite(f.normalized_time_ratio) & f.d_yardstick.gt(0)
    f['speed_decision'] = np.select([eligible_ratio & f.normalized_time_ratio.gt(.04680512383582364),f.straight_line_knots.gt(36)],
                                    ['slow_above_frozen_q99','straight_line_above_36_kn'],'retained')
    port_state = json.loads((port_root/'execution.json').read_text())
    owners = pd.concat([pd.read_csv(r['vessel_input_disposition']['path'],dtype={'vessel_uid':str})[['vessel_uid']].assign(source_shard=r['shard']) for r in port_state['results']])
    if owners.vessel_uid.duplicated().any(): raise ValueError('duplicate vessel shard')
    f['source_shard'] = f.vessel_uid_clean.map(owners.set_index('vessel_uid').source_shard)
    if f.source_shard.isna().any(): raise ValueError('missing vessel shard')
    f.to_parquet(root/'pretrim_panel.parquet',index=False)
    return guard_and_bridge(root,port_root,workers)


def guard_and_bridge(root, port_root, workers):
    """Guard root/pretrim_panel.parquet and bridge retained voyages to Option C."""
    folder=root/'guards';folder.mkdir()
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(guard_worker,s,root/'pretrim_panel.parquet',port_root,folder):s for s in core.SHARDS}
        for future in as_completed(jobs):
            try:
                r=future.result();results.append(r)
                print(json.dumps({'guard_done':len(results),'shard':r['shard'],'rows':r['rows'],'eligible':r['eligible']}),flush=True)
            except Exception: failures.append({'shard':jobs[future],'error':traceback.format_exc()})
    core.write_json(root/'guard_execution.json',{'status':'failed' if failures else 'pass','results':results,'failures':failures})
    if failures: raise RuntimeError(f'{len(failures)} guard shards failed')
    f=pd.concat([pd.read_parquet(folder/f'{s}.parquet') for s in core.SHARDS],ignore_index=True)
    f['retained']=f.speed_decision.eq('retained') & f.segmentation_eligible
    f.to_parquet(root/'candidate_decisions.parquet',index=False)
    kept=f.loc[f.retained].copy()
    old=pd.read_parquet(OLD_PANEL,columns=['candidate_id',*NATURAL_KEY,'shiptype_group','source_shard'])
    for c in ['candidate_id','vessel_uid_clean']:old[c]=old[c].astype(str)
    for c in ['departure_time','arrival_time']:old[c]=pd.to_datetime(old[c],utc=True)
    if old.duplicated(NATURAL_KEY).any() or kept.duplicated(NATURAL_KEY).any():
        raise ValueError('physical voyage key is not unique; cannot reuse by inference')
    bridge=kept[['candidate_id',*NATURAL_KEY,'shiptype_group','source_shard']].merge(old.rename(columns={'candidate_id':'old_candidate_id','source_shard':'old_source_shard','shiptype_group':'old_shiptype_group'}),
                                                               on=NATURAL_KEY,how='left',validate='one_to_one')
    matched=bridge.old_candidate_id.notna()
    if not bridge.loc[matched,'source_shard'].eq(bridge.loc[matched,'old_source_shard']).all():raise ValueError('physical voyage changed vessel shard')
    if not bridge.loc[matched,'shiptype_group'].eq(bridge.loc[matched,'old_shiptype_group']).all():raise ValueError('physical voyage changed frozen ship type')
    bridge['weather_source']=np.where(matched,'reuse_verified_option_c','rebuild_gfw')
    bridge.to_parquet(root/'physical_voyage_bridge.parquet',index=False)
    kept=kept.merge(bridge[['candidate_id','old_candidate_id','weather_source']],on='candidate_id',validate='one_to_one')
    kept.to_parquet(root/'retained_panel.parquet',index=False)
    removed=old.loc[~old.candidate_id.isin(set(bridge.old_candidate_id.dropna()))]
    removed.to_parquet(root/'old_voyages_not_reused.parquet',index=False)
    counts=kept.groupby(core.KEY,observed=True).size()
    summary={'status':'pass','all_64_guard_shards':len(results)==64,'pretrim':len(f),'retained':len(kept),
             'speed_decisions':f.speed_decision.value_counts().to_dict(),'guard_eligible':int(f.segmentation_eligible.sum()),
             'guard_reasons':f.loc[~f.segmentation_eligible,'segmentation_review_reasons'].str.split(';').explode().value_counts().to_dict(),
             'retained_reused':int(matched.sum()),'retained_rebuilt':int((~matched).sum()),'old_candidates_not_reused':len(removed),
             'combinations':len(counts),'release_combinations_ge5':int(counts.ge(5).sum()),'release_voyages_ge5':int(counts[counts.ge(5)].sum()),
             'candidate_ids_unique':not f.candidate_id.duplicated().any(),'candidate_decisions_close':int(f.retained.sum())==len(kept),
             'old_source':core.record(OLD_PANEL),'rule':core.record(RULE),'adopted':False,'network_requests':0,
             'outputs':[core.record(root/p) for p in ['candidate_decisions.parquet','retained_panel.parquet','physical_voyage_bridge.parquet']]}
    core.write_json(root/'completion_validation.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    args=p.parse_args();root=core.inside_runs(args.out);port_root=args.port_root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    base=core.module('e007_base_reuse',E006/'130_build_gfw_voyage_base.py')
    ports,results=base.port_inputs(port_root)
    state={'status':'running','experiment':'E007','source_port_root':str(port_root),'rule':core.record(RULE),
           'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'workers':args.workers,'adopted':False,'network_requests':0,'results':[]}
    start=time.monotonic()
    save=lambda:core.write_json(root/'execution.json',state)
    save()
    try:
        prep=base.build_view(root,ports,results,E006/'config/t034_anchorage_city_corrections.csv')
        state['input_view']=prep;save()
        voyage=base.run_30c(root,Path(prep['view']),Path(prep['empty_city_reference']['path']),Path(prep['mapping']['path']),args.workers,state,save)
        core.write_json(root/'voyages_execution.json',voyage)
        state.update(current_step='option_c_and_segmentation_guards');save()
        summary=finalize(root,port_root,voyage,args.workers,base.verify)
        state.update(status='pass',current_step='complete_corrected_candidate_base',summary=summary)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state.update(wall_seconds=time.monotonic()-start);save()


if __name__=='__main__':main()
