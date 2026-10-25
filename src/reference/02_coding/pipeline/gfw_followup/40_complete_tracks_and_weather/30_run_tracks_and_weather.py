"""Probe, then build new E009 tracks/weather and retain exact E007 components."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import weather_extension as weather
core=weather.core
CODING=weather.CODING;PIPE=weather.PIPE
TRACKS=core.module('e009_tracks',PIPE/'160_build_gfw_ten_minute_tracks.py')
CHECK=core.module('e009_track_check',PIPE/'170_validate_gfw_ten_minute_tracks.py')
QUALITY=CODING/'.runs/2026-10-21_2324_E009-quality-base'
PORT=CODING/'.runs/2026-10-21_2154_E009-candidate-port-view'
OLD=CODING/'.runs/2026-10-11_1907_E007-one-cell-weather'
ASSETS=CODING/'.runs/2026-10-21_2330_E009-weather-assets'
BASELINE_RUN=None
E007=OLD


def old_facts_source():
    if BASELINE_RUN is not None:return json.loads((BASELINE_RUN/'summary.json').read_text())['output']
    return json.loads((OLD/'status.json').read_text())['annual_output']


def baseline_components():
    bridge=pd.read_parquet(BASELINE_RUN/'component_bridge.parquet')
    if 'track_source_path' in bridge:return bridge
    prior=bridge.old_candidate_id.notna()
    bridge['track_source_candidate_id']=bridge.old_candidate_id.where(prior,bridge.candidate_id)
    bridge['weather_source_candidate_id']=bridge.track_source_candidate_id
    for shard in bridge.source_shard.unique():
        for legacy in [True,False]:
            selected=bridge.source_shard.eq(shard)&prior.eq(legacy)
            if not selected.any():continue
            folder=E007/'shards'/shard if legacy else BASELINE_RUN/'weather'/shard
            status=json.loads((folder/'status.json').read_text())
            bridge.loc[selected,'track_source_path']=status['sources']['track']['path']
            bridge.loc[selected,'weather_source_path']=str(folder)
    return bridge


def bindings():
    assets=json.loads((ASSETS/'manifest.json').read_text())
    for item in assets['weather_files']:
        if core.record(Path(item['path']))!=item: raise ValueError('bound weather asset changed')
    records=[core.record(p) for p in [Path(__file__),HERE/'weather_extension.py',
        PIPE/'160_build_gfw_ten_minute_tracks.py',PIPE/'170_validate_gfw_ten_minute_tracks.py',
        PIPE/'190_validate_gfw_weather.py',QUALITY/'execution.json',QUALITY/'retained_panel.parquet',ASSETS/'manifest.json',
        PORT/'execution.json']]
    records.append(old_facts_source())
    if BASELINE_RUN is not None:records.append(core.record(BASELINE_RUN/'component_bridge.parquet'))
    return records


def prepare(root,probe):
    quality=json.loads((QUALITY/'execution.json').read_text())
    if quality['status']!='pass': raise ValueError('quality base not complete')
    port_state=json.loads((PORT/'execution.json').read_text())
    if BASELINE_RUN is None:
        overlap=json.loads((PORT/'boundary_points_old_raw_leg_overlap.json').read_text())
        if overlap.get('changed_points_inside_old_raw_legs')!=0:raise ValueError('boundary overlap not cleared')
    elif any(not r.get('retained_input_points_exact') for r in port_state['results']):
        raise ValueError('all retained-voyage historical inputs must be exact')
    panel=pd.read_parquet(QUALITY/'retained_panel.parquet')
    if panel.candidate_id.duplicated().any() or panel.departure_time.ge('2026-01-01').any(): raise ValueError('wrong cohort')
    results={r['shard']:r for r in json.loads((PORT/'execution.json').read_text())['results']}
    prepared=root/'prepared';prepared.mkdir()
    tasks=[]
    shards=['shard_0032','shard_0057'] if probe else sorted(results)
    for shard in shards:
        selected=panel.loc[panel.source_shard.eq(shard) & panel.old_candidate_id.isna()].copy()
        if probe:
            # Exercise both arrival years, short and long support, and exact old twins.
            groups=[g.sort_values('total_hours').iloc[np.unique([0,len(g)-1])] for _,g in selected.groupby(selected.arrival_time.dt.year) if len(g)]
            previous=panel.loc[panel.source_shard.eq(shard) & panel.old_candidate_id.notna()].sort_values('arrival_time')
            if 'e007_candidate_id' in previous:
                old=pd.concat([previous[previous.e007_candidate_id.notna()].tail(2),previous[previous.e007_candidate_id.isna()].tail(2)])
            else:old=previous.tail(2)
            selected=pd.concat([*groups,old],ignore_index=True).drop_duplicates('candidate_id')
        selected['expected_grid_rows']=[len(TRACKS.strict_grid_ns(a,b)) for a,b in zip(selected.departure_time,selected.arrival_time)]
        f=prepared/f'{shard}.parquet';selected.to_parquet(f,index=False)
        if not len(selected): continue
        task={'shard':shard,'panel':core.record(f),'candidates':core.record(f),'points':results[shard]['source'],
              'candidate_rows':len(selected),'expected_grid_rows':int(selected.expected_grid_rows.sum()),
              'output':str(root/'tracks'/shard),'weather':str(root/'weather'/shard),'probe':probe}
        tasks.append(task)
    core.write_json(root/'tasks.json',tasks)
    return panel,tasks


def old_probe_check(task):
    panel=pd.read_parquet(task['panel']['path']);old=panel.loc[panel.old_candidate_id.notna()]
    if BASELINE_RUN is not None:
        reference=baseline_components()[['candidate_id','track_source_path','track_source_candidate_id','weather_source_path','weather_source_candidate_id']]
        refs=old[['candidate_id','old_candidate_id']].merge(reference.rename(columns={'candidate_id':'old_candidate_id'}),on='old_candidate_id',validate='one_to_one')
        frames=[]
        for path,g in refs.groupby('track_source_path'):
            folder=Path(g.weather_source_path.iloc[0]);status=json.loads((folder/'status.json').read_text());source=status['sources']['track']
            if core.record(Path(source['path']))!=source:raise ValueError('baseline track source changed')
            f=pd.read_parquet(path,filters=[('candidate_id','in',g.track_source_candidate_id.tolist())])
            f['candidate_id']=f.candidate_id.map(dict(zip(g.track_source_candidate_id,g.candidate_id)));frames.append(f)
        prior=pd.concat(frames,ignore_index=True)
    else:
        status=json.loads((OLD/'shards'/task['shard']/'status.json').read_text());source=status['sources']['track']
        if core.record(Path(source['path']))!=source: raise ValueError('old track source changed')
        prior=pd.read_parquet(source['path'],filters=[('candidate_id','in',old.old_candidate_id.tolist())])
        prior['candidate_id']=prior.candidate_id.map(dict(zip(old.old_candidate_id,old.candidate_id)))
    new=pd.read_parquet(Path(task['output'])/'ten_minute_complete_candidate.parquet',filters=[('candidate_id','in',old.candidate_id.tolist())])
    prior['candidate_id']=prior.candidate_id.astype(new.candidate_id.dtype)
    order=lambda f:f.sort_values(['candidate_id','utc']).reset_index(drop=True)
    prior,new=order(prior),order(new)
    # Gap hashes include the construction candidate ID. Earlier accepted
    # compositions also retain their original gap IDs after remapping voyages.
    # Require identical partitions and all other values; preserve the ID bridge.
    pd.testing.assert_frame_equal(prior.drop(columns='gap_id'),new.drop(columns='gap_id'),check_exact=True)
    if not prior.gap_id.isna().equals(new.gap_id.isna()): raise ValueError('gap identity missingness changed')
    gaps=pd.DataFrame({'candidate_id':new.candidate_id,'old_gap_id':prior.gap_id,'new_gap_id':new.gap_id}).dropna().drop_duplicates()
    if gaps.duplicated(['candidate_id','old_gap_id']).any() or gaps.duplicated(['candidate_id','new_gap_id']).any():
        raise ValueError('gap partition changed')
    gaps.to_parquet(Path(task['output'])/'old_gap_identity_bridge.parquet',index=False)
    prior=pd.read_parquet(old_facts_source()['path'])
    prior=prior.loc[prior.candidate_id.isin(old.old_candidate_id)].copy()
    prior['candidate_id']=prior.candidate_id.map(dict(zip(old.old_candidate_id,old.candidate_id)))
    new=pd.read_parquet(Path(task['weather'])/'voyage_weather_facts.parquet')
    new=new.loc[new.candidate_id.isin(old.candidate_id)]
    prior['candidate_id']=prior.candidate_id.astype(new.candidate_id.dtype)
    pd.testing.assert_frame_equal(prior.sort_values('candidate_id').reset_index(drop=True),
                                  new.sort_values('candidate_id').reset_index(drop=True),check_exact=True)
    return {'old_candidates':len(old),'old_track_values_exact':True,'gap_identity_bijective':True,
            'old_gap_identity_bridge':core.record(Path(task['output'])/'old_gap_identity_bridge.parquet'),
            'old_weather_facts_exact':True}


def worker(task):
    pa.set_cpu_count(1);start=time.monotonic()
    track=TRACKS.work(task)
    check=CHECK.check_shard(track)
    if any(check['problems'].values()): raise ValueError(f'track validation failed: {check}')
    core.write_json(Path(task['output'])/'validation.json',check)
    weather_check=weather.match(task,ASSETS)
    result={'shard':task['shard'],'candidates':task['candidate_rows'],'grid_rows':task['expected_grid_rows'],
            'track_validation':core.record(Path(task['output'])/'validation.json'),
            'weather_validation':core.record(Path(task['weather'])/'validation.json'),
            'facts':core.record(Path(task['weather'])/'voyage_weather_facts.parquet'),
            'seconds':time.monotonic()-start,'peak_rss_gib':track['peak_rss_gib']}
    february=Path(task['weather'])/'weather_2026-02.parquet'
    if february.exists():
        f=pd.read_parquet(february,columns=['segment_weight_hours','swh_m','position_valid'])
        result['february_hours']=float(f.segment_weight_hours.sum())
        result['february_wave_matched_hours']=float(f.loc[np.isfinite(f.swh_m),'segment_weight_hours'].sum())
        result['february_position_hours']=float(f.loc[f.position_valid,'segment_weight_hours'].sum())
    if task['probe']: result['old_parity']=old_probe_check(task)
    return result


def finalize(root,panel,results):
    source=old_facts_source()
    if core.record(Path(source['path']))!=source: raise ValueError('accepted weather facts changed')
    old=pd.read_parquet(source['path']);mapping=panel.loc[panel.old_candidate_id.notna(),['candidate_id','old_candidate_id']]
    prior=mapping.merge(old.rename(columns={'candidate_id':'old_candidate_id'}),on='old_candidate_id',how='left',validate='one_to_one')
    if len(prior)!=len(old) or prior.voyage_hours.isna().any(): raise ValueError('old facts not completely preserved')
    prior=prior.drop(columns='old_candidate_id')[old.columns]
    for r in results:
        if core.record(Path(r['facts']['path']))!=r['facts']: raise ValueError('new weather facts changed')
    added=pd.concat([pd.read_parquet(r['facts']['path']) for r in results],ignore_index=True)
    delivered=pd.concat([prior,added],ignore_index=True).sort_values('candidate_id').reset_index(drop=True)
    if delivered.candidate_id.duplicated().any() or set(delivered.candidate_id)!=set(panel.candidate_id): raise ValueError('full candidate/weather closure failed')
    expected=panel.set_index('candidate_id').total_hours
    if not np.allclose(delivered.set_index('candidate_id').voyage_hours.reindex(expected.index),expected,rtol=0,atol=1e-8): raise ValueError('full voyage hour closure failed')
    target=root/'voyage_weather_facts_2025_departures.parquet';delivered.to_parquet(target,index=False)
    # Row-level tracks and hourly weather stay at their verified source paths;
    # the bridge is the explicit identity map, avoiding copies of old large data.
    bridge=panel[['candidate_id','old_candidate_id','source_shard','weather_source']].copy()
    if BASELINE_RUN is not None:
        reference=baseline_components().set_index('candidate_id')
        for col in ['track_source_path','track_source_candidate_id','weather_source_path','weather_source_candidate_id']:
            bridge[col]=bridge.old_candidate_id.map(reference[col])
        for shard in bridge.source_shard.unique():
            selected=bridge.source_shard.eq(shard)&bridge.old_candidate_id.isna()
            bridge.loc[selected,'track_source_path']=str(root/'tracks'/shard/'ten_minute_complete_candidate.parquet')
            bridge.loc[selected,'weather_source_path']=str(root/'weather'/shard)
            for col in ['track_source_candidate_id','weather_source_candidate_id']:bridge.loc[selected,col]=bridge.loc[selected,'candidate_id']
        # Independently compare the original accepted E007 population as well.
        original_source=json.loads((E007/'status.json').read_text())['annual_output']
        if core.record(Path(original_source['path']))!=original_source:raise ValueError('original E007 facts changed')
        original=pd.read_parquet(original_source['path'])
        mapping=panel.loc[panel.e007_candidate_id.notna(),['candidate_id','e007_candidate_id']]
        actual=mapping.merge(delivered,on='candidate_id',validate='one_to_one').drop(columns='candidate_id').rename(columns={'e007_candidate_id':'candidate_id'})
        actual['candidate_id']=actual.candidate_id.astype(original.candidate_id.dtype)
        pd.testing.assert_frame_equal(original.sort_values('candidate_id').reset_index(drop=True),actual[original.columns].sort_values('candidate_id').reset_index(drop=True),check_exact=True)
    bridge.to_parquet(root/'component_bridge.parquet',index=False)
    summary={'status':'pass','retained_voyages':len(panel),'old_reused_voyages':len(prior),'new_voyages':len(added),
             'voyage_hours':float(delivered.voyage_hours.sum()),'new_voyage_hours':float(added.voyage_hours.sum()),
             'new_wave_matched_hours':float(added.wave_matched_hours.sum()),'new_unresolved_track_hours':float(added.unresolved_track_hours.sum()),
             'old_facts_exact':True,'full_hour_denominators_close':True,'new_track_grid_rows':sum(r['grid_rows'] for r in results),
             'source_old_facts':source,'output':core.record(target),'bounds_complete':False,'release_tables_complete':False,'scientific_adoption':False}
    if BASELINE_RUN is not None:
        summary.update(january_buffer_facts_exact=True,original_e007_facts_exact=True,original_e007_voyages=len(original))
        total=sum(r.get('february_hours',0) for r in results)
        matched=sum(r.get('february_wave_matched_hours',0) for r in results)
        positions=sum(r.get('february_position_hours',0) for r in results)
        if not 0<matched<=total:raise ValueError('February voyage weather support missing or invalid')
        summary.update(february_voyage_hours=total,february_wave_matched_hours=matched,
                       february_wave_coverage=matched/total,february_position_coverage=positions/total)
    core.write_json(root/'summary.json',summary)
    return summary


def main():
    global QUALITY,PORT,ASSETS,BASELINE_RUN
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--probe',action='store_true')
    p.add_argument('--verified-probe',type=Path);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--quality',type=Path,default=QUALITY);p.add_argument('--ports',type=Path,default=PORT)
    p.add_argument('--assets',type=Path,default=ASSETS);p.add_argument('--baseline-weather',type=Path);a=p.parse_args()
    QUALITY=a.quality.resolve();PORT=a.ports.resolve();ASSETS=a.assets.resolve();BASELINE_RUN=a.baseline_weather
    weather.configure(json.loads((ASSETS/'manifest.json').read_text()).get('observation_end_exclusive','2026-02-01'))
    root=core.inside_runs(a.out);root.mkdir(parents=True,exist_ok=False)
    code=bindings()
    if not a.probe:
        if a.verified_probe is None: raise ValueError('full run needs completed probe')
        verified=json.loads((a.verified_probe/'execution.json').read_text())
        if verified['status']!='pass' or not verified['probe'] or verified['bindings']!=code: raise ValueError('probe did not pass on identical inputs and code')
    state={'status':'running','probe':a.probe,'bindings':code,'workers':a.workers,'results':[],'failures':[],
           'authorization':'2026-10-22 explicit February Worker instruction' if BASELINE_RUN is not None else '2026-10-21 recorded research instruction','scientific_adoption':False}
    save=lambda:core.write_json(root/'execution.json',state);save();start=time.monotonic()
    try:
        core.write_json(root/'required_actual.json',{'status':'pass','checks':[
            {'parameter':'observation_cutoff_UTC','Required':weather.OBSERVATION_END,'Actual':weather.OBSERVATION_END,'Status':'pass'},
            {'parameter':'track_kernel','Required':'accepted production kernel without parameter overrides','Actual':core.record(PIPE/'160_build_gfw_ten_minute_tracks.py'),'Status':'pass'},
            {'parameter':'weather_calendar','Required':weather.MONTHS,'Actual':json.loads((ASSETS/'weather_config.json').read_text())['allowed_weather_months'],'Status':'pass'},
            {'parameter':'unknown_positions','Required':'retain unknown; no extrapolation','Actual':'retain unknown; no extrapolation','Status':'pass'}]})
        panel,tasks=prepare(root,a.probe)
        state.update(task_count=len(tasks),candidate_count=sum(t['candidate_rows'] for t in tasks),grid_rows=sum(t['expected_grid_rows'] for t in tasks));save()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs={pool.submit(worker,t):t['shard'] for t in tasks}
            for future in as_completed(jobs):
                try: state['results'].append(future.result())
                except BaseException: state['failures'].append({'shard':jobs[future],'error':traceback.format_exc()})
                save();print(json.dumps({'done':len(state['results']),'failed':len(state['failures'])}),flush=True)
        if state['failures']: raise RuntimeError('track/weather shards failed; see execution receipt')
        if bindings()!=code: raise ValueError('inputs or code changed during execution')
        if not a.probe: state['summary']=finalize(root,panel,state['results'])
        state['status']='pass'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state['seconds']=time.monotonic()-start;save()


if __name__=='__main__':main()
