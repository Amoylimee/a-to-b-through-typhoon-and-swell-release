"""Apply accepted C2/C3, frozen speed thresholds and port guards to E009."""
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[2]
PIPE = CODING / 'pipeline/gfw_release'
sys.path.insert(0, str(PIPE))
import step6_core as core
from port_segmentation import guard_candidates, distances
BASE = core.module('e009_quality_base', PIPE / '130_build_gfw_voyage_base.py')
CAND = CODING / '.runs/2026-10-21_2154_E009-candidate-port-view/candidates'
PORT = CAND.parent
OLD = CODING / '.runs/2026-10-11_1907_E007-one-cell-voyage-base'
NATURAL = ['vessel_uid_clean','origin_city','dest_city','departure_time','arrival_time']
Q99 = .04680512383582364
OBSERVATION_END='2026-02-01'
PROBE=False


def compare_old(old, new, columns):
    x = old[NATURAL + columns].merge(new[NATURAL + columns], on=NATURAL, how='left',
                                   suffixes=('_old','_new'), validate='one_to_one', indicator=True)
    if not x._merge.eq('both').all(): raise ValueError('old pretrim voyage disappeared')
    for c in columns:
        a, b = x[c+'_old'], x[c+'_new']
        if not (a.eq(b) | (a.isna() & b.isna())).all(): raise ValueError(f'old quality value changed: {c}')


def guard(shard, root, result):
    pa.set_cpu_count(1)
    f = pd.read_parquet(root/'pretrim_panel.parquet', filters=[('source_shard','=',shard)])
    visits = pd.read_parquet(PORT/'ports/shards'/shard/'visit_quality.parquet')
    for c in ['entry_time','exit_time','first_seen_time','last_seen_time']:
        visits[c] = pd.to_datetime(visits[c], utc=True)
    p = BASE.verify(result['source'])
    points = pd.read_feather(p, columns=['vessel_uid','UTC','lat','lon'])
    points['UTC'] = pd.to_datetime(points.UTC, utc=True)
    f = guard_candidates(f, visits, points)
    f['retained'] = f.speed_decision.eq('retained') & f.segmentation_eligible & f.temporal_scope_eligible
    f.to_parquet(root/'guards'/f'{shard}.parquet', index=False)
    return {'shard':shard,'rows':len(f),'retained':int(f.retained.sum()),'source':result['source'],
            'output':core.record(root/'guards'/f'{shard}.parquet')}


def prepare(root, workers, state, save):
    source = json.loads((CAND/'execution.json').read_text())
    if source['status'] != 'pass': raise ValueError('candidate enumeration is not complete')
    if source['observation_window_utc'][1]!=OBSERVATION_END:raise ValueError('candidate observation cutoff differs')
    src = Path(source['input_view']['view']); view = root/'input_view'; view.mkdir()
    for p in src.iterdir():
        if p.name != 'results': (view/p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
    out = view/'results'; out.mkdir()
    intermediate = out/'intermediate/2025'; intermediate.mkdir(parents=True)
    (intermediate/'city_episodes').symlink_to((src/'results/intermediate/2025/city_episodes').resolve(), target_is_directory=True)
    occurrences = intermediate/'candidate_occurrences'
    (occurrences/'candidate_passages').mkdir(parents=True)
    (occurrences/'raw_stationary_episodes').mkdir()
    for p in sorted((src/'results/intermediate/2025/candidate_occurrences/candidate_passages').glob('part-*.feather')):
        f = pd.read_feather(p)
        t = pd.to_datetime(f.departure_time, utc=True)
        f = f.loc[t.ge('2025-01-01') & t.lt('2026-01-01')].reset_index(drop=True)
        f.to_feather(occurrences/'candidate_passages'/p.name)
        stop = src/'results/intermediate/2025/candidate_occurrences/raw_stationary_episodes'/p.name
        (occurrences/'raw_stationary_episodes'/p.name).symlink_to(stop.resolve())
    city_reference = OLD/'input_view/results/support/city_snap_table_2025.csv'
    builder, _ = BASE.configure_builder(view, city_reference, Path(source['input_view']['mapping']['path']))
    episode = json.loads((intermediate/'city_episodes/output_manifest.json').read_text())
    full = pd.read_parquet(CAND/'departure_2025_candidates.parquet')
    pairs = full.groupby(['pair_id','origin_city','dest_city'], observed=True).agg(
        candidate_count_2025=('candidate_id','size'), vessel_count_2025=('vessel_uid_clean','nunique')).reset_index()
    pair_path = out/'departure_pair_manifest.csv'; pairs.to_csv(pair_path, index=False)
    episode['outputs']['pair_manifest'] = {'path':str(pair_path.relative_to(view)), 'size_bytes':pair_path.stat().st_size,
                                          'sha256':BASE.sha(pair_path)}
    state['phase'] = 'city_extension_and_yardsticks'; save()
    city = builder.build_combined_city_snaps((2025,), out, {2025:episode})
    # Preserve accepted CSV field strings as well as values: a pandas CSV
    # round trip can alter the last floating-point bit of an unchanged city.
    with city_reference.open(newline='') as handle:
        reader=csv.DictReader(handle); fields=reader.fieldnames; original_rows=list(reader)
    with city.open(newline='') as handle: generated_rows=list(csv.DictReader(handle))
    required={r['city_id'] for r in generated_rows}; old_ids={r['city_id'] for r in original_rows}
    rows=[r for r in original_rows if r['city_id'] in required]+[r for r in generated_rows if r['city_id'] not in old_ids]
    with city.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(sorted(rows,key=lambda r:r['city_id']))
    snap=city.parent/'city_snap_summary.json'
    if snap.exists():
        before=json.loads(snap.read_text());core.write_json(city.parent/'city_snap_summary_before_text_preservation.json',before)
        before['output_sha256']=BASE.sha(city);before['e009_old_field_text_preserved']=True;core.write_json(snap,before)
    original = pd.read_csv(city_reference).set_index('city_id'); updated = pd.read_csv(city).set_index('city_id')
    common = original.index.intersection(updated.index)
    pd.testing.assert_frame_equal(original.loc[common], updated.loc[common], check_exact=True)
    state['new_cities'] = sorted(set(updated.index)-set(original.index)); save()
    # Route only new pairs. Old yardstick values are preserved byte-for-value.
    prior = pd.read_feather(OLD/'input_view/results/support/2025/yardstick_table_2025.feather')
    missing = pairs.loc[~pairs.pair_id.isin(prior.pair_id)]
    fresh = prior.iloc[:0]
    if len(missing):
        missing.to_csv(pair_path,index=False)
        episode['outputs']['pair_manifest'].update(size_bytes=pair_path.stat().st_size,sha256=BASE.sha(pair_path))
        fresh_path = builder.build_year_yardstick(2025,out/'new_pairs',episode,city,workers)
        fresh = pd.read_feather(fresh_path)
    yardstick = out/'support/2025/yardstick_table_2025.feather'; yardstick.parent.mkdir(parents=True,exist_ok=True)
    y = pd.concat([prior,fresh],ignore_index=True)
    y = pairs.merge(y.drop(columns=['candidate_count_2025','vessel_count_2025','origin_city','dest_city']),on='pair_id',validate='one_to_one')
    if y.d_yardstick.isna().any(): raise ValueError('missing pair yardstick')
    y.to_feather(yardstick)
    pairs.to_csv(pair_path,index=False)
    state.update(phase='c2_c3', new_pair_count=len(missing)); save()
    part_count=2 if PROBE else 64
    c2 = builder.materialize_c2(2025,out,part_count,BASE.MODE)
    transition = builder.build_transition_yardstick(2025,out,c2,city,yardstick,workers,(2025,))
    filtering = builder.downstream_config(2025,yardstick,workers,BASE.MODE); filtering['candidate_part_count']=part_count
    if any(filtering[k] != v for k,v in BASE.PROTECTED_FILTERS.items()): raise ValueError('quality thresholds changed')
    artifact = intermediate/'filters'
    c3 = builder.rebuild.run_c3(2025,filtering,c2,transition,artifact)
    files = builder.rebuild.prepare_c4_parts(2025,filtering,artifact)
    f = pd.concat([pd.read_feather(p) for p in files],ignore_index=True)
    for c in ['departure_time','arrival_time']: f[c]=pd.to_datetime(f[c],utc=True)
    for c in ['candidate_id','vessel_uid_clean','mmsi']: f[c]=f[c].astype(str)
    # This continuation is explicitly the departure cohort, not a 2025-arrival cut.
    f['temporal_scope_eligible'] = f.departure_time.ge('2025-01-01') & f.departure_time.lt('2026-01-01') & f.arrival_time.lt(OBSERVATION_END)
    cities=pd.read_csv(city).set_index('city_id'); a,b=cities.reindex(f.origin_city),cities.reindex(f.dest_city)
    hours=(f.arrival_time-f.departure_time).dt.total_seconds()/3600
    f['straight_line_knots']=distances(a.port_lat.to_numpy(),a.port_lon.to_numpy(),b.port_lat.to_numpy(),b.port_lon.to_numpy())/(hours+1)/1.852
    ratio=np.isfinite(f.normalized_time_ratio)&f.d_yardstick.gt(0)
    f['speed_decision']=np.select([ratio & f.normalized_time_ratio.gt(Q99),f.straight_line_knots.gt(36)],
                                   ['slow_above_frozen_q99','straight_line_above_36_kn'],'retained')
    ports=json.loads((PORT/'execution.json').read_text())
    owners=pd.concat([pd.read_csv(r['vessel_input_disposition']['path'],dtype={'vessel_uid':str})[['vessel_uid']].assign(source_shard=r['shard']) for r in ports['results']])
    if owners.vessel_uid.duplicated().any(): raise ValueError('vessel shard duplicate')
    f['source_shard']=f.vessel_uid_clean.map(owners.set_index('vessel_uid').source_shard)
    if f.source_shard.isna().any() or f.candidate_id.duplicated().any(): raise ValueError('invalid candidate ownership')
    old=pd.read_parquet(OLD/'pretrim_panel.parquet')
    if PROBE:old=old.loc[old.source_shard.isin(f.source_shard.unique())]
    compare_old(old,f,['d_yardstick','normalized_time_ratio','straight_line_knots','speed_decision'])
    f.to_parquet(root/'pretrim_panel.parquet',index=False)
    state.update(phase='guard_probe',c2=c2['summary'],c3=c3['summary'],pretrim=len(f),
                 old_pretrim_quality_exact=True,city_reference=core.record(city_reference),city_output=core.record(city))
    save()


def main():
    global CAND,PORT,OLD,OBSERVATION_END,PROBE
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--candidates',type=Path,default=CAND);p.add_argument('--ports',type=Path,default=PORT)
    p.add_argument('--baseline-quality',type=Path,default=OLD);p.add_argument('--observation-end',default=OBSERVATION_END)
    p.add_argument('--probe',action='store_true');p.add_argument('--verified-probe',type=Path);a=p.parse_args()
    CAND=a.candidates.resolve();PORT=a.ports.resolve();OLD=a.baseline_quality.resolve();OBSERVATION_END=a.observation_end;PROBE=a.probe
    if OBSERVATION_END not in {'2026-02-01','2026-03-01'}:raise ValueError('unauthorized observation cutoff')
    if Q99!=.04680512383582364:raise ValueError('frozen q99 differs')
    root=core.inside_runs(a.out);root.mkdir(parents=True,exist_ok=False)
    state={'status':'running','phase':'prepare','authorization':'2026-10-22 explicit February Worker instruction' if OBSERVATION_END=='2026-03-01' else '2026-10-21 recorded research instruction',
           'analysis_departure_year':2025,'observation_end_exclusive':OBSERVATION_END,'q99':Q99,'probe':PROBE,
           'city_policy':'preserve accepted coordinates; existing builder supplies missing cities only',
           'trajectory_policy':'buffered cleaning; every historical difference requires reproduction and no retained-support overlap',
           'sources':[core.record(CAND/'execution.json'),core.record(PORT/'execution.json'),core.record(OLD/'retained_panel.parquet')],
           'code':[core.record(Path(__file__)),*BASE.code_records(),core.record(PIPE/'port_segmentation.py')],
           'scientific_adoption':False,'results':[],'failures':[]}
    save=lambda:core.write_json(root/'execution.json',state)
    save();start=time.monotonic()
    try:
        if OBSERVATION_END=='2026-03-01' and not PROBE:
            if a.verified_probe is None:raise ValueError('full quality build requires two-shard probe')
            previous=json.loads((a.verified_probe/'execution.json').read_text())
            if previous['status']!='pass' or not previous['probe'] or previous['code']!=state['code']:
                raise ValueError('quality probe did not pass with current code')
        core.write_json(root/'required_actual.json',{'status':'pass','checks':[
            {'parameter':'q99','Required':.04680512383582364,'Actual':Q99,'Status':'pass' if Q99==.04680512383582364 else 'failed'},
            {'parameter':'quality_filters','Required':BASE.PROTECTED_FILTERS,'Actual':BASE.PROTECTED_FILTERS,'Status':'pass'},
            {'parameter':'observation_cutoff_UTC','Required':OBSERVATION_END,'Actual':OBSERVATION_END,'Status':'pass'},
            {'parameter':'city_reference','Required':'preserve all baseline coordinate field strings','Actual':str(OLD/'input_view/results/support/city_snap_table_2025.csv'),'Status':'pass'}]})
        prepare(root,a.workers,state,save)
        (root/'guards').mkdir()
        results={r['shard']:r for r in json.loads((PORT/'execution.json').read_text())['results']}
        probes=sorted(results) if PROBE else ['shard_0032','shard_0057']
        with ProcessPoolExecutor(max_workers=2) as pool:
            state['results']=[f.result() for f in [pool.submit(guard,s,root,results[s]) for s in probes]]
        old=pd.read_parquet(OLD/'candidate_decisions.parquet')
        if PROBE:old=old.loc[old.source_shard.isin(probes)].copy()
        probe=pd.concat([pd.read_parquet(root/'guards'/f'{s}.parquet') for s in probes],ignore_index=True)
        compare_old(old.loc[old.source_shard.isin(probes)],probe,['segmentation_review_reasons','retained'])
        state.update(phase='all_guards',probe_old_decisions_exact=True);save()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs={pool.submit(guard,s,root,r):s for s,r in results.items() if s not in probes}
            for future in as_completed(jobs):
                state['results'].append(future.result());save();print(json.dumps({'guard_done':len(state['results'])}),flush=True)
        f=pd.concat([pd.read_parquet(root/'guards'/f'{s}.parquet') for s in sorted(results)],ignore_index=True)
        compare_old(old,f,['d_yardstick','normalized_time_ratio','speed_decision','segmentation_review_reasons','retained'])
        if a.verified_probe is not None:
            prior=pd.read_parquet(a.verified_probe/'candidate_decisions.parquet')
            compare_old(prior,f,['d_yardstick','normalized_time_ratio','speed_decision','segmentation_review_reasons','retained'])
            state['probe_quality_decisions_exact']=True
        f.to_parquet(root/'candidate_decisions.parquet',index=False)
        kept=f.loc[f.retained].copy()
        bridge=kept.merge(old.loc[old.retained,['candidate_id',*NATURAL]].rename(columns={'candidate_id':'old_candidate_id'}),on=NATURAL,how='left',validate='one_to_one')
        bridge['weather_source']=np.where(bridge.old_candidate_id.notna(),'reuse_verified_e007','rebuild_gfw')
        if OBSERVATION_END=='2026-03-01':
            prior_kept=pd.read_parquet(OLD/'retained_panel.parquet',columns=['candidate_id','old_candidate_id'])
            bridge['e007_candidate_id']=bridge.old_candidate_id.map(prior_kept.set_index('candidate_id').old_candidate_id)
            bridge['weather_source']=np.where(bridge.old_candidate_id.notna(),'reuse_verified_january_buffer','rebuild_gfw')
        bridge.to_parquet(root/'retained_panel.parquet',index=False)
        bridge.loc[bridge.old_candidate_id.isna()].to_parquet(root/'new_retained_panel.parquet',index=False)
        summary={'status':'pass','probe':PROBE,'pretrim':len(f),'retained':len(kept),'old_retained':int(old.retained.sum()),
                 'new_retained':int(bridge.old_candidate_id.isna().sum()),'old_quality_decisions_exact':True,
                 'speed_decisions':f.speed_decision.value_counts().to_dict(),
                 'guard_reasons':f.loc[~f.segmentation_eligible,'segmentation_review_reasons'].str.split(';').explode().value_counts().to_dict(),
                 'cross_year_retained':int(kept.arrival_time.ge('2026-01-01').sum()),'new_cities':state['new_cities'],
                 'track_weather_complete':False,'scientific_adoption':False}
        core.write_json(root/'summary.json',summary)
        state.update(status='pass',phase='quality_complete',summary=summary)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state['seconds']=time.monotonic()-start;save()


if __name__=='__main__':main()
