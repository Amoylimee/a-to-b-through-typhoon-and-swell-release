#!/usr/bin/env python3
"""Rebuild changed GFW voyages, and reuse exactly matched physical voyages.

Every reused identity matches vessel, both cities and both UTC boundaries;
sequence-dependent candidate IDs are remapped explicitly. No reference AIS inputs.
"""
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
import pyarrow.compute as pc
import pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent
CODING=HERE.parents[1]
ROOT=CODING.parent
E006=CODING/'pipeline/gfw_release'
sys.path.insert(0,str(E006))
import step6_core as core
TRIM=core.module('e007_segment_track_adapter',HERE/'track_weather_adapter.py')
OLD=CODING/'.runs/2026-10-09_2353_E007-option-c-voyage-set'
RULE=HERE/'config/port_segmentation_revision.json'
NAMES={'track':'ten_minute_complete_candidate.parquet','summary':'candidate_summary.parquet'}
MONTHS=[f'weather_2025-{m:02d}.parquet' for m in range(1,13)]


def remap(table, old_ids, new_ids):
    if not old_ids:
        return table.slice(0,0)
    selected=table.filter(pc.is_in(table['candidate_id'],value_set=pa.array(old_ids,pa.string())))
    position=pc.index_in(selected['candidate_id'],value_set=pa.array(old_ids,pa.string()))
    if pc.any(pc.is_null(position)).as_py(): raise ValueError('unmapped reuse candidate')
    return selected.set_column(selected.schema.get_field_index('candidate_id'),'candidate_id',pc.take(pa.array(new_ids,pa.string()),position))


def prepare(base, out, shards, probe, replay=None):
    panel=pd.read_parquet(base/'retained_panel.parquet')
    panel['expected_grid_rows']=[len(TRIM.tracks.strict_grid_ns(a,b)) for a,b in zip(panel.departure_time,panel.arrival_time)]
    folder=out/'prepared';folder.mkdir()
    tasks=[]
    for shard in shards:
        f=panel.loc[panel.source_shard.eq(shard)].sort_values('candidate_id').copy()
        if probe is not None:
            # A probe must exercise changed and unchanged cases, rather than
            # selecting only the first unchanged voyages in the shard.
            f=pd.concat([q.head(probe) for _,q in f.groupby('weather_source',sort=True)],ignore_index=True)
        path=folder/f'{shard}.parquet';f.to_parquet(path,index=False)
        changed=f.loc[f.weather_source.eq('rebuild_gfw')]
        cpath=folder/f'{shard}_changed.parquet'
        changed[[*TRIM.tracks.CANDIDATE_COLUMNS,'expected_grid_rows']].to_parquet(cpath,index=False)
        clean=TRIM.tracks.CLEAN_ROOT/'shards'/shard/'mask'/f'{shard}.feather'
        receipt=json.loads((clean.parent/'execution.json').read_text())
        point_record={'path':str(clean),'bytes':clean.stat().st_size,'sha256':receipt['outputs'][str(clean)]}
        task={'shard':shard,'panel':core.record(path),'candidates':core.record(cpath),'points':point_record,
              'output':str(out/'rebuilt_tracks'/shard),'weather':str(out/'rebuilt_weather'/shard),
              'candidate_rows':len(changed),'expected_grid_rows':int(changed.expected_grid_rows.sum()),'out':str(out)}
        if replay is not None:task['replay_rebuilt']=str(replay.resolve())
        tasks.append(task)
    return tasks


def worker(task):
    pa.set_cpu_count(1)
    started=time.monotonic();shard=task['shard'];out=Path(task['out'])
    panel=pd.read_parquet(task['panel']['path'])
    old_state=json.loads((OLD/'shards'/shard/'status.json').read_text())
    for item in [task['panel'],*old_state['outputs'].values(),old_state['sources']['track'],old_state['sources']['summary']]:
        if core.record(Path(item['path'])) != item: raise ValueError(f'bound source changed: {item["path"]}')
    clearance={'experiment':'E007','authorization':core.record(RULE),'scope':'authorized corrected GFW segmentation downstream rebuild',
               'rules':TRIM.tracks.RULES,'reference_ais_inputs':False}
    if task['candidate_rows']:
        if 'replay_rebuilt' in task:
            previous=Path(task['replay_rebuilt'])
            candidate=previous/'prepared'/f'{shard}_changed.parquet'
            if core.record(candidate)['sha256']!=task['candidates']['sha256']:raise ValueError('replayed construction candidates differ')
            folders={'output':'rebuilt_tracks','weather':'rebuilt_weather'}
            for key,folder in folders.items():
                source=previous/folder/shard
                status=json.loads((source/'status.json').read_text())
                if status['state']!='completed':raise ValueError('replayed component incomplete')
                records=status['outputs'].values() if isinstance(status['outputs'],dict) else status['outputs']
                for item in records:
                    if core.record(Path(item['path']))!=item:raise ValueError('replayed component changed')
                target=Path(task[key]);target.parent.mkdir(exist_ok=True,parents=True);target.symlink_to(source.resolve(),target_is_directory=True)
            valid=json.loads((Path(task['weather'])/'validation.json').read_text())
            if valid['status']!='pass' or not all(valid['checks'].values()):raise ValueError('replayed weather not validated')
        else:
            TRIM.tracks.work(task)
            TRIM.weather(task,clearance)
    reuse=panel.loc[panel.weather_source.eq('reuse_verified_option_c')]
    old_ids=reuse.old_candidate_id.tolist();new_ids=reuse.candidate_id.tolist()
    weather=out/'shards'/shard;tracks=out/'tracks'/shard
    weather.mkdir(parents=True);tracks.mkdir(parents=True)
    def combine(source, changed_source, destination, sort=None):
        table=pq.read_table(source)
        kept=remap(table,old_ids,new_ids)
        if task['candidate_rows'] and changed_source.exists():
            changed=pq.read_table(changed_source)
            kept=pa.concat_tables([kept,changed.cast(table.schema)])
        elif task['candidate_rows']:
            if not changed_source.name.startswith('weather_'):raise FileNotFoundError(changed_source)
            start=pd.Timestamp(changed_source.stem.removeprefix('weather_')+'-01',tz='UTC')
            end=start+pd.offsets.MonthBegin(1)
            expected=panel.weather_source.eq('rebuild_gfw')&panel.departure_time.lt(end)&panel.arrival_time.gt(start)
            if expected.any():raise ValueError('missing month containing changed voyage time')
        if sort:kept=kept.sort_by([(c,'ascending') for c in sort])
        pq.write_table(kept,destination,compression='zstd')
        return kept.num_rows
    for kind,name in NAMES.items():
        combine(Path(old_state['sources'][kind]['path']),Path(task['output'])/name,tracks/name)
    rows=0
    for name in MONTHS:
        rows+=combine(OLD/'shards'/shard/name,Path(task['weather'])/name,weather/name,['candidate_id','hour_anchor_utc'])
    combine(OLD/'shards'/shard/'voyage_weather_facts.parquet',Path(task['weather'])/'voyage_weather_facts.parquet',weather/'voyage_weather_facts.parquet',['candidate_id'])
    facts=pd.read_parquet(weather/'voyage_weather_facts.parquet')
    if set(facts.candidate_id)!=set(panel.candidate_id) or facts.candidate_id.duplicated().any():raise ValueError('composed candidate set does not close')
    original=pd.read_parquet(OLD/'shards'/shard/'voyage_weather_facts.parquet')
    original=original.loc[original.candidate_id.isin(old_ids)].copy()
    original['candidate_id']=original.candidate_id.map(dict(zip(old_ids,new_ids)))
    pd.testing.assert_frame_equal(original.sort_values('candidate_id').reset_index(drop=True),
                                  facts.loc[facts.candidate_id.isin(new_ids)].sort_values('candidate_id').reset_index(drop=True),check_exact=True)
    joined=panel[['candidate_id','departure_time','arrival_time']].merge(facts[['candidate_id','voyage_hours']],on='candidate_id',validate='one_to_one')
    if not np.allclose((joined.arrival_time-joined.departure_time).dt.total_seconds()/3600,joined.voyage_hours,rtol=0,atol=1e-8):raise ValueError('physical voyage duration changed')
    state={**{k:v for k,v in old_state.items() if k not in ['outputs','composed_from','sources']},
           'population':'E007_gfw_corrected_segmentation_2025','candidates':len(panel),'rows':rows,
           'review_clearance':clearance,
           'sources':{**old_state['sources'],'track':core.record(tracks/NAMES['track']),'summary':core.record(tracks/NAMES['summary'])},
           'segmentation_rule':core.record(RULE),'reused_exact_physical_voyages':len(reuse),'rebuilt_voyages':task['candidate_rows'],
           'source_option_c':core.record(OLD/'shards'/shard/'status.json'),
           'outputs':{p.name:core.record(p) for p in sorted(weather.glob('*.parquet'))}}
    core.write_json(weather/'status.json',state)
    check=TRIM.weather_check.validate_shard(weather)
    if check['status']!='pass' or not all(check['checks'].values()):raise ValueError('composed weather validation failed')
    result={'shard':shard,'candidates':len(panel),'rows':rows,'reused':len(reuse),'rebuilt':task['candidate_rows'],
            'reused_facts_exact':True,'validation':'pass','seconds':time.monotonic()-started}
    core.write_json(weather/'composition_validation.json',result)
    return result


def finalize(base,out,results):
    facts=pd.concat([pd.read_parquet(out/'shards'/s/'voyage_weather_facts.parquet') for s in core.SHARDS],ignore_index=True).sort_values('candidate_id').reset_index(drop=True)
    panel=pd.read_parquet(base/'retained_panel.parquet')
    if set(facts.candidate_id)!=set(panel.candidate_id) or facts.candidate_id.duplicated().any():raise ValueError('annual weather candidate closure failed')
    path=out/'voyage_weather_facts_2025.parquet';facts.to_parquet(path,index=False)
    source=json.loads((OLD/'status.json').read_text())
    totals={c:float(facts[c].sum()) for c in source['totals']}
    rows=sum(r['rows'] for r in results)
    if not np.isclose(rows,totals['voyage_hours'],rtol=0,atol=1e-6):raise ValueError('weather hour closure failed')
    receipts=[json.loads((out/'shards'/s/'validation.json').read_text()) for s in core.SHARDS]
    checks={key:all(r['checks'][key] for r in receipts) for key in receipts[0]['checks']}
    checks.update(annual_candidate_set=True,annual_candidate_uniqueness=True,all_shards_pass=True,
                  reused_facts_exact=all(r['reused_facts_exact'] for r in results),annual_hours_close=True)
    validation={'status':'pass','complete_year':True,'shards':64,'candidates':len(facts),'rows':rows,'checks':checks,'totals':totals,
                'output':core.record(path),'adopted':False,'network_requests':0,'products_modified':0}
    core.write_json(out/'validation.json',validation)
    state={'state':'completed','failures':[],'step':5,'complete_year':True,'population':'E007_gfw_corrected_segmentation_2025',
           'total_shards':64,'completed_shards':core.SHARDS,'candidates':len(facts),'rows':rows,'totals':totals,
           'annual_output':core.record(path),'adopted':False,'network_requests':0,'products_modified':0}
    core.write_json(out/'status.json',state)
    decision=json.loads((OLD/'step6_decision.json').read_text())
    cities=base/'input_view/results/support/city_snap_table_2025.csv'
    counts=panel.groupby(core.KEY,observed=True).size()
    decision.update(voyage_set='E007 corrected segmentation with unresolved cases quarantined',voyage_set_rule=core.record(RULE),
                    weather_run=str(out.relative_to(ROOT)),weather_validation_sha256=core.record(out/'validation.json')['sha256'],
                    annual_weather_sha256=core.record(path)['sha256'],gfw_city_table=str(cities.relative_to(ROOT)),
                    gfw_city_table_sha256=core.record(cities)['sha256'],required_candidates=len(facts),
                    expected_counts={'candidates':len(facts),'weather_rows':rows,'combinations':len(counts),
                                     'release_combinations':int(counts.ge(5).sum()),'release_voyages':int(counts[counts.ge(5)].sum())})
    core.write_json(out/'step6_decision.json',decision)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--shards',nargs='*');p.add_argument('--probe-candidates',type=int)
    p.add_argument('--replay-rebuilt',type=Path,help='Reuse independently validated rebuilt probe components; construction candidate hashes must match')
    args=p.parse_args();out=core.inside_runs(args.out);out.mkdir(parents=True,exist_ok=False)
    shards=sorted(args.shards or core.SHARDS)
    if not set(shards)<=set(core.SHARDS):raise ValueError('invalid shards')
    tasks=prepare(args.base.resolve(),out,shards,args.probe_candidates,args.replay_rebuilt)
    start=time.monotonic();results=[];failures=[]
    state={'state':'running','complete':shards==core.SHARDS and args.probe_candidates is None,'workers':args.workers,
           'base':str(args.base.resolve()),'rule':core.record(RULE),'code':core.record(Path(__file__)),'results':[],'failures':[]}
    core.write_json(out/'execution.json',state)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(worker,t):t['shard'] for t in tasks}
        for future in as_completed(jobs):
            try:
                r=future.result();results.append(r);print(json.dumps({'done':len(results),**r}),flush=True)
            except Exception:failures.append({'shard':jobs[future],'error':traceback.format_exc()});print('FAILED '+jobs[future],flush=True)
            state.update(results=results,failures=failures,elapsed_seconds=time.monotonic()-start)
            core.write_json(out/'execution.json',state)
    if not failures and state['complete']:state['annual']=finalize(args.base.resolve(),out,results)
    state.update(state='failed' if failures else 'completed',wall_seconds=time.monotonic()-start)
    core.write_json(out/'execution.json',state)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
