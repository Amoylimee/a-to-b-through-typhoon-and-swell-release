#!/usr/bin/env python3
"""Re-clean 2025 plus January and replay accepted visits without a calendar cut."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf

HERE = Path(__file__).resolve().parents[1]
CODING = HERE.parents[1]
PIPE = CODING/'pipeline/gfw_release'
sys.path.insert(0, str(PIPE))
from port_segmentation import visit_builder, nearest_unique, EXTRA_COLUMNS

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj; spec.loader.exec_module(obj)
    return obj

unified = module('e009_buffer_unified', PIPE/'100_build_gfw_unified_shards.py')
clean = module('e009_buffer_clean', PIPE/'110_run_gfw_cleaning.py')
ports = module('e009_buffer_ports', PIPE/'120_run_gfw_port_calls.py')
OLD_UNIFIED = CODING/'.runs/2026-10-05_1616_E006-gfw-unified-shards'
OLD_CLEAN = CODING/'.runs/2026-10-05_2305_E006-gfw-cleaning'
OLD_PORTS = CODING/'.runs/2026-10-11_1907_E007-one-cell-port-segmentation'
CORRECTIONS = PIPE/'config/t034_anchorage_city_corrections.csv'
BOUNDARY = pd.Timestamp('2026-01-01')
END = pd.Timestamp('2026-02-01')


def buffered_table(old, january, baseline_end=None, observation_end=None):
    baseline_end=pd.Timestamp(baseline_end) if baseline_end is not None else BOUNDARY
    observation_end=pd.Timestamp(observation_end) if observation_end is not None else END
    utc = january.hour_utc.dt.tz_convert('UTC').dt.tz_localize(None) + unified.STAMP
    new = pd.DataFrame({'vessel_uid': january.vessel_uid.astype('int64'), 'mmsi': january.mmsi.astype('int64'),
        'imo': january.imo.astype('Int64'), 'UTC': utc, 'UTC_10min': utc.dt.floor('10min'),
        'year': utc.dt.year.astype('int16'), 'lat': january.lat.astype('float64'), 'lon': january.lon.astype('float64'),
        'speed': np.nan, 'course': np.nan, 'heading': np.nan, 'nav_status': unified.S.UNKNOWN,
        'nav_status_code': pd.array([pd.NA]*len(january), dtype='Int16'),
        'source_provider': unified.PROVIDER, 'source_resolution_original': unified.RESOLUTION})
    if not (new.UTC.ge(baseline_end) & new.UTC.lt(observation_end)).all():
        raise ValueError('buffer outside January')
    if not (old.UTC.ge('2025-01-01') & old.UTC.lt(baseline_end)).all():
        raise ValueError('baseline outside 2025')
    joined = pd.concat([old, new[unified.S.SHARD_COLUMNS]], ignore_index=True)
    joined = joined.sort_values(['vessel_uid','UTC_10min','UTC','mmsi'], kind='stable').reset_index(drop=True)
    if joined.duplicated(['vessel_uid','UTC_10min']).any():
        raise ValueError('duplicate vessel/time bin after joining buffer')
    if not joined.year.eq(joined.UTC.dt.year).all():
        raise ValueError('observation year differs from UTC')
    return pa.Table.from_pandas(joined[unified.S.SHARD_COLUMNS], schema=unified.S.SHARD_SCHEMA, preserve_index=False)


def compare_old_visits(shard, new, old_path=None):
    old_path = old_path or OLD_PORTS/'shards'/shard/'visit_quality.parquet'
    old = pd.read_parquet(old_path)
    # An old closed visit precedes all added observations and must remain exact.
    closed = old.loc[old.exit_time.notna()].copy()
    joined = closed[['visit_id_internal']].merge(new, on='visit_id_internal', how='left', validate='one_to_one')
    if len(joined)!=len(closed):
        raise ValueError('old closed visit disappeared')
    pd.testing.assert_frame_equal(closed.reset_index(drop=True), joined[closed.columns].reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    return {'old_visits':len(old), 'old_closed_visits_exact':len(closed), 'old_open_visits':int(old.exit_time.isna().sum()),
            'source': ports.record(old_path)}


def audit_boundary_changes(task, old_points, previous, points, unified_path):
    """Reproduce every affected vessel, and reject overlap with retained support."""
    root=Path(task['root'])/'boundary_audits'/task['shard'];root.mkdir(parents=True,exist_ok=False)
    keys=['vessel_uid','UTC'];values=['lat','lon','calc_speed_knots_masked']
    merged=old_points.merge(previous,on=keys,how='outer',suffixes=('_old','_new'),indicator=True,validate='one_to_one')
    changed=merged._merge.ne('both')
    for c in values:
        changed |= ~(merged[c+'_old'].eq(merged[c+'_new']) | (merged[c+'_old'].isna() & merged[c+'_new'].isna()))
    changes=merged.loc[changed].copy();changes.to_parquet(root/'changed_historical_points.parquet',index=False)
    vessels=set(changes.vessel_uid.astype('int64'))
    retained=pd.read_parquet(task['baseline_panel']['path'],columns=['candidate_id','vessel_uid_clean','departure_time','arrival_time','source_shard'],filters=[('source_shard','=',task['shard'])])
    retained=retained[retained.vessel_uid_clean.astype('int64').isin(vessels)].copy()
    for c in ['departure_time','arrival_time']:retained[c]=pd.to_datetime(retained[c],utc=True).dt.tz_localize(None)
    joined=changes[keys].assign(vessel_uid_clean=changes.vessel_uid.astype(str)).merge(retained,on='vessel_uid_clean',how='left')
    overlap=joined.UTC.ge(joined.departure_time-pd.Timedelta(hours=1)) & joined.UTC.le(joined.arrival_time+pd.Timedelta(hours=1))
    joined.loc[overlap].to_parquet(root/'retained_support_overlap.parquet',index=False)
    audit={'status':'failed','scope':'retained_voyage_inputs','shard':task['shard'],'changed_points':len(changes),
           'affected_vessels':len(vessels),'changes_inside_retained_support':int(overlap.sum()),
           'support_padding_hours':1,'baseline_panel':task['baseline_panel'],
           'changes':ports.record(root/'changed_historical_points.parquet'),
           'old_cleaning_reproduced':False,'extended_cleaning_reproduced':False,'historical_trajectory_accepted':False}
    ports.write_json(root/'audit.json',audit)
    if overlap.any():raise ValueError('historical point difference touches a retained voyage; stop')
    columns=['vessel_uid','UTC',*values]
    for label,source,expected in [('old',Path(task['old']['path']),old_points),('extended',unified_path,points)]:
        part=pd.read_feather(source);part=part[part.vessel_uid.astype('int64').isin(vessels)].reset_index(drop=True)
        folder=root/(label+'_input');folder.mkdir();path=folder/(task['shard']+'.feather');part.to_feather(path)
        output=root/(label+'_reproduction');clean.run_shard((str(path),str(output),2.0))
        actual=pd.read_feather(output/'mask'/path.name,columns=columns)
        expected=expected[expected.vessel_uid.astype('int64').isin(vessels)][columns]
        order=lambda f:f.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(order(actual),order(expected),check_dtype=False,check_exact=True)
        audit[label+'_cleaning_reproduced' if label=='old' else 'extended_cleaning_reproduced']=True
        audit[label+'_reproduction_output']=ports.record(output/'mask'/path.name)
        ports.write_json(root/'audit.json',audit)
    audit['status']='pass';ports.write_json(root/'audit.json',audit)
    return ports.record(root/'audit.json')


def worker(task):
    pa.set_cpu_count(1)
    started = time.monotonic()
    root, shard = Path(task['root']), task['shard']
    end=pd.Timestamp(task.get('observation_end','2026-02-01'))
    baseline_end=pd.Timestamp(task.get('baseline_end','2026-01-01'))
    baseline=task.get('baseline_result')
    old_path, january_path = Path(task['old']['path']), Path(task['january']['path'])
    for item in [task['old'], task['january']]:
        if clean.sha(Path(item['path'])) != item['sha256']:
            raise ValueError('source changed: '+item['path'])
    unified_dir = root/'unified'; unified_dir.mkdir(exist_ok=True)
    source = unified_dir/(shard+'.feather')
    if source.exists():
        raise ValueError('refuse to overwrite a started shard')
    table = buffered_table(pd.read_feather(old_path), pd.read_parquet(january_path),baseline_end,end)
    pf.write_feather(table, source)
    input_rows = table.num_rows
    del table
    clean_report = clean.run_shard((str(source), str(root/'cleaning/shards'/shard), 2.0))
    masked = root/'cleaning/shards'/shard/'mask'/(shard+'.feather')
    stage = ports.load_module('e009_ports_stage', ports.PORT_STAGE/'10_build_port_calls_and_legs.py')
    stage.VESSEL_SPECS_PATH_TEMPLATE = root/'analysis_vessel_specs_2025.csv'
    if any(getattr(stage,k)!=v for k,v in ports.PORT_PARAMETERS.items()):
        raise ValueError('protected port thresholds changed')
    catalog, _ = stage.load_anchorages(ports.CATALOG)
    fixes = pd.read_csv(CORRECTIONS,dtype={'anchorage_id':str}).set_index('anchorage_id').new_analytical_city_id
    catalog['city'] = catalog.s2id.astype(str).map(fixes).fillna(catalog.analytical_city_id)
    stage.build_vessel_visits = visit_builder(catalog.city.to_numpy())
    stage.query_nearest_anchorage = nearest_unique
    stage.VISIT_OUTPUT_COLUMNS = [*stage.VISIT_OUTPUT_COLUMNS, *EXTRA_COLUMNS]
    output = root/'ports/shards'/shard; output.mkdir(parents=True, exist_ok=False)
    pipeline_task = stage.build_task(masked, 2025, output, output/'tasks', ports.CATALOG)
    result = stage.process_task(pipeline_task, output)
    if result['status']!='success' or result['n_input_rows']!=result['n_filtered_rows']:
        raise ValueError('port replay failed or dropped roster vessels: '+str(result))
    visits = pd.read_feather(result['visits_output_path'])
    visits['analytical_city_id'] = visits.anchorage_id.astype(str).map(fixes).fillna(visits.analytical_city_id)
    visits.to_parquet(output/'visit_quality.parquet',index=False)
    legs = pd.read_feather(result['voyages_output_path'])
    if visits.visit_id_internal.duplicated().any() or legs.internal_voyage_id.duplicated().any():
        raise ValueError('duplicate visit or raw-leg ID')
    high=set(visits.loc[visits.visit_confidence.eq('high'),'visit_id_internal'])
    if not (set(legs.origin_visit_id_internal)|set(legs.dest_visit_id_internal))<=high:
        raise ValueError('low-confidence raw-leg endpoint')
    if not (legs.trip_end.gt(legs.trip_start)&legs.trip_start.ge('2025-01-01')&legs.trip_end.lt(end)).all():
        raise ValueError('raw legs outside observation interval')
    old_visit_path=Path(baseline['visits_output_path']).parents[1]/'visit_quality.parquet' if baseline else None
    comparison = compare_old_visits(shard,visits,old_visit_path)
    columns=['vessel_uid','UTC','lat','lon','calc_speed_knots_masked']
    old_points=pd.read_feather(Path(baseline['source']['path']) if baseline else OLD_CLEAN/'shards'/shard/'mask'/(shard+'.feather'),columns=columns)
    points=pd.read_feather(masked,columns=columns)
    previous=points.loc[points.UTC.lt(baseline_end)].reset_index(drop=True)
    # Re-cleaning the same annual observations must not alter historical points.
    boundary_audit=None
    try:pd.testing.assert_frame_equal(previous,old_points,check_dtype=False,check_exact=True)
    except AssertionError:
        if not baseline:raise
        boundary_audit=audit_boundary_changes(task,old_points,previous,points,source)
    disposition=points.groupby('vessel_uid').agg(cleaned_rows=('UTC','size'),first_cleaned_utc=('UTC','min'),last_cleaned_utc=('UTC','max'))
    disposition['port_visits']=visits.groupby('vessel_uid').size().reindex(disposition.index.astype(str)).fillna(0).to_numpy(dtype=int)
    disposition.reset_index().to_csv(output/'vessel_input_disposition.csv',index=False)
    old_legs=pd.read_feather(Path(baseline['voyages_output_path']) if baseline else OLD_PORTS/'shards'/shard/'shard_voyages'/(shard+'_harmonized_od_voyages.feather'))
    old_ids=set(old_legs.internal_voyage_id)
    # Existing complete raw legs must also remain byte-value-equivalent.
    retained=legs.loc[legs.internal_voyage_id.isin(old_ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(retained,old_legs,check_dtype=False,check_exact=True)
    added=legs.loc[~legs.internal_voyage_id.isin(old_ids)&legs.trip_start.lt(BOUNDARY)].copy()
    added.to_parquet(output/'added_2025_departure_raw_legs.parquet',index=False)
    unresolved=visits.loc[visits.exit_time.isna()&visits.entry_time.lt(BOUNDARY)].copy()
    unresolved.to_parquet(output/'still_open_2025_visits.parquet',index=False)
    result.update(shard=shard, input_rows_with_buffer=input_rows, baseline_points_exact=boundary_audit is None,
        boundary_retained_input_audit=boundary_audit,retained_input_points_exact=True,
        baseline_raw_legs_exact=len(old_legs), old_visit_comparison=comparison,
        added_2025_departure_raw_legs=len(added), added_cross_year_arrivals=int(added.trip_end.ge(BOUNDARY).sum()),
        added_within_2025_arrivals=int(added.trip_end.lt(BOUNDARY).sum()),
        still_open_2025_visits=len(unresolved), january_cleaned_rows=int(points.UTC.ge(BOUNDARY).sum()),
        clean_report=clean_report, wall_seconds=time.monotonic()-started,
        source=ports.record(masked), vessel_input_disposition=ports.record(output/'vessel_input_disposition.csv'),
        outputs=[ports.record(Path(result[k])) for k in ['visits_output_path','voyages_output_path']])
    ports.write_json(output/'execution.json',result)
    return result


def run(buffer,root,shards,workers,baseline_visits=None,baseline_panel=None,observation_end='2026-02-01',verified_probe=None):
    root.mkdir(parents=True,exist_ok=False)
    for path,digest in ports.EXPECTED.items():
        if clean.sha(path)!=digest: raise ValueError('frozen input changed')
    summary=json.loads((buffer/'fleet_presence_unique_summary.json').read_text())
    if summary['status']!='pass' or not summary['closure'] or not summary['unique_mmsi_hour']:
        raise ValueError('January fleet buffer not verified')
    previous={};baseline_end='2026-01-01'
    if baseline_visits:
        if observation_end!='2026-03-01' or baseline_panel is None:raise ValueError('February replay requires prior retained panel and March 1 cutoff')
        state_before=json.loads((baseline_visits/'execution.json').read_text())
        if state_before['status']!='pass' or not state_before['complete_year']:raise ValueError('prior visit run not complete')
        previous={v['shard']:v for v in state_before['results']}
        baseline={s:ports.record(Path(v['source']['path']).resolve().parents[4]/'unified'/(s+'.feather')) for s,v in previous.items()}
        baseline_end='2026-02-01'
    else:
        old=json.loads((OLD_UNIFIED/'manifest.json').read_text())
        baseline={row['shard'].removesuffix('.feather'):dict(row,path=str(OLD_UNIFIED/'unified/2025'/row['shard'])) for row in old['shards']}
    january={row['shard']:row for row in summary['buckets']}
    if len(shards)!=len(set(shards)) or not set(shards)<=set(baseline): raise ValueError('invalid shards')
    specs=root/'analysis_vessel_specs_2025.csv';ports.write_analysis_specs(specs)
    code=[Path(__file__),Path(clean.__file__),Path(unified.__file__),Path(ports.__file__),
          PIPE/'port_segmentation.py',CORRECTIONS,clean.CONFIG,*clean.CLEAN.glob('*.py'),*ports.PORT_STAGE.glob('*.py')]
    state={'status':'running','analysis_year':2025,'observation_window_utc':['2025-01-01',observation_end],
        'authorization':'2026-10-22 explicit February Worker instruction' if baseline_visits else '2026-10-21 recorded research instruction', 'source_buffer':str(buffer),
        'parameters':{'speed_allowance':2.0,**ports.PORT_PARAMETERS},'shards':shards,'workers':workers,
        'code':[ports.record(p) for p in code],'specs':ports.record(specs),'catalog':ports.record(ports.CATALOG),
        'results':[],'failures':[],'scientific_adoption':False,'network_requests':0}
    state['baseline_visit_run']=str(baseline_visits) if baseline_visits else None
    state['baseline_panel']=ports.record(baseline_panel) if baseline_panel else None
    if baseline_visits and len(shards)>2:
        if verified_probe is None:raise ValueError('full replay requires two-shard probe')
        probe=json.loads((verified_probe/'execution.json').read_text())
        if (probe['status']!='pass' or len(probe['results'])!=2 or probe['code']!=state['code']
                or probe['baseline_panel']!=state['baseline_panel'] or probe['baseline_visit_run']!=state['baseline_visit_run']
                or probe['source_buffer']!=str(buffer)):
            raise ValueError('port probe code or baseline mismatch')
        state['results']=probe['results'];state['verified_probe']=ports.record(verified_probe/'execution.json')
        for result in probe['results']:
            shard=result['shard']
            source_dir=Path(result['visits_output_path']).parents[1]
            target=root/'ports/shards'/shard;target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(source_dir.resolve(),target_is_directory=True)
    checks=[{'parameter':'observation_cutoff_UTC','Required':observation_end,'Actual':observation_end,'Status':'pass'},
            {'parameter':'speed_allowance','Required':2.0,'Actual':2.0,'Status':'pass'},
            {'parameter':'port_parameters','Required':ports.PORT_PARAMETERS,'Actual':state['parameters']|{},'Status':'pass'},
            {'parameter':'frozen_catalog','Required':ports.EXPECTED[ports.CATALOG],'Actual':clean.sha(ports.CATALOG),'Status':'pass'}]
    checks[2]['Actual']={k:state['parameters'][k] for k in ports.PORT_PARAMETERS}
    ports.write_json(root/'required_actual.json',{'status':'pass','checks':checks})
    ports.write_json(root/'plan.json',state)
    print(str(root),flush=True)
    reused={v['shard'] for v in state['results']}
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(worker,{'root':str(root),'shard':s,'old':baseline[s],'january':january[s],
                                 'baseline_result':previous.get(s),'baseline_end':baseline_end,
                                 'observation_end':observation_end,'baseline_panel':state['baseline_panel']}):s for s in shards if s not in reused}
        for future in as_completed(jobs):
            try:
                result=future.result();state['results'].append(result)
                print(json.dumps({k:result[k] for k in ['shard','added_2025_departure_raw_legs','added_cross_year_arrivals','added_within_2025_arrivals','wall_seconds']}),flush=True)
            except Exception:
                state['failures'].append({'shard':jobs[future],'error':traceback.format_exc()})
                print(state['failures'][-1]['error'],flush=True)
            ports.write_json(root/'execution.json',state)
    for item in state['code']:ports.verify(item)
    state['status']='pass' if not state['failures'] else 'failed'
    state['complete_year']=len(state['results'])==64 and not state['failures']
    ports.write_json(root/'execution.json',state)
    if state['failures']:raise RuntimeError('buffered shard replay failed; inspect execution.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--buffer',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--shards',nargs='+',default=['shard_0005','shard_0032']);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--baseline-visits',type=Path);p.add_argument('--baseline-panel',type=Path)
    p.add_argument('--observation-end',default='2026-02-01');p.add_argument('--verified-probe',type=Path)
    args=p.parse_args();root=args.out.resolve()
    if not root.is_relative_to((CODING/'.runs').resolve()) or args.workers<1:raise ValueError('invalid output/workers')
    run(args.buffer.resolve(),root,args.shards,args.workers,args.baseline_visits,args.baseline_panel,args.observation_end,args.verified_probe)
