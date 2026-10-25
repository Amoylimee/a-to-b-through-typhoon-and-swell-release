#!/usr/bin/env python3
"""Enumerate 2025-departure adjacent-city candidates with January follow-up."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CODING = HERE.parents[1]
PIPE = CODING/'pipeline/gfw_release'
spec=importlib.util.spec_from_file_location('e009_candidate_base',PIPE/'130_build_gfw_voyage_base.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
OLD_BASE=CODING/'.runs/2026-10-11_1907_E007-one-cell-voyage-base'
NATURAL=['vessel_uid_clean','origin_city','dest_city','departure_time','arrival_time']


def cohort(frame, observation_end=None):
    frame=frame.copy()
    for col in ['departure_time','arrival_time']:
        frame[col]=pd.to_datetime(frame[col],utc=True)
    frame['vessel_uid_clean']=frame.vessel_uid_clean.astype(str)
    keep=frame.departure_time.ge('2025-01-01')&frame.departure_time.lt('2026-01-01')
    if observation_end is not None:keep &= frame.arrival_time.lt(observation_end)
    return frame.loc[keep].copy()


def validate_position_evidence(result, state):
    if result['baseline_points_exact']:
        return
    if result.get('retained_input_points_exact') and result.get('boundary_retained_input_audit'):
        audit=json.loads(base.verify(result['boundary_retained_input_audit']).read_text())
        if (audit['status']!='pass' or audit['scope']!='retained_voyage_inputs' or audit['shard']!=result['shard']
                or audit['changes_inside_retained_support']!=0 or not audit['old_cleaning_reproduced']
                or not audit['extended_cleaning_reproduced'] or audit['baseline_panel']!=state['baseline_panel']):
            raise ValueError('boundary differences lack retained-voyage reproduction proof')
        for name in ['changes','baseline_panel','old_reproduction_output','extended_reproduction_output']:base.verify(audit[name])
        return
    if state.get('validation_scope') != 'candidate_port_inputs_only':
        raise ValueError('unexplained changes in historical positions')
    receipt = result.get('boundary_cleaning_audit')
    if not receipt or receipt != state.get('boundary_cleaning_audit'):
        raise ValueError('missing bound boundary-cleaning audit')
    audit = json.loads(base.verify(receipt).read_text())
    if (audit.get('status') != 'pass' or audit.get('scope') != 'candidate_port_inputs_only'
            or audit.get('shard') != result['shard'] or audit.get('source_mask') != result['source']
            or audit.get('removed_2025_points') != 2
            or audit.get('removed_utc') != ['2025-12-31 22:30:00', '2025-12-31 23:30:00']
            or not audit.get('annual_cut_fallback_reproduced')
            or not audit.get('all_other_historical_positions_exact')
            or audit.get('historical_trajectory_accepted') is not False
            or audit.get('old_raw_legs_exact') != result['baseline_raw_legs_exact']
            or audit.get('old_closed_visits_exact') != result['old_visit_comparison']['old_closed_visits_exact']):
        raise ValueError('boundary audit does not certify these candidate port inputs')
    for item in [audit['source_mask'], audit['baseline_mask'], *audit['sources'], *audit['code']]:
        base.verify(item)


def run(source,root,workers,probe=False,baseline_candidates=None,observation_end='2026-02-01',verified_probe=None):
    state=json.loads((source/'execution.json').read_text())
    expected=2 if probe else 64
    if state['status']!='pass' or (not probe and not state['complete_year']) or state['failures'] or len(state['results'])!=expected:
        raise ValueError('all 64 buffered port shards must pass first')
    results=sorted(state['results'],key=lambda v:v['shard'])
    if not probe and {v['shard'] for v in results}!={f'shard_{i:04d}' for i in range(64)}:
        raise ValueError('port-shard inventory differs')
    for result in results:
        validate_position_evidence(result, state)
        for output in result['outputs']:base.verify(output)
        base.verify(result['vessel_input_disposition'])
    ids=pd.concat([pd.read_csv(v['vessel_input_disposition']['path'],dtype={'vessel_uid':str}) for v in results])
    if ids.vessel_uid.duplicated().any():raise ValueError('vessel appears in multiple shards')
    # This is the observed subset of the same frozen roster, not a new fleet definition.
    base.GFW_VESSELS=len(ids)
    root.mkdir(parents=True,exist_ok=False)
    record={'status':'running','authorization':'2026-10-22 explicit February Worker instruction' if observation_end=='2026-03-01' else '2026-10-21 recorded research instruction',
        'scope':'Candidate enumeration only; quality filters, tracks and weather remain subsequent stages.',
        'analysis_departure_window_utc':['2025-01-01','2026-01-01'],
        'observation_window_utc':['2025-01-01',observation_end],'probe':probe,
        'baseline_candidates':base.record(baseline_candidates) if baseline_candidates else None,
        'observed_frozen_roster_vessels':len(ids),'source':base.record(source/'execution.json'),
        'code':[base.record(Path(__file__)),*base.code_records()],
        'port_input_validation_scope':state.get('validation_scope','exact_historical_replay'),
        'boundary_cleaning_audit':state.get('boundary_cleaning_audit'),
        'historical_trajectory_accepted':False,
        'scientific_adoption':False,'network_requests':0}
    save=lambda:base.write_json(root/'execution.json',record)
    save()
    try:
        if observation_end=='2026-03-01' and not probe:
            if verified_probe is None:raise ValueError('full candidate build requires two-shard probe')
            verified=json.loads((verified_probe/'execution.json').read_text())
            if verified['status']!='pass' or not verified['probe'] or verified['code']!=record['code']:
                raise ValueError('candidate probe not passed with current code')
        base.write_json(root/'required_actual.json',{'status':'pass','checks':[
            {'parameter':'departure_window_UTC','Required':['2025-01-01','2026-01-01'],'Actual':['2025-01-01','2026-01-01'],'Status':'pass'},
            {'parameter':'arrival_end_exclusive','Required':observation_end,'Actual':state['observation_window_utc'][1],'Status':'pass' if state['observation_window_utc'][1]==observation_end else 'failed'},
            {'parameter':'analysis_unit','Required':'adjacent_commercial_episodes','Actual':base.MODE,'Status':'pass' if base.MODE=='adjacent_commercial_episodes' else 'failed'}]})
        if state['observation_window_utc'][1]!=observation_end or base.MODE!='adjacent_commercial_episodes':raise ValueError('candidate protected parameter mismatch')
        prep=base.build_view(root,state,results,PIPE/'config/t034_anchorage_city_corrections.csv')
        record['input_view']=prep;save()
        builder,_=base.configure_builder(Path(prep['view']),Path(prep['empty_city_reference']['path']),Path(prep['mapping']['path']))
        output=Path(prep['view'])/'results';output.mkdir()
        record['phase']='city_episodes';save()
        episode=builder.build_year_episodes(2025,output,500_000,base.MODE)
        episodes,legs,call_meta=builder.load_candidate_inputs(episode)
        record['phase']='candidate_occurrences';save()
        part_count=2 if probe else 64
        parts=builder.run_candidate_tasks(2025,output,episodes,legs,call_meta,workers,part_count,None,
                                          builder.episode_manifest_identity(episode),base.MODE)
        checked=builder.validate_candidate_outputs(2025,output,episode,parts,part_count,base.MODE)
        if checked['status']!='pass':raise ValueError('candidate enumeration validation failed')
        relative=Path('intermediate/2025/candidate_occurrences/candidate_passages')
        all_new=pd.concat([pd.read_feather(p) for p in sorted((output/relative).glob('part-*.feather'))],ignore_index=True)
        new=cohort(all_new,observation_end)
        old=cohort(pd.read_parquet(baseline_candidates)) if baseline_candidates else cohort(pd.concat([pd.read_feather(p) for p in sorted((OLD_BASE/'input_view/results'/relative).glob('part-*.feather'))],ignore_index=True))
        if probe:old=old.loc[old.vessel_uid_clean.isin(set(ids.vessel_uid))].copy()
        if new.duplicated(NATURAL).any() or old.duplicated(NATURAL).any():
            raise ValueError('physical candidate key is not unique')
        comparison=old[['candidate_id',*NATURAL]].merge(new[['candidate_id',*NATURAL]],on=NATURAL,how='outer',
            suffixes=('_old','_new'),indicator=True,validate='one_to_one')
        comparison.to_parquet(root/'candidate_comparison.parquet',index=False)
        added=new.loc[new.candidate_id.isin(set(comparison.loc[comparison._merge.eq('right_only'),'candidate_id_new']))].copy()
        removed=old.loc[old.candidate_id.isin(set(comparison.loc[comparison._merge.eq('left_only'),'candidate_id_old']))].copy()
        new.to_parquet(root/'departure_2025_candidates.parquet',index=False)
        added.to_parquet(root/'added_2025_candidates.parquet',index=False)
        removed.to_parquet(root/'old_candidates_not_preserved.parquet',index=False)
        if len(removed):raise ValueError('baseline candidate endpoints changed or disappeared')
        if verified_probe is not None:
            prior=pd.read_parquet(verified_probe/'departure_2025_candidates.parquet')
            current=new.loc[new.vessel_uid_clean.isin(set(prior.vessel_uid_clean))]
            common=[c for c in prior if c in current and c not in ['candidate_id']]
            order=lambda f:f.sort_values(NATURAL).reset_index(drop=True)
            pd.testing.assert_frame_equal(order(prior[common]),order(current[common]),check_dtype=False,check_exact=True)
            record['probe_candidates_exact_except_ids']=True
        # Explicitly retain unresolved end-of-window evidence; these are raw departures, not confirmed voyages.
        unresolved=[]
        for result in results:
            shard=result['shard']
            visits=pd.read_parquet(source/'ports/shards'/shard/'visit_quality.parquet')
            high=visits.loc[visits.visit_confidence.eq('high')].sort_values(['vessel_uid','exit_time'])
            next_entry=high.groupby('vessel_uid',sort=False).entry_time.shift(-1)
            pending=high.loc[high.exit_time.ge('2025-01-01')&high.exit_time.lt('2026-01-01')&next_entry.isna()].copy()
            pending['shard']=shard;unresolved.append(pending)
        pending=pd.concat(unresolved,ignore_index=True)
        pending.to_parquet(root/'2025_raw_departures_without_next_high_visit.parquet',index=False)
        monthly=[]
        for month in range(1,13):
            before=old.loc[old.departure_time.dt.month.eq(month)]
            after=new.loc[new.departure_time.dt.month.eq(month)]
            extra=added.loc[added.departure_time.dt.month.eq(month)]
            missing=removed.loc[removed.departure_time.dt.month.eq(month)]
            monthly.append({'month':month,'old_candidates':len(before),'buffered_candidates':len(after),
                'added':len(extra),'removed_or_changed':len(missing),
                'old_candidate_hours':float(before.total_hours.sum()),'buffered_candidate_hours':float(after.total_hours.sum())})
        pd.DataFrame(monthly).to_csv(root/'candidate_counts_by_departure_month.csv',index=False)
        summary={'status':'pass','before_candidates':len(old),'buffered_2025_departure_candidates':len(new),
            'added_candidates':len(added),'old_candidates_removed_or_changed':len(removed),
            'added_arriving_2026':int(added.arrival_time.ge('2026-01-01').sum()),
            'added_arriving_2025':int(added.arrival_time.lt('2026-01-01').sum()),
            'non_2025_departure_candidates_excluded':len(all_new)-len(new),
            'pending_2025_raw_departures_without_next_high_visit':len(pending),
            'still_open_2025_visits':sum(v['still_open_2025_visits'] for v in results),
            'adjacent_commercial_episodes':bool(new.dest_episode_sequence.sub(new.origin_episode_sequence).eq(1).all()),
            'no_intermediate_commercial_city':bool(new.n_intermediate_city_episodes.eq(0).all()),
            'row_balance':len(old)+len(added)-len(removed)==len(new),
            'quality_filters_applied':False,'weather_matched':False,'scientific_adoption':False,
            'interpretation':'Counts are pre-quality adjacent-city voyage candidates. Pending raw departures are not a measured count of missing analysis voyages. Finite follow-up does not establish departure-cohort completeness.'}
        if not all(summary[k] for k in ['adjacent_commercial_episodes','no_intermediate_commercial_city','row_balance']):
            raise ValueError('candidate comparison checks failed')
        base.write_json(root/'summary.json',summary)
        for item in record['code']:base.verify(item)
        record.update(status='pass',phase='candidate_comparison_complete',summary=summary);save()
        print(json.dumps(summary),flush=True)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--visits',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--probe',action='store_true');p.add_argument('--baseline-candidates',type=Path)
    p.add_argument('--observation-end',default='2026-02-01');p.add_argument('--verified-probe',type=Path)
    args=p.parse_args();root=args.out.resolve()
    if not root.is_relative_to((CODING/'.runs').resolve()):raise ValueError('output outside .runs')
    run(args.visits.resolve(),root,args.workers,args.probe,args.baseline_candidates,args.observation_end,args.verified_probe)
