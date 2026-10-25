"""Classify unresolved raw departures and revisit the fixed January open lists."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd


def record(path):
    path=Path(path)
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def describe(group,visits,disposition,month):
    group=group.copy();group['vessel_uid']=group.vessel_uid.astype(str)
    visits=visits.copy();visits['vessel_uid']=visits.vessel_uid.astype(str)
    for c in ['entry_time','exit_time']:visits[c]=pd.to_datetime(visits[c],utc=True)
    group['exit_time']=pd.to_datetime(group.exit_time,utc=True)
    later=visits.merge(group[['vessel_uid','exit_time']],on='vessel_uid',suffixes=('','_baseline'),validate='many_to_one')
    later=later.loc[later.entry_time.ge(later.exit_time_baseline)].copy()
    later['high']=later.visit_confidence.eq('high');later['open']=later.exit_time.isna()
    stats=later.groupby('vessel_uid').agg(later_visits=('entry_time','size'),later_high_visits=('high','sum'),
        later_open_visits=('open','sum'),first_later_visit=('entry_time','min'),last_later_visit=('entry_time','max'))
    out=group[['vessel_uid','shard','exit_time']].merge(stats,on='vessel_uid',how='left',validate='one_to_one')
    for c in ['later_visits','later_high_visits','later_open_visits']:out[c]=out[c].fillna(0).astype(int)
    out=out.merge(disposition,on='vessel_uid',how='left',validate='one_to_one')
    if out.last_cleaned_utc.isna().any():raise ValueError('departure vessel lost from cleaned disposition')
    out['latest_buffer_position_observed']=pd.to_datetime(out.last_cleaned_utc,utc=True).ge(month+'-01')
    out['followup_category']='no_later_port_visit'
    out.loc[out.later_visits.gt(0),'followup_category']='later_closed_low_confidence_visits_only'
    out.loc[out.later_open_visits.gt(0),'followup_category']='later_open_low_confidence_visit'
    out.loc[out.later_high_visits.gt(0),'followup_category']='later_high_confidence_visit_observed'
    return out


def run(candidates,ports,baseline_candidates,baseline_ports,out,month):
    out.mkdir(parents=True,exist_ok=False)
    state=json.loads((ports/'execution.json').read_text())
    summary=json.loads((candidates/'summary.json').read_text())
    if state['status']!='pass' or not state['complete_year'] or summary['status']!='pass':raise ValueError('complete candidates and ports required')
    current=pd.read_parquet(candidates/'2025_raw_departures_without_next_high_visit.parquet')
    baseline=pd.read_parquet(baseline_candidates/'2025_raw_departures_without_next_high_visit.parquet')
    if len(baseline)!=4748:raise ValueError('January pending-departure baseline changed')
    outputs={'current_pending':[],'january_pending_followup':[]};opens=[];sources=[]
    for result in sorted(state['results'],key=lambda r:r['shard']):
        shard=result['shard'];folder=ports/'ports/shards'/shard
        visit_path=folder/'visit_quality.parquet';disp_path=folder/'vessel_input_disposition.csv'
        visits=pd.read_parquet(visit_path)
        disp=pd.read_csv(disp_path,dtype={'vessel_uid':str},usecols=['vessel_uid','last_cleaned_utc'])
        sources.extend([record(visit_path),record(disp_path)])
        for name,panel in [('current_pending',current),('january_pending_followup',baseline)]:
            subset=panel[panel.shard.eq(shard)]
            if len(subset):outputs[name].append(describe(subset,visits,disp,month))
        old=pd.read_parquet(baseline_ports/'ports/shards'/shard/'still_open_2025_visits.parquet')
        if len(old):
            if old.visit_id_internal.duplicated().any():raise ValueError('duplicate baseline open visit')
            joined=old[['visit_id_internal','vessel_uid','entry_time']].merge(
                visits[['visit_id_internal','vessel_uid','entry_time','exit_time','visit_confidence']],
                on=['visit_id_internal','vessel_uid','entry_time'],how='left',validate='one_to_one',indicator=True)
            joined['category']='still_open'
            joined.loc[joined.exit_time.notna(),'category']='now_closed_low_confidence'
            joined.loc[joined.exit_time.notna()&joined.visit_confidence.eq('high'),'category']='now_closed_high_confidence'
            joined.loc[joined._merge.ne('both'),'category']='baseline_visit_identity_not_preserved'
            joined['shard']=shard;opens.append(joined.drop(columns='_merge'))
    result={'status':'pass','latest_buffer_month':month,'sources':sources,'scientific_adoption':False}
    for name,parts in outputs.items():
        frame=pd.concat(parts,ignore_index=True)
        expected=len(current) if name=='current_pending' else len(baseline)
        if len(frame)!=expected or frame.vessel_uid.duplicated().any():raise ValueError('pending identity closure failed')
        if name=='current_pending' and frame.later_high_visits.gt(0).any():raise ValueError('current pending list contains resolved departures')
        frame.to_csv(out/(name+'_details.csv'),index=False)
        result[name]={'rows':len(frame),'categories':frame.followup_category.value_counts().to_dict(),
                      'latest_buffer_position_observed':int(frame.latest_buffer_position_observed.sum())}
    open_frame=pd.concat(opens,ignore_index=True)
    if len(open_frame)!=715:raise ValueError('January open-visit baseline changed')
    open_frame.to_csv(out/'january_open_visits_followup.csv',index=False)
    result['january_open_visits']={'rows':len(open_frame),'categories':open_frame.category.value_counts().to_dict()}
    result['current_still_open_2025_visits']=summary['still_open_2025_visits']
    result['interpretation']='Raw departures and open visits can overlap. Neither is a count of missing retained voyages; do not add them or report a missing-voyage rate.'
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['candidates','ports','baseline-candidates','baseline-ports','out']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--month',default='2026-02');a=p.parse_args()
    run(a.candidates,a.ports,a.baseline_candidates,a.baseline_ports,a.out,a.month)
