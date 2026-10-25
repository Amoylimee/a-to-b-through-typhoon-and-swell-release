#!/usr/bin/env python3
"""Recheck the frozen sixteen cases against corrected, full-year GFW visits.

Dense cleaned AIS is a source reference only and is never a construction input.
This targeted sample cannot estimate an annual commercial-port accuracy rate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
HERE=Path(__file__).resolve().parent
CODING=HERE.parents[1]
sys.path.insert(0,str(CODING/'pipeline/gfw_release'))
import step6_core as core
AUDIT=CODING/'.runs/2026-10-11_1230_E007-priority-voyage-source-audit'
REF=CODING/'.runs/2026-10-11_1239_E007-priority-complete-port-reference'
CATALOG=CODING/'data/frozen_inputs/anchorage_catalog/anchorages_analytical_city_v3_dock_adjusted_inclusive.csv'
CORRECTIONS=CODING/'pipeline/gfw_release/config/t034_anchorage_city_corrections.csv'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ports',type=Path,required=True)
    p.add_argument('--base',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=core.inside_runs(args.out);out.mkdir(parents=True,exist_ok=False)
    cards=pd.read_csv(AUDIT/'card_sources.csv',dtype={'candidate_id':str})
    catalog=pd.read_csv(CATALOG,dtype={'s2id':str})
    corrections=pd.read_csv(CORRECTIONS,dtype={'anchorage_id':str}).set_index('anchorage_id').new_analytical_city_id
    catalog['city']=catalog.s2id.map(corrections).fillna(catalog.analytical_city_id)
    trees={c:BallTree(np.deg2rad(q[['lat','lon']]),metric='haversine') for c,q in catalog.groupby('city')}
    decisions=pd.read_parquet(args.base/'candidate_decisions.parquet') if args.base else None
    new_cases=[];events=[];inputs=[];port_cache={}
    for row in cards.itertuples():
        meta=json.loads((AUDIT/row.card_id/'card.json').read_text())
        begin,end=pd.Timestamp(meta['departure_time']),pd.Timestamp(meta['arrival_time'])
        ship=str(meta['ship'])
        if row.source_shard not in port_cache:
            qpath=args.ports/'shards'/row.source_shard/'visit_quality.parquet'
            port_cache[row.source_shard]=pd.read_parquet(qpath)
            for c in ['entry_time','exit_time','first_seen_time','last_seen_time']:
                port_cache[row.source_shard][c]=pd.to_datetime(port_cache[row.source_shard][c],utc=True)
            inputs.append(core.record(qpath))
        all_visits=port_cache[row.source_shard]
        v=all_visits.loc[all_visits.vessel_uid.astype(str).eq(ship)]
        v=v.loc[v.entry_time.le(end+pd.Timedelta(days=3)) & (v.exit_time.ge(begin-pd.Timedelta(days=7)) | v.exit_time.isna())]
        path=REF/row.card_id/'reference_ais_cleaned_all_periods.parquet'
        dense=pd.read_parquet(path);inputs.append(core.record(path))
        speed=next((c for c in ['calc_speed_knots_masked','calc_speed_knots'] if c in dense),None)
        flags=[c for c in ['is_residual_speed_jump','has_second_pass_issue'] if c in dense]
        usable=dense.loc[~dense[flags].fillna(False).any(axis=1)].copy() if flags else dense
        case=out/row.card_id;case.mkdir()
        v.to_parquet(case/'corrected_visits.parquet',index=False)
        for call in v.itertuples():
            stop_time=call.exit_time if pd.notna(call.exit_time) else call.last_seen_time
            q=usable.loc[usable.UTC.between(call.entry_time-pd.Timedelta(hours=1),stop_time+pd.Timedelta(hours=1))]
            if len(q) and call.analytical_city_id in trees:
                distances,_=trees[call.analytical_city_id].query(np.deg2rad(q[['lat','lon']].to_numpy()),k=1)
                near=q.loc[distances[:,0]*6371.0088<=4]
            else:near=q.iloc[:0]
            stops=near.loc[pd.to_numeric(near[speed],errors='coerce').lt(.2)] if speed else near.iloc[:0]
            support='dense_calculated_stop' if len(stops) else ('no_comparable_reference' if len(near)==0 or speed is None else 'no_dense_calculated_stop_seen')
            events.append({'card_id':row.card_id,'city':call.analytical_city_id,'entry_time':call.entry_time,'exit_time':call.exit_time,
                           'confidence':call.visit_confidence,'entry_rule':call.entry_rule,'possible_short_stop':call.possible_short_stop,
                           'ambiguous_city':call.ambiguous_city,'entry_gap_hours':call.entry_gap_hours,'exit_gap_hours':call.exit_gap_hours,
                           'reference_points_within4km':len(near),'reference_stop_points':len(stops),'reference_support':support,
                           'reference_speed_source':speed,'independent_commercial_truth':False})
        summary={'card_id':row.card_id,'original_candidate_id':row.candidate_id,'original_in_option_c':bool(row.current_option_c_candidate),
                 'new_high_visits_in_context':int(v.visit_confidence.eq('high').sum()),'annulus_entries':int(v.entry_rule.eq('stop_supported_annulus').sum()),
                 'uncertain_short_visits':int(v.possible_short_stop.sum()),'unresolved_departure_brackets':int(v.exit_gap_hours.ge(4).sum()),
                 'dense_reference_rows':len(dense),'reference_flagged_rows':len(dense)-len(usable),'independent_commercial_truth':False}
        if decisions is not None:
            f=decisions.loc[decisions.vessel_uid_clean.astype(str).eq(ship)]
            overlapping=f.loc[f.arrival_time.gt(begin) & f.departure_time.lt(end)]
            identical=f.loc[f.origin_city.eq(row.origin_city)&f.dest_city.eq(row.dest_city)&f.departure_time.eq(begin)&f.arrival_time.eq(end)]
            overlapping.to_parquet(case/'corrected_candidate_decisions.parquet',index=False)
            summary.update(identical_physical_voyage_present=len(identical),identical_physical_voyage_retained=int(identical.retained.sum()),
                           identical_review_reasons='|'.join(identical.segmentation_review_reasons.unique()),
                           overlapping_retained_candidates=int(overlapping.retained.sum()),
                           overlapping_quarantined_candidates=int((~overlapping.segmentation_eligible).sum()))
        new_cases.append(summary)
    pd.DataFrame(events).to_csv(out/'priority_visit_reference.csv',index=False)
    pd.DataFrame(new_cases).to_csv(out/'priority_case_disposition.csv',index=False)
    checks={'sixteen_original_cases':len(new_cases)==16,'all_full_references_preserved':sum(x['dense_reference_rows'] for x in new_cases)==42999,
            'no_reference_used_for_construction':True,'independent_accuracy_not_claimed':True}
    if decisions is not None:
        by={x['card_id']:x for x in new_cases}
        checks.update(long_gap_case_R04_not_released_as_original=by['R04']['identical_physical_voyage_retained']==0,
                      short_stop_case_R35_not_released_as_original=by['R35']['identical_physical_voyage_retained']==0,
                      source_anomaly_case_R58_not_released_as_original=by['R58']['identical_physical_voyage_retained']==0,
                      intermediate_Setubal_R33_not_released_as_original=by['R33']['identical_physical_voyage_retained']==0,
                      intermediate_offshore_R70_not_released_as_original=by['R70']['identical_physical_voyage_retained']==0)
    core.write_json(out/'validation.json',{'status':'pass' if all(checks.values()) else 'failed','checks':checks,'cases':new_cases,
                    'inputs':inputs,'code':core.record(Path(__file__)),'targeted_sample':True,'annual_accuracy_estimated':False,'network_requests':0})
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
