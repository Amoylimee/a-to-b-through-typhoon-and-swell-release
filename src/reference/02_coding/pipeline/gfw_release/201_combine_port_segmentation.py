#!/usr/bin/env python3
"""Combine disjoint validated port shards without rerunning the probe shards."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import pandas as pd
HERE=Path(__file__).resolve().parent
E006=HERE.parents[1]/'pipeline/gfw_release'
sys.path.insert(0,str(E006))
import step6_core as core


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--components',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    states=[json.loads((r/'execution.json').read_text()) for r in args.components]
    if any(s['status']!='pass' or s['failures'] for s in states):raise ValueError('all components must pass')
    first=states[0]
    for s in states:
        for name in ['port_parameters','catalog','rule','clean_root']:
            if s[name]!=first[name]:raise ValueError(f'component {name} differs')
        if s['specs']['sha256']!=first['specs']['sha256']:raise ValueError('component specs differ')
        for item in [*s['code'],s['rule'],s['catalog'],s['specs']]:
            if core.record(Path(item['path']))!=item:raise ValueError('component source changed')
    results=[r for s in states for r in s['results']]
    if sorted(r['shard'] for r in results)!=core.SHARDS:raise ValueError('components do not cover every shard exactly once')
    for r in results:
        for item in [*r['outputs'],r['visit_quality'],r['vessel_input_disposition']]:
            if core.record(Path(item['path']))!=item:raise ValueError('component output changed')
    out=core.inside_runs(args.out);out.mkdir(parents=True,exist_ok=False);(out/'shards').mkdir()
    for root,state in zip(args.components,states):
        for r in state['results']:(out/'shards'/r['shard']).symlink_to((root/'shards'/r['shard']).resolve(),target_is_directory=True)
    specs=out/'analysis_vessel_specs_2025.csv';shutil.copy2(first['specs']['path'],specs)
    vessels=pd.concat([pd.read_csv(r['vessel_input_disposition']['path'],dtype={'vessel_uid':str}) for r in results],ignore_index=True)
    totals={k:sum(r[k] for r in results) for k in ['n_input_rows','n_unique_vessels','n_visits','n_high_confidence_visits','n_voyages']}
    closure={'all_64_shards':len(results)==64,'cleaned_rows_close_to_step2':totals['n_input_rows']==255216589,
             'vessels_close_to_step2':len(vessels)==34884==totals['n_unique_vessels'],
             'each_vessel_in_one_shard':not vessels.vessel_uid.duplicated().any()}
    if not all(closure.values()):raise ValueError('annual row/vessel closure failed')
    counts={k:sum(r['revision_counts'][k] for r in results) for k in results[0]['revision_counts']}
    state={**first,'status':'pass','complete_year':True,'results':sorted(results,key=lambda r:r['shard']),
           'failures':[],'specs':core.record(specs),'totals':totals,'closure':closure,'revision_counts':counts,
           'components':[core.record(r/'execution.json') for r in args.components],'combination_code':core.record(Path(__file__)),
           'adopted':False,'network_requests':0}
    core.write_json(out/'execution.json',state)
    print(json.dumps({'status':'pass','closure':closure,'totals':totals,'revision_counts':counts},indent=2))


if __name__=='__main__':main()
