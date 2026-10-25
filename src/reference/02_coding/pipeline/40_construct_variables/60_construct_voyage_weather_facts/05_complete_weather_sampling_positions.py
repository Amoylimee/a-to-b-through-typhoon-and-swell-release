#!/usr/bin/env python3
"""Complete missing sampling instants for one previously matched bucket."""
from __future__ import annotations
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'25_reconstruct_companion_tracks'))
from sea_path.ocean_mask import load_ocean_mask,latlon_to_index
from sea_path.routing import SeaPathRouter,interpolate_along_path
from weather_core.complete_positions import replay_requested_positions


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()


def run(probe,output,mask):
    if output.exists():raise FileExistsError(output)
    start=time.perf_counter();s=json.loads((probe/'status.json').read_text())
    if s['state']!='completed':raise ValueError('completed source probe required')
    c=s['config'];product=Path(c['source_project'])/c['companion_product']
    adoption=json.loads((product/'adoption_manifest.json').read_text())
    output.mkdir(parents=True)
    audit={'state':'running','probe':str(probe.resolve()),'bucket':s['bucket'],'sources':{},'products_modified':0,'network_requests':0,'rescheduling':False}
    try:
        def load(relative):
            p=product/'run'/relative;sha=digest(p)
            if sha!=adoption['files'][relative]['sha256']:raise ValueError('adopted input changed')
            audit['sources'][str(p)]={'sha256':sha};return pd.read_parquet(p)
        root=f"buckets/bucket_{s['bucket']:04d}/"
        frames=[]
        for p in sorted(probe.glob('weather_*.parquet')):
            if digest(p)!=s['output_files'][p.name]['sha256']:raise ValueError('weather probe changed')
            f=pd.read_parquet(p,columns=['candidate_id','mmsi','position_time_utc','track_evidence'])
            frames.append(f.loc[f.track_evidence.eq('unresolved')])
        requests=pd.concat(frames,ignore_index=True)
        if digest(mask)!='71f6045e14acc73397efbcb5c7be6e1257c1e949cceb89dc38d001cebf0ac4e8':raise ValueError('mask changed')
        routing=HERE.parent/'25_reconstruct_companion_tracks/sea_path/routing.py'
        if digest(routing)!='3779378153454c937da19e0718b334505f4449e8b846498a363e184972546896':raise ValueError('router changed')
        ocean=load_ocean_mask(mask);router=SeaPathRouter(ocean)
        supplement,counts=replay_requested_positions(requests,load(root+'route_legs.parquet'),load(root+'scheduled_anchor_times.parquet'),load(f"inputs/anchors_{s['bucket']:04d}.parquet"),load(root+'companion_track.parquet'),router,interpolate_along_path)
        if len(supplement) and not all(ocean[latlon_to_index(a,b)] for a,b in supplement[['lat','lon']].itertuples(index=False,name=None)):raise ValueError('nonnavigable supplement')
        supplement.to_parquet(output/'supplement.parquet',index=False)
        audit.update(counts,state='completed',elapsed_seconds=time.perf_counter()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,output_sha256=digest(output/'supplement.parquet'),router_sha256=digest(routing),mask_sha256=digest(mask))
    except BaseException as e:
        audit.update(state='failed',error=repr(e));raise
    finally:(output/'status.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mask',type=Path,required=True)
    a=p.parse_args();run(a.probe,a.output,a.mask)
