"""Export missing weather instants from accepted leg schedules without rescheduling.

The product stores hourly points and leg distances, but not polylines. Replay
only legs needed for missing sampling instants, reject any changed distance or
existing position, and retain the original schedule verbatim.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def required_leg_requests(requests, legs):
    rows=[]
    valid=legs.loc[legs.leg_status.eq('resolved') & legs.schedule_status.eq('resolved')]
    grouped={k:v.sort_values('scheduled_start_epoch_s') for k,v in valid.groupby('candidate_id',sort=False)}
    for cid, group in requests.groupby('candidate_id',sort=False):
        part=grouped.get(cid)
        if part is None:continue
        t=pd.DatetimeIndex(group.position_time_utc).asi8/1e9
        starts=part.scheduled_start_epoch_s.to_numpy(float)
        ends=part.scheduled_end_epoch_s.to_numpy(float)
        if (starts[1:]<ends[:-1]-1e-7).any():raise ValueError('overlapping accepted resolved legs')
        ix=np.searchsorted(starts,t,side='right')-1
        take=(ix>=0)&(t<=ends[np.maximum(ix,0)])
        if not take.any():continue
        chosen=part.iloc[ix[take]]
        out=group.iloc[np.flatnonzero(take)].copy()
        out['gap_id']=chosen.gap_id.to_numpy()
        out['leg_index']=chosen.leg_index.to_numpy()
        rows.append(out)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def replay_requested_positions(requests, legs, anchors, inputs, original, router, interpolate):
    selected=required_leg_requests(requests,legs)
    audit={'requested_positions':len(requests),'eligible_positions':len(selected),
           'replayed_legs':0,'existing_positions_checked':0,'filled_positions':0}
    if selected.empty:return pd.DataFrame(),audit
    leg_index=legs.set_index(['gap_id','leg_index'],verify_integrity=True)
    # Endpoints in every acquisition row must agree before reducing to gap grain.
    fields=['candidate_id','mmsi','lat_a','lon_a','lat_b','lon_b']
    if inputs.groupby('gap_id')[fields].nunique(dropna=False).gt(1).any().any():
        raise ValueError('inconsistent acquisition gap endpoints')
    endpoints=inputs.drop_duplicates('gap_id').set_index('gap_id')
    scheduled={k:v.sort_values('anchor_index') for k,v in anchors.groupby('gap_id',sort=False)}
    old={k:v for k,v in original.groupby(['gap_id','leg_index'],sort=False)}
    results=[]
    for (gid,idx), targets in selected.groupby(['gap_id','leg_index'],sort=False):
        leg=leg_index.loc[(gid,idx)]
        ep=endpoints.loc[gid]
        if not targets.candidate_id.eq(ep.candidate_id).all() or not targets.mmsi.astype(str).eq(str(ep.mmsi)).all():
            raise ValueError('supplement identity mismatch')
        a=scheduled.get(gid,anchors.iloc[:0])
        if len(a) and not np.array_equal(a.anchor_index.to_numpy(),np.arange(len(a))):
            raise ValueError('anchor index sequence changed')
        nodes=[(float(ep.lat_a),float(ep.lon_a))]+list(a[['lat','lon']].itertuples(index=False,name=None))+[(float(ep.lat_b),float(ep.lon_b))]
        left,right=nodes[int(idx)],nodes[int(idx)+1]
        t0,t1=float(leg.scheduled_start_epoch_s),float(leg.scheduled_end_epoch_s)
        if not t1>t0 or leg.max_speed_knots!=25:raise ValueError('invalid frozen schedule')
        route=router.resolve_path(*left,*right,snap=False,timeout_s=5.,duration_hours=None,max_speed_knots=25.)
        if route.status!='resolved' or route.path is None:
            raise ValueError(f'accepted leg could not replay: {gid}/{idx}: {route.status}')
        if abs(route.path.total_km-float(leg.path_distance_km))>1e-8:
            raise ValueError(f'accepted path distance changed: {gid}/{idx}')
        if route.path.total_km/((t1-t0)/3600)/1.852>25+1e-9:
            raise ValueError('replayed path violates frozen speed cap')
        previous=old.get((gid,idx))
        if previous is not None:
            pt=pd.DatetimeIndex(previous.utc).asi8/1e9
            check=interpolate(route.path,t0,t1,pt)
            # Tight numerical tolerance across supported NumPy versions, never km-scale.
            if not np.allclose(check,previous[['lat','lon']].to_numpy(float),rtol=0,atol=1e-10):
                raise ValueError(f'accepted positions changed: {gid}/{idx}')
            audit['existing_positions_checked']+=len(previous)
        tt=pd.DatetimeIndex(targets.position_time_utc).asi8/1e9
        xy=interpolate(route.path,t0,t1,tt)
        result=targets[['candidate_id','mmsi','position_time_utc','gap_id','leg_index']].copy()
        result['lat'],result['lon']=xy[:,0],xy[:,1]
        result['track_evidence']='companion_resampled_accepted_schedule'
        result['companion_anchor_source']='accepted_companion_schedule'
        results.append(result);audit['replayed_legs']+=1
    output=pd.concat(results,ignore_index=True)
    if output.duplicated(['candidate_id','position_time_utc']).any():raise ValueError('duplicate supplement position')
    audit['filled_positions']=len(output)
    return output,audit


def apply_supplement(frame, supplement):
    """Fill exact missing keys, refusing to replace any existing coordinates."""
    keys=['candidate_id','position_time_utc']
    if supplement.duplicated(keys).any():raise ValueError('duplicate supplement keys')
    result=frame.copy()
    lookup=result.reset_index(names='_row').merge(supplement,on=keys,how='inner',suffixes=('_base','_supplement'),validate='one_to_one')
    if len(lookup)!=len(supplement):raise ValueError('supplement contains absent weather key')
    if not lookup.track_evidence_base.eq('unresolved').all():raise ValueError('supplement would overwrite existing evidence')
    if not lookup.mmsi_base.astype(str).eq(lookup.mmsi_supplement.astype(str)).all():raise ValueError('supplement vessel mismatch')
    for c in ['lat','lon','gap_id','track_evidence','companion_anchor_source']:
        result.loc[lookup._row,c]=lookup[c+'_supplement'].to_numpy()
    return result


def complete_product_positions(requests, product, bucket, mask_path):
    """Normal pipeline path: complete prepared requests before any weather I/O."""
    import hashlib,json,sys,time
    from pathlib import Path
    stage=Path(__file__).resolve().parents[2]/'25_reconstruct_companion_tracks'
    sys.path.insert(0,str(stage))
    from sea_path.ocean_mask import load_ocean_mask,latlon_to_index
    from sea_path.routing import SeaPathRouter,interpolate_along_path
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for x in iter(lambda:f.read(8<<20),b''):h.update(x)
        return h.hexdigest()
    product,mask_path=Path(product),Path(mask_path)
    if digest(stage/'sea_path/routing.py')!='3779378153454c937da19e0718b334505f4449e8b846498a363e184972546896':raise ValueError('router changed')
    if digest(mask_path)!='71f6045e14acc73397efbcb5c7be6e1257c1e949cceb89dc38d001cebf0ac4e8':raise ValueError('mask changed')
    manifest=json.loads((product/'adoption_manifest.json').read_text())
    hashes={}
    def load(relative):
        path=product/'run'/relative;h=digest(path)
        if h!=manifest['files'][relative]['sha256']:raise ValueError('adopted replay input changed')
        hashes[relative]=h;return pd.read_parquet(path)
    t=time.perf_counter()
    mask=load_ocean_mask(mask_path);router=SeaPathRouter(mask)
    root=f'buckets/bucket_{bucket:04d}/'
    supplement,audit=replay_requested_positions(requests,load(root+'route_legs.parquet'),load(root+'scheduled_anchor_times.parquet'),load(f'inputs/anchors_{bucket:04d}.parquet'),load(root+'companion_track.parquet'),router,interpolate_along_path)
    if len(supplement) and not all(mask[latlon_to_index(a,b)] for a,b in supplement[['lat','lon']].itertuples(index=False,name=None)):raise ValueError('nonnavigable supplement')
    audit.update(state='completed',bucket=bucket,source_hashes=hashes,elapsed_seconds=time.perf_counter()-t,rescheduling=False)
    return supplement,audit
