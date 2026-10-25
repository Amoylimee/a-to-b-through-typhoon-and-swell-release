"""Independent arithmetic checks of stored reconstruction outputs."""
import numpy as np
import pandas as pd


def distance_km(lat_a,lon_a,lat_b,lon_b):
    a,b=np.deg2rad(lat_a),np.deg2rad(lat_b)
    dlat=b-a
    dlon=np.deg2rad((np.asarray(lon_b)-np.asarray(lon_a)+180)%360-180)
    h=np.sin(dlat/2)**2+np.cos(a)*np.cos(b)*np.sin(dlon/2)**2
    return 6371.0*2*np.arcsin(np.sqrt(np.clip(h,0,1)))


def recompute_covered_hours(legs,gaps):
    success=legs.loc[legs.leg_status.eq('resolved')&legs.schedule_status.eq('resolved')].copy()
    window=gaps.set_index('gap_id').loc[success.gap_id]
    a=window.gap_start_utc.astype('int64').to_numpy()/1e9
    b=window.gap_end_utc.astype('int64').to_numpy()/1e9
    success['audit_covered_hours']=np.maximum(0,np.minimum(success.scheduled_end_epoch_s.to_numpy(),b)-
        np.maximum(success.scheduled_start_epoch_s.to_numpy(),a))/3600
    return success.groupby('gap_id').audit_covered_hours.sum()


def speed_violations(legs):
    selected=legs.loc[legs.leg_status.eq('resolved')&legs.schedule_status.eq('resolved')]
    elapsed=(selected.scheduled_end_epoch_s-selected.scheduled_start_epoch_s)/3600
    speed=selected.path_distance_km/elapsed/1.852
    return int((~np.isfinite(speed)|(speed>25+1e-8)|(elapsed<=0)).sum()), speed


def nearest_anchor_errors(track,anchors):
    scheduled=anchors.loc[anchors.scheduled_epoch_s.notna(),['gap_id','scheduled_epoch_s']].copy()
    left=track[['gap_id','utc','hours_to_nearest_anchor']].copy()
    left['epoch']=left.utc.astype('int64')/1e9
    left['_row']=np.arange(len(left))
    right=scheduled.rename(columns={'scheduled_epoch_s':'anchor_epoch'})
    merged=pd.merge_asof(left.sort_values('epoch'),right.sort_values('anchor_epoch'),
        left_on='epoch',right_on='anchor_epoch',by='gap_id',direction='nearest')
    expected=(merged.epoch-merged.anchor_epoch).abs()/3600
    actual=merged.hours_to_nearest_anchor.astype(float)
    mismatch=(expected.isna()!=actual.isna())|((expected-actual).abs()>1e-9)
    return int(mismatch.sum())
