from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from weather_core.complete_positions import required_leg_requests,replay_requested_positions


def fixture():
 t=pd.Timestamp('2025-01-01',tz='UTC').timestamp()
 requests=pd.DataFrame({'candidate_id':['v','v','v'],'mmsi':['1']*3,'position_time_utc':pd.to_datetime([t+100,t+200,t+400],unit='s',utc=True)})
 legs=pd.DataFrame({'candidate_id':['v'],'gap_id':['g'],'leg_index':[0],'leg_status':['resolved'],'schedule_status':['resolved'],'scheduled_start_epoch_s':[t],'scheduled_end_epoch_s':[t+300],'path_distance_km':[1.],'max_speed_knots':[25.]})
 inputs=pd.DataFrame({'candidate_id':['v'],'mmsi':['1'],'gap_id':['g'],'lat_a':[0.],'lon_a':[0.],'lat_b':[0.],'lon_b':[1.]})
 anchors=pd.DataFrame(columns=['gap_id','anchor_index','lat','lon'])
 old=pd.DataFrame({'gap_id':['g'],'leg_index':[0],'utc':pd.to_datetime([t+150],unit='s',utc=True),'lat':[0.],'lon':[.5]})
 return requests,legs,anchors,inputs,old


def interp(path,a,b,tt):return np.column_stack([np.zeros(len(tt)),(tt-a)/(b-a)])


class Router:
 def resolve_path(self,*args,**kwargs):
  assert kwargs==dict(snap=False,timeout_s=5.,duration_hours=None,max_speed_knots=25.)
  return SimpleNamespace(status='resolved',path=SimpleNamespace(total_km=1.))


def test_exact_schedule_and_existing_point_parity():
 f=fixture();out,a=replay_requested_positions(*f,Router(),interp)
 assert len(out)==2 and a['existing_positions_checked']==1
 np.testing.assert_allclose(out.lon,[1/3,2/3])
 assert f[1].scheduled_end_epoch_s.iloc[0]-f[1].scheduled_start_epoch_s.iloc[0]==300


def test_no_position_from_unresolved_or_outside_interval():
 r,l,*_=fixture();l['leg_status']='speed_infeasible'
 assert required_leg_requests(r,l).empty


def test_changed_old_position_fails_closed():
 f=fixture();f[-1]['lon']=.6
 with pytest.raises(ValueError,match='positions changed'):replay_requested_positions(*f,Router(),interp)


def test_changed_path_distance_fails_closed():
 f=fixture();f[1]['path_distance_km']=2
 with pytest.raises(ValueError,match='distance changed'):replay_requested_positions(*f,Router(),interp)


def test_identity_mismatch_fails_closed():
 f=fixture();f[3]['mmsi']='2'
 with pytest.raises(ValueError,match='identity mismatch'):replay_requested_positions(*f,Router(),interp)


def test_supplement_only_fills_exact_missing_key():
 from weather_core.complete_positions import apply_supplement
 t=pd.Timestamp('2025-01-01',tz='UTC')
 base=pd.DataFrame({'candidate_id':['v'],'position_time_utc':[t],'mmsi':['1'],'lat':[np.nan],'lon':[np.nan],'gap_id':[None],'track_evidence':['unresolved'],'companion_anchor_source':[None]})
 s=pd.DataFrame({'candidate_id':['v'],'position_time_utc':[t],'mmsi':['1'],'lat':[1.],'lon':[2.],'gap_id':['g'],'track_evidence':['companion_resampled_accepted_schedule'],'companion_anchor_source':['accepted_companion_schedule']})
 r=apply_supplement(base,s);assert r.lat.iloc[0]==1
 with pytest.raises(ValueError,match='overwrite'):apply_supplement(r,s)
 s['mmsi']='2'
 with pytest.raises(ValueError,match='vessel mismatch'):apply_supplement(base,s)
