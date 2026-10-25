from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from weather_core.track_adapter import voyage_segments,sample_ais,build_voyage_input,mask_uncertified_tc
from weather_core.hour_segments import centered_hour_segments,interpolate_positions


def ts(s): return pd.Timestamp(s,tz='UTC')


@pytest.mark.parametrize('start,end',[
 ('2025-01-01 00:30','2025-01-01 02:30'),
 ('2025-01-01 00:11:01','2025-01-01 04:39:07'),
 ('2025-12-31 23:51','2025-12-31 23:59'),
 ('2025-01-01 00:00','2025-01-01 00:01'),
])
def test_centred_hour_exact_parity_and_denominator(start,end):
    actual=voyage_segments(ts(start),ts(end))
    reference=centered_hour_segments(pd.DataFrame([{'physical_leg_id':'a','mmsi':'1','departure_time':ts(start),'arrival_time':ts(end)}]))
    pd.testing.assert_frame_equal(actual,reference[actual.columns])
    assert actual.segment_weight_ns.sum()==ts(end).value-ts(start).value


@pytest.mark.parametrize('hours',[4.,4.+1/3600])
def test_four_hour_boundary_dateline_exact_parity(hours):
    points=pd.DataFrame({'UTC':[ts('2025-01-01'),ts('2025-01-01')+pd.Timedelta(hours=hours)],'lat':[10.,12.],'lon':[179.,-179.],'vessel_key':['1','1']})
    targets=pd.Series([ts('2024-12-31 23:00'),ts('2025-01-01'),ts('2025-01-01 02:00'),ts('2025-01-01 06:00')])
    a,b,_=sample_ais(points,targets)
    old=interpolate_positions(points,targets)
    np.testing.assert_allclose(a,old.lat,equal_nan=True);np.testing.assert_allclose(b,old.lon,equal_nan=True)


def test_companion_exact_instant_only_and_partial_remains_unknown():
    v=SimpleNamespace(candidate_id='a',mmsi='1',departure_time=ts('2025-01-01 00:10'),arrival_time=ts('2025-01-01 02:10'))
    p=pd.DataFrame(columns=['UTC','lat','lon'])
    c=pd.DataFrame({'utc':[ts('2025-01-01 01:00'),ts('2025-01-01 02:00')],'mmsi':['1','1'],'lat':[10.,11.],'lon':[20.,21.],'gap_id':['g','g'],'anchor_source':['gfw','gfw']})
    result=build_voyage_input(v,p,c)
    assert result.track_evidence.tolist()==['unresolved','companion_reconstructed','unresolved']
    assert result.segment_weight_hours.sum()==2
    assert result.loc[1,'lat']==10


def test_companion_overlap_is_failure():
    v=SimpleNamespace(candidate_id='a',mmsi='1',departure_time=ts('2025-01-01'),arrival_time=ts('2025-01-01 02:00'))
    p=pd.DataFrame({'UTC':[v.departure_time,v.arrival_time],'lat':[10.,12.],'lon':[20.,22.]})
    c=pd.DataFrame({'utc':[ts('2025-01-01 01:00')],'mmsi':['1'],'lat':[11.],'lon':[21.],'gap_id':['g'],'anchor_source':['gfw']})
    with pytest.raises(ValueError,match='overlap'):build_voyage_input(v,p,c)


def test_no_endpoint_extrapolation_from_outside_voyage():
    v=SimpleNamespace(candidate_id='a',mmsi='1',departure_time=ts('2025-01-01'),arrival_time=ts('2025-01-01 02:00'))
    p=pd.DataFrame({'UTC':[ts('2024-12-31 23:00'),ts('2025-01-01 03:00')],'lat':[10.,12.],'lon':[20.,22.]})
    assert build_voyage_input(v,p,p.iloc[:0]).lat.isna().all()


@pytest.mark.parametrize('verified',[True,False])
def test_unknown_tc_snapshot_never_becomes_negative(verified):
    p=pd.DataFrame({'position_valid':[True,True,False],'tc_match_status':['valid_storm','valid_no_storm','invalid_position']})
    for f in ['tc_within_300km','tc_within_500km','tc_storm_present']:p[f]=[True,False,False]
    result=mask_uncertified_tc(p,full_year_verified=verified)
    assert result.tc_within_500km.iloc[0]
    assert pd.isna(result.tc_within_500km.iloc[2])
    assert pd.isna(result.tc_within_500km.iloc[1]) == (not verified)
    assert result.tc_within_500km_snapshot.iloc[1]==False


@pytest.mark.parametrize('change',[None,'sha','missing_month','gap'])
def test_verified_tc_flag_requires_matching_complete_audit(tmp_path,change):
 import json
 from weather_core.track_adapter import verify_tc_calendar_audit
 a={'sha256':'frozen','calendar_2025_within_source_span':True,'retained_month_counts':{str(i):1 for i in range(1,13)},'retained_invalid_centres':0,'within_retained_phase_gaps_over_3h':0}
 if change=='sha':a['sha256']='other'
 if change=='missing_month':del a['retained_month_counts']['12']
 if change=='gap':a['within_retained_phase_gaps_over_3h']=1
 p=tmp_path/'audit.json';p.write_text(json.dumps({'new':a}))
 c={'ibtracs_full_year_coverage_verified':True,'ibtracs_calendar_coverage_audit':str(p),'ibtracs_sha256':'frozen'}
 if change:
  with pytest.raises(ValueError):verify_tc_calendar_audit(c)
 else:assert verify_tc_calendar_audit(c)['new']==a


@pytest.mark.parametrize('start,end', [
    ('2025-02-10 13:41:34', '2025-08-20 22:37:39'),
    ('2025-01-24 05:02:03', '2025-09-03 22:10:46'),
    ('2025-06-22 13:31:29', '2025-12-14 21:50:40'),
])
def test_real_long_voyages_preserve_final_partial_hour(start, end):
    # Actual failures in annual buckets 0004 and 0008; require exact closure.
    a, b = ts(start), ts(end)
    actual = voyage_segments(a, b)
    assert actual.segment_start_utc.iloc[0] == a
    assert actual.segment_end_utc.iloc[-1] == b
    assert actual.segment_weight_ns.sum() == b.value - a.value
    assert np.array_equal(actual.segment_end_utc.iloc[:-1].array.asi8,
                          actual.segment_start_utc.iloc[1:].array.asi8)
    assert not actual.hour_anchor_utc.duplicated().any()
    v = SimpleNamespace(candidate_id='real-long', mmsi='1', departure_time=a, arrival_time=b)
    empty = pd.DataFrame(columns=['UTC', 'lat', 'lon'])
    prepared = build_voyage_input(v, empty, pd.DataFrame())
    assert prepared.segment_weight_ns.sum() == b.value - a.value
    assert prepared.track_evidence.eq('unresolved').all()
