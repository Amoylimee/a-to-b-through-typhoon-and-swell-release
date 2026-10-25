"""Guard new-month boundaries and reject invalid prepared observation inputs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

HERE=Path(__file__).resolve().parents[1]


def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,HERE/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


buffer=load('feb_buffer_test','20_build_cross_year_visits/10_rebuild_buffered_visits.py')
candidate=load('feb_candidate_test','30_build_buffered_candidates/10_enumerate_candidates.py')
weather=load('feb_weather_test','40_complete_tracks_and_weather/weather_extension.py')
chain=load('feb_chain_test','60_extend_followup/10_run_followup.py')
pending=load('feb_pending_test','50_audit_followup_censoring/20_audit_pending_followup.py')


def test_february_timestamp_keeps_year_and_excludes_march():
    old=buffer.unified.S.SHARD_SCHEMA.empty_table().to_pandas()
    f=pd.DataFrame({'vessel_uid':[1],'mmsi':[123456789],'imo':pd.array([1],dtype='Int64'),
        'hour_utc':pd.to_datetime(['2026-02-28T23:00Z'],utc=True),'lat':[0.],'lon':[1.]})
    result=buffer.buffered_table(old,f,'2026-02-01','2026-03-01').to_pandas()
    assert result.UTC.iloc[0]==pd.Timestamp('2026-02-28T23:30') and result.year.iloc[0]==2026
    f['hour_utc']=pd.to_datetime(['2026-03-01T00:00Z'],utc=True)
    with pytest.raises(ValueError):buffer.buffered_table(old,f,'2026-02-01','2026-03-01')


def test_candidate_cutoff_is_exclusive_and_2026_departures_excluded():
    f=pd.DataFrame({'vessel_uid_clean':['1','2','3'],
        'departure_time':['2025-12-31','2025-12-31','2026-01-01'],
        'arrival_time':['2026-02-28T23:30','2026-03-01T00:00','2026-02-01T00:00']})
    assert candidate.cohort(f,'2026-03-01').vessel_uid_clean.tolist()==['1']


def test_february_weather_calendar_has_fourteen_months_and_one_day_tc_margin():
    weather.configure('2026-03-01')
    assert len(weather.MONTHS)==14 and weather.MONTHS[-1]=='2026-02'
    assert weather.TC_END=='2026-03-02T00:00:00Z'
    _,source=weather.adapted_source()
    assert source.count('time_end_exclusive="2026-03-02T00:00:00Z"')==1
    assert 'month in config["allowed_weather_months"]' in source
    with pytest.raises(ValueError):weather.configure('2026-04-01')


def test_invalid_prepared_buffer_stops_before_creating_run(tmp_path,monkeypatch):
    buffer=tmp_path/'buffer';buffer.mkdir()
    (buffer/'fleet_presence_unique_summary.json').write_text(json.dumps({'status':'failed'}))
    monkeypatch.setattr(chain,'RUNS',tmp_path)
    out=tmp_path/'must_not_be_created'
    with pytest.raises(ValueError,match='prepared buffer'):
        chain.run(buffer,tmp_path/'absent_assets',out,2)
    assert not out.exists()


def test_changed_historical_point_overlapping_retained_support_stops(tmp_path):
    panel=tmp_path/'panel.parquet'
    pd.DataFrame({'candidate_id':['kept'],'vessel_uid_clean':['1'],'source_shard':['shard_0032'],
        'departure_time':pd.to_datetime(['2026-01-31T10:00Z'],utc=True),
        'arrival_time':pd.to_datetime(['2026-01-31T12:00Z'],utc=True)}).to_parquet(panel,index=False)
    old=pd.DataFrame({'vessel_uid':[1],'UTC':pd.to_datetime(['2026-01-31T11:30']),
                      'lat':[0.],'lon':[0.],'calc_speed_knots_masked':[np.nan]})
    new=old.copy();new['lat']=1.
    task={'root':str(tmp_path),'shard':'shard_0032','baseline_panel':buffer.ports.record(panel)}
    with pytest.raises(ValueError,match='touches a retained voyage'):
        buffer.audit_boundary_changes(task,old,new,new,tmp_path/'unused.feather')
    evidence=json.loads((tmp_path/'boundary_audits/shard_0032/audit.json').read_text())
    assert evidence['status']=='failed' and evidence['changes_inside_retained_support']==1


def test_pending_raw_departure_can_resolve_without_becoming_a_retained_voyage():
    group=pd.DataFrame({'vessel_uid':['1'],'shard':['shard_0000'],'exit_time':['2025-12-20']})
    visits=pd.DataFrame({'vessel_uid':['1'],'entry_time':['2026-02-01'],'exit_time':['2026-02-02'],'visit_confidence':['high']})
    disp=pd.DataFrame({'vessel_uid':['1'],'last_cleaned_utc':['2026-02-20']})
    result=pending.describe(group,visits,disp,'2026-02')
    assert result.followup_category.iloc[0]=='later_high_confidence_visit_observed'
    assert result.latest_buffer_position_observed.iloc[0]
