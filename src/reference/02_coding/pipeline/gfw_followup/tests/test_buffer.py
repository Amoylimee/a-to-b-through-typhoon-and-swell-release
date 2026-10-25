"""Regression checks for cell precision, conflict handling and year boundaries."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fleet = load('e009_test_fleet', 'prepared_inputs.py')
buffered = load('e009_test_buffered', '20_build_cross_year_visits/10_rebuild_buffered_visits.py')
candidate = load('e009_test_cohort', '30_build_buffered_candidates/10_enumerate_candidates.py')


def test_float32_cell_repeats_do_not_become_false_conflicts():
    frame = pd.DataFrame({'mmsi':[1,1], 'hour_utc':pd.to_datetime(['2026-01-01']*2,utc=True),
                          'lat':[4.5,4.5], 'lon':[138.93,138.92999267578125]})
    unique, conflicts, repeats = fleet.deduplicate(frame)
    assert len(unique)==1 and conflicts.empty and repeats==1
    assert unique.lon.iloc[0]==138.93


def test_conflicting_hour_drops_every_position_but_not_the_next_hour():
    frame = pd.DataFrame({'mmsi':[1,1,1,1],
        'hour_utc':pd.to_datetime(['2026-01-01 00:00']*3+['2026-01-01 01:00'],utc=True),
        'lat':[4.5,4.5,4.6,4.6], 'lon':[7.1,7.1,7.2,7.2]})
    unique, conflicts, repeats = fleet.deduplicate(frame)
    assert repeats==1 and len(conflicts)==2 and len(unique)==1
    assert unique.hour_utc.iloc[0]==pd.Timestamp('2026-01-01 01:00',tz='UTC')
    assert len(frame)-repeats-len(conflicts)==len(unique)


def test_january_keeps_observation_year_and_hour_midpoint():
    empty = buffered.unified.S.SHARD_SCHEMA.empty_table().to_pandas()
    january = pd.DataFrame({'vessel_uid':[10], 'mmsi':[123456789], 'imo':pd.array([1234567],dtype='Int64'),
        'hour_utc':pd.to_datetime(['2026-01-01T00:00:00Z']), 'lat':[4.5], 'lon':[7.1]})
    result = buffered.buffered_table(empty,january).to_pandas()
    assert result.UTC.iloc[0]==pd.Timestamp('2026-01-01 00:30') and result.year.iloc[0]==2026
    with pytest.raises(ValueError,match='duplicate'):
        buffered.buffered_table(empty,pd.concat([january,january]))
    january['hour_utc']=pd.to_datetime(['2026-02-01'],utc=True)
    with pytest.raises(ValueError,match='outside January'):
        buffered.buffered_table(empty,january)


def test_departure_cohort_keeps_cross_year_arrival_and_excludes_january_departure():
    frame = pd.DataFrame({'vessel_uid_clean':['1','2'],
        'departure_time':['2025-12-31 12:00','2026-01-01 12:00'],
        'arrival_time':['2026-01-05 12:00','2026-01-05 12:00']})
    result = candidate.cohort(frame)
    assert result.vessel_uid_clean.tolist()==['1']
    assert result.arrival_time.dt.year.tolist()==[2026]


def test_position_difference_cannot_enter_candidates_without_port_only_audit():
    result={'baseline_points_exact':False}
    with pytest.raises(ValueError,match='unexplained'):
        candidate.validate_position_evidence(result,{})
    with pytest.raises(ValueError,match='missing bound'):
        candidate.validate_position_evidence(result,{'validation_scope':'candidate_port_inputs_only'})


def test_boundary_audit_cannot_be_reused_for_a_different_shard(tmp_path):
    import json
    path=tmp_path/'audit.json'
    path.write_text(json.dumps({'status':'pass','scope':'candidate_port_inputs_only','shard':'shard_0057'}))
    receipt=candidate.base.record(path)
    with pytest.raises(ValueError,match='does not certify'):
        candidate.validate_position_evidence(
            {'baseline_points_exact':False,'shard':'shard_0001','boundary_cleaning_audit':receipt},
            {'validation_scope':'candidate_port_inputs_only','boundary_cleaning_audit':receipt})
