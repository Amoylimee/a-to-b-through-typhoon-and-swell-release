from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track_core.review_checks import speed_violations,nearest_anchor_errors,distance_km,recompute_covered_hours


def test_speed_check_does_not_trust_reported_speed():
    legs=pd.DataFrame(dict(leg_status=['resolved'],schedule_status=['resolved'],
        scheduled_start_epoch_s=[0.],scheduled_end_epoch_s=[3600.],path_distance_km=[60.],implied_speed_knots=[1.]))
    assert speed_violations(legs)[0]==1


def test_nearest_check_detects_wrong_anchor_distance():
    origin=pd.Timestamp('2025-01-01T00:00Z')
    track=pd.DataFrame(dict(gap_id=['a'],utc=[origin+pd.Timedelta(hours=1)],hours_to_nearest_anchor=[1.]))
    anchors=pd.DataFrame(dict(gap_id=['a','a'],scheduled_epoch_s=[origin.timestamp(),origin.timestamp()+4000]))
    assert nearest_anchor_errors(track,anchors)==1


def test_independent_coverage_clips_and_excludes_failed_leg():
    gaps=pd.DataFrame(dict(gap_id=['a'],gap_start_utc=pd.to_datetime([3600],unit='s',utc=True),
        gap_end_utc=pd.to_datetime([10800],unit='s',utc=True)))
    legs=pd.DataFrame(dict(gap_id=['a','a'],leg_status=['resolved','no_path'],schedule_status=['resolved']*2,
        scheduled_start_epoch_s=[0.,7200.],scheduled_end_epoch_s=[7200.,14400.]))
    assert recompute_covered_hours(legs,gaps).loc['a']==1


def test_distance_crosses_dateline_short_way():
    assert np.isclose(distance_km(0,179.5,0,-179.5),111.19492664)
