from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from port_segmentation import visit_builder, guard_candidates, nearest_unique


def run(lons, speeds, anchors=(0.,), cities=("A",), hours=None):
    lons = np.array(lons, dtype=float)
    times = np.datetime64("2025-01-01T00:30", "ns") + np.array(hours if hours is not None else range(len(lons))) * np.timedelta64(1, "h")
    near = np.abs(lons[:, None]-np.asarray(anchors)[None, :]).argmin(axis=1)
    distance = np.abs(lons-np.asarray(anchors)[near])*111.1950802335
    return visit_builder(cities)("v", "imo", times, np.array(speeds), np.zeros(len(lons)), lons,
                                near, distance, np.zeros(len(anchors)), np.asarray(anchors)), times


def test_stop_in_open_annulus_is_not_lost():
    v, _ = run([.01, .032, .032, .06], [5, .1, 0, 5])
    assert v[0]["has_stop"] and v[0]["visit_confidence"] == "high"


def test_stationary_annulus_can_open_marked_visit():
    v, _ = run([.06, .032, .032, .06], [5, 3, 0, 5])
    assert len(v) == 1 and v[0]["entry_rule"] == "stop_supported_annulus"
    assert v[0]["has_stop"]


def test_transit_annulus_does_not_open_visit():
    v, _ = run([.06, .032, .032, .06], [5, 3, .3, 5])
    assert not v


def test_same_city_cells_share_region_and_select_stop_cell():
    v, t = run([0, .03, .05, .05, .09], [5, 5, .1, 0, 5], anchors=(0., .05), cities=("A", "A"))
    assert len(v) == 1 and v[0]["exit_time"] == t[-1]
    assert v[0]["anchorage_idx"] == 1 and v[0]["opening_anchorage_idx"] == 0


def test_foreign_nearest_cell_does_not_hide_another_nearby_opening_city_cell():
    v,t=run([0,.025,.052,.052,.1],[5,0,5,0,5],anchors=(0.,.05,.051),cities=('A','A','B'))
    assert len(v)==1 and v[0]['exit_time']==t[-1]
    assert v[0]['ambiguous_city'] and v[0]['has_stop']


def test_unique_grid_query_is_exact_including_ties_and_repeated_positions():
    from sklearn.neighbors import BallTree
    rng=np.random.default_rng(20261011)
    anchors=np.r_[rng.uniform(-1,1,size=(100,2)),[[0,0],[0,.02]]]
    tree=BallTree(np.deg2rad(anchors),metric='haversine')
    coords=np.r_[rng.uniform(-1,1,size=(300,2)),[[0,.01],[0,.01]],np.tile([[.5,.5]],(300,1))]
    d,i=tree.query(np.deg2rad(coords),k=1)
    cached_i,cached_d=nearest_unique(coords[:,0],coords[:,1],tree)
    np.testing.assert_array_equal(cached_i,i[:,0])
    np.testing.assert_array_equal(cached_d,d[:,0]*6371.0088)


def test_different_city_cannot_donate_stop():
    v, _ = run([0, .022, .022, .06], [5, .1, 0, 5], anchors=(0., .025), cities=("A", "B"))
    assert not v[0]["has_stop"] and v[0]["ambiguous_city"]


def test_exit_sample_is_reused_for_next_entry():
    v, t = run([0, .001, .10, .10, .15], [5, 0, 5, 0, 5], anchors=(0., .10), cities=("A", "B"))
    assert len(v) == 2 and all(q["has_stop"] for q in v)
    assert v[0]["exit_time"] == v[1]["entry_time"] == t[2]


def test_continuously_observed_annulus_does_not_become_ais_gap():
    v, _ = run([0, .032, .032, .032, .032, 0, .06], [5]*7)
    assert not v[0]["has_gap"]


def test_true_consecutive_gap_is_retained():
    v, _ = run([0, 0, .06], [5]*3, hours=[0, 4, 5])
    assert v[0]["has_gap"] and v[0]["visit_confidence"] == "high"


def test_boundary_gap_is_not_lost_in_exact_point_timestamp():
    v, t = run([0, 0, .06], [5, 0, 5], hours=[0, 1, 96])
    assert v[0]["exit_gap_hours"] == 95 and v[0]["exit_bracket_start"] == t[1]


def test_two_gridded_points_do_not_manufacture_confirmed_short_stop():
    v, _ = run([-.06, 0, .01, .06], [5, 2, .6, 5])
    assert v[0]["possible_short_stop"] and not v[0]["has_stop"]
    assert v[0]["visit_confidence"] == "low"


def test_two_cell_displacement_is_not_a_possible_stop():
    # 0.03 degrees in 1 h exceeds one cell diagonal plus a 0.2 kn drift over 2 h.
    v, _ = run([-.06, -.015, .015, .06], [5, 2, 2, 5])
    assert len(v) == 1 and not v[0]["possible_short_stop"]


def test_two_hundredths_in_one_hour_remains_a_possible_stop():
    # The R35-like 2.22 km hourly displacement stays within the one-cell allowance.
    v, _ = run([-.06, 0, .02, .06], [5, 2, 1.2, 5])
    assert v[0]["possible_short_stop"] and not v[0]["has_stop"]


def visits_frame(rows, cities):
    f = pd.DataFrame(rows)
    f["analytical_city_id"] = cities
    for c in ["entry_time", "exit_time", "first_seen_time", "last_seen_time"]:
        f[c] = pd.to_datetime(f[c], utc=True)
    return f


def test_guard_quarantines_uncertain_intermediate_without_claiming_it_a_call():
    rows, t = run([0, 0, .1, .11, .2, .2, .26], [5, 0, 2, .6, 5, 0, 5],
                  anchors=(0., .10, .20), cities=("A", "B", "C"))
    v = visits_frame(rows, ["A", "B", "C"])
    f = pd.DataFrame({"vessel_uid_clean": ["v"], "origin_city": ["A"], "dest_city": ["C"],
                      "departure_time": [pd.Timestamp(t[2], tz="UTC")], "arrival_time": [pd.Timestamp(t[4], tz="UTC")]})
    g = guard_candidates(f, v)
    assert not g.segmentation_eligible.iloc[0]
    assert g.segmentation_review_reasons.iloc[0] == "unresolved_intermediate_short_stop"


def test_empty_vessel_has_no_visits():
    v, _ = run([], [])
    assert v == []


def test_different_city_cannot_donate_an_ais_gap():
    v, _ = run([0, .022, .022, .06], [5]*4, anchors=(0., .025), cities=("A", "B"), hours=[0, 1, 5, 6])
    assert not v[0]["has_gap"]
    assert v[0]["ambiguous_city"]


def test_long_departure_bracket_is_quarantined():
    rows,t=run([0,0,.06,.1,.1,.16],[5,0,5,5,0,5],anchors=(0.,.1),cities=("A","B"),hours=[0,1,6,7,8,9])
    v=visits_frame(rows,["A","B"])
    f=pd.DataFrame({'vessel_uid_clean':['v'],'origin_city':['A'],'dest_city':['B'],
                    'departure_time':[pd.Timestamp(t[2],tz='UTC')],'arrival_time':[pd.Timestamp(t[3],tz='UTC')]})
    g=guard_candidates(f,v)
    assert g.segmentation_review_reasons.iloc[0]=='unresolved_departure_bracket'


def test_impossible_observed_motion_is_quarantined_without_picking_a_good_point():
    rows,t=run([0,0,.1,.11,.2,.2,.26],[5,0,2,.6,5,0,5],anchors=(0.,.1,.2),cities=('A','B','C'))
    v=visits_frame(rows,['A','B','C'])
    f=pd.DataFrame({'vessel_uid_clean':['v'],'origin_city':['A'],'dest_city':['C'],
                    'departure_time':[pd.Timestamp(t[2],tz='UTC')],'arrival_time':[pd.Timestamp(t[4],tz='UTC')]})
    points=pd.DataFrame({'vessel_uid':['v']*3,'UTC':pd.to_datetime(t[2:5],utc=True),'lat':[0.,0.,0.],'lon':[.1,10.,.2]})
    g=guard_candidates(f,v,points)
    assert 'infeasible_observed_motion' in g.segmentation_review_reasons.iloc[0]
    assert not g.segmentation_eligible.iloc[0]
