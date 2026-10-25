from pathlib import Path
import importlib.util
import sys
import pandas as pd
import pytest

path=Path(__file__).resolve().parents[1]/'235_reuse_equivalent_segmented_bounds.py'
sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('e007_equivalence_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('known,unknown,changed,expected',[
    ((1,9),(2,8),'both',True),
    ((1,9),(0,8),'departure',False),
    ((1,9),(0,8),'arrival',True),
    ((1,9),(2,10),'arrival',False),
    ((1,9),(2,10),'departure',True),
    ((None,None),(2,8),'departure',False),
    ((None,None),(2,8),'none',True),
])
def test_only_the_port_sides_actually_used_may_affect_reuse(known,unknown,changed,expected):
    stamp=lambda h:pd.Timestamp('2025-01-01T00:00Z')+pd.Timedelta(hours=h) if h is not None else pd.NaT
    plan=pd.DataFrame({'first_known':[stamp(known[0])],'last_known':[stamp(known[1])],
                       'first_unknown':[stamp(unknown[0])],'last_unknown':[stamp(unknown[1])],
                       'weather_source':['reuse_verified_option_c']})
    old=pd.DataFrame({'departure_port_lat':[0.],'departure_port_lon':[0.],
                      'arrival_port_lat':[1.],'arrival_port_lon':[1.],'port_endpoints_connected':[True]})
    new=old.copy()
    for role in ['departure','arrival']:
        if changed in [role,'both']:new[role+'_port_lon']+=.1
    assert bool(module.classify_equivalence(plan,new,old).reusable.iloc[0]) is expected


def test_different_physical_voyage_cannot_reuse_even_with_same_ports():
    t=pd.Timestamp('2025-01-01T00:00Z')
    plan=pd.DataFrame({'first_known':[t],'last_known':[t+pd.Timedelta(hours=4)],
                       'first_unknown':[t+pd.Timedelta(hours=1)],'last_unknown':[t+pd.Timedelta(hours=3)],
                       'weather_source':['rebuild_gfw']})
    ports=pd.DataFrame({'departure_port_lat':[0.],'departure_port_lon':[0.],
                        'arrival_port_lat':[1.],'arrival_port_lon':[1.],'port_endpoints_connected':[True]})
    assert not module.classify_equivalence(plan,ports,ports).reusable.iloc[0]
