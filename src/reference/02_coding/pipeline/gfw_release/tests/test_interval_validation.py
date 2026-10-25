import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from interval_validation_core import exposure_comparison


def fixture():
    return pd.DataFrame({'is_hidden':[False,True,True], 'gfw_swh':[1.,4.,8.],
        'reference_ais_swh':[2.,4.,99.], 'wave_lo_m':[0.,3.,float('nan')],
        'wave_hi_m':[0.,5.,float('nan')], 'wave_status':['known','matched','no_wet_era5_cells'],
        'gfw_tc300':[0.,0.,1.], 'gfw_tc500':[0.,0.,1.],
        'reference_ais_tc300':[1.,0.,float('nan')], 'reference_ais_tc500':[1.,0.,float('nan')],
        'tc300_possible':[0,0,1], 'tc500_possible':[0,1,1]})


def test_same_supported_mean_domain():
    x=exposure_comparison(fixture(),'reference_ais',False)
    assert x['wave_lower_m']==2
    assert x['wave_upper_m']==3
    assert x['wave_reference_m']==2.5
    assert x['wave_inside']
    assert x['wave_unsupported_hours']==1
    assert x['tc300_compared_hours']==2


def test_known_error_is_separate_from_conditional_bounds():
    x=exposure_comparison(fixture(),'reference_ais',True)
    assert not x['tc300_inside']
    assert x['tc300_reference_hours']==1
    assert x['tc300_upper_hours']==0


def test_no_reference_is_unevaluable():
    f=fixture(); f['reference_ais_swh']=float('nan'); f['reference_ais_tc300']=float('nan')
    x=exposure_comparison(f,'reference_ais',False)
    assert x['wave_inside'] is None
    assert x['tc300_inside'] is None
    assert x['wave_compared_hours']==0


def test_missing_known_position_is_excluded_from_tc_domain():
    f=fixture(); f.loc[0,['gfw_swh','gfw_tc300','gfw_tc500']]=float('nan')
    x=exposure_comparison(f,'reference_ais',True)
    assert x['tc300_compared_hours']==1
    assert x['tc300_reference_hours']==0
    assert x['wave_compared_hours']==1
