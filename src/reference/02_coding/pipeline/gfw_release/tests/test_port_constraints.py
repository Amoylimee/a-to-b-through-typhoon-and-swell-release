import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'gfw_release'))
from port_constraints import port_ends_usable


def make_port(hours=5, connected=True):
    start = pd.Timestamp('2025-07-11T18:30:00Z')
    return SimpleNamespace(departure_time=start, arrival_time=start+pd.Timedelta(hours=hours),
        departure_port_lat=43.3370806317, departure_port_lon=5.3411529067,
        arrival_port_lat=41.3328091616, arrival_port_lon=2.1610754908,
        port_endpoints_connected=connected)


def test_restored_five_hour_voyage_cannot_be_bounded_at_25_knots():
    assert not port_ends_usable(make_port(), False)


def test_feasible_blind_voyage_keeps_port_constraints():
    assert port_ends_usable(make_port(hours=8), False)


def test_main_feasible_voyage_does_not_change_the_zero_tolerance_diagnostic():
    assert port_ends_usable(make_port(hours=6), False)


def test_observed_voyage_preserves_existing_endpoint_handling():
    assert port_ends_usable(make_port(connected=False), True)


def test_unconnected_blind_voyage_has_no_port_envelope():
    assert not port_ends_usable(make_port(connected=False), False)
