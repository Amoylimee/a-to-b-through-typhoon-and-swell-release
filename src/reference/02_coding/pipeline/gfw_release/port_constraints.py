"""Frozen port constraints and explicit handling of inconsistent blind-voyage ends.

A speed-filtered voyage may be retained at 36 kn while its missing-position
envelope still assumes 25 kn. Inconsistent port ends therefore leave the voyage
unbounded; they do not remove it or change either speed limit (Contract T015).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPEED_KM_H = 25.0 * 1.852


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2-lon1)
    a = np.sin((p2-p1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*6371.0088*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def port_ends_usable(port, has_positions: bool) -> bool:
    """Keep existing observed-end rules; audit wholly unobserved port pairs."""
    if has_positions:
        return True
    if not bool(port.port_endpoints_connected):
        return False
    # Classify conflicts under the frozen main +/-1 h rule. The zero-tolerance
    # diagnostic of other voyages must remain exactly as previously computed.
    hours = (port.arrival_time-port.departure_time).total_seconds()/3600 + 2
    distance = haversine_km(port.departure_port_lat, port.departure_port_lon,
                            port.arrival_port_lat, port.arrival_port_lon)
    return bool(np.isfinite(distance) and hours >= 0 and distance <= SPEED_KM_H*hours + 1e-6)


def port_feasibility(prepared: Path) -> dict:
    panel = pd.read_parquet(prepared/'voyage_panel.parquet', columns=['candidate_id', 'track_hours'])
    ports = pd.read_parquet(prepared/'port_endpoints.parquet')
    blind = ports.merge(panel.loc[panel.track_hours.eq(0), ['candidate_id']],
                        on='candidate_id', validate='one_to_one')
    hours = (blind.arrival_time-blind.departure_time).dt.total_seconds()/3600
    distance = haversine_km(blind.departure_port_lat, blind.departure_port_lon,
                            blind.arrival_port_lat, blind.arrival_port_lon)
    connected = blind.port_endpoints_connected.fillna(False).astype(bool)
    conflict = connected & (distance > SPEED_KM_H*(hours+2) + 1e-6)
    ids = sorted(blind.loc[conflict, 'candidate_id'].astype(str))
    return {'status': 'checked', 'no_position_voyages': len(blind), 'speed_cap_knots': 25,
            'main_port_time_tolerance_hours': 1, 'conflicting_main_port_endpoints': len(ids),
            'main_conflict_hours': float(hours[conflict].sum()),
            'conflicting_candidate_ids': ids, 'conflict_policy': 'retain_voyage_unbounded',
            'old_conflict_ids_reused': False}
