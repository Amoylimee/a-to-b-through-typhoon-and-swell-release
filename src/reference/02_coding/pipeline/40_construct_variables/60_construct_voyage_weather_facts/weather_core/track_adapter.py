"""Centred-hour quadrature over accepted voyages, with explicit track evidence.

AIS interpolation is the archived linear latitude / shortest longitude rule.
No extrapolation and no bridge across >4 hours. Companion points are consumed
only at their exact recorded instants; partial-hour midpoints are not relabelled
as integer-hour observations. Missing positions retain their time weights.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HOUR_NS = 3_600_000_000_000
HALF_NS = HOUR_NS // 2


def _hour_anchors_ns(start, end):
    first = (start + HALF_NS) // HOUR_NS * HOUR_NS
    last = -(-(end - HALF_NS) // HOUR_NS) * HOUR_NS
    # Use small integer hour indices: arange on epoch nanoseconds can lose
    # the inclusive +1 when computing its length for long voyages.
    return np.arange(first // HOUR_NS, last // HOUR_NS + 1, dtype=np.int64) * HOUR_NS


def voyage_segments(start, end):
    start, end = pd.Timestamp(start).value, pd.Timestamp(end).value
    if end <= start:
        raise ValueError('nonpositive voyage window')
    hours = _hour_anchors_ns(start, end)
    left = np.maximum(start, hours - HALF_NS)
    right = np.minimum(end, hours + HALF_NS)
    valid = right > left
    hours, left, right = hours[valid], left[valid], right[valid]
    return pd.DataFrame({
        'hour_anchor_utc': pd.to_datetime(hours, utc=True),
        'segment_start_utc': pd.to_datetime(left, utc=True),
        'segment_end_utc': pd.to_datetime(right, utc=True),
        'position_time_utc': pd.to_datetime(left + (right-left)//2, utc=True),
        'segment_weight_ns': right-left,
        'segment_weight_hours': (right-left)/HOUR_NS,
    })


def sample_ais(points, targets):
    """Sorted, identity-validated AIS within this voyage only."""
    target = pd.DatetimeIndex(targets).asi8
    lat, lon = np.full(len(target), np.nan), np.full(len(target), np.nan)
    source = np.full(len(target), 'unresolved', dtype=object)
    if points.empty:
        return lat, lon, source
    times = pd.DatetimeIndex(points.UTC).asi8
    if (np.diff(times) <= 0).any():
        raise ValueError('AIS must be sorted and unique')
    right = np.searchsorted(times, target)
    exact = (right < len(times)) & (times[np.minimum(right,len(times)-1)] == target)
    plat, plon = points.lat.to_numpy(float), points.lon.to_numpy(float)
    lat[exact], lon[exact] = plat[right[exact]], plon[right[exact]]
    source[exact] = 'observed_ais'
    idx = np.flatnonzero((~exact) & (right > 0) & (right < len(times)))
    r, l = right[idx], right[idx]-1
    keep = times[r] - times[l] <= 4*HOUR_NS
    idx, r, l = idx[keep], r[keep], l[keep]
    f = (target[idx]-times[l])/(times[r]-times[l])
    lat[idx] = plat[l] + f*(plat[r]-plat[l])
    lon[idx] = plon[l] + f*((plon[r]-plon[l]+180)%360-180)
    source[idx] = 'ais_interpolated_le4h'
    return lat, (lon+180)%360-180, source


def build_voyage_input(voyage, points, companion):
    rows = voyage_segments(voyage.departure_time, voyage.arrival_time)
    inside = points.loc[points.UTC.between(voyage.departure_time, voyage.arrival_time)]
    rows['lat'], rows['lon'], rows['track_evidence'] = sample_ais(inside,rows.position_time_utc)
    rows['candidate_id'] = voyage.candidate_id
    rows['mmsi'] = str(voyage.mmsi)
    rows['gap_id'] = pd.Series(None, index=rows.index, dtype=object)
    rows['companion_anchor_source'] = pd.Series(None,index=rows.index,dtype=object)
    if not companion.empty:
        if companion.utc.duplicated().any():
            raise ValueError('duplicate companion candidate/time key')
        if not companion.mmsi.astype(str).eq(str(voyage.mmsi)).all():
            raise ValueError('companion MMSI mismatch')
        matches = companion.set_index('utc').reindex(pd.DatetimeIndex(rows.position_time_utc))
        has = matches.lat.notna().to_numpy()
        if (has & rows.track_evidence.ne('unresolved').to_numpy()).any():
            raise ValueError('observed/companion evidence overlap')
        rows.loc[has,'lat'] = matches.lat.to_numpy()[has]
        rows.loc[has,'lon'] = matches.lon.to_numpy()[has]
        rows.loc[has,'track_evidence'] = 'companion_reconstructed'
        rows.loc[has,'gap_id'] = matches.gap_id.to_numpy()[has]
        rows.loc[has,'companion_anchor_source'] = matches.anchor_source.to_numpy()[has]
    return rows


def mask_uncertified_tc(frame, *, full_year_verified):
    """Keep snapshot diagnostics distinct from certified negative exposure."""
    frame = frame.copy()
    valid = frame.position_valid.to_numpy(bool)
    for field in ['tc_within_300km','tc_within_500km','tc_storm_present']:
        raw = frame[field].astype('boolean')
        frame[field+'_snapshot'] = raw.mask(~valid)
        frame[field] = raw.mask(~valid | ((~raw.fillna(False).to_numpy(bool)) & (not full_year_verified)))
    frame['tc_snapshot_coverage_verified'] = bool(full_year_verified)
    frame.loc[~valid,'tc_match_status'] = 'unresolved_position'
    if not full_year_verified:
        frame.loc[valid,'tc_match_status'] = 'snapshot_match_coverage_unverified'
    return frame


def verify_tc_calendar_audit(config):
    """A verified flag requires an audit of the exact selected snapshot.

    This certifies calendar support/internal continuity, not final best tracks
    or perfect real-world storm detection.
    """
    if not config['ibtracs_full_year_coverage_verified']:
        return None
    import json
    from pathlib import Path
    audit=json.loads(Path(config['ibtracs_calendar_coverage_audit']).read_text())
    selected=audit['new']
    if selected['sha256']!=config['ibtracs_sha256']:
        raise ValueError('TC calendar audit belongs to a different snapshot')
    if not selected['calendar_2025_within_source_span']:
        raise ValueError('TC snapshot does not span calendar 2025')
    if set(selected['retained_month_counts'])!={str(i) for i in range(1,13)}:
        raise ValueError('TC calendar audit lacks a 2025 month')
    if selected['retained_invalid_centres'] or selected['within_retained_phase_gaps_over_3h']:
        raise ValueError('TC snapshot has unresolved centre or continuity gaps')
    return audit
