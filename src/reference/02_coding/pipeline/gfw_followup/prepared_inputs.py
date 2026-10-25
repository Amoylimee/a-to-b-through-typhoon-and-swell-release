"""Offline checks for prepared observation buffers and the shared grid rule."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CODING = Path(__file__).resolve().parents[2]
ROSTER = CODING / 'data/frozen_inputs/roster_2025/p2/extended_sample_mmsi_2025.csv'
ROSTER_SHA256 = '32526296cde85a63489de621f29880a48b78686391f0b3eef44f3e64c8d9c078'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def deduplicate(frame):
    """T029: exact copies keep one; a conflicting vessel-hour keeps no position."""
    keys = ['mmsi', 'hour_utc', 'lat', 'lon']
    frame = frame.copy()
    # Match E006/85 exactly: provider float32 coordinates identify a 0.01-degree cell.
    frame['_cell'] = (np.rint(frame.lat.to_numpy()*100).astype(np.int64)+9000)*36001 + \
        (np.rint(frame.lon.to_numpy()*100).astype(np.int64)+18000)
    exact = frame.duplicated(['mmsi', 'hour_utc', '_cell'])
    distinct = frame.loc[~exact].copy()
    conflict = distinct.duplicated(['mmsi', 'hour_utc'], keep=False)
    unique = distinct.loc[~conflict, keys].sort_values(['mmsi', 'hour_utc']).reset_index(drop=True)
    conflicts = distinct.loc[conflict].drop(columns='_cell').copy()
    return unique, conflicts, int(exact.sum())


def validate_buffer(root, month='2026-02'):
    """Validate supplied shards before creating a run; never acquire any data.

    The input is the prepared fleet buffer, not provider responses or download
    receipts. Its summary binds all 64 shards by hash, row count and month.
    Scientific baseline, vessel-roster and weather checks remain downstream.
    """
    if month not in {'2026-01', '2026-02'}:
        raise ValueError('unsupported observation month')
    root = Path(root)
    summary = json.loads((root / 'fleet_presence_unique_summary.json').read_text())
    if (summary.get('status') != 'pass' or summary.get('month') != month
            or summary.get('probe') or summary.get('closure') is not True
            or summary.get('unique_mmsi_hour') is not True):
        raise ValueError('prepared buffer has not passed validation for ' + month)
    records = summary.get('buckets', [])
    expected = {f'shard_{i:04d}' for i in range(64)}
    if len(records) != 64 or {r['shard'] for r in records} != expected:
        raise ValueError('prepared buffer requires all 64 distinct shards')
    if sha256_file(ROSTER) != ROSTER_SHA256:
        raise ValueError('frozen vessel roster changed')
    roster = pd.read_csv(ROSTER, dtype=str, usecols=['mmsi', 'vessel_uid', 'registry_imo'])
    for key in roster:
        roster[key] = pd.to_numeric(roster[key].str.strip().str.replace(r'\.0$', '', regex=True)).astype('Int64')
    if (roster[['mmsi', 'vessel_uid']].isna().any().any()
            or roster.mmsi.duplicated().any() or roster.vessel_uid.duplicated().any()):
        raise ValueError('frozen vessel roster is not one-to-one')
    roster = roster.set_index('mmsi')
    start = pd.Timestamp(month + '-01', tz='UTC')
    end = start + pd.DateOffset(months=1)
    columns = ['mmsi', 'vessel_uid', 'imo', 'hour_utc', 'lat', 'lon']
    rows, identity = 0, {}
    for record in records:
        path = Path(record['path'])
        if not path.is_absolute():
            raise ValueError('prepared shard paths must be absolute')
        if sha256_file(path) != record['sha256']:
            raise ValueError('prepared shard checksum changed: ' + record['shard'])
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) != record['rows']:
            raise ValueError('prepared shard row count differs: ' + record['shard'])
        for key in ['mmsi', 'vessel_uid']:
            if (not pd.api.types.is_integer_dtype(frame[key])
                    or frame[key].isna().any() or frame[key].le(0).any()):
                raise ValueError('prepared identifiers must be positive integers: ' + key)
        hours = frame.hour_utc
        if (not isinstance(hours.dtype, pd.DatetimeTZDtype)
                or str(hours.dt.tz) != 'UTC' or hours.isna().any()
                or not (hours.ge(start) & hours.lt(end)).all()
                or not hours.eq(hours.dt.floor('h')).all()):
            raise ValueError('prepared hours must be UTC hours inside ' + month)
        if (not frame.lat.between(-90, 90).all()
                or not frame.lon.between(-180, 180).all()):
            raise ValueError('prepared coordinates outside geographic bounds')
        if frame.duplicated(['mmsi', 'hour_utc']).any():
            raise ValueError('duplicate prepared vessel-hour')
        expected_id = frame.mmsi.map(roster.vessel_uid)
        expected_imo = frame.mmsi.map(roster.registry_imo)
        if (expected_id.isna().any() or not frame.vessel_uid.eq(expected_id).all()
                or not (frame.imo.eq(expected_imo).fillna(False)
                        | (frame.imo.isna() & expected_imo.isna())).all()):
            raise ValueError('prepared identities differ from the frozen vessel roster')
        ids = frame[['mmsi', 'vessel_uid']].drop_duplicates()
        if ids.mmsi.duplicated().any() or ids.vessel_uid.duplicated().any():
            raise ValueError('prepared vessel identity is not one-to-one')
        bucket = int(record['shard'].split('_')[1])
        for mmsi, vessel in ids.itertuples(index=False, name=None):
            digest = hashlib.sha256(str(mmsi).encode('utf-8')).digest()
            if int.from_bytes(digest[:8], 'big') % 64 != bucket:
                raise ValueError('prepared vessel is in the wrong stable shard')
            if vessel in identity and identity[vessel] != mmsi:
                raise ValueError('prepared identity conflicts across shards')
            identity[vessel] = mmsi
        rows += len(frame)
    if rows != summary.get('output_rows') or len(identity) != summary.get('output_vessels'):
        raise ValueError('prepared buffer totals differ from its shards')
    return summary
