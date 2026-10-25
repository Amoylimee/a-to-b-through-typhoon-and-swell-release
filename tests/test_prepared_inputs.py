"""Exercise the public boundary using synthetic local inputs, without download receipts."""
import importlib.util
import json
from pathlib import Path
import socket

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / 'src/reference/02_coding/pipeline'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chain = load('public_followup', PIPE / 'gfw_followup/60_extend_followup/10_run_followup.py')
prepared = __import__('prepared_inputs')
ocean = load('public_ocean_mask', PIPE / '40_construct_variables/50_reconstruct_companion_tracks/sea_path/ocean_mask.py')


@pytest.fixture
def buffer(tmp_path, monkeypatch):
    root = tmp_path / 'buffer'
    root.mkdir()
    frame = pd.DataFrame({
        'mmsi': pd.Series([123456789], dtype='int64'),
        'vessel_uid': pd.Series([10], dtype='int64'),
        'imo': pd.Series([1234567], dtype='Int64'),
        'hour_utc': pd.to_datetime(['2026-02-28T23:00Z'], utc=True),
        'lat': [4.5], 'lon': [7.1],
    })
    roster = tmp_path / 'roster.csv'
    pd.DataFrame({'mmsi': [123456789], 'vessel_uid': [10], 'registry_imo': [1234567]}).to_csv(roster, index=False)
    monkeypatch.setattr(prepared, 'ROSTER', roster)
    monkeypatch.setattr(prepared, 'ROSTER_SHA256', prepared.sha256_file(roster))
    digest = prepared.hashlib.sha256(b'123456789').digest()
    occupied = int.from_bytes(digest[:8], 'big') % 64
    records = []
    for i in range(64):
        part = frame if i == occupied else frame.iloc[:0]
        path = root / f'shard_{i:04d}.parquet'
        part.to_parquet(path, index=False)
        records.append({'shard': path.stem, 'path': str(path), 'rows': len(part),
                        'sha256': prepared.sha256_file(path)})
    summary = {'status': 'pass', 'month': '2026-02', 'closure': True,
               'unique_mmsi_hour': True, 'probe': False, 'output_rows': 1,
               'output_vessels': 1, 'buckets': records}
    (root / 'fleet_presence_unique_summary.json').write_text(json.dumps(summary))
    return root, summary, occupied


def test_local_buffer_and_controller_inputs_need_no_acquisition_receipts(buffer, tmp_path, monkeypatch):
    root, summary, _ = buffer
    def no_network(*args, **kwargs):
        raise AssertionError('prepared-input processing attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    assets = tmp_path / 'assets'
    assets.mkdir()
    manifest = {'status': 'pass', 'observation_end_exclusive': '2026-03-01'}
    for key in ['config', 'calendar_audit', 'source_matcher', 'adapter', 'derived_matcher']:
        path = assets / key
        path.write_text('synthetic input')
        manifest[key] = chain.record(path)
    (assets / 'manifest.json').write_text(json.dumps(manifest))
    baseline = tmp_path / 'january'
    baseline.mkdir()
    (baseline / 'summary.json').write_text(json.dumps({'status': 'pass', 'voyages': 1135377, 'old_voyages': 1111800}))
    monkeypatch.setattr(chain, 'JAN_RELEASE', baseline)
    assert prepared.validate_buffer(root) == summary
    records = chain.inputs(root, assets)
    assert len(records) == 3
    assert Path(records[0]['path']).name == 'fleet_presence_unique_summary.json'
    assert not (root / 'closure.json').exists()


@pytest.mark.parametrize('case, message', [
    ('missing-shard', '64 distinct'), ('changed-file', 'checksum'),
    ('row-count', 'row count'), ('month', 'validation for'),
    ('timestamp', 'UTC hours'), ('naive-time', 'UTC hours'),
    ('duplicate', 'duplicate prepared'), ('wrong-shard', 'stable shard'),
    ('identity', 'frozen vessel roster'), ('coordinates', 'geographic'),
    ('totals', 'totals'), ('roster', 'roster changed'),
])
def test_invalid_prepared_inputs_stop_before_run_creation(buffer, tmp_path, monkeypatch, case, message):
    root, summary, occupied = buffer
    record = summary['buckets'][occupied]
    path = Path(record['path'])
    frame = pd.read_parquet(path)
    if case == 'missing-shard':
        summary['buckets'].pop()
    elif case == 'changed-file':
        path.write_bytes(path.read_bytes() + b'changed')
    elif case == 'row-count':
        record['rows'] = 2
    elif case == 'month':
        summary['month'] = '2026-01'
    elif case == 'wrong-shard':
        other = summary['buckets'][(occupied + 1) % 64]
        record['shard'], other['shard'] = other['shard'], record['shard']
    elif case == 'totals':
        summary['output_rows'] = 2
    elif case == 'roster':
        prepared.ROSTER.write_text('changed')
    else:
        if case == 'timestamp': frame['hour_utc'] = pd.to_datetime(['2026-03-01T00:00Z'], utc=True)
        elif case == 'naive-time': frame['hour_utc'] = frame.hour_utc.dt.tz_localize(None)
        elif case == 'duplicate': frame = pd.concat([frame, frame], ignore_index=True)
        elif case == 'identity': frame['vessel_uid'] = 20
        elif case == 'coordinates': frame['lat'] = float('nan')
        frame.to_parquet(path, index=False)
        record.update(rows=len(frame), sha256=prepared.sha256_file(path))
    (root / 'fleet_presence_unique_summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(chain, 'RUNS', tmp_path)
    out = tmp_path / 'must_not_exist'
    with pytest.raises(ValueError, match=message):
        chain.run(root, tmp_path / 'no_weather_needed', out, 1)
    assert not out.exists()


def test_missing_land_mask_is_not_downloaded(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('missing input triggered a download')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    raw = tmp_path / 'absent'
    with pytest.raises(FileNotFoundError, match='Prepare'):
        ocean.require_gshhg_mask(raw)
    assert not raw.exists()
    raw.mkdir()
    local = raw / ocean.GSHHG_FILENAME
    local.write_bytes(b'synthetic local input; integrity is checked by the caller')
    assert ocean.require_gshhg_mask(raw) == local
