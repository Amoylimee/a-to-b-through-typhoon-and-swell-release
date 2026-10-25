#!/usr/bin/env python3
"""Expose the one-cell composed tracks in the layout read by the V1 scripts.

The composed weather run keeps each shard's track and summary but not the
step-4 records of unresolved point pairs. Their two ends are the cleaned GFW
points bounding each unresolved gap, so they are recovered from the frozen
cleaned points. The recovery must first reproduce E006 step-4 records exactly.
    python 270_build_one_cell_track_view.py --weather <run> --out <run> [--check-shards ...]
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa
HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
sys.path.insert(0, str(CODING / 'pipeline/gfw_release'))
import step6_core as core
CLEAN = CODING / '.runs/2026-10-05_2305_E006-gfw-cleaning/shards'
E006_TRACKS = CODING / '.runs/2026-10-06_2341_E006-gfw-ten-minute-tracks/shards'
NS_MIN10 = 600_000_000_000
NS_HOUR = 3_600_000_000_000
COMPARED = ['candidate_id', 'gap_start_ns', 'gap_end_ns', 'gap_class', 'cells', 'start_lat', 'start_lon', 'end_lat', 'end_lon']


def gap_ends(shard, track_dir):
    """Unresolved point-pair gaps of one shard, with both bounding GFW points."""
    pa.set_cpu_count(1)
    cand = pd.read_parquet(track_dir / 'candidate_summary.parquet', columns=['candidate_id', 'vessel_uid_clean'])
    g = pd.read_parquet(track_dir / 'ten_minute_complete_candidate.parquet', columns=['candidate_id', 'utc', 'gap_id', 'gap_class'],
                        filters=[('gap_id', '!=', '')])
    g = g.loc[g.gap_id.notna() & g.gap_class.astype(str).str.startswith('unresolved')].copy()
    g['t'] = pd.DatetimeIndex(g.utc).asi8
    gp = g.groupby('gap_id', sort=False).agg(candidate_id=('candidate_id', 'first'), first=('t', 'min'), last=('t', 'max'),
                                             cells=('t', 'size'), gap_class=('gap_class', 'first')).reset_index(drop=True)
    gp['gap_start_ns'], gp['gap_end_ns'] = gp['first'] - NS_MIN10, gp['last'] + NS_MIN10
    if not ((gp.gap_end_ns - gp.gap_start_ns) // NS_MIN10 - 1).eq(gp.cells).all():
        raise ValueError(f'{shard}: an unresolved gap has missing interior cells')
    gp['candidate_id'] = gp.candidate_id.astype(str)
    gp = gp.merge(cand.astype(str), on='candidate_id', how='left', validate='many_to_one')
    points = pd.read_feather(CLEAN / shard / 'mask' / f'{shard}.feather', columns=['vessel_uid', 'UTC', 'lat', 'lon'])
    points = pd.DataFrame({'vessel_uid_clean': points.vessel_uid.astype(str), 't': pd.DatetimeIndex(points.UTC).asi8,
                           'lat': points.lat.to_numpy(float), 'lon': points.lon.to_numpy(float)})
    if points.duplicated(['vessel_uid_clean', 't']).any():
        raise ValueError(f'{shard}: duplicate cleaned point times')
    for side, column in [('start', 'gap_start_ns'), ('end', 'gap_end_ns')]:
        gp = gp.merge(points.rename(columns={'t': column, 'lat': f'{side}_lat', 'lon': f'{side}_lon'}),
                      on=['vessel_uid_clean', column], how='left', validate='many_to_one')
    if gp[['start_lat', 'end_lat']].isna().any().any():
        raise ValueError(f'{shard}: an unresolved gap end is not a cleaned GFW point')
    gp['gap_hours'] = (gp.gap_end_ns - gp.gap_start_ns) / NS_HOUR
    return gp[['candidate_id', 'gap_start_ns', 'gap_end_ns', 'gap_hours', 'gap_class', 'cells',
               'start_lat', 'start_lon', 'end_lat', 'end_lon']].sort_values(['candidate_id', 'gap_start_ns']).reset_index(drop=True)


def check_shard(shard):
    """The recovery must equal E006 step-4 records on the compared fields."""
    derived = gap_ends(shard, E006_TRACKS / shard)
    stored = pd.read_parquet(E006_TRACKS / shard / 'unresolved_segments.parquet', columns=COMPARED)
    stored = stored.astype({'candidate_id': str}).sort_values(['candidate_id', 'gap_start_ns']).reset_index(drop=True)
    pd.testing.assert_frame_equal(derived[COMPARED].astype({'gap_class': str}), stored.astype({'gap_class': str}),
                                  check_exact=True, check_dtype=False)
    return {'shard': shard, 'gaps': len(stored), 'exact': True}


def build_shard(shard, weather, out):
    source = weather / 'tracks' / shard
    target = out / 'shards' / shard
    target.mkdir(parents=True)
    for name in ['candidate_summary.parquet', 'ten_minute_complete_candidate.parquet']:
        (target / name).symlink_to((source / name).resolve())
    derived = gap_ends(shard, source)
    derived.to_parquet(target / 'unresolved_segments.parquet', index=False)
    return {'shard': shard, 'unresolved_gaps': len(derived), 'segments': core.record(target / 'unresolved_segments.parquet')}


def run(jobs, workers):
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(*job): job[1] for job in jobs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                failures.append({'shard': futures[future], 'error': traceback.format_exc()})
    return sorted(results, key=lambda r: r['shard']), failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weather', type=lambda v: Path(v).resolve(), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--check-shards', nargs='+', default=['shard_0005', 'shard_0009'])
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    out = core.inside_runs(args.out)
    out.mkdir(parents=True, exist_ok=False)
    state = {'state': 'running', 'weather': str(args.weather), 'clean_points': str(CLEAN), 'code': core.record(Path(__file__)),
             'adopted': False, 'network_requests': 0}
    try:
        weather = json.loads((args.weather / 'validation.json').read_text())
        if weather['status'] != 'pass':
            raise ValueError('composed weather run must pass')
        checks, failures = run([(check_shard, s) for s in args.check_shards], args.workers)
        state['recovery_reproduces_e006_step4'] = checks
        if failures or not all(c['exact'] for c in checks):
            raise ValueError(f'gap-end recovery differs from step-4 records: {failures}')
        built, failures = run([(build_shard, s, args.weather, out) for s in core.SHARDS], args.workers)
        if failures:
            state['failures'] = failures
            raise ValueError(f'{len(failures)} shards failed')
        state.update(state='pass', shards=built, unresolved_gaps=sum(r['unresolved_gaps'] for r in built))
        print(json.dumps({'state': 'pass', 'checked': checks, 'unresolved_gaps': state['unresolved_gaps']}), flush=True)
    except BaseException:
        state.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        core.write_json(out / 'execution.json', state)


if __name__ == '__main__':
    main()
