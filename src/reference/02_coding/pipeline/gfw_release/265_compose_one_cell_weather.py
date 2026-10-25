#!/usr/bin/env python3
"""Compose weather for the one-cell candidate set without recomputing kept voyages.

Previously retained voyages keep the verified corrected-segmentation weather.
Newly retained voyages reuse an exact Option C physical twin or are rebuilt
with the unchanged track and weather adapters used by 220.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
E006 = CODING / 'pipeline/gfw_release'
sys.path.insert(0, str(E006))
import step6_core as core
W = core.module('e007_one_cell_weather_adapter', HERE / '220_rebuild_segmented_weather.py')
REVISION = HERE / 'config/short_stop_allowance_revision.json'
PREVIOUS = 'previous_segmented'
COMPARED = ['old_candidate_id', 'weather_source', 'source_shard', 'shiptype_group', 'vessel_uid_clean',
            'origin_city', 'dest_city', 'departure_time', 'arrival_time']


def prepare(base, previous_base, out, shards):
    validation = json.loads((base / 'one_cell_validation.json').read_text())
    if validation['status'] != 'pass' or Path(json.loads((base / 'execution.json').read_text())['previous_base']) != previous_base:
        raise ValueError('one-cell candidate base must pass against this previous base')
    panel = pd.read_parquet(base / 'retained_panel.parquet')
    previous = pd.read_parquet(previous_base / 'retained_panel.parquet')
    joined = previous[['candidate_id', *COMPARED]].merge(panel[['candidate_id', *COMPARED]], on='candidate_id',
                                                         how='left', suffixes=('', '_new'), validate='one_to_one')
    for column in COMPARED:
        if not joined[column].astype(str).eq(joined[column + '_new'].astype(str)).all():
            raise ValueError(f'previously retained voyage changed: {column}')
    panel['composition_source'] = np.where(panel.candidate_id.isin(previous.candidate_id), PREVIOUS, panel.weather_source)
    rebuild = panel.composition_source.eq('rebuild_gfw')
    panel['expected_grid_rows'] = 0
    panel.loc[rebuild, 'expected_grid_rows'] = [len(W.TRIM.tracks.strict_grid_ns(a, b)) for a, b in
                                                zip(panel.loc[rebuild, 'departure_time'], panel.loc[rebuild, 'arrival_time'])]
    folder = out / 'prepared'
    folder.mkdir()
    panel.to_parquet(folder / 'composition_panel.parquet', index=False)
    tasks = []
    for shard in shards:
        f = panel.loc[panel.source_shard.eq(shard)].sort_values('candidate_id')
        path = folder / f'{shard}.parquet'
        f.to_parquet(path, index=False)
        changed = f.loc[f.composition_source.eq('rebuild_gfw')]
        cpath = folder / f'{shard}_changed.parquet'
        changed[[*W.TRIM.tracks.CANDIDATE_COLUMNS, 'expected_grid_rows']].to_parquet(cpath, index=False)
        clean = W.TRIM.tracks.CLEAN_ROOT / 'shards' / shard / 'mask' / f'{shard}.feather'
        receipt = json.loads((clean.parent / 'execution.json').read_text())
        tasks.append({'shard': shard, 'panel': core.record(path), 'candidates': core.record(cpath),
                      'points': {'path': str(clean), 'bytes': clean.stat().st_size, 'sha256': receipt['outputs'][str(clean)]},
                      'output': str(out / 'rebuilt_tracks' / shard), 'weather': str(out / 'rebuilt_weather' / shard),
                      'candidate_rows': len(changed), 'expected_grid_rows': int(changed.expected_grid_rows.sum()), 'out': str(out)})
    return tasks, panel.composition_source.value_counts().to_dict()


def worker(task, previous_weather):
    pa.set_cpu_count(1)
    started = time.monotonic()
    shard, out, previous_weather = task['shard'], Path(task['out']), Path(previous_weather)
    panel = pd.read_parquet(task['panel']['path'])
    prev_state = json.loads((previous_weather / 'shards' / shard / 'status.json').read_text())
    old_state = json.loads((W.OLD / 'shards' / shard / 'status.json').read_text())
    for item in [task['panel'], *prev_state['outputs'].values(), prev_state['sources']['track'], prev_state['sources']['summary'],
                 *old_state['outputs'].values(), old_state['sources']['track'], old_state['sources']['summary']]:
        if core.record(Path(item['path'])) != item:
            raise ValueError(f'bound source changed: {item["path"]}')
    clearance = {'experiment': 'E007', 'authorization': core.record(REVISION), 'segmentation_rule': core.record(W.RULE),
                 'scope': 'authorized one-cell short-stop allowance downstream rebuild', 'rules': W.TRIM.tracks.RULES, 'reference_ais_inputs': False}
    if task['candidate_rows']:
        W.TRIM.tracks.work(task)
        W.TRIM.weather(task, clearance)
    kept = panel.loc[panel.composition_source.eq(PREVIOUS), 'candidate_id'].tolist()
    twins = panel.loc[panel.composition_source.eq('reuse_verified_option_c')]
    old_ids, new_ids = twins.old_candidate_id.tolist(), twins.candidate_id.tolist()
    weather, tracks = out / 'shards' / shard, out / 'tracks' / shard
    weather.mkdir(parents=True)
    tracks.mkdir(parents=True)

    def combine(previous_source, option_c_source, changed_source, destination, sort=None):
        table = pq.read_table(previous_source)
        inside = pc.is_in(table['candidate_id'], value_set=pa.array(kept, pa.string()))
        if not pc.all(inside).as_py():
            raise ValueError(f'previous component holds a voyage that is no longer retained: {previous_source}')
        parts = [table, W.remap(pq.read_table(option_c_source), old_ids, new_ids).cast(table.schema)]
        if task['candidate_rows'] and changed_source.exists():
            parts.append(pq.read_table(changed_source).cast(table.schema))
        elif task['candidate_rows']:
            if not changed_source.name.startswith('weather_'):
                raise FileNotFoundError(changed_source)
            start = pd.Timestamp(changed_source.stem.removeprefix('weather_') + '-01', tz='UTC')
            end = start + pd.offsets.MonthBegin(1)
            if (panel.composition_source.eq('rebuild_gfw') & panel.departure_time.lt(end) & panel.arrival_time.gt(start)).any():
                raise ValueError('missing month containing a rebuilt voyage')
        result = pa.concat_tables(parts)
        if sort:
            result = result.sort_by([(c, 'ascending') for c in sort])
        pq.write_table(result, destination, compression='zstd')
        return result.num_rows

    for kind, name in W.NAMES.items():
        combine(Path(prev_state['sources'][kind]['path']), Path(old_state['sources'][kind]['path']), Path(task['output']) / name, tracks / name)
    rows = sum(combine(previous_weather / 'shards' / shard / name, W.OLD / 'shards' / shard / name, Path(task['weather']) / name,
                       weather / name, ['candidate_id', 'hour_anchor_utc']) for name in W.MONTHS)
    facts_name = 'voyage_weather_facts.parquet'
    combine(previous_weather / 'shards' / shard / facts_name, W.OLD / 'shards' / shard / facts_name, Path(task['weather']) / facts_name,
            weather / facts_name, ['candidate_id'])
    facts = pd.read_parquet(weather / facts_name)
    if set(facts.candidate_id) != set(panel.candidate_id) or facts.candidate_id.duplicated().any():
        raise ValueError('composed candidate set does not close')
    ordered = lambda f: f.sort_values('candidate_id').reset_index(drop=True)
    previous_facts = pd.read_parquet(previous_weather / 'shards' / shard / facts_name)
    if set(previous_facts.candidate_id) != set(kept):
        raise ValueError('previous weather does not cover exactly the previously retained voyages')
    pd.testing.assert_frame_equal(ordered(previous_facts), ordered(facts.loc[facts.candidate_id.isin(kept)]), check_exact=True)
    original = pd.read_parquet(W.OLD / 'shards' / shard / facts_name)
    original = original.loc[original.candidate_id.isin(old_ids)].copy()
    original['candidate_id'] = original.candidate_id.map(dict(zip(old_ids, new_ids)))
    pd.testing.assert_frame_equal(ordered(original), ordered(facts.loc[facts.candidate_id.isin(new_ids)]), check_exact=True)
    joined = panel[['candidate_id', 'departure_time', 'arrival_time']].merge(facts[['candidate_id', 'voyage_hours']], on='candidate_id', validate='one_to_one')
    if not np.allclose((joined.arrival_time - joined.departure_time).dt.total_seconds() / 3600, joined.voyage_hours, rtol=0, atol=1e-8):
        raise ValueError('physical voyage duration changed')
    state = {**{k: v for k, v in prev_state.items() if k not in ['outputs', 'sources', 'source_option_c']},
             'population': 'E007_gfw_one_cell_short_stop_2025', 'candidates': len(panel), 'rows': rows, 'review_clearance': clearance,
             'sources': {**prev_state['sources'], 'track': core.record(tracks / W.NAMES['track']), 'summary': core.record(tracks / W.NAMES['summary'])},
             'segmentation_rule': core.record(W.RULE), 'short_stop_revision': core.record(REVISION),
             'previous_segmented_voyages': len(kept), 'reused_exact_physical_voyages': len(twins), 'rebuilt_voyages': task['candidate_rows'],
             'source_previous_segmented': core.record(previous_weather / 'shards' / shard / 'status.json'),
             'source_option_c': core.record(W.OLD / 'shards' / shard / 'status.json'),
             'outputs': {p.name: core.record(p) for p in sorted(weather.glob('*.parquet'))}}
    core.write_json(weather / 'status.json', state)
    check = W.TRIM.weather_check.validate_shard(weather)
    if check['status'] != 'pass' or not all(check['checks'].values()):
        raise ValueError('composed weather validation failed')
    result = {'shard': shard, 'candidates': len(panel), 'rows': rows, 'previous': len(kept), 'reused': len(twins),
              'rebuilt': task['candidate_rows'], 'reused_facts_exact': True, 'validation': 'pass', 'seconds': time.monotonic() - started}
    core.write_json(weather / 'composition_validation.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['base', 'previous_base', 'previous_weather', 'out']:
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=lambda v: Path(v).resolve(), required=True)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--shards', nargs='*')
    args = p.parse_args()
    out = core.inside_runs(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shards = sorted(args.shards or core.SHARDS)
    if not set(shards) <= set(core.SHARDS) or len(shards) != len(set(shards)):
        raise ValueError('invalid shards')
    previous = json.loads((args.previous_weather / 'execution.json').read_text())
    if previous['state'] != 'completed' or Path(previous['base']) != args.previous_base:
        raise ValueError('previous weather must be the completed run of the previous base')
    code = [core.record(x) for x in [Path(__file__), HERE / '220_rebuild_segmented_weather.py', HERE / 'track_weather_adapter.py', REVISION]]
    state = {'state': 'running', 'complete': shards == core.SHARDS, 'workers': args.workers, 'base': str(args.base),
             'previous_base': str(args.previous_base), 'previous_weather': str(args.previous_weather),
             'previous_validation': core.record(args.previous_weather / 'validation.json'), 'rule': core.record(W.RULE),
             'revision': core.record(REVISION), 'code': code, 'results': [], 'failures': [], 'adopted': False, 'network_requests': 0}
    core.write_json(out / 'execution.json', state)
    start = time.monotonic()
    tasks, sources = prepare(args.base, args.previous_base, out, shards)
    state['composition_sources'] = sources
    results, failures = [], []
    # Largest rebuilds first keeps the second wave short.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(worker, t, args.previous_weather): t['shard'] for t in sorted(tasks, key=lambda t: -t['candidate_rows'])}
        for future in as_completed(jobs):
            try:
                r = future.result()
                results.append(r)
                print(json.dumps({'done': len(results), **r}), flush=True)
            except Exception:
                failures.append({'shard': jobs[future], 'error': traceback.format_exc()})
                print('FAILED ' + jobs[future], flush=True)
            state.update(results=sorted(results, key=lambda r: r['shard']), failures=failures, elapsed_seconds=time.monotonic() - start)
            core.write_json(out / 'execution.json', state)
    if not failures and state['complete']:
        state['annual'] = W.finalize(args.base, out, results)
        decision = json.loads((out / 'step6_decision.json').read_text())
        decision.update(voyage_set='E007 corrected segmentation, one-cell short-stop allowance, unresolved cases quarantined',
                        short_stop_revision=core.record(REVISION))
        core.write_json(out / 'step6_decision.json', decision)
    for item in code:
        if core.record(Path(item['path'])) != item:
            raise ValueError('weather composition code changed during execution')
    state.update(state='failed' if failures else 'completed', wall_seconds=time.monotonic() - start)
    core.write_json(out / 'execution.json', state)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
