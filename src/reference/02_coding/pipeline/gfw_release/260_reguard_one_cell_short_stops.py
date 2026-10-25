#!/usr/bin/env python3
"""Re-guard unchanged candidates after the one-cell short-stop allowance.

The corrected port run may differ from the previous one only by clearing
possible_short_stop flags. Every other visit field and every raw leg must be
identical, so the previous pre-guard candidates are reused and the existing
guard/bridge code reruns (config/short_stop_allowance_revision.json).
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import pandas as pd
import pyarrow as pa
HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
E006 = CODING / 'pipeline/gfw_release'
sys.path.insert(0, str(E006))
import step6_core as core
BASE = core.module('e007_one_cell_guard_base', HERE / '210_rebuild_segmented_voyage_base.py')
REVISION = HERE / 'config/short_stop_allowance_revision.json'


def compare_shard(shard, ports, previous):
    """Port outputs may differ only by clearing possible_short_stop."""
    pa.set_cpu_count(1)
    new = pd.read_parquet(ports / 'shards' / shard / 'visit_quality.parquet')
    old = pd.read_parquet(previous / 'shards' / shard / 'visit_quality.parquet')
    if list(new.columns) != list(old.columns) or len(new) != len(old):
        raise ValueError(f'{shard}: visit table shape changed')
    other = [c for c in new.columns if c != 'possible_short_stop']
    pd.testing.assert_frame_equal(new[other], old[other], check_exact=True)
    if (new.possible_short_stop & ~old.possible_short_stop).any():
        raise ValueError(f'{shard}: one-cell flag outside the previous two-cell flags')
    for folder, name in [('shard_port_visits', f'{shard}_port_visits.feather'),
                         ('shard_voyages', f'{shard}_harmonized_od_voyages.feather')]:
        a = pd.read_feather(ports / 'shards' / shard / folder / name)
        b = pd.read_feather(previous / 'shards' / shard / folder / name)
        keep = [c for c in a.columns if c != 'possible_short_stop']
        pd.testing.assert_frame_equal(a[keep], b[keep], check_exact=True)
    return {'shard': shard, 'visits': len(new), 'flags_before': int(old.possible_short_stop.sum()),
            'flags_after': int(new.possible_short_stop.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['ports', 'previous_ports', 'previous_base', 'out']:
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=lambda v: Path(v).resolve(), required=True)
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    root = core.inside_runs(args.out)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    code = [core.record(x) for x in [Path(__file__), HERE / '210_rebuild_segmented_voyage_base.py', HERE / 'port_segmentation.py',
                                     HERE / 'config/port_segmentation_revision.json', REVISION, E006 / 'step6_core.py']]
    state = {'status': 'running', 'experiment': 'E007', 'revision': core.record(REVISION), 'source_port_root': str(args.ports),
             'previous_port_root': str(args.previous_ports), 'previous_base': str(args.previous_base),
             'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'workers': args.workers, 'adopted': False, 'network_requests': 0}
    core.write_json(root / 'execution.json', state)
    try:
        port_state = json.loads((args.ports / 'execution.json').read_text())
        if port_state['status'] != 'pass' or not port_state['complete_year'] or not all(port_state['closure'].values()):
            raise ValueError('corrected port run must be a passed complete year')
        if core.record(HERE / 'port_segmentation.py') not in port_state['code']:
            raise ValueError('port run used another segmentation implementation')
        results, failures = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(compare_shard, s, args.ports, args.previous_ports): s for s in core.SHARDS}
            for future in as_completed(jobs):
                try:
                    results.append(future.result())
                except Exception:
                    failures.append({'shard': jobs[future], 'error': traceback.format_exc()})
        if failures:
            state['port_comparison_failures'] = failures
            raise ValueError(f'{len(failures)} shards differ beyond the short-stop flag')
        flags = {'before': sum(r['flags_before'] for r in results), 'after': sum(r['flags_after'] for r in results)}
        state['port_comparison'] = {'shards': len(results), 'only_short_stop_flags_cleared': True, 'flags': flags}
        core.write_json(root / 'execution.json', state)
        # Reuse the unchanged construction and record the guard code before guarding.
        pretrim = args.previous_base / 'pretrim_panel.parquet'
        reused = {'pretrim': core.record(pretrim), 'input_view': str((args.previous_base / 'input_view').resolve()),
                  'construction_record': core.record(args.previous_base / 'implementation_before_candidate_guard.json')}
        (root / 'pretrim_panel.parquet').symlink_to(pretrim)
        (root / 'input_view').symlink_to((args.previous_base / 'input_view').resolve(), target_is_directory=True)
        core.write_json(root / 'implementation_before_candidate_guard.json',
                        {'stage_when_recorded': 'before_one_cell_guard', 'code': code, 'reused_unchanged_construction': reused,
                         'source_commit': state['source_commit']})
        summary = BASE.guard_and_bridge(root, args.ports, args.workers)
        new = pd.read_parquet(root / 'candidate_decisions.parquet', columns=['candidate_id', 'speed_decision', 'segmentation_review_reasons', 'retained'])
        old = pd.read_parquet(args.previous_base / 'candidate_decisions.parquet', columns=['candidate_id', 'speed_decision', 'segmentation_review_reasons', 'retained'])
        both = old.merge(new, on='candidate_id', how='outer', suffixes=('_old', '_new'), validate='one_to_one', indicator=True)
        split = lambda s: set(filter(None, s.split(';')))
        removed = [split(a) - split(b) for a, b in zip(both.segmentation_review_reasons_old.fillna(''), both.segmentation_review_reasons_new.fillna(''))]
        added = [split(b) - split(a) for a, b in zip(both.segmentation_review_reasons_old.fillna(''), both.segmentation_review_reasons_new.fillna(''))]
        new_bridge = pd.read_parquet(root / 'physical_voyage_bridge.parquet')
        old_bridge = pd.read_parquet(args.previous_base / 'physical_voyage_bridge.parquet')
        kept = new_bridge.merge(old_bridge, on='candidate_id', how='inner', suffixes=('', '_old'))
        checks = {'same_candidates': bool(both._merge.eq('both').all()),
                  'same_speed_decisions': bool(both.speed_decision_old.eq(both.speed_decision_new).all()),
                  'no_reason_added': not any(added),
                  'only_short_stop_reasons_removed': all(r <= {'unresolved_intermediate_short_stop'} for r in removed),
                  'previous_retained_still_retained': bool((~both.retained_old | both.retained_new).all()),
                  'previous_bridge_rows_identical': len(kept) == len(old_bridge) and all(
                      kept[c].astype(str).eq(kept[c + '_old'].astype(str)).all()
                      for c in ['old_candidate_id', 'weather_source', 'source_shard', *BASE.NATURAL_KEY])}
        validation = {'status': 'pass' if all(checks.values()) else 'failed', 'checks': checks, 'port_flags': flags,
                      'candidates': len(both), 'retained_before': int(both.retained_old.sum()), 'retained_after': int(both.retained_new.sum()),
                      'newly_retained': int((both.retained_new & ~both.retained_old).sum()),
                      'short_stop_reason_cleared': int(sum('unresolved_intermediate_short_stop' in r for r in removed)),
                      'revision': core.record(REVISION), 'code': code, 'adopted': False, 'network_requests': 0}
        core.write_json(root / 'one_cell_validation.json', validation)
        for item in code:
            if core.record(Path(item['path'])) != item:
                raise ValueError('guard implementation changed during execution')
        if validation['status'] != 'pass':
            raise ValueError(f'one-cell candidate checks failed: {checks}')
        state.update(status='pass', summary=summary, one_cell_validation=core.record(root / 'one_cell_validation.json'))
        print(json.dumps({k: validation[k] for k in ['status', 'port_flags', 'retained_before', 'retained_after', 'newly_retained']}), flush=True)
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        state.update(wall_seconds=time.monotonic() - started)
        core.write_json(root / 'execution.json', state)


if __name__ == '__main__':
    main()
