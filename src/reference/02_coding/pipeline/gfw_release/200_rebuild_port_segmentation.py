#!/usr/bin/env python3
"""Replay corrected E007 port visits using the existing E006 shard machinery."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import pandas as pd
import pyarrow as pa

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
sys.path.insert(0, str(CODING / 'pipeline/gfw_release'))
import step6_core as core
from port_segmentation import visit_builder, nearest_unique, EXTRA_COLUMNS
E006 = CODING / 'pipeline/gfw_release'
CLEAN = CODING / '.runs/2026-10-05_2305_E006-gfw-cleaning'
CORRECTIONS = E006 / 'config/t034_anchorage_city_corrections.csv'
RULE = HERE / 'config/port_segmentation_revision.json'
BASE = CODING / '.runs/2026-10-06_1650_E006-gfw-port-calls'


def worker(task):
    pa.set_cpu_count(1)
    runner = core.module('e007_port_runner', E006 / '120_run_gfw_port_calls.py')
    original_load = runner.load_module
    def revised_load(name, path):
        stage = original_load(name, path)
        if name == 'e006_ports':
            catalog, _ = stage.load_anchorages(Path(task['catalog']))
            corrections = pd.read_csv(CORRECTIONS, dtype={'anchorage_id': str}).set_index('anchorage_id').new_analytical_city_id
            catalog['city'] = catalog.s2id.astype(str).map(corrections).fillna(catalog.analytical_city_id)
            stage.build_vessel_visits = visit_builder(catalog.city.to_numpy())
            stage.query_nearest_anchorage = nearest_unique
            stage.VISIT_OUTPUT_COLUMNS = [*stage.VISIT_OUTPUT_COLUMNS, *EXTRA_COLUMNS]
        return stage
    runner.load_module = revised_load
    result = runner.process_one(task)
    root = Path(task['output'])
    v = pd.read_feather(result['visits_output_path'])
    old = pd.read_feather(BASE / 'shards' / task['shard'] / 'shard_port_visits' / f"{task['shard']}_port_visits.feather")
    # The original city labels are preserved in the standard columns; add
    # corrected keys in the QC table used by the downstream candidate guard.
    corrections = pd.read_csv(CORRECTIONS, dtype={'anchorage_id': str}).set_index('anchorage_id').new_analytical_city_id
    v['analytical_city_id'] = v.anchorage_id.astype(str).map(corrections).fillna(v.analytical_city_id)
    v.to_parquet(root / 'visit_quality.parquet', index=False)
    counts = {'old_visits': len(old), 'old_high': int(old.visit_confidence.eq('high').sum()),
              'new_visits': len(v), 'new_high': int(v.visit_confidence.eq('high').sum()),
              'annulus_entries': int(v.entry_rule.eq('stop_supported_annulus').sum()),
              'possible_short_stops': int(v.possible_short_stop.sum()),
              'ambiguous_cities': int(v.ambiguous_city.sum()),
              'entry_brackets_at_least4h': int(v.entry_gap_hours.ge(4).sum()),
              'exit_brackets_at_least4h': int(v.exit_gap_hours.ge(4).sum())}
    result['revision_counts'] = counts
    result['visit_quality'] = core.record(root / 'visit_quality.parquet')
    result['revision_code'] = [core.record(HERE / 'port_segmentation.py'), core.record(Path(__file__)), core.record(RULE), core.record(CORRECTIONS)]
    core.write_json(root / 'execution.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shards', nargs='*')
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    root = core.inside_runs(args.out)
    root.mkdir(parents=True, exist_ok=False)
    runner = core.module('e007_port_inputs', E006 / '120_run_gfw_port_calls.py')
    for path, digest in runner.EXPECTED.items():
        if runner.sha(path) != digest: raise ValueError(f'frozen input changed: {path}')
    shards = sorted(args.shards or runner.SHARDS)
    if len(shards) != len(set(shards)) or not set(shards) <= set(runner.SHARDS): raise ValueError('invalid shards')
    if args.workers < 1: raise ValueError('workers must be positive')
    sources = runner.step2_sources(CLEAN)
    specs = root / 'analysis_vessel_specs_2025.csv'
    runner.write_analysis_specs(specs)
    catalog_receipt, specs_receipt = core.record(runner.CATALOG), core.record(specs)
    tasks = [{'shard': s, 'source': sources[s], 'specs': str(specs), 'specs_record': specs_receipt,
              'catalog': str(runner.CATALOG), 'catalog_record': catalog_receipt,
              'output': str(root / 'shards' / s)} for s in shards]
    results, failures = [], []
    started = time.monotonic()
    state = {'step': 'E007 corrected port segmentation', 'status': 'running', 'year': 2025,
             'complete_year': shards == runner.SHARDS, 'workers': min(args.workers, len(tasks)),
             'clean_root': str(CLEAN), 'catalog': catalog_receipt, 'specs': specs_receipt,
             'port_parameters': runner.PORT_PARAMETERS, 'rule': core.record(RULE),
             'code': [core.record(Path(__file__)), core.record(HERE/'port_segmentation.py')],
             'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'adopted': False, 'network_requests': 0, 'results': [], 'failures': []}
    core.write_json(root / 'execution.json', state)
    with ProcessPoolExecutor(max_workers=state['workers']) as pool:
        jobs = {pool.submit(worker, task): task['shard'] for task in tasks}
        for f in as_completed(jobs):
            try:
                r = f.result(); results.append(r)
                print(json.dumps({'done':len(results),'shard':r['shard'],'seconds':round(r['wall_seconds'],1),
                                  'rss_gb':round(r['peak_rss_kib']/1024**2,2), **r['revision_counts']}), flush=True)
            except Exception:
                failures.append({'shard':jobs[f], 'error':traceback.format_exc()})
                print('FAILED '+jobs[f], flush=True)
            state.update(results=sorted(results,key=lambda r:r['shard']), failures=failures,
                         elapsed_seconds=time.monotonic()-started)
            core.write_json(root / 'execution.json', state)
    totals = {k:sum(r[k] for r in results) for k in ['n_input_rows','n_unique_vessels','n_visits','n_high_confidence_visits','n_voyages']}
    if state['complete_year'] and not failures:
        vessel_ids = pd.concat([pd.read_csv(r['vessel_input_disposition']['path'], dtype={'vessel_uid':str}) for r in results])
        state['closure'] = {'all_64_shards':len(results)==64, 'cleaned_rows_close_to_step2':totals['n_input_rows']==runner.STEP2_KEPT_ROWS,
                            'vessels_close_to_step2':len(vessel_ids)==runner.STEP2_VESSELS==totals['n_unique_vessels'],
                            'each_vessel_in_one_shard':not vessel_ids.vessel_uid.duplicated().any()}
        if not all(state['closure'].values()): failures.append({'shard':'all','error':'denominator closure failed'})
    for item in [*state['code'], state['rule'], state['catalog']]:
        if core.record(Path(item['path'])) != item: raise ValueError('source changed during execution')
    state.update(status='failed' if failures else 'pass', totals=totals, failures=failures,
                 wall_seconds=time.monotonic()-started)
    core.write_json(root / 'execution.json', state)
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
