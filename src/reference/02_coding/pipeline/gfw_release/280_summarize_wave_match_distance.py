#!/usr/bin/env python3
"""A05: how far each voyage hour is from the ERA5 wave cell it was matched to.

The production matcher uses the nearest 0.5-degree ERA5 grid point when it has a
finite wave value, otherwise the nearest finite (wet) point within 117.94 km,
otherwise no wave value. All weights are voyage hours, not row counts.
    python 280_summarize_wave_match_distance.py --weather <run> --release <run> --out <run> [--review <outputs dir>]
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa
HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
sys.path.insert(0, str(CODING / 'pipeline/gfw_release'))
import step6_core as core
KEY = ['origin_city', 'dest_city', 'shiptype_group']
STEP_KM, CAP_KM = .01, 117.94
EDGES = np.arange(0, CAP_KM + 2 * STEP_KM, STEP_KM)
MATCHED = ['nearest_grid', 'nearest_finite_wet_cell']
BINS = {'nearest_grid': [0, 10, 20, 30, 40], 'nearest_finite_wet_cell': [0, 25, 50, 75, 100, CAP_KM]}


def shard_summary(shard, weather):
    pa.set_cpu_count(1)
    status, hist, maxima, per_voyage = [], {s: np.zeros(len(EDGES) - 1) for s in MATCHED}, {s: 0.0 for s in MATCHED}, []
    for path in sorted((Path(weather) / 'shards' / shard).glob('weather_*.parquet')):
        f = pd.read_parquet(path, columns=['candidate_id', 'segment_weight_hours', 'wave_match_status', 'wave_snap_distance_km', 'position_source'])
        f['position_source'] = f.position_source.astype(object).fillna('no_position')
        f['wave_match_status'] = f.wave_match_status.astype(str)
        status.append(f.groupby(['wave_match_status', 'position_source']).segment_weight_hours.agg(['sum', 'size']))
        for s in MATCHED:
            g = f.loc[f.wave_match_status.eq(s)]
            d = g.wave_snap_distance_km.to_numpy(float)
            if np.isnan(d).any() or (d > CAP_KM + 1e-8).any():
                raise ValueError(f'{path}: matched distance missing or beyond the cap')
            hist[s] += np.histogram(d, EDGES, weights=g.segment_weight_hours.to_numpy(float))[0]
            maxima[s] = max(maxima[s], float(d.max()) if len(d) else 0.0)
        f['matched_hours'] = f.segment_weight_hours.where(f.wave_match_status.isin(MATCHED), 0.0)
        f['snapped_hours'] = f.segment_weight_hours.where(f.wave_match_status.eq('nearest_finite_wet_cell'), 0.0)
        f['snapped_over50_hours'] = f.snapped_hours.where(f.wave_snap_distance_km.gt(50), 0.0)
        per_voyage.append(f.groupby('candidate_id')[['matched_hours', 'snapped_hours', 'snapped_over50_hours']].sum())
    status = pd.concat(status).groupby(level=[0, 1]).sum()
    per_voyage = pd.concat(per_voyage).groupby(level=0).sum()
    return {'shard': shard, 'status': status, 'hist': hist, 'max': maxima, 'per_voyage': per_voyage}


def quantile(hist, q):
    c = np.cumsum(hist)
    return float(EDGES[1:][np.searchsorted(c, q * c[-1])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['weather', 'release', 'out']:
        p.add_argument('--' + name, type=lambda v: Path(v).resolve(), required=True)
    p.add_argument('--review', type=lambda v: Path(v).resolve())
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    out = core.inside_runs(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((CODING / 'pipeline/40_construct_variables/60_construct_voyage_weather_facts/config/2025_ten_minute_weather.json').read_text())
    if config['maximum_wet_cell_snap_km'] != CAP_KM:
        raise ValueError('matcher snap cap differs from the frozen weather configuration')
    if json.loads((args.weather / 'validation.json').read_text())['status'] != 'pass':
        raise ValueError('weather run must pass')
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(shard_summary, s, str(args.weather)): s for s in core.SHARDS}
        for future in as_completed(jobs):
            try:
                results.append(future.result())
            except Exception:
                failures.append({'shard': jobs[future], 'error': traceback.format_exc()})
    if failures:
        core.write_json(out / 'failures.json', failures)
        raise SystemExit(1)
    status = pd.concat([r['status'] for r in results]).groupby(level=[0, 1]).sum().reset_index()
    status.columns = ['wave_match_status', 'position_source', 'hours', 'rows']
    total = status.hours.sum()
    annual = json.loads((args.weather / 'validation.json').read_text())['totals']['voyage_hours']
    if not np.isclose(total, annual, rtol=0, atol=1e-3):
        raise ValueError(f'hour closure failed: {total} vs {annual}')
    positioned = status.loc[~status.wave_match_status.eq('invalid_position'), 'hours'].sum()
    by_status = status.groupby('wave_match_status').hours.sum().rename('hours').reset_index()
    by_status['share_of_all_voyage_hours'] = by_status.hours / total
    by_status['share_of_positioned_hours'] = np.where(by_status.wave_match_status.eq('invalid_position'), np.nan, by_status.hours / positioned)
    hist = {s: sum(r['hist'][s] for r in results) for s in MATCHED}
    hist['all_matched'] = hist['nearest_grid'] + hist['nearest_finite_wet_cell']
    maxima = {s: max(r['max'][s] for r in results) for s in MATCHED}
    maxima['all_matched'] = max(maxima.values())
    for q, name in [(.5, 'p50'), (.9, 'p90'), (.99, 'p99')]:
        by_status[f'distance_km_{name}'] = by_status.wave_match_status.map({s: quantile(h, q) for s, h in hist.items()})
    by_status['distance_km_max'] = by_status.wave_match_status.map(maxima)
    matched_row = {'wave_match_status': 'all_matched', 'hours': hist['all_matched'].sum(),
                   'share_of_all_voyage_hours': hist['all_matched'].sum() / total, 'share_of_positioned_hours': hist['all_matched'].sum() / positioned,
                   **{f'distance_km_{n}': quantile(hist['all_matched'], q) for q, n in [(.5, 'p50'), (.9, 'p90'), (.99, 'p99')]},
                   'distance_km_max': maxima['all_matched']}
    by_status = pd.concat([by_status, pd.DataFrame([matched_row])], ignore_index=True)
    bins = []
    for s, edges in BINS.items():
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (EDGES[:-1] >= lo - 1e-9) & (EDGES[:-1] < hi - 1e-9)
            hours = hist[s][mask].sum()
            bins.append({'wave_match_status': s, 'distance_from_km': lo, 'distance_to_km': hi, 'hours': hours,
                         'share_of_status_hours': hours / hist[s].sum(), 'share_of_matched_hours': hours / hist['all_matched'].sum()})
    by_source = status.pivot_table(index='position_source', columns='wave_match_status', values='hours', aggfunc='sum', fill_value=0)
    by_source['hours'] = by_source.sum(axis=1)
    for s in by_source.columns.drop('hours'):
        by_source[f'share_{s}'] = by_source[s] / by_source.hours
    voyages = pd.concat([r['per_voyage'] for r in results]).groupby(level=0).sum().rename_axis('candidate_id').reset_index()
    intervals = pd.read_parquet(args.release / 'assembly/voyage_intervals.parquet', columns=['candidate_id', *KEY])
    released = pd.read_parquet(args.release / 'assembly/release_full_precision.parquet', columns=KEY)
    rec = intervals.merge(voyages, on='candidate_id', how='left', validate='one_to_one')
    if rec.matched_hours.isna().any():
        raise ValueError('a released voyage has no weather rows')
    rec = rec.groupby(KEY, observed=True)[['matched_hours', 'snapped_hours', 'snapped_over50_hours']].sum().reset_index().merge(released, on=KEY)
    if len(rec) != len(released):
        raise ValueError('record closure failed')
    rec['snapped_share'] = rec.snapped_hours / rec.matched_hours.where(rec.matched_hours > 0)
    rec['snapped_over50_share'] = rec.snapped_over50_hours / rec.matched_hours.where(rec.matched_hours > 0)
    records = pd.DataFrame([{'threshold_share_of_matched_hours': t, 'records_snapped_share_above': int(rec.snapped_share.gt(t).sum()),
                             'records_snapped_over50km_share_above': int(rec.snapped_over50_share.gt(t).sum()),
                             'records_with_matched_hours': int(rec.matched_hours.gt(0).sum())} for t in [0, .01, .05, .1, .25, .5]])
    files = {'16_wave_match_distance_by_status.csv': by_status, '17_wave_match_distance_bins.csv': pd.DataFrame(bins),
             '18_wave_match_by_position_source.csv': by_source.reset_index(), '19_records_by_snapped_wave_share.csv': records}
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)
    rec.to_parquet(out / 'record_snapped_shares.parquet', index=False)
    summary = {'status': 'pass', 'voyage_hours': float(total), 'positioned_hours': float(positioned), 'snap_cap_km': CAP_KM,
               'grid': 'ERA5 0.5-degree global wave field, frozen 2025 snapshot', 'weights': 'voyage hours (segment_weight_hours)',
               'weather': core.record(args.weather / 'validation.json'), 'release': core.record(args.release / 'validation.json'),
               'code': core.record(Path(__file__)), 'outputs': [core.record(out / n) for n in files], 'adopted_population': True,
               'network_requests': 0}
    core.write_json(out / 'summary.json', summary)
    if args.review:
        if not args.review.is_relative_to((CODING / 'outputs').resolve()):
            raise ValueError('review copies belong in 02_coding/outputs')
        for name in files:
            shutil.copy2(out / name, args.review / name)
    print(by_status.round(4).to_string())
    print(pd.DataFrame(bins).round(4).to_string())
    print(records.to_string())


if __name__ == '__main__':
    main()
