#!/usr/bin/env python3
"""Package the V1/V2 and interval checks rerun on the one-cell voyage set.

Copies the summary tables of the passed reruns, and sets their headline
numbers beside the same numbers from the runs on earlier voyage sets.
    python 275_package_one_cell_validation.py --gaps <20 run> --weather <30 run> --segmentation <60 run>
        --intervals <130 evaluate run> --out 02_coding/outputs/<YYYY-MM-DD_HHMM>_E007-...
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
RUNS = CODING / '.runs'
sys.path.insert(0, str(CODING / 'pipeline/gfw_release'))
import step6_core as core
BEFORE = {'gaps': RUNS / '2026-10-08_2235_E007-real-gap-comparison', 'weather': RUNS / '2026-10-08_2302_E007-real-gap-weather',
          'segmentation': RUNS / '2026-10-08_2308_E007-voyage-segmentation', 'intervals': RUNS / '2026-10-11_1219_E007-wave-support-masked'}
TABLES = {'01_positions_by_real_gap_length.csv': ('gaps', 'summary/positions_by_gap_length.csv'),
          '02_holdout_vs_real_gaps.csv': ('gaps', 'summary/holdout_vs_real_gaps.csv'),
          '03_gfw_voyage_time_by_gap_length.csv': ('gaps', 'summary/gfw_voyage_time_by_gap_length.csv'),
          '04_reachable_area_containment.csv': ('gaps', 'summary/reachable_area_containment_with_time_tolerance.csv'),
          '05_hourly_weather_agreement_by_gap_length.csv': ('weather', 'summary/hourly_agreement_by_gap_length.csv'),
          '06_hourly_weather_agreement_near_cyclones.csv': ('weather', 'summary/hourly_agreement_within_800km_of_a_cyclone.csv'),
          '07_record_exposure_true_vs_gfw_positions.csv': ('weather', 'summary/record_exposure_true_vs_gfw_positions.csv'),
          '08_voyage_segmentation_shares.csv': ('segmentation', 'segmentation_shares.csv'),
          '09_voyage_segmentation_by_endpoint_cells.csv': ('segmentation', 'segmentation_by_endpoint_cells.csv'),
          '10_most_missed_intermediate_cities.csv': ('segmentation', 'most_missed_intermediate_cities.csv'),
          '11_record_exposure_same_segmentation_only.csv': ('segmentation', 'record_exposure_same_segmentation_only.csv'),
          '12_masked_interval_inclusion.csv': ('intervals', 'inclusion_summary.csv'),
          '13_masked_interval_widths.csv': ('intervals', 'interval_width_summary.csv')}


def headline(roots):
    """One row per headline number, read from the summary tables of one set of runs."""
    rows = []
    add = lambda group, metric, value, unit: rows.append({'group': group, 'metric': metric, 'value': float(value), 'unit': unit})
    pos = pd.read_csv(roots['gaps'] / 'summary/positions_by_gap_length.csv')
    pos = pos.loc[pos.subset.eq('all')].set_index('length')
    for length in ['1 h', '13-24 h', '25-48 h']:
        add('positions', f'error_p50_{length}', pos.loc[length, 'error_km_p50'], 'km')
        add('positions', f'error_p90_{length}', pos.loc[length, 'error_km_p90'], 'km')
    contain = pd.read_csv(roots['gaps'] / 'summary/reachable_area_containment_with_time_tolerance.csv')
    add('reachable_area', 'contained_share', np.average(contain.contained, weights=contain.points), 'fraction')
    add('reachable_area', 'contained_share_with_30min_tolerance', np.average(contain.contained_with_tolerance, weights=contain.points), 'fraction')
    hourly = pd.read_csv(roots['weather'] / 'summary/hourly_agreement_by_gap_length.csv').set_index('length').loc['all']
    add('hourly_weather', 'paired_hours', hourly.hours, 'hours')
    add('hourly_weather', 'wave_exactly_equal_share', hourly.wave_exactly_equal_share, 'fraction')
    for name in ['ge4m', 'tc300', 'tc500']:
        add('hourly_weather', f'{name}_agree_share_among_either', hourly[f'{name}_agree_share_among_either'], 'fraction')
        add('hourly_weather', f'{name}_false_alarm_hours', hourly[f'{name}_false_alarm_hours'], 'hours')
        add('hourly_weather', f'{name}_miss_hours', hourly[f'{name}_miss_hours'], 'hours')
    record = pd.read_csv(roots['weather'] / 'summary/record_exposure_true_vs_gfw_positions.csv')
    add('record_exposure', 'records_compared', len(record), 'records')
    add('record_exposure', 'ge4m_share_abs_diff_p99', 100 * record.ge4m_share_diff.abs().quantile(.99), 'percentage_points')
    for radius in ['tc300', 'tc500']:
        add('record_exposure', f'{radius}_abs_diff_p99', record[f'{radius}_pct_diff'].abs().quantile(.99), 'percentage_points')
    shares = pd.read_csv(roots['segmentation'] / 'segmentation_shares.csv')
    shares = shares.loc[shares.scope.eq('released')].set_index('segmentation')
    add('segmentation', 'released_voyages', shares.voyages.sum(), 'voyages')
    for kind in ['same', 'reference_ais_intermediate_calls', 'same_times_other_city', 'inside_one_reference_ais_voyage']:
        add('segmentation', f'released_share_{kind}', shares.share_of_voyages.get(kind, 0.0), 'fraction')
    seg = json.loads((roots['segmentation'] / 'execution.json').read_text())
    add('segmentation', 'same_only_records_compared', seg['records_compared_same_only_ge5'], 'records')
    for metric, share in seg['share_records_diff_exceeds_ci_half_width'].items():
        add('segmentation', f'same_only_{metric}_exceeds_ci_half_width_share', share, 'fraction')
    intervals = pd.read_csv(roots['intervals'] / 'inclusion_summary.csv')
    intervals = intervals.loc[intervals.sample_set.isna() & intervals.reference.eq('gfw') & intervals.comparison.eq('fixed_known')].iloc[0]
    add('masked_intervals', 'windows', intervals.windows, 'windows')
    for name in ['wave', 'tc300', 'tc500']:
        add('masked_intervals', f'{name}_inside', intervals[f'{name}_inside_windows'], 'windows')
        add('masked_intervals', f'{name}_evaluable', intervals[f'{name}_evaluable_windows'], 'windows')
    return pd.DataFrame(rows)


def endpoint_entry_rules(segmentation, base, ports):
    """Segmentation classes by whether an endpoint visit opened as a 3-4 km stop.

    Dense-AIS voyages follow the earlier port rules, which have no stop-supported
    annulus entry; such endpoints can appear as extra GFW cuts.
    """
    seg = pd.read_parquet(segmentation / 'voyage_segmentation.parquet', columns=['candidate_id', 'released', 'segmentation'])
    panel = pd.read_parquet(base / 'retained_panel.parquet', columns=['candidate_id', 'vessel_uid_clean', 'departure_time', 'arrival_time'])
    f = panel.merge(seg, on='candidate_id', validate='one_to_one')
    f = f.loc[f.released].copy()
    visits = pd.concat([pd.read_parquet(x, columns=['vessel_uid', 'entry_time', 'exit_time', 'visit_confidence', 'entry_rule'])
                        for x in sorted((ports / 'shards').glob('*/visit_quality.parquet'))])
    visits = visits.loc[visits.visit_confidence.eq('high')].assign(vessel_uid_clean=lambda v: v.vessel_uid.astype(str))
    for c in ['entry_time', 'exit_time']:
        visits[c] = pd.to_datetime(visits[c], utc=True)
    f = f.merge(visits[['vessel_uid_clean', 'exit_time', 'entry_rule']].rename(columns={'exit_time': 'departure_time', 'entry_rule': 'origin_rule'}),
                on=['vessel_uid_clean', 'departure_time'], how='left', validate='one_to_one')
    f = f.merge(visits[['vessel_uid_clean', 'entry_time', 'entry_rule']].rename(columns={'entry_time': 'arrival_time', 'entry_rule': 'dest_rule'}),
                on=['vessel_uid_clean', 'arrival_time'], how='left', validate='one_to_one')
    if f[['origin_rule', 'dest_rule']].isna().any().any():
        raise ValueError('a released voyage endpoint has no high-confidence visit')
    f['annulus_endpoint'] = f.origin_rule.eq('stop_supported_annulus') | f.dest_rule.eq('stop_supported_annulus')
    t = f.groupby('segmentation').agg(voyages=('candidate_id', 'size'), annulus_endpoint_voyages=('annulus_endpoint', 'sum')).reset_index()
    t['share_of_released'] = t.voyages / len(f)
    t['annulus_endpoint_share'] = t.annulus_endpoint_voyages / t.voyages
    t['share_of_released_without_annulus_endpoints'] = (t.voyages - t.annulus_endpoint_voyages) / len(f)
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['gaps', 'weather', 'segmentation', 'intervals', 'base', 'ports', 'out']:
        p.add_argument('--' + name, type=lambda v: Path(v).resolve(), required=True)
    args = p.parse_args()
    if not args.out.is_relative_to((CODING / 'outputs').resolve()):
        raise ValueError('review packages live in 02_coding/outputs')
    after = {k: getattr(args, k) for k in BEFORE}
    states = {'gaps': json.loads((args.gaps / 'execution.json').read_text())['state'],
              'weather': json.loads((args.weather / 'execution.json').read_text())['state'],
              'segmentation': json.loads((args.segmentation / 'execution.json').read_text())['state'],
              'intervals': json.loads((args.intervals / 'evaluate_execution.json').read_text())['state']}
    if any(s != 'pass' for s in states.values()):
        raise ValueError(f'not every rerun passed: {states}')
    args.out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, (run, path) in TABLES.items():
        shutil.copy2(after[run] / path, args.out / name)
        files[name] = {'source': str(after[run] / path), **{k: v for k, v in core.record(args.out / name).items() if k != 'path'}}
    old, new = headline(BEFORE), headline(after)
    both = old.merge(new, on=['group', 'metric', 'unit'], suffixes=('_before', '_after'), validate='one_to_one')
    both['change'] = both.value_after - both.value_before
    both.to_csv(args.out / '00_headline_before_after.csv', index=False)
    endpoint_entry_rules(args.segmentation, args.base, args.ports).to_csv(args.out / '15_segmentation_by_endpoint_entry_rule.csv', index=False)
    git = lambda *a: subprocess.check_output(['git', *a], cwd=HERE, text=True).strip()
    manifest = {'package': args.out.name, 'branch': git('branch', '--show-current'), 'commit': git('rev-parse', 'HEAD'),
                'reruns': {k: str(v) for k, v in after.items()}, 'states': states, 'base': str(args.base), 'ports': str(args.ports),
                'compared_with': {k: str(v) for k, v in BEFORE.items()}, 'files': files, 'adopted': False}
    core.write_json(args.out / '14_review_manifest.json', manifest)
    print(both.to_string())


if __name__ == '__main__':
    main()
