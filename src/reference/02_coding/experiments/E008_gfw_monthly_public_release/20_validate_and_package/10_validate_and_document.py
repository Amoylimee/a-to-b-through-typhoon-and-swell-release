#!/usr/bin/env python3
"""Independently check monthly samples and prepare aggregate-only documentation."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import monthly_release as m


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-7, equal_nan=True)


def independent_samples(panel, table, keys):
    """Use raw voyage sums and one explicit 1000-draw array, without the bootstrap kernel."""
    eligible = table.loc[table.ships.ge(3)]
    mixed = eligible.loc[eligible.wave_coverage.between(0.01, 0.7)]
    chosen = pd.concat([eligible.nlargest(1, 'ships'), eligible.nsmallest(1, 'ships'),
                        mixed.head(3), eligible.iloc[[len(eligible)//2]],
                        table.loc[table.ships.lt(3)].head(1)]).drop_duplicates(keys)
    rows = []
    for _, row in chosen.iterrows():
        key = tuple(row[k] for k in keys)
        mask = np.ones(len(panel), dtype=bool)
        for field, value in zip(keys, key):
            mask &= panel[field].eq(value).to_numpy()
        part = panel.loc[mask]
        assert len(part) == row.voyages
        clusters = part.groupby(m.SHIP, sort=True)[m.BOOT_SUMS].sum()
        assert len(clusters) == row.ships
        for field in m.SUMS:
            # Interval sufficient statistics are checked separately by the full build.
            if field in row.index:
                close(float(part[field].sum()), float(row[field]))
        if len(clusters) < 3:
            assert row[m.CI].isna().all()
            continue
        label = '|'.join(str(x) for x in key)
        seed = int.from_bytes(hashlib.sha256(('20260923|' + label).encode()).digest()[:4], 'big')
        draws = np.random.default_rng(seed).integers(0, len(clusters), size=(1000, len(clusters)))
        sums = clusters.to_numpy()[draws].sum(axis=1)
        col = {name: sums[:, i] for i, name in enumerate(clusters.columns)}
        with np.errstate(divide='ignore', invalid='ignore'):
            estimates = {
                'mean_swh_matched': col['wave_integral_m_hours']/col['wave_matched_hours'],
                'share_wave_hours_ge4m': col['wave_ge4m_matched_hours']/col['wave_matched_hours'],
                'tc300_at_least_pct': 100*col['tc300_positive_snapshot_hours']/col['voyage_hours'],
                'tc500_at_least_pct': 100*col['tc500_positive_snapshot_hours']/col['voyage_hours'],
            }
        for metric, values in estimates.items():
            finite = values[np.isfinite(values)]
            bounds = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
            close([row[metric + '_ci95_low'], row[metric + '_ci95_high']], bounds)
        rows.append({'key': dict(zip(keys, key)), 'ships': int(row.ships), 'voyages': int(row.voyages)})
    return {'status': 'pass', 'selected_records': len(chosen), 'bootstrap_records': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    run, review = args.run.resolve(), args.review.resolve()
    assert run.is_relative_to(m.ROOT/'02_coding/.runs')
    assert review.is_relative_to(m.ROOT/'02_coding/outputs')
    state = json.loads((run/'validation.json').read_text())
    assert state['state'] == 'validated'
    binding = json.loads((run/'binding.json').read_text())
    config = json.loads((HERE/'config/authorization.json').read_text())
    panel, _ = m.load_source(Path(binding['source']), config)
    mapping = pd.read_csv(run/'private/city_code_map_PRIVATE.csv')
    assert mapping.city_id.is_unique and mapping.city_code.is_unique
    assert (run/'private').stat().st_mode & 0o077 == 0
    assert (run/'private/city_code_map_PRIVATE.csv').stat().st_mode & 0o077 == 0
    lookup = mapping.set_index('city_id').city_code
    for side in ('origin', 'dest'):
        panel[side+'_city_code'] = panel[side+'_city'].map(lookup)
    checks = {}
    tables = {}
    for name, keys in (('country', m.COUNTRY_KEY), ('city', m.CODED_KEY)):
        table = pd.read_parquet(run/'private'/f'{name}_full_precision.parquet')
        tables[name] = table
        expected = panel.groupby(keys, observed=True).agg(voyages=('candidate_id','size'), ships=(m.SHIP,'nunique'))
        pd.testing.assert_frame_equal(table.set_index(keys)[['voyages','ships']].sort_index(), expected.sort_index(), check_dtype=False)
        for stem in ('mean_swh_matched', 'share_wave_hours_ge4m', 'tc300_at_least_pct', 'tc500_at_least_pct'):
            lo, hi = table[stem+'_ci95_low'], table[stem+'_ci95_high']
            assert lo.isna().equals(hi.isna())
            assert (lo.dropna() <= hi.dropna()).all()
            assert (lo.dropna() >= 0).all()
        assert table.loc[table.ships.lt(3), m.CI].isna().all().all()
        assert table.loc[table.ships.ge(3), ['tc300_at_least_pct_ci95_low','tc500_at_least_pct_ci95_low']].notna().all().all()
        csv_path = run/'public'/state['tables'][name]['file']
        assert m.digest(csv_path) == state['tables'][name]['sha256']
        reread = pd.read_csv(csv_path, low_memory=False)
        m.public_schema(reread, keys)
        assert len(reread) == len(expected)
        checks[name] = {'all_record_voyage_and_ship_counts': 'pass', 'interval_order_and_missingness': 'pass',
                        'public_schema_and_hash': 'pass', 'independent_resamples': independent_samples(panel, table, keys)}
    # Reconstruct country means from the full-precision city table using duration weights.
    city = tables['city'].copy()
    city['_wave_integral'] = city.mean_swh_matched_m.fillna(0)*city.wave_matched_hours
    rolled = city.groupby(m.COUNTRY_KEY)[['_wave_integral','wave_matched_hours']].sum()
    target = tables['country'].set_index(m.COUNTRY_KEY).sort_index()
    close(rolled._wave_integral/rolled.wave_matched_hours, target.mean_swh_matched_m)
    checks['weighted_city_to_country_wave_mean'] = 'pass'
    guide = (HERE/'resources/03_FIELD_GUIDE.md').read_text()
    defined = re.findall(r'^\| `([^`]+)` \|', guide, re.M)
    assert len(defined) == len(set(defined)) == 52
    assert set(defined) == set(m.CODED_KEY + m.MEASURES)
    checks['field_guide_exact_52_variables'] = 'pass'
    report = {'status': 'pass', 'checks': checks, 'script_sha256': m.digest(__file__),
              'build_commit': state['source_commit'], 'source_sha256': config['source_sha256'],
              'private_map_sha256': m.digest(run/'private/city_code_map_PRIVATE.csv')}
    m.write_json(run/'independent_validation.json', report)
    public = run/'public'
    shutil.copy2(HERE/'resources/03_FIELD_GUIDE.md', public/'03_FIELD_GUIDE.md')
    readme = ['# 2025 monthly voyage exposure — release candidate', '',
              'This dataset summarises cyclone and wave exposure for voyages in the 2025 Global Fishing Watch sample. Read this overview first, then use the field guide to interpret the estimates and their coverage.', '',
              '| File | Records | Fields |', '|---|---:|---:|']
    for name in ('country', 'city'):
        item = state['tables'][name]
        readme.append(f"| `{item['file']}` | {item['rows']:,} | {item['columns']} |")
    readme += ['', 'Each record describes a directed location pair, a vessel type and a UTC departure month. Both tables cover the same 1,111,800 analysed voyages, totalling 140,832,051 hours. The full voyage duration is assigned to its departure month. All combinations are retained, including those with one voyage.', '',
               'The country table identifies the departure and arrival countries or territories. The city table adds fixed city codes. These codes preserve links between records within this release. Their numbers have no geographic meaning, and the public files contain no city-name mapping.', '',
               'Each table includes sample sizes, time coverage, cyclone exposure, wave exposure and uncertainty measures. Records with fewer than three vessels have no sampling intervals. Blank wave estimates reflect unavailable wave information. The field guide explains the denominators, flags and two reasons for missing wave means.', '',
               'The two tables are alternative geographic views of the same sample. Summing both tables would duplicate every voyage. Vessel counts and sampling intervals cannot be added across records.', '',
               'Files are UTF-8 comma-separated values (CSV), with a header row and blank cells for missing values. Hours have one decimal place, and metres, fractions and percentages have three. Full-precision values were used in calculations.', '',
               'See `03_FIELD_GUIDE.md` for all 52 variables. `04_MANIFEST.json` gives file sizes and SHA-256 checksums. This candidate has been prepared for review; repository identifiers and release licensing will be added at deposit.', '']
    (public/'00_README.md').write_text('\n'.join(readme))
    allowed = ['00_README.md', '01_release_country_pair_month.csv', '02_release_city_pair_month.csv', '03_FIELD_GUIDE.md']
    manifest = {p: {'bytes': (public/p).stat().st_size, 'sha256': m.digest(public/p)} for p in allowed}
    m.write_json(public/'04_MANIFEST.json', manifest)
    assert {p.name for p in public.iterdir()} == set(allowed + ['04_MANIFEST.json'])
    review.mkdir(parents=True, exist_ok=False)
    for filename in ('00_README.md', '03_FIELD_GUIDE.md', '04_MANIFEST.json'):
        shutil.copy2(public/filename, review/('PUBLIC_'+filename))
    shutil.copy2(run/'validation.json', review/'01_BUILD_VALIDATION.json')
    shutil.copy2(run/'independent_validation.json', review/'02_INDEPENDENT_VALIDATION.json')
    lines = ['# E008 — GFW 月度发布表核查', '',
             '先读本文件。目的为把当前已采纳 GFW 航次整理成国家对和城市编码对月度表，供改稿与发布准备使用。全量生成和独立数值核验已通过；尚未合并或公开上传。', '',
             f"代码分支 `{state['branch']}`；构建提交 `{state['source_commit']}`。采纳的科学基准仍为 `{config['research_accepted_commit']}`，此次没有改动 Contract 或 Dashboard。", '',
             f'服务器完整结果：`{run}`。公开候选文件在 `public/`；私有编号对照、全精度表和抽样证据在 `private/`。', '',
             '| 表 | 记录 | 字段 | 只有 1 航次 | 2–4 航次 | 不足 3 船 | 无浪高主值 | 无浪高范围 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, label in (('country','国家对月度'),('city','城市编码对月度')):
        t = state['tables'][name]
        vals = [t[k] for k in ('rows','columns','records_with_1_voyage','records_with_2_to_4_voyages','records_with_fewer_than_3_ships','records_without_wave_mean','records_without_wave_range')]
        lines.append('| '+label+' | '+' | '.join(f'{v:,}' for v in vals)+' |')
    lines += ['', '两表分别覆盖全部 1,111,800 航次和 140,832,051 小时。月份按 UTC 出发时刻划分，跨月航次的全部时长归入出发月。城市编号在此次 GFW 版本内固定；旧 E004 对照表保留原样。', '',
              '验证包括 8 项行为测试、15 个真实年度组合的抽样复现、全年 227,742 个组合的 38 项非抽样字段复现、月度到全年及城市到国家的加总核对，以及两表全部记录的独立航次/船数核对。另用独立计算复核新月度分组的 1,000 次抽样。公开 CSV 经字段白名单、文件哈希、精度和空值回读检查；字段说明逐项覆盖 52 个变量。', '',
              '配套证据：`01_BUILD_VALIDATION.json`、`02_INDEPENDENT_VALIDATION.json`。`PUBLIC_00_README.md`、`PUBLIC_03_FIELD_GUIDE.md` 为公开说明的副本；`PUBLIC_04_MANIFEST.json` 固定公开候选文件的哈希。', '',
              '改稿时将原来的全年城市对、≥5 航次主表说明替换为两张月度表，更新记录数、字段数和例子。国家对表可用真实国家解释例子；城市编码表的示例保持编号，不提供真实城市对照。Data Overview、船型数量表及相关图件仍需按新的发布总体逐项检查。轨迹、天气与缺失位置范围的原有计算没有重跑。', '']
    (review/'00_README.md').write_text('\n'.join(lines))
    print(json.dumps({'status': 'pass', 'review': str(review), 'public': str(public), 'checks': checks}, ensure_ascii=False))


if __name__ == '__main__':
    main()
