"""Bound observation-calendar extension around the accepted weather functions.

The production matcher has three literal calendar assumptions. Adapt only those
three expressions in memory, preserving all scientific kernels and saving the
exact derived source alongside the input receipt. Production files stay intact.
"""
from pathlib import Path
import inspect
import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CODING=HERE.parents[2]
PIPE=CODING/'pipeline/gfw_release'
STAGE=CODING/'pipeline/40_construct_variables/60_construct_voyage_weather_facts'
sys.path.insert(0,str(PIPE));sys.path.insert(0,str(STAGE))
import step6_core as core
from weather_core.tropical_cyclone import load_ibtracs_snapshot
from weather_core.track_adapter import verify_tc_calendar_audit

CONFIG=STAGE/'config/2025_ten_minute_weather.json'
WAVE=CODING/'.runs/2026-10-21_2311_E009-january-wave'
MONTHS=[f'2025-{m:02d}' for m in range(1,13)]+['2026-01']
EXTENSIONS={
    'time_end_exclusive="2026-01-02T00:00:00Z"': 'time_end_exclusive="2026-02-02T00:00:00Z"',
    '.format(month=int(month[-2:]))': '.format(year=int(month[:4]), month=int(month[-2:]))',
    'month.startswith("2025-") and wave_path.is_file()': 'month in config["allowed_weather_months"] and wave_path.is_file()',
}
OBSERVATION_END='2026-02-01'
TC_END='2026-02-02T00:00:00Z'


def configure(observation_end):
    global MONTHS,EXTENSIONS,OBSERVATION_END,TC_END
    if observation_end not in {'2026-02-01','2026-03-01'}:raise ValueError('unauthorized weather observation cutoff')
    OBSERVATION_END=observation_end
    end=pd.Timestamp(observation_end,tz='UTC')
    TC_END=(end+pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    MONTHS=pd.date_range('2025-01-01',end.tz_localize(None),freq='MS',inclusive='left').strftime('%Y-%m').tolist()
    EXTENSIONS={**EXTENSIONS,'time_end_exclusive="2026-01-02T00:00:00Z"':f'time_end_exclusive="{TC_END}"'}


def adapted_source():
    matcher=core.module('e009_source_weather',STAGE/'40_match_ten_minute_weather_shard.py')
    source=inspect.getsource(matcher.match_validated_weather_inputs)
    for before,after in EXTENSIONS.items():
        if source.count(before)!=1: raise ValueError('accepted matcher changed; calendar adaptation requires review')
        source=source.replace(before,after)
    return matcher,source


def prepare_assets(out,observation_end='2026-02-01',wave_roots=None):
    configure(observation_end)
    out.mkdir(parents=True,exist_ok=False)
    config=json.loads(CONFIG.read_text());old_audit=verify_tc_calendar_audit(config)
    wave_roots={'2026-01':WAVE,**(wave_roots or {})};extra={}
    for month in [m for m in MONTHS if m.startswith('2026-')]:
        receipt=json.loads((Path(wave_roots[month])/'execution.json').read_text())
        if receipt['status']!='pass' or not receipt['grid_matches_reference'] or receipt['hours']!=pd.Timestamp(month+'-01').days_in_month*24:
            raise ValueError('wave acquisition has not passed: '+month)
        if core.record(Path(receipt['output']['path']))!=receipt['output']:raise ValueError('wave source changed: '+month)
        extra[month]=receipt['output']
    waves=out/'waves';waves.mkdir()
    records=[]
    for month in MONTHS:
        source=Path(extra[month]['path']) if month in extra else Path(config['wave_directory'])/config['wave_filename_template'].format(month=int(month[-2:]))
        records.append(core.record(source))
        (waves/f'swh_{month}.nc').symlink_to(source.resolve())
    end=pd.Timestamp(observation_end,tz='UTC');start=end-pd.DateOffset(months=1)
    old_end=(start+pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    fixes,loader=load_ibtracs_snapshot(config['ibtracs_path'],expected_sha256=config['ibtracs_sha256'],time_end_exclusive=TC_END)
    old,_=load_ibtracs_snapshot(config['ibtracs_path'],expected_sha256=config['ibtracs_sha256'],time_end_exclusive=old_end)
    pd.testing.assert_frame_equal(old,fixes.loc[fixes.fix_time_utc.lt(old_end)].reset_index(drop=True),check_exact=True)
    window=fixes.loc[fixes.fix_time_utc.ge(start-pd.Timedelta(days=1)) & fixes.fix_time_utc.lt(TC_END)]
    gap=window.groupby(['SID','retained_phase']).fix_time_utc.diff().dt.total_seconds()/3600
    current=window.loc[window.fix_time_utc.ge(start) & window.fix_time_utc.lt(end)]
    raw=pd.read_csv(config['ibtracs_path'],skiprows=[1],usecols=['ISO_TIME'],dtype=str)
    times=pd.to_datetime(raw.ISO_TIME.str.strip(),utc=True)
    audit={'status':'pass','sha256':config['ibtracs_sha256'],'allowed_weather_months':MONTHS,
           'month':start.strftime('%Y-%m'),'observation_end_exclusive':observation_end,'tc_end_exclusive':TC_END,
           'within_source_span':bool(times.min()<=start and times.max()>=end),
           'raw_month_rows':int((times.ge(start)&times.lt(end)).sum()),
           'retained_rows':len(current),'retained_storms':int(current.SID.nunique()),
           'invalid_centres':int((~window.center_valid).sum()),'within_retained_phase_gaps_over_3h':int(gap.gt(3).sum()),
           'old_fixes_exact':True,'best_track_final':False,'old_calendar_audit':core.record(Path(config['ibtracs_calendar_coverage_audit'])),
           'interpretation':'Source calendar span and within-phase continuity only; not final best tracks or exhaustive storm detection.',
           'loader':loader}
    if not audit['within_source_span'] or audit['invalid_centres'] or audit['within_retained_phase_gaps_over_3h']:
        audit['status']='failed'
    audit_path=out/('tc_january_coverage.json' if start.month==1 else 'tc_february_coverage.json')
    core.write_json(audit_path,audit)
    if audit['status']!='pass': raise ValueError('January cyclone support audit failed')
    config.update(wave_directory=str(waves),wave_filename_template='swh_{year:04d}-{month:02d}.nc',
                  allowed_weather_months=MONTHS,e009_calendar_audit=str(audit_path),e009_observation_end=observation_end)
    core.write_json(out/'weather_config.json',config)
    _,source=adapted_source();(out/'derived_matcher.py').write_text(source)
    core.write_json(out/'manifest.json',{'status':'pass','weather_files':records,'config':core.record(out/'weather_config.json'),
                    'calendar_audit':core.record(audit_path),'source_matcher':core.record(STAGE/'40_match_ten_minute_weather_shard.py'),
                    'adapter':core.record(Path(__file__)),'derived_matcher':core.record(out/'derived_matcher.py'),
                    'calendar_replacements':EXTENSIONS,'observation_end_exclusive':observation_end,'scientific_adoption':False})
    return audit


def verify_extended_calendar(config):
    old=verify_tc_calendar_audit(config)
    audit=json.loads(Path(config['e009_calendar_audit']).read_text())
    if (audit['status']!='pass' or audit['sha256']!=config['ibtracs_sha256']
            or audit['allowed_weather_months']!=MONTHS or not audit['old_fixes_exact']
            or not audit.get('within_source_span',audit.get('january_within_source_span')) or audit['invalid_centres'] or audit['within_retained_phase_gaps_over_3h']):
        raise ValueError('unverified January cyclone coverage')
    return {'new':{**old['new'],'e009_extension':audit}}


def match(task,assets):
    binding=json.loads((assets/'manifest.json').read_text())
    configure(binding.get('observation_end_exclusive','2026-02-01'))
    for key in ['config','calendar_audit','source_matcher','adapter','derived_matcher']:
        if core.record(Path(binding[key]['path'])) != binding[key]: raise ValueError(f'weather binding changed: {key}')
    matcher,source=adapted_source()
    if source!=(assets/'derived_matcher.py').read_text(): raise ValueError('derived weather function changed')
    matcher.verify_tc_calendar_audit=verify_extended_calendar
    exec(compile(source,str(assets/'derived_matcher.py'),'exec'),matcher.__dict__)
    track=Path(task['output']);sources={k:core.record(track/name) for k,name in
        [('track','ten_minute_complete_candidate.parquet'),('summary','candidate_summary.parquet')]}
    namespace=SimpleNamespace(output=Path(task['weather']),config=assets/'weather_config.json',shard=task['shard'],max_candidates=None)
    clearance={'authorization':'E009 user-authorized observation extension','scope':'2025 departures observed before '+OBSERVATION_END,
               'weather_extension':core.record(assets/'manifest.json'),'scientific_adoption':False}
    state=matcher.match_validated_weather_inputs(namespace,clearance,sources)
    state.update(e009_calendar_extension=clearance['weather_extension'],population='E009_2025_departures',adopted=False)
    core.write_json(namespace.output/'status.json',state)
    validator=core.module('e009_weather_check',PIPE/'190_validate_gfw_weather.py')
    original=validator.load_ibtracs_snapshot
    def extended_loader(*args,**kwargs):
        if kwargs.get('time_end_exclusive')!='2026-01-02T00:00:00Z': raise ValueError('validator calendar changed')
        kwargs['time_end_exclusive']=TC_END
        return original(*args,**kwargs)
    validator.load_ibtracs_snapshot=extended_loader
    check=validator.validate_shard(namespace.output)
    if check['status']!='pass' or not all(check['checks'].values()): raise ValueError('weather validation failed')
    return check
