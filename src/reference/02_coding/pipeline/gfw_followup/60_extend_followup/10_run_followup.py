"""Run February follow-up from prepared observation shards and weather inputs."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

EXPERIMENT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from prepared_inputs import validate_buffer

CODING=EXPERIMENT.parents[1]
RUNS=CODING/'.runs'
JAN_PORTS=RUNS/'2026-10-21_2154_E009-candidate-port-view'
JAN_QUALITY=RUNS/'2026-10-21_2324_E009-quality-base'
JAN_WEATHER=RUNS/'2026-10-21_232515_E009-tracks-weather'
JAN_RELEASE=RUNS/'2026-10-21_235106_E009-bounds-monthly-release'


def record(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def inputs(buffer,assets):
    validate_buffer(buffer, '2026-02')
    manifest=json.loads((assets/'manifest.json').read_text())
    if manifest['status']!='pass' or manifest.get('observation_end_exclusive')!='2026-03-01':
        raise ValueError('February weather input gate not passed')
    for key in ['config','calendar_audit','source_matcher','adapter','derived_matcher']:
        if record(Path(manifest[key]['path']))!=manifest[key]:raise ValueError('weather input changed: '+key)
    prior=json.loads((JAN_RELEASE/'summary.json').read_text())
    if prior['status']!='pass' or prior['voyages']!=1135377 or prior['old_voyages']!=1111800:
        raise ValueError('January baseline population differs')
    return [record(buffer/'fleet_presence_unique_summary.json'),record(assets/'manifest.json'),record(JAN_RELEASE/'summary.json')]


def run(buffer,assets,root,workers):
    if not root.is_relative_to(RUNS.resolve()) or not 1<=workers<=32:raise ValueError('invalid run path or workers')
    sources=inputs(buffer,assets)  # No output directory or dependent work before gates.
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    code=[record(p) for p in sorted(EXPERIMENT.rglob('*.py')) if '__pycache__' not in p.parts and 'tests' not in p.parts]
    state={'status':'running','scope':'2025 departures; observation before 2026-03-01 UTC','inputs':sources,
           'code':code,'started_hkt':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
           'pid':os.getpid(),'workers':workers,'phases':[],'scientific_adoption':False,'public_upload':False}
    save(root/'execution.json',state)
    (root/'00_README.md').write_text('# February observation extension\n\nStart with execution.json and each stage receipt. Prepared observation shards, weather inputs and the January baseline are checked before processing. Each build stage starts with a two-shard probe.\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def execute(name,entry,args,receipt):
        if any(record(Path(v['path']))!=v for v in code):raise ValueError('implementation changed during chain')
        command=[sys.executable,'-u',str(EXPERIMENT/entry),*[str(v) for v in args]]
        log=root/'logs'/(name+'.log');phase={'name':name,'status':'running','command':command,'log':str(log)}
        state['phases'].append(phase);save(root/'execution.json',state)
        with log.open('xb') as output:
            subprocess.run(command,cwd=CODING.parent,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
        result=json.loads(receipt.read_text())
        if result.get('status')!='pass':raise ValueError(name+' did not pass')
        phase.update(status='pass',receipt=record(receipt));save(root/'execution.json',state)
        with (root/'00_README.md').open('a') as f:
            f.write(f'\n- {name}: [阶段回执]({receipt.relative_to(root)}); [日志]({log.relative_to(root)})。')
            required=receipt.parent/'required_actual.json'
            if required.exists():f.write(f' [Required / Actual / Status]({required.relative_to(root)})。')
            f.write('\n')
        print(json.dumps({'phase':name,'status':'pass','receipt':str(receipt)}),flush=True)
    try:
        port=EXPERIMENT/'20_build_cross_year_visits/10_rebuild_buffered_visits.py'
        common=['--buffer',buffer,'--baseline-visits',JAN_PORTS,'--baseline-panel',JAN_RELEASE/'preparation/all_voyage_panel.parquet','--observation-end','2026-03-01']
        execute('visits_probe',port,common+['--out',root/'visits_probe','--shards','shard_0032','shard_0057','--workers',2],root/'visits_probe/execution.json')
        execute('visits',port,common+['--out',root/'visits','--shards',*[f'shard_{i:04d}' for i in range(64)],'--workers',workers,'--verified-probe',root/'visits_probe'],root/'visits/execution.json')
        candidate=EXPERIMENT/'30_build_buffered_candidates/10_enumerate_candidates.py'
        common=['--baseline-candidates',JAN_PORTS/'candidates/departure_2025_candidates.parquet','--observation-end','2026-03-01','--workers',workers]
        execute('candidates_probe',candidate,common+['--visits',root/'visits_probe','--out',root/'candidates_probe','--probe'],root/'candidates_probe/execution.json')
        execute('candidates',candidate,common+['--visits',root/'visits','--out',root/'candidates','--verified-probe',root/'candidates_probe'],root/'candidates/execution.json')
        quality=EXPERIMENT/'40_complete_tracks_and_weather/20_build_quality_base.py'
        common=['--baseline-quality',JAN_QUALITY,'--observation-end','2026-03-01','--workers',workers]
        execute('quality_probe',quality,common+['--candidates',root/'candidates_probe','--ports',root/'visits_probe','--out',root/'quality_probe','--probe'],root/'quality_probe/execution.json')
        execute('quality',quality,common+['--candidates',root/'candidates','--ports',root/'visits','--out',root/'quality','--verified-probe',root/'quality_probe'],root/'quality/execution.json')
        tracks=EXPERIMENT/'40_complete_tracks_and_weather/30_run_tracks_and_weather.py'
        common=['--quality',root/'quality','--ports',root/'visits','--assets',assets,'--baseline-weather',JAN_WEATHER,'--workers',workers]
        execute('tracks_weather_probe',tracks,common+['--out',root/'tracks_weather_probe','--probe'],root/'tracks_weather_probe/execution.json')
        execute('tracks_weather',tracks,common+['--out',root/'tracks_weather','--verified-probe',root/'tracks_weather_probe'],root/'tracks_weather/execution.json')
        bounds=EXPERIMENT/'40_complete_tracks_and_weather/40_build_bounds_and_release.py'
        execute('bounds_monthly',bounds,['--weather',root/'tracks_weather','--quality',root/'quality','--assets',assets,
            '--baseline-release',JAN_RELEASE,'--baseline-quality',JAN_QUALITY,'--out',root/'bounds_monthly','--workers',workers],root/'bounds_monthly/execution.json')
        audit=EXPERIMENT/'50_audit_followup_censoring'
        execute('censoring_audit',audit/'10_audit_followup_censoring.py',['--panel',root/'bounds_monthly/preparation/all_voyage_panel.parquet',
            '--observation-cutoff','2026-03-01','--output',root/'censoring_audit'],root/'censoring_audit/summary.json')
        execute('monthly_comparison',audit/'30_compare_departure_months.py',['--release',root/'bounds_monthly','--baseline-release',JAN_RELEASE,
            '--out',root/'monthly_comparison'],root/'monthly_comparison/summary.json')
        execute('pending_followup',audit/'20_audit_pending_followup.py',['--candidates',root/'candidates','--ports',root/'visits',
            '--baseline-candidates',JAN_PORTS/'candidates','--baseline-ports',JAN_PORTS,'--out',root/'pending_followup','--month','2026-02'],root/'pending_followup/summary.json')
        state.update(status='pass',phase='calculation_complete_pending_review')
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:save(root/'execution.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['buffer','assets','out']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    run(a.buffer.resolve(),a.assets.resolve(),a.out.resolve(),a.workers)
