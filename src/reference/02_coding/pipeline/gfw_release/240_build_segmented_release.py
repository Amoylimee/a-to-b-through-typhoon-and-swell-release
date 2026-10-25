#!/usr/bin/env python3
"""Assemble and validate the corrected candidate release, without adoption.

The timing-only probe is a verified, unchanged component control. New annual
segmentation, weather, bounds, holdouts and bootstraps are evaluated separately.
It is never labelled a validation of the corrected voyage population.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import filecmp
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent
CODING=HERE.parents[1]
E006=CODING/'pipeline/gfw_release'
PYTHON=sys.executable
CONTROL=CODING/'.runs/2026-10-11_1238_E007-wave-support-release'


def port_audit(run):
    """Recompute the blind-port audit from this run's corrected endpoints."""
    import sys
    sys.path.insert(0,str(E006))
    from port_constraints import port_feasibility
    from step6_core import record,write_json
    audit=port_feasibility(run/'preparation')
    audit['sources']=[record(run/'preparation'/name) for name in
                     ['voyage_panel.parquet','port_endpoints.parquet']]
    audit['code']=record(E006/'port_constraints.py')
    write_json(run/'gfw_wholly_unobserved_port_feasibility.json',audit)


def resume_finalization(run,weather,review,env):
    """Retain the failed execution and rerun only audit, validation and docs."""
    import sys
    sys.path.insert(0,str(E006))
    from step6_core import record,write_json
    previous=json.loads((run/'execution.json').read_text())
    if previous['weather']!=str(weather):raise ValueError('resume weather source changed')
    if json.loads((run/'assembly/status.json').read_text())['state']!='assembled':
        raise ValueError('resume requires completed assembly')
    out=run/'finalization';out.mkdir(exist_ok=False)
    state={'state':'running','previous_execution':record(run/'execution.json'),
           'code':record(Path(__file__)),'completed':[],'adopted':False,'network_requests':0}
    write_json(out/'execution.json',state)
    try:
        port_audit(run)
        state['port_audit']=record(run/'gfw_wholly_unobserved_port_feasibility.json')
        for name,args in [('260_validate_gfw_release.py',['--run',str(run)]),
                          ('270_document_gfw_release.py',['--run',str(run),'--out',str(review)])]:
            subprocess.run([PYTHON,'-u',str(E006/name),*args],cwd=CODING.parent,env=env,check=True)
            state['completed'].append(name);write_json(out/'execution.json',state)
        state.update(state='completed',validation=record(run/'validation.json'),review=str(review))
    except BaseException as error:
        state.update(state='failed',error=repr(error));raise
    finally:write_json(out/'execution.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weather',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--bound-workers',type=int,default=32);p.add_argument('--other-workers',type=int,default=8)
    p.add_argument('--resume-finalization',action='store_true')
    args=p.parse_args();run=args.run.resolve();weather=args.weather.resolve()
    if not run.is_relative_to((CODING/'.runs').resolve()):raise ValueError('private outputs only')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
             E006_STEP6_DECISION=str(weather/'step6_decision.json'))
    if args.resume_finalization:
        resume_finalization(run,weather,args.review.resolve(),env)
        return
    run.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def stamp(*a):print(datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),*map(str,a),flush=True)
    def execute(script,*a):
        nonlocal command_process
        stamp(script.name,*a)
        command_process=subprocess.Popen([PYTHON,'-u',str(script),*map(str,a)],cwd=CODING.parent,env=env)
        while command_process.poll() is None:
            advance_bounds()
            time.sleep(1)
        advance_bounds()
        if command_process.returncode:
            raise subprocess.CalledProcessError(command_process.returncode,command_process.args)
        command_process=None
        state['completed'].append([script.name,*map(str,a)])
        write(run/'execution.json',state)
    def record(path):
        import hashlib
        path=Path(path).resolve()
        return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')
    state={'state':'running','weather':str(weather),'run':str(run),'adopted':False,'network_requests':0,
           'code':record(Path(__file__)),'completed':[]}
    write(run/'execution.json',state)
    process=None;log=None;probe_process=None;probe_log=None;command_process=None
    def advance_bounds():
        """Launch the annual census only after its two-shard gate passes.

        The holdout/bootstrap commands use independent output paths and at
        most eight workers, leaving headroom beside the 32-worker census.
        """
        nonlocal process,log,probe_process,probe_log
        if probe_process is not None and probe_process.poll() is not None:
            if probe_process.returncode:
                raise RuntimeError('corrected two-shard bound gate failed')
            probe_log.close();probe_log=None;probe_process=None
            state['completed'].append(['two_shard_bound_equivalence_gate','pass'])
            stamp('launching_annual_bounds_after_two_shard_gate')
            log=open(run/'bounds.log','w')
            process=subprocess.Popen([PYTHON,'-u',str(HERE/'235_reuse_equivalent_segmented_bounds.py'),'--prepared',str(run/'preparation'),
                         '--base',base,'--out',str(run/'bounds'),'--workers',str(args.bound_workers),
                         '--reuse-components',str(run/'bounds_equivalence_probe')],cwd=CODING.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['bounds_process']=process.pid
            write(run/'execution.json',state)
        if process is not None and process.poll() not in (None,0):
            raise RuntimeError('corrected annual bound census failed')
    try:
        execute(E006/'200_prepare_gfw_release.py','--out',run/'preparation')
        # Exercise the installed grid-support adapter on the new population,
        # including exact hour-level results, before launching the census.
        for tag,workers in [('serial',1),('parallel',2)]:
            execute(HERE/'230_bound_segmented_weather.py','--prepared',run/'preparation',
                    '--out',run/f'bounds_probe_{tag}','--workers',workers,
                    '--shards','shard_0005','shard_0009','--probe-candidates',5,'--hour-detail')
        import pandas as pd
        pairs=[]
        for shard in ['shard_0005','shard_0009']:
            for name in ['voyage_bounds.parquet','hour_bounds.parquet']:
                a,b=(run/f'bounds_probe_{tag}'/shard/name for tag in ['serial','parallel'])
                pd.testing.assert_frame_equal(pd.read_parquet(a),pd.read_parquet(b),check_exact=True)
                pairs.append({'serial':record(a),'parallel':record(b),'exact':True})
        write(run/'segmented_bounds_serial_parallel_validation.json',{'status':'pass','checks':pairs,'population':'new_corrected_candidates'})
        # Preserve the existing timing control with its own source hashes.
        # The wave-support and segmentation changes have their own direct tests.
        timing=json.loads((CONTROL/'bounds_tolerance_probe.json').read_text())
        for item in timing['receipts'].values():
            if record(item['path'])!=item:raise ValueError('timing control receipt changed')
        write(run/'bounds_tolerance_probe.json',timing)
        write(run/'timing_control_scope.json',{'scope':'Historical component proof of unchanged timestamp tolerance only; not a new-population or wave-support validation.',
              'source':record(CONTROL/'bounds_tolerance_probe.json'),'new_segmentation_validated_separately':True})
        base=json.loads((weather/'execution.json').read_text())['base']
        probe_log=open(run/'bounds_equivalence_probe.log','w')
        probe_process=subprocess.Popen([PYTHON,'-u',str(HERE/'235_reuse_equivalent_segmented_bounds.py'),'--prepared',str(run/'preparation'),
                    '--base',base,'--out',str(run/'bounds_equivalence_probe'),'--workers','2','--shards','shard_0005','shard_0009'],
                    cwd=CODING.parent,env=env,stdout=probe_log,stderr=subprocess.STDOUT)
        state['bounds_probe_process']=probe_process.pid;write(run/'execution.json',state)
        execute(E006/'220_prepare_gfw_holdouts.py','--out',run/'holdout_manifest','--workers',args.other_workers)
        subprocess.run([PYTHON,'-c',"import sys; sys.path.insert(0,sys.argv[1]); import step6_core as c; m=c.module('segmented_serial_holdout',c.HERE/'220_prepare_gfw_holdouts.py'); m.select_shard('shard_0000',sys.argv[2])",
                        str(E006),str(run/'holdout_manifest_serial')],cwd=CODING.parent,env=env,check=True)
        serial,parallel=run/'holdout_manifest_serial/shards/shard_0000.parquet',run/'holdout_manifest/shards/shard_0000.parquet'
        same=filecmp.cmp(serial,parallel,shallow=False)
        write(run/'holdout_parallel_manifest_probe.json',{'status':'pass' if same else 'failed','serial':record(serial),'parallel':record(parallel),'byte_identical':same})
        if not same:raise ValueError('holdout serial/parallel mismatch')
        execute(E006/'230_run_gfw_holdouts.py','--samples',run/'holdout_manifest/samples.parquet','--prepared',run/'preparation',
                '--out',run/'holdout','--workers',args.other_workers)
        for mode in ['base','wave']:
            execute(E006/'240_bootstrap_gfw_release.py','--mode',mode,'--prepared',run/'preparation','--out',run/f'bootstrap_{mode}','--workers',args.other_workers)
        for tag,workers in [('serial',1),('parallel',4)]:
            execute(E006/'240_bootstrap_gfw_release.py','--mode','wave','--prepared',run/'preparation',
                    '--out',run/f'bootstrap_probe_{tag}','--workers',workers,'--probe-combinations',300)
        a,b=run/'bootstrap_probe_serial/bootstrap.parquet',run/'bootstrap_probe_parallel/bootstrap.parquet'
        same=filecmp.cmp(a,b,shallow=False)
        write(run/'parallel_bootstrap_probe.json',{'serial':str(a),'parallel':str(b),'selected_groups':300,'identical_bytes':same,'sha256':record(b)['sha256']})
        if not same:raise ValueError('bootstrap serial/parallel mismatch')
        # The existing private illustration is a frozen geometry example, not
        # a corrected voyage or a release result; retain its labelled source.
        shutil.copytree(CONTROL/'reachable_example',run/'reachable_example',symlinks=True)
        stamp('waiting_for_annual_bounds')
        while probe_process is not None or (process is not None and process.poll() is None):
            advance_bounds()
            time.sleep(1)
        advance_bounds()
        execute(E006/'250_assemble_gfw_release.py','--run',run)
        port_audit(run)
        execute(E006/'260_validate_gfw_release.py','--run',run)
        execute(E006/'270_document_gfw_release.py','--run',run,'--out',args.review.resolve())
        state.update(state='completed',review=str(args.review.resolve()))
    except BaseException as error:
        state.update(state='failed',error=repr(error))
        if process is not None and process.poll() is None:
            # Keep the independently useful bound census running and identify
            # it explicitly, rather than silently abandoning child processes.
            state['bounds_process_still_running']=process.pid
        if probe_process is not None and probe_process.poll() is None:
            state['bounds_probe_process_still_running']=probe_process.pid
        if command_process is not None and command_process.poll() is None:
            state['independent_command_still_running']={'pid':command_process.pid,'command':command_process.args}
        raise
    finally:
        state.update(wall_seconds=time.monotonic()-started);write(run/'execution.json',state)
        if log:log.close()
        if probe_log:probe_log.close()


if __name__=='__main__':main()
