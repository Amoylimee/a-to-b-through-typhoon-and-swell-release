#!/usr/bin/env python3
"""E006 census bounds with the already validated E007 wave-support correction.

Both TC and wave bounds are recomputed for the corrected voyage population.
The scientific kernels are reused; only the wave grid-support adapter is
installed in the process-local module. Command-line arguments are E006's.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
E006=HERE.parents[1]/'pipeline/gfw_release'
sys.path.insert(0,str(E006))
import step6_core as core
from wave_support_bounds import install


def main():
    stage=core.module('e007_segmented_bound_stage',E006/'210_bound_gfw_missing_weather.py')
    original=stage.module
    def loader(name,path):
        kernels=original(name,path)
        if Path(path).name=='60_compute_tolerant_bounds.py':install(kernels)
        return kernels
    stage.module=loader
    stage.main()
    # Add explicit provenance for the installed method to the census receipt.
    out=Path(sys.argv[sys.argv.index('--out')+1]).resolve()
    state=__import__('json').loads((out/'status.json').read_text())
    state.update(wave_support_revision=True,segmentation_revision=True,tc_fields_recomputed=True)
    state['code'].extend(core.record(p) for p in [Path(__file__),HERE/'wave_support_bounds.py',HERE/'config/port_segmentation_revision.json',HERE/'config/wave_support_revision.json'])
    core.write_json(out/'status.json',state)


if __name__=='__main__':main()
