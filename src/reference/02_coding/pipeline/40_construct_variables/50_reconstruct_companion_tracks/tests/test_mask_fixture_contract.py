"""Exercise pytest itself so a skipped guard cannot masquerade as a pass."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

STAGE = Path(__file__).resolve().parents[1]
GUARD = ('sea_path/tests/test_sea_path_routing.py::TestPositionsInsideOneMaskCell'
         '::test_identical_positions_remain_a_legal_zero_length_route')


def run_guard(mask=None):
    env = os.environ.copy()
    env.pop('COMPANION_OCEAN_MASK', None)
    if mask is not None:
        env['COMPANION_OCEAN_MASK'] = str(mask)
    return subprocess.run([sys.executable, '-m', 'pytest', '-q', GUARD],
                          cwd=STAGE, env=env, text=True, capture_output=True)


def test_default_command_executes_real_mask_guard():
    result = run_guard()
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout and 'skipped' not in result.stdout


def test_missing_mask_fails_instead_of_skipping(tmp_path):
    result = run_guard(tmp_path/'missing.npy')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Required frozen ocean mask is missing' in result.stdout
    assert '1 error' in result.stdout and 'skipped' not in result.stdout


def test_invalid_mask_fails_instead_of_skipping(tmp_path):
    mask = tmp_path/'invalid.npy'
    np.save(mask, np.ones((2, 2), dtype=bool))
    result = run_guard(mask)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'OceanMaskIntegrityError' in result.stdout
    assert '1 error' in result.stdout and 'skipped' not in result.stdout
