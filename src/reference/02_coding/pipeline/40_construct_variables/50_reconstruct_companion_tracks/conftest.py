"""Frozen-asset fixtures; missing or invalid routing assets fail tests."""
import os
from pathlib import Path
import sys

import pytest

STAGE = Path(__file__).resolve().parent
CODING_ROOT = STAGE.parents[2]
DEFAULT_OCEAN_MASK = CODING_ROOT / "data/frozen_inputs/external/gshhg/ocean_mask_01m.npy"
sys.path.insert(0, str(STAGE))


@pytest.fixture(scope="session")
def ocean_mask_path():
    path = Path(os.environ.get("COMPANION_OCEAN_MASK", DEFAULT_OCEAN_MASK))
    if not path.is_file():
        pytest.fail(
            f"Required frozen ocean mask is missing: {path}. "
            "Use the frozen ocean-mask path or explicitly set "
            "COMPANION_OCEAN_MASK to the same verified asset. Tests must not skip.",
            pytrace=False,
        )
    return path


@pytest.fixture(scope="module")
def ocean_mask(ocean_mask_path):
    from sea_path.ocean_mask import load_ocean_mask
    return load_ocean_mask(ocean_mask_path)


@pytest.fixture(scope="module")
def router(ocean_mask):
    from sea_path.routing import SeaPathRouter
    return SeaPathRouter(ocean_mask)
