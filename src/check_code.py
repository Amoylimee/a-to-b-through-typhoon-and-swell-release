"""Check packaged source integrity; optionally run self-contained GFW tests."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REF = "src/reference/02_coding/"
TEST_PATHS = [
    REF + "pipeline/gfw_release/tests",
    REF + "pipeline/gfw_monthly_release/tests",
    REF + "pipeline/gfw_followup/tests",
    REF + "pipeline/40_construct_variables/60_construct_voyage_weather_facts/tests/test_tropical_cyclone.py",
    REF + "pipeline/40_construct_variables/60_construct_voyage_weather_facts/tests/test_era5_match.py",
    REF + "pipeline/40_construct_variables/60_construct_voyage_weather_facts/tests/test_hour_segments.py",
    REF + "pipeline/40_construct_variables/50_reconstruct_companion_tracks/tests/test_full_ten_minute_candidate.py",
    REF + "experiments/E002_2025_od_exposure_release/tests/test_aggregation.py",
    REF + "pipeline/00_environment_and_config/tests/test_ten_minute_review_gates.py",
    "tests",
]
# These exact-source integration tests need the frozen annual execution receipt,
# which is not distributed. Keep their assertions intact in the reference tree.
EXTERNAL_INPUT_TESTS = [
    REF + "pipeline/gfw_release/tests/test_step5_authorization.py::test_user_authorization_binds_the_actual_gfw_baseline",
    REF + "pipeline/gfw_release/tests/test_step5_authorization.py::test_changed_weather_configuration_is_rejected",
]


def check_integrity():
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    copied = links = 0
    for item in manifest["files"]:
        path = ROOT / item["destination"]
        expected = item.get("release_sha256", item["source_sha256"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source checksum mismatch: " + str(path.relative_to(ROOT)))
        if "symlink_target" in item:
            if not path.is_symlink() or os.readlink(path) != item["symlink_target"]:
                raise ValueError("Stage entry point changed: " + str(path))
            if not path.resolve().is_relative_to((ROOT / "src/reference").resolve()):
                raise ValueError("Stage entry point escapes the reference tree: " + str(path))
            links += 1
        copied += 1
    ignored = {".git", ".venv", "__pycache__", ".runs", "inputs", "build"}
    paths = [p for p in ROOT.rglob("*.py") if not set(p.relative_to(ROOT).parts) & ignored]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"PASS: {copied} copied-file hashes, {links} stage symlinks and {len(paths)} Python sources.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true", help="also run the selected tests; requires environment.yml")
    args = parser.parse_args()
    check_integrity()
    if args.tests:
        print("Excluding two annual-input integration tests (frozen execution receipt not bundled).", flush=True)
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        command = [sys.executable, "-m", "pytest", "-q", *TEST_PATHS]
        command += ["--deselect=" + node for node in EXTERNAL_INPUT_TESTS]
        raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
