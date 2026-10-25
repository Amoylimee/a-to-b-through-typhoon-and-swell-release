"""Resolve the facts described by a guarded run command.

A gate that inspects a fixed configuration while the run is free to select another one
does not gate anything: it approves A and runs B. The guarded operation's selected
config can only be known by parsing the same argv the entry will parse.

This module does exactly that, once, so the gated stage entry and the seal inside the
migrated upstream file derive identical facts from the identical command line. The
facts are:

  config_path              the config the run will actually load
  inputs                   every input file that config and command line name,
                           including the per-bucket directories the stage really reads
  input_snapshot_sha256    the snapshot over those inputs
  year                     the departure year this run will process
  candidate_scope_sha256   the candidate set, i.e. the voyage table both gates are about

`year` and `candidate_scope_sha256` exist so the approved scope can be compared with the
run rather than merely being present in the decision record.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from input_snapshot import expand_tree, sha256_file, snapshot_of

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent


class RunFactsUnresolved(Exception):
    """The run's own inputs could not be determined, so nothing may be cleared."""


@dataclass(frozen=True)
class RunFacts:
    operation: str
    config_path: Path
    inputs: tuple[Path, ...]
    input_snapshot_sha256: str
    year: int
    candidate_scope_sha256: str
    candidate_scope_path: Path

    def as_dict(self) -> dict:
        return {"operation": self.operation,
                "config_path": str(self.config_path),
                "input_snapshot_sha256": self.input_snapshot_sha256,
                "year": self.year,
                "candidate_scope_sha256": self.candidate_scope_sha256,
                "candidate_scope_path": str(self.candidate_scope_path),
                "input_count": len(self.inputs)}


# Per operation: how its entry selects a config, which further inputs the command line
# names, and which config keys carry the year and candidate set. Retired 64-bucket
# entrypoints and their route-facts configs are not defaults here.
CONTRACTS: dict[str, dict] = {
    "companion_reconstruction": {
        "stage": PIPELINE / "40_construct_variables" / "50_reconstruct_companion_tracks",
        "config_flag": "--config",
        "config_default": None,          # upstream marks it required
        # Both are required=True upstream, so a run that omits them cannot be described.
        "extra_input_flags": ["--quality-audit", "--pilot-status"],
        "required_input_flags": ["--quality-audit", "--pilot-status"],
        "year_key": "departure_year",
        "candidate_key": "voyage_table",         # relative to source_project
        "config_file_keys": ["ocean_mask_path"],  # absolute paths in the config
        "config_project_keys": ["voyage_table", "coverage_manifest"],
        # Only summary.json of the acquisition run is opened by prepare(); the rest of
        # that 234 GB store is not read here, so it is not declared as an input.
        "config_project_files": [("source_target_root", "anchor_target_manifest.json"),
                                 ("gfw_run", "summary.json")],
        # prepare() reads candidate_scope/bucket_NNNN.parquet and gaps/bucket_NNNN.parquet.
        "config_project_trees": [("source_target_root", ["candidate_scope", "gaps"])],
    },
    "weather_matching": {
        "stage": PIPELINE / "40_construct_variables" / "60_construct_voyage_weather_facts",
        "config_flag": "--config",
        "config_default": None,
        "extra_input_flags": [],
        "year_key": "year",
        "candidate_key": "voyages",
        "config_file_keys": ["ocean_mask_path", "ibtracs_path",
                             "ibtracs_calendar_coverage_audit"],
        "config_project_keys": ["voyages"],
        "config_project_files": [],
        # The data the weather computation really consumes: the companion tracks it
        # samples and the AIS buckets it reads per bucket.
        "config_project_trees": [("companion_product", None), ("ais_bucket_root", None)],
        "optional_file_keys": ["position_supplement"],
        "wave_series": ("wave_directory", "wave_filename_template", range(1, 13)),
    },
}


def _parse(argv: list[str], contract: dict) -> tuple[Path | None, list[Path], set[str]]:
    """Read the same flags the upstream entry reads; ignore everything else."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(contract["config_flag"], type=Path, default=None)
    for flag in contract["extra_input_flags"]:
        parser.add_argument(flag, type=Path, default=None)
    known, _ = parser.parse_known_args(argv)

    config = getattr(known, contract["config_flag"].lstrip("-").replace("-", "_"))
    extras, given = [], set()
    for flag in contract["extra_input_flags"]:
        value = getattr(known, flag.lstrip("-").replace("-", "_"))
        if value is not None:
            extras.append(Path(value))
            given.add(flag)
    return config, extras, given


def resolve_run_facts(operation: str, argv: list[str]) -> RunFacts:
    """Derive the facts of the run `argv` describes, or refuse to produce any."""
    if operation not in CONTRACTS:
        raise ValueError(f"unknown operation {operation}")
    contract = CONTRACTS[operation]

    config_path, extras, given_flags = _parse(list(argv), contract)
    missing_flags = [f for f in contract.get("required_input_flags", [])
                     if f not in given_flags]
    if missing_flags:
        raise RunFactsUnresolved(
            f"{operation}: the upstream entry requires {missing_flags} and they were "
            f"not given, so the inputs this run would read are unknown.")
    if config_path is None:
        if contract["config_default"] is None:
            raise RunFactsUnresolved(
                f"{operation}: the run selects its config with "
                f"{contract['config_flag']} and none was given, so the config that "
                f"would actually be used is unknown. A gate cannot approve an unknown "
                f"configuration.")
        config_path = contract["stage"] / contract["config_default"]
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise RunFactsUnresolved(f"{operation}: config not found: {config_path}")

    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise RunFactsUnresolved(f"{operation}: config is not valid JSON: {error}")

    missing_keys = [k for k in (contract["year_key"], contract["candidate_key"],
                                "source_project") if k not in config]
    if missing_keys:
        raise RunFactsUnresolved(
            f"{operation}: config {config_path} is missing {sorted(missing_keys)}")

    project = Path(config["source_project"])
    inputs: list[Path] = [config_path, *extras]
    for key in contract["config_file_keys"]:
        if key not in config:
            raise RunFactsUnresolved(f"{operation}: config declares no {key!r}")
        inputs.append(Path(config[key]))
    for key in contract["config_project_keys"]:
        inputs.append(project / config[key])
    for root_key, name in contract["config_project_files"]:
        if root_key not in config:
            raise RunFactsUnresolved(f"{operation}: config declares no {root_key!r}")
        inputs.append(project / config[root_key] / name)
    for key in contract.get("optional_file_keys", []):
        if config.get(key):
            inputs.append(Path(config[key]))
    for root_key, subdirs in contract.get("config_project_trees", []):
        if root_key not in config:
            raise RunFactsUnresolved(f"{operation}: config declares no {root_key!r}")
        try:
            inputs += expand_tree(project / config[root_key], subdirs)
        except FileNotFoundError as error:
            raise RunFactsUnresolved(f"{operation}: {error}")
    if "wave_series" in contract:
        dir_key, template_key, months = contract["wave_series"]
        wave_dir = Path(config[dir_key])
        inputs += [wave_dir / config[template_key].format(month=m) for m in months]

    try:
        snapshot, _ = snapshot_of(inputs)
    except FileNotFoundError as error:
        # One refusal type for callers: a run whose declared inputs are not all present
        # cannot be described, so it cannot be cleared.
        raise RunFactsUnresolved(f"{operation}: {error}")

    year = config[contract["year_key"]]
    if not isinstance(year, int):
        raise RunFactsUnresolved(
            f"{operation}: config {contract['year_key']} is {year!r}, expected an int")

    candidate_path = project / config[contract["candidate_key"]]
    if not candidate_path.is_file():
        raise RunFactsUnresolved(
            f"{operation}: the candidate set {candidate_path} does not exist, so the "
            f"approved candidate scope cannot be compared with the run.")

    return RunFacts(operation=operation, config_path=config_path,
                    inputs=tuple(inputs), input_snapshot_sha256=snapshot,
                    year=year, candidate_scope_sha256=sha256_file(candidate_path),
                    candidate_scope_path=candidate_path)
