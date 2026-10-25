"""Compute the input snapshot a gated stage is about to consume.

A CP1 or CP2 decision is bound to the inputs the user actually reviewed, so a gated
stage has to name its inputs and hash them before it asks to be cleared. This module
produces that one number.

The snapshot is the SHA-256 over a canonical, sorted listing of
``<sha256>  <path>`` lines, one per input file. It is deterministic, order independent
and easy to reproduce by hand, and it changes whenever any declared input changes.

Directories are expanded file by file, so replacing a file, deleting one or adding one
all change the snapshot: a replacement changes a line's hash, a deletion removes a line
and an addition adds one. This matters because the data a stage really consumes is
mostly per-bucket directories -- the companion tracks and the AIS buckets the weather
stage reads -- and a gate that only pinned the config and a few scalar assets would
still call a swapped bucket "the batch you approved".

Missing inputs are an error, never a skipped line. A snapshot computed over a partial
input set would let a decision clear a run it was never made about. An empty declared
directory is an error for the same reason.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHUNK = 1 << 22


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_tree(root: Path, subdirs: list[str] | None = None) -> list[Path]:
    """Every file under `root`, or under each named subdirectory of it.

    Used for the per-bucket directories a stage actually consumes. An absent or empty
    directory raises, because a snapshot that silently covers nothing is worse than no
    snapshot at all.
    """
    root = Path(root)
    roots = [root] if not subdirs else [root / name for name in subdirs]
    files: list[Path] = []
    for base in roots:
        if not base.is_dir():
            raise FileNotFoundError(
                f"cannot compute an input snapshot: the declared input directory "
                f"{base} does not exist.")
        found = [p for p in base.rglob("*") if p.is_file()]
        if not found:
            raise FileNotFoundError(
                f"cannot compute an input snapshot: the declared input directory "
                f"{base} is empty, so the snapshot would cover none of the data the "
                f"run consumes.")
        files += found
    return sorted(files)


def snapshot_of(paths: list[Path]) -> tuple[str, list[dict]]:
    """Return (snapshot_sha256, manifest) over the given input files."""
    if not paths:
        raise ValueError("an input snapshot needs at least one input file")
    resolved = [Path(p).resolve() for p in paths]
    missing = [str(p) for p in resolved if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot compute an input snapshot: these declared inputs are missing "
            f"{missing}. A snapshot over a partial input set would let a decision "
            "clear a run it was never made about.")

    entries = sorted(({"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                      for p in resolved), key=lambda e: e["path"])
    blob = "\n".join(f"{e['sha256']}  {e['path']}" for e in entries).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), entries


def stage_input_snapshot(config_path: Path, path_keys: list[str]) -> tuple[str, list[dict]]:
    """Snapshot a stage config plus the input files it names under `path_keys`."""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"stage config not found: {config_path}")
    config = json.loads(config_path.read_text())
    inputs = [config_path]
    for key in path_keys:
        if key not in config:
            raise KeyError(f"stage config {config_path} declares no {key!r}")
        inputs.append(Path(config[key]))
    return snapshot_of(inputs)
