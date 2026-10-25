"""Read the accepted route-free 2025 candidate root with manifest closure checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather


HERE = Path(__file__).resolve().parent
CODING_ROOT = HERE.parents[1]
CANDIDATE_ROOT_MANIFEST = (
    CODING_ROOT
    / ".runs/2026-09-10_1030_E001-2025-candidate-root/assembly_manifest.json"
)
EXPECTED_MANIFEST_SHA256 = "b4802787db3da22b3ac1955a688b48c44a126cf815f84592290d5adc7a14a2a0"
EXPECTED_CANDIDATES = 1_354_168
EXPECTED_PARTS = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidate_root(
    manifest_path: Path,
    columns: tuple[str, ...],
    *,
    expected_manifest_sha256: str | None = None,
    expected_rows: int | None = None,
    expected_parts: int | None = None,
) -> pd.DataFrame:
    """Read listed 30e parts and fail closed on manifest, schema, or key drift."""

    manifest_path = Path(manifest_path).resolve()
    if expected_manifest_sha256 and sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError(f"candidate-root manifest hash mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    year = manifest.get("years", {}).get("2025", {})
    records = year.get("parts", [])
    row_count = year.get("row_count")
    part_count = year.get("part_count")
    if (
        manifest.get("status") != "pass"
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count <= 0
        or not isinstance(records, list)
        or isinstance(part_count, bool)
        or not isinstance(part_count, int)
        or part_count <= 0
    ):
        raise ValueError("candidate-root manifest is not a passing 2025 assembly")
    if year.get("candidate_enumeration_mode") != "adjacent_commercial_episodes":
        raise ValueError("candidate-root grain is not adjacent commercial episodes")
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError("candidate-root row count changed")
    if expected_parts is not None and len(records) != expected_parts:
        raise ValueError("candidate-root part count changed")
    if part_count != len(records):
        raise ValueError("candidate-root manifest part inventory is inconsistent")
    part_rows = [record.get("rows") for record in records]
    if any(
        isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
        for rows in part_rows
    ):
        raise ValueError("candidate-root part inventory has a missing or invalid row count")
    if sum(part_rows) != row_count:
        raise ValueError("candidate-root part rows do not sum to the manifest total")

    frames: list[pd.DataFrame] = []
    for record in records:
        available = set(record.get("columns", []))
        missing = set(columns) - available
        if missing:
            raise ValueError(f"candidate-root part lacks requested columns: {sorted(missing)}")
        if not isinstance(record.get("path"), str) or not record["path"].strip():
            raise ValueError("candidate-root part inventory has a missing path")
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise ValueError("candidate-root part inventory has a missing SHA-256")
        path = Path(record["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"candidate-root part hash mismatch: {path}")
        frame = feather.read_table(path, columns=list(columns), memory_map=True).to_pandas()
        if len(frame) != record.get("rows"):
            raise ValueError(f"candidate-root part row count changed: {path}")
        frames.append(frame)

    candidates = pd.concat(frames, ignore_index=True)
    if len(candidates) != row_count:
        raise ValueError("candidate-root rows do not close to the manifest")
    if candidates["candidate_id"].isna().any() or candidates["candidate_id"].duplicated().any():
        raise ValueError("candidate-root candidate_id is null or duplicated")
    for column in ("candidate_id", "mmsi", "vessel_uid_clean"):
        candidates[column] = candidates[column].astype("string")
    candidates["departure_year"] = pd.to_datetime(
        candidates["departure_time"], utc=True
    ).dt.year.astype("int16")
    return candidates


def load_accepted_2025_candidates(columns: tuple[str, ...]) -> pd.DataFrame:
    """Read the frozen 30e 2025 candidate root used by the accepted products."""

    return read_candidate_root(
        CANDIDATE_ROOT_MANIFEST,
        columns,
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        expected_rows=EXPECTED_CANDIDATES,
        expected_parts=EXPECTED_PARTS,
    )
