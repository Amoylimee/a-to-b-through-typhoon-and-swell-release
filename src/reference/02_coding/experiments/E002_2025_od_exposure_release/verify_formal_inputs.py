#!/usr/bin/env python3
"""Verify the adopted 2025 product manifests and payload hashes read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any


CODING_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = {
    "companion_tracks": CODING_ROOT / "data/processed/products/companion_tracks/2025",
    "voyage_weather_facts": CODING_ROOT / "data/processed/products/voyage_weather_facts/2025",
}
EXPECTED_CANDIDATES = 1_354_168


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_one(root: Path, relative: str, expected: dict[str, Any]) -> dict[str, Any]:
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return {"path": relative, "ok": False, "error": "unsafe manifest path"}
    path = root / "run" / Path(*relative_path.parts)
    if not path.is_file() or path.is_symlink():
        return {"path": relative, "ok": False, "error": "missing, non-file, or symlink"}
    stat = path.stat()
    actual_sha = sha256_file(path)
    expected_bytes = int(expected["bytes"])
    expected_sha = str(expected["sha256"])
    return {
        "path": relative,
        "ok": stat.st_size == expected_bytes and actual_sha == expected_sha,
        "expected_bytes": expected_bytes,
        "actual_bytes": stat.st_size,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "hardlink_count": stat.st_nlink,
    }


def verify_product(name: str, root: Path, workers: int) -> dict[str, Any]:
    manifest_path = root / "adoption_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "adopted" or manifest.get("formal_product_adoption") is not True:
        raise ValueError(f"{name}: manifest is not formally adopted")
    if int(manifest.get("year", 0)) != 2025:
        raise ValueError(f"{name}: unexpected product year")
    if int(manifest.get("candidates", -1)) != EXPECTED_CANDIDATES:
        raise ValueError(f"{name}: candidate count differs from Contract")
    entries = manifest.get("source_files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{name}: source_files manifest missing")

    total_bytes = sum(int(value["bytes"]) for value in entries.values())
    if total_bytes != int(manifest["source_bytes"]):
        raise ValueError(f"{name}: source_bytes does not close over manifest files")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_verify_one, root, relative, expected)
            for relative, expected in entries.items()
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed % 500 == 0 or completed == len(futures):
                print(f"{name}: {completed:,}/{len(futures):,} payload files checked", flush=True)

    bad = [result for result in results if not result.get("ok")]
    links = [result["path"] for result in results if result.get("hardlink_count", 1) > 1]
    return {
        "product": name,
        "root": str(root),
        "adoption_manifest_sha256": manifest_sha,
        "state": manifest["state"],
        "formal_product_adoption": manifest["formal_product_adoption"],
        "year": manifest["year"],
        "candidates": manifest["candidates"],
        "shards": manifest["shards"],
        "manifest_payload_files": len(entries),
        "expected_payload_bytes": total_bytes,
        "actual_payload_bytes": sum(int(x.get("actual_bytes", 0)) for x in results),
        "matched_payload_files": len(results) - len(bad),
        "mismatch_count": len(bad),
        "hardlinked_payload_files": links,
        "failures": bad[:50],
        "ok": not bad and not links,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise SystemExit("workers must be in [1, 16]")
    report = {
        "scope": "read-only formal 2025 adopted product verification",
        "workers": args.workers,
        "products": [
            verify_product(name, root, args.workers) for name, root in PRODUCTS.items()
        ],
    }
    report["ok"] = all(product["ok"] for product in report["products"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
