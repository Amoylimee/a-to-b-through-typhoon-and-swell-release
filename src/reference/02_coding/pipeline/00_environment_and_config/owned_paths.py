"""Resolve historical input provenance to the repository's owned data copies.

Historical manifests remain byte-for-byte unchanged. A registered source path is
translated to its current local copy; missing local copies never fall back to a
source repository. Code provenance and archive-only paths are not input aliases.
"""
from functools import lru_cache
import json
from pathlib import Path

CODING = Path(__file__).resolve().parents[2]

@lru_cache(maxsize=1)
def mappings():
    registry = json.loads((Path(__file__).parent / 'config/frozen_inputs_2025.json').read_text())
    result = []
    for group in registry['groups']:
        for item in group['items']:
            source = Path(registry['source_repos'][item.get('source_repo', group.get('source_repo'))]['root']) / item['src']
            target = CODING / 'data' / item['dst']
            result.append((source, target))
    return sorted(result, key=lambda pair: len(str(pair[0])), reverse=True)

def local_input(path):
    path = Path(path)
    if path.is_relative_to(CODING): return path
    for source, target in mappings():
        if path == source or path.is_relative_to(source):
            result = target / path.relative_to(source)
            if not result.exists(): raise FileNotFoundError('Owned input is missing: ' + str(result))
            if not result.resolve().is_relative_to(CODING.resolve()):
                raise ValueError('Input still resolves outside this repository: ' + str(result))
            return result
    registry = json.loads((Path(__file__).parent / 'config/frozen_inputs_2025.json').read_text())
    if any(path.is_relative_to(Path(r['root'])) for r in registry['source_repos'].values()):
        raise ValueError('Historical path has no registered owned input: ' + str(path))
    return path
