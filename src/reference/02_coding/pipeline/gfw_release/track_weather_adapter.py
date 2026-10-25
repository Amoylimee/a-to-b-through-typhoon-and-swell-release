"""Track/weather adapter used by the accepted GFW voyage construction."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import step6_core as core

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
WEATHER_STAGE = CODING / 'pipeline/40_construct_variables/60_construct_voyage_weather_facts'
WEATHER_CONFIG = WEATHER_STAGE / 'config/2025_ten_minute_weather.json'
load = core.module
tracks = load('gfw_accepted_tracks', HERE / '160_build_gfw_ten_minute_tracks.py')
weather_check = load('gfw_accepted_weather_check', HERE / '190_validate_gfw_weather.py')
record, write_json, verify = tracks.record, tracks.write_json, tracks.verify

def weather(task: dict, clearance: dict) -> dict:
    """Frozen step-5 matcher and E006's shard validator on one trimmed-voyage track shard."""
    step4 = json.loads((Path(task["output"]) / "status.json").read_text())
    outputs = {Path(item["path"]).name: item for item in step4["outputs"]}
    sources = {"track": outputs["ten_minute_complete_candidate.parquet"], "summary": outputs["candidate_summary.parquet"]}
    for item in sources.values():
        verify(item)
    matcher = load("e007_weather_matcher", WEATHER_STAGE / "40_match_ten_minute_weather_shard.py")
    out = Path(task["weather"])
    namespace = SimpleNamespace(output=out, config=WEATHER_CONFIG, shard=task["shard"], max_candidates=None)
    result = matcher.match_validated_weather_inputs(namespace, clearance, sources)
    result.update(population="E007_gfw_speed_trimmed_2025", step=5, interpolation_gap_max_hours=48.0, adopted=False)
    write_json(out / "status.json", result)
    receipt = weather_check.validate_shard(out)
    return {"shard": task["shard"], "candidates": result["candidates"], "rows": result["rows"],
            "validation": receipt["status"], "seconds": result["elapsed_seconds"]}
