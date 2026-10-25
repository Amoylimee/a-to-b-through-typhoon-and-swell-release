#!/usr/bin/env python3
"""Bind GFW stage-5 facts to original voyages and GFW-derived city endpoints."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from step6_core import *


def prepare(out: Path) -> dict:
    started = time.perf_counter()
    out = inside_runs(out)
    out.mkdir(parents=True, exist_ok=False)
    source = binding()
    state = source["weather_state"]
    columns = ["candidate_id", "mmsi", SHIP, "source_shard", "departure_time", "arrival_time", *KEY]
    pieces = []
    inputs = []
    for shard in SHARDS:
        receipt = json.loads((source["weather_run"] / "shards" / shard / "status.json").read_text())
        item = receipt["sources"]["summary"]
        if record(Path(item["path"])) != item:
            raise ValueError(f"GFW voyage summary changed: {shard}")
        pieces.append(pd.read_parquet(item["path"], columns=columns))
        inputs.append(item)
    candidates = pd.concat(pieces, ignore_index=True)
    weather = pd.read_parquet(source["annual"]["path"])
    panel = candidates.merge(weather, on="candidate_id", how="outer", validate="one_to_one", indicator=True)
    if not panel._merge.eq("both").all() or len(panel) != source["counts"]["candidates"] \
            or panel[["candidate_id", SHIP, *KEY]].isna().any().any():
        raise ValueError("GFW candidate/weather join does not close")
    panel.drop(columns="_merge", inplace=True)
    numeric_close((panel.arrival_time-panel.departure_time).dt.total_seconds()/3600,
                  panel.voyage_hours, "original voyage denominator")
    grouped = aggregate.aggregate_panel(panel)
    # The inherited initial U1 helper also returns historical position-hour rates;
    # retain them only in private preparation, never in a release table.
    panel.to_parquet(out / "voyage_panel.parquet", index=False)
    grouped.to_parquet(out / "all_combinations.parquet", index=False)
    cities = pd.read_csv(source["cities"]["path"], usecols=["city_id", "port_lat", "port_lon"])
    if cities.city_id.duplicated().any():
        raise ValueError("GFW city coordinate table is ambiguous")
    ports = panel[["candidate_id", "source_shard", "departure_time", "arrival_time", *KEY]].copy()
    for role, city in (("departure", "origin_city"), ("arrival", "dest_city")):
        ref = cities.rename(columns={"city_id": city, "port_lat": f"{role}_port_lat", "port_lon": f"{role}_port_lon"})
        ports = ports.merge(ref, on=city, how="left", validate="many_to_one")
    ports["port_endpoints_connected"] = np.isfinite(ports[["departure_port_lat", "departure_port_lon",
                                                            "arrival_port_lat", "arrival_port_lon"]]).all(axis=1)
    ports.to_parquet(out / "port_endpoints.parquet", index=False)
    totals = {c: float(panel[c].sum()) for c in aggregate.WEATHER_SUMS}
    for c, value in totals.items():
        numeric_close(value, state["totals"][c], c)
    result = {"status": "pass", "step": 6, "year": 2025, "population": "E006_gfw_presence_2025",
              "candidates": len(panel), "combinations": len(grouped),
              "release_combinations_ge5": int(grouped.voyages.ge(5).sum()),
              "release_voyages_ge5": int(grouped.loc[grouped.voyages.ge(5), "voyages"].sum()),
              "connected_port_endpoints": int(ports.port_endpoints_connected.sum()),
              "totals": totals, "source_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "authorization": source["authorization"], "annual_weather": source["annual"],
              "weather_validation": source["validation"], "gfw_city_table": source["cities"],
              "candidate_summaries": inputs, "network_requests": 0, "products_modified": 0,
              "adopted": False, "elapsed_seconds": time.perf_counter()-started,
              "outputs": {p.name: record(p) for p in out.glob("*.parquet")}}
    write_json(out / "preparation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = prepare(parser.parse_args().out)
    print(json.dumps({k: result[k] for k in ("status", "candidates", "combinations", "release_combinations_ge5", "release_voyages_ge5", "connected_port_endpoints", "elapsed_seconds")}, indent=2))
