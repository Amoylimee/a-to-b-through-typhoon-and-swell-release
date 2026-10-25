"""E008 monthly public tables from the adopted GFW voyage-level estimates.

The accepted aggregation and bootstrap kernels are imported, not reimplemented.
The interval assembly below follows 250_assemble_gfw_release.py and is checked
against every annual accepted record before a monthly table can be written.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPELINE = ROOT / "02_coding/pipeline/gfw_release"
sys.path.insert(0, str(PIPELINE))
import step6_core as core

ASSEMBLY = core.module("e008_accepted_assembly", PIPELINE / "250_assemble_gfw_release.py")
BOOT = core.module("e008_accepted_bootstrap", PIPELINE / "240_bootstrap_gfw_release.py")
DOC = core.module("e008_accepted_document", PIPELINE / "270_document_gfw_release.py")
SEEDS = core.module("e008_accepted_seed", core.E002 / "30_release/70_clip_full_voyage_wave.py")
CITY = ["origin_city", "dest_city", "shiptype_group"]
COUNTRIES = ["origin_country", "dest_country"]
MONTH = "departure_month"
COUNTRY_KEY = [*COUNTRIES, "shiptype_group", MONTH]
CODED_KEY = [*COUNTRIES, "origin_city_code", "dest_city_code", "shiptype_group", MONTH]
SHIP = core.SHIP
MEASURES = ASSEMBLY.RELEASE_COLUMNS[3:]
CI = [c for c in MEASURES if "_ci95_" in c]
POINT_FIELDS = [c for c in MEASURES if c not in CI]
BOUND_SUMS = ["main_unbounded_hours", "main_wave_counted_hours", "main_wave_min_integral",
              "main_wave_max_integral", "main_tc300_possible_hours", "main_tc500_possible_hours"]
SUMS = [*core.aggregate.WEATHER_SUMS, *BOUND_SUMS]
ADDITIVE = [*SUMS, "voyages", "voyages_with_tc300", "voyages_with_tc500"]
BOOT_SUMS = list(dict.fromkeys([*BOOT.BASE_COLUMNS, *BOOT.WAVE_COLUMNS]))
METRICS = ["mean_swh_matched_m", "share_wave_hours_ge4m", "tc300_at_least_pct", "tc500_at_least_pct"]
TEXT_FIELDS = {"wave_mean_missing_reason"}
BOOL_FIELDS = {"single_ship", "wave_mean_was_constrained", "sampling_interval_unavailable_fewer_than_3_ships"}
COUNT_FIELDS = {"voyages", "ships", "voyages_with_tc300", "voyages_with_tc500"}
SHIP_TYPES = {"bulk_carrier", "container", "crude_oil_tanker", "gas_tanker", "general_cargo",
              "product_chemical_tanker", "reefer", "ro_ro_vehicle", "other_retained"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def note(message):
    print(datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"), message, flush=True)


def aggregate(panel, keys):
    # This imported kernel exposes its grouping columns as a module constant.
    # Restore that constant even on failure; no accepted source file is changed.
    previous = core.aggregate.GROUP_COLUMNS
    try:
        core.aggregate.GROUP_COLUMNS = list(keys)
        g = core.aggregate.aggregate_panel(panel)
    finally:
        core.aggregate.GROUP_COLUMNS = previous
    bounds = panel.groupby(keys, observed=True, sort=True, dropna=False)[BOUND_SUMS].sum().reset_index()
    g = g.merge(bounds, on=keys, validate="one_to_one")
    ratio = core.safe_ratio
    g["unbounded_hours"] = g.main_unbounded_hours
    g["wave_unknown_counted_hours"] = g.main_wave_counted_hours
    denominator = g.wave_matched_hours + g.main_wave_counted_hours
    g["mean_swh_min_possible_m"] = ratio(g.wave_integral_m_hours + g.main_wave_min_integral, denominator)
    g["mean_swh_max_possible_m"] = ratio(g.wave_integral_m_hours + g.main_wave_max_integral, denominator)
    g["mean_swh_possible_range_m"] = g.mean_swh_max_possible_m - g.mean_swh_min_possible_m
    g["wave_bound_hours"] = denominator
    g["wave_bound_coverage"] = ratio(denominator, g.voyage_hours)
    g["wave_unboundable_hours"] = g.voyage_hours - denominator
    g["mean_swh_constrained_m"] = np.clip(g.mean_swh_matched_m, g.mean_swh_min_possible_m, g.mean_swh_max_possible_m)
    g["mean_swh_constrained_shift_m"] = g.mean_swh_constrained_m - g.mean_swh_matched_m
    g["wave_mean_was_constrained"] = g.mean_swh_constrained_shift_m.abs().gt(1e-9).astype("boolean")
    g.loc[g.mean_swh_matched_m.isna(), "wave_mean_was_constrained"] = pd.NA
    g["wave_mean_missing_reason"] = pd.Series(pd.NA, index=g.index, dtype="string")
    none = g.wave_matched_hours.eq(0)
    g.loc[none & denominator.gt(0), "wave_mean_missing_reason"] = "no_matched_wave_hours_reachable_bounds_available"
    g.loc[denominator.eq(0), "wave_mean_missing_reason"] = "no_matched_or_reachable_wave_hours"
    for r in (300, 500):
        prefix = f"tc{r}"
        g[f"{prefix}_hours_at_least"] = g[f"tc{r}_positive_snapshot_hours"]
        g[f"{prefix}_bounded_possible_hours"] = g[f"main_tc{r}_possible_hours"] - g.main_unbounded_hours
        g[f"{prefix}_hours_at_most"] = g[f"{prefix}_hours_at_least"] + g[f"main_tc{r}_possible_hours"]
        g[f"{prefix}_at_least_pct"] = 100 * ratio(g[f"{prefix}_hours_at_least"], g.voyage_hours)
        g[f"{prefix}_at_most_pct"] = 100 * ratio(g[f"{prefix}_hours_at_most"], g.voyage_hours)
    g["single_ship"] = g.ships.eq(1)
    g["sampling_interval_unavailable_fewer_than_3_ships"] = g.ships.lt(3)
    return g


def compare(left, right, keys, fields, label, *, atol=1e-7):
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(label + ": duplicate keys")
    joined = left[keys + fields].merge(right[keys + fields], on=keys, how="outer",
                                     validate="one_to_one", suffixes=("", "_ref"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError(label + ": key sets differ")
    worst = 0.0
    for field in fields:
        a, b = joined[field], joined[field + "_ref"]
        if field in TEXT_FIELDS | BOOL_FIELDS:
            if not a.astype("string").fillna("").equals(b.astype("string").fillna("")):
                raise ValueError(label + ": " + field)
        else:
            x, y = a.to_numpy(float), b.to_numpy(float)
            tol = 0 if field in COUNT_FIELDS else atol
            if not np.allclose(x, y, rtol=1e-12 if tol else 0, atol=tol, equal_nan=True):
                raise ValueError(label + ": " + field)
            difference = np.abs(x - y)
            if np.isfinite(difference).any():
                worst = max(worst, float(np.nanmax(difference)))
    return {"status": "pass", "rows": len(joined), "fields": len(fields), "max_absolute_difference": worst}


def adds_up(fine, coarse, keys, label):
    rolled = fine.groupby(keys, observed=True, sort=True)[ADDITIVE].sum().reset_index()
    return compare(rolled, coarse, keys, ADDITIVE, label, atol=1e-6)


def city_codes(cities, path):
    cities = sorted(set(cities))
    path = Path(path)
    if path.exists():
        mapping = pd.read_csv(path)
        if set(mapping.city_id) != set(cities):
            raise ValueError("Frozen city-code map has a different city population")
    else:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        rng = np.random.default_rng(secrets.randbits(128))
        rows = []
        series = pd.Series(cities)
        for country, members in series.groupby(series.str[:3], sort=True):
            if len(members) > 9999:
                raise ValueError("Country exceeds four-digit city-code capacity")
            order = rng.permutation(len(members)) + 1
            rows.extend({"city_id": city, "city_code": f"{country}_c{rank:04d}"}
                        for city, rank in zip(members, order))
        mapping = pd.DataFrame(rows)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            mapping.to_csv(f, index=False)
    if mapping.city_id.duplicated().any() or mapping.city_code.duplicated().any():
        raise ValueError("City-code map is not one to one")
    if not mapping.city_code.str.fullmatch(r"[A-Z]{3}_c[0-9]{4}").all():
        raise ValueError("Malformed city code")
    if not mapping.city_id.str[:3].equals(mapping.city_code.str[:3]):
        raise ValueError("City-code country mismatch")
    return mapping


def bootstrap_one(key, block, keys):
    block = block.sort_values(SHIP)
    row = dict(zip(keys, key))
    row["ships"] = len(block)
    for mode, columns in (("base", BOOT.BASE_COLUMNS), ("wave", BOOT.WAVE_COLUMNS)):
        result = BOOT.bootstrap_group(key, block[columns].to_numpy(float), mode, SEEDS)
        for field, value in result.items():
            if field in METRICS or "_ci95_" in field or field.endswith("_valid_bootstrap_draws"):
                row[field.replace("mean_swh_matched_m_ci95_", "mean_swh_matched_ci95_")] = value
    return row


def bootstrap_part(path, keys):
    path = Path(path)
    target = path.with_name(path.stem + "_bootstrap.parquet")
    receipt = target.with_suffix(".json")
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["input_sha256"] != digest(path) or old["output_sha256"] != digest(target):
            raise ValueError("Bootstrap checkpoint changed")
        return old
    started = time.perf_counter()
    frame = pd.read_parquet(path)
    rows = [bootstrap_one(key, block, keys) for key, block in frame.groupby(keys, observed=True, sort=True)]
    result = pd.DataFrame(rows)
    result.to_parquet(target, index=False)
    report = {"part": path.stem, "records": len(result), "output": str(target),
              "input_sha256": digest(path), "output_sha256": digest(target),
              "seconds": round(time.perf_counter() - started, 3)}
    write_json(receipt, report)
    return report


def bootstrap(panel, table, keys, directory, workers):
    directory.mkdir(parents=True, exist_ok=True)
    eligible = table.loc[table.ships.ge(3), keys]
    sub = panel.merge(eligible, on=keys, how="inner", validate="many_to_one")
    clusters = sub.groupby([*keys, SHIP], observed=True, sort=True)[BOOT_SUMS].sum().reset_index()
    group_numbers = clusters.groupby(keys, observed=True, sort=True).ngroup()
    clusters["_part"] = group_numbers % workers
    paths = []
    for number, block in clusters.groupby("_part", sort=True):
        p = directory / f"part_{number:03d}.parquet"
        contents = block.drop(columns="_part")
        if p.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(p), contents.reset_index(drop=True), check_exact=True)
        else:
            contents.to_parquet(p, index=False)
        paths.append(p)
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(bootstrap_part, p, keys) for p in paths]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            note(f"{directory.name}: {len(reports)}/{len(paths)} partitions complete")
    if reports:
        results = pd.concat([pd.read_parquet(r["output"]) for r in reports], ignore_index=True)
        check = table.merge(results, on=keys, how="inner", validate="one_to_one", suffixes=("", "_boot"))
        if len(check) != len(eligible) or not check.ships.eq(check.ships_boot).all():
            raise ValueError("Bootstrap group or vessel count differs")
        for metric in METRICS:
            core.numeric_close(check[metric], check[metric + "_boot"], "bootstrap " + metric)
        table = table.merge(results[keys + CI], on=keys, how="left", validate="one_to_one")
        audit = {metric: {"records_with_fewer_than_1000_valid_draws": int(results[metric + "_valid_bootstrap_draws"].lt(1000).sum()),
                          "minimum_valid_draws": int(results[metric + "_valid_bootstrap_draws"].min())}
                 for metric in METRICS}
    else:
        for field in CI:
            table[field] = np.nan
        audit = {}
    if table.loc[table.ships.lt(3), CI].notna().any().any():
        raise ValueError("Sampling interval supplied for fewer than three ships")
    return table, {"eligible_records": len(eligible), "replicates": 1000, "seed": core.SEED,
                   "minimum_ships": 3, "valid_draws": audit, "partitions": reports}


def check_values(g, keys):
    tol = 1e-7
    if g.duplicated(keys).any() or g.voyages.lt(1).any() or g.ships.lt(1).any():
        raise ValueError("Invalid record key or sample size")
    if g.ships.gt(g.voyages).any() or g.voyage_hours.le(0).any():
        raise ValueError("Invalid ship count or duration")
    for c in ["position_coverage", "wave_coverage", "wave_bound_coverage", "share_wave_hours_ge4m"]:
        if ((g[c] < -tol) | (g[c] > 1 + tol)).any():
            raise ValueError("Fraction outside [0, 1]: " + c)
    if (g.mean_swh_min_possible_m > g.mean_swh_max_possible_m + tol).any():
        raise ValueError("Wave range reversed")
    for r in (300, 500):
        lo, hi = g[f"tc{r}_at_least_pct"], g[f"tc{r}_at_most_pct"]
        if ((lo < -tol) | (hi > 100 + tol) | (lo > hi + tol)).any():
            raise ValueError("Cyclone range invalid")
    core.numeric_close(g.track_hours + g.unresolved_track_hours, g.voyage_hours, "position time")
    core.numeric_close(g.wave_matched_hours + g.wave_unmatched_hours, g.voyage_hours, "wave time")
    core.numeric_close(g.wave_bound_hours + g.wave_unboundable_hours, g.voyage_hours, "wave-bound time")


def public_schema(frame, keys):
    if frame.columns.tolist() != keys + MEASURES:
        raise ValueError("Public schema differs from explicit field allowlist")
    for col in COUNTRIES:
        if not frame[col].str.fullmatch(r"[A-Z]{3}").all():
            raise ValueError("Invalid public country code")
    for col in [k for k in keys if k.endswith("_city_code")]:
        if not frame[col].str.fullmatch(r"[A-Z]{3}_c[0-9]{4}").all():
            raise ValueError("City name or invalid value in public code column")
        country = "origin_country" if col.startswith("origin") else "dest_country"
        if not frame[col].str[:3].equals(frame[country]):
            raise ValueError("Coded city and country disagree")
    if not frame[MONTH].str.fullmatch(r"2025-(0[1-9]|1[0-2])").all():
        raise ValueError("Public departure time is not a 2025 month")
    if not set(frame.shiptype_group) <= SHIP_TYPES:
        raise ValueError("Unknown vessel type")
    allowed_reasons = {"no_matched_wave_hours_reachable_bounds_available", "no_matched_or_reachable_wave_hours"}
    if not set(frame.wave_mean_missing_reason.dropna()) <= allowed_reasons:
        raise ValueError("Unexpected free text in public table")
    for col in [c for c in MEASURES if c not in BOOL_FIELDS | TEXT_FIELDS]:
        pd.to_numeric(frame[col], errors="raise")


def export_table(table, keys, path):
    full = table[keys + MEASURES].sort_values(keys).reset_index(drop=True)
    public_schema(full, keys)
    check_values(full, keys)
    DOC.rounded_csv(full, path)
    reread = pd.read_csv(path)
    public_schema(reread, keys)
    if not full[keys].equals(reread[keys]):
        raise ValueError("Public CSV changed record keys")
    for field in MEASURES:
        if field in TEXT_FIELDS | BOOL_FIELDS:
            if not full[field].astype("string").fillna("").equals(reread[field].astype("string").fillna("")):
                raise ValueError("Public CSV changed flag or missing-value reason: " + field)
        else:
            places = DOC.places(field)
            tolerance = 0.5 * 10 ** (-places) + 1e-10 if places is not None else 0
            if not np.allclose(full[field], reread[field], rtol=0, atol=tolerance, equal_nan=True):
                raise ValueError("Public CSV rounding failed: " + field)
    # Distinct fields and missing entries survive the CSV round trip unchanged.
    if not full[MEASURES].isna().equals(reread[MEASURES].isna()):
        raise ValueError("Public CSV changed missingness")
    return {"file": path.name, "rows": len(full), "columns": len(full.columns),
            "bytes": path.stat().st_size, "sha256": digest(path), "round_trip": "pass",
            "schema_allowlist": "pass", "voyages": int(full.voyages.sum()),
            "voyage_hours": float(full.voyage_hours.sum()),
            "records_with_1_voyage": int(full.voyages.eq(1).sum()),
            "records_with_2_to_4_voyages": int(full.voyages.between(2, 4).sum()),
            "records_with_fewer_than_3_ships": int(full.ships.lt(3).sum()),
            "records_without_wave_mean": int(full.mean_swh_matched_m.isna().sum()),
            "records_without_wave_range": int(full.mean_swh_min_possible_m.isna().sum()),
            "records_by_month": {k:int(v) for k,v in full.groupby(MONTH).size().items()}}


def load_source(source, config):
    records = {}
    for name, expected in config["source_sha256"].items():
        path = source / name
        if digest(path) != expected:
            raise ValueError("Frozen input changed: " + name)
        records[name] = {"sha256": expected, "bytes": path.stat().st_size}
    if json.loads((source / "validation.json").read_text())["status"] != "pass":
        raise ValueError("Adopted source validation has not passed")
    columns = ["candidate_id", *CITY, SHIP, "departure_time", *SUMS]
    panel = pd.read_parquet(source / "assembly/voyage_intervals.parquet", columns=columns)
    if len(panel) != config["voyages"] or panel.candidate_id.duplicated().any():
        raise ValueError("Voyage population changed")
    if panel[[*CITY, SHIP, "candidate_id"]].isna().any().any():
        raise ValueError("Missing key or ship identity")
    for side in ("origin", "dest"):
        city = panel[side + "_city"]
        if not city.str.fullmatch(r"[A-Z]{3}:.+").all():
            raise ValueError("City has no country prefix")
        panel[side + "_country"] = city.str[:3]
    departure = pd.to_datetime(panel.departure_time, utc=True)
    if departure.isna().any() or departure.dt.year.ne(2025).any():
        raise ValueError("Departure outside 2025 UTC")
    panel[MONTH] = departure.dt.strftime("%Y-%m")
    if panel[SUMS].isna().any().any() or not np.isfinite(panel[SUMS].to_numpy(float)).all():
        raise ValueError("Missing or non-finite voyage sufficient statistic")
    if abs(float(panel.voyage_hours.sum()) - config["voyage_hours"]) > 1e-6:
        raise ValueError("Voyage duration changed")
    return panel, records


def probe(panel, source, directory):
    """Real-data checks of kernel parity and serial/parallel equivalence."""
    directory.mkdir(parents=True, exist_ok=False)
    reference = pd.read_parquet(source / "assembly/release_full_precision.parquet")
    choices = pd.concat([reference.loc[reference.ships.lt(3)].head(2),
                         reference.loc[reference.ships.ge(3)].iloc[np.linspace(0, int(reference.ships.ge(3).sum())-1, 10, dtype=int)],
                         reference.nlargest(1, "ships"),
                         reference.loc[reference.mean_swh_matched_m.isna()].head(2)]).drop_duplicates(CITY)
    chosen = panel.merge(choices[CITY], on=CITY, validate="many_to_one")
    clusters = chosen.groupby([*CITY, SHIP], sort=True, observed=True)[BOOT_SUMS].sum().reset_index()
    expected = pd.DataFrame([bootstrap_one(key, block, CITY) for key, block in clusters.groupby(CITY, sort=True, observed=True)])
    compare(expected, choices, CITY, [*METRICS, *CI, "ships"], "accepted annual bootstrap probe")
    split = clusters.groupby(CITY, sort=True, observed=True).ngroup() % 2
    paths = []
    for n in (0, 1):
        path = directory / f"probe_{n}.parquet"
        clusters.loc[split.eq(n)].to_parquet(path, index=False)
        paths.append(path)
    with ProcessPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(bootstrap_part, paths, [CITY, CITY]))
    actual = pd.concat([pd.read_parquet(r["output"]) for r in reports])
    compare(actual, expected, CITY, [*METRICS, *CI, "ships"], "serial/parallel bootstrap", atol=0)
    return {"status": "pass", "annual_groups": len(choices), "voyages": len(chosen),
            "accepted_bootstrap_parity": "pass", "serial_parallel_parity": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((ROOT / "02_coding/.runs").resolve()):
        parser.error("private outputs must stay under this worktree's 02_coding/.runs")
    if args.workers < 1 or args.workers > 24:
        parser.error("workers must be between 1 and 24")
    config = json.loads((HERE / "config/authorization.json").read_text())
    sources = [Path(__file__), HERE / "config/authorization.json",
               HERE / "10_build_monthly_release/10_build_tables.py", PIPELINE / "step6_core.py",
               PIPELINE / "250_assemble_gfw_release.py",
               PIPELINE / "240_bootstrap_gfw_release.py", PIPELINE / "270_document_gfw_release.py",
               Path(core.aggregate.__file__), Path(SEEDS.__file__)]
    binding = {"code": {str(p.relative_to(ROOT)): digest(p) for p in sources},
               "source": str(args.source.resolve()), "source_sha256": config["source_sha256"],
               "workers": args.workers, "probe": args.probe}
    if args.resume:
        if json.loads((run / "binding.json").read_text()) != binding:
            raise ValueError("Resume binding differs")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "binding.json", binding)
    private = run / "private"
    private.mkdir(exist_ok=True, mode=0o700)
    state = {"state": "running", "adopted": False, "uploaded": False, "network_requests": 0,
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()}
    write_json(run / "status.json", state)
    try:
        panel, receipts = load_source(args.source, config)
        note(f"Verified {len(panel):,} adopted voyages")
        if args.probe:
            state["probe"] = probe(panel, args.source, private / "bootstrap_probe")
            state["state"] = "probe_pass"
            return
        # The full build requires a separately completed probe from identical code.
        probe_path = run.parent / (run.name + "-probe") / "status.json"
        gate = json.loads(probe_path.read_text())
        probe_binding = json.loads(probe_path.with_name("binding.json").read_text())
        if gate["state"] != "probe_pass" or probe_binding["code"] != binding["code"] or probe_binding["source_sha256"] != binding["source_sha256"]:
            raise ValueError("Matching real-data probe gate not passed")
        checks = {"real_data_probe": gate["probe"]}
        annual = aggregate(panel, CITY)
        checks["all_annual_records"] = compare(annual, pd.read_parquet(args.source / "assembly/all_combinations.parquet"),
                                                CITY, POINT_FIELDS, "annual aggregation")
        released = annual.loc[annual.voyages.ge(5)]
        checks["accepted_release"] = compare(released, pd.read_parquet(args.source / "assembly/release_full_precision.parquet"),
                                               CITY, POINT_FIELDS, "accepted release")
        if len(released) != 39793 or int(released.voyages.sum()) != 826279:
            raise ValueError("Accepted annual release census changed")
        note("All annual point estimates reproduce the adopted GFW result")
        mapping = city_codes(pd.concat([panel.origin_city, panel.dest_city]), private / "city_code_map_PRIVATE.csv")
        lookup = mapping.set_index("city_id").city_code
        for side in ("origin", "dest"):
            panel[side + "_city_code"] = panel[side + "_city"].map(lookup)
            annual[side + "_city_code"] = annual[side + "_city"].map(lookup)
            annual[side + "_country"] = annual[side + "_city"].str[:3]
        country = aggregate(panel, COUNTRY_KEY)
        city = aggregate(panel, CODED_KEY)
        if (len(country), len(city)) != (config["country_records"], config["city_records"]):
            raise ValueError("Monthly group census differs from preflight")
        checks["city_months_to_year"] = adds_up(city, annual, CODED_KEY[:-1], "months to year")
        checks["city_pairs_to_countries"] = adds_up(city, country, COUNTRY_KEY, "city pairs to country pair")
        for table, keys in ((country, COUNTRY_KEY), (city, CODED_KEY)):
            check_values(table, keys)
            if int(table.voyages.sum()) != config["voyages"]:
                raise ValueError("Some retained voyages are missing or duplicated")
        note(f"Closure passed: country={len(country):,}; coded city={len(city):,}")
        summaries = {}
        bootstrap_reports = {}
        public = run / "public"
        public.mkdir(exist_ok=True)
        for name, table, keys, filename in (("country", country, COUNTRY_KEY, "01_release_country_pair_month.csv"),
                                             ("city", city, CODED_KEY, "02_release_city_pair_month.csv")):
            table, bootstrap_reports[name] = bootstrap(panel, table, keys, private / ("bootstrap_" + name), args.workers)
            table[keys + MEASURES].sort_values(keys).to_parquet(private / (name + "_full_precision.parquet"), index=False)
            summaries[name] = export_table(table, keys, public / filename)
            note(f"{name} CSV written and independently reread")
        for name, expected in config["source_sha256"].items():
            if digest(args.source / name) != expected:
                raise ValueError("Source changed during execution")
        checks["inputs_unchanged"] = "pass"
        checks["city_map_one_to_one"] = {"status": "pass", "cities": len(mapping)}
        state.update(state="validated", checks=checks, source_receipts=receipts, tables=summaries,
                     bootstrap=bootstrap_reports, countries=int(pd.unique(panel[COUNTRIES].to_numpy().ravel()).size),
                     cities=len(mapping), ships=int(panel[SHIP].nunique()),
                     city_pairs=int(city[["origin_city_code", "dest_city_code"]].drop_duplicates().shape[0]),
                     country_pairs=int(country[COUNTRIES].drop_duplicates().shape[0]),
                     private_map_sha256=digest(private / "city_code_map_PRIVATE.csv"))
        write_json(run / "validation.json", state)
        note("Monthly release checks passed; candidate files remain local and unadopted")
    except BaseException as error:
        state.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(run / "status.json", state)


if __name__ == "__main__":
    main()
