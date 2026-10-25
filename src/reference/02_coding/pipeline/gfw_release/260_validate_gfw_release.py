#!/usr/bin/env python3
"""Independent source, denominator, bounds, holdout and bootstrap checks for E006."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from step6_core import *


def receipt_matches(item):
    if record(Path(item["path"])) != item:
        raise ValueError(f"receipt changed: {item['path']}")


def seed_for(key):
    value = "|".join(str(part) for part in key)
    return int.from_bytes(hashlib.sha256(f"{SEED}|{value}".encode()).digest()[:4], "big")


def independent_bootstrap(block):
    columns = ["wave_integral_m_hours", "wave_matched_hours", "wave_ge4m_matched_hours",
               "tc300_positive_snapshot_hours", "tc500_positive_snapshot_hours", "voyage_hours"]
    clusters = block.groupby(SHIP, sort=True)[columns].sum().to_numpy(float)
    if len(clusters) < 3:
        return {name: (np.nan, np.nan) for name in (
            "mean_swh_matched_m", "share_wave_hours_ge4m", "tc300_at_least_pct", "tc500_at_least_pct")}
    key = tuple(block.iloc[0][KEY])
    # One matrix, independently calculated ratios; production uses batches of 100.
    draws = np.random.default_rng(seed_for(key)).integers(0, len(clusters), size=(1000, len(clusters)))
    total = clusters[draws].sum(axis=1)
    def ratio(a, b):
        return np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)
    values = {"mean_swh_matched_m": ratio(total[:, 0], total[:, 1]),
              "share_wave_hours_ge4m": ratio(total[:, 2], total[:, 1]),
              "tc300_at_least_pct": 100*ratio(total[:, 3], total[:, 5]),
              "tc500_at_least_pct": 100*ratio(total[:, 4], total[:, 5])}
    return {name: np.quantile(v[np.isfinite(v)], [.025, .975], method="linear")
            if np.isfinite(v).any() else (np.nan, np.nan) for name, v in values.items()}


def validate(root: Path):
    source = binding(); revised = revision(); root = inside_runs(root)
    checks, details = {}, {}
    def gate(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(f"validation gate failed: {name}")
    prep = json.loads((root / "preparation/preparation.json").read_text())
    states = {name: json.loads((root / name / "status.json").read_text())
              for name in ("bounds", "holdout_manifest", "holdout", "bootstrap_base", "bootstrap_wave", "assembly")}
    for name, state in states.items():
        gate(f"{name}_complete", state["state"] == ("assembled" if name == "assembly" else "completed")
             and not state.get("failures"))
        for item in state.get("code", []): receipt_matches(item)
        for item in state.get("outputs", {}).values(): receipt_matches(item)
        if "output" in state: receipt_matches(state["output"])
    for item in prep["outputs"].values(): receipt_matches(item)
    for item in states["bounds"]["weather_assets"].values():
        receipt_matches({k: item[k] for k in ("path", "bytes", "sha256")})
    gate("frozen_input_and_code_hashes", True)
    # The tolerance probe reruns the adapter at 0 h against the previously validated census and
    # checks, hour by hour, that widening the known ends only widens the bounds.
    probe = json.loads((root/"bounds_tolerance_probe.json").read_text())
    for item in probe["receipts"].values(): receipt_matches(item)
    gate("bounds_zero_tolerance_reproduces_validated_census", probe["zero_tolerance"]["identical"]
         and probe["zero_tolerance"]["voyages"] > 0 and len(probe["zero_tolerance"]["fields"]) >= 6)
    gate("bounds_tolerance_only_widens", probe["status"] == "pass"
         and probe["tolerance_hours"] == revised["fix_time_tolerance_hours"] and probe["widening"]["hours"] > 0
         and all(probe["widening"]["checks"].values()))
    gate("bounds_census_uses_revised_tolerance", states["bounds"]["fix_time_tolerance_hours"]
         == revised["fix_time_tolerance_hours"] and not states["bounds"]["hour_detail"])
    parallel = json.loads((root/"parallel_bootstrap_probe.json").read_text())
    gate("bootstrap_serial_parallel_repeatability", parallel["identical_bytes"])
    parallel = json.loads((root/"holdout_parallel_manifest_probe.json").read_text())
    for key in ("serial", "parallel"): receipt_matches(parallel[key])
    gate("holdout_manifest_serial_parallel_repeatability", parallel["byte_identical"])
    example = json.loads((root/"reachable_example/example.json").read_text())
    receipt_matches(example["figure"])
    gate("private_reachable_area_example", example["complete_grid"] and example["tc500_possible"]
         and example["swh_min_m"] <= example["swh_max_m"] and not example["public_export"])
    gate("bounds_census_64_shards", sorted(states["bounds"]["completed"]) == SHARDS
         and states["bounds"]["probe_candidates"] is None)
    gate("holdout_census_64_shards", sorted(states["holdout"]["completed"]) == SHARDS
         and states["holdout"]["pilot_per_stratum"] == 0)
    panel = pd.read_parquet(root / "assembly/voyage_intervals.parquet")
    original = pd.read_parquet(root / "preparation/voyage_panel.parquet")
    grouped = pd.read_parquet(root / "assembly/all_combinations.parquet").set_index(KEY).sort_index()
    release = pd.read_parquet(root / "assembly/release_full_precision.parquet").set_index(KEY).sort_index()
    gate("original_voyage_identity", len(panel) == source["counts"]["candidates"] and not panel.candidate_id.duplicated().any()
         and set(panel.candidate_id) == set(original.candidate_id) and not panel[[*KEY, SHIP]].isna().any().any())
    same = panel.set_index("candidate_id")[[c for c in original if c != "candidate_id"]].sort_index()
    pd.testing.assert_frame_equal(same, original.set_index("candidate_id").sort_index(), check_exact=True)
    gate("original_voyage_facts_unchanged", True)
    identities = ["candidate_id", "mmsi", SHIP, "source_shard", "departure_time", "arrival_time", *KEY]
    raw_sources = []
    for item in prep["candidate_summaries"]:
        receipt_matches(item)
        raw_sources.append(pd.read_parquet(item["path"], columns=identities))
    raw_identities = pd.concat(raw_sources, ignore_index=True).set_index("candidate_id").sort_index()
    pd.testing.assert_frame_equal(panel[identities].set_index("candidate_id").sort_index(),
                                  raw_identities, check_exact=True, check_dtype=False)
    annual = pd.read_parquet(source["annual"]["path"]).set_index("candidate_id").sort_index()
    pd.testing.assert_frame_equal(panel.set_index("candidate_id")[annual.columns].sort_index(),
                                  annual, check_exact=True, check_dtype=False)
    gate("independent_raw_gfw_identity_and_weather_join", True)
    numeric_close(panel.unknown_hours, panel.unresolved_track_hours, "all unknown voyage hours")
    numeric_close(panel.track_hours+panel.unresolved_track_hours, panel.voyage_hours, "position denominator")
    numeric_close(panel.wave_matched_hours+panel.wave_unmatched_hours, panel.voyage_hours, "wave denominator")
    for field, total in prep["totals"].items(): numeric_close(panel[field].sum(), total, field)
    gate("annual_numerators_and_denominators", True)
    port_check = json.loads((root/"gfw_wholly_unobserved_port_feasibility.json").read_text())
    # Independently identify impossible blind-voyage ends; never waive this check
    # just because the speed filter and the bounding model have different caps.
    ports = pd.read_parquet(root/'preparation/port_endpoints.parquet')
    blind = ports.merge(panel.loc[panel.track_hours.eq(0), ['candidate_id']], on='candidate_id', validate='one_to_one')
    p1, p2 = np.radians(blind.departure_port_lat), np.radians(blind.arrival_port_lat)
    dl = np.radians(blind.arrival_port_lon-blind.departure_port_lon)
    aa = np.sin((p2-p1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    distance = 2*6371.0088*np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
    duration = (blind.arrival_time-blind.departure_time).dt.total_seconds()/3600
    conflict = blind.port_endpoints_connected.fillna(False).astype(bool) & (distance > 25*1.852*(duration+2)+1e-6)
    conflict_ids = sorted(blind.loc[conflict, 'candidate_id'].astype(str))
    gate('gfw_wholly_unobserved_port_audit_matches',
         port_check['no_position_voyages'] == len(blind)
         and port_check['conflicting_main_port_endpoints'] == len(conflict_ids)
         and port_check.get('conflicting_candidate_ids', []) == conflict_ids
         and abs(port_check['main_conflict_hours']-duration[conflict].sum()) < 1e-7
         and not port_check['old_conflict_ids_reused'])
    impossible = panel.loc[panel.candidate_id.astype(str).isin(conflict_ids)]
    gate('gfw_conflicting_ports_explicitly_unbounded',
         np.allclose(impossible.main_unbounded_hours, impossible.unknown_hours, rtol=0, atol=1e-7)
         and np.allclose(impossible.main_wave_counted_hours, 0, rtol=0, atol=1e-7)
         and all(np.allclose(impossible[f'main_tc{r}_possible_hours'], impossible.unknown_hours, rtol=0, atol=1e-7)
                 for r in (300, 500)))
    bounds_rows = 0
    for shard in SHARDS:
        status = json.loads((root / "bounds" / shard / "status.json").read_text())
        receipt_matches(status["output"]); bounds_rows += status["weather_rows"]
        block = panel.loc[panel.source_shard.eq(shard)]
        numeric_close(block.unknown_hours.sum(), status["totals"]["unknown_hours"], shard)
        for rule in ("main", "zero"):
            outcomes = [c for c in block if c.startswith(rule+"_wave_") and c.endswith("_hours")]
            numeric_close(block[outcomes].sum(axis=1), block.unknown_hours, rule+" wave status closure")
            gate(f"{shard}_{rule}_bounds", bool(
                (block[f"{rule}_wave_min_integral"] <= block[f"{rule}_wave_max_integral"]+1e-7).all()
                and (block[f"{rule}_tc300_possible_hours"] <= block[f"{rule}_tc500_possible_hours"]+1e-7).all()
                and (block[f"{rule}_tc500_possible_hours"] <= block.unknown_hours+1e-7).all()))
    gate("hour_window_census", bounds_rows == source["counts"]["weather_rows"])
    numeric_close(grouped.voyages.sum(), len(panel), "all group voyage census")
    gate("unique_annual_directed_group_grain", len(grouped) == source["counts"]["combinations"] and not grouped.index.duplicated().any())
    selected = grouped.loc[grouped.voyages.ge(5)]
    gate("release_threshold_and_identity", release.index.equals(selected.index)
         and len(release) == source["counts"]["release_combinations"]
         and int(release.voyages.sum()) == source["counts"]["release_voyages"])
    common = [c for c in release if c in selected]
    pd.testing.assert_frame_equal(release[common], selected[common], check_exact=True, check_dtype=False)
    gate("all_release_points_and_flags_match_verified_groups", True)
    below = json.loads((root / "assembly/below_threshold_summary.json").read_text())
    gate("below_threshold_closure", below["combinations"]+len(release) == len(grouped)
         and below["voyages"]+release.voyages.sum() == len(panel))
    sums = [*aggregate.WEATHER_SUMS, "main_wave_counted_hours", "main_wave_min_integral",
            "main_wave_max_integral", "main_unbounded_hours", "main_tc300_possible_hours", "main_tc500_possible_hours"]
    independent = panel.groupby(KEY, observed=True, sort=True)[sums].sum()
    numeric_close(independent, grouped[sums], "independent group sums")
    numeric_close(panel.groupby(KEY, observed=True).size(), grouped.voyages, "group voyage counts")
    numeric_close(panel.groupby(KEY, observed=True)[SHIP].nunique(), grouped.ships, "distinct ship counts")
    support = independent.wave_matched_hours+independent.main_wave_counted_hours
    lower = (independent.wave_integral_m_hours+independent.main_wave_min_integral)/support.where(support > 0)
    upper = (independent.wave_integral_m_hours+independent.main_wave_max_integral)/support.where(support > 0)
    observed = independent.wave_integral_m_hours/independent.wave_matched_hours.where(independent.wave_matched_hours > 0)
    numeric_close(lower, grouped.mean_swh_min_possible_m, "independent wave lower")
    numeric_close(upper, grouped.mean_swh_max_possible_m, "independent wave upper")
    numeric_close(observed, grouped.mean_swh_matched_m, "independent matched mean")
    constrained = np.maximum(lower, np.minimum(upper, observed))
    numeric_close(constrained, grouped.mean_swh_constrained_m, "independent constrained mean")
    numeric_close(constrained-observed, grouped.mean_swh_constrained_shift_m, "constrained shift")
    flag, has_mean = grouped.wave_mean_was_constrained, observed.notna().to_numpy()
    gate("constrained_flag", np.array_equal(flag.isna().to_numpy(), ~has_mean)
         and np.array_equal(flag.to_numpy()[has_mean].astype(bool),
                            (constrained-observed).abs().gt(1e-9).to_numpy()[has_mean]))
    numeric_close(upper-lower, grouped.mean_swh_possible_range_m, "wave bound width")
    numeric_close(support, grouped.wave_bound_hours, "wave support hours")
    numeric_close(support/independent.voyage_hours, grouped.wave_bound_coverage, "wave support coverage")
    numeric_close(independent.voyage_hours-support, grouped.wave_unboundable_hours, "unsupported wave hours")
    gate("wave_support_within_voyage", bool((grouped.wave_unboundable_hours >= -1e-7).all()
         and (grouped.wave_bound_hours >= grouped.wave_matched_hours-1e-7).all()))
    for radius in (300, 500):
        positive = independent[f"tc{radius}_positive_snapshot_hours"]
        possible = independent[f"main_tc{radius}_possible_hours"]
        numeric_close(grouped[f"tc{radius}_at_least_pct"], 100*positive/independent.voyage_hours, "full duration TC least")
        numeric_close(grouped[f"tc{radius}_at_most_pct"], 100*(positive+possible)/independent.voyage_hours, "full duration TC most")
        numeric_close(grouped[f"tc{radius}_hours_at_most"], grouped[f"tc{radius}_hours_at_least"]+
                      grouped[f"tc{radius}_bounded_possible_hours"]+grouped.unbounded_hours, "TC upper decomposition")
        gate(f"tc{radius}_physical_range", bool((grouped[f"tc{radius}_at_least_pct"] >= 0).all()
             and (grouped[f"tc{radius}_at_most_pct"] <= 100+1e-7).all()
             and (grouped[f"tc{radius}_at_least_pct"] <= grouped[f"tc{radius}_at_most_pct"]+1e-7).all()))
    for col, num, den in (("position_coverage", "track_hours", "voyage_hours"),
                          ("wave_coverage", "wave_matched_hours", "voyage_hours"),
                          ("share_wave_hours_ge4m", "wave_ge4m_matched_hours", "wave_matched_hours")):
        numeric_close(grouped[col], independent[num]/independent[den].where(independent[den] > 0), col)
    gate("independent_aggregation_and_estimands", True)
    present = release.loc[release.mean_swh_constrained_m.notna()]
    gate("constrained_wave_inside_possible_bounds", bool(
        (present.mean_swh_constrained_m >= present.mean_swh_min_possible_m-1e-7).all()
        and (present.mean_swh_constrained_m <= present.mean_swh_max_possible_m+1e-7).all()))
    intervals = [c for c in release if "_ci95_" in c]
    gate("fewer_than_3_ships_intervals_empty", release.loc[release.ships.lt(3), intervals].isna().all().all()
         and release.sampling_interval_unavailable_fewer_than_3_ships.eq(release.ships.lt(3)).all())
    for col in intervals:
        if col.endswith("_low"):
            gate(col+"_ordered", (release[col].dropna() <= release[col.replace("_low", "_high")].dropna()+1e-7).all())
    choice = np.unique(np.r_[np.linspace(0, len(release)-1, 30, dtype=int),
                              np.flatnonzero(release.ships.ge(3))[:3],
                              np.flatnonzero(release.wave_unknown_counted_hours.gt(0))[:3]])
    for key in release.iloc[choice].index:
        block = panel.loc[np.logical_and.reduce([panel[c].eq(v).to_numpy() for c, v in zip(KEY, key)])]
        for metric, values in independent_bootstrap(block).items():
            stem = "mean_swh_matched" if metric == "mean_swh_matched_m" else metric
            numeric_close(release.loc[key, [stem+"_ci95_low", stem+"_ci95_high"]].to_numpy(float),
                          values, "independent ship bootstrap: "+metric)
    gate("bootstrap_independent_reproduction", True)
    details["independently_bootstrapped_groups"] = len(choice)
    for mode in ("base", "wave"):
        gate(mode+"_bootstrap_frozen_parameters", states["bootstrap_"+mode]["replicates"] == 1000
             and states["bootstrap_"+mode]["seed"] == SEED and states["bootstrap_"+mode]["minimum_ships"] == 3)
        saved = pd.read_parquet(root/f"bootstrap_{mode}/bootstrap.parquet").set_index(KEY).sort_index()
        if mode == "wave":
            saved = saved.rename(columns={f"mean_swh_matched_m_ci95_{side}":
                                           f"mean_swh_matched_ci95_{side}" for side in ("low","high")})
        columns = [c for c in saved if "_ci95_" in c]
        gate(mode+"_bootstrap_release_keys", saved.index.equals(release.index))
        numeric_close(release[columns], saved[columns], mode+" complete release sampling intervals")
    wave_boot = pd.read_parquet(root/"bootstrap_wave/bootstrap.parquet")
    gate("wave_bootstrap_matched_estimand", states["bootstrap_wave"].get("wave_estimand") == "matched_mean")
    available_boot = wave_boot.loc[wave_boot.ships.ge(3) & wave_boot.mean_swh_matched_m.notna()]
    draw_counts = available_boot.mean_swh_matched_m_valid_bootstrap_draws
    details["wave_bootstrap_groups_fewer_than_1000_valid_draws"] = int(draw_counts.lt(1000).sum())
    details["wave_bootstrap_min_valid_draws"] = int(draw_counts.min()) if len(draw_counts) else None
    manifest = pd.read_parquet(root / "holdout_manifest/samples.parquet")
    main = manifest.loc[manifest.sample_set.eq("gap_length_x_shiptype")]
    storm = manifest.loc[manifest.sample_set.eq("tc500km_additional")]
    strata = main.groupby(["nominal_gap_hours", "shiptype_group"], observed=True).size()
    gate("hourly_holdout_sample_layers", len(strata) == 45 and strata.eq(500).all()
         and set(main.nominal_gap_hours) == {2,4,6,12,24} and len(storm) == 500 and len(manifest) == 23000
         and not manifest.sample_id.duplicated().any())
    outcomes = pd.read_parquet(root / "holdout/all_outcomes.parquet")
    gate("holdout_all_samples_accounted", set(outcomes.sample_id) == set(manifest.sample_id)
         and len(outcomes) == len(manifest) and not outcomes.sample_id.duplicated().any())
    gate("heldout_points_not_leaked", outcomes.heldout_points_leaked.eq(0).all())
    paired = []
    for shard in SHARDS:
        report = json.loads((root / "holdout/shards" / shard / "status.json").read_text())
        for item in report["outputs"].values(): receipt_matches(item)
        annual_source = json.loads((source["weather_run"] / "shards" / shard / "status.json").read_text())["sources"]["track"]
        gate(shard+"_holdout_source", report["source_track"] == annual_source)
        paired.append(pd.read_parquet(root / "holdout/shards" / shard / "point_results.parquet"))
    points = pd.concat(paired, ignore_index=True)
    gate("holdout_point_identity", not points.duplicated(["sample_id", "UTC"]).any()
         and points.UTC.dt.minute.eq(30).all() and points.UTC.eq(points.UTC_10min).all())
    counts = points.groupby("sample_id").size().reindex(manifest.sample_id).to_numpy()
    numeric_close(counts, manifest.truth_observation_count.to_numpy(), "all hidden points accounted")
    windows = points.merge(manifest[["sample_id", "gap_start_utc", "gap_end_utc"]], on="sample_id", validate="many_to_one")
    gate("truth_inside_hidden_window", (windows.UTC > windows.gap_start_utc).all()
         and (windows.UTC < windows.gap_end_utc).all())
    storm_ids = storm.sample_id
    gate("storm_sample_truth_near_tc500", points.loc[points.sample_id.isin(storm_ids)].groupby("sample_id").tc500_truth.max().eq(1).all())
    numeric_close(points.wave_difference_m, points.swh_pred_m-points.swh_truth_m, "wave point difference")
    for radius in (300,500):
        valid = points[f"tc{radius}_pred"].notna()
        numeric_close(points.loc[valid, f"tc{radius}_flip"],
                      points.loc[valid, f"tc{radius}_pred"].ne(points.loc[valid, f"tc{radius}_truth"]).astype(float), "TC flip")
    metrics = pd.read_csv(root / "holdout/metrics_by_stratum.csv").set_index(["sample_set", "nominal_gap_hours", "shiptype_group"])
    for key, block in points.groupby(["sample_set", "nominal_gap_hours", "shiptype_group"], sort=True):
        row = metrics.loc[key]
        numeric_close([row.truth_points, row.predicted_points, row.wave_paired_points],
                      [len(block), block.lat_pred.notna().sum(), block.wave_difference_m.notna().sum()], "stratum counts")
        for name, value in (("position_error_km", block.position_error_km),
                            ("wave_absolute_difference_m", block.wave_difference_m.abs())):
            for q in (.5,.9,.95): numeric_close(row[f"{name}_p{int(q*100)}"], value.quantile(q), "stratum quantile")
    gate("holdout_metrics_independent_reduction", True)
    # Check actual observed source points independently for a fixed spread of samples.
    selected_truth = manifest.sort_values("sample_id").iloc[np.linspace(0,len(manifest)-1,18,dtype=int)]
    for shard, selected_samples in selected_truth.groupby("source_shard"):
        path = json.loads((root / "holdout/shards" / shard / "status.json").read_text())["source_track"]["path"]
        raw = pd.read_parquet(path, columns=["candidate_id", "observation_time_utc", "lat", "lon", "source_provider"],
                              filters=[("candidate_id", "in", selected_samples.candidate_id.tolist()), ("track_evidence", "=", "observed_ais")])
        gate(shard+"_truth_is_observed_gfw", raw.source_provider.str.startswith("gfw").all())
        for sample in selected_samples.itertuples(index=False):
            expected = raw.loc[raw.candidate_id.eq(sample.candidate_id) & raw.observation_time_utc.gt(sample.gap_start_utc)
                               & raw.observation_time_utc.lt(sample.gap_end_utc)].sort_values("observation_time_utc")
            actual = points.loc[points.sample_id.eq(sample.sample_id)].sort_values("UTC")
            numeric_close(actual[["lat_truth", "lon_truth"]].to_numpy(), expected[["lat", "lon"]].to_numpy(), "real GFW truth coordinates")
            gate(sample.sample_id+"_truth_time", np.array_equal(actual.UTC.to_numpy(), expected.observation_time_utc.to_numpy()))
    gate("independent_observed_gfw_truth", True)
    details.update(candidates=len(panel), all_combinations=len(grouped), release_combinations=len(release),
        release_voyages=int(release.voyages.sum()), below_combinations=below["combinations"], below_voyages=int(below["voyages"]),
        fewer_than_3_ships=int(release.ships.lt(3).sum()), holdout_samples=len(manifest), holdout_truth_points=len(points),
        holdout_predicted_points=int(points.lat_pred.notna().sum()), unresolved_holdout_windows=int(outcomes.status.ne("predicted").sum()),
        wave_unknown_counted_hours=float(panel.main_wave_counted_hours.sum()),
        constrained_combinations=int(release.wave_mean_was_constrained.fillna(False).astype(bool).sum()),
        constrained_shift_abs_p50_m=float(release.mean_swh_constrained_shift_m.abs().quantile(.5)),
        constrained_shift_abs_p99_m=float(release.mean_swh_constrained_shift_m.abs().quantile(.99)),
        constrained_shift_abs_max_m=float(release.mean_swh_constrained_shift_m.abs().max()),
        fix_time_tolerance_hours=revised["fix_time_tolerance_hours"],
        wave_unknown_unsupported_hours=float((panel.unknown_hours-panel.main_wave_counted_hours).sum()),
        known_position_wave_unmatched_hours=float((panel.wave_unmatched_hours-panel.unknown_hours).sum()),
        unbounded_hours=float(panel.main_unbounded_hours.sum()),
        no_position_voyages=port_check["no_position_voyages"],
        conflicting_main_port_endpoints=port_check["conflicting_main_port_endpoints"],
        conflicting_port_voyages_retained_unbounded=len(conflict_ids),
        totals={c: float(panel[c].sum()) for c in aggregate.WEATHER_SUMS})
    result = {"status": "pass", "checks": checks, "details": details, "network_requests": 0,
              "products_modified": 0, "adopted": False, "authorization": source["authorization"],
              "component_receipts": {k: record(root/k/"status.json") for k in states}}
    result.update(provenance([HERE / "step6_core.py", Path(__file__), DECISION, REVISION,
                              HERE / 'port_constraints.py',
                              HERE / "200_prepare_gfw_release.py"]))
    write_json(root / "validation.json", result)
    print(json.dumps({"status": result["status"], "passed_checks": len(checks), **details}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate(args.run)
    except Exception as exc:
        write_json(inside_runs(args.run)/"validation_failure.json", {"status": "failed", "error": repr(exc)})
        raise
