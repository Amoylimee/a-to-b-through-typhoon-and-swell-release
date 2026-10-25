"""Read-only manuscript summaries of E008 tables and frozen E007 comparisons.

No trajectories, environmental matches, intervals, or release values are refitted.
Private per-voyage sufficient statistics remain in --out/private. Only public/
contains aggregate figure data and manuscript evidence suitable for Git.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ANNUAL = ["origin_city", "dest_city", "shiptype_group"]
COUNTRY = ["origin_country", "dest_country", "shiptype_group", "departure_month"]
CITY = ["origin_city_code", "dest_city_code", "shiptype_group", "departure_month"]
SUMS = ["paired_hours", "wave_paired_hours", "h4_true", "h4_gfw",
        "tc300_true", "tc300_gfw", "tc500_true", "tc500_gfw"]


def receipt(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def one_shard(job):
    weather, points = job
    f = pd.read_parquet(weather, columns=["cand", "swh_true", "swh_gfw",
                                           "tc300_true", "tc300_gfw", "tc500_true", "tc500_gfw"])
    paired = f.swh_true.notna() & f.swh_gfw.notna()
    f["paired_hours"] = 1
    f["wave_paired_hours"] = paired.astype("int64")
    f["h4_true"] = (paired & f.swh_true.ge(4)).astype("int64")
    f["h4_gfw"] = (paired & f.swh_gfw.ge(4)).astype("int64")
    summed = f.groupby("cand", observed=True)[SUMS].sum().reset_index()
    candidates = pd.read_parquet(points, columns=["cand", "candidate_id"])
    result = summed.merge(candidates, on="cand", how="left", validate="one_to_one")
    assert result.candidate_id.notna().all()
    return result.drop(columns="cand"), [receipt(weather), receipt(points)]


def shares(frame, keys):
    s = frame.groupby(keys, observed=True)[SUMS].sum()
    for source in ["true", "gfw"]:
        s[f"ge4m_share_{source}"] = s[f"h4_{source}"] / s.wave_paired_hours.replace(0, np.nan)
        for radius in [300, 500]:
            s[f"tc{radius}_pct_{source}"] = 100 * s[f"tc{radius}_{source}"] / s.paired_hours
    for metric in ["ge4m_share", "tc300_pct", "tc500_pct"]:
        s[f"{metric}_diff"] = s[f"{metric}_gfw"] - s[f"{metric}_true"]
    return s.reset_index()


def curve(values, **labels):
    x = np.sort(np.asarray(values, dtype=float))
    assert len(x) and np.isfinite(x).all() and (x >= 0).all()
    grid = np.unique(np.r_[0, np.geomspace(1e-8, max(x[-1], 1e-8), 1800),
                           np.quantile(x, [.5, .9, .95, .99]), x[-1]])
    return [dict(labels, value=float(v), cumulative_pct=float(p), n=len(x))
            for v, p in zip(grid, 100 * np.searchsorted(x, grid, side="right") / len(x))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coding", type=Path, required=True)
    parser.add_argument("--monthly-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    runs = args.coding / ".runs"
    assembly = runs / "2026-10-11_1907_E007-one-cell-release/assembly"
    weather = runs / "2026-10-13_2006_E007-one-cell-real-gap-weather"
    points = runs / "2026-10-13_2006_E007-one-cell-real-gap-comparison"
    segmentation = runs / "2026-10-13_2006_E007-one-cell-voyage-segmentation/voyage_segmentation.parquet"
    assert json.loads((weather / "execution.json").read_text())["state"] == "pass"
    public, private = args.out / "public", args.out / "private"
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(mode=0o700)
    voyage_path = assembly / "voyage_intervals.parquet"
    panel = pd.read_parquet(voyage_path, columns=["candidate_id", *ANNUAL, "vessel_uid_clean", "departure_time"])
    assert len(panel) == 1111800 and panel.candidate_id.is_unique
    panel["departure_month"] = pd.to_datetime(panel.departure_time, utc=True).dt.strftime("%Y-%m")
    mapping_path = args.monthly_run / "private/city_code_map_PRIVATE.csv"
    mapping = pd.read_csv(mapping_path).set_index("city_id").city_code
    for role in ["origin", "dest"]:
        panel[f"{role}_country"] = panel[f"{role}_city"].str[:3]
        panel[f"{role}_city_code"] = panel[f"{role}_city"].map(mapping)
        assert panel[f"{role}_city_code"].notna().all()
    inputs = [receipt(voyage_path), receipt(mapping_path), receipt(segmentation)]
    pairs = sorted((weather / "shards").glob("*/hourly_weather_pairs.parquet"))
    assert len(pairs) == 64
    jobs = [(p, points / "shards" / p.parent.name / "candidates.parquet") for p in pairs]
    # The first shard is the small probe; it is reused in the complete summary.
    first, source = one_shard(jobs[0])
    assert first.paired_hours.sum() == 1048330
    print("Probe passed: one shard, candidate mapping and hour counts", flush=True)
    blocks = [first]
    inputs.extend(source)
    with ThreadPoolExecutor(args.workers) as pool:
        for number, (block, source) in enumerate(pool.map(one_shard, jobs[1:]), start=2):
            blocks.append(block)
            inputs.extend(source)
            if number % 8 == 0:
                print(f"Summarized {number}/64 frozen shards", flush=True)
    totals = pd.concat(blocks, ignore_index=True)
    assert totals.candidate_id.is_unique
    assert totals.paired_hours.sum() == 67349525
    assert totals.wave_paired_hours.sum() == 67107550
    paired = totals.merge(panel, on="candidate_id", validate="one_to_one", how="left")
    assert paired.departure_month.notna().all()
    paired.to_parquet(private / "paired_voyage_sums.parquet", index=False)
    # Reproduce the accepted annual comparison before changing its grouping.
    old_path = weather / "summary/record_exposure_true_vs_gfw_positions.csv"
    old = pd.read_csv(old_path)
    legacy = shares(paired, ANNUAL).merge(old[ANNUAL], on=ANNUAL, validate="one_to_one")
    legacy = legacy.set_index(ANNUAL).sort_index()
    reference = old.set_index(ANNUAL).sort_index()
    assert legacy.index.equals(reference.index) and len(legacy) == 39616
    fields = ["paired_hours", "wave_paired_hours", "ge4m_share_diff", "tc300_pct_diff", "tc500_pct_diff"]
    for field in fields:
        np.testing.assert_allclose(legacy[field], reference[field], atol=1e-12, rtol=0, equal_nan=True)
    print("Annual parity passed: 39,616 records and all paired-hour/share fields", flush=True)
    inputs.append(receipt(old_path))
    summary = {"voyages": len(panel), "ships": int(panel.vessel_uid_clean.nunique()),
               "cities": len(set(panel.origin_city) | set(panel.dest_city)),
               "countries": len(set(panel.origin_country) | set(panel.dest_country)),
               "city_pairs": len(panel.drop_duplicates(["origin_city", "dest_city"])),
               "country_pairs": len(panel.drop_duplicates(["origin_country", "dest_country"])),
               "paired_hours": int(totals.paired_hours.sum()),
               "wave_paired_hours": int(totals.wave_paired_hours.sum()), "levels": {}}
    records, curves, ranges, range_curves, types = [], [], [], [], []
    for level, keys, count in [("country", COUNTRY, 122736), ("city", CITY, 566294)]:
        path = args.monthly_run / "private" / f"{level}_full_precision.parquet"
        d = pd.read_parquet(path)
        inputs.append(receipt(path))
        assert len(d) == count and d.voyages.sum() == 1111800 and not d.duplicated(keys).any()
        expected = panel.groupby(keys).size().sort_index()
        pd.testing.assert_series_equal(expected, d.set_index(keys).voyages.sort_index(), check_names=False)
        stats = {"records": len(d), "median_voyages": float(d.voyages.median()),
                 "median_ships": float(d.ships.median()), "single_voyage": int(d.voyages.eq(1).sum()),
                 "under_3_ships": int(d.ships.lt(3).sum()),
                 "position_coverage": float(d.track_hours.sum() / d.voyage_hours.sum()),
                 "wave_coverage": float(d.wave_matched_hours.sum() / d.voyage_hours.sum()),
                 "wave_means_available": int(d.mean_swh_constrained_m.notna().sum()),
                 "median_wave_mean_m": float(d.mean_swh_constrained_m.median())}
        summary["levels"][level] = stats
        for shiptype, group in d.groupby("shiptype_group"):
            types.append(dict(level=level, shiptype_group=shiptype, records=len(group), voyages=int(group.voyages.sum())))
        record = shares(paired, keys)
        assert record.paired_hours.sum() == 67349525
        assert record.wave_paired_hours.sum() == 67107550
        check = record.merge(d[keys], on=keys, how="left", validate="one_to_one", indicator=True)
        assert check._merge.eq("both").all()
        record.to_parquet(private / f"{level}_record_comparison.parquet", index=False)
        for metric, label, scale in [("ge4m_share", "Wave ≥4 m", 100),
                                     ("tc300_pct", "Cyclone ≤300 km", 1),
                                     ("tc500_pct", "Cyclone ≤500 km", 1)]:
            x = record[f"{metric}_diff"].dropna().abs().to_numpy() * scale
            row = dict(level=level, metric=metric, records=len(x), identical_pct=float(100 * (x == 0).mean()))
            row.update({f"p{q}_pp": float(np.quantile(x, q/100)) for q in [50, 90, 95, 99, 100]})
            records.append(row)
            curves.extend(curve(x, metric=f"{level}_record_share", group=label))
        for metric, lo, hi in [("tc300", "tc300_at_least_pct", "tc300_at_most_pct"),
                                ("tc500", "tc500_at_least_pct", "tc500_at_most_pct"),
                                ("wave", "mean_swh_min_possible_m", "mean_swh_max_possible_m")]:
            assert d[lo].isna().equals(d[hi].isna())
            x = (d[hi] - d[lo]).dropna().to_numpy()
            assert (x >= -1e-12).all()
            x[np.abs(x) <= 1e-12] = 0
            v = x[x > 0] if metric != "wave" else x
            row = dict(level=level, metric=metric, total=len(d), finite=len(x), missing=len(d)-len(x),
                       zero=int((x == 0).sum()), zero_pct=float(100 * (x == 0).sum() / len(d)),
                       positive=int((x > 0).sum()), distribution_n=len(v))
            row.update({f"p{q}": float(np.quantile(v, q/100)) for q in [50, 90, 95, 99, 100]})
            ranges.append(row)
            range_curves.extend(curve(v, level=level, metric=metric))
    seg = pd.read_parquet(segmentation)
    assert seg.candidate_id.is_unique and set(seg.candidate_id) == set(panel.candidate_id)
    counts = seg.groupby("segmentation").size()
    summary["segmentation"] = {k: {"voyages": int(v), "pct": float(100*v/len(seg))} for k, v in counts.items()}
    (public / "manuscript_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for filename, rows in [("record_difference_summary.csv", records), ("record_cdf.csv", curves),
                           ("range_summary.csv", ranges), ("range_cdf.csv", range_curves),
                           ("vessel_type_summary.csv", types)]:
        pd.DataFrame(rows).to_csv(public / filename, index=False)
    (public / "provenance.json").write_text(json.dumps({"state": "pass", "sources": inputs,
        "checks": {"annual_record_parity": True, "monthly_keys_and_voyage_counts": True,
                   "paired_hour_totals_at_both_levels": True, "segmentation_population": True},
        "range_zero_tolerance": 1e-12,
        "record_comparison": "Unweighted paired hours within each departure-month record; wave shares use hours with wave values at both positions.",
        "privacy": "Only distributions and population summaries in public; no record keys or voyage identifiers.",
        "code_sha256": receipt(Path(__file__))["sha256"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
