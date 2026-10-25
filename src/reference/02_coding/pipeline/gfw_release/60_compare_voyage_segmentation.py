#!/usr/bin/env python3
"""E007 V2: are GFW-version voyages cut at the same port calls as voyages from dense AIS?

Both versions apply the same port-call and voyage rules; the reference AIS version sees the vessel
every few minutes, the GFW version once an hour. For each GFW-version voyage, the same
vessel's reference AIS version voyages are laid against it (6 h tolerance at each end):
- same: one reference AIS voyage with the same departure and arrival times and the same cities;
- same_times_other_city: the times agree but an endpoint city differs;
- reference_ais_intermediate_calls: two or more consecutive reference AIS voyages span it, so the GFW version
  passed over one or more intermediate city calls that dense AIS detected;
- inside_one_reference_ais_voyage: it lies inside a single longer reference AIS voyage, so the GFW version
  cut a call that dense AIS did not;
- other_overlap, no_reference_ais_voyage_nearby, vessel_not_in_reference_ais.
Each voyage end is also classed by its anchorage cell: a dock-flagged cell, or a cell at a
location with no dock-flagged cell (where stopping at the anchorage counts as arriving).
Finally, released-record exposure is recomputed from the 'same' voyages only and compared
with the released values.

    python 60_compare_voyage_segmentation.py --out <new .runs dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pyarrow.feather as ft
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODING = HERE.parents[1]
STEP6 = CODING / ".runs/2026-10-11_1907_E007-one-cell-release/assembly"
GFW_BASE = CODING / ".runs/2026-10-11_1907_E007-one-cell-voyage-base"
# Default to the adopted population; an explicit environment selects another frozen run.
STEP6 = Path(os.environ.get("E007_RELEASE_ASSEMBLY", STEP6))
GFW_BASE = Path(os.environ.get("E007_GFW_BASE", GFW_BASE))
REFERENCE_AIS_ROOT = CODING / ".runs/2026-09-10_1030_E001-2025-candidate-root/2025"
CATALOG = CODING / "data/frozen_inputs/anchorage_catalog/anchorages_analytical_city_v3_dock_adjusted_inclusive.csv"
KEY = ["origin_city", "dest_city", "shiptype_group"]
TOL = pd.Timedelta(hours=6)
SUMS = ["voyage_hours", "wave_matched_hours", "wave_integral_m_hours", "wave_ge4m_matched_hours",
        "tc300_positive_snapshot_hours", "tc500_positive_snapshot_hours"]


def classify(gfw: pd.DataFrame, reference_ais: pd.DataFrame) -> pd.DataFrame:
    out = []
    reference_ais = reference_ais.sort_values(["ship", "departure_time"])
    groups = {ship: block for ship, block in reference_ais.groupby("ship", sort=False)}
    for ship, block in gfw.groupby("ship", sort=False):
        k = groups.get(ship)
        res = pd.DataFrame({"candidate_id": block.candidate_id.to_numpy()})
        if k is None:
            res["segmentation"] = "vessel_not_in_reference_ais"
            res["reference_ais_voyages_spanned"] = 0
            res["missed_cities"] = ""
            out.append(res)
            continue
        kd, ka = pd.DatetimeIndex(k.departure_time).asi8, pd.DatetimeIndex(k.arrival_time).asi8
        ko, kt = k.origin_city.to_numpy(object), k.dest_city.to_numpy(object)
        gd, ga = pd.DatetimeIndex(block.departure_time).asi8, pd.DatetimeIndex(block.arrival_time).asi8
        go, gt = block.origin_city.to_numpy(object), block.dest_city.to_numpy(object)
        tol = TOL.value
        i0 = np.searchsorted(kd, gd - tol, "left")
        i1 = np.searchsorted(ka, ga + tol, "right") - 1
        # reference AIS voyages of one vessel do not overlap (checked in main), so arrivals sort with departures.
        n = np.maximum(0, i1 - i0 + 1)
        seg, spanned, missed = [], [], []
        for j in range(len(block)):
            a, b = i0[j], i1[j]
            if n[j] >= 1:
                dep_ok = abs(kd[a] - gd[j]) <= tol
                arr_ok = abs(ka[b] - ga[j]) <= tol
                if dep_ok and arr_ok and n[j] == 1:
                    seg.append("same" if (ko[a] == go[j] and kt[b] == gt[j]) else "same_times_other_city")
                elif dep_ok and arr_ok:
                    seg.append("reference_ais_intermediate_calls")
                    missed.append((j, ";".join(kt[a:b])))
                else:
                    seg.append("other_overlap")
                spanned.append(int(n[j]))
                continue
            c = np.searchsorted(kd, gd[j] + tol, "right") - 1
            if c >= 0 and kd[c] <= gd[j] + tol and ka[c] >= ga[j] - tol:
                seg.append("inside_one_reference_ais_voyage")
            elif c >= 0 and ka[c] > gd[j] - tol or (c + 1 < len(kd) and kd[c + 1] < ga[j] + tol):
                seg.append("other_overlap")
            else:
                seg.append("no_reference_ais_voyage_nearby")
            spanned.append(0)
        res["segmentation"] = seg
        res["reference_ais_voyages_spanned"] = spanned
        res["missed_cities"] = ""
        for j, cities in missed:
            res.iat[j, res.columns.get_loc("missed_cities")] = cities
        out.append(res)
    return pd.concat(out, ignore_index=True)


def pooled(frame: pd.DataFrame) -> pd.DataFrame:
    s = frame.groupby(KEY, observed=True)[SUMS].sum()
    s["voyages"] = frame.groupby(KEY, observed=True).size()
    return pd.DataFrame({
        "voyages": s.voyages,
        "wave_mean_matched_m": s.wave_integral_m_hours / s.wave_matched_hours.where(s.wave_matched_hours > 0),
        "share_wave_hours_ge4m": s.wave_ge4m_matched_hours / s.wave_matched_hours.where(s.wave_matched_hours > 0),
        "tc300_at_least_pct": 100 * s.tc300_positive_snapshot_hours / s.voyage_hours,
        "tc500_at_least_pct": 100 * s.tc500_positive_snapshot_hours / s.voyage_hours,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((CODING / ".runs").resolve()):
        raise ValueError("outputs must stay inside 02_coding/.runs")
    out.mkdir(parents=True, exist_ok=False)

    voyages = pd.read_parquet(STEP6 / "voyage_intervals.parquet",
                              columns=["candidate_id", "vessel_uid_clean", *KEY, "departure_time", "arrival_time", *SUMS])
    voyages["ship"] = voyages.vessel_uid_clean.astype(str)
    reference_ais = pd.concat([ft.read_table(f, columns=["vessel_uid_clean", "origin_city", "dest_city", "departure_time", "arrival_time"]).to_pandas()
                       for f in sorted(glob.glob(str(REFERENCE_AIS_ROOT / "*.feather")))])
    reference_ais["ship"] = reference_ais.vessel_uid_clean.astype(str)
    if (reference_ais.sort_values(["ship", "departure_time"]).groupby("ship").departure_time.shift(-1)
            < reference_ais.sort_values(["ship", "departure_time"]).arrival_time).any():
        raise ValueError("reference AIS voyages of one vessel overlap in time")
    seg = classify(voyages, reference_ais)
    voyages = voyages.merge(seg, on="candidate_id", how="left", validate="one_to_one")

    # Endpoint anchorage cells: the leg leaving the origin city and the leg reaching the destination.
    legs = pd.concat([pd.read_csv(f, dtype={"vessel_id": str, "trip_start_anchorage_s2id": str, "trip_end_anchorage_s2id": str}) for f in sorted(glob.glob(str(GFW_BASE / "input_view/our_od/2025/*.csv")))])
    for c in ("trip_start", "trip_end"):
        legs[c] = pd.to_datetime(legs[c].str.replace(" UTC", "", regex=False), utc=True)
    legs = legs.rename(columns={"vessel_id": "ship"})
    voyages = voyages.merge(legs[["ship", "trip_start", "trip_start_anchorage_s2id"]].rename(columns={"trip_start": "departure_time"}),
                            on=["ship", "departure_time"], how="left", validate="one_to_one")
    voyages = voyages.merge(legs[["ship", "trip_end", "trip_end_anchorage_s2id"]].rename(columns={"trip_end": "arrival_time"}),
                            on=["ship", "arrival_time"], how="left", validate="one_to_one")
    catalog = pd.read_csv(CATALOG, usecols=["s2id", "dock"], dtype=str).drop_duplicates("s2id").set_index("s2id").dock
    # 15 catalog cells carry no dock value; like dock=False they are not dock-flagged.
    dock_value_missing = int(catalog.isna().sum())
    catalog = catalog.str.lower().eq("true")
    for end, column in (("origin", "trip_start_anchorage_s2id"), ("dest", "trip_end_anchorage_s2id")):
        flag = voyages[column].astype(str).map(catalog)
        if flag.isna().any():
            raise ValueError(f"{end} anchorage cells missing from the catalog: {int(flag.isna().sum())}")
        voyages[f"{end}_end_at_dock_cell"] = flag.astype(bool)

    # Later releases name the matched-hour wave mean interval mean_swh_matched.
    wave_ci = "mean_swh_matched" if "mean_swh_matched_ci95_low" in pq.read_schema(STEP6 / "release_full_precision.parquet").names else "mean_swh_full_voyage"
    released = pd.read_parquet(STEP6 / "release_full_precision.parquet", columns=[*KEY, "voyages", f"{wave_ci}_ci95_low",
                                                                                   f"{wave_ci}_ci95_high", "tc300_at_least_pct_ci95_low",
                                                                                   "tc300_at_least_pct_ci95_high", "tc500_at_least_pct_ci95_low",
                                                                                   "tc500_at_least_pct_ci95_high"])
    voyages["released"] = pd.MultiIndex.from_frame(voyages[KEY]).isin(pd.MultiIndex.from_frame(released[KEY]))
    voyages[["candidate_id", *KEY, "released", "segmentation", "reference_ais_voyages_spanned", "missed_cities",
             "origin_end_at_dock_cell", "dest_end_at_dock_cell", "voyage_hours"]].to_parquet(out / "voyage_segmentation.parquet", index=False)

    def shares(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        g = frame.groupby([*by, "segmentation"], observed=True).agg(voyages=("candidate_id", "size"), hours=("voyage_hours", "sum")).reset_index()
        tot = g.groupby(by, observed=True)[["voyages", "hours"]].transform("sum") if by else g[["voyages", "hours"]].sum()
        g["share_of_voyages"] = g.voyages / tot["voyages"] if by else g.voyages / tot["voyages"]
        g["share_of_hours"] = g.hours / tot["hours"] if by else g.hours / tot["hours"]
        return g

    shares(voyages.assign(scope=np.where(voyages.released, "released", "below_threshold")), ["scope"]).to_csv(out / "segmentation_shares.csv", index=False)
    shares(voyages.loc[voyages.released], ["shiptype_group"]).to_csv(out / "segmentation_by_shiptype.csv", index=False)
    voyages["endpoint_cells"] = np.select(
        [voyages.origin_end_at_dock_cell & voyages.dest_end_at_dock_cell, ~voyages.origin_end_at_dock_cell & ~voyages.dest_end_at_dock_cell],
        ["both_dock", "both_no_dock_location"], "one_each")
    shares(voyages.loc[voyages.released], ["endpoint_cells"]).to_csv(out / "segmentation_by_endpoint_cells.csv", index=False)
    missed = voyages.loc[voyages.released & voyages.segmentation.eq("reference_ais_intermediate_calls"), "missed_cities"].str.split(";").explode()
    missed.value_counts().rename_axis("missed_city").rename("times_missed").head(200).to_csv(out / "most_missed_intermediate_cities.csv")

    # Exposure from voyages cut the same way in both versions, against all voyages.
    rel = voyages.loc[voyages.released]
    full, same = pooled(rel), pooled(rel.loc[rel.segmentation.eq("same")])
    compare = full.join(same, rsuffix="_same_only", how="inner").reset_index().merge(released, on=KEY)
    compare = compare.loc[compare.voyages_same_only >= 5]
    for metric, ci in (("wave_mean_matched_m", wave_ci), ("tc300_at_least_pct", "tc300_at_least_pct"),
                       ("tc500_at_least_pct", "tc500_at_least_pct")):
        compare[f"{metric}_diff"] = compare[f"{metric}_same_only"] - compare[metric]
        half = (compare[f"{ci}_ci95_high"] - compare[f"{ci}_ci95_low"]) / 2
        compare[f"{metric}_diff_exceeds_ci_half_width"] = compare[f"{metric}_diff"].abs() > half
    compare["share_wave_hours_ge4m_diff"] = compare.share_wave_hours_ge4m_same_only - compare.share_wave_hours_ge4m
    compare.to_csv(out / "record_exposure_same_segmentation_only.csv", index=False)

    q = lambda s: {p: float(s.abs().quantile(p)) for p in (0.5, 0.9, 0.99)}
    counts = voyages.loc[voyages.released].segmentation.value_counts()
    summary = {
        "released_voyages": int(voyages.released.sum()),
        "released_segmentation_shares": (counts / counts.sum()).round(4).to_dict(),
        "released_endpoint_cells": voyages.loc[voyages.released].endpoint_cells.value_counts().to_dict(),
        "released_dest_end_at_no_dock_location_share": float((~voyages.loc[voyages.released, "dest_end_at_dock_cell"]).mean()),
        "records_compared_same_only_ge5": len(compare),
        "abs_diff_quantiles": {m: q(compare[f"{m}_diff"].dropna()) for m in
                               ("wave_mean_matched_m", "share_wave_hours_ge4m", "tc300_at_least_pct", "tc500_at_least_pct")},
        "share_records_diff_exceeds_ci_half_width": {m: float(compare[f"{m}_diff_exceeds_ci_half_width"].mean()) for m in
                                                     ("wave_mean_matched_m", "tc300_at_least_pct", "tc500_at_least_pct")},
    }
    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()
    execution = {"step": "E007 V2: voyage segmentation against dense-AIS voyages", "state": "pass", "git_head": git("rev-parse", "HEAD"),
                 "git_dirty_experiment": bool(git("status", "--porcelain", "--", ".")), "tolerance_hours": 6, "catalog_cells_without_dock_value": dock_value_missing,
                 "reference_ais_root": str(REFERENCE_AIS_ROOT.relative_to(CODING)), "gfw_base": str(GFW_BASE), "release_assembly": str(STEP6), "wave_interval": wave_ci, **summary}
    (out / "execution.json").write_text(json.dumps(execution, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
