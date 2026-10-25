"""E007 city-bound port visits, with explicit unresolved observation evidence.

Adopted on 2026-10-13 and integrated into the pipeline on 2026-10-14. Numerical
3/4 km, 0.2/0.5 kn and 4 h thresholds are fixed. Annexed stop-entry and uncertainty
rules are recorded in config/port_segmentation_revision.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ENTRY_KM, EXIT_KM, STOP_ENTER, STOP_EXIT, GAP_HOURS = 3., 4., .2, .5, 4.
EARTH_KM = 6371.0088
EXTRA_COLUMNS = ["opening_anchorage_idx", "entry_rule", "entry_bracket_start",
                 "exit_bracket_start", "entry_gap_hours", "exit_gap_hours",
                 "possible_short_stop", "ambiguous_city", "stop_anchorage_idx"]


def distances(lat1, lon1, lat2, lon2):
    a, b, c, d = (np.deg2rad(v) for v in (lat1, lon1, lat2, lon2))
    q = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*EARTH_KM*np.arcsin(np.sqrt(np.clip(q, 0, 1)))


def cell_allowance_km(lat):
    """A full-cell diagonal, without an undocumented cell-centre assumption."""
    return np.sqrt((EARTH_KM*np.deg2rad(.01))**2 +
                   (EARTH_KM*np.deg2rad(.01)*np.cos(np.deg2rad(lat)))**2)


def nearest_unique(lat, lon, tree):
    """Exact original BallTree query, evaluating each repeated grid point once."""
    if len(lat)==0:
        return np.empty(0,dtype=np.int64),np.empty(0,dtype=float)
    coordinates,inverse=np.unique(np.column_stack([lat,lon]),axis=0,return_inverse=True)
    distance,index=tree.query(np.deg2rad(coordinates),k=1)
    return index[:,0].astype(np.int64)[inverse],(distance[:,0]*EARTH_KM)[inverse]


def visit_builder(anchor_cities):
    cities = np.asarray(anchor_cities, dtype=str)
    city_trees = {}

    def build(vessel_uid, registry_imo_clean, times, speeds, latitudes, longitudes,
              nearest_idx, nearest_dist_km, anchorage_latitudes, anchorage_longitudes):
        times = times.astype("datetime64[ns]")
        n = len(times)
        if n == 0:
            return []
        if (np.diff(times.astype("int64")) <= 0).any():
            raise ValueError("port state requires unique chronological vessel points")
        visits, v, seq = [], None, 0
        allowance = cell_allowance_km(np.asarray(latitudes, dtype=float))
        # Calculate all pair geometry once, rather than repeatedly running
        # NumPy scalar kernels in the state-machine loop.
        dt_hours = np.diff(times.astype('int64'))/3.6e12
        pair_distance = distances(np.asarray(latitudes)[:-1], np.asarray(longitudes)[:-1],
                                  np.asarray(latitudes)[1:], np.asarray(longitudes)[1:])
        # Both samples are binned to the 0.01-degree grid by one rule, so their
        # displacement is uncertain by at most one cell diagonal, not one per
        # sample (config/short_stop_allowance_revision.json).
        possible_stop = np.r_[False, np.maximum(0.,pair_distance-np.maximum(allowance[:-1],allowance[1:]))/(dt_hours+1)/1.852 < STOP_ENTER]

        def open_new(i):
            near = int(nearest_idx[i])
            bracket = times[i-1] if i else times[i]
            return {"anchor": near, "city": cities[near], "first": i, "last": i,
                    "last_entry": i, "entry_bracket": bracket,
                    "entry_gap": float((times[i]-bracket)/np.timedelta64(1, "h")),
                    "entry_rule": "nominal_3km" if nearest_dist_km[i] <= ENTRY_KM else "stop_supported_annulus",
                    "has_stop": False, "has_gap": False, "in_stop": False,
                    "possible_stop": False, "ambiguous_city": False,
                    "stop_anchor": -1, "count": 0}

        def close(end):
            nonlocal seq
            seq += 1
            complete = end is not None
            exit_time = times[end] if complete else pd.NaT
            last = times[v["last"]]
            exit_gap = float((exit_time-last)/np.timedelta64(1, "h")) if complete else np.nan
            representative = v["stop_anchor"] if v["stop_anchor"] >= 0 else v["anchor"]
            return {"vessel_uid": vessel_uid, "registry_imo_clean": registry_imo_clean,
                    "visit_id_internal": f"{vessel_uid}_v{seq:05d}",
                    "anchorage_idx": representative, "opening_anchorage_idx": v["anchor"],
                    "entry_time": times[v["first"]], "exit_time": exit_time,
                    "first_seen_time": times[v["first"]], "last_seen_time": last,
                    "has_entry": True, "has_exit": complete,
                    "has_stop": v["has_stop"], "has_gap": v["has_gap"],
                    "visit_confidence": "high" if complete and (v["has_stop"] or v["has_gap"]) else "low",
                    "n_ais_points_in_visit": v["count"] + int(complete),
                    "entry_rule": v["entry_rule"], "entry_bracket_start": v["entry_bracket"],
                    "exit_bracket_start": last, "entry_gap_hours": v["entry_gap"],
                    "exit_gap_hours": exit_gap,
                    "possible_short_stop": bool(not v["has_stop"] and not v["has_gap"] and v["count"] >= 2 and v["possible_stop"]),
                    "ambiguous_city": v["ambiguous_city"], "stop_anchorage_idx": v["stop_anchor"]}

        for i in range(n):
            near, near_d, speed = int(nearest_idx[i]), float(nearest_dist_km[i]), speeds[i]
            if v is not None:
                same = cities[near] == v["city"]
                # A different city cannot donate its spatial support.
                visit_d = near_d if same else float(distances(latitudes[i], longitudes[i],
                                            anchorage_latitudes[v["anchor"]], anchorage_longitudes[v["anchor"]]))
                if not same and visit_d > EXIT_KM:
                    # The closest catalog cell may be foreign, while a second
                    # cell still belongs to the opening city's support region.
                    # Query the complete city, not only the locked entry cell.
                    if v['city'] not in city_trees:
                        indexes=np.flatnonzero(cities==v['city'])
                        city_trees[v['city']]=BallTree(np.deg2rad(np.column_stack([
                            np.asarray(anchorage_latitudes)[indexes],np.asarray(anchorage_longitudes)[indexes]])),metric='haversine')
                    d,_=city_trees[v['city']].query(np.deg2rad([[latitudes[i],longitudes[i]]]),k=1)
                    visit_d=float(d[0,0]*EARTH_KM)
                if visit_d > EXIT_KM:
                    visits.append(close(i))
                    v = None  # re-evaluate this very sample below
            if v is None:
                nominal = near_d <= ENTRY_KM
                stopped_annulus = near_d <= EXIT_KM and np.isfinite(speed) and speed < STOP_ENTER
                if not (nominal or stopped_annulus):
                    continue
                v = open_new(i)
                same, visit_d = True, near_d
            previous = v["last"]
            if v["count"] and previous == i-1:
                dt = dt_hours[i-1]
                if dt >= GAP_HOURS:
                    if same and cities[int(nearest_idx[previous])] == v["city"]:
                        v["has_gap"] = True
                    else:
                        v["ambiguous_city"] = True
                if np.isfinite(speed):
                    if possible_stop[i]:
                        v["possible_stop"] = True
            v["count"] += 1
            v["last"] = i
            if visit_d <= ENTRY_KM:
                v["last_entry"] = i
            if np.isfinite(speed):
                if speed < STOP_ENTER:
                    if same:
                        v["has_stop"] = True
                        v["in_stop"] = True
                        if v["stop_anchor"] < 0:
                            v["stop_anchor"] = near
                    else:
                        v["ambiguous_city"] = True
                elif v["in_stop"] and speed > STOP_EXIT:
                    v["in_stop"] = False
        if v is not None:
            visits.append(close(None))
        return visits
    return build


def guard_candidates(frame, visits, points=None):
    """Return reason columns for unresolved boundaries/visits/source motion.

    Endpoints use the last high visit ending at departure and first high visit
    beginning at arrival. Calls may belong to same-city episodes; matching
    by their boundary times handles these without using unstable visit IDs.
    """
    frame = frame.copy()
    reason_sets = [set() for _ in range(len(frame))]
    groups = {str(s): g.sort_values("entry_time") for s, g in visits.groupby("vessel_uid", sort=False)}
    p_groups = {str(s): g.sort_values("UTC") for s, g in points.groupby("vessel_uid", sort=False)} if points is not None else {}
    for ship, ix in frame.groupby("vessel_uid_clean", sort=False).indices.items():
        calls = groups.get(str(ship))
        if calls is None:
            raise ValueError(f"candidate without port history: {ship}")
        high = calls.loc[calls.visit_confidence.eq("high")]
        end_map = {pd.Timestamp(v.exit_time): v for v in high.itertuples()}
        start_map = {pd.Timestamp(v.entry_time): v for v in high.itertuples()}
        issues = calls.loc[calls.possible_short_stop | calls.ambiguous_city |
                           calls.entry_gap_hours.ge(GAP_HOURS) | calls.exit_gap_hours.ge(GAP_HOURS)]
        bad_motion = []
        p = p_groups.get(str(ship))
        if p is not None and len(p) > 1:
            t = pd.to_datetime(p.UTC, utc=True).astype("int64").to_numpy()
            hours = np.diff(t)/3.6e12
            distance = distances(p.lat.to_numpy()[:-1], p.lon.to_numpy()[:-1], p.lat.to_numpy()[1:], p.lon.to_numpy()[1:])
            bad_motion = list(zip(t[:-1][distance/(hours+1)/1.852 > 36], t[1:][distance/(hours+1)/1.852 > 36]))
        for j in ix:
            row = frame.iloc[j]
            departure, arrival = pd.Timestamp(row.departure_time), pd.Timestamp(row.arrival_time)
            origin, dest = end_map.get(departure), start_map.get(arrival)
            reasons = reason_sets[j]
            if origin is None or dest is None:
                reasons.add("endpoint_visit_not_recovered")
            if origin is not None:
                if origin.exit_gap_hours >= GAP_HOURS: reasons.add("unresolved_departure_bracket")
                if origin.ambiguous_city: reasons.add("ambiguous_origin_city")
            if dest is not None:
                if dest.entry_gap_hours >= GAP_HOURS: reasons.add("unresolved_arrival_bracket")
                if dest.ambiguous_city: reasons.add("ambiguous_destination_city")
            for call in issues.itertuples():
                if call.analytical_city_id in (row.origin_city, row.dest_city): continue
                # No observation after the last sample: do not fabricate a closed
                # intermediate visit. Its observed support still overlaps.
                end = call.exit_time if pd.notna(call.exit_time) else call.last_seen_time
                if end > departure and call.entry_time < arrival:
                    if call.possible_short_stop: reasons.add("unresolved_intermediate_short_stop")
                    if call.ambiguous_city: reasons.add("unresolved_intermediate_city")
                    if call.entry_gap_hours >= GAP_HOURS or call.exit_gap_hours >= GAP_HOURS:
                        reasons.add("unresolved_intermediate_boundary")
            if any(a < arrival.value and b > departure.value for a, b in bad_motion):
                reasons.add("infeasible_observed_motion")
    frame["segmentation_review_reasons"] = [";".join(sorted(r)) for r in reason_sets]
    frame["segmentation_eligible"] = frame.segmentation_review_reasons.eq("")
    return frame
