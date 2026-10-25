# Monthly voyage exposure — field guide

The two tables describe the 2025 Global Fishing Watch (GFW) voyage sample at country-pair and coded city-pair levels. Each row combines a departure location, an arrival location, a vessel type and a departure month. The direction of travel is preserved. All analysed voyages contribute to both tables, including groups with one voyage.

`departure_month` uses Coordinated Universal Time (UTC). The full duration of a voyage is assigned to its departure month, including time spent sailing in the following month. Monthly records therefore describe voyages departing in that month.

The country table has 50 fields. The city table adds `origin_city_code` and `dest_city_code`, giving 52 fields. A city code is stable within this release and has the form `CHN_c0001`. Its four-digit number is assigned within a country and has no geographic meaning. The public files contain no city-name lookup table, coordinates, vessel identifiers or exact departure and arrival times.

## Exposure, coverage and intervals

Cyclone exposure shares use the total duration of the contributing voyages. Lower shares count time when a known position falls within the stated radius of a cyclone centre. Upper shares also include missing-position time when exposure remains possible under the study's travel constraints. Time without a usable travel constraint contributes to the upper estimate at both radii.

Mean significant wave height and the high-wave share use hours with matched wave values. The wave-height range also includes missing-position hours for which valid wave values can be found within the reachable area. `wave_bound_coverage` gives the share of voyage time represented by this range. The matched mean and the range can describe different amounts of time; the matched mean can fall outside the range. The constrained mean is a secondary comparison measure obtained by limiting the matched mean to that range.

The 95% sampling intervals describe variation across vessel samples. Each of 1,000 resamples draws vessels with replacement within the record, keeping all voyages of each selected vessel together. The interval uses the 2.5th and 97.5th percentiles. The random seed is 20260923, with a deterministic seed derived for each record. Intervals are blank for records with fewer than three vessels. Wave intervals use resamples with a positive matched-wave duration; the number of usable resamples can be below 1,000. Sampling intervals and ranges for missing positions describe different sources of uncertainty.

An empty cell means that the value is unavailable or the flag does not apply. It does not represent zero. `wave_mean_missing_reason` distinguishes records with a wave-height range but no matched mean from those with neither. This field is blank when a matched mean is available.

Hours are written to one decimal place; metres, fractions and percentages to three. Counts are integers and flags are `True` or `False`. Calculations use full precision. Independently rounded fields can differ in their last digit when added or divided.

## Fields

| Variable | Unit | Meaning |
|---|---|---|
| `origin_country` | Code | Country or territory of departure. |
| `dest_country` | Code | Country or territory of arrival. |
| `origin_city_code` | Code | Coded departure city; present in the city table only. |
| `dest_city_code` | Code | Coded arrival city; present in the city table only. |
| `shiptype_group` | Category | Vessel type. The nine categories are listed below. |
| `departure_month` | YYYY-MM | Month of departure in UTC. |
| `voyages` | Count | Number of analysed voyages in the record. |
| `ships` | Count | Number of distinct vessels. Each vessel is counted once within the record. |
| `voyage_hours` | h | Total duration of the contributing voyages. |
| `track_hours` | h | Duration with recorded or reconstructed positions at the hourly evaluation times. |
| `position_coverage` | 0–1 | `track_hours` divided by `voyage_hours`. |
| `unresolved_track_hours` | h | Duration without a position; `voyage_hours` minus `track_hours`. |
| `wave_matched_hours` | h | Duration with positions and valid matched significant wave height. |
| `wave_unmatched_hours` | h | `voyage_hours` minus `wave_matched_hours`. |
| `wave_ge4m_matched_hours` | h | Matched duration with significant wave height of at least 4 m. |
| `wave_coverage` | 0–1 | `wave_matched_hours` divided by `voyage_hours`. |
| `mean_swh_matched_m` | m | Duration-weighted mean significant wave height over matched time. This is the primary wave estimate. |
| `mean_swh_matched_ci95_low` | m | Lower endpoint of the 95% sampling interval for the matched mean. |
| `mean_swh_matched_ci95_high` | m | Upper endpoint of the 95% sampling interval for the matched mean. |
| `mean_swh_min_possible_m` | m | Lower mean wave height over matched time and missing-position time with usable reachable-area wave values. |
| `mean_swh_max_possible_m` | m | Upper mean wave height over the same time. |
| `mean_swh_possible_range_m` | m | Upper mean wave height minus lower mean wave height. |
| `wave_bound_hours` | h | Duration covered by the wave-height range; `wave_matched_hours` plus `wave_unknown_counted_hours`. |
| `wave_bound_coverage` | 0–1 | `wave_bound_hours` divided by `voyage_hours`. |
| `wave_unboundable_hours` | h | Duration outside the wave-height range; `voyage_hours` minus `wave_bound_hours`. |
| `mean_swh_constrained_m` | m | Secondary estimate obtained by limiting the matched mean to the lower and upper possible means. |
| `mean_swh_constrained_shift_m` | m | Constrained mean minus matched mean. |
| `wave_mean_was_constrained` | Flag | Whether limiting the matched mean changed it by more than 10⁻⁹ m; blank when the matched mean is unavailable. |
| `wave_mean_missing_reason` | Text | Reason for an unavailable matched mean; the two values are defined below. |
| `share_wave_hours_ge4m` | 0–1 | `wave_ge4m_matched_hours` divided by `wave_matched_hours`. |
| `share_wave_hours_ge4m_ci95_low` | 0–1 | Lower endpoint of the 95% sampling interval for the high-wave share. |
| `share_wave_hours_ge4m_ci95_high` | 0–1 | Upper endpoint of the 95% sampling interval for the high-wave share. |
| `voyages_with_tc300` | Count | Voyages with at least one known-position hour within 300 km of a cyclone centre. |
| `voyages_with_tc500` | Count | Voyages with at least one known-position hour within 500 km of a cyclone centre. |
| `single_ship` | Flag | Whether the record contains one vessel. |
| `sampling_interval_unavailable_fewer_than_3_ships` | Flag | Whether the record has fewer than three vessels and sampling intervals are omitted. |
| `unbounded_hours` | h | Missing-position duration without a usable travel constraint, included in upper cyclone exposure. |
| `wave_unknown_counted_hours` | h | Missing-position duration with usable reachable-area wave values, included in the wave-height range. |
| `tc300_hours_at_least` | h | Known-position duration within 300 km of a cyclone centre. |
| `tc300_bounded_possible_hours` | h | Missing-position duration with a usable travel constraint that allows exposure within 300 km. |
| `tc300_hours_at_most` | h | `tc300_hours_at_least` plus `tc300_bounded_possible_hours` plus `unbounded_hours`. |
| `tc300_at_least_pct` | % | Lower cyclone exposure share; 100 × `tc300_hours_at_least` / `voyage_hours`. |
| `tc300_at_most_pct` | % | Upper cyclone exposure share; 100 × `tc300_hours_at_most` / `voyage_hours`. |
| `tc300_at_least_pct_ci95_low` | % | Lower endpoint of the 95% sampling interval for the lower 300 km share. |
| `tc300_at_least_pct_ci95_high` | % | Upper endpoint of the 95% sampling interval for the lower 300 km share. |
| `tc500_hours_at_least` | h | Known-position duration within 500 km of a cyclone centre. |
| `tc500_bounded_possible_hours` | h | Missing-position duration with a usable travel constraint that allows exposure within 500 km. |
| `tc500_hours_at_most` | h | `tc500_hours_at_least` plus `tc500_bounded_possible_hours` plus `unbounded_hours`. |
| `tc500_at_least_pct` | % | Lower cyclone exposure share; 100 × `tc500_hours_at_least` / `voyage_hours`. |
| `tc500_at_most_pct` | % | Upper cyclone exposure share; 100 × `tc500_hours_at_most` / `voyage_hours`. |
| `tc500_at_least_pct_ci95_low` | % | Lower endpoint of the 95% sampling interval for the lower 500 km share. |
| `tc500_at_least_pct_ci95_high` | % | Upper endpoint of the 95% sampling interval for the lower 500 km share. |

## Vessel types and missing-value reasons

The vessel types are `bulk_carrier`, `container`, `crude_oil_tanker`, `gas_tanker`, `general_cargo`, `product_chemical_tanker`, `reefer`, `ro_ro_vehicle` and `other_retained`.

`no_matched_wave_hours_reachable_bounds_available` means that a matched wave mean is unavailable, while a wave-height range is available from reachable-area wave values. `no_matched_or_reachable_wave_hours` means that neither the matched mean nor the range is available.

## Combining records

Voyage counts and durations can be added across months or city pairs within either table. Means, fractions and percentages use their corresponding duration weights. For example, a combined matched wave mean uses `wave_matched_hours` as weights. Vessel counts require deduplication across groups and cannot be obtained by adding `ships`. The country table calculates them directly from vessel identities before those identities are removed. Sampling intervals also require resampling the combined vessel sample; averaging interval endpoints does not reproduce them.

The country and city tables contain the same voyages at different geographic levels. Adding both tables would count the voyages twice. Comparisons across departure months retain the full voyage-duration convention described above.
