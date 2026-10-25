"""Behavioral checks for regrouping, missingness, sampling and public fields."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monthly_release as m


def voyage(cid, ship, month="2025-07", origin="CHN:A", dest="JPN:B", *, hours=10,
           located=10, matched=10, integral=20, counted=0, low=0, high=0,
           tc=0, possible=0, unbounded=0):
    return {"candidate_id": cid, m.SHIP: ship, "origin_city": origin, "dest_city": dest,
            "origin_country": origin[:3], "dest_country": dest[:3], "shiptype_group": "container",
            m.MONTH: month, "voyage_hours": hours, "track_hours": located,
            "unresolved_track_hours": hours-located, "wave_matched_hours": matched,
            "wave_unmatched_hours": hours-matched, "wave_integral_m_hours": integral,
            "wave_ge4m_matched_hours": 0, "tc300_positive_snapshot_hours": tc,
            "tc500_positive_snapshot_hours": tc, "main_wave_counted_hours": counted,
            "main_wave_min_integral": low, "main_wave_max_integral": high,
            "main_unbounded_hours": unbounded, "main_tc300_possible_hours": possible,
            "main_tc500_possible_hours": possible}


def test_regrouping_preserves_durations_and_counts_ships_once():
    panel = pd.DataFrame([voyage("1", "same", origin="CHN:A"),
                          voyage("2", "same", origin="CHN:C"),
                          voyage("3", "same", month="2025-08"),
                          voyage("4", "other", hours=20, integral=20, matched=20, located=20)])
    city = m.aggregate(panel, [*m.CITY, m.MONTH])
    annual = m.aggregate(panel, m.CITY)
    country = m.aggregate(panel, m.COUNTRY_KEY)
    m.adds_up(city, annual, m.CITY, "months")
    city = city.assign(origin_country=city.origin_city.str[:3], dest_country=city.dest_city.str[:3])
    m.adds_up(city, country, m.COUNTRY_KEY, "countries")
    july = country.loc[country.departure_month.eq("2025-07")].iloc[0]
    assert july.ships == 2 and july.voyages == 3
    assert july.mean_swh_matched_m == 60/40
    assert country.voyages.sum() == 4  # Single-voyage months are retained.


def test_matched_mean_is_preserved_even_outside_missing_position_range():
    panel = pd.DataFrame([voyage("1", "s", located=2, matched=2, integral=20, counted=8)])
    row = m.aggregate(panel, m.CITY).iloc[0]
    assert row.mean_swh_matched_m == 10
    assert row.mean_swh_min_possible_m == row.mean_swh_max_possible_m == 2
    assert row.mean_swh_constrained_m == 2 and row.mean_swh_constrained_shift_m == -8
    assert bool(row.wave_mean_was_constrained)


def test_missing_wave_mean_can_have_a_range_and_is_not_zero():
    panel = pd.DataFrame([voyage("1", "s", located=0, matched=0, integral=0, counted=10,
                                 low=10, high=30, possible=10)])
    row = m.aggregate(panel, m.CITY).iloc[0]
    assert pd.isna(row.mean_swh_matched_m)
    assert (row.mean_swh_min_possible_m, row.mean_swh_max_possible_m) == (1, 3)
    assert row.wave_mean_missing_reason == "no_matched_wave_hours_reachable_bounds_available"


def test_unbounded_hours_are_counted_once_in_cyclone_upper_share():
    panel = pd.DataFrame([voyage("1", "s", located=0, matched=0, integral=0,
                                 unbounded=10, possible=10)])
    row = m.aggregate(panel, m.CITY).iloc[0]
    assert row.tc300_at_least_pct == 0 and row.tc300_at_most_pct == 100
    assert row.tc300_bounded_possible_hours == 0
    assert pd.isna(row.mean_swh_min_possible_m)


def test_bootstrap_uses_whole_ship_clusters_and_finite_wave_draws():
    panel = pd.DataFrame([voyage("1", "a", matched=10, integral=20),
                          voyage("2", "a", matched=10, integral=30),
                          voyage("3", "b", located=0, matched=0, integral=0),
                          voyage("4", "c", located=0, matched=0, integral=0)])
    clusters = panel.groupby(m.SHIP)[m.BOOT_SUMS].sum().reset_index()
    key = ("CHN", "JPN", "container", "2025-07")
    result = m.bootstrap_one(key, clusters, m.COUNTRY_KEY)
    rng = np.random.default_rng(m.SEEDS.stable_seed(key))
    sampled = rng.integers(0, 3, size=(1000, 3))
    wave = []
    for draw in sampled:
        ships = clusters.iloc[draw]
        if ships.wave_matched_hours.sum() > 0:
            wave.append(ships.wave_integral_m_hours.sum()/ships.wave_matched_hours.sum())
    assert result["ships"] == 3
    assert result["mean_swh_matched_m"] == 2.5
    assert result["mean_swh_matched_m_valid_bootstrap_draws"] == len(wave) < 1000
    assert result["mean_swh_matched_ci95_low"] == np.quantile(wave, .025)
    assert result["mean_swh_matched_ci95_high"] == np.quantile(wave, .975)
    few = m.bootstrap_one(key, clusters.iloc[:2], m.COUNTRY_KEY)
    assert all(pd.isna(few[c]) for c in m.CI)


def test_codes_are_one_to_one_country_consistent_and_reused(tmp_path):
    path = tmp_path / "private" / "map.csv"
    cities = ["CHN:A", "CHN:B", "JPN:A"]
    first = m.city_codes(cities, path)
    second = m.city_codes(list(reversed(cities)), path)
    pd.testing.assert_frame_equal(first, second)
    assert first.city_code.nunique() == 3
    assert first.city_code.str.fullmatch(r"[A-Z]{3}_c[0-9]{4}").all()
    assert path.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="different city population"):
        m.city_codes(cities + ["CHN:C"], path)


def test_public_allowlist_rejects_names_extra_fields_and_invalid_codes():
    frame = m.aggregate(pd.DataFrame([voyage("1", "s")]), m.COUNTRY_KEY)
    for c in m.CI:
        frame[c] = np.nan
    public = frame[m.COUNTRY_KEY + m.MEASURES]
    m.public_schema(public, m.COUNTRY_KEY)
    with pytest.raises(ValueError, match="allowlist"):
        m.public_schema(public.assign(origin_city="CHN:A"), m.COUNTRY_KEY)
    coded = public.assign(origin_city_code="CHN:A", dest_city_code="JPN_c0001")[m.CODED_KEY + m.MEASURES]
    with pytest.raises(ValueError, match="City name"):
        m.public_schema(coded, m.CODED_KEY)


def test_csv_round_trip_keeps_nulls_counts_and_precision(tmp_path):
    frame = m.aggregate(pd.DataFrame([voyage("1", "s"),
                                     voyage("2", "t", month="2025-08", located=0, matched=0,
                                            integral=0, possible=10, unbounded=10)]), m.COUNTRY_KEY)
    for c in m.CI:
        frame[c] = np.nan
    report = m.export_table(frame, m.COUNTRY_KEY, tmp_path / "country.csv")
    assert report["rows"] == 2 and report["columns"] == 50
    assert report["records_without_wave_mean"] == 1
    assert report["voyages"] == 2
