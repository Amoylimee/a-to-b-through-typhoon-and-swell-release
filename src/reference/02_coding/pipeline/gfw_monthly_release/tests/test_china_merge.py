"""Regrouping must preserve cities and deduplicate vessels across old regions."""
from pathlib import Path
import importlib.util
import pandas as pd
import pytest
from test_monthly_release import voyage

path = Path(__file__).resolve().parents[1] / "20_finalize_public_release/10_merge_china_regions.py"
spec = importlib.util.spec_from_file_location("china_merge", path)
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


def test_colliding_city_suffixes_are_reassigned_and_existing_china_codes_survive():
    source = pd.DataFrame({"city_id":["CHN:A","HKG:B","TWN:C","MAC:D","JPN:E"],
                           "city_code":["CHN_c0001","HKG_c0001","TWN_c0001","MAC_c0001","JPN_c0001"]})
    result = merge.merge_city_codes(source).set_index("city_id")
    assert result.city_code.is_unique and len(result) == 5
    assert result.loc["CHN:A","city_code"] == "CHN_c0001"
    assert result.loc["JPN:E","city_code"] == "JPN_c0001"
    assert result.loc[["HKG:B","TWN:C","MAC:D"],"city_code"].str.startswith("CHN_").all()
    assert source.loc[1,"city_code"] == "HKG_c0001"


def test_merge_counts_same_ship_once_and_uses_duration_weighted_values():
    panel = pd.DataFrame([voyage("a","same",origin="CHN:A",hours=10,matched=10,located=10,integral=10,tc=1),
                          voyage("b","same",origin="TWN:B",hours=30,matched=30,located=30,integral=90,tc=3),
                          voyage("c","other",origin="HKG:C",hours=10,matched=0,located=0,integral=0,counted=10,low=10,high=40,possible=10)])
    merged = merge.relabel(panel, {})
    table = merge.m.aggregate(merged,merge.m.COUNTRY_KEY)
    assert len(table) == 1
    row = table.iloc[0]
    assert row.voyages == 3 and row.ships == 2
    assert row.mean_swh_matched_m == 2.5
    assert row.mean_swh_min_possible_m == 110/50
    assert row.mean_swh_max_possible_m == 140/50
    assert row.tc300_at_least_pct == pytest.approx(8)
    assert row.tc300_at_most_pct == pytest.approx(28)
    assert row.wave_bound_coverage == 1


def test_cross_region_voyage_becomes_china_to_china_without_merging_cities():
    frame = pd.DataFrame({"origin_country":["TWN"],"dest_country":["HKG"],
                          "origin_city_code":["TWN_c0001"],"dest_city_code":["HKG_c0001"],"voyages":[7]})
    result = merge.relabel(frame,{"TWN_c0001":"CHN_c0002","HKG_c0001":"CHN_c0003"})
    assert result.origin_country.iloc[0] == result.dest_country.iloc[0] == "CHN"
    assert result.origin_city_code.iloc[0] != result.dest_city_code.iloc[0]
    assert result.voyages.iloc[0] == 7
