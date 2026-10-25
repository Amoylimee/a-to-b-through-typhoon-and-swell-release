#!/usr/bin/env python3
"""Merge HKG/TWN/MAC into CHN using frozen voyage sufficient statistics.

Keep the public field names and preserve every physical city and voyage. Reuse
the accepted aggregation kernel, including distinct-vessel counting and wave
integrals. No trajectory, weather matching or bootstrap calculation is run.
"""
from pathlib import Path
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import monthly_release as m

REGIONS = {"HKG": "CHN", "TWN": "CHN", "MAC": "CHN"}
TABLES = {"country": "01_release_country_pair_month.csv", "city": "02_release_city_pair_month.csv"}
FILES = {"00_README.md", "03_FIELD_GUIDE.md", *TABLES.values()}
EXAMPLE = {"origin_country": "SGP", "dest_country": "CHN", "shiptype_group": "container", "departure_month": "2025-09"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipt(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": m.digest(path)}


def merge_city_codes(mapping):
    """Preserve existing CHN codes and append the other cities without collisions."""
    result = mapping.copy()
    require(result.city_id.is_unique and result.city_code.is_unique, "Source map is not one-to-one")
    require(result.city_id.str[:3].eq(result.city_code.str[:3]).all(), "Original city prefixes disagree")
    result["previous_city_code"] = result.city_code
    existing = result.loc[result.city_code.str.startswith("CHN_"), "city_code"]
    last = int(existing.str[-4:].astype(int).max()) if len(existing) else 0
    changed = result.loc[result.city_code.str[:3].isin(REGIONS)].sort_values("city_code")
    require(last + len(changed) <= 9999, "CHN code capacity exceeded")
    for number, index in enumerate(changed.index, start=last + 1):
        result.loc[index, "city_code"] = f"CHN_c{number:04d}"
    require(result.city_code.is_unique, "Merged city codes collide")
    require(result.city_code.str[:3].eq(result.city_id.str[:3].replace(REGIONS)).all(), "Merged prefix mismatch")
    return result


def relabel(frame, lookup):
    result = frame.copy()
    for side in ("origin", "dest"):
        result[side + "_country"] = result[side + "_country"].replace(REGIONS)
        if side + "_city_code" in result:
            result[side + "_city_code"] = result[side + "_city_code"].map(lookup)
            require(result[side + "_city_code"].notna().all(), "Unmapped city code")
    return result


def export(frame, fields, path):
    keys = m.COUNTRY_KEY if "origin_city_code" not in fields else m.CODED_KEY
    frame = frame[fields].sort_values(keys).reset_index(drop=True)
    m.check_values(frame, keys)
    require(not any(re.search("ci95|sampling|bootstrap", c) for c in frame), "Sampling field remains")
    for side in ("origin", "dest"):
        require(not frame[side + "_country"].isin(REGIONS).any(), "Unmerged country code")
        if side + "_city_code" in frame:
            require(frame[side + "_city_code"].str[:3].eq(frame[side + "_country"]).all(), "City-country mismatch")
    m.DOC.rounded_csv(frame, path)
    reread = pd.read_csv(path, low_memory=False)
    require(reread.columns.tolist() == fields, "Public schema changed")
    require(frame[keys].equals(reread[keys]), "Public keys changed on export")
    for field in fields[len(keys):]:
        if field in m.TEXT_FIELDS | m.BOOL_FIELDS:
            require(frame[field].astype("string").fillna("").equals(reread[field].astype("string").fillna("")), "Flag/text changed: " + field)
        else:
            places = m.DOC.places(field)
            tolerance = 0.5 * 10 ** (-places) + 1e-10 if places is not None else 0
            require(np.allclose(frame[field], reread[field], atol=tolerance, rtol=0, equal_nan=True), "Export precision: " + field)
    require(frame.isna().equals(reread.isna()), "Missingness changed on export")
    return frame, {"records": len(frame), "fields": len(fields), "voyages": int(frame.voyages.sum()),
                   "voyage_hours": float(frame.voyage_hours.sum()), "bytes": path.stat().st_size,
                   "sha256": m.digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-public", required=True, type=Path)
    parser.add_argument("--monthly-run", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    args = parser.parse_args()
    source, monthly, adopted, run, review = [p.resolve() for p in (args.source_public, args.monthly_run, args.source, args.run, args.review)]
    require(not run.exists() and not review.exists(), "Preserve existing output directories")
    require(run.is_relative_to((m.ROOT / "02_coding/.runs").resolve()), "Private output must remain under .runs")
    require(not any(run.is_relative_to(p) or review.is_relative_to(p) for p in (source, monthly, adopted)), "Output overlaps source")
    manifest = json.loads((source / "04_MANIFEST.json").read_text())
    require(set(manifest) == FILES, "Unexpected public manifest")
    receipts = [receipt(source / name) for name in sorted(FILES | {"04_MANIFEST.json"})]
    for record in receipts:
        name = Path(record["path"]).name
        if name in manifest:
            require({k: record[k] for k in ("bytes", "sha256")} == manifest[name], "Source manifest mismatch")
    config = json.loads((HERE.parent / "config/authorization.json").read_text())
    panel, _ = m.load_source(adopted, config)
    receipts += [receipt(adopted / name) for name in config["source_sha256"]]
    mapping_path = monthly / "private/city_code_map_PRIVATE.csv"
    receipts.append(receipt(mapping_path))
    old_mapping = pd.read_csv(mapping_path)
    mapping = merge_city_codes(old_mapping)
    require(set(mapping.city_id) == set(panel.origin_city) | set(panel.dest_city), "City population changed")
    lookup = mapping.set_index("previous_city_code").city_code
    old = {}
    schemas = {}
    for level, filename in TABLES.items():
        path = monthly / "private" / f"{level}_full_precision.parquet"
        receipts.append(receipt(path))
        schemas[level] = pd.read_csv(source / filename, nrows=0).columns.tolist()
        old[level] = pd.read_parquet(path)[schemas[level]]
    fields = [f for f in schemas["country"] if f not in m.COUNTRY_KEY]
    # Reproduce every old country record before changing its grouping.
    original = m.aggregate(panel, m.COUNTRY_KEY)
    checks = {"old_country_full_precision": m.compare(original, old["country"], m.COUNTRY_KEY, fields, "Original country parity")}
    # All identity keys remain private. Only their distinct counts are exported.
    for side in ("origin", "dest"):
        panel[side + "_country"] = panel[side + "_country"].replace(REGIONS)
    country = m.aggregate(panel, m.COUNTRY_KEY)
    city = relabel(old["city"], lookup)
    require(len(country) == 119475 and len(city) == 566294, "Unexpected merged record count")
    for side in ("origin", "dest"):
        panel[side + "_city_code"] = panel[side + "_city"].map(mapping.set_index("city_id").city_code)
    # Independent group membership, vessel counts and additive-field closure.
    additive = [c for c in schemas["country"] if c.endswith("_hours") or c in {"voyages", "voyages_with_tc300", "voyages_with_tc500"}]
    for level, table, keys in (("country", country, m.COUNTRY_KEY), ("city", city, m.CODED_KEY)):
        expected = panel.groupby(keys).agg(voyages=(m.SHIP, "size"), ships=(m.SHIP, "nunique"))
        actual = table.set_index(keys)[["voyages", "ships"]].sort_index()
        pd.testing.assert_frame_equal(expected.sort_index(), actual, check_dtype=False)
        require(int(table.voyages.sum()) == 1111800 and table.voyage_hours.sum() == 140832051, "Population changed")
        checks[level + "_membership_and_vessel_counts"] = "pass"
    rolled = city.groupby(m.COUNTRY_KEY)[additive].sum().reset_index()
    checks["city_to_country"] = m.compare(rolled, country, m.COUNTRY_KEY, additive, "City-country closure")
    unmapped = ~original.origin_country.isin(["CHN", *REGIONS]) & ~original.dest_country.isin(["CHN", *REGIONS])
    unaffected = country.loc[~country.origin_country.eq("CHN") & ~country.dest_country.eq("CHN")]
    checks["unchanged_non_china_records"] = m.compare(original.loc[unmapped], unaffected, m.COUNTRY_KEY, fields, "Unchanged non-China groups", atol=0)
    before_ships = relabel(original, lookup).groupby(m.COUNTRY_KEY).ships.sum()
    duplicate_ship_records = int((before_ships - country.set_index(m.COUNTRY_KEY).ships).gt(0).sum())
    run.mkdir(parents=True)
    public, private = run / "public", run / "private"
    public.mkdir()
    private.mkdir(mode=0o700)
    mapping.to_csv(private / "city_code_map_PRIVATE.csv", index=False)
    summaries = {}
    for level, frame in (("country", country), ("city", city)):
        frame, summaries[level] = export(frame, schemas[level], public / TABLES[level])
        frame.to_parquet(private / f"{level}_full_precision.parquet", index=False)
    # Independently compare the city CSV against the old strings: only keys change.
    old_strings = pd.read_csv(source / TABLES["city"], dtype=str, keep_default_na=False)
    expected_strings = relabel(old_strings, lookup).sort_values(m.CODED_KEY).reset_index(drop=True)
    actual_strings = pd.read_csv(public / TABLES["city"], dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(expected_strings, actual_strings)
    checks["city_all_non_key_cells_unchanged"] = "pass"
    text = (source / "00_README.md").read_text()
    text = text.replace("122,736", f"{len(country):,}")
    old_sentence = "Hong Kong and Taiwan have separate codes (`HKG` and `TWN`) and are labelled Hong Kong, China and Taiwan, China."
    statement = "China is coded `CHN`, including Hong Kong, China and Taiwan, China. Macao, China also maps to `CHN` but contributes no voyages in this sample. Country-pair records combine these locations before calculating vessel counts and exposure measures."
    require(old_sentence in text, "Expected region documentation missing")
    text = text.replace(old_sentence, statement)
    text = text.replace("These codes preserve links between records within this release.", "These codes preserve links between records within this release. Hong Kong and Taiwan use distinct `CHN_cNNNN` city codes, preserving their separate cities.")
    text = text.replace("repository identifiers and release licensing will be added at deposit.", "the reserved Zenodo DOI is `10.5281/zenodo.23453528`. This DOI will become active after publication; release licensing and the archive version remain to be finalised.")
    (public / "00_README.md").write_text(text)
    guide = (source / "03_FIELD_GUIDE.md").read_text().replace(old_sentence, statement)
    guide = guide.replace("Its four-digit number is assigned within a country", "Its four-digit number is assigned within a country or region")
    require(set(re.findall(r"^\| `([^`]+)` \|", guide, re.M)) == set(schemas["city"]), "Guide/schema mismatch")
    (public / "03_FIELD_GUIDE.md").write_text(guide)
    m.write_json(public / "04_MANIFEST.json", {name: {k:v for k,v in receipt(public/name).items() if k!='path'} for name in sorted(FILES)})
    require({p.name for p in public.iterdir()} == FILES | {"04_MANIFEST.json"}, "Unexpected public files")
    require(all(receipt(r["path"]) == r for r in receipts), "Frozen input changed")
    example = pd.read_csv(public / TABLES["country"], low_memory=False)
    example = example.loc[np.logical_and.reduce([example[k].eq(v) for k,v in EXAMPLE.items()])]
    require(len(example) == 1, "Example is not unique")
    report = {"status":"pass", "user_instruction":'Merge HKG, TWN and MAC into CHN without changing field names.',
              "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "country_code_mapping":REGIONS, "sources":receipts, "checks":checks, "tables":summaries,
              "cities":len(mapping), "countries_regions":len(set(panel.origin_country)|set(panel.dest_country)),
              "country_pairs":len(country[["origin_country","dest_country"]].drop_duplicates()),
              "city_pairs":len(city[["origin_city_code","dest_city_code"]].drop_duplicates()),
              "changed_city_codes":int(mapping.city_code.ne(mapping.previous_city_code).sum()),
              "records_requiring_vessel_deduplication":duplicate_ship_records,
              "source_files_unchanged":True, "script":receipt(Path(__file__)), "public":str(public)}
    m.write_json(run / "validation.json", report)
    review.mkdir(parents=True)
    for name in ("00_README.md","03_FIELD_GUIDE.md","04_MANIFEST.json"):
        shutil.copy2(public/name,review/("PUBLIC_"+name))
    shutil.copy2(run / "validation.json",review / "01_VALIDATION.json")
    example.to_csv(review / "03_MANUSCRIPT_EXAMPLE.csv",index=False)
    (review / "00_README.md").write_text(
        "# E008 — 中国相关地区统一归入 CHN\n\n"
        "用户于 2026-10-20 要求保留字段名，将香港、台湾及澳门的国家/地区代码统一归入 CHN。"
        "从冻结逐航次全精度统计重新聚合国家表，按船舶身份去重后计算船数；波浪与台风指标由小时数及积分重新计算。"
        "澳门没有分析航次。没有重建航迹、匹配天气或计算抽样区间。\n\n"
        f"公开目录：`{public}`。五个文件，字段名及 41/43 字段结构保持不变。\n\n"
        f"国家表 {len(country):,} 条，城市表 {len(city):,} 条；各含 1,111,800 航次、140,832,051 小时。"
        "193 个国家/地区、8,124 个国家对；5,366 个城市、166,107 个城市对。"
        "香港及台湾的 22 个城市改用不冲突的 CHN 编号，既有 CHN 编号不变。"
        f"{duplicate_ship_records:,} 个国家组合需要对原分组间重复出现的船舶去重。\n\n"
        "核验见 `01_VALIDATION.json`：旧国家表全精度复现；新组合成员和船数逐组核对；城市数值逐单元格不变；"
        "国家/城市小时数闭合；不涉及中国的国家记录不变；源文件哈希不变。\n\n"
        f"旧包保留：`{source}`。本目录是当前上传候选，旧候选不再用于上传。"
        "Zenodo 沿用预留 DOI `10.5281/zenodo.23453528`（草稿 23453528），本次未上传、发布、commit 或 push。"
        "数据许可及归档版本仍待最终发布时落实。\n")
    print(json.dumps({k:report[k] for k in ("status","tables","countries_regions","country_pairs","changed_city_codes","records_requiring_vessel_deduplication","public")},indent=2),flush=True)


if __name__ == "__main__":
    main()
