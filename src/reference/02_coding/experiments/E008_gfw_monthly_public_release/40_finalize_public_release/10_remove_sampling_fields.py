#!/usr/bin/env python3
"""Prepare the current public schema from validated E008 CSVs without recalculation."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from itertools import zip_longest
from pathlib import Path


REMOVED = (
    "mean_swh_matched_ci95_low", "mean_swh_matched_ci95_high",
    "share_wave_hours_ge4m_ci95_low", "share_wave_hours_ge4m_ci95_high",
    "tc300_at_least_pct_ci95_low", "tc300_at_least_pct_ci95_high",
    "tc500_at_least_pct_ci95_low", "tc500_at_least_pct_ci95_high",
    "sampling_interval_unavailable_fewer_than_3_ships",
)
TABLES = (
    ("01_release_country_pair_month.csv", 122736, 50),
    ("02_release_city_pair_month.csv", 566294, 52),
)
PUBLIC_FILES = {"00_README.md", "03_FIELD_GUIDE.md", *[t[0] for t in TABLES]}
EXAMPLE = dict(origin_country="SGP", dest_country="CHN",
               shiptype_group="container", departure_month="2025-09")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def file_record(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replace_once(text, old, new):
    require(text.count(old) == 1, "Unexpected source documentation: " + old)
    return text.replace(old, new)


def project_csv(source, target, expected_columns):
    """Use strings throughout so dates, precision, flags and empty cells survive."""
    with source.open(newline="", encoding="utf-8") as src:
        reader = csv.reader(src)
        header = next(reader)
        require(len(header) == len(set(header)) == expected_columns, "Unexpected source schema")
        require(set(REMOVED) <= set(header), "Source is missing a removal field")
        indexes = [i for i, field in enumerate(header) if field not in REMOVED]
        with target.open("x", newline="", encoding="utf-8") as dst:
            writer = csv.writer(dst, lineterminator="\n")
            writer.writerow([header[i] for i in indexes])
            for row in reader:
                require(len(row) == len(header), "Malformed source CSV row")
                writer.writerow([row[i] for i in indexes])


def verify_csv(source, target, expected_rows):
    """Independently reread both files and compare every retained cell by name."""
    nulls = Counter()
    keys_seen = set()
    rows = voyages = 0
    hours = Decimal(0)
    example = None
    with source.open(newline="", encoding="utf-8") as src, target.open(newline="", encoding="utf-8") as dst:
        original, final = csv.DictReader(src), csv.DictReader(dst)
        fields = final.fieldnames
        require(fields == [f for f in original.fieldnames if f not in REMOVED], "Column order changed")
        require(len(fields) == len(set(fields)), "Duplicate output columns")
        require(not (set(fields) & set(REMOVED)), "Sampling field remains")
        keys = ["origin_country", "dest_country", "shiptype_group", "departure_month"]
        if "origin_city_code" in fields:
            keys += ["origin_city_code", "dest_city_code"]
        for before, after in zip_longest(original, final):
            require(before is not None and after is not None, "Row count changed")
            require(None not in before and None not in after, "Malformed CSV row")
            require(all(after[field] == before[field] for field in fields),
                    "Retained value or row order changed at record " + str(rows + 1))
            key = tuple(after[field] for field in keys)
            require(key not in keys_seen, "Duplicate record key")
            keys_seen.add(key)
            rows += 1
            voyages += int(after["voyages"])
            hours += Decimal(after["voyage_hours"])
            nulls.update(field for field in fields if after[field] == "")
            if "origin_city_code" not in fields and all(after[k] == v for k, v in EXAMPLE.items()):
                require(example is None, "Duplicate manuscript example")
                example = after
    require(rows == expected_rows, "Unexpected record count")
    require(voyages == 1111800, "Analysed voyage count changed")
    return {"rows": rows, "columns": len(fields), "fields": fields,
            "voyages": voyages, "released_voyage_hours_sum": str(hours),
            "unique_keys": len(keys_seen), "blank_cells_by_field": dict(nulls),
            "every_retained_cell_and_row_order": "identical", **file_record(target)}, example


def clean_guide(text):
    text = replace_once(text, "country table has 50 fields", "country table has 41 fields")
    text = replace_once(text, "giving 52 fields", "giving 43 fields")
    text = replace_once(text, "## Exposure, coverage and intervals", "## Exposure, coverage and ranges")
    paragraphs = text.split("\n\n")
    removed = [p for p in paragraphs if p.startswith("The 95% sampling intervals describe")]
    require(len(removed) == 1, "Unexpected sampling overview")
    text = "\n\n".join(p for p in paragraphs if p not in removed)
    text = replace_once(text, " Sampling intervals also require resampling the combined vessel sample; averaging interval endpoints does not reproduce them.", "")
    lines = []
    dropped = []
    for line in text.splitlines():
        match = re.match(r"^\| `([^`]+)` \|", line)
        if match and match[1] in REMOVED:
            dropped.append(match[1])
        else:
            lines.append(line)
    require(set(dropped) == set(REMOVED), "Field guide removal set differs")
    return "\n".join(lines) + "\n"


def clean_readme(text):
    replacements = (
        ("122,736 | 50 |", "122,736 | 41 |"),
        ("566,294 | 52 |", "566,294 | 43 |"),
        ("uncertainty measures. Records with fewer than three vessels have no sampling intervals.", "ranges for missing positions."),
        ("Vessel counts and sampling intervals cannot be added across records.", "Vessel counts cannot be added across records."),
        ("for all 52 variables", "for all 43 variables"),
    )
    for old, new in replacements:
        text = replace_once(text, old, new)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-public", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    source, run, review = (p.resolve() for p in (args.source_public, args.run, args.review))
    require(source.is_dir(), "Source public directory does not exist")
    require(not run.exists() and not review.exists(), "Output paths already exist; preserve prior artifacts")
    require(not run.is_relative_to(source) and not review.is_relative_to(source), "Output must not be inside source")
    require(not run.is_relative_to(review) and not review.is_relative_to(run), "Run and review paths must be separate")
    manifest_path = source / "04_MANIFEST.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(set(source_manifest) == PUBLIC_FILES, "Unexpected input manifest members")
    require({p.name for p in source.iterdir()} == PUBLIC_FILES | {"04_MANIFEST.json"}, "Unexpected source public files")
    source_records = {name: file_record(source / name) for name in sorted(PUBLIC_FILES | {"04_MANIFEST.json"})}
    for name, record in source_manifest.items():
        require(source_records[name] == record, "Source manifest mismatch: " + name)

    run.mkdir(parents=True, exist_ok=False)
    public = run / "public"
    public.mkdir()
    report = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_public": str(source), "public": str(public), "removed_fields": list(REMOVED),
              "source_files": source_records, "script_sha256": digest(Path(__file__)), "tables": {}}
    script_dir = Path(__file__).resolve().parent
    report["checkout_commit"] = subprocess.check_output(
        ["git", "-C", str(script_dir), "rev-parse", "HEAD"], text=True).strip()
    example = None
    for filename, rows, columns in TABLES:
        project_csv(source / filename, public / filename, columns)
        table, selected = verify_csv(source / filename, public / filename, rows)
        require(table["columns"] == columns - len(REMOVED), "Output column count differs")
        report["tables"][filename] = table
        if selected is not None:
            example = selected
        print(json.dumps({"file": filename, "rows": rows, "fields": table["columns"],
                          "retained_cells": "identical"}), flush=True)
    require(example is not None, "Manuscript example missing")

    guide = clean_guide((source / "03_FIELD_GUIDE.md").read_text(encoding="utf-8"))
    readme = clean_readme((source / "00_README.md").read_text(encoding="utf-8"))
    for text in (guide, readme):
        require(not re.search(r"bootstrap|resampl|sampling interval|ci95", text, re.I), "Sampling documentation remains")
    definitions = re.findall(r"^\| `([^`]+)` \|", guide, re.M)
    city_fields = report["tables"][TABLES[1][0]]["fields"]
    require(len(definitions) == len(set(definitions)) == 43 and set(definitions) == set(city_fields),
            "Field guide does not exactly cover the final city schema")
    require(set(city_fields) - set(report["tables"][TABLES[0][0]]["fields"]) == {"origin_city_code", "dest_city_code"},
            "Country and city schemas differ unexpectedly")
    (public / "03_FIELD_GUIDE.md").write_text(guide, encoding="utf-8")
    (public / "00_README.md").write_text(readme, encoding="utf-8")
    write_json(public / "04_MANIFEST.json", {name: file_record(public / name) for name in sorted(PUBLIC_FILES)})
    require({p.name for p in public.iterdir()} == PUBLIC_FILES | {"04_MANIFEST.json"}, "Unexpected public files")
    require(all(file_record(source / name) == record for name, record in source_records.items()), "Source changed during packaging")
    report.update(status="pass", source_files_unchanged=True, field_guide_exact_schema=True)
    write_json(run / "validation.json", report)

    review.mkdir(parents=True, exist_ok=False)
    for name in ("00_README.md", "03_FIELD_GUIDE.md", "04_MANIFEST.json"):
        shutil.copy2(public / name, review / ("PUBLIC_" + name))
    shutil.copy2(run / "validation.json", review / "01_VALIDATION.json")
    with (review / "03_MANUSCRIPT_EXAMPLE.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(example), lineterminator="\n")
        writer.writeheader()
        writer.writerow(example)
    (review / "00_README.md").write_text(
        "# E008 — 删除抽样字段后的月度发布候选\n\n"
        "2026-10-17 决定将稿件删除 bootstrap 的处理同步到实际候选数据。"
        "本次从已验证的公开候选 CSV 删除 8 个抽样区间端点和 1 个区间不可用标记，"
        "同步字段指南与说明，并生成新的文件哈希。保留 single_ship、主值、覆盖率和缺失位置范围。\n\n"
        "| 表 | 记录 | 字段 | 分析航次 |\n|---|---:|---:|---:|\n"
        "| 国家对月度 | 122,736 | 41 | 1,111,800 |\n"
        "| 城市编码对月度 | 566,294 | 43 | 1,111,800 |\n\n"
        f"服务器新候选：`{public}`。\n\n"
        f"原始候选：`{source}`，全部文件的哈希在运行前后保持一致。\n\n"
        "核验逐行、逐字段比较所有保留单元格的字符串值，确认行序、数值精度、空值和标记均不变；"
        "检查唯一键、各表航次总数、字段指南与实际 43 字段逐一对应。结果见 `01_VALIDATION.json`。"
        "未重跑轨迹、天气、聚合或缺失位置范围计算；原有科学验证继续见"
        "[原 E008 核验](../2026-10-16_2130_E008-monthly-public-release/00_README.md)。\n\n"
        "`PUBLIC_` 文件为新候选的公开说明、字段指南和清单副本；"
        "`03_MANUSCRIPT_EXAMPLE.csv` 是新国家表中的新加坡→中国集装箱船 2025-09 记录。"
        "候选数据尚未公开上传；归档版本和许可仍待发布时落实。\n",
        encoding="utf-8")
    print(json.dumps({"status": "pass", "public": str(public), "review": str(review)}), flush=True)


if __name__ == "__main__":
    main()
