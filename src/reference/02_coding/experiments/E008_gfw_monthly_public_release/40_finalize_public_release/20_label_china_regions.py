#!/usr/bin/env python3
"""Relabel location wording in the public documentation; copy the tables byte for byte.

The user asked on 2026-10-18 that the release name Hong Kong and Taiwan as parts of China.
The tables carry codes only, so only 00_README.md and 03_FIELD_GUIDE.md change: "country or
territory" becomes "country or region", and both documents state that HKG and TWN are labelled
Hong Kong, China and Taiwan, China. Macao (MAC) has no analysed voyages, so it is not named. The CSVs are copied unchanged and
their SHA-256 values are checked against the source manifest.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TABLES = ("01_release_country_pair_month.csv", "02_release_city_pair_month.csv")
PUBLIC_FILES = {"00_README.md", "03_FIELD_GUIDE.md", *TABLES}
CHINA_REGIONS = ("HKG", "TWN")
LABEL_SENTENCE = ("Hong Kong and Taiwan have separate codes (`HKG` and `TWN`) and are labelled "
                  "Hong Kong, China and Taiwan, China.")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def file_record(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replace_once(text, old, new):
    require(text.count(old) == 1, "Unexpected source documentation: " + old)
    return text.replace(old, new)


def relabel_readme(text):
    return replace_once(
        text,
        "The country table identifies the departure and arrival countries or territories.",
        "The country table identifies the departure and arrival countries or regions. " + LABEL_SENTENCE)


def relabel_guide(text):
    text = replace_once(text, "| `origin_country` | Code | Country or territory of departure. |",
                        "| `origin_country` | Code | Country or region of departure. |")
    text = replace_once(text, "| `dest_country` | Code | Country or territory of arrival. |",
                        "| `dest_country` | Code | Country or region of arrival. |")
    anchor = "The public files contain no city-name lookup table, coordinates, vessel identifiers or exact departure and arrival times.\n"
    return replace_once(text, anchor, anchor + "\nLocations are identified by country or region. " + LABEL_SENTENCE + "\n")


def region_counts(path):
    """Rows in which each China region code appears as origin or destination."""
    counts = Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for code in CHINA_REGIONS:
                if code in (row["origin_country"], row["dest_country"]):
                    counts[code] += 1
    return {code: counts[code] for code in CHINA_REGIONS}


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
    source_manifest = json.loads((source / "04_MANIFEST.json").read_text(encoding="utf-8"))
    require(set(source_manifest) == PUBLIC_FILES, "Unexpected input manifest members")
    require({p.name for p in source.iterdir()} == PUBLIC_FILES | {"04_MANIFEST.json"}, "Unexpected source public files")
    source_records = {name: file_record(source / name) for name in sorted(PUBLIC_FILES | {"04_MANIFEST.json"})}
    for name, record in source_manifest.items():
        require(source_records[name] == record, "Source manifest mismatch: " + name)

    run.mkdir(parents=True, exist_ok=False)
    public = run / "public"
    public.mkdir()
    report = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_public": str(source), "public": str(public), "source_files": source_records,
              "script_sha256": digest(Path(__file__)), "tables": {}}
    report["checkout_commit"] = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], text=True).strip()
    for name in TABLES:
        shutil.copyfile(source / name, public / name)
        require(file_record(public / name) == source_records[name], "Table changed while copying: " + name)
        report["tables"][name] = {**file_record(public / name), "byte_identical_to_source": True,
                                  "rows_with_region_code": region_counts(public / name)}
    require(all(report["tables"][TABLES[0]]["rows_with_region_code"][c] > 0 for c in CHINA_REGIONS),
            "A China region code is absent from the country table")

    readme = relabel_readme((source / "00_README.md").read_text(encoding="utf-8"))
    guide = relabel_guide((source / "03_FIELD_GUIDE.md").read_text(encoding="utf-8"))
    for text in (readme, guide):
        require(not re.search(r"territor", text, re.I), "Territory wording remains")
        require(text.count("Taiwan, China") == 1 and text.count("Hong Kong, China") == 1, "Region labels missing")
    old_fields = re.findall(r"^\| `([^`]+)` \|", (source / "03_FIELD_GUIDE.md").read_text(encoding="utf-8"), re.M)
    require(re.findall(r"^\| `([^`]+)` \|", guide, re.M) == old_fields and len(old_fields) == 43, "Field rows changed")
    (public / "00_README.md").write_text(readme, encoding="utf-8")
    (public / "03_FIELD_GUIDE.md").write_text(guide, encoding="utf-8")
    write_json(public / "04_MANIFEST.json", {name: file_record(public / name) for name in sorted(PUBLIC_FILES)})
    require({p.name for p in public.iterdir()} == PUBLIC_FILES | {"04_MANIFEST.json"}, "Unexpected public files")
    require(all(file_record(source / name) == record for name, record in source_records.items()), "Source changed during packaging")
    report.update(status="pass", source_files_unchanged=True, changed_files=["00_README.md", "03_FIELD_GUIDE.md", "04_MANIFEST.json"])
    write_json(run / "validation.json", report)

    review.mkdir(parents=True, exist_ok=False)
    for name in ("00_README.md", "03_FIELD_GUIDE.md", "04_MANIFEST.json"):
        shutil.copy2(public / name, review / ("PUBLIC_" + name))
    shutil.copy2(run / "validation.json", review / "01_VALIDATION.json")
    counts = report["tables"][TABLES[0]]["rows_with_region_code"]
    (review / "00_README.md").write_text(
        "# E008 — 香港、台湾标注为中国的月度发布候选\n\n"
        "2026-10-18 决定发布文件明确香港、台湾属于中国。发布表只含代码，没有国名，"
        "所以本次只改说明文字：README 与字段指南中的 “country or territory” 改为 “country or region”，"
        "并写明 HKG、TWN 分别标注为 Hong Kong, China 和 Taiwan, China。澳门（MAC）没有分析航次，不出现在表中。\n\n"
        "首次运行 `2026-10-18_0050_…` 因国家表没有 MAC 记录在核验步骤停止，未写出说明或清单，目录保留。"
        "两张 CSV 逐字节复制，SHA-256 与上一候选一致，数据没有任何改动；字段指南的 43 个字段行不变。"
        f"国家对表中含 HKG/TWN 的记录数为 {counts['HKG']:,}/{counts['TWN']:,}。"
        "结果见 `01_VALIDATION.json`。\n\n"
        f"服务器新候选：`{public}`。\n\n"
        f"上一候选：`{source}`（[本地说明](../2026-10-17_2208_E008-public-release-without-sampling/00_README.md)），"
        "全部文件哈希在运行前后保持一致。\n\n"
        "`PUBLIC_` 文件为新候选的公开说明、字段指南和清单副本。候选数据尚未公开上传；归档版本和许可仍待发布时落实。\n",
        encoding="utf-8")
    print(json.dumps({"status": "pass", "public": str(public), "review": str(review), "region_rows": counts}), flush=True)


if __name__ == "__main__":
    main()
