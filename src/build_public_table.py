"""Export a GFW monthly aggregate to the current 41/43-field public schema.

Input is a full-precision monthly aggregate, not individual voyages or rounded
public data. Reuses the accepted shared validation, rounding and field projection.
Country aggregates must already apply the adopted HKG/TWN/MAC-to-CHN regrouping.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MONTHLY = ROOT / "src/reference/02_coding/pipeline/gfw_monthly_release"
HISTORICAL = ROOT / "src/reference/02_coding/experiments/E008_gfw_monthly_public_release"


def load_methods():
    sys.path.insert(0, str(MONTHLY))
    import monthly_release
    path = HISTORICAL / "40_finalize_public_release/10_remove_sampling_fields.py"
    spec = importlib.util.spec_from_file_location("public_schema_projection", path)
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)
    return monthly_release, projection


def export_monthly(source, output, level):
    source, output = Path(source), Path(output)
    if level not in {"country", "city"}:
        raise ValueError("level must be country or city")
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output path.")
    if output.exists():
        raise FileExistsError(output)
    methods, projection = load_methods()
    import numpy as np
    import pandas as pd

    if source.suffix.lower() == ".parquet":
        table = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        table = pd.read_csv(source)
    else:
        raise ValueError("Input must be a full-precision .parquet or .csv aggregate.")
    keys = methods.COUNTRY_KEY if level == "country" else methods.CODED_KEY
    required = set(keys + methods.MEASURES) - set(projection.REMOVED)
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for side in ("origin", "dest"):
        if table[side + "_country"].isin({"HKG", "TWN", "MAC"}).any():
            raise ValueError("Regroup HKG/TWN/MAC into CHN from voyage statistics before export; do not relabel rounded aggregates.")
    # The historical exporter validates 50/52 fields. Supply placeholders only
    # for fields discarded by the accepted final projection; no resampling.
    for field in methods.CI:
        table[field] = np.nan
    table["sampling_interval_unavailable_fewer_than_3_ships"] = table.ships.lt(3)
    with tempfile.TemporaryDirectory(prefix="gfw-monthly-export-") as temporary:
        intermediate = Path(temporary) / "intermediate.csv"
        final = Path(temporary) / "public.csv"
        report = methods.export_table(table, keys, intermediate)
        projection.project_csv(intermediate, final, expected_columns=len(keys + methods.MEASURES))
        report.update(file=output.name, columns=len(keys + methods.MEASURES) - len(projection.REMOVED),
                      bytes=final.stat().st_size, sha256=projection.digest(final))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as target, final.open("rb") as src:
            shutil.copyfileobj(src, target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", choices=("country", "city"), required=True)
    args = parser.parse_args()
    report = export_monthly(args.input, args.output, args.level)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
