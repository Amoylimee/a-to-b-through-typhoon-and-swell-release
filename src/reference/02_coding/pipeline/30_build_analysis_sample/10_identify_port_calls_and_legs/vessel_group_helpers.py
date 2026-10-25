"""Frozen vessel grouping helpers used by port-call and leg construction.

Copied from the frozen legacy implementation so this repository is
self-contained. The definitions are intentionally unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().upper().split())


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def map_shiptype_group(detailed_type: Any) -> str:
    text = normalize_text(detailed_type)

    if "CONTAINER" in text:
        return "container"
    if "BULK" in text or text in {"ORE CARRIER", "OBO CARRIER"}:
        return "bulk_carrier"
    if "OIL PRODUCTS" in text or "CHEMICAL" in text or "OIL/CHEMICAL" in text:
        return "product_chemical_tanker"
    if "CRUDE OIL" in text:
        return "crude_oil_tanker"
    if "LNG" in text or "LPG" in text:
        return "gas_tanker"
    if "GENERAL CARGO" in text:
        return "general_cargo"
    if "RO-RO" in text or "VEHICLE" in text:
        return "ro_ro_vehicle"
    if "REEFER" in text:
        return "reefer"

    return "other_retained"


def build_size_bin(df: pd.DataFrame) -> pd.Series:
    dwt = numeric_column(df, "deadweight_tonnage")
    teu = numeric_column(df, "teu_capacity")
    gt = numeric_column(df, "gross_tonnage")

    size_value = dwt.copy()
    size_source = pd.Series("dwt", index=df.index, dtype="object")

    use_teu = size_value.isna() & teu.notna()
    size_value.loc[use_teu] = teu.loc[use_teu]
    size_source.loc[use_teu] = "teu"

    use_gt = size_value.isna() & gt.notna()
    size_value.loc[use_gt] = gt.loc[use_gt]
    size_source.loc[use_gt] = "gt"

    bins = pd.Series("unknown", index=df.index, dtype="object")

    dwt_mask = size_source.eq("dwt") & size_value.notna()
    bins.loc[dwt_mask & size_value.lt(20_000)] = "dwt_lt20k"
    bins.loc[dwt_mask & size_value.between(20_000, 79_999, inclusive="both")] = "dwt_20k_80k"
    bins.loc[dwt_mask & size_value.between(80_000, 149_999, inclusive="both")] = "dwt_80k_150k"
    bins.loc[dwt_mask & size_value.ge(150_000)] = "dwt_ge150k"

    teu_mask = size_source.eq("teu") & size_value.notna()
    bins.loc[teu_mask & size_value.lt(3_000)] = "teu_lt3k"
    bins.loc[teu_mask & size_value.between(3_000, 7_999, inclusive="both")] = "teu_3k_8k"
    bins.loc[teu_mask & size_value.between(8_000, 14_999, inclusive="both")] = "teu_8k_15k"
    bins.loc[teu_mask & size_value.ge(15_000)] = "teu_ge15k"

    gt_mask = size_source.eq("gt") & size_value.notna()
    bins.loc[gt_mask & size_value.lt(20_000)] = "gt_lt20k"
    bins.loc[gt_mask & size_value.between(20_000, 79_999, inclusive="both")] = "gt_20k_80k"
    bins.loc[gt_mask & size_value.ge(80_000)] = "gt_ge80k"

    return bins
