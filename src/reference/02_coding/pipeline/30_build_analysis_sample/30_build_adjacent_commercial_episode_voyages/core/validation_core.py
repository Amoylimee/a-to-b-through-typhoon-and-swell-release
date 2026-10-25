"""Shared, non-binding comparison helpers for C5."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


REQUIRED_SHADOW_LABEL = "shadow_diagnostic_only"


def compare_shadow_parameters(
    baseline: pd.DataFrame,
    shadow: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    value_column: str = "parameter_value",
) -> pd.DataFrame:
    """Compare matched 2021 and 2025 parameter estimates without promoting 2025.

    Both frames must use identical keys and units. Missing groups are retained
    as explicit left/right-only rows so coverage changes cannot disappear.
    """

    keys = list(key_columns)
    required = set(keys) | {value_column}
    for name, frame in (("baseline", baseline), ("shadow", shadow)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} frame is missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} frame has duplicate parameter keys")

    left = baseline[keys + [value_column]].rename(
        columns={value_column: "value_2021_frozen"}
    )
    right = shadow[keys + [value_column]].rename(
        columns={value_column: "value_2025_shadow"}
    )
    output = left.merge(right, on=keys, how="outer", indicator=True, validate="1:1")
    output["coverage_status"] = output.pop("_merge").map(
        {
            "both": "matched",
            "left_only": "2021_only",
            "right_only": "2025_only",
        }
    )
    output["absolute_change"] = (
        output["value_2025_shadow"] - output["value_2021_frozen"]
    )
    baseline_values = output["value_2021_frozen"].to_numpy(dtype=float)
    output["percentage_change"] = np.divide(
        output["absolute_change"].to_numpy(dtype=float) * 100.0,
        baseline_values,
        out=np.full(len(output), np.nan, dtype=float),
        where=np.isfinite(baseline_values) & (baseline_values != 0),
    )
    output["value_2021_role"] = "frozen_operational"
    output["value_2025_role"] = REQUIRED_SHADOW_LABEL
    output["eligible_for_rule_replacement"] = False
    return output.sort_values(keys, kind="stable").reset_index(drop=True)


def assign_suez_dependence_bin(values: pd.Series) -> pd.Categorical:
    """Assign frozen, descriptive Suez-dependence bins."""

    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 0.0, 0.25, 0.5, 0.75, np.inf],
        labels=["zero", "low", "medium", "high", "very_high"],
        include_lowest=True,
        right=True,
    )
