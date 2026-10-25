"""Generic AIS identity, bucketing, filtering, and deduplication helpers.

These helpers are shared by weather matching and weather coverage diagnostics.
They preserve the accepted MMSI/API identity join and deterministic bucket
rules without depending on route measurement code.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

VESSEL_BUCKET_HASH_VERSION = "sha256-mmsi-utf8-first8-big-mod-v1"
AIS_IDENTITY_JOIN_POLICY = (
    "accepted_physical_leg_exact_mmsi_api_pair_keep_else_drop_never_relabel_v1"
)
AIS_SOURCE_COLUMNS = ("vessel_uid", "mmsi", "UTC", "lat", "lon")
AIS_BUCKET_COLUMNS = (
    "vessel_uid",
    "mmsi",
    "UTC",
    "lat",
    "lon",
    "source_year",
    "source_shard",
    "source_row_ordinal",
    "vessel_bucket",
)
def _normalise_identifier(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return canonical decimal identifiers and a validity mask.

    String identifiers remain strings, including genuine leading zeroes.  The
    ``.0`` suffix is removed only for legacy numeric-looking values.
    """

    value = series.astype("string").str.strip()
    numeric_float = value.str.fullmatch(r"[0-9]+[.]0+", na=False)
    value = value.mask(numeric_float, value.str.replace(r"[.]0+$", "", regex=True))
    valid = value.notna() & value.str.fullmatch(r"[0-9]+", na=False)
    return value, valid


def normalise_required_identifier(series: pd.Series, label: str) -> pd.Series:
    """Canonicalise an identifier column and reject missing/non-decimal keys."""

    value, valid = _normalise_identifier(series)
    if not valid.all():
        raise ValueError(f"{label} contains {int((~valid).sum())} invalid identifiers")
    return value


def stable_vessel_bucket(mmsi: str, bucket_count: int) -> int:
    """Map one canonical MMSI to a stable bucket using versioned SHA-256."""

    if int(bucket_count) <= 0:
        raise ValueError("bucket_count must be positive")
    canonical, valid = _normalise_identifier(pd.Series([mmsi], dtype="string"))
    if not bool(valid.iloc[0]):
        raise ValueError(f"invalid MMSI for vessel bucket: {mmsi!r}")
    digest = hashlib.sha256(str(canonical.iloc[0]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % int(bucket_count)


def stable_vessel_buckets(mmsi: pd.Series, bucket_count: int) -> pd.Series:
    """Vector wrapper retaining the frozen scalar hash definition."""

    canonical = normalise_required_identifier(mmsi, "mmsi")
    unique = canonical.drop_duplicates()
    lookup = {
        value: stable_vessel_bucket(str(value), bucket_count)
        for value in unique.astype(str)
    }
    return canonical.astype(str).map(lookup).astype(np.int16)


def prepare_ais_batch(
    frame: pd.DataFrame,
    *,
    source_year: int,
    source_shard: str,
    source_row_offset: int,
    bucket_count: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select valid AIS observations from one streaming source record batch."""

    if missing := sorted(set(AIS_SOURCE_COLUMNS) - set(frame.columns)):
        raise ValueError(f"cleaned AIS shard missing columns: {missing}")
    work = frame.loc[:, AIS_SOURCE_COLUMNS].copy()
    vessel_uid, valid_api = _normalise_identifier(work["vessel_uid"])
    mmsi, valid_mmsi = _normalise_identifier(work["mmsi"])
    utc = pd.to_datetime(work["UTC"], utc=True, errors="coerce")
    lat = pd.to_numeric(work["lat"], errors="coerce")
    lon = pd.to_numeric(work["lon"], errors="coerce")
    valid_time = utc.notna()
    wrong_year = valid_time & utc.dt.year.ne(int(source_year))
    if wrong_year.any():
        raise ValueError(
            f"cleaned AIS shard for {source_year} contains {int(wrong_year.sum())} wrong-year timestamps"
        )
    valid_coordinate = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & lat.between(-90.0, 90.0)
        & lon.between(-180.0, 180.0)
    )
    valid = valid_api & valid_mmsi & valid_time & valid_coordinate
    ordinal = np.arange(
        int(source_row_offset), int(source_row_offset) + len(work), dtype=np.int64
    )
    selected_index = work.index[valid]
    output = pd.DataFrame(
        {
            "vessel_uid": vessel_uid.loc[selected_index].astype("string").to_numpy(),
            "mmsi": mmsi.loc[selected_index].astype("string").to_numpy(),
            "UTC": utc.loc[selected_index].to_numpy(),
            "lat": lat.loc[selected_index].astype(np.float64).to_numpy(),
            "lon": lon.loc[selected_index].astype(np.float64).to_numpy(),
            "source_year": np.int16(source_year),
            "source_shard": str(source_shard),
            "source_row_ordinal": ordinal[valid.to_numpy()],
        }
    ).reset_index(drop=True)
    if not output.empty:
        output["vessel_bucket"] = stable_vessel_buckets(
            output["mmsi"], bucket_count
        ).to_numpy()
    else:
        output["vessel_bucket"] = pd.Series(dtype=np.int16)
    output = output.loc[:, AIS_BUCKET_COLUMNS]
    audit = {
        "input_rows": int(len(work)),
        "valid_rows": int(valid.sum()),
        "rejected_identifier_rows": int((~(valid_api & valid_mmsi)).sum()),
        "rejected_timestamp_rows": int((~valid_time).sum()),
        "rejected_coordinate_rows": int((~valid_coordinate).sum()),
    }
    return output, audit


def deduplicate_vessel_ais(points: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Deterministically deduplicate same-vessel UTC observations."""

    required = set(AIS_BUCKET_COLUMNS)
    if missing := sorted(required - set(points.columns)):
        raise ValueError(f"bucket AIS missing columns: {missing}")
    work = points.loc[:, AIS_BUCKET_COLUMNS].copy()
    work["vessel_uid"] = normalise_required_identifier(
        work["vessel_uid"], "AIS vessel_uid"
    )
    work["mmsi"] = normalise_required_identifier(work["mmsi"], "AIS mmsi")
    work["UTC"] = pd.to_datetime(work["UTC"], utc=True, errors="coerce")
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    if work[["UTC", "lat", "lon"]].isna().any().any():
        raise ValueError("bucket AIS contains invalid required values")
    if (~work["lat"].between(-90.0, 90.0) | ~work["lon"].between(-180.0, 180.0)).any():
        raise ValueError("bucket AIS contains invalid coordinates")
    work = work.sort_values(
        ["mmsi", "UTC", "source_year", "source_shard", "source_row_ordinal"],
        kind="stable",
    ).reset_index(drop=True)
    duplicate = work.duplicated(["mmsi", "UTC"], keep=False)
    if duplicate.any():
        disagreement = (
            work.loc[duplicate]
            .groupby(["mmsi", "UTC"], sort=False, observed=True)[
                ["vessel_uid", "lat", "lon"]
            ]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if disagreement.any():
            raise ValueError("same-vessel duplicate UTC AIS observations disagree")
    before = len(work)
    work = work.drop_duplicates(["mmsi", "UTC"], keep="first").reset_index(drop=True)
    return work, int(before - len(work))


def validate_ais_physical_identity(
    points: pd.DataFrame, physical_legs: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the authoritative physical-leg identity join without relabelling.

    The accepted physical-leg ``(mmsi, vessel_uid_clean)`` pair is the
    authority for this join. AIS observations from an irrelevant MMSI are
    outside the bucket's physical-leg scope. Observations from a relevant MMSI
    whose source API identity disagrees are removed from analysis inputs and
    retained as exact source-locator audit evidence. A physical-leg MMSI
    mapping to multiple canonical API identities remains a hard failure.
    """

    required_points = {
        "mmsi",
        "vessel_uid",
        "source_year",
        "source_shard",
        "source_row_ordinal",
    }
    required_legs = {"mmsi", "vessel_uid_clean"}
    if missing := sorted(required_points - set(points.columns)):
        raise ValueError(f"AIS identity columns missing: {missing}")
    if missing := sorted(required_legs - set(physical_legs.columns)):
        raise ValueError(f"physical-leg identity columns missing: {missing}")
    canonical_points = points.copy()
    canonical_points["mmsi"] = normalise_required_identifier(
        canonical_points["mmsi"], "AIS identity-join mmsi"
    )
    canonical_points["vessel_uid"] = normalise_required_identifier(
        canonical_points["vessel_uid"], "AIS identity-join vessel_uid"
    )
    expected = physical_legs[["mmsi", "vessel_uid_clean"]].copy()
    expected["mmsi"] = normalise_required_identifier(
        expected["mmsi"], "physical-leg identity-join mmsi"
    )
    expected["vessel_uid_clean"] = normalise_required_identifier(
        expected["vessel_uid_clean"], "physical-leg identity-join vessel_uid_clean"
    )
    expected = expected.drop_duplicates()
    if expected["mmsi"].duplicated().any():
        raise ValueError("one physical-leg MMSI maps to multiple API ship identities")
    api_by_mmsi = expected.set_index("mmsi")["vessel_uid_clean"].astype(str)
    relevant = canonical_points.loc[
        canonical_points["mmsi"].isin(api_by_mmsi.index)
    ].copy()
    expected_api = relevant["mmsi"].map(api_by_mmsi).astype("string")
    mismatch = ~relevant["vessel_uid"].astype("string").eq(expected_api)
    mismatch_rows = relevant.loc[
        mismatch,
        ["mmsi", "vessel_uid", "source_year", "source_shard", "source_row_ordinal"],
    ].copy()
    mismatch_rows["expected_vessel_uid_clean"] = expected_api.loc[mismatch].to_numpy()
    if not mismatch_rows.empty:
        for column in ("source_year", "source_row_ordinal"):
            values = mismatch_rows[column]
            if (
                pd.api.types.is_bool_dtype(values.dtype)
                or not pd.api.types.is_integer_dtype(values.dtype)
                or values.isna().any()
            ):
                raise ValueError(
                    f"AIS identity mismatch {column} must contain exact integers"
                )
            if (
                (values < np.iinfo(np.int64).min).any()
                or (values > np.iinfo(np.int64).max).any()
                or (column == "source_year" and (values <= 0).any())
                or (column == "source_row_ordinal" and (values < 0).any())
            ):
                raise ValueError(
                    f"AIS identity mismatch {column} is outside its valid range"
                )
        mismatch_rows["source_year"] = mismatch_rows["source_year"].astype(
            np.int64
        )
        mismatch_rows["source_shard"] = mismatch_rows["source_shard"].astype("string")
        mismatch_rows["source_row_ordinal"] = mismatch_rows[
            "source_row_ordinal"
        ].astype(np.int64)
        if mismatch_rows["source_shard"].isna().any() or mismatch_rows[
            "source_shard"
        ].eq("").any():
            raise ValueError("AIS identity mismatch contains a missing source shard")
        source_key = ["source_year", "source_shard", "source_row_ordinal"]
        if mismatch_rows.duplicated(source_key).any():
            raise ValueError("AIS identity mismatch source locator is not unique")

    breakdown: list[dict[str, Any]] = []
    group_columns = [
        "mmsi",
        "expected_vessel_uid_clean",
        "vessel_uid",
        "source_year",
        "source_shard",
    ]
    for keys, group in mismatch_rows.groupby(
        group_columns, sort=True, observed=True, dropna=False
    ):
        mmsi, expected_api_value, observed_api, source_year, source_shard = keys
        ordinals = sorted(int(value) for value in group["source_row_ordinal"])
        breakdown.append(
            {
                "mmsi": str(mmsi),
                "expected_vessel_uid_clean": str(expected_api_value),
                "observed_vessel_uid": str(observed_api),
                "source_year": int(source_year),
                "source_shard": str(source_shard),
                "rows": len(ordinals),
                "source_row_ordinals": ordinals,
            }
        )

    matched = relevant.loc[~mismatch].copy().reset_index(drop=True)
    audit: dict[str, Any] = {
        "policy": AIS_IDENTITY_JOIN_POLICY,
        "bucket_input_rows": int(len(canonical_points)),
        "relevant_mmsi_rows_before_identity_filter": int(len(relevant)),
        "relevant_physical_vessel_rows": int(len(matched)),
        "irrelevant_vessel_rows_skipped": int(len(canonical_points) - len(relevant)),
        "api_identity_mismatch_rows": int(mismatch.sum()),
        "api_identity_mismatch_group_count": len(breakdown),
        "api_identity_mismatch_source_breakdown": breakdown,
    }
    if (
        audit["bucket_input_rows"]
        != audit["irrelevant_vessel_rows_skipped"]
        + audit["relevant_physical_vessel_rows"]
        + audit["api_identity_mismatch_rows"]
    ):
        raise AssertionError("AIS identity-join audit row closure failed")
    if sum(item["rows"] for item in breakdown) != audit["api_identity_mismatch_rows"]:
        raise AssertionError("AIS identity-join source breakdown does not close")
    return matched, audit
