"""The one AIS shard schema every provider is ingested into.

Prepared input feeds can differ in column names, types and navigational-status
encoding. Cleaning and later stages read the common schema below. The input
adapter supplies source provenance explicitly; this schema does not infer a
provider from the observation year.

The required set is deliberately small. `cleaning_common.py` needs only
`vessel_uid, UTC, lon, lat, speed, course`, treats `heading, mmsi, imo` as optional,
and the port-call step downstream consumes columns the cleaning step derives rather
than raw ones. The twenty-odd further columns carried by the legacy shards
(`AIS_new`, `turn`, `draught`, `DateHour`, `destination`, `source_shard_file`, and so
on) are read by nothing, and several cannot be filled for all five years, so carrying
them would manufacture a difference between providers where none is needed.

## Downsampling

One row per (`vessel_uid`, `UTC_10min`) bin, holding the earliest raw position in
that bin, with its original `UTC` preserved rather than snapped to the grid. Empty
bins produce no row. This rule was recovered from the 2021 and 2025 shards rather
than chosen; see the accepted report in
`outputs/2026-08-07_2350_E009_downsampling-rule/`.

## Why timestamps are naive

All positions are UTC, and an earlier version of this schema said so explicitly by
storing a timezone. That broke the accepted cleaning code in a way worth recording,
because the failure was silent about its real cause.

`run_outlier_cleaner` sends each trajectory through movingpandas, which strips the
timezone on the way out — but trajectories shorter than two points take a shortcut and
are appended untouched, keeping theirs. The `UTC` column then holds a mix of aware and
naive timestamps. `pd.concat` accepts it; `sort_values` raises `Cannot compare tz-naive
and tz-aware timestamps`, which pandas re-wraps as a categorical-ordering error that
names neither timezones nor the column involved.

The legacy 2021 and 2025 feeds are naive `datetime64[ns]`, so that branch never fired
before. Matching them is also what this experiment is for: the years should differ in
their raw feed, never in a type the downstream code handles differently.

## Navigational status

2025 and reference AIS carry ITU numeric codes. The 2021 archive carries text labels that
collapse everything outside the five common states into a single `Others` bucket, so
the numeric code cannot be recovered for those rows. `nav_status` is therefore defined
at the granularity all three feeds can express, and `nav_status_code` keeps the
original number wherever one exists.

Nothing in port-call detection reads either field — that logic uses position, speed
thresholds of 0.2 and 0.5 knots, and a 4-hour AIS gap. Status is carried for
diagnostics, which is why the 2021 collapse is tolerable rather than blocking.
"""

from __future__ import annotations

import pyarrow as pa

# --- Navigational status --------------------------------------------------

UNDER_WAY_ENGINE = "under_way_engine"
AT_ANCHOR = "at_anchor"
MOORED = "moored"
UNDER_WAY_SAILING = "under_way_sailing"
FISHING = "fishing"
OTHER = "other"
UNKNOWN = "unknown"

NAV_STATUS_VALUES = (
    UNDER_WAY_ENGINE,
    AT_ANCHOR,
    MOORED,
    UNDER_WAY_SAILING,
    FISHING,
    OTHER,
    UNKNOWN,
)

# ITU codes shared by the 2025 and reference AIS feeds
NAV_STATUS_FROM_CODE = {
    0: UNDER_WAY_ENGINE,
    1: AT_ANCHOR,
    2: OTHER,   # not under command
    3: OTHER,   # restricted manoeuvrability
    4: OTHER,   # constrained by draught
    5: MOORED,
    6: OTHER,   # aground
    7: FISHING,
    8: UNDER_WAY_SAILING,
}

# text labels used by the 2021 archive
NAV_STATUS_FROM_LABEL = {
    "UsingEngine": UNDER_WAY_ENGINE,
    "Anchor": AT_ANCHOR,
    "Moored": MOORED,
    "Sailing": UNDER_WAY_SAILING,
    "Fishing": FISHING,
    "Others": OTHER,
}


def nav_status_from_code(code: object) -> str:
    """Map an ITU navigational status code to the harmonised vocabulary.

    Codes 9 to 15 are reserved, undefined, or safety-related, and 99 is a
    provider-specific unknown. None of them names a state the 2021 feed can
    distinguish, so they land in `other` alongside the codes that do.
    """
    if code is None:
        return UNKNOWN
    try:
        value = int(code)
    except (TypeError, ValueError):
        return UNKNOWN
    return NAV_STATUS_FROM_CODE.get(value, OTHER)


def nav_status_from_label(label: object) -> str:
    if label is None:
        return UNKNOWN
    return NAV_STATUS_FROM_LABEL.get(str(label).strip(), OTHER)


# --- The shard schema -----------------------------------------------------

# 511 is the AIS sentinel for "heading not available"; it is stored as null so no
# consumer mistakes it for a bearing.
HEADING_UNAVAILABLE = 511

SHARD_SCHEMA = pa.schema(
    [
        # identity
        pa.field("vessel_uid", pa.int64(), nullable=False),
        pa.field("mmsi", pa.int64(), nullable=True),
        pa.field("imo", pa.int64(), nullable=True),
        # time — naive nanoseconds, matching the legacy feeds; see the note below
        pa.field("UTC", pa.timestamp("ns"), nullable=False),
        pa.field("UTC_10min", pa.timestamp("ns"), nullable=False),
        pa.field("year", pa.int16(), nullable=False),
        # position and motion
        pa.field("lat", pa.float64(), nullable=False),
        pa.field("lon", pa.float64(), nullable=False),
        pa.field("speed", pa.float64(), nullable=True),
        pa.field("course", pa.float64(), nullable=True),
        pa.field("heading", pa.float64(), nullable=True),
        # navigational status
        pa.field("nav_status", pa.string(), nullable=False),
        pa.field("nav_status_code", pa.int16(), nullable=True),
        # provenance
        pa.field("source_provider", pa.string(), nullable=False),
        pa.field("source_resolution_original", pa.string(), nullable=False),
    ]
)

SHARD_COLUMNS = [field.name for field in SHARD_SCHEMA]

# what `cleaning_common.py` requires of any shard it reads
CLEANING_REQUIRED_COLUMNS = ["vessel_uid", "UTC", "lon", "lat", "speed", "course"]
CLEANING_OPTIONAL_COLUMNS = ["heading", "mmsi", "imo"]

BIN_SECONDS = 600


def validate_table(table: pa.Table, year: int) -> None:
    """Raise if a built shard departs from the schema or the invariants."""
    missing = [c for c in CLEANING_REQUIRED_COLUMNS if c not in table.schema.names]
    if missing:
        raise ValueError(f"shard is missing required columns: {missing}")
    if list(table.schema.names) != SHARD_COLUMNS:
        raise ValueError(
            f"shard columns {list(table.schema.names)} do not match the schema"
        )
    for field in SHARD_SCHEMA:
        actual = table.schema.field(field.name).type
        if actual != field.type:
            raise ValueError(
                f"column {field.name} has type {actual}, expected {field.type}"
            )

    import pyarrow.compute as pc

    if table.num_rows == 0:
        return
    if pc.any(pc.not_equal(table["year"], year)).as_py():
        raise ValueError(f"shard carries rows outside year {year}")
    lat, lon = table["lat"], table["lon"]
    if pc.any(pc.or_(pc.less(lat, -90.0), pc.greater(lat, 90.0))).as_py():
        raise ValueError("latitude out of range")
    if pc.any(pc.or_(pc.less(lon, -180.0), pc.greater(lon, 180.0))).as_py():
        raise ValueError("longitude out of range")
    allowed = pa.array(NAV_STATUS_VALUES)
    if pc.any(pc.invert(pc.is_in(table["nav_status"], value_set=allowed))).as_py():
        raise ValueError("nav_status carries a value outside the vocabulary")
