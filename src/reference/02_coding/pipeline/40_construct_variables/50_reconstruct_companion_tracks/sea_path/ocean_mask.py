"""
Preprocessing utility for the GMT/GSHHG Earth Mask.

Reads a locally prepared official GMT 1-arc-minute pixel-registered Earth Mask,
converts it to a boolean NumPy array where True = navigable ocean,
and saves it for fast loading by the routing algorithm.

Source: GMT/GSHHG Earth Mask 01m pixel-registered
        https://www.generic-mapping-tools.org/remote-datasets/earth-mask.html

GSHHG mask encoding:
    0 = ocean (beyond shoreline)
    1 = land (inside shoreline)
    2 = lake (within land)
    3 = island in lake
    4 = pond in island in lake

Only mask == 0 cells are treated as navigable ocean.
Lakes, inland seas, and all land are non-navigable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

GSHHG_URL = (
    "https://oceania.generic-mapping-tools.org"
    "/server/earth/earth_mask/earth_mask_01m_p.grd"
)
GSHHG_FILENAME = "earth_mask_01m_p.grd"

RESOLUTION_DEG = 1 / 60  # 1 arc-minute ≈ 0.01667°
N_LAT = 10800
N_LON = 21600

LAT_MIN = -89.991667
LAT_MAX = 89.991667
LON_MIN = -179.991667
LON_MAX = 179.991667


# ---------------------------------------------------------------------------
# Frozen identity of the routing inputs
# ---------------------------------------------------------------------------
# The 233 MB derived ocean mask is deliberately not committed (see .gitignore),
# yet every routing result depends on it.  Its identity is frozen here instead.
# Rebuilding it from a different GSHHG release, a different resolution, or a
# changed override list moves these digests, and the guard then refuses to run
# rather than silently returning a different geography.
#
# Recorded in CONTRACT.md section 3.  Changing a value here is a protected-
# parameter change and needs an explicit user decision, not a worker edit.

FROZEN_GSHHG_SOURCE_SHA256 = (
    "c058a5e52493dcde749ea3e26ba8a6c5485d362706dd63a4f86345348439a209"
)
FROZEN_OCEAN_MASK_SHA256 = (
    "71f6045e14acc73397efbcb5c7be6e1257c1e949cceb89dc38d001cebf0ac4e8"
)
FROZEN_OCEAN_CELL_COUNT = 153_736_744
FROZEN_CHANNEL_EDGE_COUNT = 108
FROZEN_CHANNEL_EDGE_SHA256 = (
    "8f92261c6ddcb107860b871fb582f39d798280a9380372b40e17da983765c8d9"
)


class OceanMaskIntegrityError(RuntimeError):
    """Raised when a routing input does not match its frozen identity."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


class OceanMaskMetadata(NamedTuple):
    resolution_deg: float
    n_lat: int
    n_lon: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    lat_direction: str
    lon_convention: str
    registration: str
    ocean_cell_count: int
    land_cell_count: int
    ocean_fraction: float


def require_gshhg_mask(dest_dir: Path) -> Path:
    """Require the GMT/GSHHG input locally; acquisition is outside this release."""
    dest_path = dest_dir / GSHHG_FILENAME
    if dest_path.exists():
        print(f"  GSHHG file already exists: {dest_path}")
        return dest_path
    raise FileNotFoundError(f"Prepare {dest_path} before running; source: {GSHHG_URL}")


def read_gshhg_netcdf(nc_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the GSHHG NetCDF and return (mask, lat, lon) arrays.

    Returns
    -------
    mask : int8 array, shape (10800, 21600), values 0–4
    lat  : float64 array, shape (10800,), south-to-north
    lon  : float64 array, shape (21600,), -179.99 to 179.99
    """
    import netCDF4 as nc

    ds = nc.Dataset(str(nc_path))
    try:
        mask = np.asarray(ds.variables["z"][:], dtype=np.int8)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    finally:
        ds.close()

    assert mask.shape == (N_LAT, N_LON), f"Unexpected shape: {mask.shape}"
    assert lat.shape == (N_LAT,)
    assert lon.shape == (N_LON,)
    assert np.isclose(lat[0], LAT_MIN, atol=1e-4), f"Unexpected lat start: {lat[0]}"
    assert np.isclose(lat[-1], LAT_MAX, atol=1e-4), f"Unexpected lat end: {lat[-1]}"
    assert np.isclose(lon[0], LON_MIN, atol=1e-4), f"Unexpected lon start: {lon[0]}"
    assert np.isclose(lon[-1], LON_MAX, atol=1e-4), f"Unexpected lon end: {lon[-1]}"

    step_lat = lat[1] - lat[0]
    step_lon = lon[1] - lon[0]
    assert np.isclose(step_lat, RESOLUTION_DEG, atol=1e-4), f"Unexpected lat step: {step_lat}"
    assert np.isclose(step_lon, RESOLUTION_DEG, atol=1e-4), f"Unexpected lon step: {step_lon}"

    return mask, lat, lon


def build_ocean_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Convert GSHHG integer mask to boolean ocean mask.

    Only cells with value 0 (ocean) are navigable.
    Land (1), lakes (2), islands in lakes (3), ponds (4) are all non-navigable.
    """
    return raw_mask == 0


def latlon_to_index(lat_val: float, lon_val: float) -> tuple[int, int]:
    """Convert latitude/longitude to grid indices.

    Parameters
    ----------
    lat_val : float, degrees north (-90 to 90)
    lon_val : float, degrees east (-180 to 180)

    Returns
    -------
    (i_lat, i_lon) : grid indices into the (10800, 21600) array
    """
    i_lat = int(round((lat_val - LAT_MIN) / RESOLUTION_DEG))
    i_lon = int(round((lon_val - LON_MIN) / RESOLUTION_DEG))
    i_lat = max(0, min(N_LAT - 1, i_lat))
    i_lon = max(0, min(N_LON - 1, i_lon))
    return i_lat, i_lon


def index_to_latlon(i_lat: int, i_lon: int) -> tuple[float, float]:
    """Convert grid indices back to cell-center latitude/longitude."""
    lat_val = LAT_MIN + i_lat * RESOLUTION_DEG
    lon_val = LON_MIN + i_lon * RESOLUTION_DEG
    return lat_val, lon_val


def save_ocean_mask(ocean: np.ndarray, dest_path: Path) -> None:
    """Save the boolean ocean mask as a .npy file."""
    assert ocean.shape == (N_LAT, N_LON)
    assert ocean.dtype == np.bool_
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest_path, ocean)
    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"  Saved ocean mask: {dest_path} ({size_mb:.1f} MB)")


def load_ocean_mask(mask_path: Path, *, verify: bool = True) -> np.ndarray:
    """Load the pre-built boolean ocean mask.

    ``verify`` checks the file against its frozen SHA-256 and cell count before
    the array is handed to a router.  Hashing the 233 MB file costs about 0.2 s,
    so the guard stays on by default; pass ``verify=False`` only to inspect a
    mask that is deliberately being rebuilt.
    """
    ocean = np.load(mask_path)
    if ocean.shape != (N_LAT, N_LON):
        raise OceanMaskIntegrityError(f"Unexpected shape: {ocean.shape}")
    if ocean.dtype != np.bool_:
        raise OceanMaskIntegrityError(f"Unexpected dtype: {ocean.dtype}")
    if verify:
        verify_routing_inputs(mask_path=Path(mask_path), ocean=ocean)
    return ocean


def compute_metadata(ocean: np.ndarray) -> OceanMaskMetadata:
    """Compute summary metadata for the ocean mask."""
    ocean_count = int(ocean.sum())
    total = ocean.size
    return OceanMaskMetadata(
        resolution_deg=RESOLUTION_DEG,
        n_lat=N_LAT,
        n_lon=N_LON,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        lat_direction="south_to_north",
        lon_convention="minus180_to_180",
        registration="pixel_center",
        ocean_cell_count=ocean_count,
        land_cell_count=total - ocean_count,
        ocean_fraction=ocean_count / total,
    )


MARITIME_CHANNEL_OVERRIDES: dict[str, list[tuple[float, float]]] = {
    "suez_canal": [
        (31.158333, 32.291667),
        (31.158333, 32.308333),
        (31.141667, 32.325),
        (31.125, 32.325),
        (31.108333, 32.325),
        (31.091667, 32.325),
        (31.075, 32.325),
        (31.058333, 32.325),
        (31.041667, 32.325),
        (31.025, 32.325),
        (31.008333, 32.325),
        (30.991667, 32.325),
        (30.975, 32.325),
        (30.958333, 32.325),
        (30.941667, 32.325),
        (30.925, 32.325),
        (30.908333, 32.325),
        (30.891667, 32.325),
        (30.875, 32.325),
        (30.858333, 32.325),
        (30.841667, 32.325),
        (30.825, 32.325),
        (30.808333, 32.325),
        (30.791667, 32.325),
        (30.775, 32.325),
        (30.758333, 32.325),
        (30.741667, 32.308333),
        (30.725, 32.325),
        (30.708333, 32.325),
        (30.691667, 32.308333),
        (30.675, 32.308333),
        (30.658333, 32.308333),
        (30.641667, 32.308333),
        (30.625, 32.308333),
        (30.608333, 32.308333),
        (30.591667, 32.291667),
        (30.575, 32.308333),
        (30.558333, 32.291667),
        (30.541667, 32.291667),
        (30.525, 32.291667),
        (30.508333, 32.308333),
        (30.491667, 32.308333),
        (30.475, 32.325),
        (30.458333, 32.341667),
        (30.441667, 32.358333),
        (30.425, 32.375),
        (30.408333, 32.391667),
        (30.391667, 32.408333),
        (30.375, 32.425),
        (30.358333, 32.441667),
        (30.341667, 32.458333),
        (30.325, 32.458333),
        (30.308333, 32.458333),
        (30.291667, 32.475),
        (30.275, 32.491667),
        (30.258333, 32.475),
        (30.241667, 32.475),
        (30.225, 32.458333),
        (30.208333, 32.441667),
        (30.191667, 32.425),
        (30.175, 32.425),
        (30.158333, 32.425),
        (30.141667, 32.441667),
        (30.125, 32.441667),
        (30.108333, 32.441667),
        (30.091667, 32.458333),
        (30.075, 32.458333),
        (30.058333, 32.458333),
        (30.041667, 32.475),
        (30.025, 32.491667),
        (30.008333, 32.491667),
        (29.991667, 32.508333),
        (29.975, 32.525),
        (29.958333, 32.525),
    ],
    "panama_canal": [
        (9.375, -79.858333),
        (9.358333, -79.841667),
        (9.341667, -79.825),
        (9.325, -79.808333),
        (9.308333, -79.808333),
        (9.291667, -79.825),
        (9.275, -79.808333),
        (9.258333, -79.808333),
        (9.241667, -79.825),
        (9.225, -79.825),
        (9.208333, -79.808333),
        (9.191667, -79.791667),
        (9.175, -79.775),
        (9.158333, -79.791667),
        (9.141667, -79.775),
        (9.125, -79.758333),
        (9.108333, -79.741667),
        (9.091667, -79.725),
        (9.075, -79.708333),
        (9.058333, -79.691667),
        (9.041667, -79.675),
        (9.025, -79.658333),
        (9.008333, -79.641667),
        (9.008333, -79.625),
        (8.991667, -79.608333),
        (8.975, -79.591667),
    ],
}

# Named, resolution-specific diagonal transitions where the official 01m mask
# represents a real navigable strait only as corner-touching ocean cells.  The
# Suez, Panama, and Bosporus edges below are the minimal sea-to-channel or
# south-to-north connections; unrelated corner contacts inside the same
# diagnostic boxes remain forbidden.
MARITIME_CHANNEL_EDGE_OVERRIDES: dict[
    str, list[tuple[tuple[int, int], tuple[int, int]]]
] = {
    "suez_canal": [
        # Mediterranean entrance to the first ordered canal cell.
        ((7268, 12736), (7269, 12737)),
    ],
    "panama_canal": [
        # Caribbean entrance to the first ordered canal cell.
        ((5963, 6009), (5962, 6008)),
        # Last ordered canal cell to the Pacific entrance water cell.
        ((5938, 6024), (5937, 6025)),
        ((5937, 6025), (5936, 6026)),
    ],
    "bosporus": [
        ((7861, 12540), (7862, 12541)),
        ((7862, 12541), (7863, 12542)),
        ((7863, 12542), (7864, 12543)),
        ((7867, 12544), (7868, 12543)),
        ((7869, 12543), (7870, 12544)),
        ((7870, 12544), (7871, 12545)),
    ],
}


def maritime_channel_edges(
    channels: dict[str, list[tuple[float, float]]] | None = None,
) -> frozenset[tuple[tuple[int, int], tuple[int, int]]]:
    """Return explicit grid edges along the hand-verified canal centre-lines.

    Ordinary diagonal movement is conservative and may not pass between land
    cells that touch at a corner.  Named Suez and Panama centre-lines, plus the
    separately enumerated Suez, Panama, and Bosporus entrance/strait edges,
    describe real navigable channels that the 1-arc-minute raster cannot
    represent safely.  Keeping these exceptions as named edges prevents the
    global corner rule from silently opening unrelated coastlines.
    """
    source = MARITIME_CHANNEL_OVERRIDES if channels is None else channels
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for name, points in source.items():
        ordered_cells: list[tuple[int, int]] = []
        for point in points:
            cell = latlon_to_index(*point)
            if not ordered_cells or cell != ordered_cells[-1]:
                ordered_cells.append(cell)

        for left, right in zip(ordered_cells, ordered_cells[1:]):
            di = abs(right[0] - left[0])
            raw_dj = abs(right[1] - left[1])
            dj = min(raw_dj, N_LON - raw_dj)
            if max(di, dj) != 1:
                raise ValueError(
                    f"Non-adjacent {name} channel cells: {left} -> {right}"
                )
            edges.add((left, right) if left <= right else (right, left))

    if channels is None:
        for name, named_edges in MARITIME_CHANNEL_EDGE_OVERRIDES.items():
            for left, right in named_edges:
                di = abs(right[0] - left[0])
                raw_dj = abs(right[1] - left[1])
                dj = min(raw_dj, N_LON - raw_dj)
                if max(di, dj) != 1:
                    raise ValueError(
                        f"Non-adjacent {name} channel edge: {left} -> {right}"
                    )
                edges.add((left, right) if left <= right else (right, left))

    return frozenset(edges)


def channel_edge_digest(
    edges: frozenset[tuple[tuple[int, int], tuple[int, int]]] | None = None,
) -> str:
    """Return a stable digest of the declared sub-grid channel edge set.

    These edges are the only permitted corner-touching transitions in the whole
    router, so an unnoticed addition would silently open a coastline.  Hashing
    the canonical sorted set makes any change visible.
    """
    resolved = maritime_channel_edges() if edges is None else edges
    return hashlib.sha256(repr(sorted(resolved)).encode("utf-8")).hexdigest()


def verify_routing_inputs(
    *,
    mask_path: Path | None = None,
    ocean: np.ndarray | None = None,
) -> dict[str, object]:
    """Check the routing inputs against their frozen identity.

    Verifies whichever inputs are supplied: the mask file digest, the loaded
    array's navigable-cell count, and always the declared channel edge set.
    Raises :class:`OceanMaskIntegrityError` listing every mismatch at once, so a
    rebuild does not have to be diagnosed one failure at a time.
    """
    report: dict[str, object] = {}
    problems: list[str] = []

    if mask_path is not None:
        actual = sha256_file(mask_path)
        report["mask_sha256"] = actual
        if actual != FROZEN_OCEAN_MASK_SHA256:
            problems.append(
                f"ocean mask SHA-256 is {actual}, "
                f"expected {FROZEN_OCEAN_MASK_SHA256}"
            )

    if ocean is not None:
        cells = int(ocean.sum())
        report["ocean_cell_count"] = cells
        report["ocean_fraction"] = cells / ocean.size
        if cells != FROZEN_OCEAN_CELL_COUNT:
            problems.append(
                f"navigable cell count is {cells}, "
                f"expected {FROZEN_OCEAN_CELL_COUNT}"
            )

    edges = maritime_channel_edges()
    digest = channel_edge_digest(edges)
    report["channel_edge_count"] = len(edges)
    report["channel_edge_sha256"] = digest
    if len(edges) != FROZEN_CHANNEL_EDGE_COUNT:
        problems.append(
            f"declared channel edge count is {len(edges)}, "
            f"expected {FROZEN_CHANNEL_EDGE_COUNT}"
        )
    if digest != FROZEN_CHANNEL_EDGE_SHA256:
        problems.append(
            f"declared channel edge digest is {digest}, "
            f"expected {FROZEN_CHANNEL_EDGE_SHA256}"
        )

    report["verified"] = not problems
    if problems:
        raise OceanMaskIntegrityError(
            "Routing inputs do not match their frozen identity recorded in "
            "CONTRACT.md section 3:\n  - " + "\n  - ".join(problems)
        )
    return report


def apply_maritime_channel_overrides(ocean: np.ndarray) -> dict[str, int]:
    """Open narrow man-made channels that are sub-grid at 01m.

    Modifies the mask in place.  Returns the number of cells opened
    per channel (cells already ocean are not counted).
    """
    report: dict[str, int] = {}
    for name, cells in MARITIME_CHANNEL_OVERRIDES.items():
        opened = 0
        for lat, lon in cells:
            i, j = latlon_to_index(lat, lon)
            if not ocean[i, j]:
                ocean[i, j] = True
                opened += 1
        report[name] = opened
    return report


def preprocess(
    raw_dir: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> OceanMaskMetadata:
    """Offline preprocessing pipeline: read -> convert -> save.

    Parameters
    ----------
    raw_dir : directory containing the prepared GSHHG NetCDF file
    output_path : where to write the boolean .npy ocean mask
    force : if True, overwrite existing output

    Returns the mask metadata.
    """
    if output_path.exists() and not force:
        print(f"Ocean mask already exists: {output_path}")
        ocean = load_ocean_mask(output_path)
        return compute_metadata(ocean)

    print("Step 1: Locate prepared GSHHG Earth Mask")
    nc_path = require_gshhg_mask(raw_dir)

    nc_hash = hashlib.sha256(nc_path.read_bytes()).hexdigest()
    print(f"  SHA-256: {nc_hash}")

    print("Step 2: Read NetCDF")
    raw_mask, lat, lon = read_gshhg_netcdf(nc_path)
    unique, counts = np.unique(raw_mask, return_counts=True)
    for val, cnt in zip(unique, counts):
        label = {0: "ocean", 1: "land", 2: "lake", 3: "island_in_lake", 4: "pond"}.get(
            val, "unknown"
        )
        print(f"  value {val} ({label}): {cnt:,} cells ({cnt/raw_mask.size*100:.2f}%)")

    print("Step 3: Build boolean ocean mask (only value 0 = navigable)")
    ocean = build_ocean_mask(raw_mask)
    assert ocean.dtype == np.bool_

    print("Step 4: Apply maritime channel overrides")
    channel_report = apply_maritime_channel_overrides(ocean)
    for name, opened in channel_report.items():
        print(f"  {name}: {opened} cells opened")

    print("Step 5: Save")
    save_ocean_mask(ocean, output_path)

    meta = compute_metadata(ocean)
    print(f"\nOcean mask metadata:")
    print(f"  Resolution: {meta.resolution_deg:.6f} deg (1 arc-minute)")
    print(f"  Shape: ({meta.n_lat}, {meta.n_lon})")
    print(f"  Lat: {meta.lat_min} to {meta.lat_max} ({meta.lat_direction})")
    print(f"  Lon: {meta.lon_min} to {meta.lon_max} ({meta.lon_convention})")
    print(f"  Registration: {meta.registration}")
    print(f"  Ocean cells: {meta.ocean_cell_count:,} ({meta.ocean_fraction:.4%})")
    print(f"  Non-ocean cells: {meta.land_cell_count:,}")

    return meta
