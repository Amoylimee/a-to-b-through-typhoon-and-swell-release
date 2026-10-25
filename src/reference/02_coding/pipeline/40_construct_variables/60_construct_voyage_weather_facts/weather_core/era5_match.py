"""Cached, exact-hour ERA5 SWH matching on a canonical tile union."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr

from .geo import chord_for_km, great_circle_km, lon_lat_to_unit_xyz, normalize_longitude


GRID_STEP_DEGREES = 0.5
GRID_HALF_STEP_DEGREES = GRID_STEP_DEGREES / 2.0
OVERLAP_TOLERANCE_M = 1e-5
HIGH_WAVE_THRESHOLD_M = 4.0


@dataclass(frozen=True)
class Era5Match:
    swh_m: float
    matched_lat: float
    matched_lon: float
    snap_distance_km: float
    match_status: str
    source_path: str | None


def _coordinate_name(dataset: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in dataset.coords:
            return name
    raise ValueError(f"ERA5 dataset lacks coordinate alternatives {names}")


def _validated_axis(values, *, longitude: bool, source: str) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    label = "longitude" if longitude else "latitude"
    if raw.ndim != 1 or len(raw) == 0 or not np.isfinite(raw).all():
        raise ValueError(f"{source}: {label} must be a nonempty finite 1-D coordinate")
    if longitude:
        if ((raw < -180.0) | (raw > 360.0)).any():
            raise ValueError(f"{source}: longitude coordinate outside [-180,360]")
    elif ((raw < -90.0) | (raw > 90.0)).any():
        raise ValueError(f"{source}: latitude coordinate outside [-90,90]")
    if not np.allclose(raw * 2.0, np.rint(raw * 2.0), rtol=0.0, atol=1e-9):
        raise ValueError(f"{source}: {label} coordinates are offset from the 0.5 degree lattice")
    if len(raw) > 1:
        difference = np.diff(raw)
        if not (np.all(difference > 0.0) or np.all(difference < 0.0)):
            raise ValueError(f"{source}: {label} must be strictly monotone")
        if not np.allclose(np.abs(difference), GRID_STEP_DEGREES, rtol=0.0, atol=1e-9):
            raise ValueError(f"{source}: {label} grid is not exactly 0.5 degrees")
    canonical = normalize_longitude(raw) if longitude else raw
    return np.unique(np.round(canonical * 2.0).astype(np.int16)).astype(np.float64) / 2.0


def _cell_code(latitude, longitude) -> np.ndarray:
    lat_q = np.rint(np.asarray(latitude, dtype=np.float64) * 2.0).astype(np.int64)
    lon_q = np.rint(normalize_longitude(longitude) * 2.0).astype(np.int64)
    return (lat_q + 180) * 1000 + (lon_q + 360)


def _decode_cell_code(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    code = np.asarray(code, dtype=np.int64)
    lat_q = code // 1000 - 180
    lon_q = code % 1000 - 360
    return lat_q.astype(np.float64) / 2.0, lon_q.astype(np.float64) / 2.0


def _nearest_linear(axis: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(axis, target, side="left")
    left = np.clip(right - 1, 0, len(axis) - 1)
    right = np.clip(right, 0, len(axis) - 1)
    choose_right = np.abs(axis[right] - target) < np.abs(axis[left] - target)
    index = np.where(choose_right, right, left)
    return axis[index], np.abs(axis[index] - target)


def _nearest_circular(axis: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extended = np.concatenate([axis - 360.0, axis, axis + 360.0])
    nearest, _ = _nearest_linear(extended, target)
    canonical = normalize_longitude(nearest)
    distance = np.abs(((target - canonical + 180.0) % 360.0) - 180.0)
    return canonical, distance


class _Tile:
    def __init__(
        self,
        *,
        source: str,
        raw_latitude: np.ndarray,
        raw_longitude: np.ndarray,
        dataset: xr.Dataset | None = None,
        fixed_values: np.ndarray | None = None,
    ) -> None:
        self.source = str(source)
        self.raw_latitude = np.asarray(raw_latitude, dtype=np.float64)
        self.raw_longitude = np.asarray(raw_longitude, dtype=np.float64)
        self.latitude = _validated_axis(self.raw_latitude, longitude=False, source=self.source)
        self.longitude = _validated_axis(self.raw_longitude, longitude=True, source=self.source)
        self.dataset = dataset
        self.fixed_values = fixed_values
        raw_lon_grid, raw_lat_grid = np.meshgrid(self.raw_longitude, self.raw_latitude)
        self.flat_codes = _cell_code(raw_lat_grid.ravel(), raw_lon_grid.ravel())
        self.union_mapping: np.ndarray | None = None
        self.group_order: np.ndarray | None = None
        self.group_starts: np.ndarray | None = None
        self.group_counts: np.ndarray | None = None
        self.unique_union_indices: np.ndarray | None = None
        self.time_name: str | None = None
        self.lat_name: str | None = None
        self.lon_name: str | None = None
        self.extra_dims: dict[str, int] = {}

    def configure_dataset(self) -> pd.DatetimeIndex:
        if self.dataset is None:
            return pd.DatetimeIndex([])
        dataset = self.dataset
        if "swh" not in dataset.data_vars:
            raise ValueError(f"{self.source}: ERA5 asset lacks swh")
        self.lat_name = _coordinate_name(dataset, ("latitude", "lat"))
        self.lon_name = _coordinate_name(dataset, ("longitude", "lon"))
        self.time_name = _coordinate_name(dataset, ("valid_time", "time"))
        allowed = {self.time_name, self.lat_name, self.lon_name}
        for dimension in dataset["swh"].dims:
            if dimension not in allowed:
                if dataset.sizes[dimension] != 1:
                    raise ValueError(f"{self.source}: extra SWH dimension {dimension} is not singleton")
                self.extra_dims[dimension] = 0
        times = pd.DatetimeIndex(pd.to_datetime(dataset[self.time_name].values, utc=True))
        if times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError(f"{self.source}: ERA5 times must be unique and increasing")
        if not (times == times.floor("h")).all():
            raise ValueError(f"{self.source}: ERA5 time coordinate is not exact hourly UTC")
        return times

    def read_values(self, time_index: int) -> np.ndarray:
        if self.fixed_values is not None:
            value = np.asarray(self.fixed_values, dtype=np.float64)
        else:
            assert self.dataset is not None
            assert self.time_name and self.lat_name and self.lon_name
            data = self.dataset["swh"].isel(
                {self.time_name: int(time_index), **self.extra_dims}
            ).transpose(self.lat_name, self.lon_name)
            value = np.asarray(data.values, dtype=np.float64)
        expected = (len(self.raw_latitude), len(self.raw_longitude))
        if value.shape != expected:
            raise ValueError(f"{self.source}: unexpected SWH shape {value.shape}, required {expected}")
        return value.ravel()


class _CanonicalGrid:
    def __init__(self, tiles: Iterable[_Tile]) -> None:
        self.tiles = tuple(sorted(tiles, key=lambda tile: tile.source))
        if not self.tiles:
            raise ValueError("ERA5 canonical grid requires at least one tile")
        self.codes = np.unique(np.concatenate([tile.flat_codes for tile in self.tiles]))
        self.latitude, self.longitude = _decode_cell_code(self.codes)
        self.xyz = lon_lat_to_unit_xyz(self.longitude, self.latitude)
        self.tree = cKDTree(self.xyz)
        self.source = np.empty(len(self.codes), dtype=object)
        self.source[:] = None
        for tile in self.tiles:
            mapping = np.searchsorted(self.codes, tile.flat_codes)
            if not np.array_equal(self.codes[mapping], tile.flat_codes):
                raise RuntimeError("ERA5 canonical cell mapping failed")
            tile.union_mapping = mapping
            order = np.argsort(mapping, kind="stable")
            ordered = mapping[order]
            unique, starts, counts = np.unique(
                ordered, return_index=True, return_counts=True
            )
            tile.group_order = order
            tile.group_starts = starts
            tile.group_counts = counts
            tile.unique_union_indices = unique
            unset = pd.isna(self.source[unique])
            self.source[unique[unset]] = tile.source

    @staticmethod
    def _collapse_tile(tile: _Tile, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert tile.group_order is not None
        assert tile.group_starts is not None
        assert tile.group_counts is not None
        assert tile.unique_union_indices is not None
        ordered = values[tile.group_order]
        if np.isinf(ordered).any():
            raise ValueError(f"{tile.source}: SWH contains infinite values")
        finite = np.isfinite(ordered)
        finite_count = np.add.reduceat(finite.astype(np.int64), tile.group_starts)
        mask_conflict = (finite_count > 0) & (finite_count < tile.group_counts)
        if mask_conflict.any():
            raise ValueError(f"{tile.source}: duplicate seam has inconsistent NaN mask")
        minimum = np.minimum.reduceat(np.where(finite, ordered, np.inf), tile.group_starts)
        maximum = np.maximum.reduceat(np.where(finite, ordered, -np.inf), tile.group_starts)
        both_sides = (
            np.logical_or.reduceat(finite & (ordered < HIGH_WAVE_THRESHOLD_M), tile.group_starts)
            & np.logical_or.reduceat(finite & (ordered >= HIGH_WAVE_THRESHOLD_M), tile.group_starts)
        )
        if np.any((finite_count > 0) & (maximum - minimum > OVERLAP_TOLERANCE_M)):
            raise ValueError(f"{tile.source}: duplicate seam SWH differs by more than 1e-5 m")
        if both_sides.any():
            raise ValueError(f"{tile.source}: duplicate seam flips the 4 m threshold")
        collapsed = np.where(finite_count > 0, minimum, np.nan)
        return tile.unique_union_indices, collapsed

    def merge_hour(self, time_index: int) -> np.ndarray:
        merged = np.full(len(self.codes), np.nan, dtype=np.float64)
        assigned = np.zeros(len(self.codes), dtype=bool)
        for tile in self.tiles:
            index, value = self._collapse_tile(tile, tile.read_values(time_index))
            overlap = assigned[index]
            if overlap.any():
                prior = merged[index[overlap]]
                new = value[overlap]
                prior_finite, new_finite = np.isfinite(prior), np.isfinite(new)
                if np.any(prior_finite != new_finite):
                    raise ValueError("overlapping ERA5 tiles have inconsistent NaN masks")
                both = prior_finite & new_finite
                if np.any(np.abs(prior[both] - new[both]) > OVERLAP_TOLERANCE_M):
                    raise ValueError("overlapping ERA5 tiles differ by more than 1e-5 m")
                if np.any((prior[both] >= 4.0) != (new[both] >= 4.0)):
                    raise ValueError("overlapping ERA5 tiles flip the 4 m threshold")
            new_cell = ~overlap
            merged[index[new_cell]] = value[new_cell]
            assigned[index] = True
        return merged

    def _nominal_indices(self, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        best_index = np.full(len(latitude), -1, dtype=np.int64)
        best_distance = np.full(len(latitude), np.inf, dtype=np.float64)
        for tile in self.tiles:
            nearest_lat, lat_delta = _nearest_linear(tile.latitude, latitude)
            nearest_lon, lon_delta = _nearest_circular(tile.longitude, longitude)
            covered = (
                (lat_delta <= GRID_HALF_STEP_DEGREES + 1e-12)
                & (lon_delta <= GRID_HALF_STEP_DEGREES + 1e-12)
            )
            if not covered.any():
                continue
            candidate_code = _cell_code(nearest_lat, nearest_lon)
            candidate_index = np.searchsorted(self.codes, candidate_code)
            if np.any(self.codes[candidate_index] != candidate_code):
                raise RuntimeError("analytic ERA5 nominal index is absent from canonical union")
            distance = np.asarray(
                great_circle_km(longitude, latitude, nearest_lon, nearest_lat), dtype=np.float64
            )
            better = covered & (
                (distance < best_distance - 1e-12)
                | (
                    np.isclose(distance, best_distance, rtol=0.0, atol=1e-12)
                    & ((best_index < 0) | (candidate_index < best_index))
                )
            )
            best_distance[better] = distance[better]
            best_index[better] = candidate_index[better]
        return best_index

    def match_batch(
        self,
        latitude,
        longitude,
        values: np.ndarray,
        *,
        maximum_wet_cell_snap_km: float,
    ) -> pd.DataFrame:
        raw_lat = np.asarray(latitude, dtype=np.float64)
        raw_lon = np.asarray(longitude, dtype=np.float64)
        if raw_lat.shape != raw_lon.shape:
            raise ValueError("ERA5 latitude and longitude arrays must have the same shape")
        raw_lat, raw_lon = raw_lat.ravel(), raw_lon.ravel()
        count = len(raw_lat)
        result_value = np.full(count, np.nan)
        result_lat = np.full(count, np.nan)
        result_lon = np.full(count, np.nan)
        result_distance = np.full(count, np.nan)
        result_status = np.full(count, "invalid_position", dtype=object)
        result_source = np.full(count, None, dtype=object)
        valid = (
            np.isfinite(raw_lat)
            & np.isfinite(raw_lon)
            & (raw_lat >= -90.0)
            & (raw_lat <= 90.0)
            & (raw_lon >= -180.0)
            & (raw_lon <= 180.0)
        )
        if not valid.any():
            return pd.DataFrame(
                {
                    "swh_m": result_value, "matched_lat": result_lat,
                    "matched_lon": result_lon, "snap_distance_km": result_distance,
                    "match_status": result_status, "source_path": result_source,
                }
            )
        canonical_lon = normalize_longitude(raw_lon)
        nominal = np.full(count, -1, dtype=np.int64)
        nominal[valid] = self._nominal_indices(raw_lat[valid], canonical_lon[valid])
        outside = valid & (nominal < 0)
        result_status[outside] = "outside_assets"
        inside = valid & (nominal >= 0)
        finite_nominal = inside & np.isfinite(values[np.maximum(nominal, 0)])
        if finite_nominal.any():
            cell = nominal[finite_nominal]
            result_value[finite_nominal] = values[cell]
            result_lat[finite_nominal] = self.latitude[cell]
            result_lon[finite_nominal] = self.longitude[cell]
            result_distance[finite_nominal] = great_circle_km(
                canonical_lon[finite_nominal], raw_lat[finite_nominal],
                self.longitude[cell], self.latitude[cell],
            )
            result_status[finite_nominal] = "nearest_grid"
            result_source[finite_nominal] = self.source[cell]
        fallback = np.flatnonzero(inside & ~finite_nominal)
        any_finite = np.isfinite(values).any()
        cap = float(maximum_wet_cell_snap_km)
        if len(fallback) and not any_finite:
            result_status[fallback] = "no_finite_wet_cell"
            fallback = np.asarray([], dtype=np.int64)
        fallback_xyz = lon_lat_to_unit_xyz(
            canonical_lon[fallback], raw_lat[fallback]
        ) if len(fallback) else np.empty((0, 3))
        neighbor_lists = self.tree.query_ball_point(
            fallback_xyz, chord_for_km(cap + 1e-9)
        ) if len(fallback) else []
        for row, candidate_neighbors in zip(fallback, neighbor_lists):
            neighbors = np.asarray(candidate_neighbors, dtype=np.int64)
            neighbors = neighbors[np.isfinite(values[neighbors])]
            if len(neighbors) == 0:
                result_status[row] = "wet_cell_beyond_cap"
                continue
            distance = np.asarray(
                great_circle_km(
                    np.full(len(neighbors), canonical_lon[row]),
                    np.full(len(neighbors), raw_lat[row]),
                    self.longitude[neighbors], self.latitude[neighbors],
                )
            )
            eligible = distance <= cap + 1e-9
            if not eligible.any():
                result_status[row] = "wet_cell_beyond_cap"
                continue
            neighbors, distance = neighbors[eligible], distance[eligible]
            chosen_order = np.lexsort((neighbors, distance))
            cell = neighbors[chosen_order[0]]
            result_value[row] = values[cell]
            result_lat[row] = self.latitude[cell]
            result_lon[row] = self.longitude[cell]
            result_distance[row] = distance[chosen_order[0]]
            result_status[row] = "nearest_finite_wet_cell"
            result_source[row] = self.source[cell]
        return pd.DataFrame(
            {
                "swh_m": result_value, "matched_lat": result_lat,
                "matched_lon": result_lon, "snap_distance_km": result_distance,
                "match_status": result_status, "source_path": result_source,
            }
        )


def match_era5_hour_arrays(
    latitude: float,
    longitude: float,
    blocks: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    *,
    maximum_wet_cell_snap_km: float = 117.94,
) -> Era5Match:
    """One-point test/probe adapter using the same canonical batch engine."""

    tiles = [
        _Tile(
            source=str(source), raw_latitude=np.asarray(lat), raw_longitude=np.asarray(lon),
            fixed_values=np.asarray(value),
        )
        for lat, lon, value, source in blocks
    ]
    grid = _CanonicalGrid(tiles)
    values = grid.merge_hour(0)
    row = grid.match_batch(
        [latitude], [longitude], values,
        maximum_wet_cell_snap_km=maximum_wet_cell_snap_km,
    ).iloc[0]
    return Era5Match(**row.to_dict())


class Era5MonthlyMatcher:
    """Open monthly tiles once; reuse canonical grid and a bounded hour cache."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        *,
        maximum_wet_cell_snap_km: float = 117.94,
        cached_hours: int = 2,
    ) -> None:
        self.paths = tuple(sorted({Path(path).resolve() for path in paths}, key=str))
        if not self.paths:
            raise ValueError("ERA5 matcher requires at least one monthly tile")
        self.maximum_wet_cell_snap_km = float(maximum_wet_cell_snap_km)
        self.cached_hours = max(1, int(cached_hours))
        self._datasets: list[xr.Dataset] = []
        tiles: list[_Tile] = []
        times: pd.DatetimeIndex | None = None
        try:
            for path in self.paths:
                dataset = xr.open_dataset(path)
                self._datasets.append(dataset)
                lat_name = _coordinate_name(dataset, ("latitude", "lat"))
                lon_name = _coordinate_name(dataset, ("longitude", "lon"))
                tile = _Tile(
                    source=str(path),
                    raw_latitude=np.asarray(dataset[lat_name].values),
                    raw_longitude=np.asarray(dataset[lon_name].values),
                    dataset=dataset,
                )
                tile_times = tile.configure_dataset()
                if times is None:
                    times = tile_times
                elif not times.equals(tile_times):
                    raise ValueError("ERA5 monthly tiles do not share the identical hourly time grid")
                tiles.append(tile)
            self.times = times if times is not None else pd.DatetimeIndex([])
            self.grid = _CanonicalGrid(tiles)
            self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        except Exception:
            self.close()
            raise

    def preload_swh(self) -> int:
        """Decode each monthly SWH array once, preserving matching semantics.

        Some source files use 186-hour compressed chunks. Hour-at-a-time lazy
        reads otherwise decompress a large chunk repeatedly. Return resident
        array bytes so the caller can account for this memory tradeoff.
        """
        for dataset in self._datasets:
            dataset["swh"].load()
        return sum(dataset["swh"].nbytes for dataset in self._datasets)

    def close(self) -> None:
        for dataset in getattr(self, "_datasets", []):
            dataset.close()
        self._datasets = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _values_for_index(self, index: int) -> np.ndarray:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        value = self.grid.merge_hour(index)
        self._cache[index] = value
        while len(self._cache) > self.cached_hours:
            self._cache.popitem(last=False)
        return value

    def match(self, positions: pd.DataFrame) -> pd.DataFrame:
        required = {"hour_anchor_utc", "lat", "lon"}
        if missing := sorted(required - set(positions.columns)):
            raise ValueError(f"ERA5 positions missing columns: {missing}")
        frame = positions.reset_index(drop=True).copy()
        result_columns = [
            "swh_m", "matched_lat", "matched_lon", "snap_distance_km",
            "match_status", "source_path",
        ]
        if frame.empty:
            return frame.assign(
                swh_m=pd.Series(dtype="float64"), matched_lat=pd.Series(dtype="float64"),
                matched_lon=pd.Series(dtype="float64"), snap_distance_km=pd.Series(dtype="float64"),
                match_status=pd.Series(dtype="object"), source_path=pd.Series(dtype="object"),
            )
        frame["hour_anchor_utc"] = pd.to_datetime(
            frame["hour_anchor_utc"], utc=True, errors="coerce"
        )
        if frame["hour_anchor_utc"].isna().any():
            raise ValueError("ERA5 positions contain invalid hour anchors")
        if not frame["hour_anchor_utc"].eq(frame["hour_anchor_utc"].dt.floor("h")).all():
            raise ValueError("ERA5 requires exact integer UTC hour anchors including nanoseconds")
        output = pd.DataFrame(index=frame.index, columns=result_columns)
        for hour, group in frame.groupby("hour_anchor_utc", sort=True):
            time_index = self.times.get_indexer([hour])[0]
            if time_index < 0:
                raise KeyError(f"ERA5 exact hour absent: {hour}")
            matched = self.grid.match_batch(
                group["lat"].to_numpy(), group["lon"].to_numpy(),
                self._values_for_index(int(time_index)),
                maximum_wet_cell_snap_km=self.maximum_wet_cell_snap_km,
            )
            output.loc[group.index, result_columns] = matched[result_columns].to_numpy()
        for column in ("swh_m", "matched_lat", "matched_lon", "snap_distance_km"):
            output[column] = pd.to_numeric(output[column], errors="coerce")
        return pd.concat([frame, output], axis=1)
