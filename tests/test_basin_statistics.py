"""Tests for ``WatershedRaster.statistics`` (P17)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import Point

from digitalrivers import DEM


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def _build(z: np.ndarray, cell_size: float = 1.0):
    dem = _make_dem(z, cell_size=cell_size)
    fd = dem.flow_direction(method="d8")
    return dem, fd


def test_area_km2_matches_cell_count_for_unit_cell():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd = _build(z)
    ws = fd.basins()
    df = ws.statistics()
    # With cell_size=1 m, each cell is 1 m^2; area_km2 = cell_count / 1e6.
    arr = ws.read_array()
    for bid, row in df.iterrows():
        cells = int((arr == bid).sum())
        assert row["area_km2"] == pytest.approx(cells / 1.0e6)


def test_elevation_stats_match_basin_dem_values():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd = _build(z)
    ws = fd.basins()
    df = ws.statistics(dem=dem)
    assert "min_elev" in df.columns
    assert "max_elev" in df.columns
    assert "mean_elev" in df.columns
    # Hypsometric integral is in [0, 1].
    assert ((df["hypsometric_integral"] >= 0)
            & (df["hypsometric_integral"] <= 1)).all()


def test_drainage_density_uses_stream_length():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=1)
    ws = fd.basins()
    df = ws.statistics(dem=dem, streams=sr)
    assert "drainage_density_km_per_km2" in df.columns
    # Non-negative.
    assert (df["drainage_density_km_per_km2"] >= 0).all()


def test_metrics_subset_filters_columns():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd = _build(z)
    ws = fd.basins()
    df = ws.statistics(dem=dem, metrics=["area_km2"])
    assert list(df.columns) == ["area_km2"]


def test_centroid_in_output_when_dem_passed():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd = _build(z)
    ws = fd.basins()
    df = ws.statistics(dem=dem)
    assert "centroid_x" in df.columns
    assert "centroid_y" in df.columns


def test_no_basins_returns_empty():
    z = np.full((3, 3), 5.0, dtype=np.float32)
    dem, fd = _build(z)
    ws = fd.basins()
    df = ws.statistics()
    # Flat surface has no defined direction → all cells are outlets;
    # the dataframe length matches the basin count.
    assert len(df) == ws.basin_count


class TestBasinCountCachingAndCentroid:
    """Lazy-property and centroid-always coverage (N5, I4)."""

    def _build_ws(self):
        z = np.array(
            [[5, 5, 5], [5, 1, 5], [5, 5, 5]], dtype=np.float32
        )
        dem, fd = _build(z)
        return fd.basins()

    def test_basin_count_cached_on_second_access(self):
        ws = self._build_ws()
        first = ws.basin_count
        second = ws.basin_count
        assert first == second
        assert ws._basin_count == first

    def test_basin_count_internal_cache_starts_none(self):
        ws = self._build_ws()
        assert ws._basin_count is None
        _ = ws.basin_count
        assert ws._basin_count is not None

    def test_centroid_returned_with_no_inputs_at_all(self):
        ws = self._build_ws()
        df = ws.statistics()
        assert "centroid_x" in df.columns
        assert "centroid_y" in df.columns
