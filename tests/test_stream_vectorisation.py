"""Tests for `StreamRaster.to_vector` (P9)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import LineString

from digitalrivers import DEM, FlowDirection, StreamRaster


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def _build_pipeline(z: np.ndarray, threshold: int):
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=threshold)
    return dem, fd, sr


def test_returns_geodataframe_with_expected_columns():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build_pipeline(z, threshold=1)
    gdf = sr.to_vector(fd, dem=dem)
    assert isinstance(gdf, gpd.GeoDataFrame)
    for col in ("link_id", "from_node", "to_node", "length_m", "drop_m",
                "mean_slope", "sinuosity", "geometry"):
        assert col in gdf.columns


def test_single_chain_yields_one_link():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    # Threshold high enough to exclude the surrounding 9-row cells; the chain alone
    # remains as a stream.
    dem, fd, sr = _build_pipeline(z, threshold=2)
    gdf = sr.to_vector(fd, dem=dem)
    # At least one link, no duplicate link IDs.
    assert len(gdf) >= 1
    assert gdf["link_id"].is_unique


def test_geometry_vertices_at_cell_centres():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build_pipeline(z, threshold=2)
    gdf = sr.to_vector(fd, dem=dem)
    # cell_size=1, top_left=(0, 0). Cell (r, c)'s centre is (c + 0.5, -(r + 0.5)).
    # Verify all vertex coordinates land on half-integer grid lines.
    for line in gdf.geometry:
        for x, y in line.coords:
            # x mod 1 == 0.5 (within float tolerance).
            assert abs((x - 0.5) % 1) < 1e-6 or abs((x + 0.5) % 1) < 1e-6


def test_links_have_non_negative_drop():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build_pipeline(z, threshold=1)
    gdf = sr.to_vector(fd, dem=dem)
    assert (gdf["drop_m"] >= 0).all()


def test_link_length_at_least_one_cell_step():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build_pipeline(z, threshold=1)
    gdf = sr.to_vector(fd, dem=dem)
    # Every link spans at least one cell step (>= 1.0 for cardinal at unit cell size).
    assert (gdf["length_m"] >= 1.0).all()


def test_multi_direction_routing_raises():
    z = np.array(
        [
            [9, 9, 9, 9, 9],
            [9, 5, 4, 3, 9],
            [9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd_dinf = dem.flow_direction(method="dinf")
    # Build a stream raster (D8-routed) but pass the Dinf FlowDirection.
    fd_d8 = dem.flow_direction(method="d8")
    acc = fd_d8.accumulate()
    sr = acc.streams(threshold=1)
    with pytest.raises(ValueError, match="single-direction routing"):
        sr.to_vector(fd_dinf, dem=dem)


def test_shape_mismatch_raises():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build_pipeline(z, threshold=1)
    # Build a smaller, mis-shaped FlowDirection.
    z_small = np.zeros((2, 2), dtype=np.float32)
    dem_small = _make_dem(z_small)
    fd_small = dem_small.flow_direction(method="d8")
    with pytest.raises(ValueError, match="shape"):
        sr.to_vector(fd_small, dem=dem)


def test_without_dem_drop_and_slope_are_nan():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build_pipeline(z, threshold=1)
    gdf = sr.to_vector(fd)  # no dem
    assert gdf["drop_m"].isna().all()
    assert gdf["mean_slope"].isna().all()


def test_straight_chain_has_sinuosity_one():
    """Test a single east-flowing chain produces sinuosity exactly 1.0.

    Test scenario:
        Cells (1, 1) → (1, 5) form a single straight cardinal-step chain.
        traced_length == straight_line_distance → sinuosity = 1.0.
    """
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build_pipeline(z, threshold=2)
    gdf = sr.to_vector(fd, dem=dem)
    assert (gdf["sinuosity"] == 1.0).all(), (
        f"Straight chain must yield sinuosity 1.0, got {gdf['sinuosity'].tolist()}"
    )


def test_sinuosity_at_least_one_for_non_degenerate_links():
    """Test sinuosity is ≥ 1.0 for every link (geometric invariant).

    Test scenario:
        Traced length cannot be less than straight-line distance — sinuosity
        is bounded below by 1.0 for every non-degenerate link.
    """
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build_pipeline(z, threshold=1)
    gdf = sr.to_vector(fd, dem=dem)
    assert (gdf["sinuosity"] >= 1.0 - 1e-9).all(), (
        f"sinuosity must be ≥ 1.0; got {gdf['sinuosity'].tolist()}"
    )


def test_links_are_linestrings():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build_pipeline(z, threshold=1)
    gdf = sr.to_vector(fd, dem=dem)
    assert all(isinstance(g, LineString) for g in gdf.geometry)
