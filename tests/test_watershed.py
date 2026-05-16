"""Tests for ``FlowDirection.watershed`` and ``WatershedRaster`` (P13)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import Point

from digitalrivers import DEM, FlowDirection, WatershedRaster


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def _world_xy(row: int, col: int) -> tuple[float, float]:
    """Cell centre in world coords for cell_size=1, top_left=(0, 0)."""
    return (col + 0.5, -(row + 0.5))


def test_single_pour_point_captures_chain():
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
    # Pour point at (row=1, col=5) — the outlet.
    pts = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(*_world_xy(1, 5))], crs=4326,
    )
    ws = fd.watershed(pts)
    assert type(ws) is WatershedRaster
    assert ws.basin_count == 1
    arr = ws.read_array()
    # Every cell in the chain (row 1) plus their upstream cells (rows 0 and 2)
    # is part of basin 1.
    assert (arr[1, :] == 1).all()


def test_two_pour_points_inner_outer():
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
    # Outer basin at the outlet (1, 5); inner at (1, 3).
    pts = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(*_world_xy(1, 5)), Point(*_world_xy(1, 3))],
        crs=4326,
    )
    ws = fd.watershed(pts, require_unique_basins=False)
    assert ws.basin_count == 2
    arr = ws.read_array()
    # Outer pour point (id=1) labels its cell.
    assert arr[1, 5] == 1
    # Inner pour point (id=2) labels its cell and any upstream cells; basin 2
    # overwrites basin 1 along the shared path.
    assert arr[1, 3] == 2


def test_two_pour_points_unique_basins():
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
    pts = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(*_world_xy(1, 5)), Point(*_world_xy(1, 3))],
        crs=4326,
    )
    ws = fd.watershed(pts, require_unique_basins=True)
    arr = ws.read_array()
    # First-come-first-serve: the outlet wins its cell and the chain back to
    # the inner point's cell. The inner point and its upstream get basin 2.
    assert arr[1, 5] == 1
    # Inner pour-point cell itself is already claimed by basin 1's reverse walk,
    # so basin 2 cannot start. Behaviour is "first seed wins".
    assert arr[1, 3] == 1


def test_outlets_attribute_matches_input_count():
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
    pts = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(*_world_xy(1, 5)), Point(*_world_xy(1, 3))],
        crs=4326,
    )
    ws = fd.watershed(pts)
    assert len(ws.outlets) == 2


def test_to_polygons_emits_one_geometry_per_basin():
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
    pts = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(*_world_xy(1, 5))], crs=4326,
    )
    ws = fd.watershed(pts)
    poly = ws.to_polygons()
    assert len(poly) == 1
    assert poly.iloc[0]["basin_id"] == 1


def test_multi_direction_routing_rejected():
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
    pts = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(*_world_xy(1, 3))], crs=4326,
    )
    with pytest.raises(ValueError, match="single-direction"):
        fd_dinf.watershed(pts)


def test_outside_envelope_point_skipped():
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
    pts = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(*_world_xy(1, 5)), Point(1000.0, -1000.0)],
        crs=4326,
    )
    ws = fd.watershed(pts)
    # Only one basin was created (the outside-envelope point was skipped).
    arr = ws.read_array()
    assert set(np.unique(arr)) - {0} == {1}


class TestWatershedD8ReversedOrder:
    """N7 regression: ``watershed_d8`` non-unique mode (reversed BFS)
    preserves the "later-seed-wins" contract AND keeps the run linear
    on overlapping fan-ins."""

    def test_three_overlapping_seeds_last_seed_wins(self):
        from digitalrivers._flow.watershed import watershed_d8

        fdir = np.array([[6, 6, 6, 6, 6, -1]], dtype=np.int32)
        out = watershed_d8(fdir, [(0, 1), (0, 3), (0, 5)], [1, 2, 3])
        # Forward "last-wins" → seed 3 covers the chain.
        assert int(out[0, 0]) == 3
        assert int(out[0, 5]) == 3

    def test_unique_mode_first_claim_wins(self):
        from digitalrivers._flow.watershed import watershed_d8

        fdir = np.array([[6, 6, 6, 6, 6, -1]], dtype=np.int32)
        out = watershed_d8(
            fdir, [(0, 1), (0, 3), (0, 5)], [1, 2, 3],
            require_unique_basins=True,
        )
        assert int(out[0, 0]) == 1
        assert int(out[0, 1]) == 1

    def test_large_overlapping_run_completes_quickly(self):
        """200-cell chain × 5 overlapping seeds finishes well under 1s."""
        import time

        from digitalrivers._flow.watershed import watershed_d8

        n = 200
        fdir = np.full((1, n), 6, dtype=np.int32)
        fdir[0, -1] = -1
        seeds = [(0, k) for k in (10, 50, 100, 150, n - 1)]
        ids = [1, 2, 3, 4, 5]
        t0 = time.perf_counter()
        out = watershed_d8(fdir, seeds, ids)
        elapsed = time.perf_counter() - t0
        assert int(out[0, 0]) == 5
        assert elapsed < 1.0
