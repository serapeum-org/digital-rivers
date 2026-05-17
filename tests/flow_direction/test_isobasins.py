"""Tests for `FlowDirection.isobasins` (W-7)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, WatershedRaster


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def _build_pipeline(z: np.ndarray, threshold: int, cell_size: float = 1.0):
    dem = _make_dem(z, cell_size=cell_size)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=threshold)
    return dem, fd, acc, sr


class TestFlowDirectionIsobasins:
    """Tests for `FlowDirection.isobasins`."""

    def test_returns_typed_watershed_raster(self):
        """Test isobasins returns a `WatershedRaster` with the expected routing.

        Test scenario:
            A small east-flowing chain with target_area_km2 large enough to
            produce a single basin should still return a typed WatershedRaster.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc, sr = _build_pipeline(z, threshold=1)
        # cell_size=1.0 deg → cell_area_km2 ≈ tiny; pick a huge target so we
        # fall back to a single basin at the outlet.
        ws = fd.isobasins(sr, acc, target_area_km2=1e9)
        assert type(ws) is WatershedRaster
        assert ws.routing == "d8"
        assert ws.basin_count <= 1

    def test_small_target_produces_multiple_basins(self):
        """Test a small target area yields multiple sub-basins.

        Test scenario:
            With a small target area (one cell), every stream cell becomes
            its own sub-basin seed.
        """
        # Use a single straight stream chain. cell_size=1 → cell area is
        # roughly the cell-area-in-degrees-squared / 1e6 km² which is tiny
        # (since EPSG 4326 is degrees). Use a 30-m equivalent by setting
        # cell_size=30 m via target / cell_area calc directly.
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1, 0],
                [9, 9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc, sr = _build_pipeline(z, threshold=1)
        # target so small that every stream cell is its own seed bucket.
        # cell_area_km2 for a 4326 1-degree cell is ~12000 km² — too big to
        # split. We get the target_cells = max(1, round(target_area / a)) =
        # 1 when target equals cell_area, so target equals a → every cell.
        # Approximating: pick target equal to the area of one cell. Compute it.
        gt = fd.geotransform
        cell_area_km2 = abs(gt[1] * gt[5]) / 1e6
        ws = fd.isobasins(sr, acc, target_area_km2=cell_area_km2)
        # At least 2 sub-basins for a 6-cell chain.
        assert ws.basin_count >= 2

    def test_outlets_dict_has_one_entry_per_basin(self):
        """Test the returned outlets dict has one entry per basin.

        Test scenario:
            Whatever the number of seeds placed, the resulting outlets dict
            maps basin_id -> (row, col) for each label.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1, 0],
                [9, 9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc, sr = _build_pipeline(z, threshold=1)
        gt = fd.geotransform
        cell_area_km2 = abs(gt[1] * gt[5]) / 1e6
        ws = fd.isobasins(sr, acc, target_area_km2=cell_area_km2 * 2)
        assert len(ws.outlets) == ws.basin_count

    def test_non_positive_target_raises(self):
        """Test target_area_km2 <= 0 raises ValueError.

        Test scenario:
            Zero or negative target areas are not meaningful; reject them.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc, sr = _build_pipeline(z, threshold=1)
        with pytest.raises(ValueError, match="positive"):
            fd.isobasins(sr, acc, target_area_km2=0.0)
        with pytest.raises(ValueError, match="positive"):
            fd.isobasins(sr, acc, target_area_km2=-1.0)

    def test_multi_direction_routing_rejected(self):
        """Test multi-direction routing input raises ValueError.

        Test scenario:
            isobasins requires single-direction routing; passing a
            multi-flow FlowDirection raises.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd_dinf = dem.flow_direction(method="dinf")
        fd_d8 = dem.flow_direction(method="d8")
        acc = fd_d8.accumulate()
        sr = acc.streams(threshold=1)
        with pytest.raises(ValueError, match="single-direction"):
            fd_dinf.isobasins(sr, acc, target_area_km2=1e-3)

    def test_shape_mismatch_raises(self):
        """Test mismatched shapes raise ValueError.

        Test scenario:
            Stream raster with different shape from flow direction must
            be rejected.
        """
        z_big = np.zeros((4, 4), dtype=np.float32)
        z_small = np.zeros((2, 2), dtype=np.float32)
        dem_big = _make_dem(z_big)
        dem_small = _make_dem(z_small)
        fd_big = dem_big.flow_direction(method="d8")
        acc_big = fd_big.accumulate()
        acc_small = dem_small.flow_direction(method="d8").accumulate()
        sr_small = acc_small.streams(threshold=1)
        with pytest.raises(ValueError, match="shape mismatch"):
            fd_big.isobasins(sr_small, acc_big, target_area_km2=1.0)
