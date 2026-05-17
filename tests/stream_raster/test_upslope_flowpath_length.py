"""Tests for `FlowDirection.upslope_flowpath_length` (W-9)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

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


class TestUpslopeFlowpathLength:
    """Tests for `FlowDirection.upslope_flowpath_length`."""

    def test_single_chain_lengths_increase_along_flow(self):
        """Test lengths strictly increase along an east-flowing single chain.

        Test scenario:
            cell_size=1, single-row chain flowing east. The leftmost cell is
            a source (length 0); each successive cell adds one cardinal step;
            the rightmost cell sits 4 cell widths downstream.
        """
        fdir = np.array([[6, 6, 6, 6, -1]], dtype=np.int32)
        fd_ds = Dataset.create_from_array(
            fdir, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-1,
        )
        from digitalrivers import FlowDirection
        fd = FlowDirection.from_dataset(fd_ds, routing="d8")
        out = fd.upslope_flowpath_length()
        arr = out.read_array()
        assert abs(arr[0, 4] - 4.0) < 1e-5, f"Outlet length {arr[0, 4]}"
        assert arr[0, 0] == 0.0, f"Source must be 0, got {arr[0, 0]}"

    def test_diagonal_steps_use_sqrt_two(self):
        """Test diagonal step contributions are scaled by sqrt(2).

        Test scenario:
            A single SE→ diagonal-only chain. After one SE step the
            downstream cell's length is sqrt(2).
        """
        # 3x3, single diagonal:
        #   (0, 0) → SE → (1, 1) → SE → (2, 2)
        fdir = np.array(
            [[7, -1, -1], [-1, 7, -1], [-1, -1, -1]],
            dtype=np.int32,
        )
        fd_ds = Dataset.create_from_array(
            fdir, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-1,
        )
        from digitalrivers import FlowDirection
        fd = FlowDirection.from_dataset(fd_ds, routing="d8")
        out = fd.upslope_flowpath_length()
        arr = out.read_array()
        sqrt2 = float(2.0 ** 0.5)
        assert abs(arr[1, 1] - sqrt2) < 1e-5, f"Single-diag length {arr[1, 1]}"
        assert abs(arr[2, 2] - 2 * sqrt2) < 1e-5, f"Two-diag length {arr[2, 2]}"

    def test_sources_have_zero_length(self):
        """Test cells with no upstream neighbour hold length 0.

        Test scenario:
            On a constructed fdir grid with no inflow to (0, 0), that cell is
            a source and must hold 0.0.
        """
        fdir = np.array([[6, 6, -1], [-1, -1, -1], [-1, -1, -1]], dtype=np.int32)
        fd_ds = Dataset.create_from_array(
            fdir, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-1,
        )
        from digitalrivers import FlowDirection
        fd = FlowDirection.from_dataset(fd_ds, routing="d8")
        out = fd.upslope_flowpath_length()
        arr = out.read_array()
        assert arr[0, 0] == 0.0, f"Source must be 0, got {arr[0, 0]}"

    def test_dataset_dtype_is_float32(self):
        """Test the returned dataset uses float32 storage.

        Test scenario:
            Output is float32 with no_data_value -9999.0.
        """
        z = np.array(
            [
                [9, 9, 9],
                [9, 5, 4],
                [9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd = dem.flow_direction(method="d8")
        out = fd.upslope_flowpath_length()
        assert out.read_array().dtype == np.float32

    def test_multi_direction_routing_rejected(self):
        """Test multi-direction routing raises ValueError.

        Test scenario:
            upslope_flowpath_length needs a single-direction FlowDirection;
            multi-flow inputs must be rejected.
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
        with pytest.raises(ValueError, match="single-direction"):
            fd_dinf.upslope_flowpath_length()
