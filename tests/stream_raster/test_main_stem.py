"""Tests for `StreamRaster.main_stem` (W-4)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, StreamRaster


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


def _stream_raster_from_mask(sm: np.ndarray) -> StreamRaster:
    ds = Dataset.create_from_array(
        sm.astype(np.uint8), top_left_corner=(0.0, 0.0), cell_size=1.0,
        epsg=4326, no_data_value=0,
    )
    return StreamRaster.from_dataset(ds, threshold=1, routing="d8")


class TestStreamRasterMainStem:
    """Tests for `StreamRaster.main_stem`."""

    def test_single_chain_entire_path_is_main_stem(self):
        """Test a single-head chain: every stream cell sits on the main stem.

        Test scenario:
            With one head and one outlet, the longest path covers all stream
            cells. `main_stem` must return a mask matching the stream mask.
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
        mask = sr.main_stem(fd)
        sm = sr.read_array().astype(bool)
        assert mask[sm].all(), "All stream cells must be on the main stem"
        assert not mask[~sm].any(), "Non-stream cells must be False"

    def test_y_junction_main_stem_traces_lower_index_head(self):
        """Test Y-junction main stem walks through the lower-linear-index head.

        Test scenario:
            Two equal-length heads at (0, 0) and (0, 2) tie on upstream length.
            The tie-break picks the lower linear index → head (0, 0) is on
            the main stem; head (0, 2) is not.
        """
        sm = np.zeros((4, 3), dtype=bool)
        sm[0, 0] = sm[0, 2] = True
        sm[1, 1] = sm[2, 1] = sm[3, 1] = True
        from digitalrivers import FlowDirection
        fdir = np.array(
            [[7, -1, 1], [-1, 0, -1], [-1, 0, -1], [-1, -1, -1]],
            dtype=np.int32,
        )
        # Wrap the raw mask + fdir into typed objects.
        sr = _stream_raster_from_mask(sm)
        fdir_ds = Dataset.create_from_array(
            fdir, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-1,
        )
        fd = FlowDirection.from_dataset(fdir_ds, routing="d8")
        mask = sr.main_stem(fd)
        assert mask[0, 0], "Lower-index head must be on the main stem"
        assert not mask[0, 2], "Higher-index head must NOT be on the main stem"
        # Trunk cells are on the main stem.
        assert mask[1, 1] and mask[2, 1] and mask[3, 1]

    def test_explicit_outlet_traces_subnetwork(self):
        """Test passing an outlet explicitly traces from that specific cell.

        Test scenario:
            Pass the outlet location directly; the returned mask must contain
            that cell and walk upstream from it.
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
        # Pour-point at the rightmost stream cell.
        outlet_rc = (1, 5)
        mask = sr.main_stem(fd, outlet=outlet_rc)
        assert mask[outlet_rc], "Passed outlet must be on the main stem"

    def test_outlet_outside_grid_raises(self):
        """Test passing an out-of-grid outlet raises ValueError.

        Test scenario:
            outlet=(100, 100) outside the raster bounds must raise
            ValueError with a clear message.
        """
        sm = np.zeros((3, 3), dtype=bool)
        sm[0, :] = True
        from digitalrivers import FlowDirection
        fdir = np.array([[6, 6, -1], [-1, -1, -1], [-1, -1, -1]], dtype=np.int32)
        sr = _stream_raster_from_mask(sm)
        fdir_ds = Dataset.create_from_array(
            fdir, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-1,
        )
        fd = FlowDirection.from_dataset(fdir_ds, routing="d8")
        with pytest.raises(ValueError, match="outside the raster"):
            sr.main_stem(fd, outlet=(100, 100))

    def test_outlet_non_stream_raises(self):
        """Test passing an outlet at a non-stream cell raises ValueError.

        Test scenario:
            outlet pointing at a cell where the stream mask is False must
            raise ValueError.
        """
        sm = np.zeros((3, 3), dtype=bool)
        sm[0, :] = True
        from digitalrivers import FlowDirection
        fdir = np.array([[6, 6, -1], [-1, -1, -1], [-1, -1, -1]], dtype=np.int32)
        sr = _stream_raster_from_mask(sm)
        fdir_ds = Dataset.create_from_array(
            fdir, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-1,
        )
        fd = FlowDirection.from_dataset(fdir_ds, routing="d8")
        with pytest.raises(ValueError, match="not a stream cell"):
            sr.main_stem(fd, outlet=(1, 1))

    def test_multi_direction_routing_rejected(self):
        """Test multi-direction routing input raises ValueError.

        Test scenario:
            main_stem requires a single-direction FlowDirection; passing
            a dinf-routed one must be rejected.
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
            sr.main_stem(fd_dinf)

    def test_shape_mismatch_raises(self):
        """Test mismatched stream / flow-direction shapes raise ValueError.

        Test scenario:
            A FlowDirection raster of a different shape must be rejected.
        """
        z_big = np.zeros((4, 4), dtype=np.float32)
        z_small = np.zeros((2, 2), dtype=np.float32)
        dem_big = _make_dem(z_big)
        dem_small = _make_dem(z_small)
        fd_small = dem_small.flow_direction(method="d8")
        acc_big = dem_big.flow_direction(method="d8").accumulate()
        sr_big = acc_big.streams(threshold=1)
        with pytest.raises(ValueError, match="shape"):
            sr_big.main_stem(fd_small)

    def test_empty_stream_returns_zero_mask(self):
        """Test an empty stream raster returns an all-False mask without error.

        Test scenario:
            With no stream cells the helper returns an all-False mask of the
            input shape.
        """
        from digitalrivers import FlowDirection
        sm = np.zeros((2, 3), dtype=bool)
        fdir = np.full((2, 3), -1, dtype=np.int32)
        sr = _stream_raster_from_mask(sm)
        fdir_ds = Dataset.create_from_array(
            fdir, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-1,
        )
        fd = FlowDirection.from_dataset(fdir_ds, routing="d8")
        mask = sr.main_stem(fd)
        assert mask.shape == sm.shape
        assert not mask.any()
