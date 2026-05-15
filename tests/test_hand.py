"""Tests for ``DEM.hand`` (P11)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, FlowDirection, StreamRaster
from digitalrivers._hand import hand_d8


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


# ----- Kernel-level -----------------------------------------------------------------------

class TestHandD8:
    def test_stream_cells_are_zero(self):
        # Simple chain: all stream cells. HAND should be 0 everywhere.
        elev = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float64)
        fdir = np.array([[6, 6, 6, 6, -1]], dtype=np.int32)
        sm = np.ones(elev.shape, dtype=bool)
        hand = hand_d8(elev, fdir, sm)
        np.testing.assert_array_equal(hand, np.zeros_like(elev))

    def test_hand_equals_drop_to_drain(self):
        # Two-row strip: top row drains south into the bottom row (stream).
        # elev top row = [10, 8], bottom row = [2, 1].
        elev = np.array(
            [
                [10.0, 8.0],
                [2.0, 1.0],
            ],
            dtype=np.float64,
        )
        fdir = np.array(
            [
                [0, 0],  # both top cells flow south
                [-1, -1],  # bottom row are stream cells (no flow needed)
            ],
            dtype=np.int32,
        )
        sm = np.array([[False, False], [True, True]], dtype=bool)
        hand = hand_d8(elev, fdir, sm)
        # HAND[0, 0] = elev[0, 0] - elev[1, 0] = 10 - 2 = 8.
        # HAND[0, 1] = elev[0, 1] - elev[1, 1] = 8 - 1 = 7.
        # Stream cells: 0.
        assert hand[0, 0] == pytest.approx(8.0)
        assert hand[0, 1] == pytest.approx(7.0)
        assert hand[1, 0] == 0.0
        assert hand[1, 1] == 0.0

    def test_orphan_cell_is_nan(self):
        # Cell that flows to nowhere; no stream downstream.
        elev = np.array([[5.0]], dtype=np.float64)
        fdir = np.array([[-1]], dtype=np.int32)
        sm = np.zeros(elev.shape, dtype=bool)
        hand = hand_d8(elev, fdir, sm)
        assert np.isnan(hand[0, 0])

    def test_no_data_cells_stay_nan(self):
        elev = np.array(
            [
                [np.nan, 3.0],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        )
        fdir = np.array(
            [
                [-1, 1],
                [-1, -1],
            ],
            dtype=np.int32,
        )
        sm = np.array([[False, False], [True, False]], dtype=bool)
        hand = hand_d8(elev, fdir, sm)
        assert np.isnan(hand[0, 0])


# ----- DEM.hand end-to-end ---------------------------------------------------------------

class TestDEMHand:
    def test_returns_dataset(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, sr = _build_pipeline(z, threshold=1)
        out = dem.hand(sr, fd)
        assert isinstance(out, Dataset)
        assert out.shape == dem.shape

    def test_stream_cells_have_zero_hand(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, sr = _build_pipeline(z, threshold=1)
        out = dem.hand(sr, fd)
        out_arr = out.read_array()
        sr_mask = sr.read_array().astype(bool)
        # Every stream cell has HAND == 0 (within float tolerance).
        np.testing.assert_allclose(out_arr[sr_mask], 0.0, atol=1e-4)

    def test_hand_non_negative_in_catchment(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, sr = _build_pipeline(z, threshold=1)
        out = dem.hand(sr, fd)
        out_arr = out.read_array()
        no_val = out.no_data_value[0]
        valid = out_arr != no_val
        assert np.all(out_arr[valid] >= -1e-4)

    def test_multi_direction_routing_rejected(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd_d8, sr = _build_pipeline(z, threshold=1)
        fd_dinf = dem.flow_direction(method="dinf")
        with pytest.raises(ValueError, match="single-direction"):
            dem.hand(sr, fd_dinf)

    def test_shape_mismatch_rejected(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, sr = _build_pipeline(z, threshold=1)
        # Build a smaller dem for size mismatch.
        small = _make_dem(np.zeros((2, 2), dtype=np.float32))
        fd_small = small.flow_direction(method="d8")
        with pytest.raises(ValueError, match="Shape mismatch"):
            dem.hand(sr, fd_small)
