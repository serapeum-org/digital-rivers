"""Tests for ``DEM.hand`` (P11)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, FlowDirection, StreamRaster
from digitalrivers._streams.hand import hand_d8


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


# --- Orphan-chain memoisation (I5 fix) -------------------------------------


class TestHandOrphanMemoisation:
    """``hand_d8`` must memoise unreachable cells so a long orphan chain
    does not pessimise to O(N²) (I5 fix)."""

    def test_orphan_chain_all_nan(self):
        """A long downhill chain that never reaches a stream stays NaN."""
        n = 50
        elev = np.arange(n, dtype=np.float64).reshape(1, n)[::-1]
        # Single-row chain pointing east (6) all the way; last cell is a sink.
        fdir = np.full((1, n), 6, dtype=np.int32)
        fdir[0, -1] = -1
        # No stream cells at all → every cell is an orphan.
        stream_mask = np.zeros((1, n), dtype=bool)
        out = hand_d8(elev, fdir, stream_mask)
        assert np.isnan(out).all()

    def test_orphan_chain_completes_quickly(self):
        """200-cell orphan chain finishes well under one second; the
        memoisation prevents the O(L²) re-walk."""
        import time

        n = 200
        elev = np.arange(n, dtype=np.float64).reshape(1, n)[::-1]
        fdir = np.full((1, n), 6, dtype=np.int32)
        fdir[0, -1] = -1
        stream_mask = np.zeros((1, n), dtype=bool)
        t0 = time.perf_counter()
        out = hand_d8(elev, fdir, stream_mask)
        elapsed = time.perf_counter() - t0
        assert np.isnan(out).all()
        assert elapsed < 1.0, (
            f"hand_d8 on 200-cell orphan chain took {elapsed:.3f}s — "
            f"unreachable-memoisation may be broken"
        )

    def test_mixed_reachable_and_orphan_paths(self):
        """A two-row grid where one row reaches a stream and the other
        doesn't: reachable cells get finite HAND, orphans stay NaN."""
        elev = np.array(
            [
                [4.0, 3.0, 2.0, 1.0],  # reaches the stream at (0, 3)
                [9.0, 8.0, 7.0, 6.0],  # orphan: walks east into a sink
            ],
            dtype=np.float64,
        )
        # Row 0 walks east (6) into the stream at col 3.
        # Row 1 walks east (6) but the rightmost cell is a sink (-1).
        fdir = np.array(
            [[6, 6, 6, -1], [6, 6, 6, -1]], dtype=np.int32
        )
        stream_mask = np.zeros((2, 4), dtype=bool)
        stream_mask[0, 3] = True
        out = hand_d8(elev, fdir, stream_mask)
        # Row 0: reaches the stream, every value finite.
        assert np.isfinite(out[0]).all()
        assert float(out[0, 0]) == pytest.approx(3.0)
        # Row 1: orphan — all NaN.
        assert np.isnan(out[1]).all()
