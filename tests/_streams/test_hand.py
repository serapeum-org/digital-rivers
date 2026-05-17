"""Tests for `DEM.hand` (P11)."""
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
    """`hand_d8` must memoise unreachable cells so a long orphan chain
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

    def test_euclidean_hand_zero_at_stream_cells(self):
        """Test euclidean HAND is zero on every stream cell.

        Test scenario:
            With method='euclidean', stream cells must hold 0 (each is its
            own nearest stream cell).
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
        out = dem.hand(sr, method="euclidean")
        arr = out.read_array()
        sm = sr.read_array().astype(bool)
        assert (arr[sm] == 0).all(), "Stream cells must have HAND = 0"

    def test_euclidean_hand_uses_nearest_2d_stream(self):
        """Test euclidean HAND uses 2-D-nearest stream — not flow-path-nearest.

        Test scenario:
            A non-stream cell sitting one cell row above a stream cell
            (without any explicit flow direction tying them) must report
            HAND = elev_self - elev_nearest_stream.
        """
        # 3-row grid; the middle row is the stream chain at elevation 1,
        # row 0 sits at elevation 5 above it.
        z = np.array(
            [
                [5, 5, 5, 5, 5],
                [1, 1, 1, 1, 1],
                [9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            z, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-9999.0,
        )
        dem = DEM(ds.raster)
        sm = np.zeros((3, 5), dtype=bool)
        sm[1, :] = True
        sm_ds = Dataset.create_from_array(
            sm.astype(np.uint8), top_left_corner=(0.0, 0.0), cell_size=1.0,
            epsg=4326, no_data_value=0,
        )
        sr = StreamRaster.from_dataset(sm_ds, threshold=1, routing="d8")
        out = dem.hand(sr, method="euclidean")
        arr = out.read_array()
        # Row 0 cells are 4 m above the nearest stream cell directly below.
        assert (arr[0, :] == 4.0).all(), f"Got {arr[0, :].tolist()}"

    def test_euclidean_hand_warns_and_drops_streams_on_nodata(self):
        """L2 regression: stream cells on DEM no-data are dropped with a UserWarning.

        Test scenario:
            One of the two stream cells sits on a no-data DEM position. The
            method must (a) emit a UserWarning and (b) still produce a valid
            HAND grid by dropping the bad stream cell — not silently
            corrupt the output.
        """
        import warnings
        z = np.array(
            [
                [10.0, 10.0, np.nan, 10.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            z, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-9999.0,
        )
        dem = DEM(ds.raster)
        # Stream cells at (0, 0) — valid — and (0, 2) — on no-data.
        sm = np.zeros((2, 4), dtype=bool)
        sm[0, 0] = True
        sm[0, 2] = True
        sm_ds = Dataset.create_from_array(
            sm.astype(np.uint8), top_left_corner=(0.0, 0.0), cell_size=1.0,
            epsg=4326, no_data_value=0,
        )
        sr = StreamRaster.from_dataset(sm_ds, threshold=1, routing="d8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = dem.hand(sr, method="euclidean")
            arr = out.read_array()
        assert any(issubclass(w.category, UserWarning) for w in caught), (
            f"Expected a UserWarning; got: {[str(w.message) for w in caught]}"
        )
        # Non-stream, non-nodata cells must have finite HAND (no NaN
        # propagation from the bad stream cell).
        no_val = float(dem.no_data_value[0])
        finite = arr[(arr != no_val) & np.isfinite(arr)]
        assert finite.size > 0

    def test_euclidean_hand_no_stream_raises(self):
        """Test euclidean HAND raises when the stream raster has no stream cells.

        Test scenario:
            With no streams, HAND is undefined; the API must raise rather
            than emit zeros or NaN silently.
        """
        z = np.array([[5, 5], [5, 5]], dtype=np.float32)
        ds = Dataset.create_from_array(
            z, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-9999.0,
        )
        dem = DEM(ds.raster)
        sm = np.zeros((2, 2), dtype=bool)
        sm_ds = Dataset.create_from_array(
            sm.astype(np.uint8), top_left_corner=(0.0, 0.0), cell_size=1.0,
            epsg=4326, no_data_value=0,
        )
        sr = StreamRaster.from_dataset(sm_ds, threshold=1, routing="d8")
        with pytest.raises(ValueError, match="no stream cells"):
            dem.hand(sr, method="euclidean")

    def test_invalid_method_raises(self):
        """Test an unknown method= raises ValueError.

        Test scenario:
            Caller passes method='bogus' — must raise with a clear message.
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
        with pytest.raises(ValueError, match="method must be"):
            dem.hand(sr, fd, method="bogus")

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
