"""Tests for `StreamRaster.prune_short` (W-5)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, FlowDirection, StreamRaster


def _stream_raster_from_mask(sm: np.ndarray, cell_size: float = 1.0) -> StreamRaster:
    ds = Dataset.create_from_array(
        sm.astype(np.uint8), top_left_corner=(0.0, 0.0), cell_size=cell_size,
        epsg=4326, no_data_value=0,
    )
    return StreamRaster.from_dataset(ds, threshold=1, routing="d8")


def _fd_from_array(fdir: np.ndarray, cell_size: float = 1.0) -> FlowDirection:
    fdir_ds = Dataset.create_from_array(
        fdir, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-1,
    )
    return FlowDirection.from_dataset(fdir_ds, routing="d8")


class TestStreamRasterPruneShort:
    """Tests for `StreamRaster.prune_short`."""

    def test_returns_typed_stream_raster_preserving_tags(self):
        """Test prune_short returns a StreamRaster with the same routing / threshold.

        Test scenario:
            With a threshold below every link length, no cells are pruned; the
            returned object must still be a `StreamRaster` and carry the same
            `routing` and `threshold` tags.
        """
        sm = np.array([[True, True, True, True]], dtype=bool)
        fd_arr = np.array([[6, 6, 6, -1]], dtype=np.int32)
        sr = _stream_raster_from_mask(sm)
        fd = _fd_from_array(fd_arr)
        out = sr.prune_short(fd, min_length_m=0.0)
        assert type(out) is StreamRaster
        assert out.routing == "d8"
        assert out.threshold == sr.threshold

    def test_short_headwater_link_is_pruned(self):
        """Test a short headwater link below threshold is removed.

        Test scenario:
            A Y-junction where one headwater link is only 1 cell long.
            min_length_m=2 must prune that head; the other (also length 1
            from head to confluence) is also pruned by symmetry.
        """
        # Y-junction: heads at (0, 0) and (0, 2); confluence at (1, 1); trunk
        # at (2, 1) and (3, 1).
        sm = np.zeros((4, 3), dtype=bool)
        sm[0, 0] = sm[0, 2] = True
        sm[1, 1] = sm[2, 1] = sm[3, 1] = True
        fdir = np.array(
            [[7, -1, 1], [-1, 0, -1], [-1, 0, -1], [-1, -1, -1]],
            dtype=np.int32,
        )
        sr = _stream_raster_from_mask(sm)
        fd = _fd_from_array(fdir)
        # Each headwater link spans one diagonal step = sqrt(2) ≈ 1.414.
        out = sr.prune_short(fd, min_length_m=2.0)
        out_arr = out.read_array().astype(bool)
        # Both heads pruned (each is one diagonal step from confluence).
        assert not out_arr[0, 0]
        assert not out_arr[0, 2]
        # Internal trunk cells preserved.
        assert out_arr[1, 1]
        assert out_arr[2, 1]
        assert out_arr[3, 1]

    def test_long_headwater_link_is_preserved(self):
        """Test a headwater link above the threshold survives the prune.

        Test scenario:
            With min_length_m=0.5, even a one-cell-step headwater link is
            longer than the threshold so nothing is pruned.
        """
        sm = np.zeros((4, 3), dtype=bool)
        sm[0, 0] = sm[0, 2] = True
        sm[1, 1] = sm[2, 1] = sm[3, 1] = True
        fdir = np.array(
            [[7, -1, 1], [-1, 0, -1], [-1, 0, -1], [-1, -1, -1]],
            dtype=np.int32,
        )
        sr = _stream_raster_from_mask(sm)
        fd = _fd_from_array(fdir)
        out = sr.prune_short(fd, min_length_m=0.5)
        out_arr = out.read_array().astype(bool)
        assert (out_arr == sm).all(), "Nothing should be pruned"

    def test_internal_links_never_pruned(self):
        """Test internal (between-confluence) links survive even if short.

        Test scenario:
            A network with two heads merging at one confluence, then another
            head joining downstream — the segment between the two confluences
            is internal and must survive even if shorter than the threshold.
        """
        # Layout (5 rows, 5 cols):
        #   row 0: H1 . . . H2     (heads at (0,0), (0,4))
        #   row 1: .  X  X  .  .   (confluence at (1, 2))
        #   row 2: . . X . .       (trunk)
        #   row 3: . . X . H3      (H3 at (3, 4))
        #   row 4: . . X X .       (second confluence at (4, 2))
        sm = np.zeros((5, 5), dtype=bool)
        sm[0, 0] = sm[0, 4] = True       # H1, H2
        sm[1, 1] = sm[1, 2] = True       # first confluence + step
        sm[1, 3] = True                   # H2's path-cell
        sm[2, 2] = sm[3, 2] = True       # internal trunk between confluences
        sm[3, 4] = True                   # H3
        sm[4, 2] = sm[4, 3] = True       # second confluence + step from H3
        # Outlet at (4, 2).
        fdir = np.full((5, 5), -1, dtype=np.int32)
        fdir[0, 0] = 7   # SE into (1, 1)
        fdir[1, 1] = 6   # E into (1, 2)
        fdir[0, 4] = 1   # SW into (1, 3)
        fdir[1, 3] = 2   # W into (1, 2)
        fdir[1, 2] = 0   # S into (2, 2)
        fdir[2, 2] = 0   # S into (3, 2)
        fdir[3, 4] = 1   # SW into (4, 3)
        fdir[4, 3] = 2   # W into (4, 2)
        fdir[3, 2] = 0   # S into (4, 2)
        fdir[4, 2] = -1  # outlet
        sr = _stream_raster_from_mask(sm)
        fd = _fd_from_array(fdir)
        out = sr.prune_short(fd, min_length_m=0.5)
        out_arr = out.read_array().astype(bool)
        # Internal trunk cells (2, 2) and (3, 2) must survive regardless.
        assert out_arr[2, 2]
        assert out_arr[3, 2]
        # Second confluence (4, 2) survives.
        assert out_arr[4, 2]

    def test_negative_threshold_raises(self):
        """Test min_length_m < 0 raises ValueError.

        Test scenario:
            A negative length threshold is meaningless; the API must reject
            it with a clear error.
        """
        sm = np.array([[True, True, True, True]], dtype=bool)
        fdir = np.array([[6, 6, 6, -1]], dtype=np.int32)
        sr = _stream_raster_from_mask(sm)
        fd = _fd_from_array(fdir)
        with pytest.raises(ValueError, match="non-negative"):
            sr.prune_short(fd, min_length_m=-1.0)

    def test_multi_direction_routing_rejected(self):
        """Test multi-direction FlowDirection is rejected.

        Test scenario:
            prune_short requires single-direction routing; passing a dinf
            FlowDirection must raise.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            z, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            no_data_value=-9999.0,
        )
        dem = DEM(ds.raster)
        fd_dinf = dem.flow_direction(method="dinf")
        fd_d8 = dem.flow_direction(method="d8")
        acc = fd_d8.accumulate()
        sr = acc.streams(threshold=1)
        with pytest.raises(ValueError, match="single-direction"):
            sr.prune_short(fd_dinf, min_length_m=2.0)

    def test_shape_mismatch_raises(self):
        """Test mismatched stream / flow-direction shapes raise.

        Test scenario:
            A FlowDirection raster of a different shape must be rejected.
        """
        sm = np.ones((3, 3), dtype=bool)
        fdir_small = np.full((2, 2), -1, dtype=np.int32)
        sr = _stream_raster_from_mask(sm)
        fd = _fd_from_array(fdir_small)
        with pytest.raises(ValueError, match="shape"):
            sr.prune_short(fd, min_length_m=1.0)
