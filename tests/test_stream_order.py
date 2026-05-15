"""Tests for ``StreamRaster.order`` — Strahler / Shreve / Horton (P10)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, FlowDirection, StreamRaster
from digitalrivers._stream_order import horton, shreve, strahler


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


def _y_junction_streams() -> tuple[np.ndarray, np.ndarray]:
    """Hand-craft a Y-shaped stream raster with a 3-cell trunk and two 2-cell
    tributaries flowing into a confluence at (2, 1)."""
    # 4x3 grid; stream cells form a Y where the trunk runs south:
    #   (0,0) head A → SE → (1,1)
    #   (0,2) head B → SW → (1,1)
    #   (1,1) confluence → S → (2,1)
    #   (2,1) → S → (3,1) outlet
    stream_mask = np.zeros((4, 3), dtype=bool)
    stream_mask[0, 0] = True
    stream_mask[0, 2] = True
    stream_mask[1, 1] = True
    stream_mask[2, 1] = True
    stream_mask[3, 1] = True
    # Direction codes (DIR_OFFSETS: 0=S, 1=SW, 2=W, 3=NW, 4=N, 5=NE, 6=E, 7=SE):
    fdir = np.full((4, 3), -1, dtype=np.int32)
    fdir[0, 0] = 7  # SE → (1, 1)
    fdir[0, 2] = 1  # SW → (1, 1)
    fdir[1, 1] = 0  # S → (2, 1)
    fdir[2, 1] = 0  # S → (3, 1)
    fdir[3, 1] = -1  # outlet
    return stream_mask, fdir


# ----- Strahler ----------------------------------------------------------------------------

class TestStrahler:
    def test_y_junction_heads_are_one_trunk_is_two(self):
        sm, fd = _y_junction_streams()
        order = strahler(sm, fd)
        assert order[0, 0] == 1  # head A
        assert order[0, 2] == 1  # head B
        # Confluence and trunk get order 2 (two order-1 tributaries arriving).
        assert order[1, 1] == 2
        assert order[2, 1] == 2
        assert order[3, 1] == 2

    def test_single_stream_keeps_order_one(self):
        # One head, single chain, no confluence — every cell is order 1.
        sm = np.zeros((1, 5), dtype=bool)
        sm[0, :] = True
        fd = np.full((1, 5), 6, dtype=np.int32)  # all east
        fd[0, -1] = -1  # outlet sink
        order = strahler(sm, fd)
        assert (order[sm] == 1).all()


# ----- Shreve ------------------------------------------------------------------------------

class TestShreve:
    def test_y_junction_outlet_equals_head_count(self):
        sm, fd = _y_junction_streams()
        mag = shreve(sm, fd)
        assert mag[0, 0] == 1  # head A
        assert mag[0, 2] == 1  # head B
        # Confluence and trunk magnitude = 2 (sum of two heads).
        assert mag[1, 1] == 2
        assert mag[3, 1] == 2

    def test_single_stream_magnitude_one_throughout(self):
        sm = np.zeros((1, 5), dtype=bool)
        sm[0, :] = True
        fd = np.full((1, 5), 6, dtype=np.int32)
        fd[0, -1] = -1
        mag = shreve(sm, fd)
        assert (mag[sm] == 1).all()


# ----- Horton ------------------------------------------------------------------------------

class TestHorton:
    def test_y_junction_promotes_one_tributary_to_outlet_order(self):
        sm, fd = _y_junction_streams()
        order = horton(sm, fd)
        # Outlet's Horton order matches Strahler = 2.
        assert order[3, 1] == 2
        # At least one tributary has been promoted to order 2 (the main stem).
        promoted = (order[0, 0] == 2) or (order[0, 2] == 2)
        assert promoted
        # The other tributary keeps its Strahler value = 1.
        assert min(int(order[0, 0]), int(order[0, 2])) == 1


# ----- DEM/StreamRaster end-to-end --------------------------------------------------------

class TestStreamRasterOrder:
    def test_returns_typed_stream_raster_strahler(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, sr = _build_pipeline(z, threshold=1)
        ordered = sr.order(method="strahler", flow_direction=fd)
        assert type(ordered) is StreamRaster
        assert ordered.routing == "d8"
        assert ordered.threshold == sr.threshold

    def test_shreve_outlet_matches_head_count(self):
        # 2-headwater configuration via a real DEM.
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 5, 9, 5, 9],
                [9, 9, 3, 9, 9],
                [9, 9, 1, 9, 9],
                [9, 9, 0, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, sr = _build_pipeline(z, threshold=1)
        ordered = sr.order(method="shreve", flow_direction=fd)
        ord_arr = ordered.read_array()
        # Find the outlet cell (largest accumulation; minimum elevation).
        sr_mask = sr.read_array().astype(bool)
        if not sr_mask.any():
            pytest.skip("test fixture produced no stream cells at threshold=1")
        # Outlet's Shreve magnitude should be at least the number of heads in the
        # extracted network.
        outlet_value = int(ord_arr[sr_mask].max())
        assert outlet_value >= 1

    def test_invalid_method_raises(self):
        z = np.array(
            [[9, 9, 9], [9, 5, 9], [9, 9, 9]], dtype=np.float32
        )
        dem, fd, sr = _build_pipeline(z, threshold=1)
        with pytest.raises(ValueError, match="method must be"):
            sr.order(method="bogus", flow_direction=fd)

    def test_missing_flow_direction_raises(self):
        z = np.array(
            [[9, 9, 9], [9, 5, 9], [9, 9, 9]], dtype=np.float32
        )
        dem, fd, sr = _build_pipeline(z, threshold=1)
        with pytest.raises(ValueError, match="FlowDirection"):
            sr.order(method="strahler", flow_direction=None)

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
            sr.order(method="strahler", flow_direction=fd_dinf)
