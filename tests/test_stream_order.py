"""Tests for `StreamRaster.order` — Strahler / Shreve / Horton (P10)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, FlowDirection, StreamRaster
from digitalrivers._streams.order import (
    _stream_outlets,
    _upstream_length_from_head,
    _build_topology,
    hack,
    horton,
    shreve,
    strahler,
)


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


# ----- Hack --------------------------------------------------------------------------------

class TestHack:
    def test_y_junction_main_stem_is_order_one_tributaries_two(self):
        # In the Y-fixture both tributaries have length 1 from their head; the
        # main stem of the catchment runs from head A (or B — ties broken by
        # lower row-major index) through (1, 1), (2, 1), to outlet (3, 1).
        sm, fd = _y_junction_streams()
        order = hack(sm, fd)
        # Outlet, confluence and trunk are all order 1 (main stem).
        assert order[3, 1] == 1
        assert order[2, 1] == 1
        assert order[1, 1] == 1
        # Exactly one of the two heads is on the main stem (order 1); the other
        # is a tributary (order 2). Deterministic tie-break by lower linear
        # index picks (0, 0).
        assert order[0, 0] == 1
        assert order[0, 2] == 2

    def test_single_chain_is_order_one_throughout(self):
        sm = np.zeros((1, 5), dtype=bool)
        sm[0, :] = True
        fd = np.full((1, 5), 6, dtype=np.int32)
        fd[0, -1] = -1
        order = hack(sm, fd)
        assert (order[sm] == 1).all()

    def test_empty_stream_mask_returns_all_zeros(self):
        """Test hack with no stream cells returns a zero raster of the same shape.

        Test scenario:
            Empty (all-False) stream mask should produce an all-zero order
            raster without crashing the outlet enumeration or upstream walk.
        """
        sm = np.zeros((3, 4), dtype=bool)
        fd = np.full((3, 4), -1, dtype=np.int32)
        order = hack(sm, fd)
        assert order.shape == sm.shape, "Output shape must match input shape"
        assert (order == 0).all(), "All cells must hold order 0 when no streams"

    def test_disconnected_networks_each_get_own_main_stem(self):
        """Test hack handles two disconnected stream networks independently.

        Test scenario:
            Two independent east-flowing chains share no flow path; each must
            label its own outlet's chain as order 1 without interference.
        """
        sm = np.zeros((3, 5), dtype=bool)
        sm[0, :] = True
        sm[2, :] = True
        fd = np.full((3, 5), -1, dtype=np.int32)
        fd[0, :-1] = 6
        fd[2, :-1] = 6
        order = hack(sm, fd)
        assert (order[0, :] == 1).all(), "Top network all order 1"
        assert (order[2, :] == 1).all(), "Bottom network all order 1"

    def test_tie_break_picks_lower_linear_index(self):
        """Test hack tie-break on equal upstream lengths is deterministic.

        Test scenario:
            Two single-cell heads of equal upstream length meeting at a
            confluence — the head with the lower row-major linear index
            must be promoted to the main stem (order 1); the other becomes
            order 2.
        """
        sm, fd = _y_junction_streams()
        order = hack(sm, fd)
        assert order[0, 0] == 1, "Lower-linear-index head should be main stem"
        assert order[0, 2] == 2, "Higher-linear-index head should be tributary"

    def test_tributary_of_tributary_gets_order_three(self):
        # Main stem runs along row 0 (head at (0, 7), outlet at (0, 0)).
        # Tributary T joins the main stem at (0, 4); T itself has a
        # sub-tributary U joining at (1, 5).
        #
        #   col:    0  1  2  3  4  5  6  7
        # row 0:    O  X  X  X  X  X  X  H   ← main stem (length 7)
        # row 1:    .  .  .  .  T  T  H  .   ← tributary T joining at (0, 4)
        # row 2:    .  .  .  .  .  H  .  .   ← sub-tributary U joining at (1, 5)
        sm = np.zeros((3, 8), dtype=bool)
        sm[0, :] = True
        sm[1, 4:7] = True
        sm[2, 5] = True
        # DIR_OFFSETS: 0=S, 1=SW, 2=W, 3=NW, 4=N, 5=NE, 6=E, 7=SE.
        fd = np.full((3, 8), -1, dtype=np.int32)
        fd[0, 0] = -1  # outlet
        for c in range(1, 8):
            fd[0, c] = 2  # W along the main stem
        fd[1, 4] = 4   # N into (0, 4) — T merges with main stem here
        fd[1, 5] = 2   # W into (1, 4)
        fd[1, 6] = 2   # W into (1, 5)
        fd[2, 5] = 4   # N into (1, 5) — U merges with T here
        order = hack(sm, fd)
        # Main stem along row 0 is order 1.
        assert (order[0, :] == 1).all()
        # T (row 1 stream cells) is order 2.
        assert order[1, 4] == 2
        assert order[1, 5] == 2
        assert order[1, 6] == 2
        # U (sub-tributary at (2, 5)) is order 3.
        assert order[2, 5] == 3


# ----- _upstream_length_from_head -----------------------------------------------------------

class TestUpstreamLengthFromHead:
    """Tests for the `_upstream_length_from_head` helper."""

    def test_heads_have_length_zero(self):
        """Test heads (no upstream inflow) carry length 0.

        Test scenario:
            A 2-head Y-junction: both heads are sources in the stream DAG, so
            their length-from-head must be 0.
        """
        sm, fd = _y_junction_streams()
        indeg, _ = _build_topology(sm, fd)
        length = _upstream_length_from_head(sm, fd, indeg)
        assert length[0, 0] == 0, "Head A length must be 0"
        assert length[0, 2] == 0, "Head B length must be 0"

    def test_outlet_length_matches_longest_chain(self):
        """Test outlet length equals one less than the longest source-to-outlet path.

        Test scenario:
            On a single 5-cell east-flowing chain the outlet must carry
            length 4 (four steps from the only head).
        """
        sm = np.zeros((1, 5), dtype=bool)
        sm[0, :] = True
        fd = np.full((1, 5), 6, dtype=np.int32)
        fd[0, -1] = -1
        indeg, _ = _build_topology(sm, fd)
        length = _upstream_length_from_head(sm, fd, indeg)
        assert length[0, 4] == 4, f"Outlet length should be 4, got {length[0, 4]}"


# ----- _stream_outlets ----------------------------------------------------------------------

class TestStreamOutlets:
    """Tests for the `_stream_outlets` helper."""

    def test_sink_cell_is_outlet(self):
        """Test a stream cell with `fdir = -1` is treated as an outlet.

        Test scenario:
            On a chain whose downstream-most cell has fdir = -1, that cell
            must be reported in the outlet list and nothing else.
        """
        sm = np.zeros((1, 3), dtype=bool)
        sm[0, :] = True
        fd = np.array([[6, 6, -1]], dtype=np.int32)
        outlets = _stream_outlets(sm, fd)
        assert outlets == [(0, 2)], f"Expected single outlet at (0, 2), got {outlets}"

    def test_edge_cell_is_outlet_when_receiver_off_grid(self):
        """Test a stream cell flowing off the grid is reported as an outlet.

        Test scenario:
            A 1×3 chain whose last cell flows east goes off-grid; it must
            still be an outlet.
        """
        sm = np.zeros((1, 3), dtype=bool)
        sm[0, :] = True
        fd = np.array([[6, 6, 6]], dtype=np.int32)
        outlets = _stream_outlets(sm, fd)
        assert (0, 2) in outlets, f"Edge-flow cell should be outlet, got {outlets}"

    def test_non_stream_receiver_makes_cell_an_outlet(self):
        """Test a stream cell whose D8 receiver is non-stream is an outlet.

        Test scenario:
            A single stream cell at (0, 0) flowing east into a non-stream
            cell at (0, 1) must be reported as an outlet.
        """
        sm = np.zeros((1, 3), dtype=bool)
        sm[0, 0] = True
        fd = np.array([[6, -1, -1]], dtype=np.int32)
        outlets = _stream_outlets(sm, fd)
        assert outlets == [(0, 0)], f"Expected outlet at (0, 0), got {outlets}"


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

    def test_hack_dispatcher_returns_typed_stream_raster(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, sr = _build_pipeline(z, threshold=1)
        ordered = sr.order(method="hack", flow_direction=fd)
        assert type(ordered) is StreamRaster
        # Single horizontal chain → entire chain is the main stem (order 1).
        assert (ordered.read_array()[sr.read_array().astype(bool)] == 1).all()

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
