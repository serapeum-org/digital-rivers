"""Comprehensive tests for DEM.accumulate_flow and DEM.flow_accumulation.

Tests cover the iterative stack-based DFS that counts upstream cells
draining into a given cell using D8 flow direction.  Includes unit
tests for every edge case and end-to-end integration tests with the
Coello dataset.
"""
import numpy as np
import pytest
from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers.dem import DEM, DIR_OFFSETS


@pytest.fixture()
def make_dem():
    """Factory fixture that creates a DEM from a 2-D elevation array.

    Returns:
        Callable that accepts a numpy array and returns a DEM instance.
    """

    def _make(elev: np.ndarray) -> DEM:
        ds = Dataset.create_from_array(
            elev.astype(np.float32),
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999,
        )
        return DEM(ds.raster)

    return _make


class TestAccumulateFlowOutOfBounds:
    """Verify behaviour when the target cell is outside the grid."""

    def test_negative_row(self, make_dem):
        """Out-of-bounds row (negative) returns 0 without modifying acc.

        Test scenario:
            r=-1 is outside the grid; the method should short-circuit
            and return 0.
        """
        dem = make_dem(np.array([[100.0]]))
        flow_dir = np.array([[0]], dtype=np.int32)
        acc = np.full_like(flow_dir, -1)
        result = dem.accumulate_flow(-1, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 0, f"Expected 0 for out-of-bounds row, got {result}"
        assert acc[0, 0] == -1, "acc should not be modified for out-of-bounds call"

    def test_negative_col(self, make_dem):
        """Out-of-bounds column (negative) returns 0.

        Test scenario:
            c=-1 is outside the grid.
        """
        dem = make_dem(np.array([[100.0]]))
        flow_dir = np.array([[0]], dtype=np.int32)
        acc = np.full_like(flow_dir, -1)
        result = dem.accumulate_flow(0, -1, flow_dir, acc, DIR_OFFSETS)
        assert result == 0, f"Expected 0 for out-of-bounds col, got {result}"

    def test_row_beyond_grid(self, make_dem):
        """Out-of-bounds row (>= rows) returns 0.

        Test scenario:
            r=1 on a 1-row grid is out of bounds.
        """
        dem = make_dem(np.array([[100.0]]))
        flow_dir = np.array([[0]], dtype=np.int32)
        acc = np.full_like(flow_dir, -1)
        result = dem.accumulate_flow(1, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 0, f"Expected 0 for row beyond grid, got {result}"

    def test_col_beyond_grid(self, make_dem):
        """Out-of-bounds column (>= cols) returns 0.

        Test scenario:
            c=1 on a 1-column grid is out of bounds.
        """
        dem = make_dem(np.array([[100.0]]))
        flow_dir = np.array([[0]], dtype=np.int32)
        acc = np.full_like(flow_dir, -1)
        result = dem.accumulate_flow(0, 1, flow_dir, acc, DIR_OFFSETS)
        assert result == 0, f"Expected 0 for col beyond grid, got {result}"


class TestAccumulateFlowCachedCells:
    """Verify that already-cached cells return immediately."""

    def test_cached_zero(self, make_dem):
        """A cell with acc=0 (cached, no upstream) returns 0 immediately.

        Test scenario:
            Pre-set acc[0,0]=0 and verify the method returns 0 without
            re-processing.
        """
        dem = make_dem(np.array([[100.0]]))
        flow_dir = np.array([[0]], dtype=np.int32)
        acc = np.array([[0]], dtype=np.int32)
        result = dem.accumulate_flow(0, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 0, f"Expected cached value 0, got {result}"

    def test_cached_positive(self, make_dem):
        """A cell with acc=5 (cached) returns 5 immediately.

        Test scenario:
            Pre-set acc[0,0]=5, confirm cached value returned unchanged.
        """
        dem = make_dem(np.array([[100.0]]))
        flow_dir = np.array([[0]], dtype=np.int32)
        acc = np.array([[5]], dtype=np.int32)
        result = dem.accumulate_flow(0, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 5, f"Expected cached value 5, got {result}"


class TestAccumulateFlowNoUpstream:
    """Cell with no upstream neighbours (headwater)."""

    def test_isolated_single_cell(self, make_dem):
        """A 1x1 grid has no neighbours, accumulation is 0.

        Test scenario:
            Single cell grid — no neighbour can point toward it.
        """
        dem = make_dem(np.array([[100.0]]))
        flow_dir = np.array([[0]], dtype=np.int32)
        acc = np.full((1, 1), -1, dtype=np.int32)
        result = dem.accumulate_flow(0, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 0, f"Expected 0 upstream for single cell, got {result}"
        assert acc[0, 0] == 0, f"acc should be cached as 0, got {acc[0, 0]}"

    def test_no_neighbour_points_inward(self, make_dem):
        """A 3x3 grid where all neighbours flow away from the center.

        Test scenario:
            Center cell at (1,1).  All 8 neighbours flow outward (away
            from center).  The center should have 0 upstream cells.

            Flow directions are set so each neighbour flows *away* from
            the center:
              - (0,0) flows NW (dir 3): offset (-1,-1), away from center
              - (0,1) flows N  (dir 4): offset (0,-1),  away from center
              - (0,2) flows NE (dir 5): offset (1,-1),  away from center
              - (1,0) flows W  (dir 2): offset (-1,0),  away from center
              - (1,2) flows E  (dir 6): offset (1,0),   away from center
              - (2,0) flows SW (dir 1): offset (-1,1),  away from center
              - (2,1) flows S  (dir 0): offset (0,1),   away from center
              - (2,2) flows SE (dir 7): offset (1,1),   away from center
        """
        dem = make_dem(np.ones((3, 3), dtype=np.float32) * 100)
        flow_dir = np.array([
            [3, 4, 5],
            [2, 0, 6],
            [1, 0, 7],
        ], dtype=np.int32)
        acc = np.full((3, 3), -1, dtype=np.int32)
        result = dem.accumulate_flow(1, 1, flow_dir, acc, DIR_OFFSETS)
        assert result == 0, f"Expected 0 upstream when all neighbours flow away, got {result}"


class TestAccumulateFlowSingleUpstream:
    """One neighbour drains into the target cell."""

    def test_one_cell_flows_south_into_target(self, make_dem):
        """A 2-row grid where the top cell flows south into the bottom cell.

        Test scenario:
            Grid:
                row 0: dir=0 (South, offset (0,1)) → flows into row 1
                row 1: dir=0 (irrelevant, this is the target)
            Target (1,0) should have 1 upstream cell.
        """
        dem = make_dem(np.array([[200.0], [100.0]]))
        # Direction 0 = South = (col_off=0, row_off=1)
        flow_dir = np.array([[0], [0]], dtype=np.int32)
        acc = np.full((2, 1), -1, dtype=np.int32)
        result = dem.accumulate_flow(1, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 1, f"Expected 1 upstream cell, got {result}"
        assert acc[1, 0] == 1, f"acc[1,0] should be 1, got {acc[1, 0]}"
        assert acc[0, 0] == 0, f"acc[0,0] (headwater) should be 0, got {acc[0, 0]}"

    def test_one_cell_flows_east_into_target(self, make_dem):
        """A 1x3 row where the left cell flows east into the middle cell.

        Test scenario:
            Grid 1x3:
                (0,0) dir=6 (East) → flows into (0,1)
                (0,1) dir=6 (East) → target
                (0,2) dir=6 (East) → flows away
            Target (0,1) should have 1 upstream cell from (0,0).
        """
        dem = make_dem(np.array([[300.0, 200.0, 100.0]]))
        flow_dir = np.array([[6, 6, 6]], dtype=np.int32)
        acc = np.full((1, 3), -1, dtype=np.int32)
        result = dem.accumulate_flow(0, 1, flow_dir, acc, DIR_OFFSETS)
        assert result == 1, f"Expected 1 upstream cell, got {result}"


class TestAccumulateFlowChain:
    """Linear chain of cells where each drains into the next."""

    def test_three_cell_south_chain(self, make_dem):
        """A 3x1 column where cells flow south in a chain.

        Test scenario:
            (0,0) dir=0 (S) → (1,0) dir=0 (S) → (2,0) target
            Target (2,0) should have 2 upstream cells.
        """
        dem = make_dem(np.array([[300.0], [200.0], [100.0]]))
        flow_dir = np.array([[0], [0], [0]], dtype=np.int32)
        acc = np.full((3, 1), -1, dtype=np.int32)
        result = dem.accumulate_flow(2, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 2, f"Expected 2 upstream cells in chain, got {result}"
        assert acc[0, 0] == 0, f"Headwater acc should be 0, got {acc[0, 0]}"
        assert acc[1, 0] == 1, f"Middle cell acc should be 1, got {acc[1, 0]}"
        assert acc[2, 0] == 2, f"Outlet acc should be 2, got {acc[2, 0]}"

    def test_five_cell_east_chain(self, make_dem):
        """A 1x5 row flowing east — outlet gets 4 upstream cells.

        Test scenario:
            All cells flow East (dir=6).  The rightmost cell is the
            target and should accumulate all 4 upstream cells.
        """
        dem = make_dem(np.array([[500, 400, 300, 200, 100]], dtype=np.float32))
        flow_dir = np.array([[6, 6, 6, 6, 6]], dtype=np.int32)
        acc = np.full((1, 5), -1, dtype=np.int32)
        result = dem.accumulate_flow(0, 4, flow_dir, acc, DIR_OFFSETS)
        assert result == 4, f"Expected 4 upstream cells, got {result}"


class TestAccumulateFlowMultipleUpstream:
    """Target cell receives flow from multiple neighbours."""

    def test_two_tributaries(self, make_dem):
        """Two cells both flow into the same target.

        Test scenario:
            Grid 1x3 row:
                (0,0) dir=6 (E) → flows into (0,1)
                (0,2) dir=2 (W) → flows into (0,1)
                (0,1) dir=0 (S, flows away, no downstream in grid)
            Target (0,1) should have 2 upstream cells.
        """
        dem = make_dem(np.array([[200.0, 100.0, 200.0]]))
        flow_dir = np.array([[6, 0, 2]], dtype=np.int32)
        acc = np.full((1, 3), -1, dtype=np.int32)
        result = dem.accumulate_flow(0, 1, flow_dir, acc, DIR_OFFSETS)
        assert result == 2, f"Expected 2 tributaries, got {result}"

    def test_fan_shaped_drainage(self, make_dem):
        """All 8 cells in a 3x3 grid drain into the bottom-center target.

        Test scenario:
            3x3 grid.  Target at (2,1) on the bottom edge (flows south
            out of grid — no cycle).  All other 8 cells eventually
            drain into (2,1) through chains:
              (0,0)→SE→(1,1)→S→(2,1)
              (0,1)→S→(1,1)→S→(2,1)
              (0,2)→S→(1,2)→S→(2,2)→W→(2,1)
              (1,0)→S→(2,0)→E→(2,1)
              (1,1)→S→(2,1)
              (1,2)→S→(2,2)→W→(2,1)
              (2,0)→E→(2,1)
              (2,2)→W→(2,1)
            Total: 8 upstream cells.
        """
        dem = make_dem(np.ones((3, 3), dtype=np.float32) * 100)
        flow_dir = np.array([
            [7, 0, 0],
            [0, 0, 0],
            [6, 0, 2],
        ], dtype=np.int32)
        acc = np.full((3, 3), -1, dtype=np.int32)
        result = dem.accumulate_flow(2, 1, flow_dir, acc, DIR_OFFSETS)
        assert result == 8, f"Expected 8 upstream cells, got {result}"

    def test_tributary_with_own_upstream(self, make_dem):
        """A tributary that itself has upstream cells contributes its total.

        Test scenario:
            Grid 3x2:
                (0,0) dir=0 (S)→(1,0)   (0,1) dir=0 (S)→(1,1)
                (1,0) dir=0 (S)→(2,0)   (1,1) dir=0 (S)→(2,1)
                (2,0) target             (2,1) dir=2 (W)→(2,0)
            Full drainage: (0,0)→(1,0)→(2,0), (0,1)→(1,1)→(2,1)→(2,0).
            Target (2,0) gets 5 upstream cells.
        """
        dem = make_dem(np.array([
            [300.0, 100.0],
            [200.0, 100.0],
            [100.0, 150.0],
        ]))
        flow_dir = np.array([
            [0, 0],
            [0, 0],
            [0, 2],
        ], dtype=np.int32)
        acc = np.full((3, 2), -1, dtype=np.int32)
        result = dem.accumulate_flow(2, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 5, f"Expected 5 upstream cells, got {result}"


class TestAccumulateFlowBoundaryConditions:
    """Cells on the grid boundary with limited neighbours."""

    def test_corner_cell_no_upstream(self, make_dem):
        """Top-left corner cell (0,0) with no upstream.

        Test scenario:
            In a 3x3 grid where all cells flow south, the top-left
            corner has only 3 possible upstream neighbours (N, NW, NE)
            which are all out of bounds.  Result should be 0.
        """
        dem = make_dem(np.ones((3, 3), dtype=np.float32) * 100)
        flow_dir = np.full((3, 3), 0, dtype=np.int32)  # all flow south
        acc = np.full((3, 3), -1, dtype=np.int32)
        result = dem.accumulate_flow(0, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 0, f"Expected 0 for top-left corner, got {result}"

    def test_edge_cell_receives_from_interior(self, make_dem):
        """Edge cell receives flow from an interior cell.

        Test scenario:
            Grid 2x2:
                (0,0) dir=6 (E) → (0,1)
                (1,0) dir=0       (1,1) dir=0
            Target (0,1) on right edge receives from (0,0).
        """
        dem = make_dem(np.array([
            [200.0, 100.0],
            [200.0, 100.0],
        ]))
        flow_dir = np.array([
            [6, 0],
            [0, 0],
        ], dtype=np.int32)
        acc = np.full((2, 2), -1, dtype=np.int32)
        result = dem.accumulate_flow(0, 1, flow_dir, acc, DIR_OFFSETS)
        assert result == 1, f"Expected 1 upstream from interior, got {result}"


class TestAccumulateFlowCaching:
    """Verify the accumulation array is correctly populated as a cache."""

    def test_calling_twice_returns_cached(self, make_dem):
        """Second call returns the cached value without re-computation.

        Test scenario:
            Compute accumulation for a cell, then call again.  The
            second call should hit acc >= 0 and return immediately.
        """
        dem = make_dem(np.array([[200.0], [100.0]]))
        flow_dir = np.array([[0], [0]], dtype=np.int32)
        acc = np.full((2, 1), -1, dtype=np.int32)

        result1 = dem.accumulate_flow(1, 0, flow_dir, acc, DIR_OFFSETS)
        result2 = dem.accumulate_flow(1, 0, flow_dir, acc, DIR_OFFSETS)
        assert result1 == result2 == 1, (
            f"Both calls should return 1, got {result1} and {result2}"
        )

    def test_upstream_cells_cached_after_processing(self, make_dem):
        """After processing outlet, all upstream cells should be cached.

        Test scenario:
            Chain of 4 cells flowing south.  After computing the outlet,
            all intermediate cells should have their acc cached.
        """
        dem = make_dem(np.array([[400.0], [300.0], [200.0], [100.0]]))
        flow_dir = np.array([[0], [0], [0], [0]], dtype=np.int32)
        acc = np.full((4, 1), -1, dtype=np.int32)
        dem.accumulate_flow(3, 0, flow_dir, acc, DIR_OFFSETS)

        assert acc[0, 0] == 0, f"Headwater should be cached as 0, got {acc[0, 0]}"
        assert acc[1, 0] == 1, f"Second cell should be cached as 1, got {acc[1, 0]}"
        assert acc[2, 0] == 2, f"Third cell should be cached as 2, got {acc[2, 0]}"
        assert acc[3, 0] == 3, f"Outlet should be cached as 3, got {acc[3, 0]}"


class TestAccumulateFlowDiagonal:
    """Flow along diagonal directions."""

    def test_diagonal_southeast_chain(self, make_dem):
        """Chain flowing southeast, plus incidental south-flowing cells.

        Test scenario:
            3x3 grid with all dir=0 except diagonal:
                (0,0) dir=7 (SE)→(1,1)   (0,1) dir=0 (S)→(1,1)  (0,2) dir=0 (S)→(1,2)
                (1,0) dir=0 (S)→(2,0)    (1,1) dir=7 (SE)→(2,2)  (1,2) dir=0 (S)→(2,2)
                (2,0) dir=0 (out)         (2,1) dir=0 (out)        (2,2) target
            Cells draining into (2,2): (1,1)→(2,2), (1,2)→(2,2),
            and transitively (0,0)→(1,1), (0,1)→(1,1), (0,2)→(1,2).
            Total: 5 upstream cells.
        """
        dem = make_dem(np.array([
            [300.0, 0.0, 0.0],
            [0.0, 200.0, 0.0],
            [0.0, 0.0, 100.0],
        ]))
        flow_dir = np.array([
            [7, 0, 0],
            [0, 7, 0],
            [0, 0, 0],
        ], dtype=np.int32)
        acc = np.full((3, 3), -1, dtype=np.int32)
        result = dem.accumulate_flow(2, 2, flow_dir, acc, DIR_OFFSETS)
        assert result == 5, f"Expected 5 upstream cells, got {result}"

    def test_diagonal_northwest_single(self, make_dem):
        """Single diagonal cell plus a south-chaining cell flow into target.

        Test scenario:
            2x2 grid:
                (0,0) target       (0,1) dir=0 (S)→(1,1)
                (1,0) dir=0 (out)  (1,1) dir=3 (NW)→(0,0)
            (1,1) flows NW into (0,0), and (0,1) flows S into (1,1).
            Target (0,0) gets 2 upstream: (1,1) and (0,1)→(1,1).
        """
        dem = make_dem(np.array([
            [100.0, 200.0],
            [200.0, 300.0],
        ]))
        flow_dir = np.array([
            [0, 0],
            [0, 3],
        ], dtype=np.int32)
        acc = np.full((2, 2), -1, dtype=np.int32)
        result = dem.accumulate_flow(0, 0, flow_dir, acc, DIR_OFFSETS)
        assert result == 2, f"Expected 2 upstream cells, got {result}"


class TestAccumulateFlowComplexTopology:
    """Complex drainage patterns combining chains and tributaries."""

    def test_y_shaped_network(self, make_dem):
        """Y-shaped drainage: two branches merge into a single channel.

        Test scenario:
            Grid 4x3:
                (0,0) dir=7 (SE)  (0,1) dir=0      (0,2) dir=1 (SW)
                (1,0) dir=0       (1,1) dir=0 (S)   (1,2) dir=0
                (2,0) dir=0       (2,1) dir=0 (S)   (2,2) dir=0
                (3,0) dir=0       (3,1) target       (3,2) dir=0

            Branch 1: (0,0) → (1,1) via SE
            Branch 2: (0,2) → (1,1) via SW
            Trunk: (1,1) → (2,1) → (3,1) via S

            Full network:
              (0,0)→SE→(1,1), (0,1)→S→(1,1), (0,2)→SW→(1,1)
              (1,0)→S→(2,0), (1,1)→S→(2,1), (1,2)→S→(2,2)
              (2,0)→S→(3,0), (2,1)→S→(3,1), (2,2)→S→(3,2)
            Only cells whose chain reaches (3,1): (0,0)→(1,1)→(2,1)→(3,1),
            (0,1)→(1,1)→(2,1)→(3,1), (0,2)→(1,1)→(2,1)→(3,1),
            (1,1)→(2,1)→(3,1), (2,1)→(3,1) = 5 upstream cells.
        """
        dem = make_dem(np.array([
            [400.0, 400.0, 400.0],
            [300.0, 300.0, 300.0],
            [200.0, 200.0, 200.0],
            [100.0, 100.0, 100.0],
        ]))
        flow_dir = np.array([
            [7, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int32)
        acc = np.full((4, 3), -1, dtype=np.int32)
        result = dem.accumulate_flow(3, 1, flow_dir, acc, DIR_OFFSETS)
        assert result == 5, f"Expected 5 upstream cells in Y-network, got {result}"


class TestAccumulateFlowCustomDirOffsets:
    """Verify the method works with a custom dir_offsets mapping."""

    def test_subset_offsets(self, make_dem):
        """Custom dir_offsets with only 4 cardinal directions.

        Test scenario:
            Use a subset of DIR_OFFSETS (only cardinal directions).
            Target at (2,1) on the bottom edge.
            Cells (0,0) and (1,0) have dir=7 (SE) — not in cardinal
            set, so their direction is never matched as upstream.
            But cells with dir=0 (S) chain through cardinal directions.
            Full reachable network via cardinal_only:
              (0,1)→S→(1,1)→S→(2,1)
              (0,2)→S→(1,2)→S→(2,2)→W→(2,1)
              (2,0)→E→(2,1)
            Total: 6 upstream.  Compare to 8 with full DIR_OFFSETS.
        """
        cardinal_only = {
            0: (0, 1),   # South
            2: (-1, 0),  # West
            4: (0, -1),  # North
            6: (1, 0),   # East
        }
        dem = make_dem(np.ones((3, 3), dtype=np.float32) * 100)
        flow_dir = np.array([
            [7, 0, 0],
            [7, 0, 0],
            [6, 0, 2],
        ], dtype=np.int32)
        acc = np.full((3, 3), -1, dtype=np.int32)
        result = dem.accumulate_flow(2, 1, flow_dir, acc, cardinal_only)
        assert result == 6, f"Expected 6 cardinal upstream, got {result}"

        # Verify fewer upstream than with full DIR_OFFSETS
        acc_full = np.full((3, 3), -1, dtype=np.int32)
        result_full = dem.accumulate_flow(2, 1, flow_dir, acc_full, DIR_OFFSETS)
        assert result_full > result, (
            f"Full offsets ({result_full}) should find more upstream than cardinal ({result})"
        )


class TestFlowAccumulationEndToEnd:
    """End-to-end tests using flow_accumulation on hand-crafted flow dirs."""

    def test_uniform_south_chain(self, make_dem):
        """5-cell column all flowing south — outlet accumulates 4.

        Test scenario:
            Hand-craft a flow direction array for a 5x1 column where
            every cell flows south (dir=0).  The bottom cell should
            accumulate 4 upstream cells.
        """
        dem = make_dem(np.array([[500], [400], [300], [200], [100]], dtype=np.float32))
        flow_dir = np.array([[0], [0], [0], [0], [0]], dtype=np.int32)
        acc = np.full((5, 1), -1, dtype=np.int32)
        dem.accumulate_flow(4, 0, flow_dir, acc, DIR_OFFSETS)

        assert acc[0, 0] == 0, f"Top cell should be 0, got {acc[0, 0]}"
        assert acc[4, 0] == 4, f"Bottom cell should accumulate 4, got {acc[4, 0]}"

    def test_v_shaped_valley(self, make_dem):
        """V-shaped valley: two slopes converge on a center channel.

        Test scenario:
            3x3 grid with hand-crafted flow directions modelling a
            valley along column 1:
              (0,0)→E→(0,1)  (0,1)→S→(1,1)  (0,2)→W→(0,1)
              (1,0)→E→(1,1)  (1,1)→S→(2,1)  (1,2)→W→(1,1)
              (2,0)→E→(2,1)  (2,1) target     (2,2)→W→(2,1)
            Total upstream of (2,1): all 8 other cells = 8.
        """
        dem = make_dem(np.array([
            [200, 100, 200],
            [200, 50, 200],
            [200, 25, 200],
        ], dtype=np.float32))
        flow_dir = np.array([
            [6, 0, 2],
            [6, 0, 2],
            [6, 0, 2],
        ], dtype=np.int32)
        acc = np.full((3, 3), -1, dtype=np.int32)
        result = dem.accumulate_flow(2, 1, flow_dir, acc, DIR_OFFSETS)
        assert result == 8, f"Valley outlet should accumulate 8, got {result}"


@pytest.mark.slow
class TestFlowAccumulationCoello:
    """Integration tests with the Coello test dataset."""

    def test_coello_accumulation_basic_properties(
        self,
        coello_dem_4000: gdal.Dataset,
        coello_flow_direction_4000: gdal.Dataset,
    ):
        """Flow accumulation on the Coello basin has expected properties.

        Test scenario:
            - Output is a Dataset with int32 dtype.
            - No-data cells remain no-data.
            - All valid cells have non-negative accumulation.
            - The maximum accumulation is consistent with the number
              of valid cells.
        """
        dem = DEM(coello_dem_4000)
        fd = DEM(coello_flow_direction_4000)
        acc_ds = dem.flow_accumulation(fd)

        arr = acc_ds.read_array()
        assert acc_ds.dtype == ["int32"], f"Expected int32, got {acc_ds.dtype}"

        no_data = Dataset.default_no_data_value
        valid_mask = arr != no_data
        valid_cells = arr[valid_mask]

        assert np.all(valid_cells >= 0), "All valid cells should have non-negative accumulation"

        n_valid = valid_cells.size
        max_acc = valid_cells.max()
        assert max_acc < n_valid, (
            f"Max accumulation ({max_acc}) should be less than total valid cells ({n_valid})"
        )

    def test_coello_accumulation_no_data_matches_dem(
        self,
        coello_dem_4000: gdal.Dataset,
        coello_flow_direction_4000: gdal.Dataset,
    ):
        """No-data cells in accumulation match no-data cells in the DEM.

        Test scenario:
            Cells that are no-data in the DEM (outside the basin) should
            also be no-data in the accumulation raster.
        """
        dem = DEM(coello_dem_4000)
        fd = DEM(coello_flow_direction_4000)
        acc_ds = dem.flow_accumulation(fd)

        acc_arr = acc_ds.read_array()
        elev = dem.values
        no_data = Dataset.default_no_data_value

        dem_nodata_mask = np.isnan(elev)
        acc_nodata_mask = acc_arr == no_data

        assert np.array_equal(dem_nodata_mask, acc_nodata_mask), (
            "No-data pattern in accumulation should match DEM no-data pattern"
        )

    def test_coello_headwater_cells_are_zero(
        self,
        coello_dem_4000: gdal.Dataset,
        coello_flow_direction_4000: gdal.Dataset,
    ):
        """Some cells in the Coello basin should be headwaters (acc=0).

        Test scenario:
            Ridge cells and boundary cells with no upstream neighbours
            should have accumulation = 0.  At least some must exist.
        """
        dem = DEM(coello_dem_4000)
        fd = DEM(coello_flow_direction_4000)
        acc_ds = dem.flow_accumulation(fd)
        arr = acc_ds.read_array()

        no_data = Dataset.default_no_data_value
        valid = arr[arr != no_data]
        headwaters = np.sum(valid == 0)
        assert headwaters > 0, (
            f"Expected at least some headwater cells (acc=0), found {headwaters}"
        )

    def test_coello_accumulation_sum_consistency(
        self,
        coello_dem_4000: gdal.Dataset,
        coello_flow_direction_4000: gdal.Dataset,
    ):
        """Sum of all accumulation values equals expected drainage total.

        Test scenario:
            In a fully connected D8 network with N valid cells, the sum
            of all accumulation values equals N*(N-1)/2 only for a
            single linear chain.  For a tree, the sum equals N - 1
            (each cell except the outlet is counted once as an upstream
            contributor somewhere).  We verify sum >= N-1.
        """
        dem = DEM(coello_dem_4000)
        fd = DEM(coello_flow_direction_4000)
        acc_ds = dem.flow_accumulation(fd)
        arr = acc_ds.read_array()

        no_data = Dataset.default_no_data_value
        valid = arr[arr != no_data]
        n_valid = valid.size
        total_acc = int(valid.sum())

        assert total_acc >= n_valid - 1, (
            f"Total accumulation ({total_acc}) should be >= N-1 ({n_valid - 1})"
        )
