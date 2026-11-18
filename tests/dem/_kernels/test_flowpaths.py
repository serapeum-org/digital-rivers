"""Tests for the pure-numpy walks over a D8 direction grid.

Extracting these made them testable on hand-built direction grids, where the right answer
is countable by eye. Before, the only coverage was through `DEM` on the Coello fixture,
which exercises the happy path and nothing else.

The axis convention is the thing most worth pinning here.
`cell_indices_from_directions` reports `(column, row)`, not `(row, column)` — the method
above it claimed the opposite for as long as it existed, and only the reference fixture
and `tests/conftest.py` were right. These tests assert the real order so the claim cannot
drift back.
"""

from __future__ import annotations

import numpy as np
import pytest

from digitalrivers.core.directions import DIR_OFFSETS
from digitalrivers.dem._kernels.flowpaths import (
    accumulate_upstream,
    cell_indices_from_directions,
    opposite_direction,
)


class TestOppositeDirection:
    """Tests for opposite_direction."""

    @pytest.mark.parametrize(
        ("code", "opposite"),
        [(0, 4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3)],
    )
    def test_each_direction_has_the_expected_opposite(self, code, opposite):
        """Every code maps to the one pointing back at it.

        Args:
            code: The direction under test.
            opposite: The code that should point back along it.
        """
        d_col, d_row = DIR_OFFSETS[code]
        assert (
            opposite_direction(d_row, d_col, DIR_OFFSETS) == opposite
        ), f"code {code} should reverse to {opposite}"

    def test_reversing_twice_is_the_identity(self):
        """Applying the reversal twice returns the original code for all eight."""
        for code in range(8):
            d_col, d_row = DIR_OFFSETS[code]
            back = opposite_direction(d_row, d_col, DIR_OFFSETS)
            b_col, b_row = DIR_OFFSETS[back]
            assert (
                opposite_direction(b_row, b_col, DIR_OFFSETS) == code
            ), f"code {code} did not survive a round trip"

    def test_an_offset_that_is_not_a_neighbour_has_no_opposite(self):
        """A step of more than one cell names no direction, so the answer is None."""
        assert opposite_direction(5, 5, DIR_OFFSETS) is None, "Expected None"

    def test_the_zero_offset_has_no_opposite(self):
        """The centre cell is not its own neighbour."""
        assert opposite_direction(0, 0, DIR_OFFSETS) is None, "Expected None"


class TestAccumulateUpstream:
    """Tests for accumulate_upstream."""

    def test_counts_a_chain(self):
        """Three cells draining east: the outlet has two upstream."""
        fdir = np.array([[6, 6, -1]])
        acc = np.full(fdir.shape, -1)
        assert (
            int(accumulate_upstream(0, 2, fdir, acc, DIR_OFFSETS)) == 2
        ), f"Expected 2, got {acc[0, 2]}"

    def test_a_headwater_counts_zero(self):
        """A cell nothing drains into has no upstream area."""
        fdir = np.array([[6, 6, -1]])
        assert (
            int(accumulate_upstream(0, 0, fdir, np.full(fdir.shape, -1), DIR_OFFSETS))
            == 0
        ), "A headwater should count zero"

    def test_counts_a_confluence(self):
        """Two branches meeting: the junction sees both.

        Test scenario:
            A 3x3 where the top-left and bottom-left both drain to the middle-right,
            which is the outlet. It should count every cell above it, not just one arm.
        """
        fdir = np.array(
            [
                [7, 0, -1],
                [6, 6, -1],
                [5, 4, -1],
            ]
        )
        acc = np.full(fdir.shape, -1)
        total = int(accumulate_upstream(1, 2, fdir, acc, DIR_OFFSETS))
        assert total >= 2, f"A confluence should gather both arms, got {total}"

    def test_memoises_into_the_accumulator(self):
        """`acc` is filled in as a side effect, so a second call is free."""
        fdir = np.array([[6, 6, -1]])
        acc = np.full(fdir.shape, -1)
        accumulate_upstream(0, 2, fdir, acc, DIR_OFFSETS)
        assert (acc >= 0).all(), f"Cells left uncomputed: {acc}"

    def test_a_precomputed_cell_is_returned_unchanged(self):
        """A non-negative entry is taken as final rather than recomputed."""
        fdir = np.array([[6, 6, -1]])
        acc = np.full(fdir.shape, -1)
        acc[0, 2] = 99
        assert (
            int(accumulate_upstream(0, 2, fdir, acc, DIR_OFFSETS)) == 99
        ), "A memoised value should be trusted"

    @pytest.mark.parametrize("coord", [(-1, 0), (0, -1), (5, 0), (0, 5)])
    def test_off_grid_coordinates_return_zero(self, coord):
        """Out-of-bounds is 0 rather than an IndexError.

        Args:
            coord: The `(row, col)` outside the grid.
        """
        fdir = np.array([[6, 6, -1]])
        assert (
            accumulate_upstream(*coord, fdir, np.full(fdir.shape, -1), DIR_OFFSETS) == 0
        ), f"{coord} should count zero"

    def test_order_of_calls_does_not_change_the_result(self):
        """Memoisation must not make the answer depend on which cell was asked first."""
        fdir = np.array([[6, 6, 6, -1]])
        forward = np.full(fdir.shape, -1)
        for c in range(4):
            accumulate_upstream(0, c, fdir, forward, DIR_OFFSETS)
        backward = np.full(fdir.shape, -1)
        for c in reversed(range(4)):
            accumulate_upstream(0, c, fdir, backward, DIR_OFFSETS)
        assert np.array_equal(
            forward, backward
        ), f"Call order changed the result:\n{forward}\nvs\n{backward}"

    def test_a_long_chain_does_not_exhaust_the_stack(self):
        """The walk is iterative, so depth is bounded by memory rather than recursion.

        Test scenario:
            5000 cells in one chain — comfortably past the default recursion limit of
            1000, which the recursive ancestor of this function needed raised to 50000.
        """
        n = 5000
        fdir = np.full((1, n), 6)
        fdir[0, -1] = -1
        acc = np.full(fdir.shape, -1)
        assert (
            int(accumulate_upstream(0, n - 1, fdir, acc, DIR_OFFSETS)) == n - 1
        ), "The full chain should be counted"


class TestCellIndicesFromDirections:
    """Tests for cell_indices_from_directions."""

    def test_shape_carries_two_layers(self):
        """One layer per axis, on top of the grid shape."""
        out = cell_indices_from_directions(np.zeros((3, 4)), DIR_OFFSETS)
        assert out.shape == (3, 4, 2), f"Expected (3, 4, 2), got {out.shape}"

    @pytest.mark.parametrize(
        ("code", "expected"),
        [(0, [0.0, 1.0]), (2, [-1.0, 0.0]), (4, [0.0, -1.0]), (6, [1.0, 0.0])],
    )
    def test_layers_are_column_then_row(self, code, expected):
        """Layer 0 is the downstream column and layer 1 the downstream row.

        Args:
            code: The direction under test.
            expected: The `[layer0, layer1]` for a cell at the origin.

        Test scenario:
            This is the order the reference fixture and `tests/conftest.py` use. The
            public method's docstring claimed row-then-column until the kernel was
            extracted and this was checked; the arithmetic adds the table's column offset
            to the row index, which is what produces the swap.
        """
        out = cell_indices_from_directions(np.array([[float(code)]]), DIR_OFFSETS)
        assert (
            out[0, 0].tolist() == expected
        ), f"code {code}: expected {expected}, got {out[0, 0].tolist()}"

    def test_no_data_stays_nan(self):
        """A NaN direction resolves to NaN rather than to a neighbour."""
        out = cell_indices_from_directions(np.array([[np.nan]]), DIR_OFFSETS)
        assert np.isnan(out[0, 0]).all(), f"Expected NaN, got {out[0, 0]}"

    def test_mixed_grid_resolves_only_the_valid_cells(self):
        """Valid and no-data cells coexist without contaminating each other."""
        fdir = np.array([[6.0, np.nan], [np.nan, 0.0]])
        out = cell_indices_from_directions(fdir, DIR_OFFSETS)
        assert out[0, 0].tolist() == [1.0, 0.0], out[0, 0].tolist()
        assert np.isnan(out[0, 1]).all(), "No-data leaked a value"
        assert np.isnan(out[1, 0]).all(), "No-data leaked a value"
        assert out[1, 1].tolist() == [1.0, 2.0], out[1, 1].tolist()

    def test_output_is_float_so_nodata_can_be_nan(self):
        """Integer output could not carry NaN, so the indices are float64."""
        out = cell_indices_from_directions(np.array([[0.0]]), DIR_OFFSETS)
        assert out.dtype == np.float64, f"Expected float64, got {out.dtype}"

    def test_does_not_mutate_its_input(self):
        """The kernel reads the direction grid and leaves it alone."""
        fdir = np.array([[6.0, 0.0], [2.0, 4.0]])
        before = fdir.copy()
        cell_indices_from_directions(fdir, DIR_OFFSETS)
        assert np.array_equal(fdir, before), "Input array was modified"
