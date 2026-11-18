"""Tests for the pure-numpy surface kernels behind the morphometric indices.

These could not exist before the kernels were extracted: every case here is a hand-written
grid, so the assertions are against arithmetic with a known answer rather than against
whatever the Coello fixture happens to contain.

Two properties carry most of the weight. The slope kernel must divide diagonal runs by
`sqrt(2)` — without that, diagonal slopes read ~41% too steep and D8 routing biases toward
the diagonals, which is the kind of error that produces a plausible-looking wrong network.
And the focal kernel must exclude no-data from its window rather than averaging it in,
which is what stops a hole dragging every cell around it toward the fill value.
"""

from __future__ import annotations

import numpy as np
import pytest

from digitalrivers.dem._kernels.surface import (
    eight_direction_slopes,
    focal_window_stats,
)

SQRT2 = float(np.sqrt(2.0))


class TestEightDirectionSlopes:
    """Tests for eight_direction_slopes."""

    def test_shape_is_one_layer_per_direction(self):
        """The output carries eight slopes per cell, in `DIR_OFFSETS` order."""
        s = eight_direction_slopes(np.zeros((4, 5), dtype=np.float32), 1.0)
        assert s.shape == (4, 5, 8), f"Expected (4, 5, 8), got {s.shape}"

    def test_flat_surface_has_zero_slope_everywhere_in_bounds(self):
        """A constant surface slopes nowhere; only the off-grid neighbours are NaN."""
        s = eight_direction_slopes(np.full((5, 5), 3.0, dtype=np.float32), 1.0)
        interior = s[1:-1, 1:-1, :]
        assert np.allclose(interior, 0.0), f"Interior is not flat:\n{interior}"

    def test_slope_is_positive_downhill(self):
        """A cell above its neighbour has a positive slope toward it."""
        z = np.zeros((3, 3), dtype=np.float32)
        z[1, 1] = 1.0
        s = eight_direction_slopes(z, 1.0)
        assert s[1, 1, 0] > 0, f"Expected positive slope downhill, got {s[1, 1, 0]}"
        assert s[0, 1, 0] < 0, f"Expected negative slope uphill, got {s[0, 1, 0]}"

    def test_cardinal_slope_is_rise_over_cell_size(self):
        """A one-metre drop over a two-metre cell is a slope of 0.5."""
        z = np.zeros((3, 3), dtype=np.float32)
        z[1, 1] = 1.0
        s = eight_direction_slopes(z, 2.0)
        assert np.isclose(s[1, 1, 0], 0.5), f"Expected 0.5, got {s[1, 1, 0]}"

    def test_diagonal_run_is_longer_by_root_two(self):
        """The diagonal divisor is `cell_size * sqrt(2)`, not `cell_size`.

        Test scenario:
            The single most consequential constant in the kernel. Dividing a diagonal by
            the cell size instead makes diagonal slopes read about 41% too steep, and D8
            then prefers the diagonals over the cardinals on a surface where it should
            not.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        z[1, 1] = 1.0
        s = eight_direction_slopes(z, 1.0)
        assert np.isclose(
            s[1, 1, 1], 1.0 / SQRT2, atol=1e-6
        ), f"Diagonal slope {s[1, 1, 1]} is not 1/sqrt(2)"
        assert np.isclose(
            s[1, 1, 0] / s[1, 1, 1], SQRT2, atol=1e-5
        ), "Cardinal and diagonal slopes are not in the sqrt(2) ratio"

    @pytest.mark.parametrize(
        ("direction", "expected_offset"),
        [(0, (1, 0)), (2, (0, -1)), (4, (-1, 0)), (6, (0, 1))],
    )
    def test_each_cardinal_layer_reads_the_right_neighbour(
        self, direction, expected_offset
    ):
        """Layer `k` compares the centre against the neighbour `DIR_OFFSETS[k]` names.

        Args:
            direction: The direction code under test.
            expected_offset: The `(row, col)` step that code means.

        Test scenario:
            Puts a single low cell at that offset and asserts only that layer sees it, so
            a transposed or rotated layer ordering fails here rather than silently
            re-routing every downstream product.
        """
        z = np.zeros((5, 5), dtype=np.float32)
        dr, dc = expected_offset
        z[2 + dr, 2 + dc] = -1.0
        s = eight_direction_slopes(z, 1.0)
        assert s[2, 2, direction] > 0.9, (
            f"Layer {direction} should see the drop at {expected_offset}, "
            f"got {s[2, 2, direction]}"
        )

    def test_off_grid_neighbours_are_nan_not_zero(self):
        """Edge cells report NaN outward, so `nanmax` finds the steepest in-bounds drop."""
        s = eight_direction_slopes(np.zeros((3, 3), dtype=np.float32), 1.0)
        assert np.isnan(s[0, 0, 4]), "North of the top-left corner should be NaN"
        assert np.isnan(s[0, 0, 2]), "West of the top-left corner should be NaN"
        assert not np.isnan(s[0, 0, 0]), "South of the top-left corner is in bounds"

    def test_no_data_neighbour_yields_nan(self):
        """A NaN neighbour produces a NaN slope rather than propagating a number."""
        z = np.zeros((3, 3), dtype=np.float32)
        z[2, 1] = np.nan
        s = eight_direction_slopes(z, 1.0)
        assert np.isnan(s[1, 1, 0]), "Slope toward a no-data cell should be NaN"

    def test_does_not_mutate_its_input(self):
        """The kernel pads a copy; the caller's elevations are untouched."""
        z = np.arange(9, dtype=np.float32).reshape(3, 3)
        before = z.copy()
        eight_direction_slopes(z, 1.0)
        assert np.array_equal(z, before), "Input array was modified"


class TestFocalWindowStats:
    """Tests for focal_window_stats."""

    def test_returns_the_surface_alongside_the_statistics(self):
        """The first element is the input as float64, so callers need not re-cast."""
        z, _mean, _sd = focal_window_stats(np.full((3, 3), 2.0, dtype=np.float32), 3)
        assert z.dtype == np.float64, f"Expected float64, got {z.dtype}"
        assert np.allclose(z, 2.0), "The returned surface is not the input"

    def test_flat_surface_has_mean_equal_to_itself_and_zero_deviation(self):
        """Nothing varies, so every window's mean is the value and its SD is zero."""
        _z, mean, sd = focal_window_stats(np.full((5, 5), 7.0), 3)
        assert np.allclose(mean, 7.0), f"Mean drifted from 7.0:\n{mean}"
        assert np.allclose(sd, 0.0), f"SD is not zero on a flat surface:\n{sd}"

    def test_mean_of_a_known_window(self):
        """A 3x3 window over 0..8 averages to 4."""
        _z, mean, _sd = focal_window_stats(
            np.arange(9, dtype=np.float64).reshape(3, 3), 3
        )
        assert np.isclose(mean[1, 1], 4.0), f"Expected 4.0, got {mean[1, 1]}"

    def test_standard_deviation_of_a_known_window(self):
        """The SD of 0..8 is its population SD, not the sample one."""
        _z, _mean, sd = focal_window_stats(
            np.arange(9, dtype=np.float64).reshape(3, 3), 3
        )
        expected = float(np.std(np.arange(9, dtype=np.float64)))
        assert np.isclose(sd[1, 1], expected), f"Expected {expected}, got {sd[1, 1]}"

    def test_no_data_is_excluded_from_the_window_not_averaged_in(self):
        """A hole must not drag its neighbours' mean toward the fill value.

        Test scenario:
            Every valid cell is 10.0 with one NaN in the middle. The kernel zero-fills
            internally, so if it averaged over the whole window rather than over the
            valid count, the neighbouring means would come out near 8.9 instead of 10.0.
        """
        z = np.full((5, 5), 10.0)
        z[2, 2] = np.nan
        _z, mean, sd = focal_window_stats(z, 3)
        assert np.isclose(
            mean[1, 1], 10.0
        ), f"A no-data neighbour dragged the mean to {mean[1, 1]}"
        # Not exactly zero: E[x^2] - E[x]^2 cancels to ~1e-14 at this magnitude and the
        # square root lifts that to ~1e-7. That residual is the cancellation the clip in
        # the kernel exists to keep non-negative, so the tolerance names it rather than
        # pretending the arithmetic is exact.
        assert np.isclose(
            sd[1, 1], 0.0, atol=1e-6
        ), f"Deviation around a hole should be ~0, got {sd[1, 1]}"

    def test_window_with_nothing_valid_is_nan(self):
        """A fully no-data window yields NaN rather than zero."""
        _z, mean, sd = focal_window_stats(np.full((3, 3), np.nan), 3)
        assert np.isnan(mean).all(), "Mean should be NaN with nothing to average"
        assert np.isnan(sd).all(), "SD should be NaN with nothing to average"

    def test_sd_is_nan_only_where_the_mean_is(self):
        """The variance is clipped before the root, tying the two together.

        Test scenario:
            Callers divide by the SD and guard on it being zero. If cancellation could
            leave a negative variance, `sqrt` would give NaN at a cell whose mean is
            finite, and that guard would not fire. The clip is what prevents it.
        """
        z = np.full((8, 8), 1e8)
        z[0:3, 0:3] = np.nan
        _z, mean, sd = focal_window_stats(z, 3)
        assert np.array_equal(
            np.isnan(sd), np.isnan(mean)
        ), "SD and mean disagree about which cells are undefined"

    def test_window_of_one_is_the_surface_itself(self):
        """A 1x1 window means each cell is its own neighbourhood."""
        z = np.arange(9, dtype=np.float64).reshape(3, 3)
        _z, mean, sd = focal_window_stats(z, 1)
        assert np.allclose(mean, z), f"1x1 mean is not the surface:\n{mean}"
        assert np.allclose(sd, 0.0), f"1x1 SD is not zero:\n{sd}"

    def test_larger_window_smooths_more(self):
        """A wider window flattens the mean surface toward the global average."""
        rng = np.random.default_rng(1337)
        z = rng.normal(0.0, 1.0, (21, 21))
        _z, mean3, _ = focal_window_stats(z, 3)
        _z, mean9, _ = focal_window_stats(z, 9)
        assert float(mean9.std()) < float(
            mean3.std()
        ), "The wider window did not smooth further"

    @pytest.mark.parametrize("window", [0, -1])
    def test_window_below_one_raises(self, window):
        """A window has to contain at least the cell itself.

        Args:
            window: The rejected value under test.
        """
        with pytest.raises(ValueError, match="window must be >= 1") as exc_info:
            focal_window_stats(np.zeros((3, 3)), window)
        assert str(window) in str(exc_info.value), str(exc_info.value)

    def test_does_not_mutate_its_input(self):
        """The kernel reads the surface and leaves it alone."""
        z = np.arange(25, dtype=np.float64).reshape(5, 5)
        before = z.copy()
        focal_window_stats(z, 3)
        assert np.array_equal(z, before), "Input array was modified"
