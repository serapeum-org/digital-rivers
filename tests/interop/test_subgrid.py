"""Tests for `digitalrivers.interop.subgrid.subgrid_table`.

A sub-grid table is what lets a coarse solver behave as if it resolved fine topography:
for each coarse cell it records what fraction of the block lies below each depth level.
Get the fractions wrong and the model's storage curve is wrong, silently — the raster
still looks fine.

The flat-block branch is the one worth pinning. A block where every cell is the same
elevation has no range to bin across, and the natural loop produces nothing at all; the
column would then vanish from the DataFrame for that row, which is worse than a wrong
number because it changes the table's shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from digitalrivers.interop.subgrid import subgrid_table


@pytest.fixture(scope="function")
def ramp_4x4() -> np.ndarray:
    """A 4x4 elevation ramp, 0..15 row-major.

    With `scale_factor=2` this is four 2x2 blocks, each holding four distinct values, so
    every fraction column exercises the non-flat branch.

    Returns:
        np.ndarray: The 4x4 float64 ramp.
    """
    return np.arange(16, dtype=np.float64).reshape(4, 4)


class TestSubgridTable:
    """Tests for subgrid_table."""

    def test_indexes_by_coarse_cell(self, ramp_4x4):
        """The table is indexed by the coarse `(row, col)`, one row per block."""
        df = subgrid_table(ramp_4x4, scale_factor=2, n_bins=3)
        assert list(df.index.names) == ["row", "col"], f"Index: {df.index.names}"
        assert sorted(df.index) == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ], f"Unexpected blocks: {sorted(df.index)}"

    def test_column_count_is_two_plus_n_bins(self, ramp_4x4):
        """Every table carries `z_min`, `z_max` and one fraction column per bin."""
        df = subgrid_table(ramp_4x4, scale_factor=2, n_bins=3)
        assert list(df.columns) == [
            "z_min",
            "z_max",
            "frac_below_1",
            "frac_below_2",
            "frac_below_3",
        ], f"Unexpected columns: {list(df.columns)}"

    def test_z_min_and_z_max_bound_the_block(self, ramp_4x4):
        """Each row reports the elevation range of its own block, not the raster's."""
        df = subgrid_table(ramp_4x4, scale_factor=2, n_bins=2)
        assert df.loc[(0, 0), "z_min"] == 0.0, df.loc[(0, 0), "z_min"]
        assert df.loc[(0, 0), "z_max"] == 5.0, df.loc[(0, 0), "z_max"]
        assert df.loc[(1, 1), "z_min"] == 10.0, df.loc[(1, 1), "z_min"]
        assert df.loc[(1, 1), "z_max"] == 15.0, df.loc[(1, 1), "z_max"]

    def test_top_bin_always_holds_the_whole_block(self, ramp_4x4):
        """The last bin edge is `z_max`, so every cell is at or below it."""
        df = subgrid_table(ramp_4x4, scale_factor=2, n_bins=4)
        assert (
            df["frac_below_4"] == 1.0
        ).all(), f"Top bin should be 1.0 everywhere:\n{df['frac_below_4']}"

    def test_fractions_increase_with_depth(self, ramp_4x4):
        """Fractions are cumulative, so each column is >= the one before it."""
        df = subgrid_table(ramp_4x4, scale_factor=2, n_bins=4)
        cols = [f"frac_below_{k}" for k in range(1, 5)]
        values = df[cols].to_numpy()
        assert (
            np.diff(values, axis=1) >= 0
        ).all(), f"Fractions are not monotonic:\n{values}"

    def test_flat_block_reports_every_fraction_as_one(self):
        """A block with no elevation range still gets a full set of fraction columns.

        Test scenario:
            A uniform 2x2 block has `z_max == z_min`, so there is no range to bin. Every
            bin edge sits at that single elevation, which every cell is at or below, so
            all fractions are 1.0. Dropping them would change the table's shape.
        """
        flat = np.full((2, 2), 7.0)
        df = subgrid_table(flat, scale_factor=2, n_bins=3)
        assert list(df.columns) == [
            "z_min",
            "z_max",
            "frac_below_1",
            "frac_below_2",
            "frac_below_3",
        ], f"Flat block dropped its fraction columns: {list(df.columns)}"
        assert (
            df.loc[(0, 0), ["frac_below_1", "frac_below_2", "frac_below_3"]] == 1.0
        ).all(), f"Flat block fractions wrong:\n{df.loc[(0, 0)]}"

    def test_blocks_with_no_finite_cell_are_omitted(self):
        """An all-no-data block has nothing to describe and gets no row."""
        arr = np.arange(16, dtype=np.float64).reshape(4, 4)
        arr[0:2, 0:2] = np.nan
        df = subgrid_table(arr, scale_factor=2, n_bins=2)
        assert (0, 0) not in df.index, "All-NaN block should have been skipped"
        assert len(df) == 3, f"Expected 3 surviving blocks, got {len(df)}"

    def test_partial_no_data_uses_only_the_finite_cells(self):
        """A block with some `NaN` is described by what is left, not by zeros."""
        arr = np.array(
            [[1.0, np.nan], [np.nan, 3.0]],
            dtype=np.float64,
        )
        df = subgrid_table(arr, scale_factor=2, n_bins=2)
        assert df.loc[(0, 0), "z_min"] == 1.0, df.loc[(0, 0), "z_min"]
        assert df.loc[(0, 0), "z_max"] == 3.0, df.loc[(0, 0), "z_max"]
        assert (
            df.loc[(0, 0), "frac_below_1"] == 0.5
        ), f"Half the finite cells are at or below the midpoint: {df.loc[(0, 0)]}"

    def test_remainder_rows_and_columns_are_dropped(self):
        """A raster that does not divide evenly contributes only its whole blocks."""
        arr = np.arange(25, dtype=np.float64).reshape(5, 5)
        df = subgrid_table(arr, scale_factor=2, n_bins=2)
        assert len(df) == 4, f"5x5 at scale 2 gives four whole blocks, got {len(df)}"
        assert (1, 2) not in df.index, "A partial block was included"

    def test_single_bin_is_all_ones(self, ramp_4x4):
        """With one bin the only edge is `z_max`, so the fraction is 1.0 everywhere."""
        df = subgrid_table(ramp_4x4, scale_factor=2, n_bins=1)
        assert (df["frac_below_1"] == 1.0).all(), f"\n{df}"

    @pytest.mark.parametrize("scale_factor", [-1, 0, 1])
    def test_scale_factor_below_two_raises(self, ramp_4x4, scale_factor):
        """A coarse cell must aggregate at least a 2x2 block to mean anything.

        Args:
            scale_factor: The rejected value under test.
        """
        with pytest.raises(ValueError, match="scale_factor must be >= 2") as exc_info:
            subgrid_table(ramp_4x4, scale_factor=scale_factor, n_bins=2)
        assert str(scale_factor) in str(
            exc_info.value
        ), f"Message omits the bad value: {exc_info.value}"

    @pytest.mark.parametrize("n_bins", [-1, 0])
    def test_n_bins_below_one_raises(self, ramp_4x4, n_bins):
        """A table with no depth levels carries no information.

        Args:
            n_bins: The rejected value under test.
        """
        with pytest.raises(ValueError, match="n_bins must be >= 1") as exc_info:
            subgrid_table(ramp_4x4, scale_factor=2, n_bins=n_bins)
        assert str(n_bins) in str(
            exc_info.value
        ), f"Message omits the bad value: {exc_info.value}"

    def test_does_not_mutate_its_input(self, ramp_4x4):
        """The kernel reads the elevation array and leaves it untouched."""
        before = ramp_4x4.copy()
        subgrid_table(ramp_4x4, scale_factor=2, n_bins=3)
        assert np.array_equal(ramp_4x4, before), "Input array was modified"

    def test_is_deterministic(self, ramp_4x4):
        """Two runs over the same input produce the same table."""
        a = subgrid_table(ramp_4x4, scale_factor=2, n_bins=3)
        b = subgrid_table(ramp_4x4, scale_factor=2, n_bins=3)
        assert a.equals(b), "Repeated calls disagree"
