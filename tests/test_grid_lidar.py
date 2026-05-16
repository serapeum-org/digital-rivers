"""Tests for ``grid_lidar_points`` (P34 backfill — gridding half)."""
from __future__ import annotations

import numpy as np
import pytest

from digitalrivers._phase4_stubs import grid_lidar_points


def test_grid_min_picks_lowest_per_cell():
    # Two points in cell (0,0): z=5 and z=2; min → 2.
    # One point in cell (0,1): z=4.
    xs = np.array([0.1, 0.2, 1.1])
    ys = np.array([0.1, 0.2, 0.1])
    zs = np.array([5.0, 2.0, 4.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 1.0),
        aggregate="min", epsg=3857,
    )
    arr = ds.read_array()
    # Row 0 contains both cells; min in cell (0,0)=2, cell (0,1)=4.
    assert arr.shape == (1, 2)
    assert float(arr[0, 0]) == 2.0
    assert float(arr[0, 1]) == 4.0


def test_grid_max_picks_highest_per_cell():
    xs = np.array([0.1, 0.2, 1.1])
    ys = np.array([0.1, 0.2, 0.1])
    zs = np.array([5.0, 2.0, 4.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 1.0),
        aggregate="max", epsg=3857,
    )
    arr = ds.read_array()
    assert float(arr[0, 0]) == 5.0


def test_grid_mean_averages_per_cell():
    xs = np.array([0.1, 0.2])
    ys = np.array([0.1, 0.2])
    zs = np.array([4.0, 6.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 1.0, 1.0),
        aggregate="mean", epsg=3857,
    )
    arr = ds.read_array()
    assert float(arr[0, 0]) == 5.0


def test_grid_median_per_cell():
    xs = np.array([0.1, 0.2, 0.3])
    ys = np.array([0.1, 0.2, 0.3])
    zs = np.array([1.0, 5.0, 9.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 1.0, 1.0),
        aggregate="median", epsg=3857,
    )
    arr = ds.read_array()
    assert float(arr[0, 0]) == 5.0


def test_grid_empty_cells_get_nodata():
    # Single point in a 2x2 grid leaves three empty cells.
    xs = np.array([0.1])
    ys = np.array([0.1])
    zs = np.array([3.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 2.0),
        aggregate="min", epsg=3857,
    )
    arr = ds.read_array()
    no_data = ds.no_data_value[0]
    # 3 of 4 cells should be no-data.
    assert (arr == no_data).sum() == 3


def test_grid_invalid_aggregate_rejected():
    xs = ys = zs = np.array([0.0])
    with pytest.raises(ValueError, match="aggregate"):
        grid_lidar_points(xs, ys, zs, cell_size=1.0, aggregate="bogus")


def test_grid_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="same length"):
        grid_lidar_points(
            np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0]),
            cell_size=1.0,
        )


def test_grid_defaults_to_bounds_from_points():
    xs = np.array([0.0, 2.0])
    ys = np.array([0.0, 2.0])
    zs = np.array([1.0, 2.0])
    ds = grid_lidar_points(xs, ys, zs, cell_size=1.0, aggregate="min")
    # Bounds = (0,0,2,2) → 2x2 grid.
    arr = ds.read_array()
    assert arr.shape == (2, 2)


def test_grid_min_keeps_first_when_second_higher():
    """Cover the 'else' branch where a later point is NOT lower than the
    existing min."""
    xs = np.array([0.1, 0.2])
    ys = np.array([0.1, 0.2])
    zs = np.array([1.0, 5.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 1.0, 1.0),
        aggregate="min", epsg=3857,
    )
    arr = ds.read_array()
    assert float(arr[0, 0]) == 1.0


def test_grid_max_keeps_first_when_second_lower():
    """Cover the dual 'else' branch in the max-aggregate loop."""
    xs = np.array([0.1, 0.2])
    ys = np.array([0.1, 0.2])
    zs = np.array([5.0, 1.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 1.0, 1.0),
        aggregate="max", epsg=3857,
    )
    arr = ds.read_array()
    assert float(arr[0, 0]) == 5.0


def test_grid_single_point_input():
    """Single point produces a 1x1 cell with that point's z."""
    xs = np.array([0.5])
    ys = np.array([0.5])
    zs = np.array([42.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 1.0, 1.0),
        aggregate="min", epsg=3857,
    )
    arr = ds.read_array()
    assert arr.shape == (1, 1)
    assert float(arr[0, 0]) == 42.0


def test_grid_points_exactly_on_bounds_clipped_into_grid():
    """A point at the lower-right corner clips into the last cell."""
    xs = np.array([2.0])
    ys = np.array([0.0])
    zs = np.array([3.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 1.0),
        aggregate="min", epsg=3857,
    )
    arr = ds.read_array()
    # Bounded grid is 1x2; the point (x=2, y=0) at the SE corner clips to
    # column 1 (the rightmost valid column).
    assert arr.shape == (1, 2)
    assert float(arr[0, 1]) == 3.0


def test_grid_returns_no_data_sentinel():
    """The returned Dataset advertises -9999.0 as its no-data sentinel."""
    xs = np.array([0.5])
    ys = np.array([0.5])
    zs = np.array([1.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 2.0),
        aggregate="min",
    )
    assert float(ds.no_data_value[0]) == -9999.0


def test_grid_dense_cell_aggregates_many_points():
    """1,000 points dropped into a single cell aggregate correctly."""
    rng = np.random.default_rng(seed=1337)
    xs = rng.uniform(0.1, 0.9, size=1000)
    ys = rng.uniform(0.1, 0.9, size=1000)
    zs = rng.uniform(-100.0, 100.0, size=1000)
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 1.0, 1.0),
        aggregate="mean", epsg=3857,
    )
    arr = ds.read_array()
    assert abs(float(arr[0, 0]) - float(zs.mean())) < 0.5
