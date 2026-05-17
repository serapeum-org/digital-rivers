"""Tests for `digitalrivers.lidar` (grid_lidar_points + umbrella stub)."""
from __future__ import annotations

import numpy as np
import pytest

from digitalrivers.lidar import LasPoints, grid_lidar_points

# Gridding tests run without laspy. The LAS I/O tests (below the
# `requires_laspy` marker) skip when it's missing.
try:
    import laspy  # noqa: F401
    HAS_LASPY = True
except ImportError:  # pragma: no cover — environment-specific
    HAS_LASPY = False

requires_laspy = pytest.mark.skipif(
    not HAS_LASPY, reason="laspy required for LAS I/O tests"
)


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


class TestLasPoints:
    """Tests for the `LasPoints` record class."""

    def test_constructor_requires_matching_lengths(self):
        """Test mismatched x/y/z lengths raise ValueError.

        Test scenario:
            Constructing LasPoints with arrays of different lengths must be
            rejected.
        """
        with pytest.raises(ValueError, match="same length"):
            LasPoints(
                x=np.array([0.0]),
                y=np.array([0.0, 1.0]),
                z=np.array([0.0]),
            )

    def test_len_returns_point_count(self):
        """Test `len(LasPoints)` equals the number of points.

        Test scenario:
            A cloud with 3 points reports length 3.
        """
        pts = LasPoints(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([0.0, 1.0, 2.0]),
            z=np.array([0.0, 1.0, 2.0]),
        )
        assert len(pts) == 3

    def test_subset_filters_all_fields_in_sync(self):
        """Test `subset` retains the same indices across every field.

        Test scenario:
            With intensity and classification populated, a bool mask must
            select the same rows in all populated arrays.
        """
        pts = LasPoints(
            x=np.array([0.0, 1.0, 2.0, 3.0]),
            y=np.array([0.0, 1.0, 2.0, 3.0]),
            z=np.array([0.0, 1.0, 2.0, 3.0]),
            intensity=np.array([10, 20, 30, 40], dtype=np.uint16),
            classification=np.array([2, 5, 2, 6], dtype=np.uint8),
        )
        mask = np.array([True, False, True, False])
        sub = pts.subset(mask)
        assert len(sub) == 2
        np.testing.assert_array_equal(sub.x, [0.0, 2.0])
        np.testing.assert_array_equal(sub.intensity, [10, 30])
        np.testing.assert_array_equal(sub.classification, [2, 2])

    def test_subset_rejects_wrong_shape_mask(self):
        """Test subset rejects a mask of the wrong length.

        Test scenario:
            A mask shorter or longer than the point count must raise.
        """
        pts = LasPoints(
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 1.0]),
            z=np.array([0.0, 1.0]),
        )
        with pytest.raises(ValueError, match="shape"):
            pts.subset(np.array([True, True, True]))


@requires_laspy
class TestLasIO:
    """Tests for `read_las` / `write_las`. Skipped when laspy is missing."""

    def test_write_then_read_roundtrip(self, tmp_path):
        """Test writing a LasPoints to disk and reading it back yields equivalent data.

        Args:
            tmp_path: pytest builtin fixture providing a temporary directory.

        Test scenario:
            A small LasPoints with intensity and classification is written
            to `.las` and read back. Coordinates / intensity / classification
            must match the source within laspy's quantisation precision.
        """
        from digitalrivers.lidar import read_las, write_las
        pts = LasPoints(
            x=np.array([100.0, 101.5, 103.25]),
            y=np.array([200.0, 200.5, 201.0]),
            z=np.array([50.0, 51.0, 52.5]),
            intensity=np.array([100, 200, 300], dtype=np.uint16),
            classification=np.array([2, 2, 5], dtype=np.uint8),
        )
        path = str(tmp_path / "roundtrip.las")
        write_las(pts, path)
        back = read_las(path)
        # LAS stores coordinates as scaled int32s, so allow a small tolerance
        # on xyz (typically ≤ 0.001).
        np.testing.assert_allclose(back.x, pts.x, atol=0.01)
        np.testing.assert_allclose(back.y, pts.y, atol=0.01)
        np.testing.assert_allclose(back.z, pts.z, atol=0.01)
        np.testing.assert_array_equal(back.intensity, pts.intensity)
        np.testing.assert_array_equal(back.classification, pts.classification)

    def test_read_las_returns_las_points(self, tmp_path):
        """Test read_las returns a LasPoints instance.

        Args:
            tmp_path: pytest tmp_path fixture.

        Test scenario:
            After writing a tiny LAS file, read_las returns a LasPoints
            with the expected length.
        """
        from digitalrivers.lidar import read_las, write_las
        pts = LasPoints(
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 1.0]),
            z=np.array([0.0, 1.0]),
        )
        path = str(tmp_path / "tiny.las")
        write_las(pts, path)
        back = read_las(path)
        assert isinstance(back, LasPoints)
        assert len(back) == 2
