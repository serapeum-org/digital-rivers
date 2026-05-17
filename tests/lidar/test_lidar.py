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


class TestInterpolationGridders:
    """Tests for the spatial-interpolation aggregates (idw / nn / tin / rbf)."""

    def test_idw_returns_finite_grid(self):
        """Test IDW produces a finite grid of the requested shape.

        Test scenario:
            Three points cover a 2×2 grid; IDW interpolation must fill
            every cell with a finite z value.
        """
        xs = np.array([0.0, 1.0, 0.5])
        ys = np.array([0.0, 0.0, 1.0])
        zs = np.array([10.0, 20.0, 15.0])
        ds = grid_lidar_points(
            xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 2.0),
            aggregate="idw", epsg=3857, idw_k=3,
        )
        arr = ds.read_array()
        assert arr.shape == (2, 2)
        assert np.isfinite(arr).all()

    def test_idw_exact_point_match(self):
        """Test IDW at a cell centred on a point returns that point's z.

        Test scenario:
            A point exactly at the cell centre triggers the exact-hit branch:
            the cell takes the point's z (rather than IDW-blending its
            neighbours).
        """
        # Cell centres at (0.5, 0.5) and (1.5, 0.5) for a 2×1 grid with
        # cell_size=1.0 and bounds (0,0,2,1). Place a point exactly at
        # (0.5, 0.5).
        xs = np.array([0.5, 1.9])
        ys = np.array([0.5, 0.5])
        zs = np.array([42.0, 5.0])
        ds = grid_lidar_points(
            xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 1.0),
            aggregate="idw", epsg=3857, idw_k=2,
        )
        arr = ds.read_array()
        assert abs(float(arr[0, 0]) - 42.0) < 1e-5

    def test_nn_takes_nearest_z(self):
        """Test nearest-neighbour gridding takes the closest point's z.

        Test scenario:
            Two points at known locations; each cell must take the z of the
            spatially-closest point.
        """
        xs = np.array([0.5, 1.5])
        ys = np.array([0.5, 0.5])
        zs = np.array([100.0, 200.0])
        ds = grid_lidar_points(
            xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 1.0),
            aggregate="nn", epsg=3857,
        )
        arr = ds.read_array()
        assert float(arr[0, 0]) == 100.0
        assert float(arr[0, 1]) == 200.0

    def test_tin_interpolates_inside_hull(self):
        """Test TIN interpolation produces finite values inside the convex hull.

        Test scenario:
            A triangle of three points; the cell centre inside the hull
            yields a finite interpolated z.
        """
        xs = np.array([0.0, 2.0, 1.0])
        ys = np.array([0.0, 0.0, 2.0])
        zs = np.array([0.0, 0.0, 6.0])
        ds = grid_lidar_points(
            xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 2.0),
            aggregate="tin", epsg=3857,
        )
        arr = ds.read_array()
        # The cell-centre (1, 1) is at the centroid of the triangle; the
        # interpolant should be 0+0+6 / 3 = 2.0 (linear interpolation).
        assert np.isfinite(arr).any()

    def test_rbf_finite_grid_interpolation(self):
        """Test RBF interpolation populates the full grid with finite values.

        Test scenario:
            Five points across a 3×3 grid; the thin-plate-spline RBF
            interpolant should produce finite z at every cell.
        """
        xs = np.array([0.0, 2.0, 0.0, 2.0, 1.0])
        ys = np.array([0.0, 0.0, 2.0, 2.0, 1.0])
        zs = np.array([0.0, 10.0, 10.0, 0.0, 5.0])
        ds = grid_lidar_points(
            xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 3.0, 3.0),
            aggregate="rbf", epsg=3857,
        )
        arr = ds.read_array()
        assert np.isfinite(arr).all()

    def test_count_aggregate_returns_density(self):
        """Test the count aggregate returns per-cell point density.

        Test scenario:
            Three points all in the same cell return 3 at that cell; empty
            cells return 0 (which becomes the no-data sentinel via dtype
            coercion in the wider helper).
        """
        xs = np.array([0.1, 0.2, 0.3])
        ys = np.array([0.1, 0.2, 0.3])
        zs = np.array([1.0, 2.0, 3.0])
        ds = grid_lidar_points(
            xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 1.0, 1.0),
            aggregate="count", epsg=3857,
        )
        arr = ds.read_array()
        assert float(arr[0, 0]) == 3.0


class TestDetectTrees:
    """Tests for `lidar.detect_trees` (local-maxima on a CHM)."""

    def _make_chm(self, arr: np.ndarray, cell_size: float = 1.0):
        from pyramids.dataset import Dataset
        ds = Dataset.create_from_array(
            arr.astype(np.float32), top_left_corner=(0.0, 0.0),
            cell_size=cell_size, epsg=3857, no_data_value=-9999.0,
        )
        return ds

    def test_single_peak_detected(self):
        """Test a single elevated cell on a flat CHM is detected as a tree top.

        Test scenario:
            5×5 CHM with z=0 everywhere except z=15 at (2, 2). detect_trees
            should return exactly one top at that cell.
        """
        from digitalrivers.lidar import detect_trees
        z = np.zeros((5, 5), dtype=np.float32)
        z[2, 2] = 15.0
        chm = self._make_chm(z)
        gdf = detect_trees(chm, min_height_m=2.0)
        assert len(gdf) == 1
        row = gdf.iloc[0]
        assert int(row["row"]) == 2 and int(row["col"]) == 2
        assert float(row["height_m"]) == 15.0

    def test_below_threshold_skipped(self):
        """Test cells below `min_height_m` are not reported.

        Test scenario:
            A CHM with a 1.5 m peak below the 2.0 m threshold returns an
            empty GeoDataFrame.
        """
        from digitalrivers.lidar import detect_trees
        z = np.zeros((3, 3), dtype=np.float32)
        z[1, 1] = 1.5
        chm = self._make_chm(z)
        gdf = detect_trees(chm, min_height_m=2.0)
        assert len(gdf) == 0

    def test_geometry_at_cell_centre(self):
        """Test the geometry of each tree top sits at the cell centre.

        Test scenario:
            With cell_size=1, top_left=(0, 0), the tree top at (row, col) =
            (2, 2) maps to world coordinates (2.5, -2.5).
        """
        from digitalrivers.lidar import detect_trees
        z = np.zeros((5, 5), dtype=np.float32)
        z[2, 2] = 10.0
        chm = self._make_chm(z)
        gdf = detect_trees(chm, min_height_m=2.0)
        pt = gdf.iloc[0]["geometry"]
        assert abs(pt.x - 2.5) < 1e-5
        assert abs(pt.y - (-2.5)) < 1e-5

    def test_two_well_separated_peaks_both_detected(self):
        """Test two peaks beyond each other's window are both detected.

        Test scenario:
            A large CHM with peaks at (2, 2) and (10, 10) are both reported
            when the variable window radius is small enough that they don't
            overlap.
        """
        from digitalrivers.lidar import detect_trees
        z = np.zeros((15, 15), dtype=np.float32)
        z[2, 2] = 5.0
        z[10, 10] = 5.0
        chm = self._make_chm(z)
        # radius_fn keeps the window tiny so the two peaks don't compete.
        gdf = detect_trees(
            chm, min_height_m=2.0, radius_fn=lambda h: 1.0,
        )
        assert len(gdf) == 2


class TestClassifyGround:
    """Tests for `lidar.classify_ground` (Zhang 2003 tophat filter)."""

    def test_flat_terrain_all_ground(self):
        """Test a perfectly flat point cloud labels every point as ground.

        Test scenario:
            All points share z=0; the tophat opening leaves the grid flat,
            so every point sits at the opening height and is ground (class 2).
        """
        from digitalrivers.lidar import classify_ground
        n = 50
        rng = np.random.default_rng(42)
        xs = rng.uniform(0, 10, size=n)
        ys = rng.uniform(0, 10, size=n)
        zs = np.zeros(n)
        pts = LasPoints(x=xs, y=ys, z=zs)
        out = classify_ground(pts, cell_size=1.0)
        assert (out == 2).all()

    def test_tall_outlier_marked_non_ground(self):
        """Test a single tall point above flat terrain is classified as non-ground.

        Test scenario:
            A dense set of ground points at z=0 plus one outlier at z=10.
            The opening at the outlier's cell stays near 0, so the outlier
            exceeds the threshold and gets class 1 (non-ground).
        """
        from digitalrivers.lidar import classify_ground
        rng = np.random.default_rng(42)
        xs = np.concatenate([rng.uniform(0, 10, size=200), [5.0]])
        ys = np.concatenate([rng.uniform(0, 10, size=200), [5.0]])
        zs = np.concatenate([np.zeros(200), [10.0]])
        pts = LasPoints(x=xs, y=ys, z=zs)
        out = classify_ground(pts, cell_size=1.0, slope_threshold=1.0)
        assert out[-1] == 1, f"Outlier must be non-ground; got {out[-1]}"

    def test_axelsson_raises_not_implemented(self):
        """Test the Axelsson path raises NotImplementedError with a helpful message.

        Test scenario:
            classify_ground(method='axelsson') must signal the
            unimplemented branch rather than fall through silently.
        """
        from digitalrivers.lidar import classify_ground
        pts = LasPoints(
            x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
        )
        with pytest.raises(NotImplementedError, match="Axelsson"):
            classify_ground(pts, method="axelsson")

    def test_unknown_method_raises(self):
        """Test an unknown method= raises ValueError.

        Test scenario:
            method='bogus' must be rejected with a clear error.
        """
        from digitalrivers.lidar import classify_ground
        pts = LasPoints(
            x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
        )
        with pytest.raises(ValueError, match="method must be"):
            classify_ground(pts, method="bogus")

    def test_invalid_geometry_args_rejected(self):
        """Test invalid cell_size / window_cells / slope_threshold raise.

        Test scenario:
            Negative cell_size, window_cells < 1, and negative threshold
            must all be rejected.
        """
        from digitalrivers.lidar import classify_ground
        pts = LasPoints(
            x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
        )
        with pytest.raises(ValueError, match="cell_size"):
            classify_ground(pts, cell_size=0)
        with pytest.raises(ValueError, match="window_cells"):
            classify_ground(pts, window_cells=0)
        with pytest.raises(ValueError, match="slope_threshold"):
            classify_ground(pts, slope_threshold=-0.1)


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

    def _placeholder(self):  # unused — anchor for the class above
        pass


class TestClipMergeFilter:
    """Tests for `clip`, `merge`, `filter_classes`."""

    def test_clip_keeps_only_points_inside_polygon(self):
        """Test clip keeps only points inside the polygon.

        Test scenario:
            With a square polygon covering x,y ∈ [0, 1] and points at
            (0.5, 0.5) and (2.0, 2.0), clip retains only the first.
        """
        from shapely.geometry import box
        from digitalrivers.lidar import clip
        pts = LasPoints(
            x=np.array([0.5, 2.0]),
            y=np.array([0.5, 2.0]),
            z=np.array([1.0, 2.0]),
        )
        polygon = box(0, 0, 1, 1)
        out = clip(pts, polygon)
        assert len(out) == 1
        assert out.x[0] == 0.5

    def test_clip_inverse_keeps_only_points_outside(self):
        """Test clip(inverse=True) drops points inside, keeps those outside.

        Test scenario:
            Inverse clip on the same fixture drops the inside point and keeps
            the outside one.
        """
        from shapely.geometry import box
        from digitalrivers.lidar import clip
        pts = LasPoints(
            x=np.array([0.5, 2.0]),
            y=np.array([0.5, 2.0]),
            z=np.array([1.0, 2.0]),
        )
        polygon = box(0, 0, 1, 1)
        out = clip(pts, polygon, inverse=True)
        assert len(out) == 1
        assert out.x[0] == 2.0

    def test_merge_concatenates_two_clouds(self):
        """Test merge stacks two `LasPoints` end-to-end.

        Test scenario:
            Two single-point clouds merge into a two-point cloud preserving
            the inputs' order.
        """
        from digitalrivers.lidar import merge
        a = LasPoints(
            x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
        )
        b = LasPoints(
            x=np.array([1.0]), y=np.array([1.0]), z=np.array([1.0]),
        )
        out = merge(a, b)
        assert len(out) == 2
        np.testing.assert_array_equal(out.x, [0.0, 1.0])

    def test_merge_rejects_empty_input(self):
        """Test merge with no clouds raises ValueError.

        Test scenario:
            Calling merge() with no arguments must raise.
        """
        from digitalrivers.lidar import merge
        with pytest.raises(ValueError, match="at least one"):
            merge()

    def test_merge_drops_optional_field_when_only_some_inputs_have_it(self):
        """Test merge drops intensity when not every input carries it.

        Test scenario:
            One input has intensity, the other does not — the merged
            cloud's intensity must be empty (size 0).
        """
        from digitalrivers.lidar import merge
        a = LasPoints(
            x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
            intensity=np.array([100], dtype=np.uint16),
        )
        b = LasPoints(
            x=np.array([1.0]), y=np.array([1.0]), z=np.array([1.0]),
        )
        out = merge(a, b)
        assert out.intensity.size == 0

    def test_filter_classes_keeps_only_ground(self):
        """Test filter_classes(classes={2}) keeps only ground points.

        Test scenario:
            A cloud mixing ground (2) and vegetation (5) points returns
            only the ground subset when filtered on {2}.
        """
        from digitalrivers.lidar import filter_classes
        pts = LasPoints(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([0.0, 1.0, 2.0]),
            z=np.array([0.0, 1.0, 2.0]),
            classification=np.array([2, 5, 2], dtype=np.uint8),
        )
        out = filter_classes(pts, {2})
        assert len(out) == 2
        np.testing.assert_array_equal(out.classification, [2, 2])

    def test_filter_classes_rejects_cloud_without_classification(self):
        """Test filter_classes raises when classification is empty.

        Test scenario:
            A cloud with no classification data cannot be filtered; raise.
        """
        from digitalrivers.lidar import filter_classes
        pts = LasPoints(
            x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
        )
        with pytest.raises(ValueError, match="no classification"):
            filter_classes(pts, {2})


@requires_laspy
class TestLasIORoundTrip:
    """Round-trip tests gated on laspy availability."""

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
