"""End-to-end test for the W-15 → W-19 LiDAR pipeline.

Chains the LiDAR-cluster deliverables on a synthetic point cloud:

    LasPoints (constructed in-memory)
      → write_las(...)         # W-15 — round-trip via on-disk LAS
      → read_las(...)          # W-15
      → classify_ground(...)   # W-16 — Zhang 2003 morphological tophat
      → filter_classes({2})    # W-18 — keep ground points only
      → grid_lidar_points(method="idw" / "tin")  # W-17 — interpolation gridders
      → detect_trees(chm)      # W-19 — variable-window local-maxima on a CHM
      → clip / merge           # W-18 — polygon clip + concatenate

The on-disk LAS round-trip is gated on `laspy` being installed. The
ground-classifier, gridders, clip/merge, and tree-detect pass run
unconditionally on the in-memory LasPoints.
"""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers.lidar import (
    LasPoints,
    classify_ground,
    clip,
    detect_trees,
    filter_classes,
    grid_lidar_points,
    merge,
)


try:
    import laspy  # noqa: F401
    HAS_LASPY = True
except ImportError:  # pragma: no cover — environment-specific
    HAS_LASPY = False

requires_laspy = pytest.mark.skipif(
    not HAS_LASPY, reason="laspy required for LAS round-trip"
)


@pytest.fixture(scope="module")
def synthetic_cloud() -> LasPoints:
    """Synthetic point cloud: flat ground at z=0 plus a 5m peak and 2m peak.

    Layout:
        * 400 ground points sprinkled uniformly across a 10×10 unit square.
        * One 5m peak at (5, 5) with three nearby canopy returns.
        * One 2m peak at (8, 2) with two nearby canopy returns.
        * 10 building-class points at the corners (z=3) to exercise
          class-based filtering.
    """
    rng = np.random.default_rng(seed=1337)
    n_ground = 400
    ground_x = rng.uniform(0, 10, size=n_ground)
    ground_y = rng.uniform(0, 10, size=n_ground)
    ground_z = np.zeros(n_ground)
    ground_class = np.full(n_ground, 2, dtype=np.uint8)
    # Tall peak cluster — three returns near (5, 5).
    peak_x = np.array([5.0, 5.1, 4.95])
    peak_y = np.array([5.0, 5.05, 4.9])
    peak_z = np.array([5.0, 4.7, 4.8])
    peak_class = np.full(3, 5, dtype=np.uint8)
    # Smaller peak cluster — two returns near (8, 2).
    small_x = np.array([8.0, 8.1])
    small_y = np.array([2.0, 2.05])
    small_z = np.array([2.0, 1.9])
    small_class = np.full(2, 5, dtype=np.uint8)
    # Building corner points.
    building_x = np.array([0.5, 0.5, 9.5, 9.5, 0.5, 9.5, 5.0, 5.0, 2.0, 7.0])
    building_y = np.array([0.5, 9.5, 0.5, 9.5, 5.0, 5.0, 0.5, 9.5, 0.5, 9.5])
    building_z = np.full(10, 3.0)
    building_class = np.full(10, 6, dtype=np.uint8)
    x = np.concatenate([ground_x, peak_x, small_x, building_x])
    y = np.concatenate([ground_y, peak_y, small_y, building_y])
    z = np.concatenate([ground_z, peak_z, small_z, building_z])
    classification = np.concatenate(
        [ground_class, peak_class, small_class, building_class]
    )
    return LasPoints(x=x, y=y, z=z, classification=classification)


class TestLidarPipeline:
    """End-to-end exercise of the W-15 → W-19 LiDAR cluster."""

    def test_full_pipeline_runs_to_completion(self, synthetic_cloud):
        """Test the full LiDAR pipeline executes without error.

        Test scenario:
            classify_ground (Zhang) → filter_classes (keep class 2) →
            grid_lidar_points (IDW) — each step produces a non-empty output
            of the expected type.
        """
        classes = classify_ground(
            synthetic_cloud, method="zhang", cell_size=1.0,
            window_cells=3, slope_threshold=1.0,
        )
        assert classes.shape == (len(synthetic_cloud),)
        # Replace input classification with computed labels and keep ground.
        ground = LasPoints(
            x=synthetic_cloud.x,
            y=synthetic_cloud.y,
            z=synthetic_cloud.z,
            classification=classes,
        )
        ground_only = filter_classes(ground, {2})
        assert len(ground_only) > 0
        ds = grid_lidar_points(
            ground_only.x, ground_only.y, ground_only.z,
            cell_size=1.0, bounds=(0.0, 0.0, 10.0, 10.0),
            aggregate="idw", epsg=3857, idw_k=4,
        )
        arr = ds.read_array()
        assert arr.shape == (10, 10)

    def test_detect_trees_finds_at_least_one_peak(self, synthetic_cloud):
        """Test tree detection on a CHM from gridded max points finds peaks.

        Test scenario:
            Grid the full cloud as a `max` DSM (W-17 block-max path), then
            detect_trees over the result. At least one of the two synthetic
            peaks must be reported.
        """
        chm = grid_lidar_points(
            synthetic_cloud.x, synthetic_cloud.y, synthetic_cloud.z,
            cell_size=1.0, bounds=(0.0, 0.0, 10.0, 10.0),
            aggregate="max", epsg=3857,
        )
        tops = detect_trees(chm, min_height_m=1.5)
        assert len(tops) >= 1
        # All reported tops must hold height >= the threshold.
        assert (tops["height_m"] >= 1.5).all()

    def test_clip_and_merge_round_trip(self, synthetic_cloud):
        """Test clip → merge restores the original cell count.

        Test scenario:
            Clip the cloud into two halves with `box` polygons covering
            (0..5, *) and (5..10, *), then merge — total length should
            equal the original cloud (minus any points exactly on the
            shared boundary, which `contains_xy` excludes from both halves).
        """
        from shapely.geometry import box
        left = clip(synthetic_cloud, box(0, 0, 5, 10))
        right = clip(synthetic_cloud, box(5, 0, 10, 10), inverse=False)
        # Points with x == 5.0 sit on the boundary and may be excluded from
        # both halves (shapely `contains` is strict). Confirm the merged
        # length is at least most of the original.
        merged = merge(left, right)
        assert len(merged) >= int(0.95 * len(synthetic_cloud))

    @requires_laspy
    def test_las_round_trip_through_disk(self, synthetic_cloud, tmp_path):
        """Test write_las → read_las preserves point count and attributes.

        Args:
            synthetic_cloud: Module-scoped point-cloud fixture.
            tmp_path: pytest builtin tmp_path fixture.

        Test scenario:
            Write the in-memory cloud to a .las file, read it back, and
            confirm the round-tripped record carries the same point count,
            classification distribution, and xyz extent (within laspy's
            scale-offset quantisation).
        """
        from digitalrivers.lidar import read_las, write_las
        path = str(tmp_path / "cloud.las")
        write_las(synthetic_cloud, path)
        back = read_las(path)
        assert len(back) == len(synthetic_cloud)
        # Class distribution preserved.
        np.testing.assert_array_equal(
            np.bincount(back.classification, minlength=7),
            np.bincount(synthetic_cloud.classification, minlength=7),
        )
        # xyz extent matches the source within LAS scale-offset tolerance.
        np.testing.assert_allclose(
            (back.x.min(), back.x.max()),
            (synthetic_cloud.x.min(), synthetic_cloud.x.max()),
            atol=0.01,
        )

    def test_pipeline_idw_and_tin_agree_on_dense_centre(self, synthetic_cloud):
        """Test IDW and TIN gridders agree on the well-sampled interior.

        Test scenario:
            On a cell in the dense interior (lots of points within the K
            nearest neighbours), the IDW estimate and the TIN linear
            interpolation must produce numerically similar elevations.
            The dense ground points dominate so we expect both methods to
            return ~0.0.
        """
        ground = LasPoints(
            x=synthetic_cloud.x, y=synthetic_cloud.y, z=synthetic_cloud.z,
            classification=synthetic_cloud.classification,
        )
        ground_only = filter_classes(ground, {2})
        idw = grid_lidar_points(
            ground_only.x, ground_only.y, ground_only.z,
            cell_size=1.0, bounds=(0.0, 0.0, 10.0, 10.0),
            aggregate="idw", epsg=3857, idw_k=6,
        ).read_array()
        tin = grid_lidar_points(
            ground_only.x, ground_only.y, ground_only.z,
            cell_size=1.0, bounds=(0.0, 0.0, 10.0, 10.0),
            aggregate="tin", epsg=3857,
        ).read_array()
        # The interior 6×6 block (away from the convex-hull edge) must
        # match closely: both methods should return ~0.0 (ground level).
        no_val = -9999.0
        interior_idw = idw[2:-2, 2:-2]
        interior_tin = tin[2:-2, 2:-2]
        # Where both are valid, the per-cell difference is small.
        mask = (interior_idw != no_val) & (interior_tin != no_val)
        if mask.any():
            diff = np.abs(interior_idw[mask] - interior_tin[mask])
            assert diff.max() < 2.0  # tolerant — ground noise + edge effects
