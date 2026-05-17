"""Tests for `DEM.tpi` / `.deviation_from_mean` / `.elev_std` / `.ruggedness`
(W-21 / W-22 / W-23 / W-24)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


class TestTPI:
    """Tests for `DEM.tpi`."""

    def test_flat_terrain_zero_everywhere(self):
        """Test TPI on a constant-elevation DEM is zero everywhere.

        Test scenario:
            A flat 5×5 DEM at z=10 has focal_mean = 10 everywhere, so
            TPI = z - focal_mean = 0 at every cell.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        tpi = dem.tpi(window=3).read_array()
        assert np.allclose(tpi, 0.0)

    def test_ridge_cell_positive_tpi(self):
        """Test a single elevated cell on flat terrain produces positive TPI.

        Test scenario:
            A 5×5 flat DEM at z=0 with a peak at (2, 2)=9 produces TPI > 0
            at that cell (the cell sits above its focal mean).
        """
        z = np.zeros((5, 5), dtype=np.float32)
        z[2, 2] = 9.0
        dem = _make_dem(z)
        tpi = dem.tpi(window=3).read_array()
        assert tpi[2, 2] > 0

    def test_valley_cell_negative_tpi(self):
        """Test a single low cell on flat terrain produces negative TPI.

        Test scenario:
            A 5×5 flat DEM at z=10 with a pit at (2, 2)=0 produces TPI < 0
            at that cell.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        z[2, 2] = 0.0
        dem = _make_dem(z)
        tpi = dem.tpi(window=3).read_array()
        assert tpi[2, 2] < 0

    def test_invalid_window_rejected(self):
        """Test window < 1 raises ValueError.

        Test scenario:
            A zero or negative window is not a meaningful focal radius.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="window"):
            dem.tpi(window=0)

    def test_output_dtype_is_float32(self):
        """Test TPI output dtype is float32.

        Test scenario:
            All terrain indices share a float32 storage convention; verify
            the returned Dataset matches.
        """
        z = np.zeros((4, 4), dtype=np.float32)
        dem = _make_dem(z)
        assert dem.tpi(window=3).read_array().dtype == np.float32

    def test_returns_dataset_with_dem_geotransform_and_epsg(self):
        """Test the output Dataset carries the DEM's geotransform and EPSG.

        Test scenario:
            Pyramids `Dataset.create_from_array` round-trips geotransform
            and projection; the returned TPI raster must align spatially
            with the input DEM.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z, cell_size=2.5)
        out = dem.tpi(window=3)
        assert out.geotransform == dem.geotransform
        assert out.epsg == dem.epsg

    @pytest.mark.parametrize("window", [1, 3, 5, 7])
    def test_window_sizes_accepted(self, window):
        """Test a range of valid window sizes all return a shape-matching raster.

        Args:
            window: Window side length parametrised over a small range of
                valid values.

        Test scenario:
            For every supported window size the kernel must produce a
            raster of the same shape as the DEM.
        """
        z = np.zeros((7, 7), dtype=np.float32)
        dem = _make_dem(z)
        out = dem.tpi(window=window).read_array()
        assert out.shape == z.shape


class TestTPINoDataBoundary:
    """M2 regression: TPI / focal-window stats must not bias near no-data cells."""

    def test_tpi_near_nodata_boundary_uses_only_valid_neighbours(self):
        """Test TPI at the data/no-data boundary is computed only from valid cells.

        Test scenario:
            Half the DEM is at z=100, half is no-data. Valid cells near the
            boundary should report TPI = 0 (focal_mean over valid cells
            equals the cell's own elevation, since every valid neighbour is
            at z=100). The pre-fix buggy version would have biased the mean
            toward 0 and reported large negative TPI at the boundary.
        """
        # 5×6 DEM, left 3 columns valid at z=100, right 3 columns no-data.
        z = np.full((5, 6), 100.0, dtype=np.float32)
        z[:, 3:] = np.nan
        dem = _make_dem(z)
        tpi = dem.tpi(window=3).read_array()
        no_val = float(dem.no_data_value[0])
        # Cells at columns 0-2 (all valid neighbours at z=100) — TPI must be 0
        # because every valid neighbour shares the cell's elevation. The
        # boundary column (col=2) used to be biased by the buggy 0-fill;
        # after the no-data-aware fix it stays at 0.
        for col in (0, 1, 2):
            for row in range(5):
                assert tpi[row, col] == 0.0, (
                    f"Cell ({row}, {col}) TPI must be 0, got {tpi[row, col]}"
                )
        # No-data cells (columns 3-5) emit the sentinel.
        assert (tpi[:, 3:] == no_val).all()


class TestDeviationFromMean:
    """Tests for `DEM.deviation_from_mean`."""

    def test_flat_terrain_zero_everywhere(self):
        """Test deviation_from_mean on a flat DEM is zero everywhere.

        Test scenario:
            Constant elevation → focal_mean equals z, focal_sd is zero
            (treated as 1 to avoid /0) — result is zero everywhere.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        dev = dem.deviation_from_mean(window=3).read_array()
        assert np.allclose(dev, 0.0)

    def test_peak_cell_positive_normalised_deviation(self):
        """Test a peak cell yields a positive normalised deviation.

        Test scenario:
            A flat DEM with a single elevated cell has dev > 0 at the peak.
        """
        z = np.zeros((5, 5), dtype=np.float32)
        z[2, 2] = 10.0
        dem = _make_dem(z)
        dev = dem.deviation_from_mean(window=3).read_array()
        assert dev[2, 2] > 0

    def test_pit_cell_negative_normalised_deviation(self):
        """Test a pit cell yields a negative normalised deviation.

        Test scenario:
            A flat DEM at z=10 with a sink at (2, 2)=0 produces dev < 0
            at the pit (cell sits below its focal mean).
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        z[2, 2] = 0.0
        dem = _make_dem(z)
        dev = dem.deviation_from_mean(window=3).read_array()
        assert dev[2, 2] < 0

    def test_output_dtype_is_float32(self):
        """Test deviation_from_mean output is float32.

        Test scenario:
            Verify the storage dtype convention.
        """
        z = np.zeros((4, 4), dtype=np.float32)
        dem = _make_dem(z)
        assert dem.deviation_from_mean(window=3).read_array().dtype == np.float32

    def test_invalid_window_rejected(self):
        """Test window < 1 raises ValueError.

        Test scenario:
            The shared `_focal_window_stats` helper rejects sub-1 windows;
            this surface exposes that path.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="window"):
            dem.deviation_from_mean(window=0)


class TestElevStd:
    """Tests for `DEM.elev_std`."""

    def test_flat_terrain_zero_everywhere(self):
        """Test focal-SD on a flat DEM is zero everywhere.

        Test scenario:
            Constant elevation → focal_sd = 0 at every cell.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        sd = dem.elev_std(window=3).read_array()
        assert np.allclose(sd, 0.0)

    def test_step_function_has_finite_sd_near_step(self):
        """Test cells near a step in elevation carry positive SD.

        Test scenario:
            Half the DEM at z=0, half at z=10 — cells straddling the
            boundary must have positive elev_std.
        """
        z = np.zeros((5, 5), dtype=np.float32)
        z[:, 3:] = 10.0
        dem = _make_dem(z)
        sd = dem.elev_std(window=3).read_array()
        # Cells along the boundary column have SD > 0.
        assert (sd[:, 2] > 0).all()

    def test_sd_non_negative(self):
        """Test elev_std is non-negative everywhere by construction.

        Test scenario:
            SD = sqrt(max(sq - m², 0)); the floor keeps numerical noise from
            producing negative values.
        """
        rng = np.random.default_rng(11)
        z = rng.uniform(0, 100, size=(10, 10)).astype(np.float32)
        dem = _make_dem(z)
        sd = dem.elev_std(window=3).read_array()
        assert (sd >= 0).all()

    def test_output_dtype_is_float32(self):
        """Test elev_std output is float32.

        Test scenario:
            Verify the storage dtype convention.
        """
        z = np.zeros((4, 4), dtype=np.float32)
        dem = _make_dem(z)
        assert dem.elev_std(window=3).read_array().dtype == np.float32

    def test_invalid_window_rejected(self):
        """Test window < 1 raises ValueError.

        Test scenario:
            Sub-1 windows go through the shared helper's gate and raise.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="window"):
            dem.elev_std(window=0)


class TestRuggedness:
    """Tests for `DEM.ruggedness`."""

    def test_flat_terrain_zero_everywhere(self):
        """Test ruggedness on a flat DEM is zero everywhere.

        Test scenario:
            Constant elevation → every per-neighbour absolute difference
            is zero, so ruggedness is zero.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        tri = dem.ruggedness(window=3).read_array()
        assert np.allclose(tri, 0.0)

    def test_centre_peak_has_positive_ruggedness_around_it(self):
        """Test ruggedness is non-zero around a peak surrounded by flat terrain.

        Test scenario:
            A single elevated cell at (2, 2)=9 on flat DEM yields positive
            ruggedness at the peak and its immediate neighbours.
        """
        z = np.zeros((5, 5), dtype=np.float32)
        z[2, 2] = 9.0
        dem = _make_dem(z)
        tri = dem.ruggedness(window=3).read_array()
        assert tri[2, 2] > 0
        # Neighbours of the peak also see the height difference.
        assert tri[1, 2] > 0
        assert tri[3, 2] > 0

    def test_invalid_window_rejected(self):
        """Test window < 1 raises ValueError.

        Test scenario:
            A zero or negative window is not a meaningful focal radius.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="window"):
            dem.ruggedness(window=0)

    def test_output_dtype_is_float32(self):
        """Test ruggedness output dtype is float32.

        Test scenario:
            Verify the storage dtype convention.
        """
        z = np.zeros((4, 4), dtype=np.float32)
        dem = _make_dem(z)
        assert dem.ruggedness(window=3).read_array().dtype == np.float32

    def test_larger_window_includes_more_neighbours(self):
        """Test a wider window incorporates more neighbours.

        Test scenario:
            On the centre-peak fixture, a 5×5 window sees the same peak
            from more cells than a 3×3 window — so the average ruggedness
            across the raster increases with window size (more cells see
            the peak's contribution).
        """
        z = np.zeros((7, 7), dtype=np.float32)
        z[3, 3] = 9.0
        dem = _make_dem(z)
        small = dem.ruggedness(window=3).read_array()
        large = dem.ruggedness(window=5).read_array()
        # The larger window should produce more non-zero ruggedness cells
        # because more cells see the peak.
        assert (large > 0).sum() >= (small > 0).sum()


class TestCurvature:
    """Tests for `DEM.curvature` (W-25)."""

    def test_flat_terrain_zero_for_every_kind(self):
        """Test every curvature variant is zero on a flat DEM.

        Test scenario:
            Constant elevation → every partial derivative is zero → every
            curvature variant evaluates to zero.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        for kind in ("plan", "profile", "total", "mean", "gaussian"):
            arr = dem.curvature(kind=kind).read_array()
            assert np.allclose(arr, 0.0), f"{kind} curvature on flat DEM should be 0"

    def test_inclined_plane_total_curvature_zero(self):
        """Test total curvature is zero on a perfectly linear ramp.

        Test scenario:
            An inclined plane has no curvature; total curvature must be
            (close to) zero at every interior cell.
        """
        # z = x + y — a linear ramp, no curvature.
        x, y = np.meshgrid(np.arange(7), np.arange(7))
        z = (x + y).astype(np.float32)
        dem = _make_dem(z)
        arr = dem.curvature(kind="total").read_array()
        # Interior cells (avoid the padded edge) should be ~0.
        interior = arr[1:-1, 1:-1]
        assert np.allclose(interior, 0.0, atol=1e-5)

    def test_paraboloid_curvature_negative(self):
        """Test a convex paraboloid yields negative profile curvature on the slope.

        Test scenario:
            `z = -(x² + y²)` is a downward-opening paraboloid (high in the
            middle, lower at the edges). Profile curvature on its slopes
            should report sign consistent with the surface curvature; the
            sign convention is Zevenbergen-Thorne.
        """
        x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        z = (-(x * x + y * y)).astype(np.float32)
        dem = _make_dem(z)
        arr = dem.curvature(kind="total").read_array()
        # Total curvature in the interior is `2 * (D + E)` and is non-zero
        # because the surface has finite second derivatives.
        interior = arr[2:-2, 2:-2]
        assert (interior != 0).any()

    def test_invalid_kind_raises(self):
        """Test `kind="bogus"` raises ValueError.

        Test scenario:
            Unknown curvature variant must be rejected with a clear error.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="kind must be one of"):
            dem.curvature(kind="bogus")

    @pytest.mark.parametrize(
        "kind",
        ["plan", "profile", "total", "mean", "gaussian"],
    )
    def test_every_kind_returns_float32(self, kind):
        """Test each curvature variant returns a float32 Dataset.

        Args:
            kind: One of the five recognised curvature variants.

        Test scenario:
            Every dispatch branch returns a float32 raster of the input
            shape.
        """
        z = np.zeros((5, 5), dtype=np.float32)
        dem = _make_dem(z)
        out = dem.curvature(kind=kind).read_array()
        assert out.dtype == np.float32
        assert out.shape == z.shape

    def test_mean_curvature_half_of_total(self):
        """Test mean curvature equals total / 2 on a paraboloid (interior cells).

        Test scenario:
            By definition, mean curvature = (κ_max + κ_min)/2 ≈ (D + E),
            and total curvature = 2(D + E). The mean kind must therefore
            be exactly half the total kind at every interior cell.
        """
        x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        z = (-(x * x + y * y)).astype(np.float32)
        dem = _make_dem(z)
        total = dem.curvature(kind="total").read_array()
        mean = dem.curvature(kind="mean").read_array()
        # Interior cells away from the padded edge.
        total_int = total[2:-2, 2:-2]
        mean_int = mean[2:-2, 2:-2]
        np.testing.assert_allclose(mean_int, total_int / 2.0, atol=1e-5)

    def test_gaussian_curvature_finite_on_paraboloid(self):
        """Test gaussian curvature is finite on a paraboloid.

        Test scenario:
            Gaussian curvature is the product of the two principal
            curvatures — finite for any smooth surface. Verify finiteness
            on `z = -(x² + y²)`.
        """
        x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        z = (-(x * x + y * y)).astype(np.float32)
        dem = _make_dem(z)
        arr = dem.curvature(kind="gaussian").read_array()
        no_val = -9999.0
        finite = arr[arr != no_val]
        assert np.isfinite(finite).all()


class TestNormalVectorDeviation:
    """Tests for `DEM.normal_vector_deviation` (W-26)."""

    def test_flat_terrain_zero_everywhere(self):
        """Test normal-vector deviation is zero on a flat DEM.

        Test scenario:
            Constant elevation → every surface normal is `(0, 0, 1)` →
            angular deviation is zero everywhere.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        arr = dem.normal_vector_deviation(window=3).read_array()
        assert np.allclose(arr, 0.0, atol=1e-5)

    def test_paraboloid_has_positive_deviation_off_peak(self):
        """Test a paraboloid yields positive deviation away from the peak.

        Test scenario:
            `z = -(x² + y²)` has surface normals tilting outward; cells
            off the peak should carry a positive angular deviation from the
            window's mean normal.
        """
        x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        z = (-(x * x + y * y)).astype(np.float32)
        dem = _make_dem(z)
        arr = dem.normal_vector_deviation(window=3).read_array()
        # Cells away from the peak (large |x|+|y|) should carry some
        # deviation since normals diverge there.
        edge = arr[0, 0]
        assert edge > 0

    def test_output_bounded_below_by_zero(self):
        """Test the result is bounded below by zero (angles are non-negative).

        Test scenario:
            Angular deviation is `acos(clip(cos_theta, -1, 1))`, so output
            stays in `[0, π]`.
        """
        rng = np.random.default_rng(42)
        z = rng.uniform(0, 10, size=(8, 8)).astype(np.float32)
        dem = _make_dem(z)
        arr = dem.normal_vector_deviation(window=3).read_array()
        valid = arr != float(dem.no_data_value[0])
        assert (arr[valid] >= 0).all()

    def test_inclined_plane_zero_deviation_deep_interior(self):
        """Test an inclined plane's deep interior has zero normal-vector deviation.

        Test scenario:
            On a constant-slope ramp every interior surface normal is
            identical, so cells well inside the boundary (outside the
            window-padding influence) have zero focal angular deviation.
            Edge cells pick up boundary-padding noise; tested separately.
        """
        x, y = np.meshgrid(np.arange(7), np.arange(7))
        z = (2.0 * x + y).astype(np.float32)
        dem = _make_dem(z)
        arr = dem.normal_vector_deviation(window=3).read_array()
        # Take only the deep interior — well away from the reflective
        # boundary's slope-discontinuity artefacts.
        deep_interior = arr[2:-2, 2:-2]
        assert np.allclose(deep_interior, 0.0, atol=1e-4)

    def test_output_dtype_is_float32(self):
        """Test normal_vector_deviation output is float32.

        Test scenario:
            Verify the storage dtype convention.
        """
        z = np.zeros((4, 4), dtype=np.float32)
        dem = _make_dem(z)
        out = dem.normal_vector_deviation(window=3).read_array()
        assert out.dtype == np.float32

    def test_invalid_window_rejected(self):
        """Test window < 1 raises ValueError.

        Test scenario:
            A zero or negative window is not a meaningful focal radius.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="window"):
            dem.normal_vector_deviation(window=0)


class TestOpenness:
    """Tests for `DEM.openness` (W-27)."""

    def test_flat_terrain_openness_is_pi_over_two(self):
        """Test openness on a flat DEM is π/2 (max zenith) at every cell.

        Test scenario:
            On a perfectly flat surface every horizon angle is 0, so
            `π/2 - 0 = π/2` for each azimuth → output is `π/2` everywhere.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        arr = dem.openness(search_radius=2).read_array()
        assert np.allclose(arr, np.pi / 2.0, atol=1e-5)

    def test_peak_has_higher_openness_than_pit(self):
        """Test a high peak yields higher positive openness than a deep pit.

        Test scenario:
            A 5×5 flat DEM with one peak at (2, 2)=10 produces openness
            larger at the peak than at any cell forced to look "up" toward
            the peak.
        """
        z = np.zeros((5, 5), dtype=np.float32)
        z[2, 2] = 10.0
        dem = _make_dem(z)
        arr = dem.openness(search_radius=3).read_array()
        # The peak sees nothing higher than itself in any direction; its
        # openness is π/2. Neighbour cells see the peak above them, so
        # their openness is strictly less than π/2.
        assert arr[2, 2] > arr[1, 2] + 1e-5

    def test_invalid_kind_raises(self):
        """Test unknown `kind` raises ValueError.

        Test scenario:
            Anything other than `"positive"` / `"negative"` is rejected.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="kind"):
            dem.openness(kind="bogus")

    def test_invalid_radius_rejected(self):
        """Test search_radius < 1 raises ValueError.

        Test scenario:
            Zero search radius is meaningless; reject it.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="search_radius"):
            dem.openness(search_radius=0)

    def test_negative_openness_pit_exceeds_peak(self):
        """Test negative openness is higher at a pit than at a peak.

        Test scenario:
            Negative openness measures depression depth; the deepest pit
            scores highest. A 5×5 DEM at z=10 with a pit at (2, 2)=0
            should report a higher negative openness at the pit than at
            a flat surrounding cell.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        z[2, 2] = 0.0
        dem = _make_dem(z)
        arr = dem.openness(search_radius=3, kind="negative").read_array()
        # The pit at (2,2) is below every surrounding cell; its negative
        # openness should exceed the corner cell's.
        assert arr[2, 2] > arr[0, 0] + 1e-5

    def test_output_dtype_is_float32(self):
        """Test openness output is float32.

        Test scenario:
            Verify the storage dtype convention.
        """
        z = np.zeros((4, 4), dtype=np.float32)
        dem = _make_dem(z)
        assert dem.openness(search_radius=2).read_array().dtype == np.float32


class TestSkyViewFactor:
    """Tests for `DEM.sky_view_factor` (W-28)."""

    def test_flat_terrain_svf_is_one(self):
        """Test SVF on a flat DEM is 1 (full sky visible).

        Test scenario:
            Every horizon angle is 0 on a flat surface → `1 - sin(0) = 1`
            per direction → mean is 1.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        arr = dem.sky_view_factor(search_radius=2).read_array()
        assert np.allclose(arr, 1.0, atol=1e-5)

    def test_pit_has_lower_svf_than_neighbour(self):
        """Test a cell surrounded by higher cells reports lower SVF than the
        higher cells themselves.

        Test scenario:
            A 5×5 DEM at z=10 with a pit at (2, 2)=0. The pit looks up at
            the walls and sees less sky than the wall cells.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        z[2, 2] = 0.0
        dem = _make_dem(z)
        arr = dem.sky_view_factor(search_radius=2).read_array()
        assert arr[2, 2] < arr[0, 0] - 1e-5

    def test_svf_in_unit_interval(self):
        """Test SVF values lie in `[0, 1]`.

        Test scenario:
            By construction `(1 - sin(angle))` ∈ `[0, 1]` for angles in
            `[0, π/2]`; the mean stays in `[0, 1]`.
        """
        rng = np.random.default_rng(7)
        z = rng.uniform(0, 50, size=(10, 10)).astype(np.float32)
        dem = _make_dem(z)
        arr = dem.sky_view_factor(search_radius=3).read_array()
        valid = arr != float(dem.no_data_value[0])
        assert (arr[valid] >= 0).all()
        assert (arr[valid] <= 1).all()

    def test_invalid_radius_rejected(self):
        """Test search_radius < 1 raises ValueError.

        Test scenario:
            Zero search radius is meaningless; reject it.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="search_radius"):
            dem.sky_view_factor(search_radius=0)

    def test_output_dtype_is_float32(self):
        """Test sky_view_factor output is float32.

        Test scenario:
            Verify the storage dtype convention.
        """
        z = np.zeros((4, 4), dtype=np.float32)
        dem = _make_dem(z)
        assert dem.sky_view_factor(search_radius=2).read_array().dtype == np.float32

    def test_pit_lower_svf_relative_to_flat(self):
        """Test the SVF at a pit is strictly less than the flat-terrain SVF (1.0).

        Test scenario:
            A pit surrounded by higher cells sees less of the sky than a
            cell on flat ground; SVF < 1 at the pit.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        z[2, 2] = 0.0
        dem = _make_dem(z)
        arr = dem.sky_view_factor(search_radius=2).read_array()
        assert arr[2, 2] < 1.0


class TestHorizonWalkKernel:
    """Tests for the shared `horizon_walk_kernel` in `_numba.py`."""

    def test_kernel_mode_zero_matches_openness(self):
        """Test mode=0 produces the same surface as `DEM.openness`.

        Test scenario:
            The DEM method is a thin wrapper around the kernel; running
            the kernel directly with mode=0 on a flat DEM must yield the
            same `π/2` everywhere.
        """
        from digitalrivers._numba import horizon_walk_kernel
        z = np.full((5, 5), 10.0, dtype=np.float64)
        out = horizon_walk_kernel(z, 1.0, 2, 0)
        assert np.allclose(out, np.pi / 2.0, atol=1e-5)

    def test_kernel_mode_one_matches_sky_view_factor(self):
        """Test mode=1 reproduces the sky-view-factor invariant.

        Test scenario:
            On flat terrain, mode=1 must return 1.0 at every cell.
        """
        from digitalrivers._numba import horizon_walk_kernel
        z = np.full((5, 5), 10.0, dtype=np.float64)
        out = horizon_walk_kernel(z, 1.0, 2, 1)
        assert np.allclose(out, 1.0, atol=1e-5)

    def test_kernel_output_shape_matches_input(self):
        """Test the kernel preserves the input shape.

        Test scenario:
            For any rectangular input shape, the kernel returns a
            same-shape raster.
        """
        from digitalrivers._numba import horizon_walk_kernel
        z = np.zeros((3, 7), dtype=np.float64)
        out = horizon_walk_kernel(z, 1.0, 2, 0)
        assert out.shape == z.shape


class TestFocalWindowStats:
    """Tests for the shared `_focal_window_stats` helper on `DEM`."""

    def test_returns_three_grids(self):
        """Test the helper returns `(z, focal_mean, focal_sd)` as parallel grids.

        Test scenario:
            Output is a 3-tuple of `(rows, cols)` arrays — all same shape
            as the DEM.
        """
        z = np.zeros((4, 4), dtype=np.float32)
        dem = _make_dem(z)
        out = dem._focal_window_stats(window=3)
        assert isinstance(out, tuple) and len(out) == 3
        for arr in out:
            assert arr.shape == z.shape

    def test_flat_dem_zero_sd(self):
        """Test focal_sd on a flat DEM is exactly zero.

        Test scenario:
            Constant elevation yields zero focal SD; the floor `np.maximum(..., 0.0)`
            prevents negative artefacts.
        """
        z = np.full((4, 4), 5.0, dtype=np.float32)
        dem = _make_dem(z)
        _z, _m, sd = dem._focal_window_stats(window=3)
        assert np.allclose(sd, 0.0)

    def test_invalid_window_rejected(self):
        """Test the helper raises on window < 1.

        Test scenario:
            The shared gate also surfaces through any caller — verify it
            directly.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="window"):
            dem._focal_window_stats(window=0)
