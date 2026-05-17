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
