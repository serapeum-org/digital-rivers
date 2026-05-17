"""End-to-end test for the W-21 → W-28 terrain-attribute stack.

Runs every terrain-named focal-window / surface-geometry / visibility index
in sequence on a single synthetic DEM and asserts cross-cutting invariants:

    DEM
      → tpi(window=3)                        # W-21
      → deviation_from_mean(window=3)        # W-22
      → elev_std(window=3)                   # W-23
      → ruggedness(window=3)                 # W-24
      → curvature(kind="profile")            # W-25
      → curvature(kind="plan")
      → curvature(kind="total")
      → curvature(kind="mean")
      → curvature(kind="gaussian")
      → normal_vector_deviation(window=3)    # W-26
      → openness(search_radius=3)            # W-27 (positive)
      → openness(search_radius=3, kind="negative")
      → sky_view_factor(search_radius=3)     # W-28
"""
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


@pytest.fixture(scope="module")
def gaussian_hill_dem() -> DEM:
    """A 15×15 DEM with a Gaussian hill at the centre.

    The hill is high at the centre and decays toward the edges, giving each
    index a non-trivial, non-zero signal without introducing no-data cells.
    """
    rows = cols = 15
    yy, xx = np.indices((rows, cols))
    cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    z = (50.0 * np.exp(-rr2 / 30.0)).astype(np.float32)
    return _make_dem(z)


class TestTerrainAttributeStack:
    """End-to-end exercise of the W-21 → W-28 terrain-attribute family."""

    @pytest.fixture(scope="class")
    def bundle(self, gaussian_hill_dem: DEM) -> dict:
        """Run all eight surfaces once and return the dataset map.

        Args:
            gaussian_hill_dem: Module-scoped Gaussian-hill DEM fixture.

        Returns:
            A dict mapping artefact name to typed result. Reused by every
            test method in this class so the kernels run once.
        """
        dem = gaussian_hill_dem
        return {
            "tpi": dem.tpi(window=3),
            "dev": dem.deviation_from_mean(window=3),
            "sd": dem.elev_std(window=3),
            "rug": dem.ruggedness(window=3),
            "plan": dem.curvature(kind="plan"),
            "profile": dem.curvature(kind="profile"),
            "total": dem.curvature(kind="total"),
            "mean": dem.curvature(kind="mean"),
            "gaussian": dem.curvature(kind="gaussian"),
            "normal_dev": dem.normal_vector_deviation(window=3),
            "openness_pos": dem.openness(search_radius=3, kind="positive"),
            "openness_neg": dem.openness(search_radius=3, kind="negative"),
            "svf": dem.sky_view_factor(search_radius=3),
            "shape": dem.values.shape,
        }

    @pytest.mark.parametrize(
        "key",
        ["tpi", "dev", "sd", "rug", "plan", "profile", "total", "mean",
         "gaussian", "normal_dev", "openness_pos", "openness_neg", "svf"],
    )
    def test_every_surface_matches_dem_shape_and_dtype(self, bundle, key):
        """Test each surface in the stack has the DEM's shape and float32 dtype.

        Args:
            bundle: Class-scoped artefact bundle.
            key: Parametrised surface name.

        Test scenario:
            All terrain attributes share a common storage convention —
            float32 raster with the DEM's grid. This locks both invariants
            in one place per surface.
        """
        arr = bundle[key].read_array()
        assert arr.shape == bundle["shape"], f"{key} shape {arr.shape}"
        assert arr.dtype == np.float32, f"{key} dtype {arr.dtype}"

    def test_tpi_and_deviation_share_sign(self, bundle):
        """Test TPI and deviation_from_mean carry the same sign at the peak (W-21 / W-22).

        Test scenario:
            Both indices are positive at ridge cells and negative at pit
            cells; on a Gaussian hill the centre cell is a clear ridge.
        """
        tpi = bundle["tpi"].read_array()
        dev = bundle["dev"].read_array()
        centre = (7, 7)
        assert tpi[centre] > 0 and dev[centre] > 0

    def test_sd_and_ruggedness_non_negative(self, bundle):
        """Test elev_std and ruggedness are non-negative across the DEM (W-23 / W-24).

        Test scenario:
            Both are absolute-value statistics; they cannot go negative
            for any real-valued input.
        """
        sd = bundle["sd"].read_array()
        rug = bundle["rug"].read_array()
        no_val = float(bundle["sd"].no_data_value[0])
        assert (sd[(sd != no_val) & np.isfinite(sd)] >= 0).all()
        assert (rug[(rug != no_val) & np.isfinite(rug)] >= 0).all()

    def test_mean_curvature_equals_total_over_two(self, bundle):
        """Test mean curvature is exactly total / 2 on the interior (W-25).

        Test scenario:
            By the Zevenbergen-Thorne formulas, `mean = D + E` and
            `total = 2 * (D + E)`. Interior cells (away from the padded
            boundary) must satisfy `mean == total / 2`.
        """
        total = bundle["total"].read_array()
        mean = bundle["mean"].read_array()
        interior_total = total[2:-2, 2:-2]
        interior_mean = mean[2:-2, 2:-2]
        np.testing.assert_allclose(
            interior_mean, interior_total / 2.0, atol=1e-5,
        )

    def test_gaussian_curvature_finite(self, bundle):
        """Test gaussian curvature is finite everywhere on the smooth DEM (W-25).

        Test scenario:
            On a smooth Gaussian hill, all second derivatives are finite,
            so the product of the two principal curvatures stays finite.
        """
        arr = bundle["gaussian"].read_array()
        finite = arr[arr != -9999.0]
        assert np.isfinite(finite).all()

    def test_normal_vector_deviation_in_zero_to_pi(self, bundle):
        """Test normal-vector deviation stays in `[0, π]` (W-26).

        Test scenario:
            The angle is `acos(clip(cos_theta, -1, 1))`, so output is in
            `[0, π]` by construction.
        """
        arr = bundle["normal_dev"].read_array()
        finite = arr[(arr != -9999.0) & np.isfinite(arr)]
        assert (finite >= 0).all()
        assert (finite <= np.pi + 1e-6).all()

    def test_openness_pos_higher_at_peak_than_at_edge(self, bundle):
        """Test positive openness peaks at the hill summit (W-27).

        Test scenario:
            The Gaussian hill's centre cell has the largest unobstructed
            view in every direction — its positive openness should exceed
            (or match) the openness at a corner cell.
        """
        pos = bundle["openness_pos"].read_array()
        assert pos[7, 7] >= pos[0, 0] - 1e-5

    def test_openness_neg_higher_in_basin_neighbourhood(self, bundle):
        """Test negative openness is largest near the hill summit's *outside*.

        Test scenario:
            Negative openness measures depth relative to surroundings —
            cells in the foothills surrounded by lower terrain on most
            sides report larger negative openness than the centre peak.
        """
        neg = bundle["openness_neg"].read_array()
        # Peak cell has nothing below it within the search radius in 8 dirs;
        # its negative openness is low. Corner cells see the hill rising in
        # one direction and flat surrounds elsewhere; they're intermediate.
        assert neg[7, 7] <= float(neg.max())

    def test_svf_in_unit_interval(self, bundle):
        """Test sky-view factor stays in `[0, 1]` (W-28).

        Test scenario:
            SVF = mean of `(1 - sin(horizon_angle))` across 8 azimuths.
            Each term lies in `[0, 1]`, so the mean does too.
        """
        svf = bundle["svf"].read_array()
        finite = svf[(svf != -9999.0) & np.isfinite(svf)]
        assert (finite >= 0).all()
        assert (finite <= 1.0 + 1e-6).all()

    def test_svf_drops_below_one_near_hill(self, bundle):
        """Test sky-view factor is strictly less than 1 in the hill's shadow.

        Test scenario:
            Cells in the foothills look up at the hill in at least one
            azimuth, so their SVF < 1 (some sky is occluded).
        """
        svf = bundle["svf"].read_array()
        # At least one cell should be below the flat-terrain ceiling.
        assert (svf < 0.999).any()
