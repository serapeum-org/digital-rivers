"""Tests for the GDAL-backed `Terrain` ruggedness derivatives and viewshed.

Covers `Terrain.roughness` / `.tpi` / `.tri` (PB-3 — `gdal.DEMProcessing`
ruggedness modes) and `Terrain.viewshed` (PD-1 — `gdal.ViewshedGenerate`).
All fixtures are built in-memory with `Dataset.create_from_array`, are
deterministic, and touch no network or external files (except the explicit
`path=` write-to-GeoTIFF tests, which use pytest's `tmp_path`).
"""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers.terrain import Terrain

NO_DATA = -9999.0


def _make_terrain(
    arr: np.ndarray, cell_size: float = 1.0, epsg: int = 32636
) -> Terrain:
    """Wrap a 2-D array in an in-memory `Terrain` dataset.

    Args:
        arr: Elevation array; cast to float32 for the raster band.
        cell_size: Square pixel size in CRS units. Defaults to 1.0.
        epsg: Projected CRS code (a metric CRS keeps viewshed distances
            sensible). Defaults to 32636 (UTM 36N).

    Returns:
        Terrain: Dataset wrapping `arr` with no-data set to `-9999.0`.
    """
    ds = Dataset.create_from_array(
        arr.astype(np.float32, copy=True),
        top_left_corner=(0.0, 0.0),
        cell_size=cell_size,
        epsg=epsg,
        no_data_value=NO_DATA,
    )
    return Terrain(ds.raster)


@pytest.fixture(scope="function")
def flat_dem() -> Terrain:
    """A 5x5 constant-elevation DEM (z=10)."""
    return _make_terrain(np.full((5, 5), 10.0, dtype=np.float32))


@pytest.fixture(scope="function")
def peak_dem() -> Terrain:
    """A 7x7 flat DEM (z=0) with a single z=5 peak at the centre (3, 3).

    Chosen so that interior cell (1, 1) lies outside the peak's 3x3
    window (clean zero) while (2, 2) and (3, 3) include it — making the
    ruggedness indices hand-computable.
    """
    z = np.zeros((7, 7), dtype=np.float32)
    z[3, 3] = 5.0
    return _make_terrain(z)


class TestRoughness:
    """Tests for `Terrain.roughness` (GDAL `Roughness` — max−min in 3x3)."""

    def test_flat_interior_is_zero(self, flat_dem):
        """Test roughness is zero across the interior of a flat DEM.

        Test scenario:
            Every 3x3 window on a constant surface has max == min, so
            roughness is 0 at every interior cell (edges are no-data
            without compute_edges).
        """
        out = flat_dem.roughness().read_array()
        interior = out[1:-1, 1:-1]
        assert np.allclose(interior, 0.0), f"Flat interior must be 0, got {interior}"

    def test_known_value_around_peak(self, peak_dem):
        """Test roughness equals the peak height where the window sees the peak.

        Test scenario:
            With a z=5 peak on flat z=0 terrain, any 3x3 window containing
            the peak has max=5, min=0 → roughness=5. Cell (1, 1) does not
            see the peak → roughness=0.
        """
        out = peak_dem.roughness().read_array()
        assert out[3, 3] == pytest.approx(
            5.0
        ), f"Peak roughness should be 5, got {out[3, 3]}"
        assert out[2, 2] == pytest.approx(
            5.0
        ), f"Diag neighbour should be 5, got {out[2, 2]}"
        assert out[1, 1] == pytest.approx(
            0.0
        ), f"Cell away from peak should be 0, got {out[1, 1]}"

    def test_output_dtype_is_float32(self, flat_dem):
        """Test the roughness raster is stored as float32.

        Test scenario:
            GDAL DEMProcessing ruggedness modes emit float32.
        """
        assert flat_dem.roughness().read_array().dtype == np.float32

    def test_nodata_value_is_minus_9999(self, flat_dem):
        """Test the output no-data sentinel is -9999.0.

        Test scenario:
            GDAL writes -9999.0 as the ruggedness no-data value; edge cells
            carry it when compute_edges is False.
        """
        out = flat_dem.roughness()
        assert out.no_data_value[0] == pytest.approx(NO_DATA)
        assert out.read_array()[0, 0] == pytest.approx(
            NO_DATA
        ), "Edge should be no-data by default"

    def test_compute_edges_fills_edges(self, flat_dem):
        """Test compute_edges=True computes edge cells instead of no-data.

        Test scenario:
            With compute_edges=True every cell — including the boundary —
            is computed from its available partial window, so a flat DEM
            yields 0 everywhere with no no-data cells.
        """
        out = flat_dem.roughness(compute_edges=True).read_array()
        assert not np.any(out == NO_DATA), "compute_edges should leave no no-data cells"
        assert np.allclose(
            out, 0.0
        ), f"Flat DEM with edges must be all 0, got unique {np.unique(out)}"

    def test_geometry_preserved(self, peak_dem):
        """Test the result aligns spatially with the input DEM.

        Test scenario:
            Geotransform and EPSG must round-trip through DEMProcessing.
        """
        out = peak_dem.roughness()
        assert out.geotransform == peak_dem.geotransform
        assert out.epsg == peak_dem.epsg

    def test_band_index_selects_band(self):
        """Test band= selects the requested band on a multi-band raster.

        Test scenario:
            A 2-band raster whose second band is flat must yield zero
            roughness when band=1 is selected, regardless of band 0.
        """
        rough_band = np.zeros((5, 5), dtype=np.float32)
        rough_band[2, 2] = 50.0
        flat_band = np.full((5, 5), 7.0, dtype=np.float32)
        ds = Dataset.create_from_array(
            np.stack([rough_band, flat_band]),
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=32636,
            no_data_value=NO_DATA,
        )
        out = Terrain(ds.raster).roughness(band=1).read_array()
        assert np.allclose(
            out[1:-1, 1:-1], 0.0
        ), "Flat second band must give 0 roughness"

    def test_path_writes_geotiff(self, peak_dem, tmp_path):
        """Test path= writes a readable GeoTIFF that matches the in-memory result.

        Args:
            peak_dem: Terrain fixture with a central peak.
            tmp_path: pytest temporary directory.

        Test scenario:
            Writing to disk and reopening must reproduce the same array as
            the in-memory computation.
        """
        out_path = str(tmp_path / "roughness.tif")
        peak_dem.roughness(path=out_path)
        reopened = Dataset.read_file(out_path).read_array()
        assert reopened.shape == (7, 7), f"Unexpected shape on reload: {reopened.shape}"
        assert reopened[3, 3] == pytest.approx(
            5.0
        ), "Reloaded peak roughness should be 5"


class TestTPI:
    """Tests for `Terrain.tpi` (GDAL `TPI` — z minus 8-neighbour mean)."""

    def test_flat_interior_is_zero(self, flat_dem):
        """Test TPI is zero across the interior of a flat DEM.

        Test scenario:
            On constant terrain each cell equals its neighbour mean → TPI=0.
        """
        out = flat_dem.tpi().read_array()
        assert np.allclose(out[1:-1, 1:-1], 0.0), "Flat interior TPI must be 0"

    def test_peak_positive_pit_negative(self):
        """Test a peak yields positive TPI and a pit yields negative TPI.

        Test scenario:
            A z=5 peak sits above its neighbour mean (TPI>0); cells around
            it sit below their (peak-inflated) neighbour mean (TPI<0).
        """
        z = np.zeros((7, 7), dtype=np.float32)
        z[3, 3] = 5.0
        out = _make_terrain(z).tpi().read_array()
        assert out[3, 3] > 0, f"Peak TPI should be positive, got {out[3, 3]}"
        assert (
            out[2, 2] < 0
        ), f"Neighbour-of-peak TPI should be negative, got {out[2, 2]}"

    def test_known_peak_value_excludes_centre(self):
        """Test the peak's TPI equals its height (8-neighbour mean is zero).

        Test scenario:
            GDAL's TPI excludes the centre cell from the focal mean. For a
            z=5 peak whose eight neighbours are all 0, TPI = 5 - 0 = 5.
        """
        z = np.zeros((7, 7), dtype=np.float32)
        z[3, 3] = 5.0
        out = _make_terrain(z).tpi().read_array()
        assert out[3, 3] == pytest.approx(
            5.0
        ), f"Peak TPI should be 5.0, got {out[3, 3]}"

    def test_output_dtype_is_float32(self, flat_dem):
        """Test the TPI raster is float32.

        Test scenario:
            DEMProcessing TPI emits float32.
        """
        assert flat_dem.tpi().read_array().dtype == np.float32

    def test_path_writes_geotiff(self, flat_dem, tmp_path):
        """Test path= writes a readable GeoTIFF.

        Args:
            flat_dem: Flat Terrain fixture.
            tmp_path: pytest temporary directory.

        Test scenario:
            The written file reopens with the DEM's shape.
        """
        out_path = str(tmp_path / "tpi.tif")
        flat_dem.tpi(path=out_path)
        assert Dataset.read_file(out_path).read_array().shape == (5, 5)


class TestTRI:
    """Tests for `Terrain.tri` (GDAL `TRI` — Riley / Wilson variants)."""

    def test_flat_interior_is_zero(self, flat_dem):
        """Test TRI is zero across the interior of a flat DEM.

        Test scenario:
            No elevation differences → zero ruggedness everywhere.
        """
        out = flat_dem.tri().read_array()
        assert np.allclose(out[1:-1, 1:-1], 0.0), "Flat interior TRI must be 0"

    def test_default_is_riley_root_sum_square(self):
        """Test the default algorithm is Riley's root-sum-square form.

        Test scenario:
            For a z=5 peak with eight z=0 neighbours, Riley TRI is
            sqrt(8 * 5**2) = sqrt(200) ≈ 14.142 at the peak.
        """
        z = np.zeros((7, 7), dtype=np.float32)
        z[3, 3] = 5.0
        out = _make_terrain(z).tri().read_array()
        assert out[3, 3] == pytest.approx(
            np.sqrt(200.0), rel=1e-4
        ), f"Default (Riley) peak TRI should be sqrt(200), got {out[3, 3]}"

    def test_wilson_is_mean_absolute_difference(self):
        """Test algorithm='Wilson' is the mean-absolute-difference form.

        Test scenario:
            For the same peak, Wilson TRI is mean(|5-0|) over 8 neighbours
            = 5.0 at the peak — distinct from the Riley default.
        """
        z = np.zeros((7, 7), dtype=np.float32)
        z[3, 3] = 5.0
        out = _make_terrain(z).tri(algorithm="Wilson").read_array()
        assert out[3, 3] == pytest.approx(
            5.0
        ), f"Wilson peak TRI should be 5.0, got {out[3, 3]}"

    def test_riley_and_wilson_differ(self):
        """Test the two algorithms produce different rasters on rough terrain.

        Test scenario:
            Riley (root-sum-square) and Wilson (mean-abs-diff) coincide
            only in trivial cases; on a peak they must differ.
        """
        z = np.zeros((7, 7), dtype=np.float32)
        z[3, 3] = 5.0
        terrain = _make_terrain(z)
        riley = terrain.tri(algorithm="Riley").read_array()[3, 3]
        wilson = terrain.tri(algorithm="Wilson").read_array()[3, 3]
        assert riley != pytest.approx(
            wilson
        ), "Riley and Wilson TRI should differ at a peak"

    def test_output_dtype_is_float32(self, flat_dem):
        """Test the TRI raster is float32.

        Test scenario:
            DEMProcessing TRI emits float32.
        """
        assert flat_dem.tri().read_array().dtype == np.float32

    def test_path_writes_geotiff(self, flat_dem, tmp_path):
        """Test path= writes a readable GeoTIFF.

        Args:
            flat_dem: Flat Terrain fixture.
            tmp_path: pytest temporary directory.

        Test scenario:
            The written file reopens with the DEM's shape.
        """
        out_path = str(tmp_path / "tri.tif")
        flat_dem.tri(path=out_path)
        assert Dataset.read_file(out_path).read_array().shape == (5, 5)


class TestViewshed:
    """Tests for `Terrain.viewshed` (GDAL `ViewshedGenerate`)."""

    def test_flat_terrain_all_visible(self, flat_dem):
        """Test every cell of a flat DEM is visible from an observer.

        Test scenario:
            With no obstructing relief, an observer standing above a flat
            surface sees every cell, so the output is all visible_value
            (255 by default).
        """
        out = flat_dem.viewshed(observer_x=2.5, observer_y=-2.5).read_array()
        assert np.unique(out).tolist() == [
            255
        ], f"Flat DEM should be fully visible, got {np.unique(out)}"

    def test_output_is_binary_visible_invisible(self):
        """Test the output contains only the visible / invisible sentinels.

        Test scenario:
            A tall central wall hides cells behind it, so the raster holds
            both 255 (visible) and 0 (invisible) and nothing else.
        """
        z = np.zeros((9, 9), dtype=np.float32)
        z[:, 4] = 100.0
        out = _make_terrain(z).viewshed(observer_x=0.5, observer_y=-0.5).read_array()
        assert set(np.unique(out).tolist()) <= {
            0,
            255,
        }, f"Viewshed must be binary, got {np.unique(out)}"
        assert (
            out == 0
        ).any(), "A tall wall should hide some cells (expected invisible cells)"

    def test_custom_visible_invisible_values(self, flat_dem):
        """Test visible_value / invisible_value override the output coding.

        Test scenario:
            A fully-visible flat DEM should be filled entirely with the
            custom visible_value (1) and never the invisible_value (9).
        """
        out = flat_dem.viewshed(
            observer_x=2.5, observer_y=-2.5, visible_value=1.0, invisible_value=9.0
        ).read_array()
        assert np.unique(out).tolist() == [
            1
        ], f"Custom visible value not applied, got {np.unique(out)}"

    @pytest.mark.parametrize("mode", ["max", "min", "edge", "diagonal"])
    def test_all_modes_accepted(self, flat_dem, mode):
        """Test every supported cell-evaluation mode runs and returns a raster.

        Args:
            flat_dem: Flat Terrain fixture.
            mode: One of the four GDAL viewshed modes.

        Test scenario:
            Each mode string maps to a valid GVM_* constant; on flat
            terrain all return a fully-visible 5x5 raster.
        """
        out = flat_dem.viewshed(observer_x=2.5, observer_y=-2.5, mode=mode).read_array()
        assert out.shape == (5, 5), f"mode={mode!r} produced shape {out.shape}"

    def test_invalid_mode_raises(self, flat_dem):
        """Test an unknown mode raises ValueError.

        Test scenario:
            Any mode outside {max, min, edge, diagonal} is rejected before
            the GDAL call.
        """
        with pytest.raises(ValueError, match="mode must be one of"):
            flat_dem.viewshed(observer_x=2.5, observer_y=-2.5, mode="bogus")

    def test_max_distance_clips_extent(self, flat_dem):
        """Test max_distance limits the computed output window.

        Test scenario:
            GDAL clips the viewshed raster to the observer's reachable
            window when max_distance is set, so a small radius yields a
            raster smaller than the full 5x5 DEM.
        """
        full = flat_dem.viewshed(observer_x=2.5, observer_y=-2.5).read_array()
        clipped = flat_dem.viewshed(
            observer_x=2.5, observer_y=-2.5, max_distance=1.0
        ).read_array()
        assert (
            clipped.size < full.size
        ), f"max_distance should shrink output: {clipped.shape} vs {full.shape}"

    def test_path_writes_geotiff(self, flat_dem, tmp_path):
        """Test path= writes a readable GeoTIFF viewshed.

        Args:
            flat_dem: Flat Terrain fixture.
            tmp_path: pytest temporary directory.

        Test scenario:
            The written file reopens as a fully-visible 5x5 raster.
        """
        out_path = str(tmp_path / "viewshed.tif")
        flat_dem.viewshed(observer_x=2.5, observer_y=-2.5, path=out_path)
        assert Dataset.read_file(out_path).read_array().shape == (5, 5)
