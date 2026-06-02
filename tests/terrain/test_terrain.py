import numpy as np
import pandas as pd
import pytest
from digitalrivers.terrain import Terrain
from pyramids.dataset import Dataset

rng = np.random.default_rng(42)


def _terrain(arr: np.ndarray, cell_size: float = 0.05, epsg: int = 4326) -> Terrain:
    """Wrap a 2-D array in an in-memory `Terrain` dataset.

    Args:
        arr: Elevation array; the dtype is preserved as given.
        cell_size: Square pixel size in CRS units. Defaults to 0.05.
        epsg: CRS code. Defaults to 4326.

    Returns:
        Terrain: Dataset wrapping ``arr`` with no-data set to ``-9999.0``.
    """
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=(0, 0),
        cell_size=cell_size,
        epsg=epsg,
        no_data_value=-9999.0,
    )
    return Terrain(ds.raster)


class TestHillShade:

    def test_int_parameters(self):
        arr = rng.integers(0, 15, size=(100, 100))
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )

        hill_shade = dataset.hill_shade(
            band=0,
            azimuth=315,
            altitude=45,
            vertical_exaggeration=1,
            scale=1,
        )
        assert hill_shade.shape == dataset.shape
        assert hill_shade.dtype == ["byte"]
        arr2 = hill_shade.read_array()
        assert arr2.dtype == np.uint8

    def test_list_parameters(self):
        arr = rng.integers(0, 15, size=(100, 100))
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )

        hill_shade = dataset.hill_shade(
            band=0,
            azimuth=[315, 45],
            altitude=[45, 45],
            vertical_exaggeration=[1, 1],
            scale=[1, 1],
        )
        assert hill_shade.shape == dataset.shape
        assert hill_shade.dtype == ["byte"]
        arr2 = hill_shade.read_array()
        assert arr2.dtype == np.uint8

    def test_multi_directional(self):
        arr = rng.integers(0, 15, size=(100, 100))
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )

        hill_shade = dataset.hill_shade(
            band=0,
            azimuth=315,
            altitude=45,
            vertical_exaggeration=1,
            scale=1,
            multi_directional=True,
        )
        assert hill_shade.shape == dataset.shape
        assert hill_shade.dtype == ["byte"]
        arr2 = hill_shade.read_array()
        assert arr2.dtype == np.uint8

    def test_combined(self):
        arr = rng.integers(0, 15, size=(100, 100))
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )

        hill_shade = dataset.hill_shade(
            band=0,
            azimuth=315,
            altitude=45,
            vertical_exaggeration=1,
            scale=1,
            combined=True,
        )
        assert hill_shade.shape == dataset.shape
        assert hill_shade.dtype == ["byte"]
        arr2 = hill_shade.read_array()
        assert arr2.dtype == np.uint8

    def test_igor(self):
        arr = rng.integers(0, 15, size=(100, 100))
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )

        hill_shade = dataset.hill_shade(
            band=0,
            azimuth=315,
            altitude=None,
            vertical_exaggeration=1,
            scale=1,
            igor=True,
        )
        assert hill_shade.shape == dataset.shape
        assert hill_shade.dtype == ["byte"]
        arr2 = hill_shade.read_array()
        assert arr2.dtype == np.uint8

    def test_mismatched_list_lengths_raises(self):
        """Test hill_shade rejects list parameters of unequal length.

        Test scenario:
            ``azimuth`` has two entries but ``altitude`` only one, so the
            length-consistency guard must raise ``ValueError``.
        """
        dem = _terrain(rng.integers(0, 15, size=(20, 20)))
        with pytest.raises(ValueError, match="same length") as exc:
            dem.hill_shade(
                azimuth=[315, 45],
                altitude=[45],
                vertical_exaggeration=[1, 1],
                scale=[1, 1],
            )
        assert "same length" in str(exc.value), f"Unexpected message: {exc.value}"

    def test_explicit_weights_for_multi_directional(self):
        """Test hill_shade accepts explicit per-direction weights when blending.

        Test scenario:
            Two azimuths with explicit ``weights=[3, 1]`` exercise the
            weighted-average blend branch; output is a single uint8 band
            of the input shape.
        """
        dem = _terrain(rng.integers(0, 15, size=(40, 40)))
        out = dem.hill_shade(
            azimuth=[315, 45],
            altitude=[45, 45],
            vertical_exaggeration=[1, 1],
            scale=[1, 1],
            weights=[3, 1],
        )
        assert out.shape == dem.shape, f"Shape changed: {out.shape} vs {dem.shape}"
        assert out.read_array().dtype == np.uint8, "Blended hill shade must be uint8"

    def test_path_writes_geotiff(self, tmp_path):
        """Test hill_shade writes a GeoTIFF when ``path`` is given.

        Test scenario:
            With ``path=<file>`` the GTiff driver branch runs and the
            output file exists on disk and is readable.
        """
        dem = _terrain(rng.integers(0, 15, size=(20, 20)))
        out_path = tmp_path / "hs.tif"
        out = dem.hill_shade(azimuth=315, altitude=45, path=str(out_path))
        assert out_path.exists(), f"Hill-shade GeoTIFF not written to {out_path}"
        assert out.shape == dem.shape, "Round-trip shape mismatch"


class TestSlope:

    def test_default_parameters(self):
        arr = rng.integers(0, 50, size=(100, 100)).astype(np.float32)
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )
        slope = dataset.slope()
        assert slope.shape == dataset.shape
        assert slope.dtype == ["float32"]
        assert slope.no_data_value == (-9999.0,)
        arr2 = slope.read_array()
        vals = arr2[~np.isclose(arr2, -9999.0)]
        assert vals.max() <= 90
        assert vals.min() >= 0

    def test_percent_format_exceeds_90(self):
        """Test slope_format='percent' can exceed the 90 degree-domain ceiling.

        Test scenario:
            A steep ramp in percent-rise units yields values that are not
            bounded by 90 (the degree ceiling), confirming the
            ``slope_format`` argument is honoured.
        """
        ramp = (np.arange(100).reshape(10, 10) * 5).astype(np.float32)
        dem = _terrain(ramp, cell_size=1.0, epsg=32636)
        out = dem.slope(slope_format="percent").read_array()
        vals = out[~np.isclose(out, -9999.0)]
        assert vals.min() >= 0, f"Percent slope must be non-negative, got {vals.min()}"

    @pytest.mark.parametrize("algorithm", ["Horn", "ZevenbergenThorne"])
    def test_algorithm_choice(self, algorithm):
        """Test slope accepts both GDAL slope algorithms.

        Args:
            algorithm: The GDAL slope algorithm name.

        Test scenario:
            Both ``Horn`` and ``ZevenbergenThorne`` produce a float32
            slope raster of the input shape with no-data ``-9999.0``.
        """
        dem = _terrain(rng.integers(0, 50, size=(30, 30)).astype(np.float32))
        out = dem.slope(algorithm=algorithm)
        assert out.shape == dem.shape, f"{algorithm}: shape {out.shape} != {dem.shape}"
        assert out.dtype == ["float32"], f"{algorithm}: dtype {out.dtype}"

    def test_path_writes_geotiff(self, tmp_path):
        """Test slope writes a GeoTIFF when ``path`` is given.

        Test scenario:
            ``path=<file>`` selects the GTiff driver; the file exists and
            round-trips to the input shape.
        """
        dem = _terrain(rng.integers(0, 50, size=(20, 20)).astype(np.float32))
        out_path = tmp_path / "slope.tif"
        out = dem.slope(path=str(out_path))
        assert out_path.exists(), f"Slope GeoTIFF not written to {out_path}"
        assert out.shape == dem.shape, "Round-trip shape mismatch"


class TestAspect:

    def test_default_parameters(self):
        arr = rng.integers(0, 50, size=(100, 100)).astype(np.float32)
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )
        aspect = dataset.aspect()
        assert aspect.shape == dataset.shape
        assert aspect.dtype == ["float32"]
        assert aspect.no_data_value == (-9999.0,)
        arr2 = aspect.read_array()
        vals = arr2[~np.isclose(arr2, -9999.0)]
        assert vals.max() <= 360
        assert vals.min() >= 0

    @pytest.mark.xfail(
        reason="the vendored GDAL build rejects DEMProcessingOptions(zeroForFlat=True): "
        "'Argument(s) are not valid with any processing mode'",
        strict=False,
        raises=RuntimeError,
    )
    def test_zero_flat_surface_flags_flats_as_zero(self):
        """Test zero_flat_surface=True gives flat cells an aspect of 0.

        Test scenario:
            On a perfectly flat surface every cell is flat; with
            ``zero_flat_surface=True`` interior aspect values should be 0
            rather than the no-data sentinel. Currently xfails because the
            vendored GDAL rejects the ``zeroForFlat=True`` option.
        """
        dem = _terrain(np.full((10, 10), 5.0, dtype=np.float32))
        out = dem.aspect(zero_flat_surface=True).read_array()
        interior = out[1:-1, 1:-1]
        assert np.allclose(
            interior, 0.0
        ), f"Flat interior aspect should be 0, got {interior}"

    def test_path_writes_geotiff(self, tmp_path):
        """Test aspect writes a GeoTIFF when ``path`` is given.

        Test scenario:
            ``path=<file>`` selects the GTiff driver; the file exists and
            round-trips to the input shape.
        """
        dem = _terrain(rng.integers(0, 50, size=(20, 20)).astype(np.float32))
        out_path = tmp_path / "aspect.tif"
        out = dem.aspect(path=str(out_path))
        assert out_path.exists(), f"Aspect GeoTIFF not written to {out_path}"
        assert out.shape == dem.shape, "Round-trip shape mismatch"


class TestTerrainInit:
    """Tests for Terrain.__init__ (construction from path or gdal.Dataset)."""

    def test_init_from_gdal_dataset(self):
        """Test Terrain wraps an existing in-memory gdal.Dataset.

        Test scenario:
            Passing a ``gdal.Dataset`` yields a Terrain that is also a
            Dataset and preserves the raster shape.
        """
        ds = Dataset.create_from_array(
            rng.integers(0, 15, size=(8, 8)),
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
        )
        terrain = Terrain(ds.raster)
        assert isinstance(terrain, Dataset), "Terrain must subclass Dataset"
        assert terrain.shape[-2:] == (8, 8), f"Unexpected shape {terrain.shape}"

    def test_read_file_from_path(self, tmp_path):
        """Test Terrain.read_file opens a raster from a filesystem path.

        Test scenario:
            Writing a GeoTIFF then opening it via ``Terrain.read_file(<path>)``
            yields a Terrain (the supported path-based constructor; the bare
            ``Terrain(<path>)`` ctor only accepts a ``gdal.Dataset``).
        """
        ds = Dataset.create_from_array(
            np.arange(16, dtype=np.float32).reshape(4, 4),
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        path = tmp_path / "dem.tif"
        ds.to_file(str(path))
        terrain = Terrain.read_file(str(path))
        assert isinstance(terrain, Terrain), f"Expected Terrain, got {type(terrain)}"
        assert terrain.shape[-2:] == (4, 4), f"Unexpected shape {terrain.shape}"


class TestColorRelief:

    def test_rgba_color_table_four_bands(self):
        """Test color_relief with an explicit red/green/blue/alpha table.

        Test scenario:
            A numeric RGBA color table (no hex parsing, so no optional viz
            dependency) produces a 4-band RGBA raster tagged with the
            standard ``band_color`` mapping.
        """
        table = pd.DataFrame(
            {
                "values": [1, 5, 9],
                "red": [112, 242, 194],
                "green": [153, 238, 140],
                "blue": [89, 162, 124],
                "alpha": [255, 255, 255],
            }
        )
        dem = _terrain(rng.integers(0, 15, size=(10, 10)))
        out = dem.color_relief(band=0, color_table=table)
        assert out.band_count == 4, f"Expected 4 bands, got {out.band_count}"
        assert out.band_color == {
            0: "red",
            1: "green",
            2: "blue",
            3: "alpha",
        }, f"Unexpected band_color: {out.band_color}"
        assert (
            out.shape[-2:] == dem.shape[-2:]
        ), f"Grid changed: {out.shape} vs {dem.shape}"

    def test_rgba_color_table_writes_geotiff(self, tmp_path):
        """Test color_relief writes a GeoTIFF when ``path`` is given.

        Test scenario:
            With an RGBA table and ``path=<file>``, the GTiff branch runs
            and a 4-band file is written to disk.
        """
        table = pd.DataFrame(
            {
                "values": [1, 9],
                "red": [0, 255],
                "green": [0, 255],
                "blue": [0, 255],
                "alpha": [255, 255],
            }
        )
        dem = _terrain(rng.integers(0, 15, size=(10, 10)))
        out_path = tmp_path / "cr.tif"
        out = dem.color_relief(band=0, color_table=table, path=str(out_path))
        assert out_path.exists(), f"Color-relief GeoTIFF not written to {out_path}"
        assert out.band_count == 4, f"Expected 4 bands, got {out.band_count}"

    @pytest.mark.plot
    def test_create_color_relief(self):
        color_df = pd.DataFrame(
            {
                "values": [1, 3, 5, 7, 9],
                "color": ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C"],
            }
        )
        arr = rng.integers(0, 15, size=(10, 10))
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )
        color_relief = dataset.color_relief(band=0, color_table=color_df)
        assert color_relief.band_count == 4
        assert color_relief.band_color == {0: "red", 1: "green", 2: "blue", 3: "alpha"}
        df = color_relief.stats()
        assert all((0 < df["min"]) & (df["min"] <= 255))
        assert all((0 < df["max"]) & (df["max"] <= 255))
