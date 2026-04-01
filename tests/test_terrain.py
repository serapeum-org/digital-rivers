import numpy as np
import pandas as pd
import pytest
from digitalrivers.terrain import Terrain
from pyramids.dataset import Dataset


class TestHillShade:

    def test_int_parameters(self):
        arr = np.random.randint(0, 15, size=(100, 100))
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
        arr = np.random.randint(0, 15, size=(100, 100))
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
        arr = np.random.randint(0, 15, size=(100, 100))
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
        arr = np.random.randint(0, 15, size=(100, 100))
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
        arr = np.random.randint(0, 15, size=(100, 100))
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


class TestSlope:

    def test_default_parameters(self):
        arr = np.random.randint(0, 50, size=(100, 100)).astype(np.float32)
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )
        slope = dataset.slope()
        assert slope.shape == dataset.shape
        assert slope.dtype == ["float32"]
        assert slope.no_data_value == [-9999.0]
        # check if the values are from 0 to 90
        arr2 = slope.read_array()
        vals = arr2[~np.isclose(arr2, -9999.0)]
        assert vals.max() <= 90
        assert vals.min() >= 0


class TestAspect:

    def test_default_parameters(self):
        arr = np.random.randint(0, 50, size=(100, 100)).astype(np.float32)
        dataset = Terrain(
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
            ).raster
        )
        aspect = dataset.aspect()
        assert aspect.shape == dataset.shape
        assert aspect.dtype == ["float32"]
        assert aspect.no_data_value == [-9999.0]
        # check if the values are from 0 to 90
        arr2 = aspect.read_array()
        vals = arr2[~np.isclose(arr2, -9999.0)]
        assert vals.max() <= 360
        assert vals.min() >= 0


class TestColorRelief:

    @pytest.mark.plot
    def test_create_color_relief(self):
        color_df = pd.DataFrame(
            {
                "values": [1, 3, 5, 7, 9],
                "color": ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C"],
            }
        )
        arr = np.random.randint(0, 15, size=(10, 10))
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