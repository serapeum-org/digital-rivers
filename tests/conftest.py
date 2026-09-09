# Import pyramids FIRST — before anything else, and keep it first.
# The osgeo bindings live inside the pyramids wheel; importing pyramids is
# what puts `_vendor/osgeo` on sys.path. Ten test modules under tests/ do
# `from osgeo import gdal` before they import digitalrivers, so this line is
# what makes their imports resolve. Nothing installs a top-level osgeo any
# more, so sorting this below the osgeo import breaks collection everywhere.
import pyramids  # noqa: F401  # isort:skip

from typing import Dict
import pytest
import numpy as np
from osgeo import gdal
from geopandas import GeoDataFrame
import geopandas as gpd


@pytest.fixture(scope="module")
def coello_df_4000() -> gdal.Dataset:
    return gdal.Open("tests/data/coello/fd4000.tif")


@pytest.fixture(scope="module")
def coello_dem_4000() -> gdal.Dataset:
    return gdal.Open("tests/data/coello/coello-dem-4000.tif")


@pytest.fixture
def coello_slope() -> np.ndarray:
    return np.load("tests/data/coello/slope.npy")


@pytest.fixture
def coello_max_slope() -> np.ndarray:
    return np.load("tests/data/coello/coello-max-slope.npy")


@pytest.fixture(scope="module")
def coello_flow_direction_4000() -> gdal.Dataset:
    return gdal.Open("tests/data/coello/flow-direction-with-outfall.tif")


@pytest.fixture
def flow_direction_array_cells_indices() -> np.ndarray:
    return np.load("tests/data/coello/flow_direction_array.npy")


@pytest.fixture(scope="module")
def coello_flow_accumulation_4000() -> gdal.Dataset:
    return gdal.Open("tests/data/coello/flow-accumulation.tif")


@pytest.fixture(scope="module")
def coello_outfall() -> GeoDataFrame:
    """Point Geometry of the Coello river outfall"""
    return gpd.read_file("tests/data/coello/coello-outfall.geojson")
