"""Shared pytest fixtures for the digital-rivers test suite.

Every fixture here serves the Coello river basin dataset in `tests/data/coello/`: a 14x13
grid of 4000 m cells in EPSG:32618, plus the reference arrays the DEM tests compare their
output against. Raster fixtures hand back the raw `gdal.Dataset`, not a `DEM`, so each test
decides which typed class to wrap it in.

The `import pyramids` below is order-critical and must stay the first import in this module:
it is what puts the osgeo bindings vendored inside the pyramids wheel on `sys.path`, and the
test modules that do `from osgeo import gdal` rely on that having already happened. The
comment on the import itself carries the full rationale.
"""

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
    """Coello D8 flow direction in the ESRI encoding, as a `uint16` raster.

    Reads `tests/data/coello/fd4000.tif`: a 14x13 grid whose cells carry the ESRI
    powers-of-two direction codes (1, 2, 4, ..., 128) with 255 for no-data. This is the
    foreign-encoding counterpart to `coello_flow_direction_4000`, which uses the
    digitalrivers 0-7 encoding.
    """
    return gdal.Open("tests/data/coello/fd4000.tif")


@pytest.fixture(scope="module")
def coello_dem_4000() -> gdal.Dataset:
    """Coello elevation raster at 4000 m resolution, the input of most DEM tests.

    Reads `tests/data/coello/coello-dem-4000.tif`: a 14x13 single-band `int32` grid in
    EPSG:32618 whose no-data value is `2147483647`. Wrap it in `DEM(...)` to exercise sink
    filling, slope and flow routing.
    """
    return gdal.Open("tests/data/coello/coello-dem-4000.tif")


@pytest.fixture
def coello_slope() -> np.ndarray:
    """Reference per-direction slopes for `coello_dem_4000`.

    Loads `tests/data/coello/slope.npy`: a `(13, 14, 8)` `float32` array holding one slope
    per D8 direction per cell, in the `DIR_OFFSETS` order. It is the expected output of
    `DEM._get_8_direction_slopes()`.
    """
    return np.load("tests/data/coello/slope.npy")


@pytest.fixture
def coello_max_slope() -> np.ndarray:
    """Reference steepest-downhill slope per cell for `coello_dem_4000`.

    Loads `tests/data/coello/coello-max-slope.npy`: a `(13, 14)` `float32` array that is the
    expected value grid of `DEM.slope()`.
    """
    return np.load("tests/data/coello/coello-max-slope.npy")


@pytest.fixture(scope="module")
def coello_flow_direction_4000() -> gdal.Dataset:
    """Coello D8 flow direction with the outfall forced, in the digitalrivers encoding.

    Reads `tests/data/coello/flow-direction-with-outfall.tif`: a 14x13 `int32` grid carrying
    direction indices 0-7 with `-9999` for no-data. It serves both as the expected result of
    `DEM.flow_direction(forced_direction=...)` and as the routing input of the flow
    accumulation tests.
    """
    return gdal.Open("tests/data/coello/flow-direction-with-outfall.tif")


@pytest.fixture
def flow_direction_array_cells_indices() -> np.ndarray:
    """Reference downstream-neighbour indices for `coello_dem_4000`.

    Loads `tests/data/coello/flow_direction_array.npy`: a `(13, 14, 2)` `float64` array
    giving each cell's `(column, row)` downstream neighbour, with `np.nan` where the cell has
    no downhill neighbour. It is the expected output of
    `DEM.convert_flow_direction_to_cell_indices()`.
    """
    return np.load("tests/data/coello/flow_direction_array.npy")


@pytest.fixture(scope="module")
def coello_flow_accumulation_4000() -> gdal.Dataset:
    """Reference upstream-cell counts for the Coello basin at 4000 m.

    Reads `tests/data/coello/flow-accumulation.tif`: a 14x13 `int32` grid with `-9999` for
    no-data, holding the expected result of `DEM.flow_accumulation(flow_direction)` driven by
    `coello_flow_direction_4000`.
    """
    return gdal.Open("tests/data/coello/flow-accumulation.tif")


@pytest.fixture(scope="module")
def coello_outfall() -> GeoDataFrame:
    """Point geometry of the Coello river outfall.

    Reads `tests/data/coello/coello-outfall.geojson` into a single-row `GeoDataFrame` in
    EPSG:4326. Tests reproject it onto the DEM's CRS and attach a `direction` column before
    passing it to `DEM.flow_direction(forced_direction=...)`.
    """
    return gpd.read_file("tests/data/coello/coello-outfall.geojson")
