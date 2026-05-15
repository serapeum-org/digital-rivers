import numpy as np
from osgeo import gdal
from geopandas import GeoDataFrame
from digitalrivers.dem import DEM
from digitalrivers.flow_direction import FlowDirection
from pyramids.dataset import Dataset


def test_create_dem_instance(coello_dem_4000: gdal.Dataset):
    dem = DEM(coello_dem_4000)
    assert isinstance(dem, DEM)
    assert hasattr(dem, "crs")
    assert hasattr(dem, "epsg")
    assert hasattr(dem, "band_count")


class TestProperties:
    def test_values(self, coello_dem_4000: gdal.Dataset):
        """Test if the 'values' property actually replaces the no data values with np.nan"""
        dem = DEM(coello_dem_4000)
        arr = dem.values
        assert isinstance(arr, np.ndarray)
        assert np.isnan(arr[0, 0])


def test_fill_sinks_deprecated_alias(coello_dem_4000: gdal.Dataset):
    """``DEM.fill_sinks`` now aliases ``fill_depressions(method='priority_flood', epsilon=0.1)``
    and emits a DeprecationWarning. The historical pixel-equality fixture
    (``elev_sink_free``) was computed by the old single-pass algorithm and is no
    longer reproducible — behavioural assertions take its place."""
    import warnings

    dem = DEM(coello_dem_4000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dem_filled = dem.fill_sinks()
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    assert isinstance(dem_filled, DEM)
    assert dem_filled.shape == dem.shape
    # The fill cannot reduce elevations, only raise them.
    original = dem.values
    filled = dem_filled.values
    valid = ~np.isnan(original) & ~np.isnan(filled)
    assert np.all(filled[valid] >= original[valid])
    # inplace branch returns None.
    assert dem.fill_sinks(inplace=True) is None


class TestSlope:
    def test_get_8_direction_slopes(
        self,
        coello_dem_4000: gdal.Dataset,
        coello_slope: np.ndarray,
    ):
        dem = DEM(coello_dem_4000)
        slope = dem._get_8_direction_slopes()
        assert isinstance(slope, np.ndarray)
        assert np.allclose(slope, coello_slope, equal_nan=True)

    def test_slope(
        self,
        coello_dem_4000: gdal.Dataset,
        coello_max_slope: np.ndarray,
    ):
        dem = DEM(coello_dem_4000)
        slope = dem.slope()
        assert isinstance(slope, DEM)
        assert slope.shape == dem.shape
        assert np.allclose(slope.values, coello_max_slope, equal_nan=True)


class TestFlowDirection:
    def test_flow_direction(
        self,
        coello_dem_4000: gdal.Dataset,
        coello_outfall: GeoDataFrame,
        coello_flow_direction_4000: gdal.Dataset,
    ):
        """Test if the flow direction is calculated correctly.
        The test sets the flow direction of the outfall to 6 (east) and checks if the flow direction
        """
        dem = DEM(coello_dem_4000)
        flow_direction_validation = DEM(coello_flow_direction_4000)
        coello_outfall.to_crs(dem.epsg, inplace=True)
        coello_outfall["direction"] = 6
        fd = dem.flow_direction(forced_direction=coello_outfall)
        # Strict-type assertion: must be exactly FlowDirection, not a DEM
        # (today's behaviour without P1) and not just any Dataset subclass.
        assert type(fd) is FlowDirection
        assert fd.routing == "d8"
        assert fd.encoding == "digitalrivers"
        assert fd.no_data_value == (Dataset.default_no_data_value,)
        assert fd.dtype == ["int32"]
        arr = fd.read_array()
        # check that the no data value is set correctly in the array.
        assert arr[0, 0] == Dataset.default_no_data_value
        arr_validation = flow_direction_validation.read_array()
        assert np.array_equal(arr, arr_validation, equal_nan=True)


def test_flow_accumulation(
    coello_dem_4000: gdal.Dataset,
    coello_flow_direction_4000: gdal.Dataset,
    coello_flow_accumulation_4000: gdal.Dataset,
):
    dem = DEM(coello_dem_4000)
    flow_direction = DEM(coello_flow_direction_4000)
    acc = dem.flow_accumulation(flow_direction)
    assert isinstance(acc, Dataset)
    assert acc.no_data_value == (Dataset.default_no_data_value,)
    assert acc.dtype == ["int32"]
    arr = acc.read_array()
    assert arr[0, 0] == Dataset.default_no_data_value
    arr_validation = coello_flow_accumulation_4000.ReadAsArray()
    assert np.array_equal(arr, arr_validation, equal_nan=True)


def test_flow_direction_array_cells_indices(
    coello_dem_4000: gdal.Dataset,
    flow_direction_array_cells_indices: np.ndarray,
):
    dem = DEM(coello_dem_4000)
    fd_cell = dem.convert_flow_direction_to_cell_indices()
    assert isinstance(fd_cell, np.ndarray)
    assert fd_cell.shape == (dem.rows, dem.columns, 2)
    assert np.array_equal(fd_cell, flow_direction_array_cells_indices, equal_nan=True)
