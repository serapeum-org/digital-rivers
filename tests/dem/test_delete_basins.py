"""Comprehensive tests for DEM.delete_basins.

Tests cover the static method that filters a basin-ID raster to keep
only the first (lowest-valued) basin, replacing all others with the
no-data value.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers.dem import DEM

NO_DATA = -9999


@pytest.fixture()
def make_basin_dataset():
    """Factory fixture that creates a Dataset from a 2-D basin-ID array.

    Returns:
        Callable accepting a numpy array and returning a Dataset with
        no-data = -9999.
    """

    def _make(arr: np.ndarray) -> Dataset:
        return Dataset.create_from_array(
            arr.astype(np.int32),
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=NO_DATA,
        )

    return _make


def _read_result(path: str) -> np.ndarray:
    """Read the output GeoTIFF written by delete_basins.

    Args:
        path: File path to a GeoTIFF.

    Returns:
        2-D numpy array of pixel values.
    """
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()
    ds = None
    return arr


class TestDeleteBasinsHappyPath:
    """Nominal cases: multiple basins, only the lowest ID is kept."""

    def test_two_basins_keeps_lowest(self, make_basin_dataset, tmp_path):
        """Two basin IDs (1 and 2) — only basin 1 is retained.

        Test scenario:
            3x3 grid with basin 1 in the left column and basin 2 in
            the right column.  After delete_basins, basin-2 cells
            become no-data.
        """
        arr = np.array([
            [1, 1, 2],
            [1, 2, 2],
            [1, 1, 2],
        ], dtype=np.int32)
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "two_basins.tif")

        DEM.delete_basins(ds, out)
        result = _read_result(out)

        expected = np.array([
            [1, 1, NO_DATA],
            [1, NO_DATA, NO_DATA],
            [1, 1, NO_DATA],
        ], dtype=np.int32)
        assert np.array_equal(result, expected), (
            f"Expected only basin 1 retained.\nGot:\n{result}"
        )

    def test_three_basins_keeps_lowest(self, make_basin_dataset, tmp_path):
        """Three basin IDs (1, 5, 10) — only basin 1 is retained.

        Test scenario:
            2x3 grid with basins 1, 5, 10.  After delete_basins, only
            cells with basin 1 survive.
        """
        arr = np.array([
            [1, 5, 10],
            [1, 5, 10],
        ], dtype=np.int32)
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "three_basins.tif")

        DEM.delete_basins(ds, out)
        result = _read_result(out)

        expected = np.array([
            [1, NO_DATA, NO_DATA],
            [1, NO_DATA, NO_DATA],
        ], dtype=np.int32)
        assert np.array_equal(result, expected), (
            f"Expected only basin 1 retained.\nGot:\n{result}"
        )

    def test_non_contiguous_basin_ids(self, make_basin_dataset, tmp_path):
        """Basin IDs 3 and 7 (non-contiguous) — basin 3 is kept.

        Test scenario:
            np.unique returns sorted values, so the first basin is
            the one with the lowest ID (3), not the first encountered
            in raster scan order.
        """
        arr = np.array([
            [7, 7],
            [3, 3],
        ], dtype=np.int32)
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "non_contiguous.tif")

        DEM.delete_basins(ds, out)
        result = _read_result(out)

        expected = np.array([
            [NO_DATA, NO_DATA],
            [3, 3],
        ], dtype=np.int32)
        assert np.array_equal(result, expected), (
            f"Expected basin 3 (lowest) retained.\nGot:\n{result}"
        )


class TestDeleteBasinsSingleBasin:
    """Only one basin ID present — output should be identical to input."""

    def test_single_basin_unchanged(self, make_basin_dataset, tmp_path):
        """Single basin ID with no-data border — all basin cells preserved.

        Test scenario:
            2x3 grid: basin 5 in center, no-data on edges.
            Output should match input exactly.
        """
        arr = np.array([
            [NO_DATA, 5, NO_DATA],
            [NO_DATA, 5, NO_DATA],
        ], dtype=np.int32)
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "single_basin.tif")

        DEM.delete_basins(ds, out)
        result = _read_result(out)

        assert np.array_equal(result, arr), (
            f"Single-basin grid should be unchanged.\nGot:\n{result}"
        )

    def test_single_basin_fills_entire_grid(self, make_basin_dataset, tmp_path):
        """Single basin fills the entire grid — no cells removed.

        Test scenario:
            3x3 grid entirely filled with basin 2, no no-data cells.
        """
        arr = np.full((3, 3), 2, dtype=np.int32)
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "full_basin.tif")

        DEM.delete_basins(ds, out)
        result = _read_result(out)

        assert np.array_equal(result, arr), (
            f"Full single-basin grid should be unchanged.\nGot:\n{result}"
        )


class TestDeleteBasinsAllNoData:
    """Every cell is no-data — nothing to keep or remove."""

    def test_all_nodata_unchanged(self, make_basin_dataset, tmp_path):
        """Grid of all no-data values — output matches input.

        Test scenario:
            2x2 grid where every cell is the no-data value.
            No basins exist, so nothing changes.
        """
        arr = np.full((2, 2), NO_DATA, dtype=np.int32)
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "all_nodata.tif")

        DEM.delete_basins(ds, out)
        result = _read_result(out)

        assert np.array_equal(result, arr), (
            f"All-nodata grid should be unchanged.\nGot:\n{result}"
        )


class TestDeleteBasinsNoDataPreservation:
    """No-data cells in the input remain no-data in the output."""

    def test_nodata_border_preserved(self, make_basin_dataset, tmp_path):
        """No-data cells surrounding basins stay as no-data.

        Test scenario:
            3x3 grid with no-data border and two basins inside.
            After filtering, only the kept basin remains; removed
            basin cells become no-data; original no-data stays.
        """
        arr = np.array([
            [NO_DATA, NO_DATA, NO_DATA],
            [NO_DATA, 1, 2],
            [NO_DATA, NO_DATA, NO_DATA],
        ], dtype=np.int32)
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "border.tif")

        DEM.delete_basins(ds, out)
        result = _read_result(out)

        expected = np.array([
            [NO_DATA, NO_DATA, NO_DATA],
            [NO_DATA, 1, NO_DATA],
            [NO_DATA, NO_DATA, NO_DATA],
        ], dtype=np.int32)
        assert np.array_equal(result, expected), (
            f"No-data border should be preserved.\nGot:\n{result}"
        )


class TestDeleteBasinsInputValidation:
    """Invalid inputs raise the expected exceptions."""

    def test_non_string_path_raises_type_error(self, make_basin_dataset):
        """Passing a non-string path raises TypeError.

        Test scenario:
            path=123 (integer) should raise TypeError with a message
            mentioning the invalid input.
        """
        arr = np.array([[1]], dtype=np.int32)
        ds = make_basin_dataset(arr)

        with pytest.raises(TypeError, match="string type"):
            DEM.delete_basins(ds, 123)

    def test_none_path_raises_type_error(self, make_basin_dataset):
        """Passing path=None raises TypeError.

        Test scenario:
            None is not a string, so TypeError is expected.
        """
        arr = np.array([[1]], dtype=np.int32)
        ds = make_basin_dataset(arr)

        with pytest.raises(TypeError, match="string type"):
            DEM.delete_basins(ds, None)


class TestDeleteBasinsOutputFile:
    """Verify the output is a valid GeoTIFF with correct metadata."""

    def test_output_is_valid_geotiff(self, make_basin_dataset, tmp_path):
        """Output file can be opened by GDAL as a valid raster.

        Test scenario:
            Write output, re-open with GDAL, and confirm band count,
            dimensions, and no-data value are correct.
        """
        arr = np.array([
            [1, 2],
            [1, 2],
        ], dtype=np.int32)
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "valid.tif")

        DEM.delete_basins(ds, out)

        assert os.path.exists(out), f"Output file should exist at {out}"
        result_ds = gdal.Open(out)
        assert result_ds is not None, "GDAL should be able to open the output"
        assert result_ds.RasterCount == 1, (
            f"Expected 1 band, got {result_ds.RasterCount}"
        )
        assert result_ds.RasterYSize == 2, (
            f"Expected 2 rows, got {result_ds.RasterYSize}"
        )
        assert result_ds.RasterXSize == 2, (
            f"Expected 2 cols, got {result_ds.RasterXSize}"
        )
        result_ds = None


class TestDeleteBasinsLargeGrid:
    """Larger grids with many basin IDs."""

    def test_ten_basins_keeps_lowest(self, make_basin_dataset, tmp_path):
        """10 basin IDs spread across a 10x10 grid — only lowest kept.

        Test scenario:
            Each row has a different basin ID (1–10).  After filtering,
            only row with basin 1 survives; all others become no-data.
        """
        arr = np.zeros((10, 10), dtype=np.int32)
        for i in range(10):
            arr[i, :] = i + 1
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "ten_basins.tif")

        DEM.delete_basins(ds, out)
        result = _read_result(out)

        assert np.all(result[0, :] == 1), (
            f"First row should all be basin 1, got {result[0, :]}"
        )
        assert np.all(result[1:, :] == NO_DATA), (
            "All rows except first should be no-data"
        )


class TestDeleteBasinsIdempotence:
    """Running delete_basins on already-filtered output is idempotent."""

    def test_double_application_unchanged(self, make_basin_dataset, tmp_path):
        """Applying delete_basins twice produces the same result.

        Test scenario:
            Run delete_basins once, read back, create a new Dataset
            from the result, run again.  Output should be identical.
        """
        arr = np.array([
            [1, 2, 3],
            [1, 2, 3],
        ], dtype=np.int32)
        ds = make_basin_dataset(arr)
        out1 = str(tmp_path / "pass1.tif")
        out2 = str(tmp_path / "pass2.tif")

        DEM.delete_basins(ds, out1)
        result1 = _read_result(out1)

        ds2 = Dataset(gdal.Open(out1))
        DEM.delete_basins(ds2, out2)
        result2 = _read_result(out2)

        assert np.array_equal(result1, result2), (
            f"Double application should be idempotent.\n"
            f"Pass 1:\n{result1}\nPass 2:\n{result2}"
        )


class TestDeleteBasinsEndToEnd:
    """Full round-trip: create basin raster, filter, read back, verify."""

    def test_round_trip_with_mixed_basins_and_nodata(
        self, make_basin_dataset, tmp_path
    ):
        """Create a realistic basin raster, filter it, verify pixel by pixel.

        Test scenario:
            4x4 grid simulating a catchment delineation output:
              - No-data border (row 0, col 0)
              - Basin 1: main basin (6 cells)
              - Basin 2: small side basin (3 cells)
              - Basin 3: tiny basin (1 cell)
            After delete_basins: only basin 1 cells survive.
        """
        arr = np.array([
            [NO_DATA, 1, 1, NO_DATA],
            [NO_DATA, 1, 2, 2],
            [1, 1, 2, 3],
            [NO_DATA, 1, NO_DATA, NO_DATA],
        ], dtype=np.int32)
        ds = make_basin_dataset(arr)
        out = str(tmp_path / "realistic.tif")

        DEM.delete_basins(ds, out)
        result = _read_result(out)

        expected = np.array([
            [NO_DATA, 1, 1, NO_DATA],
            [NO_DATA, 1, NO_DATA, NO_DATA],
            [1, 1, NO_DATA, NO_DATA],
            [NO_DATA, 1, NO_DATA, NO_DATA],
        ], dtype=np.int32)
        assert np.array_equal(result, expected), (
            f"Round-trip result mismatch.\n"
            f"Expected:\n{expected}\nGot:\n{result}"
        )

        n_basin1_in = np.sum(arr == 1)
        n_basin1_out = np.sum(result == 1)
        assert n_basin1_in == n_basin1_out, (
            f"Basin 1 cell count should be preserved: "
            f"input={n_basin1_in}, output={n_basin1_out}"
        )

        n_nodata_out = np.sum(result == NO_DATA)
        n_total = result.size
        assert n_nodata_out == n_total - n_basin1_out, (
            f"Every non-basin-1 cell should be no-data: "
            f"no-data={n_nodata_out}, expected={n_total - n_basin1_out}"
        )
