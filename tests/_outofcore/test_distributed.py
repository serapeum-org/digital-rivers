"""Tests for the dask backend (B7): dask fill/accumulation == serial, bit-for-bit / allclose."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from pyramids.dataset import Dataset

pytest.importorskip("dask")

from digitalrivers._numba import (  # noqa: E402
    _DIR_DC_I32,
    _DIR_DR_I32,
    d8_flow_direction_numba,
    kahn_accumulate_d8_numba,
    priority_flood_numba,
)
from digitalrivers._outofcore.accumulate import flow_accumulation_tiled  # noqa: E402
from digitalrivers._outofcore.fill import fill_depressions_tiled  # noqa: E402

NODATA = -9999.0
DR, DC = _DIR_DR_I32, _DIR_DC_I32


def _fill_baseline(arr):
    f = priority_flood_numba(arr.astype(np.float64), arr == NODATA, 0.0, DR, DC)
    return np.where(np.isnan(f), NODATA, f).astype(np.float32)


def _dem_array(seed=0, shape=(13, 17)):
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.0, 100.0, size=shape).astype(np.float32)
    arr[3:7, 3:7] = 2.0
    arr[5, 5] = 0.0
    return arr


def _fdir_array(seed=0, shape=(13, 17)):
    rng = np.random.default_rng(seed)
    dem = rng.uniform(0.0, 100.0, size=shape).astype(np.float64)
    filled = priority_flood_numba(dem, np.zeros(shape, dtype=bool), 0.0, DR, DC)
    return d8_flow_direction_numba(filled, 1.0, np.int32(-1), DR, DC)


class TestDaskFill:
    @pytest.mark.parametrize("scheduler", ["threads", "synchronous"])
    @pytest.mark.parametrize("tile", [(5, 5), (3, 3), (13, 17)])
    def test_dask_fill_matches_serial(self, scheduler, tile):
        arr = _dem_array()
        base = _fill_baseline(arr)
        with tempfile.TemporaryDirectory() as tmp:
            dem = Dataset.create_from_array(
                arr,
                top_left_corner=(0, 0),
                cell_size=1.0,
                epsg=4326,
                no_data_value=NODATA,
                driver_type="GTiff",
                path=os.path.join(tmp, "d.tif"),
            )
            out = fill_depressions_tiled(
                dem,
                os.path.join(tmp, "o.tif"),
                tile_rows=tile[0],
                tile_cols=tile[1],
                workers=4,
                scheduler=scheduler,
            )
            try:
                np.testing.assert_array_equal(
                    np.asarray(out.read_array()).astype(np.float32), base
                )
            finally:
                dem.close()
                out.close()

    def test_dask_fill_rejects_mem_source(self):
        arr = _dem_array()
        dem = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
        )  # MEM
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError):
                fill_depressions_tiled(
                    dem, os.path.join(tmp, "o.tif"), tile_rows=5, tile_cols=5, workers=2
                )


class TestDaskAccumulation:
    @pytest.mark.parametrize("scheduler", ["threads", "synchronous"])
    @pytest.mark.parametrize("tile", [(5, 5), (3, 3)])
    def test_dask_accum_matches_serial(self, scheduler, tile):
        fd = _fdir_array()
        base = kahn_accumulate_d8_numba(fd, np.ones(fd.shape), DR, DC)
        with tempfile.TemporaryDirectory() as tmp:
            fdds = Dataset.create_from_array(
                fd.astype(np.float32),
                top_left_corner=(0, 0),
                cell_size=1.0,
                epsg=4326,
                no_data_value=-1,
                driver_type="GTiff",
                path=os.path.join(tmp, "fd.tif"),
            )
            out = flow_accumulation_tiled(
                fdds,
                os.path.join(tmp, "a.tif"),
                tile_rows=tile[0],
                tile_cols=tile[1],
                workers=4,
                scheduler=scheduler,
            )
            try:
                np.testing.assert_allclose(
                    np.asarray(out.read_array()).astype(np.float64), base
                )
            finally:
                fdds.close()
                out.close()
