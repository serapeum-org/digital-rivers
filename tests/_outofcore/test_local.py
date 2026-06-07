"""Tests for halo-tiled local stencil ops (B0): tiled slope == whole-array slope."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers._outofcore.local import max_slope_2d
from digitalrivers.dem import DEM


def _dem(arr: np.ndarray) -> DEM:
    ds = Dataset.create_from_array(
        arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326, no_data_value=-9999.0
    )
    return DEM(ds.raster)


def _terrain(seed: int, shape=(15, 19)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 500.0, size=shape).astype(np.float32)


class TestMaxSlope2d:
    def test_matches_dem_internal_kernel(self):
        arr = _terrain(0)
        dem = _dem(arr)
        whole = np.nanmax(dem._get_8_direction_slopes(), axis=2)
        block = max_slope_2d(arr, dem.cell_size)
        np.testing.assert_allclose(
            np.nan_to_num(block), np.nan_to_num(whole), rtol=1e-6
        )


class TestTiledSlope:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("tile", [(15, 19), (5, 5), (4, 7), (8, 8)])
    def test_tiled_equals_whole_array(self, seed, tile):
        arr = _terrain(seed)
        whole = np.asarray(_dem(arr).slope(engine="in_memory").read_array()).astype(
            np.float32
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = _dem(arr).slope(
                engine="tiled", out_path=os.path.join(tmp, "s.tif"), tile_size=tile
            )
            try:
                got = np.asarray(out.read_array()).astype(np.float32)
                np.testing.assert_allclose(got, whole, rtol=1e-6)
            finally:
                out.close()

    def test_tiled_requires_out_path(self):
        with pytest.raises(ValueError):
            _dem(_terrain(0)).slope(engine="tiled")

    def test_auto_small_is_in_memory_identical(self):
        arr = _terrain(3)
        auto = np.asarray(_dem(arr).slope(engine="auto").read_array()).astype(
            np.float32
        )
        in_mem = np.asarray(_dem(arr).slope(engine="in_memory").read_array()).astype(
            np.float32
        )
        np.testing.assert_array_equal(auto, in_mem)
