"""Equivalence tests for tiled depression fill (B3): tiled == whole-array, bit-for-bit (epsilon=0)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers._numba import _DIR_DC_I32, _DIR_DR_I32, priority_flood_numba
from digitalrivers._outofcore.fill import fill_depressions_tiled

NODATA = -9999.0


def _baseline(arr: np.ndarray) -> np.ndarray:
    """Whole-array Priority-Flood (epsilon=0), nodata written back as the sentinel, float32."""
    filled = priority_flood_numba(
        arr.astype(np.float64), arr == NODATA, 0.0, _DIR_DR_I32, _DIR_DC_I32
    )
    return np.where(np.isnan(filled), NODATA, filled).astype(np.float32)


def _dem_with_seam_pit(seed: int, shape=(13, 17), nodata_patch=False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.0, 100.0, size=shape).astype(np.float32)
    arr[3:7, 3:7] = 2.0  # a depression that straddles small tile seams
    arr[5, 5] = 0.0
    if nodata_patch:
        arr[0, 0:3] = NODATA
    return arr


def _run_tiled(arr: np.ndarray, tile, cache="evict", scratch=None) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        dem = Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=NODATA,
            driver_type="GTiff",
            path=os.path.join(tmp, "dem.tif"),
        )
        out = fill_depressions_tiled(
            dem,
            os.path.join(tmp, "out.tif"),
            tile_rows=tile[0],
            tile_cols=tile[1],
            cache=cache,
            scratch_dir=(
                os.path.join(tmp, "scratch")
                if scratch is None and cache == "cache"
                else scratch
            ),
        )
        try:
            return np.asarray(out.read_array()).astype(np.float32)
        finally:
            dem.close()
            out.close()


class TestEquivalence:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("tile", [(13, 17), (5, 5), (3, 3), (4, 6), (7, 4)])
    def test_tiled_equals_whole_array(self, seed, tile):
        arr = _dem_with_seam_pit(seed)
        np.testing.assert_array_equal(_run_tiled(arr, tile), _baseline(arr))

    @pytest.mark.parametrize("tile", [(5, 5), (3, 3)])
    def test_with_nodata(self, tile):
        arr = _dem_with_seam_pit(7, nodata_patch=True)
        np.testing.assert_array_equal(_run_tiled(arr, tile), _baseline(arr))

    @pytest.mark.parametrize("cache", ["evict", "retain", "cache"])
    def test_memory_modes_agree(self, cache):
        arr = _dem_with_seam_pit(1)
        np.testing.assert_array_equal(
            _run_tiled(arr, (5, 5), cache=cache), _baseline(arr)
        )

    def test_larger_random_dem(self):
        rng = np.random.default_rng(99)
        arr = rng.uniform(0.0, 200.0, size=(40, 55)).astype(np.float32)
        arr[10:20, 10:20] = 1.0  # broad basin
        np.testing.assert_array_equal(_run_tiled(arr, (16, 16)), _baseline(arr))


class TestDtypePreservation:
    def test_float64_dem_preserves_dtype_and_matches_in_memory(self):
        # M2: a float64 DEM must yield a float64 tiled fill, bit-for-bit with the whole-array fill.
        rng = np.random.default_rng(2)
        arr = rng.uniform(0.0, 100.0, size=(13, 17)).astype(np.float64)
        arr[3:7, 3:7] = 2.0
        arr[5, 5] = 0.0
        base = priority_flood_numba(arr, arr == NODATA, 0.0, _DIR_DR_I32, _DIR_DC_I32)
        base = np.where(np.isnan(base), NODATA, base)  # float64
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
                dem, os.path.join(tmp, "o.tif"), tile_rows=5, tile_cols=5
            )
            try:
                got = np.asarray(out.read_array())
                assert got.dtype == np.float64
                np.testing.assert_array_equal(got, base)
            finally:
                dem.close()
                out.close()


class TestGuards:
    def test_epsilon_exact_mode_raises(self):
        # epsilon>0 with eps_fill='exact' is the deferred byte-identical mode -> NotImplementedError.
        arr = _dem_with_seam_pit(0)
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
            try:
                with pytest.raises(NotImplementedError):
                    fill_depressions_tiled(
                        dem,
                        os.path.join(tmp, "o.tif"),
                        tile_rows=5,
                        tile_cols=5,
                        epsilon=0.01,
                        eps_fill="exact",
                    )
            finally:
                dem.close()

    def test_monotone_with_workers_warns(self):
        # L2: workers>1 is ignored for the serial monotone (epsilon>0) path -> a warning, not silence.
        arr = _dem_with_seam_pit(0)
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
            try:
                with pytest.warns(UserWarning):
                    out = fill_depressions_tiled(
                        dem,
                        os.path.join(tmp, "o.tif"),
                        tile_rows=5,
                        tile_cols=5,
                        epsilon=0.001,
                        workers=4,
                    )
                out.close()
            finally:
                dem.close()

    def test_epsilon_monotone_mode_runs(self):
        # epsilon>0 with the default eps_fill='monotone' produces a result (does not raise).
        arr = _dem_with_seam_pit(0)
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
                dem, os.path.join(tmp, "o.tif"), tile_rows=5, tile_cols=5, epsilon=0.001
            )
            try:
                assert np.asarray(out.read_array()).shape == arr.shape
            finally:
                dem.close()
                out.close()
