"""Tests for the tiled monotone (epsilon>0) fill — eps_fill='monotone' (B6).

These assert only the guarantees the monotone terracing actually provides:
- tiled == whole-array reference, bit-for-bit (tiling adds no error);
- epsilon -> 0 reduces to the exact fill_0;
- flat-free for SMALL epsilon on smooth terrain (the supported regime).
They do NOT assert universal flat-free / byte-identity with the in-memory Barnes kernel — see the module
docstring: that needs the global priority-flood order (deferred eps_fill='exact').
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers._numba import _DIR_DC_I32, _DIR_DR_I32, priority_flood_numba
from digitalrivers._outofcore.fill_monotone import (
    fill_depressions_monotone_tiled,
    monotone_fill_reference,
)
from digitalrivers.dem import DEM

NODATA = -9999.0
DR, DC = _DIR_DR_I32, _DIR_DC_I32


def _fill0(arr):
    return priority_flood_numba(arr.astype(np.float64), arr == NODATA, 0.0, DR, DC)


def _noisy_pit(seed, shape=(14, 18)):
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.0, 100.0, size=shape).astype(np.float32)
    arr[3:8, 4:12] = 2.0
    arr[5, 7] = 0.0
    return arr


def _smooth_bowl(shape=(40, 50)):
    """A smooth paraboloid bowl — real elevation steps >> a small epsilon, so terracing stays flat-free."""
    r, c = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    cr, cc = shape[0] / 2, shape[1] / 2
    bowl = ((r - cr) ** 2 + (c - cc) ** 2).astype(
        np.float32
    )  # rises ~quadratically from the centre pit
    return bowl


def _interior_minima(surface):
    s = surface
    bad = 0
    for i in range(1, s.shape[0] - 1):
        for j in range(1, s.shape[1] - 1):
            if not (s[i - 1 : i + 2, j - 1 : j + 2] < s[i, j]).any():
                bad += 1
    return bad


def _run_tiled(arr, tile, epsilon):
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
        out = fill_depressions_monotone_tiled(
            dem,
            os.path.join(tmp, "o.tif"),
            epsilon=epsilon,
            tile_rows=tile[0],
            tile_cols=tile[1],
        )
        try:
            return np.asarray(out.read_array()).astype(np.float64)
        finally:
            dem.close()
            out.close()


class TestTiledEqualsReference:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("tile", [(14, 18), (5, 5), (3, 3), (4, 7)])
    @pytest.mark.parametrize("epsilon", [0.5, 0.01])
    def test_tiled_matches_whole_array_reference(self, seed, tile, epsilon):
        arr = _noisy_pit(seed)
        ref = monotone_fill_reference(
            arr.astype(np.float64), _fill0(arr), epsilon, NODATA
        )
        np.testing.assert_allclose(_run_tiled(arr, tile, epsilon), ref)

    def test_epsilon_zero_reduces_to_fill0(self):
        arr = _noisy_pit(0)
        f0 = _fill0(arr).astype(np.float32)
        ref = monotone_fill_reference(
            arr.astype(np.float64), _fill0(arr), 0.0, NODATA
        ).astype(np.float32)
        np.testing.assert_array_equal(ref, f0)


class TestFlatFreeSmallEpsilon:
    @pytest.mark.parametrize("tile", [(40, 50), (16, 16), (7, 9)])
    def test_smooth_bowl_is_flat_free(self, tile):
        # On smooth terrain with a tiny epsilon, the terracing removes all interior flats.
        arr = _smooth_bowl()
        out = _run_tiled(arr, tile, epsilon=1e-3)
        assert _interior_minima(out) == 0

    def test_reference_smooth_bowl_flat_free(self):
        arr = _smooth_bowl()
        ref = monotone_fill_reference(arr.astype(np.float64), _fill0(arr), 1e-3, NODATA)
        assert _interior_minima(ref) == 0


class TestGuards:
    def test_barnes_mode_raises(self):
        # eps_fill='barnes' (classic step-count) depends on global order -> not tileable.
        arr = _noisy_pit(0)
        with tempfile.TemporaryDirectory() as tmp:
            dem = DEM(
                Dataset.create_from_array(
                    arr,
                    top_left_corner=(0, 0),
                    cell_size=1.0,
                    epsg=4326,
                    no_data_value=NODATA,
                ).raster
            )
            with pytest.raises(NotImplementedError):
                dem.fill_depressions(
                    engine="tiled",
                    out_path=os.path.join(tmp, "o.tif"),
                    epsilon=0.1,
                    eps_fill="barnes",
                )

    def test_invalid_eps_fill_raises(self):
        arr = _noisy_pit(0)
        with tempfile.TemporaryDirectory() as tmp:
            dem = DEM(
                Dataset.create_from_array(
                    arr,
                    top_left_corner=(0, 0),
                    cell_size=1.0,
                    epsg=4326,
                    no_data_value=NODATA,
                ).raster
            )
            with pytest.raises(ValueError):
                dem.fill_depressions(
                    engine="tiled",
                    out_path=os.path.join(tmp, "o.tif"),
                    epsilon=0.1,
                    eps_fill="bogus",
                )

    def test_public_api_small_epsilon_flat_free(self):
        arr = _smooth_bowl()
        with tempfile.TemporaryDirectory() as tmp:
            dem = DEM(
                Dataset.create_from_array(
                    arr,
                    top_left_corner=(0, 0),
                    cell_size=1.0,
                    epsg=4326,
                    no_data_value=NODATA,
                ).raster
            )
            out = dem.fill_depressions(
                engine="tiled",
                out_path=os.path.join(tmp, "o.tif"),
                epsilon=1e-3,
                tile_size=16,
            )
            try:
                assert _interior_minima(np.asarray(out.read_array())) == 0
            finally:
                out.close()
