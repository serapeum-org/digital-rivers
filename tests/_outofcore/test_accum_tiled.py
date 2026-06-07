"""Equivalence tests for tiled flow accumulation (B4): tiled == whole-array (D8/Rho8)."""

from __future__ import annotations

import os
import tempfile
import types

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers._numba import (
    _DIR_DC_I32,
    _DIR_DR_I32,
    d8_flow_direction_numba,
    kahn_accumulate_d8_numba,
    priority_flood_numba,
)
from digitalrivers._outofcore.accumulate import flow_accumulation_tiled

DR, DC = _DIR_DR_I32, _DIR_DC_I32


def _make_fdir(seed: int, shape=(13, 17)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dem = rng.uniform(0.0, 100.0, size=shape).astype(np.float64)
    filled = priority_flood_numba(dem, np.zeros(shape, dtype=bool), 0.0, DR, DC)
    return d8_flow_direction_numba(filled, 1.0, np.int32(-1), DR, DC)


def _fdir_dataset(fd: np.ndarray, path: str) -> Dataset:
    return Dataset.create_from_array(
        fd.astype(np.float32),
        top_left_corner=(0, 0),
        cell_size=1.0,
        epsg=4326,
        no_data_value=-1,
        driver_type="GTiff",
        path=path,
    )


def _run(fd: np.ndarray, tile, weights=None):
    with tempfile.TemporaryDirectory() as tmp:
        fdds = _fdir_dataset(fd, os.path.join(tmp, "fd.tif"))
        wds = None
        if weights is not None:
            wds = Dataset.create_from_array(
                weights.astype(np.float32),
                top_left_corner=(0, 0),
                cell_size=1.0,
                epsg=4326,
                driver_type="GTiff",
                path=os.path.join(tmp, "w.tif"),
            )
        out = flow_accumulation_tiled(
            fdds,
            os.path.join(tmp, "a.tif"),
            weights=wds,
            tile_rows=tile[0],
            tile_cols=tile[1],
        )
        try:
            return np.asarray(out.read_array()).astype(np.float64)
        finally:
            fdds.close()
            out.close()
            if wds is not None:
                wds.close()


class TestEquivalence:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("tile", [(13, 17), (5, 5), (3, 3), (4, 6), (7, 4)])
    def test_unweighted_matches_whole_array(self, seed, tile):
        fd = _make_fdir(seed)
        base = kahn_accumulate_d8_numba(fd, np.ones(fd.shape), DR, DC)
        np.testing.assert_allclose(_run(fd, tile), base)

    def test_weighted_matches_whole_array(self):
        fd = _make_fdir(0, shape=(12, 12))
        rng = np.random.default_rng(5)
        wt = rng.uniform(0.5, 3.0, size=fd.shape)
        base = kahn_accumulate_d8_numba(fd, wt, DR, DC)
        np.testing.assert_allclose(_run(fd, (5, 5), weights=wt), base)

    def test_river_across_seam_total(self):
        # A single column draining downward (all flow to one outlet) split by a horizontal seam.
        rows, cols = 9, 1
        fd = np.zeros((rows, cols), dtype=np.int32)  # all flow South (code 0)
        fd[-1, 0] = -1  # outlet sink
        base = kahn_accumulate_d8_numba(fd, np.ones(fd.shape), DR, DC)
        np.testing.assert_allclose(_run(fd, (3, 1)), base)
        assert base[-1, 0] == rows - 1  # outlet receives all upstream cells


class TestGuards:
    @pytest.mark.parametrize("routing", ["dinf", "mfd_quinn", "mfd_holmgren"])
    def test_divergent_routing_raises(self, routing):
        dummy = types.SimpleNamespace(routing=routing, rows=4, columns=4)
        with pytest.raises(NotImplementedError):
            flow_accumulation_tiled(dummy, "unused.tif", tile_rows=2, tile_cols=2)
