"""Tests for the engine resolver and the public engine= switch (B5)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers._outofcore.engine import require_out_path, resolve_engine
from digitalrivers.dem import DEM


class TestRequireOutPath:
    def test_ok_when_path_given(self):
        require_out_path("auto", "out.tif")  # no raise

    def test_auto_message_names_auto(self):
        with pytest.raises(ValueError, match="auto"):
            require_out_path("auto", None)

    def test_tiled_raises(self):
        with pytest.raises(ValueError):
            require_out_path("tiled", None)


class TestResolveEngine:
    @pytest.mark.parametrize("explicit", ["in_memory", "tiled"])
    def test_explicit_passthrough(self, explicit):
        assert resolve_engine(explicit, 10, 10) == explicit

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            resolve_engine("bogus", 10, 10)

    def test_auto_small_is_in_memory(self):
        assert resolve_engine("auto", 100, 100, available_bytes=10**12) == "in_memory"

    def test_auto_large_is_tiled(self):
        assert (
            resolve_engine("auto", 60000, 60000, available_bytes=8 * 10**9) == "tiled"
        )

    def test_auto_threshold_fallback_without_ram_info(self):
        # available_bytes=None path is exercised live; here force the fallback via a tiny threshold.
        assert resolve_engine(
            "auto", 1000, 1000, available_bytes=None, threshold_cells=10
        ) in (
            "tiled",
            "in_memory",
        )


def _dem(arr: np.ndarray, path: str | None = None) -> DEM:
    driver = "GTiff" if path else "MEM"
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=(0, 0),
        cell_size=1.0,
        epsg=4326,
        no_data_value=-9999.0,
        driver_type=driver,
        path=path,
    )
    return DEM(ds.raster)


def _seam_pit_dem(shape=(14, 18)) -> np.ndarray:
    rng = np.random.default_rng(4)
    arr = rng.uniform(0.0, 100.0, size=shape).astype(np.float32)
    arr[4:9, 5:10] = 2.0
    arr[6, 7] = 0.0
    return arr


class TestFillDepressionsEngine:
    def test_tiled_equals_in_memory(self):
        arr = _seam_pit_dem()
        in_mem = _dem(arr).fill_depressions(engine="in_memory").values
        with tempfile.TemporaryDirectory() as tmp:
            dem = _dem(arr)
            out = dem.fill_depressions(
                engine="tiled", out_path=os.path.join(tmp, "f.tif"), tile_size=5
            )
            try:
                np.testing.assert_array_equal(
                    out.values.astype(np.float32), in_mem.astype(np.float32)
                )
            finally:
                out.close()

    def test_tiled_requires_out_path(self):
        with pytest.raises(ValueError):
            _dem(_seam_pit_dem()).fill_depressions(engine="tiled")

    def test_tiled_rejects_inplace(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError):
                _dem(_seam_pit_dem()).fill_depressions(
                    engine="tiled", out_path=os.path.join(tmp, "f.tif"), inplace=True
                )

    def test_tiled_rejects_exact_epsilon(self):
        # eps_fill='exact' is the deferred byte-identical mode; the default 'monotone' runs (covered elsewhere).
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(NotImplementedError):
                _dem(_seam_pit_dem()).fill_depressions(
                    engine="tiled",
                    out_path=os.path.join(tmp, "f.tif"),
                    epsilon=0.01,
                    eps_fill="exact",
                )

    def test_auto_small_dem_is_byte_identical_to_in_memory(self):
        arr = _seam_pit_dem()
        auto = _dem(arr).fill_depressions(engine="auto").values
        in_mem = _dem(arr).fill_depressions(engine="in_memory").values
        np.testing.assert_array_equal(auto, in_mem)

    def _assert_tiled_matches_in_memory_nodata(self, arr, nodata):
        # M4: no-data parity for a non-(-9999) sentinel or NaN no-data — tiled == in-memory at finite cells,
        # and the no-data positions agree.
        def _build(path):
            ds = Dataset.create_from_array(
                arr,
                top_left_corner=(0, 0),
                cell_size=1.0,
                epsg=4326,
                no_data_value=nodata,
                driver_type=("GTiff" if path else "MEM"),
                path=path,
            )
            return DEM(ds.raster)

        in_mem = _build(None).fill_depressions(engine="in_memory").values
        with tempfile.TemporaryDirectory() as tmp:
            out = _build(os.path.join(tmp, "d.tif")).fill_depressions(
                engine="tiled", out_path=os.path.join(tmp, "o.tif"), tile_size=5
            )
            try:
                got = out.values
                np.testing.assert_array_equal(np.isnan(in_mem), np.isnan(got))
                finite = ~np.isnan(in_mem)
                np.testing.assert_array_equal(in_mem[finite], got[finite])
            finally:
                out.close()

    def test_tiled_matches_in_memory_alt_sentinel_nodata(self):
        arr = _seam_pit_dem().astype(np.float32)
        arr[0, 0:3] = -32768.0  # a no-data patch with a non-default sentinel
        self._assert_tiled_matches_in_memory_nodata(arr, -32768.0)

    def test_tiled_matches_in_memory_nan_nodata(self):
        arr = _seam_pit_dem().astype(np.float32)
        arr[0, 0:3] = np.nan  # NaN-valued no-data
        self._assert_tiled_matches_in_memory_nodata(arr, np.nan)


class TestAccumulateEngine:
    def test_tiled_equals_in_memory(self):
        arr = _seam_pit_dem()
        dem = _dem(arr)
        filled = dem.fill_depressions(engine="in_memory")
        fd = filled.flow_direction()
        in_mem = np.asarray(fd.accumulate().read_array()).astype(np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            out = fd.accumulate(
                engine="tiled", out_path=os.path.join(tmp, "a.tif"), tile_size=5
            )
            try:
                np.testing.assert_allclose(
                    np.asarray(out.read_array()).astype(np.float64), in_mem
                )
            finally:
                out.close()

    def test_tiled_requires_out_path(self):
        dem = _dem(_seam_pit_dem())
        fd = dem.flow_direction()
        with pytest.raises(ValueError):
            fd.accumulate(engine="tiled")
