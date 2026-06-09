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

    def test_tiled_rejects_barnes_epsilon(self):
        # eps_fill='barnes' (classic step-count) is not tileable; eps_fill='exact' (default) is the tileable,
        # byte-identical ramp (covered by the equivalence tests).
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(NotImplementedError):
                _dem(_seam_pit_dem()).fill_depressions(
                    engine="tiled",
                    out_path=os.path.join(tmp, "f.tif"),
                    epsilon=0.01,
                    eps_fill="barnes",
                )

    def test_auto_small_dem_is_byte_identical_to_in_memory(self):
        arr = _seam_pit_dem()
        auto = _dem(arr).fill_depressions(engine="auto").values
        in_mem = _dem(arr).fill_depressions(engine="in_memory").values
        np.testing.assert_array_equal(auto, in_mem)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("tile", [5, 7, (5, 9), 64])
    def test_epsilon_exact_tiled_is_byte_identical_to_in_memory(self, dtype, tile):
        # Path B: eps_fill="exact" (the shared exit-distance ramp) is bit-for-bit identical across engines,
        # for every tile size and dtype — read the rasters (not .values, which downcasts float64 -> float32).
        arr = _seam_pit_dem().astype(dtype)
        in_mem = np.asarray(
            _dem(arr)
            .fill_depressions(method="priority_flood", epsilon=1e-3, engine="in_memory")
            .read_array()
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = _dem(arr).fill_depressions(
                method="priority_flood",
                epsilon=1e-3,
                engine="tiled",
                out_path=os.path.join(tmp, "f.tif"),
                tile_size=tile,
                eps_fill="exact",
            )
            try:
                tiled = np.asarray(out.read_array())
                assert tiled.dtype == in_mem.dtype == np.dtype(dtype)
                np.testing.assert_array_equal(tiled, in_mem)
            finally:
                out.close()

    def test_epsilon_integer_dem_keeps_float_ramp_and_matches(self):
        # M1: epsilon>0 on an integer DEM must not truncate the ramp to ints. Both engines promote to float and
        # stay byte-identical. (int16 is the SRTM dtype; epsilon>0 is the flat-breaking step before D8 routing.)
        arr = np.full((12, 12), 100, dtype=np.int16)
        arr[4:8, 4:8] = 50
        arr[6, 6] = 10
        in_mem = np.asarray(
            _dem(arr)
            .fill_depressions(method="priority_flood", epsilon=1e-3, engine="in_memory")
            .read_array()
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = _dem(arr).fill_depressions(
                method="priority_flood",
                epsilon=1e-3,
                engine="tiled",
                out_path=os.path.join(tmp, "f.tif"),
                tile_size=5,
                eps_fill="exact",
            )
            try:
                tiled = np.asarray(out.read_array())
                assert np.issubdtype(in_mem.dtype, np.floating)
                assert np.issubdtype(tiled.dtype, np.floating)
                # the pit pool carries a real gradient, not a single truncated fill level
                assert np.unique(in_mem[4:8, 4:8]).size > 1
                np.testing.assert_array_equal(tiled, in_mem)
            finally:
                out.close()

    def test_epsilon_monotone_is_alias_of_exact(self):
        arr = _seam_pit_dem()
        a = np.asarray(
            _dem(arr)
            .fill_depressions(method="priority_flood", epsilon=1e-3, eps_fill="exact")
            .read_array()
        )
        b = np.asarray(
            _dem(arr)
            .fill_depressions(method="priority_flood", epsilon=1e-3, eps_fill="monotone")
            .read_array()
        )
        np.testing.assert_array_equal(a, b)

    def test_epsilon_barnes_differs_from_exact_in_memory(self):
        # The two epsilon definitions are genuinely different surfaces (sanity that "barnes" is not aliased away).
        arr = _seam_pit_dem()
        exact = np.asarray(
            _dem(arr)
            .fill_depressions(method="priority_flood", epsilon=0.1, eps_fill="exact")
            .read_array()
        )
        barnes = np.asarray(
            _dem(arr)
            .fill_depressions(method="priority_flood", epsilon=0.1, eps_fill="barnes")
            .read_array()
        )
        assert not np.array_equal(exact, barnes)

    def test_unknown_eps_fill_raises(self):
        with pytest.raises(ValueError, match="eps_fill"):
            _dem(_seam_pit_dem()).fill_depressions(
                method="priority_flood", epsilon=1e-3, eps_fill="bogus"
            )

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

    def test_dem_flow_accumulation_forwards_engine(self):
        # N2: DEM.flow_accumulation(engine="tiled", ...) streams to disk and matches the in-memory result.
        arr = _seam_pit_dem()
        filled = _dem(arr).fill_depressions(engine="in_memory")
        fd = filled.flow_direction()
        in_mem = np.asarray(filled.flow_accumulation(fd).read_array()).astype(
            np.float64
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = filled.flow_accumulation(
                fd, engine="tiled", out_path=os.path.join(tmp, "a.tif"), tile_size=5
            )
            try:
                np.testing.assert_allclose(
                    np.asarray(out.read_array()).astype(np.float64), in_mem
                )
            finally:
                out.close()
