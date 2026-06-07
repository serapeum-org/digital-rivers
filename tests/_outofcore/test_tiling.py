"""Tests for digitalrivers._outofcore.tiling."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from pyramids.dataset import Dataset

import types

from digitalrivers._outofcore.tiling import (
    TileSpec,
    gid,
    plan_tiles,
    read_tile,
    require_single_band,
    write_core,
)


class TestRequireSingleBand:
    def test_single_band_ok(self):
        require_single_band(types.SimpleNamespace(band_count=1))  # no raise

    def test_multiband_raises(self):
        with pytest.raises(ValueError):
            require_single_band(types.SimpleNamespace(band_count=3))


from digitalrivers.cloud_io import tile_windows


class TestPlanTiles:
    def test_exact_partition_no_overlap(self):
        specs = plan_tiles(4, 5, 2, 3, halo=0)
        cores = [(s.row_off, s.col_off, s.n_rows, s.n_cols) for s in specs]
        assert cores == [(0, 0, 2, 3), (0, 3, 2, 2), (2, 0, 2, 3), (2, 3, 2, 2)]

    def test_ids_are_row_major_contiguous(self):
        specs = plan_tiles(10, 10, 4, 4)
        assert [s.tid for s in specs] == list(range(len(specs)))
        assert [(s.row, s.col) for s in specs[:3]] == [(0, 0), (0, 1), (0, 2)]

    def test_cores_match_cloud_io_tile_windows_when_halo_zero(self):
        # plan_tiles cores must tile identically to cloud_io.tile_windows
        arr = np.zeros((37, 53), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
        )
        win = list(tile_windows(ds, tile_rows=16, tile_cols=16, overlap=0))
        specs = plan_tiles(37, 53, 16, 16, halo=0)
        assert [(s.row_off, s.col_off, s.n_rows, s.n_cols) for s in specs] == win

    def test_cores_cover_domain_exactly(self):
        rows, cols = 30, 40
        covered = np.zeros((rows, cols), dtype=int)
        for s in plan_tiles(rows, cols, 7, 9):
            covered[
                s.row_off : s.row_off + s.n_rows, s.col_off : s.col_off + s.n_cols
            ] += 1
        assert np.all(covered == 1)  # no gaps, no overlap

    @pytest.mark.parametrize("bad", [(0, 4), (4, -1)])
    def test_rejects_nonpositive_tile_size(self, bad):
        with pytest.raises(ValueError):
            plan_tiles(10, 10, bad[0], bad[1])

    def test_rejects_negative_halo(self):
        with pytest.raises(ValueError):
            plan_tiles(10, 10, 4, 4, halo=-1)


class TestTileSpecGeometry:
    def test_interior_halo_bounds_and_core_slice(self):
        s = TileSpec(0, 0, 0, 2, 2, 3, 3, halo=1)
        assert s.halo_bounds(10, 10) == (1, 1, 6, 6)
        assert s.core_slice(10, 10) == (slice(1, 4), slice(1, 4))

    def test_corner_halo_clips_at_edges(self):
        s = TileSpec(0, 0, 0, 0, 0, 3, 3, halo=1)
        assert s.halo_bounds(10, 10) == (0, 0, 4, 4)
        assert s.core_slice(10, 10) == (slice(0, 3), slice(0, 3))


class TestGid:
    def test_basic(self):
        assert gid(2, 3, 8) == 19

    def test_overflow_safe_past_2_31(self):
        value = gid(50000, 50000, 51810)
        assert value > 2**31
        assert isinstance(value, int)


def _valued_raster(rows: int, cols: int, path: str) -> Dataset:
    """A GTiff whose cell value is ``row * 1000 + col`` (unique per cell)."""
    arr = (np.arange(rows)[:, None] * 1000 + np.arange(cols)[None, :]).astype(
        np.float32
    )
    return Dataset.create_from_array(
        arr,
        top_left_corner=(0, 0),
        cell_size=1.0,
        epsg=4326,
        driver_type="GTiff",
        path=path,
    )


class TestReadWriteRoundTrip:
    """The crux: read_tile + write_core must bridge pyramids' two window conventions correctly."""

    @pytest.mark.parametrize("halo", [0, 1, 2])
    @pytest.mark.parametrize("tile", [(8, 8), (5, 7), (16, 16)])
    def test_read_tile_core_matches_source(self, halo, tile):
        rows, cols = 13, 17
        with tempfile.TemporaryDirectory() as tmp:
            src = _valued_raster(rows, cols, os.path.join(tmp, "src.tif"))
            try:
                source = (
                    np.arange(rows)[:, None] * 1000 + np.arange(cols)[None, :]
                ).astype(np.float32)
                for s in plan_tiles(rows, cols, tile[0], tile[1], halo=halo):
                    arr, core = read_tile(src, s, rows, cols)
                    got = arr[core]
                    want = source[
                        s.row_off : s.row_off + s.n_rows,
                        s.col_off : s.col_off + s.n_cols,
                    ]
                    assert got.shape == (s.n_rows, s.n_cols)
                    np.testing.assert_array_equal(got, want)
            finally:
                src.close()

    @pytest.mark.parametrize("tile", [(8, 8), (5, 7)])
    def test_tile_then_untile_reconstructs_raster(self, tile):
        rows, cols = 13, 17
        with tempfile.TemporaryDirectory() as tmp:
            src = _valued_raster(rows, cols, os.path.join(tmp, "src.tif"))
            out = Dataset.create_empty(
                rows,
                cols,
                dtype="float32",
                geo=src.geotransform,
                epsg=src.epsg,
                no_data_value=-9999.0,
                driver_type="GTiff",
                path=os.path.join(tmp, "out.tif"),
            )
            try:
                source = (
                    np.arange(rows)[:, None] * 1000 + np.arange(cols)[None, :]
                ).astype(np.float32)
                for s in plan_tiles(rows, cols, tile[0], tile[1], halo=1):
                    arr, core = read_tile(src, s, rows, cols)
                    write_core(out, s, arr[core])
                np.testing.assert_array_equal(np.asarray(out.read_array()), source)
            finally:
                src.close()
                out.close()

    def test_write_core_shape_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Dataset.create_empty(
                6,
                6,
                dtype="float32",
                geo=(0, 1, 0, 0, 0, -1),
                epsg=4326,
                no_data_value=-9999.0,
                driver_type="GTiff",
                path=os.path.join(tmp, "o.tif"),
            )
            try:
                s = TileSpec(0, 0, 0, 0, 0, 3, 3, halo=1)
                with pytest.raises(ValueError):
                    write_core(out, s, np.zeros((2, 2), dtype=np.float32))
            finally:
                out.close()
