"""Tests for `digitalrivers.cloud_io` (tile_windows, write_cog, umbrellas)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference, Window

from digitalrivers import cloud_io


def test_dask_backend_umbrella_raises():
    """`dask_backend` is a deferred umbrella stub pointing at tile_windows."""
    with pytest.raises(NotImplementedError, match="tile_windows"):
        cloud_io.dask_backend()


def test_cloud_storage_umbrella_raises():
    """`cloud_storage` (Zarr/S3/GCS factories) is deferred."""
    with pytest.raises(NotImplementedError, match="write_cog"):
        cloud_io.cloud_storage()


def test_tile_windows_partitions_dataset_into_tiles():
    """`tile_windows` yields edge-clipped x-first `Window` tiles."""
    ds = Dataset.from_array(
        np.zeros((10, 10), dtype=np.float32),
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    )
    wins = list(cloud_io.tile_windows(ds, tile_rows=4, tile_cols=4))
    # 10 / 4 = 3 row stripes (4, 4, 2) and 3 col stripes (4, 4, 2) = 9 tiles.
    assert len(wins) == 9
    assert all(isinstance(w, Window) for w in wins)
    assert Window(col_off=8, row_off=8, cols=2, rows=2) in wins


def test_tile_windows_are_readable_windows():
    """Each yielded window reads exactly the tile it names.

    The regression guard for the transposed-read defect: the generator used to
    yield a row-first tuple while documenting it as a `read_array` window, so an
    off-diagonal non-square tile silently read the wrong region.
    """
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    ds = Dataset.from_array(
        arr,
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    )
    for win in cloud_io.tile_windows(ds, tile_rows=3, tile_cols=3):
        tile = np.asarray(ds.read_array(window=win))
        expected = arr[
            win.row_off : win.row_off + win.rows,
            win.col_off : win.col_off + win.cols,
        ]
        assert np.array_equal(tile, expected), f"{win} read the wrong region"


def test_tile_windows_cover_every_cell_exactly_once():
    """With no overlap the tiles partition the raster: no gaps, no double-cover."""
    ds = Dataset.from_array(
        np.zeros((13, 17), dtype=np.float32),
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    )
    covered = np.zeros((13, 17), dtype=int)
    for win in cloud_io.tile_windows(ds, tile_rows=5, tile_cols=7):
        covered[
            win.row_off : win.row_off + win.rows,
            win.col_off : win.col_off + win.cols,
        ] += 1
    assert np.all(covered == 1)


def test_tile_windows_invalid_sizes_raise():
    ds = Dataset.from_array(
        np.zeros((4, 4), dtype=np.float32),
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    )
    with pytest.raises(ValueError, match="tile_rows"):
        list(cloud_io.tile_windows(ds, tile_rows=0, tile_cols=2))
    with pytest.raises(ValueError, match="overlap"):
        list(cloud_io.tile_windows(ds, tile_rows=2, tile_cols=2, overlap=-1))


def test_write_cog_writes_a_file(tmp_path):
    """`write_cog` writes a COG via GDAL's COG driver."""
    z = np.arange(64, dtype=np.float32).reshape(8, 8)
    ds = Dataset.from_array(
        z,
        geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326),
    )
    out = tmp_path / "out.tif"
    written = cloud_io.write_cog(ds, str(out))
    assert os.path.exists(written)


def test_write_cog_output_is_internally_tiled(tmp_path):
    """The COG writer's output must have block-tiled internal layout."""
    from osgeo import gdal

    z = np.arange(64, dtype=np.float32).reshape(8, 8)
    ds = Dataset.from_array(
        z,
        geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326),
    )
    written = cloud_io.write_cog(ds, str(tmp_path / "out.tif"))
    handle = gdal.Open(written)
    block_size = handle.GetRasterBand(1).GetBlockSize()
    assert block_size[0] > 0 and block_size[1] > 0
    handle = None
