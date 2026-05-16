"""Tests for ``digitalrivers.cloud_io`` (tile_windows, write_cog, umbrellas)."""
from __future__ import annotations

import os

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import cloud_io


def test_dask_backend_umbrella_raises():
    """``dask_backend`` is a deferred umbrella stub pointing at tile_windows."""
    with pytest.raises(NotImplementedError, match="tile_windows"):
        cloud_io.dask_backend()


def test_cloud_storage_umbrella_raises():
    """``cloud_storage`` (Zarr/S3/GCS factories) is deferred."""
    with pytest.raises(NotImplementedError, match="write_cog"):
        cloud_io.cloud_storage()


def test_tile_windows_partitions_dataset_into_tiles():
    """``tile_windows`` yields edge-clipped ``(row, col, h, w)`` windows."""
    ds = Dataset.create_from_array(
        np.zeros((10, 10), dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    wins = list(cloud_io.tile_windows(ds, tile_rows=4, tile_cols=4))
    # 10 / 4 = 3 row stripes (4, 4, 2) and 3 col stripes (4, 4, 2) = 9 tiles.
    assert len(wins) == 9
    assert (8, 8, 2, 2) in wins


def test_tile_windows_invalid_sizes_raise():
    ds = Dataset.create_from_array(
        np.zeros((4, 4), dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    with pytest.raises(ValueError, match="tile_rows"):
        list(cloud_io.tile_windows(ds, tile_rows=0, tile_cols=2))
    with pytest.raises(ValueError, match="overlap"):
        list(cloud_io.tile_windows(ds, tile_rows=2, tile_cols=2, overlap=-1))


def test_write_cog_writes_a_file(tmp_path):
    """``write_cog`` writes a COG via GDAL's COG driver."""
    z = np.arange(64, dtype=np.float32).reshape(8, 8)
    ds = Dataset.create_from_array(
        z, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
    )
    out = tmp_path / "out.tif"
    written = cloud_io.write_cog(ds, str(out))
    assert os.path.exists(written)


def test_write_cog_output_is_internally_tiled(tmp_path):
    """The COG writer's output must have block-tiled internal layout."""
    from osgeo import gdal

    z = np.arange(64, dtype=np.float32).reshape(8, 8)
    ds = Dataset.create_from_array(
        z, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
    )
    written = cloud_io.write_cog(ds, str(tmp_path / "out.tif"))
    handle = gdal.Open(written)
    block_size = handle.GetRasterBand(1).GetBlockSize()
    assert block_size[0] > 0 and block_size[1] > 0
    handle = None
