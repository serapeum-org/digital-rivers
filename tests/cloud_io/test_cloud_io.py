"""Tests for `digitalrivers.cloud_io` (tile_windows, write_cog, umbrellas)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from osgeo import gdal
from pyramids.dataset import Dataset, GeoReference, Window
from pyramids.dataset.cog import Compression

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


class TestWriteCogCompression:
    """`write_cog`'s `compress` argument selects the GDAL compression method."""

    @pytest.fixture()
    def dataset(self) -> Dataset:
        """An 8x8 float32 raster, the smallest thing the COG driver will take."""
        return Dataset.from_array(
            np.arange(64, dtype=np.float32).reshape(8, 8),
            geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326),
        )

    @pytest.mark.parametrize(
        "compress, expected",
        [
            ("deflate", "DEFLATE"),
            ("lzw", "LZW"),
            ("zstd", "ZSTD"),
            ("none", None),
        ],
    )
    def test_compress_reaches_the_written_file(
        self, dataset: Dataset, tmp_path, compress: str, expected: str | None
    ):
        """Test each accepted value produces that compression on disk.

        Args:
            dataset: Fixture raster.
            tmp_path: pytest temporary directory.
            compress: The value passed to `write_cog`.
            expected: GDAL's `IMAGE_STRUCTURE` COMPRESSION tag, `None` for none.

        Test scenario:
            Pins the wrapper's accepted value set. `"none"` is included because
            it is how a caller asks for an uncompressed COG.
        """
        written = cloud_io.write_cog(
            dataset, str(tmp_path / f"{compress}.tif"), compress=compress
        )
        handle = gdal.Open(written)
        actual = handle.GetMetadata("IMAGE_STRUCTURE").get("COMPRESSION")
        handle = None
        assert (
            actual == expected
        ), f"compress={compress!r} wrote COMPRESSION={actual!r}, expected {expected!r}"

    def test_compress_is_case_insensitive(self, dataset: Dataset, tmp_path):
        """Test an upper-case value is accepted and behaves like its lower-case form.

        Args:
            dataset: Fixture raster.
            tmp_path: pytest temporary directory.

        Test scenario:
            The docstring promises case-insensitivity, so `"DEFLATE"` must land the
            same as `"deflate"`.
        """
        written = cloud_io.write_cog(
            dataset, str(tmp_path / "upper.tif"), compress="DEFLATE"
        )
        handle = gdal.Open(written)
        actual = handle.GetMetadata("IMAGE_STRUCTURE").get("COMPRESSION")
        handle = None
        assert actual == "DEFLATE", f"Expected DEFLATE, got {actual!r}"

    def test_only_methods_this_gdal_declares_are_parametrised(self):
        """Test the parametrised methods are ones the installed COG driver declares.

        Test scenario:
            Guards the case above against a GDAL build compiled without one of the
            optional codecs, where the write would silently fall back rather than
            produce the asserted tag.
        """
        declared = cloud_io._cog_compression_methods()
        for method in ("DEFLATE", "LZW", "ZSTD", "NONE"):
            assert (
                method in declared
            ), f"{method} not declared by this GDAL's COG driver"

    @pytest.mark.parametrize("bad", ["bogus", "raw", ""])
    def test_unknown_compress_raises(self, dataset: Dataset, tmp_path, bad: str):
        """Test an unrecognised value raises instead of writing an uncompressed file.

        Args:
            dataset: Fixture raster.
            tmp_path: pytest temporary directory.
            bad: A value the COG driver does not accept.

        Test scenario:
            GDAL does not fail on an unknown COMPRESS — it warns and writes the
            file with no compression at all. On a helper meant for continental
            DEMs that turns a typo into a silently enormous output, so the
            wrapper rejects it up front. `"raw"` is covered explicitly: it reads
            like "no compression" but is not a GDAL method.
        """
        with pytest.raises(ValueError, match="not a COG compression method"):
            cloud_io.write_cog(dataset, str(tmp_path / "bad.tif"), compress=bad)

    def test_compression_level_is_left_to_gdal(
        self, dataset: Dataset, tmp_path, mocker
    ):
        """Test the method is passed as a `Compression`, not as a named profile.

        Args:
            dataset: Fixture raster.
            tmp_path: pytest temporary directory.
            mocker: pytest-mock fixture.

        Test scenario:
            Passing the bare string `"deflate"` selects a pyramids *profile*, which
            pins `LEVEL: 9` — maximum effort on a helper meant for continental DEMs.
            An explicit `Compression(compress=...)` leaves the level unset, so GDAL
            picks its own default. Guards the argument's meaning, not just its name.
        """
        spy = mocker.spy(type(dataset), "to_cog")
        cloud_io.write_cog(dataset, str(tmp_path / "level.tif"))
        passed = spy.call_args.kwargs["compression"]
        assert isinstance(
            passed, Compression
        ), f"Expected a Compression, got {type(passed).__name__}"
        assert (
            passed.compress == "DEFLATE"
        ), f"Expected DEFLATE, got {passed.compress!r}"
        assert (
            passed.level is None
        ), f"Compression level should be GDAL's default, got {passed.level!r}"


def test_write_cog_output_is_internally_tiled(tmp_path):
    """The COG writer's output must have block-tiled internal layout."""
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
