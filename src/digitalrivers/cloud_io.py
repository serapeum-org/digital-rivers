"""Cloud-optimised raster I/O and chunked-tile streaming.

Two working halves ship today plus two umbrella stubs for the still-deferred
features:

* :func:`tile_windows` — chunked-iteration helper that yields
  `pyramids.dataset.Window` tiles for streaming a continental DEM through
  any per-tile algorithm without materialising the full raster in memory.
* :func:`write_cog` — Cloud-Optimised GeoTIFF writer; a thin convenience
  wrapper that delegates to pyramids' `Dataset.to_cog`.

Deferred (umbrella raises `NotImplementedError` with a deferral note):

* :func:`dask_backend` — full Dask-graph integration on top of
  `tile_windows`.
* :func:`cloud_storage` — Zarr / S3 / GCS read & write factories.
"""

from __future__ import annotations

from pyramids.dataset import Window
from pyramids.dataset.cog import Compression


def tile_windows(
    dataset,
    tile_rows: int = 1024,
    tile_cols: int = 1024,
    overlap: int = 0,
):
    """Iterate `Window` tiles over a Dataset.

    Yields one `pyramids.dataset.Window` per tile so callers can stream a
    continental DEM through any per-tile algorithm without ever
    materialising the full raster in memory. A `Window` is what
    `Dataset.read_array(window=...)` and `Dataset.write_array(window=...)`
    both take, so a tile can be read and written back with the same object.

    Tile size defaults match the COG / Cloud-Optimised GeoTIFF spec
    (512×512 or 1024×1024 internal tiles).

    Args:
        dataset: A pyramids `Dataset` (or subclass).
        tile_rows: Tile height in cells. Defaults to 1024.
        tile_cols: Tile width in cells. Defaults to 1024.
        overlap: Cells of overlap between adjacent tiles. Useful for
            algorithms that need neighbour context (slopes, flow
            direction, dilations). Default 0.

    Yields:
        `Window(col_off, row_off, cols, rows)` in row-major order. Edge
        tiles are clipped to the dataset bounds.

    Note:
        The window is **x-first** — column before row — matching GDAL and
        `read_array`. Earlier releases yielded a bare row-first
        `(row_off, col_off, n_rows, n_cols)` tuple while documenting it as
        a `read_array` window, so feeding it straight to `read_array`
        silently read a transposed region. Unpack as
        `col_off, row_off, cols, rows` if you still need the four numbers.

    Examples:
        - Iterate a 5x5 dataset in 3x3 tiles with no overlap:

            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalrivers.cloud_io import tile_windows
            >>> ds = Dataset.from_array(
            ...     np.zeros((5, 5), dtype=np.float32),
            ...     geo_ref=GeoReference(
            ...         top_left_corner=(0, 0),
            ...         cell_size=1.0,
            ...         epsg=4326,
            ...     ),
            ... )
            >>> windows = list(tile_windows(ds, tile_rows=3, tile_cols=3))
            >>> [tuple(w) for w in windows]
            [(0, 0, 3, 3), (3, 0, 2, 3), (0, 3, 3, 2), (3, 3, 2, 2)]
            >>> windows[1]
            Window(col_off=3, row_off=0, cols=2, rows=3)

        - Each window reads the tile it names:

            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalrivers.cloud_io import tile_windows
            >>> arr = np.arange(25, dtype=np.float32).reshape(5, 5)
            >>> ds = Dataset.from_array(
            ...     arr,
            ...     geo_ref=GeoReference(
            ...         top_left_corner=(0, 0),
            ...         cell_size=1.0,
            ...         epsg=4326,
            ...     ),
            ... )
            >>> win = list(tile_windows(ds, tile_rows=3, tile_cols=3))[1]
            >>> tile = np.asarray(ds.read_array(window=win))
            >>> bool((tile == arr[0:3, 3:5]).all())
            True
    """
    rows = dataset.rows
    cols = dataset.columns
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile_rows and tile_cols must be positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    step_r = max(1, tile_rows - overlap)
    step_c = max(1, tile_cols - overlap)
    for r_off in range(0, rows, step_r):
        n_r = min(tile_rows, rows - r_off)
        if n_r <= 0:
            break
        for c_off in range(0, cols, step_c):
            n_c = min(tile_cols, cols - c_off)
            if n_c <= 0:
                break
            yield Window(col_off=c_off, row_off=r_off, cols=n_c, rows=n_r)


def dask_backend(*args, **kwargs):
    """Dask / chunked-tile backend for continental DEMs — umbrella stub.

    Full Dask-graph integration remains deferred. The chunked-iteration
    half ships as :func:`tile_windows` — callers process continental DEMs
    by looping `for win in tile_windows(ds): chunk = ds.read_array(window=win)`
    without loading the full mosaic in memory.

    References:
        Dask documentation: https://docs.dask.org/
        rioxarray chunked I/O.
    """
    raise NotImplementedError(
        "dask_backend umbrella API deferred. Use "
        "digitalrivers.cloud_io.tile_windows for per-tile streaming."
    )


def write_cog(dataset, path: str, compress: str = "deflate") -> str:
    """Cloud-Optimised GeoTIFF writer.

    Thin convenience wrapper that delegates to pyramids' `Dataset.to_cog`,
    the canonical COG writer. COG is the standard cloud-native format for
    raster data: internally tiled, internally overviewed, and indexable by
    HTTP range requests — the foundation of every modern STAC-based pipeline.
    Reach for `dataset.to_cog(...)` directly when you need the full option
    matrix (overviews, blocksize, tiling scheme, reprojection, etc.).

    Args:
        dataset: Any `pyramids.Dataset` (or subclass — DEM,
            FlowDirection, Accumulation, etc.).
        path: Output `.tif` path.
        compress: GDAL compression method (`"deflate"` default, `"lzw"`,
            `"zstd"`, `"none"`). Case-insensitive. The method alone is set;
            the compression *level* is left at GDAL's default, and the
            predictor is resolved from the band dtype by pyramids.

    Returns:
        The output path on success.

    Raises:
        DriverNotExistError: If the GDAL build lacks the COG driver.
        FileNotFoundError: If the parent directory does not exist.
        FailedToSaveError: If GDAL's COG `CreateCopy` fails.

    Examples:
        - Write a 5x5 DEM as a COG:

            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalrivers.cloud_io import write_cog
            >>> import tempfile, os
            >>> arr = np.arange(25, dtype=np.float32).reshape(5, 5)
            >>> ds = Dataset.from_array(
            ...     arr,
            ...     geo_ref=GeoReference(
            ...         top_left_corner=(0, 0),
            ...         cell_size=1.0,
            ...         epsg=4326,
            ...     ),
            ... )
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     out_path = os.path.join(tmpdir, "out.tif")
            ...     result = write_cog(ds, out_path)
            ...     os.path.exists(result)
            True
    """
    # `Compression(compress=...)` sets the method only. Passing the bare string
    # would select a named pyramids *profile* instead, and the "deflate" profile
    # pins LEVEL=9 — maximum effort on a helper meant for continental DEMs.
    return str(dataset.to_cog(path, compression=Compression(compress=compress.upper())))


def cloud_storage(*args, **kwargs):
    """Zarr / S3 / GCS factories — umbrella stub.

    The COG write half is shipped under :func:`write_cog`. Zarr writers
    and S3 / GCS read factories remain deferred pending a follow-up PR.
    """
    raise NotImplementedError(
        "cloud_storage umbrella API deferred. The COG write half is "
        "available via digitalrivers.cloud_io.write_cog."
    )
