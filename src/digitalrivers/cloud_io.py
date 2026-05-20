"""Cloud-optimised raster I/O and chunked-tile streaming.

Two working halves ship today plus two umbrella stubs for the still-deferred
features:

* :func:`tile_windows` — chunked-iteration helper that yields
  GDAL-compatible `(row_off, col_off, n_rows, n_cols)` windows for
  streaming a continental DEM through any per-tile algorithm without
  materialising the full raster in memory.
* :func:`write_cog` — Cloud-Optimised GeoTIFF writer wrapping GDAL's
  built-in COG driver.

Deferred (umbrella raises `NotImplementedError` with a deferral note):

* :func:`dask_backend` — full Dask-graph integration on top of
  `tile_windows`.
* :func:`cloud_storage` — Zarr / S3 / GCS read & write factories.
"""

from __future__ import annotations

from osgeo import gdal


def tile_windows(
    dataset,
    tile_rows: int = 1024,
    tile_cols: int = 1024,
    overlap: int = 0,
):
    """Iterate `(row_off, col_off, n_rows, n_cols)` tile windows over a Dataset.

    Yields one window per tile so callers can stream a continental DEM
    through any per-tile algorithm without ever materialising the full
    raster in memory. Each window is a GDAL-compatible
    `(xoff, yoff, xsize, ysize)` quadruple ready to pass into
    `Dataset.read_array(window=...)`.

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
        `(row_off, col_off, n_rows, n_cols)` int tuples in row-major
        order. Edge tiles are clipped to the dataset bounds.

    Examples:
        - Iterate a 5x5 dataset in 3x3 tiles with no overlap:

            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from digitalrivers.cloud_io import tile_windows
            >>> ds = Dataset.create_from_array(
            ...     np.zeros((5, 5), dtype=np.float32),
            ...     top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
            ... )
            >>> windows = list(tile_windows(ds, tile_rows=3, tile_cols=3))
            >>> [(w[0], w[1], w[2], w[3]) for w in windows]
            [(0, 0, 3, 3), (0, 3, 3, 2), (3, 0, 2, 3), (3, 3, 2, 2)]
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
            yield (r_off, c_off, n_r, n_c)


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

    Writes a pyramids `Dataset` to a COG file using GDAL's built-in COG
    driver. COG is the standard cloud-native format for raster data:
    internally tiled, internally overviewed, and indexable by HTTP range
    requests — the foundation of every modern STAC-based pipeline.

    Args:
        dataset: Any `pyramids.Dataset` (or subclass — DEM,
            FlowDirection, Accumulation, etc.).
        path: Output `.tif` path.
        compress: GDAL compression option (`"deflate"` default,
            `"lzw"`, `"zstd"`, `"none"`).

    Returns:
        The output path on success.

    Raises:
        RuntimeError: If GDAL's COG driver fails (older GDAL builds may
            need `"GTIFF"` with manual COG options instead).

    Examples:
        - Write a 5x5 DEM as a COG:

            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from digitalrivers.cloud_io import write_cog
            >>> import tempfile, os
            >>> arr = np.arange(25, dtype=np.float32).reshape(5, 5)
            >>> ds = Dataset.create_from_array(
            ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
            ... )
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     out_path = os.path.join(tmpdir, "out.tif")
            ...     result = write_cog(ds, out_path)
            ...     os.path.exists(result)
            True
    """
    src = dataset.raster
    driver = gdal.GetDriverByName("COG")
    if driver is None:
        raise RuntimeError(
            "GDAL COG driver not available; upgrade GDAL >= 3.1 or write "
            "via GTIFF with TILED=YES + COPY_SRC_OVERVIEWS=YES manually."
        )
    options = [f"COMPRESS={compress.upper()}"]
    out = driver.CreateCopy(path, src, 0, options)
    if out is None:
        raise RuntimeError(f"GDAL COG driver failed to write {path}")
    out = None  # flush / close
    return path


def cloud_storage(*args, **kwargs):
    """Zarr / S3 / GCS factories — umbrella stub.

    The COG write half is shipped under :func:`write_cog`. Zarr writers
    and S3 / GCS read factories remain deferred pending a follow-up PR.
    """
    raise NotImplementedError(
        "cloud_storage umbrella API deferred. The COG write half is "
        "available via digitalrivers.cloud_io.write_cog."
    )
