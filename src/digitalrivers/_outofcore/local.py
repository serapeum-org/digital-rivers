"""Halo-tiled local (stencil) terrain ops — out-of-core, no dask required (B0).

Local terrain derivatives (slope, aspect, hillshade, curvature, TPI/TRI, ...) are *stencils*: each output cell
depends only on a fixed-radius neighbourhood, so they tile trivially with a halo and need no boundary
reconciliation. :func:`tiled_stencil` streams such an op tile-by-tile over the :mod:`_outofcore` tiling primitives
— constant memory regardless of raster size — and is **bit-for-bit identical** to the whole-array result because
each block kernel sees the same neighbourhood (the halo) the whole-array kernel would.

A2 (``pyramids.Dataset.map_overlap``) was declined upstream, so this consumes the existing tiling helpers (B1)
directly instead of dask's ``map_overlap``.
"""

from __future__ import annotations

import numpy as np
from pyramids.dataset import Dataset

from digitalrivers._outofcore.tiling import (
    plan_tiles,
    read_tile,
    require_single_band,
    write_core,
)


def max_slope_2d(elev: np.ndarray, cell_size: float) -> np.ndarray:
    """Maximum downhill D8 slope per cell — pure 2-D kernel (matches ``DEM._get_8_direction_slopes`` + nanmax).

    The block is NaN-padded by one cell, so cells on the block boundary see NaN neighbours; when called on a
    halo-expanded tile, the tile's *core* cells therefore use real neighbours and domain-edge cells fall back to
    NaN exactly as the whole-array computation does.

    Args:
        elev: `(rows, cols)` elevation block (NaN at no-data).
        cell_size: Square cell side length in map units.

    Returns:
        `(rows, cols)` float32 maximum-slope array.

    Examples:
        - The centre of an east-facing ramp drains to its lower (east) neighbour at slope ``(2-1)/1``:
            ```python
            >>> import numpy as np
            >>> from digitalrivers._outofcore.local import max_slope_2d
            >>> z = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
            >>> float(max_slope_2d(z, 1.0)[1, 1])
            1.0

            ```
        - A flat block has zero slope everywhere:
            ```python
            >>> import numpy as np
            >>> from digitalrivers._outofcore.local import max_slope_2d
            >>> float(max_slope_2d(np.zeros((3, 3)), 1.0)[1, 1])
            0.0

            ```
    """
    elev = np.asarray(elev, dtype=np.float32)
    rows, cols = elev.shape
    d = float(cell_size)
    dd = d * np.sqrt(2.0)
    padded = np.full((rows + 2, cols + 2), np.nan, dtype=np.float32)
    padded[1:-1, 1:-1] = elev
    centre = padded[1:-1, 1:-1]
    diffs = (
        (centre - padded[2:, 1:-1], d),  # 0 S
        (centre - padded[2:, :-2], dd),  # 1 SW
        (centre - padded[1:-1, :-2], d),  # 2 W
        (centre - padded[:-2, :-2], dd),  # 3 NW
        (centre - padded[:-2, 1:-1], d),  # 4 N
        (centre - padded[:-2, 2:], dd),  # 5 NE
        (centre - padded[1:-1, 2:], d),  # 6 E
        (centre - padded[2:, 2:], dd),  # 7 SE
    )
    slopes = np.full((rows, cols, 8), np.nan, dtype=np.float32)
    for k, (diff, dist) in enumerate(diffs):
        slopes[:, :, k] = diff / dist
    with np.errstate(invalid="ignore"):
        all_nan = np.all(np.isnan(slopes), axis=2)
        out = np.where(all_nan, np.nan, np.nanmax(slopes, axis=2)).astype(np.float32)
    return out


def tiled_stencil(
    dataset,
    block_fn,
    out_path: str,
    *,
    depth: int = 1,
    tile_size: int | tuple[int, int] = 2048,
    input_nodata: float | None = None,
    out_nodata: float = -9999.0,
    dtype: str = "float32",
):
    """Stream a local stencil op over ``dataset`` tile-by-tile and write the result to ``out_path``.

    Args:
        dataset: Source `pyramids` ``Dataset`` (or subclass).
        block_fn: Callable mapping a 2-D halo-expanded block to a same-shaped result array.
        out_path: GeoTIFF to create and fill.
        depth: Halo (kernel radius) read on each side. Defaults to 1.
        tile_size: Core tile size (int or ``(rows, cols)``). Defaults to 2048.
        input_nodata: If given, cells equal to it are converted to NaN before ``block_fn`` (so kernels that pad
            with NaN treat no-data correctly).
        out_nodata: No-data sentinel for the output raster.
        dtype: Output dtype.

    Returns:
        The result `pyramids` ``Dataset`` opened on ``out_path``.
    """
    require_single_band(dataset)
    rows, cols = dataset.rows, dataset.columns
    tile_rows, tile_cols = (
        (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
    )
    specs = plan_tiles(rows, cols, tile_rows, tile_cols, halo=depth)
    out = Dataset.create_empty(
        rows,
        cols,
        dtype=dtype,
        geo=dataset.geotransform,
        epsg=dataset.epsg,
        no_data_value=out_nodata,
        driver_type="GTiff",
        path=out_path,
    )
    for spec in specs:
        block, core = read_tile(dataset, spec, rows, cols)
        block = np.asarray(block, dtype=np.float64)
        if input_nodata is not None and not np.isnan(input_nodata):
            block = np.where(block == input_nodata, np.nan, block)
        result = np.asarray(block_fn(block))
        core_res = result[core]
        core_res = np.where(np.isnan(core_res), out_nodata, core_res).astype(dtype)
        write_core(out, spec, core_res)
    return out
