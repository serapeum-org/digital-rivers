"""Tile planning and halo-aware tile I/O over `pyramids`.

The building blocks every tiled algorithm shares:

* :class:`TileSpec` — one tile's geometry: its **core** region (the cells it owns and writes back) and the
  **halo** ring (core + ``halo`` cells on each side, clipped at the domain edge) read in for neighbour context.
* :func:`plan_tiles` — partition a ``rows × cols`` raster into row-major :class:`TileSpec` tiles.
* :func:`read_tile` / :func:`write_core` — read a tile's halo-expanded window and write a tile's core back, each
  using the **correct** `pyramids` window convention (see below).
* :func:`gid` — an overflow-safe global cell id (``int64``) for keying perimeter graphs past 2³¹ cells.

.. note::
   `pyramids` uses **two different** window conventions, so callers must not pass a window tuple from one API to
   the other:

   * ``Dataset.read_array(window=...)`` takes GDAL order ``(xoff, yoff, xsize, ysize)`` = ``(col, row, n_cols,
     n_rows)``.
   * ``Dataset.write_array(window=...)`` takes ``(row_off, col_off, n_rows, n_cols)``.

   :func:`read_tile` and :func:`write_core` encapsulate that asymmetry so the rest of the package can think purely
   in ``(row, col, n_rows, n_cols)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    """Geometry of one tile in a row-major tile grid.

    Attributes:
        tid: Flat tile id, ``0 .. n_tiles - 1`` in row-major order.
        row: Tile row index in the tile grid.
        col: Tile column index in the tile grid.
        row_off: Row origin of the **core** region in the full raster.
        col_off: Column origin of the **core** region in the full raster.
        n_rows: Core height in cells (edge tiles are clipped).
        n_cols: Core width in cells (edge tiles are clipped).
        halo: Neighbour ring read on each side, clipped at the domain edge. ``1`` is enough for D8 (each cell
            needs its 8 neighbours); local stencils use their kernel radius.
    """

    tid: int
    row: int
    col: int
    row_off: int
    col_off: int
    n_rows: int
    n_cols: int
    halo: int = 1

    def halo_bounds(self, full_rows: int, full_cols: int) -> tuple[int, int, int, int]:
        """Return the half-open halo-expanded bounds ``(r0, c0, r1, c1)`` clipped to the domain.

        The halo window is ``[r0:r1, c0:c1]`` in full-raster coordinates.

        Examples:
            >>> TileSpec(0, 0, 0, 2, 2, 3, 3, halo=1).halo_bounds(10, 10)
            (1, 1, 6, 6)
            >>> TileSpec(0, 0, 0, 0, 0, 3, 3, halo=1).halo_bounds(10, 10)
            (0, 0, 4, 4)
        """
        r0 = max(0, self.row_off - self.halo)
        c0 = max(0, self.col_off - self.halo)
        r1 = min(full_rows, self.row_off + self.n_rows + self.halo)
        c1 = min(full_cols, self.col_off + self.n_cols + self.halo)
        return r0, c0, r1, c1

    def core_slice(self, full_rows: int, full_cols: int) -> tuple[slice, slice]:
        """Return the ``(rows, cols)`` slice that selects the core out of a halo-expanded array.

        Examples:
            >>> TileSpec(0, 0, 0, 2, 2, 3, 3, halo=1).core_slice(10, 10)
            (slice(1, 4, None), slice(1, 4, None))
            >>> TileSpec(0, 0, 0, 0, 0, 3, 3, halo=1).core_slice(10, 10)
            (slice(0, 3, None), slice(0, 3, None))
        """
        r0, c0, _, _ = self.halo_bounds(full_rows, full_cols)
        rs = self.row_off - r0
        cs = self.col_off - c0
        return slice(rs, rs + self.n_rows), slice(cs, cs + self.n_cols)


def plan_tiles(
    rows: int,
    cols: int,
    tile_rows: int,
    tile_cols: int,
    halo: int = 1,
) -> list[TileSpec]:
    """Partition a ``rows × cols`` raster into row-major :class:`TileSpec` tiles.

    The cores tile the domain exactly — no overlap, edge tiles clipped — mirroring
    ``digitalrivers.cloud_io.tile_windows`` but carrying a ``halo`` for neighbour context.

    Args:
        rows: Full raster height in cells.
        cols: Full raster width in cells.
        tile_rows: Core tile height in cells.
        tile_cols: Core tile width in cells.
        halo: Neighbour ring carried on each :class:`TileSpec`. Defaults to 1 (D8).

    Returns:
        Row-major list of :class:`TileSpec`.

    Raises:
        ValueError: If ``tile_rows``/``tile_cols`` are not positive or ``halo`` is negative.

    Examples:
        >>> specs = plan_tiles(4, 5, 2, 3, halo=0)
        >>> [(s.row_off, s.col_off, s.n_rows, s.n_cols) for s in specs]
        [(0, 0, 2, 3), (0, 3, 2, 2), (2, 0, 2, 3), (2, 3, 2, 2)]
        >>> [s.tid for s in specs]
        [0, 1, 2, 3]
    """
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile_rows and tile_cols must be positive")
    if halo < 0:
        raise ValueError("halo must be >= 0")

    specs: list[TileSpec] = []
    tid = 0
    for trow, r_off in enumerate(range(0, rows, tile_rows)):
        n_r = min(tile_rows, rows - r_off)
        for tcol, c_off in enumerate(range(0, cols, tile_cols)):
            n_c = min(tile_cols, cols - c_off)
            specs.append(TileSpec(tid, trow, tcol, r_off, c_off, n_r, n_c, halo))
            tid += 1
    return specs


def read_tile(dataset, spec: TileSpec, full_rows: int, full_cols: int):
    """Read a tile's halo-expanded window and return ``(array, core_slice)``.

    Issues one bounded ``Dataset.read_array(window=...)`` over the halo window, translating to the GDAL
    ``(xoff, yoff, xsize, ysize)`` order that `read_array` expects. The returned ``core_slice`` selects the
    tile's core back out of the halo-expanded ``array``.

    Args:
        dataset: A `pyramids` ``Dataset`` (or subclass) to read from.
        spec: The tile to read.
        full_rows: Full raster height (for halo clipping).
        full_cols: Full raster width (for halo clipping).

    Returns:
        Tuple ``(array, core_slice)`` where ``array`` is the 2-D halo-expanded tile and ``core_slice`` is the
        ``(rows, cols)`` slice of its core.
    """
    r0, c0, r1, c1 = spec.halo_bounds(full_rows, full_cols)
    n_r, n_c = r1 - r0, c1 - c0
    # read_array uses GDAL order: (xoff, yoff, xsize, ysize) == (col, row, n_cols, n_rows)
    arr = np.asarray(dataset.read_array(window=(c0, r0, n_c, n_r)))
    return arr, spec.core_slice(full_rows, full_cols)


def write_core(dataset, spec: TileSpec, core_array: np.ndarray) -> None:
    """Write a tile's core region back into ``dataset`` at the tile's core offset.

    Translates to the ``(row_off, col_off, n_rows, n_cols)`` order that ``Dataset.write_array`` expects (note:
    this differs from the order :func:`read_tile` passes to ``read_array``).

    Args:
        dataset: A disk-backed `pyramids` ``Dataset`` (e.g. from ``Dataset.create_empty``).
        spec: The tile whose core is being written.
        core_array: The core data, shape ``(spec.n_rows, spec.n_cols)``.

    Raises:
        ValueError: If ``core_array`` does not match the tile core shape.
    """
    if core_array.shape != (spec.n_rows, spec.n_cols):
        raise ValueError(
            f"core_array shape {core_array.shape} != tile core ({spec.n_rows}, {spec.n_cols})"
        )
    # write_array uses (row_off, col_off, n_rows, n_cols)
    dataset.write_array(
        core_array, window=(spec.row_off, spec.col_off, spec.n_rows, spec.n_cols)
    )


def require_single_band(dataset) -> None:
    """Raise ``ValueError`` unless ``dataset`` is single-band.

    The tiled algorithms assume `read_array` returns a 2-D array; a multi-band raster would yield a 3-D block and
    confusing downstream failures, so reject it up front with a clear message.
    """
    bands = getattr(dataset, "band_count", 1)
    if bands != 1:
        raise ValueError(
            f"out-of-core tiled operations require a single-band raster; got {bands} bands "
            "(select one band first)"
        )


def gid(row: int, col: int, full_cols: int) -> int:
    """Return the overflow-safe global cell id ``row * full_cols + col`` as a Python ``int`` (int64-safe).

    Perimeter graphs key on this. ``full_cols * full_rows`` can exceed 2³¹ for continental DEMs (e.g. 51810² ≈
    2.7e9), so the multiply is done in Python ``int`` (arbitrary precision) rather than ``int32``.

    Examples:
        >>> gid(2, 3, 8)
        19
        >>> gid(50000, 50000, 51810) > 2**31
        True
    """
    return int(row) * int(full_cols) + int(col)
