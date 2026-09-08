"""Shared constructors for test rasters.

Thirty test modules each carried their own byte-identical `_make_dem`, which is
the bulk of the suite's duplicated-code measure. They live here instead, so the
raster-building convention is stated once.
"""

from __future__ import annotations

import numpy as np
from pyramids.dataset import Dataset, GeoReference

from digitalrivers import DEM

NO_DATA = -9999.0
"""Sentinel every test raster declares, matching the fixtures on disk."""


def make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    """Build an in-memory `DEM` from a 2-D elevation array.

    `NaN` cells are written out as the `NO_DATA` sentinel, so an array can be
    authored with `np.nan` holes and still round-trip through GDAL.

    Args:
        arr: 2-D elevation array. Cast to `float32`; the caller's array is not
            modified.
        cell_size: Square cell size in the raster's CRS units. Defaults to 1.0.

    Returns:
        A `DEM` over a fresh in-memory raster at the origin, in EPSG:4326.
    """
    disk = arr.astype(np.float32, copy=True)
    disk[np.isnan(disk)] = NO_DATA
    ds = Dataset.from_array(
        disk,
        geo_ref=GeoReference(
            top_left_corner=(0.0, 0.0),
            cell_size=cell_size,
            epsg=4326,
        ),
        no_data_value=NO_DATA,
    )
    return DEM(ds.raster)


def channel_z() -> np.ndarray:
    """Return the 3x6 single-channel elevation grid the suite tests against.

    Row 1 falls monotonically west to east (`5 4 3 2 1`) between two walls of
    `9`, so D8 routes the whole row to the outlet at `(1, 5)` and every
    derivative — flow direction, accumulation, streams, watersheds, ordering —
    has one unambiguous answer.

    A fresh array is returned on every call rather than a shared constant, so a
    test that conditions or burns into the grid cannot affect any other.

    Returns:
        A `(3, 6)` `float32` array.
    """
    return np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )


def twin_channel_z() -> np.ndarray:
    """Return the 6x6 two-channel grid the upscaling tests use.

    Two identical parallel channels (rows 2 and 3) between walls of `9`. The pair
    is what makes a 2x upscale meaningful: both fine rows fall into one coarse
    row, so the coarse flow direction has an unambiguous answer to check against.

    A fresh array is returned on every call, so a test that conditions the grid
    cannot affect any other.

    Returns:
        A `(6, 6)` `float32` array.
    """
    return np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
