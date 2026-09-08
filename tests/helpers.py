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
