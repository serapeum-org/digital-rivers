"""Deprecated import path for the watershed raster.

This module moved to :mod:`digitalrivers.watershed.raster` in the package restructure. Importing from here still
works and returns the same objects, but warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.watershed_raster import WatershedRaster

    # after
    from digitalrivers.watershed.raster import WatershedRaster

The names re-exported from the package root are unaffected: `from digitalrivers import
WatershedRaster` was correct before and is still correct.
"""

from __future__ import annotations

import warnings

from digitalrivers.watershed.raster import (  # noqa: F401
    WatershedRaster,
)

__all__ = [
    "WatershedRaster",
]

warnings.warn(
    "digitalrivers.watershed_raster has moved to digitalrivers.watershed.raster; "
    "this import path is removed in 0.6.0.",
    DeprecationWarning,
    stacklevel=2,
)
