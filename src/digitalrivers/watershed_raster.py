"""Deprecated import path for the watershed raster.

This module moved to :mod:`digitalrivers.watershed.raster` in the package restructure. Importing
from here still works and warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.watershed_raster import WatershedRaster

    # after
    from digitalrivers.watershed.raster import WatershedRaster

The name re-exported from the package root is unaffected: `from digitalrivers import
WatershedRaster` was correct before and is still correct.

It re-exports `WatershedRaster` and nothing else. Names that this module used to expose
incidentally — it never declared `__all__`, so anything it imported was reachable through it —
are not carried over. `VALID_ROUTING`, `META_CLASS` and `META_ROUTING` now live in
`digitalrivers.core.metadata`; `kahn_max_upslope_length` in
`digitalrivers.flow._kernels.accumulation`.
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
