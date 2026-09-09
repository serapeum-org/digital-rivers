"""Deprecated import path for the COG / tiled-window helpers.

This module moved to :mod:`digitalrivers.interop.cloud_io` in the package restructure. Importing from here still
works and returns the same objects, but warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.cloud_io import tile_windows

    # after
    from digitalrivers.interop.cloud_io import tile_windows

The names re-exported from the package root are unaffected: `from digitalrivers import
tile_windows` was correct before and is still correct.
"""

from __future__ import annotations

import warnings

from digitalrivers.interop.cloud_io import (  # noqa: F401
    tile_windows,
    write_cog,
    dask_backend,
    cloud_storage,
)

__all__ = [
    "tile_windows",
    "write_cog",
    "dask_backend",
    "cloud_storage",
]

warnings.warn(
    "digitalrivers.cloud_io has moved to digitalrivers.interop.cloud_io; "
    "this import path is removed in 0.6.0.",
    DeprecationWarning,
    stacklevel=2,
)
