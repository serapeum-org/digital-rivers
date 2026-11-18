"""Deprecated import path for the COG / tiled-window helpers.

This module moved to :mod:`digitalrivers.interop.cloud_io` in the package restructure. Importing
from here still works and warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.cloud_io import tile_windows

    # after
    from digitalrivers.interop.cloud_io import tile_windows

`tile_windows` has never been re-exported from the package root, so `from digitalrivers import
tile_windows` raises `ImportError` on this version and on every earlier one. Import it from
`digitalrivers.interop.cloud_io`.

It re-exports `tile_windows`, `write_cog`, `dask_backend` and `cloud_storage` and nothing else.
Names that this module used to expose incidentally — it never declared `__all__`, so anything it
imported was reachable through it — are not carried over.
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
