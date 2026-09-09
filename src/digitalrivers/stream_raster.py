"""Deprecated import path for the stream raster.

This module moved to :mod:`digitalrivers.streams.raster` in the package restructure. Importing from here still
works and returns the same objects, but warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.stream_raster import StreamRaster

    # after
    from digitalrivers.streams.raster import StreamRaster

The names re-exported from the package root are unaffected: `from digitalrivers import
StreamRaster` was correct before and is still correct.
"""

from __future__ import annotations

import warnings

from digitalrivers.streams.raster import (  # noqa: F401
    StreamRaster,
)

__all__ = [
    "StreamRaster",
]

warnings.warn(
    "digitalrivers.stream_raster has moved to digitalrivers.streams.raster; "
    "this import path is removed in 0.6.0.",
    DeprecationWarning,
    stacklevel=2,
)
