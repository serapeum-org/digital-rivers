"""Deprecated import path for the stream raster.

This module moved to :mod:`digitalrivers.streams.raster` in the package restructure. Importing
from here still works and warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.stream_raster import StreamRaster

    # after
    from digitalrivers.streams.raster import StreamRaster

The name re-exported from the package root is unaffected: `from digitalrivers import
StreamRaster` was correct before and is still correct.

It re-exports `StreamRaster` and nothing else. Names that this module used to expose
incidentally — it never declared `__all__`, so anything it imported was reachable through it —
are not carried over. `VALID_ROUTING`, `META_CLASS`, `META_ROUTING` and `META_THRESHOLD` now
live in `digitalrivers.core.metadata`; the `strahler` / `horton` / `shreve` / `hack` /
`topological` ordering kernels in `digitalrivers.streams._kernels.order`.
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
