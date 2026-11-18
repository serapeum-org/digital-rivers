"""Deprecated import path for the flow-direction raster.

This module moved to :mod:`digitalrivers.flow.direction` in the package restructure. Importing
from here still works and warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.flow_direction import FlowDirection

    # after
    from digitalrivers.flow.direction import FlowDirection

The name re-exported from the package root is unaffected: `from digitalrivers import
FlowDirection` was correct before and is still correct.

It re-exports `FlowDirection` and nothing else. Names that this module used to expose
incidentally — it never declared `__all__`, so anything it imported was reachable through it —
are not carried over. `VALID_ROUTING`, `VALID_ENCODING`, `META_CLASS`, `META_ROUTING` and
`META_ENCODING` now live in `digitalrivers.core.metadata`; `ihu_upscale`,
`kahn_max_upslope_length` and `watershed_d8` in the private kernel packages.
"""

from __future__ import annotations

import warnings

from digitalrivers.flow.direction import (  # noqa: F401
    FlowDirection,
)

__all__ = [
    "FlowDirection",
]

warnings.warn(
    "digitalrivers.flow_direction has moved to digitalrivers.flow.direction; "
    "this import path is removed in 0.6.0.",
    DeprecationWarning,
    stacklevel=2,
)
