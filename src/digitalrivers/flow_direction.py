"""Deprecated import path for the flow-direction raster.

This module moved to :mod:`digitalrivers.flow.direction` in the package restructure. Importing from here still
works and returns the same objects, but warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.flow_direction import FlowDirection

    # after
    from digitalrivers.flow.direction import FlowDirection

The names re-exported from the package root are unaffected: `from digitalrivers import
FlowDirection` was correct before and is still correct.
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
