"""Deprecated import path for topobathy fusion.

This module moved to :mod:`digitalrivers.dem.fusion` in the package restructure. Importing from here still
works and returns the same objects, but warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.fusion import topobathy_fusion

    # after
    from digitalrivers.dem.fusion import topobathy_fusion

The names re-exported from the package root are unaffected: `from digitalrivers import
topobathy_fusion` was correct before and is still correct.
"""

from __future__ import annotations

import warnings

from digitalrivers.dem.fusion import (  # noqa: F401
    topobathy_fusion,
)

__all__ = [
    "topobathy_fusion",
]

warnings.warn(
    "digitalrivers.fusion has moved to digitalrivers.dem.fusion; "
    "this import path is removed in 0.6.0.",
    DeprecationWarning,
    stacklevel=2,
)
