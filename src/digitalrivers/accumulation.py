"""Deprecated import path for the flow-accumulation raster.

This module moved to :mod:`digitalrivers.flow.accumulation` in the package restructure. Importing from here still
works and returns the same objects, but warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.accumulation import Accumulation

    # after
    from digitalrivers.flow.accumulation import Accumulation

The names re-exported from the package root are unaffected: `from digitalrivers import
Accumulation` was correct before and is still correct.
"""

from __future__ import annotations

import warnings

from digitalrivers.flow.accumulation import (  # noqa: F401
    Accumulation,
)

__all__ = [
    "Accumulation",
]

warnings.warn(
    "digitalrivers.accumulation has moved to digitalrivers.flow.accumulation; "
    "this import path is removed in 0.6.0.",
    DeprecationWarning,
    stacklevel=2,
)
