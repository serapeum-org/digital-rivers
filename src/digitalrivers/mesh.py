"""Deprecated import path for the triangle-mesh container.

This module moved to :mod:`digitalrivers.interop.mesh` in the package restructure. Importing from here still
works and returns the same objects, but warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.mesh import Mesh

    # after
    from digitalrivers.interop.mesh import Mesh

The names re-exported from the package root are unaffected: `from digitalrivers import
Mesh` was correct before and is still correct.
"""

from __future__ import annotations

import warnings

from digitalrivers.interop.mesh import (  # noqa: F401
    Mesh,
)

__all__ = [
    "Mesh",
]

warnings.warn(
    "digitalrivers.mesh has moved to digitalrivers.interop.mesh; "
    "this import path is removed in 0.6.0.",
    DeprecationWarning,
    stacklevel=2,
)
