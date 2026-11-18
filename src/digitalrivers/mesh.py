"""Deprecated import path for the triangle-mesh container.

This module moved to :mod:`digitalrivers.interop.mesh` in the package restructure. Importing
from here still works and warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.mesh import Mesh

    # after
    from digitalrivers.interop.mesh import Mesh

The name re-exported from the package root is unaffected: `from digitalrivers import Mesh` was
correct before and is still correct.

It re-exports `Mesh` and nothing else. Names that this module used to expose incidentally — it
never declared `__all__`, so anything it imported was reachable through it — are not carried
over.
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
