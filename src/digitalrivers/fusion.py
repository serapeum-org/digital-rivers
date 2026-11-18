"""Deprecated import path for topobathy fusion.

This module moved to :mod:`digitalrivers.dem.fusion` in the package restructure. Importing from
here still works and warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.fusion import topobathy_fusion

    # after
    from digitalrivers.dem.fusion import topobathy_fusion

`topobathy_fusion` has never been re-exported from the package root, so `from digitalrivers
import topobathy_fusion` raises `ImportError` on this version and on every earlier one. Import
it from `digitalrivers.dem.fusion`.

It re-exports `topobathy_fusion` and nothing else. Names that this module used to expose
incidentally — it never declared `__all__`, so anything it imported was reachable through it —
are not carried over.
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
