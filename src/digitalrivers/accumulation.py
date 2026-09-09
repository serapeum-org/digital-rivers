"""Deprecated import path for the flow-accumulation raster.

This module moved to :mod:`digitalrivers.flow.accumulation` in the package restructure.
Importing from here still works and warns; the shim is removed in 0.6.0.

    # before
    from digitalrivers.accumulation import Accumulation

    # after
    from digitalrivers.flow.accumulation import Accumulation

The name re-exported from the package root is unaffected: `from digitalrivers import
Accumulation` was correct before and is still correct.

It re-exports `Accumulation` and nothing else. Names that this module used to expose
incidentally — it never declared `__all__`, so anything it imported was reachable through it —
are not carried over. `VALID_ROUTING`, `META_CLASS` and `META_ROUTING` now live in
`digitalrivers.core.metadata`.
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
