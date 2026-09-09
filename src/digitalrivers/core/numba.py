"""Numba availability shim: the `njit` / `prange` decorators and the enabled predicate.

Every JIT kernel in the package decorates with the `njit` exported here rather than
importing `numba` directly, so there is exactly one place that decides whether the JIT
path is live.

Disabling Numba
---------------

Set `DIGITALRIVERS_DISABLE_NUMBA=1` (or simply have no Numba installed) and the decorators
degrade to no-ops, so the decorated functions run as plain Python and produce bit-for-bit
identical output. Used for debugging (step-through in an IDE) and for CI on platforms
without Numba wheels.

The flag is read once, at import. Tests that flip the env var must therefore drop this
module from `sys.modules` — along with any kernel module that imported these decorators —
and import it again, not merely re-read the variable.

**This module imports `numba` eagerly.** `import digitalrivers` must not pull Numba into
the process, so nothing may import it at module scope outside the `_kernels` layer; the
kernel modules that do are themselves only ever imported lazily, inside the function that
needs them.
"""

from __future__ import annotations

import os

__all__ = ["njit", "prange", "is_numba_enabled"]

_USE_NUMBA = os.environ.get("DIGITALRIVERS_DISABLE_NUMBA", "0") != "1"

if _USE_NUMBA:
    try:
        from numba import njit, prange  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — environment without numba
        _USE_NUMBA = False

if not _USE_NUMBA:

    def njit(*args, **kwargs):  # type: ignore[no-redef]  # noqa: F811
        """No-op decorator used when Numba is unavailable / disabled."""
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    def prange(*args, **kwargs):  # type: ignore[no-redef]  # noqa: F811
        """Plain `range`, standing in for Numba's parallel range."""
        return range(*args, **kwargs)


def is_numba_enabled() -> bool:
    """Return True if Numba JIT is currently active for this process."""
    return _USE_NUMBA
