"""Shared, hydrology-agnostic primitives that every domain in the package builds on.

Three modules, deliberately kept free of any dependency on the rest of `digitalrivers`
(this package is the bottom of the stack — nothing here may import a domain module):

* :mod:`directions` — the 8-neighbour direction convention: `DIR_OFFSETS` plus the
  parallel `(row, col)` offset arrays every routing, ordering, and walk kernel indexes.
* :mod:`metadata` — the `DR_*` GeoTIFF tag keys and valid-value sets the typed result
  classes write, plus `resolve_no_val`.
* :mod:`numba` — the Numba availability shim: the `njit` / `prange` decorators (real or
  no-op) and `is_numba_enabled`.

Nothing is re-exported here on purpose. `core.numba` imports `numba` eagerly, and
`import digitalrivers` must not pull Numba into the process, so importing this package
must not drag that module in with it. Import the submodule you need directly.
"""
