"""The 8-neighbour direction convention shared by every routing, ordering, and walk kernel.

Direction codes run `0=S, 1=SW, 2=W, 3=NW, 4=N, 5=NE, 6=E, 7=SE`. This is the convention
digital-rivers writes into rasters tagged `DR_ENCODING="digitalrivers"`, and the one every
docstring in the package means when it says "`DIR_OFFSETS` order".

Two shapes of the same table are exported, because callers genuinely need different ones:

* `DIR_OFFSETS` — `{direction: (column_offset, row_offset)}`. Note the `(col, row)` order:
  it is the transpose of the array form below. Both encode the same convention (direction
  `0` is south, i.e. `row + 1`); they differ only in argument order. Do not re-order
  either form — `DEM.accumulate_flow` and `DEM.convert_flow_direction_to_cell_indices`
  unpack the tuples positionally.
* `DIR_DR_* / DIR_DC_*` — parallel `(row, col)` offset arrays, one pair per dtype. Numba
  compiles a separate specialisation per argument dtype, so the `int8` and `int32`
  variants are both real and neither is redundant: the vectorised routing kernels in
  `flow._kernels.routing` work in `int8`, while every `@njit` kernel and every
  pure-Python walk works in `int32`. **Pass the variant the call site already used** —
  swapping one for the other silently changes a kernel's compiled signature.

`INV_DIR[k]` is the direction code a neighbour sitting at offset `k` would carry if it
flowed *into* the centre cell. Every upstream (reverse) walk indexes it.

These arrays are module-level and therefore shared by every importer. Treat them as
immutable — nothing in the package writes to them. They are deliberately left writable
rather than flagged read-only, because `ndarray.flags.writeable` is part of the type Numba
compiles against, and flipping it would change the signature of every `@njit` kernel that
takes them.

Not every 8-direction table in the package belongs here:
`dem._kernels.morphometry.horizon_walk_kernel` walks in *azimuth* order
(E, NE, N, NW, W, SW, S, SE), a different labelling that happens to cover the same eight
neighbours. It stays local to that kernel.

The forms are cross-checked by `tests/core/test_directions.py`; the examples below are
the same invariants written so they run as doctests.

Examples:
    - The dict form and the array form describe the same eight neighbours:

        >>> from digitalrivers.core.directions import DIR_OFFSETS
        >>> from digitalrivers.core.directions import DIR_DR_I32, DIR_DC_I32
        >>> pairs = list(zip(DIR_DC_I32.tolist(), DIR_DR_I32.tolist()))
        >>> [DIR_OFFSETS[k] for k in range(8)] == pairs
        True

    - Reversing a direction twice is the identity:

        >>> from digitalrivers.core.directions import INV_DIR
        >>> INV_DIR[INV_DIR].tolist() == list(range(8))
        True

    - And a direction's offsets cancel against its inverse's:

        >>> all(
        ...     DIR_DR_I32[k] + DIR_DR_I32[INV_DIR[k]] == 0
        ...     and DIR_DC_I32[k] + DIR_DC_I32[INV_DIR[k]] == 0
        ...     for k in range(8)
        ... )
        True
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DIR_OFFSETS",
    "DIR_DR_I8",
    "DIR_DC_I8",
    "DIR_DR_I32",
    "DIR_DC_I32",
    "INV_DIR",
]

#: D8 direction offsets mapping direction index to `(column_offset, row_offset)`.
#:
#: Directions follow the convention:
#:   0 = South (bottom), 1 = Southwest (bottom-left), 2 = West (left),
#:   3 = Northwest (top-left), 4 = North (top), 5 = Northeast (top-right),
#:   6 = East (right), 7 = Southeast (bottom-right).
DIR_OFFSETS = {
    0: (0, 1),  # bottom
    1: (-1, 1),  # bottom left
    2: (-1, 0),  # left
    3: (-1, -1),  # top left
    4: (0, -1),  # top
    5: (1, -1),  # top right
    6: (1, 0),  # right
    7: (1, 1),  # bottom right
}

# The same convention as row / column deltas, in DIR_OFFSETS index order.
_DR = [1, 1, 0, -1, -1, -1, 0, 1]
_DC = [0, -1, -1, -1, 0, 1, 1, 1]

#: Row offsets as `int8` — the dtype `flow._kernels.routing`'s vectorised kernels use.
DIR_DR_I8 = np.array(_DR, dtype=np.int8)
#: Column offsets as `int8`.
DIR_DC_I8 = np.array(_DC, dtype=np.int8)

#: Row offsets as `int32` — the dtype every `@njit` kernel is compiled against.
DIR_DR_I32 = np.array(_DR, dtype=np.int32)
#: Column offsets as `int32`.
DIR_DC_I32 = np.array(_DC, dtype=np.int32)

#: `INV_DIR[k]` is the direction code of a neighbour at offset `k` flowing *into* centre.
INV_DIR = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int32)
