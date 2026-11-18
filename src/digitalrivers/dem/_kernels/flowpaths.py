"""Pure-numpy walks over a D8 direction grid.

The array kernels behind `RoutingMixin`'s legacy per-cell surface. Each takes a direction
grid and returns a value or an array, so they can be exercised on a hand-written 3x3
fixture rather than through a raster.

`digitalrivers.flow._kernels.accumulation` holds the vectorised accumulation the typed
classes use; what is here is the older per-cell path that `DEM.accumulate_flow` and
`DEM.convert_flow_direction_to_cell_indices` still expose.

Numba-free by design — the mixin imports these at module scope.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "opposite_direction",
    "accumulate_upstream",
    "cell_indices_from_directions",
]


def opposite_direction(dr: int, dc: int, dir_offsets: dict) -> int | None:
    """Return the direction code that points back along `(dr, dc)`.

    Used to decide whether a neighbour drains *into* the centre cell: a neighbour sitting
    at offset `(dr, dc)` flows inward exactly when its own direction code equals this.

    Args:
        dr: Row offset to the neighbour.
        dc: Column offset to the neighbour.
        dir_offsets: The direction table, mapping code to `(col_offset, row_offset)`.

    Returns:
        The direction code pointing from the neighbour back to the centre, or `None` when
        `(dr, dc)` names no neighbour in the table.

    Examples:
        - South and north are opposites:

            >>> from digitalrivers.core.directions import DIR_OFFSETS
            >>> from digitalrivers.dem._kernels.flowpaths import opposite_direction
            >>> opposite_direction(1, 0, DIR_OFFSETS)
            4

        - and reversing twice returns the original code:

            >>> from digitalrivers.core.directions import DIR_OFFSETS
            >>> from digitalrivers.dem._kernels.flowpaths import opposite_direction
            >>> col, row = DIR_OFFSETS[5]
            >>> back = opposite_direction(row, col, DIR_OFFSETS)
            >>> col2, row2 = DIR_OFFSETS[back]
            >>> opposite_direction(row2, col2, DIR_OFFSETS)
            5

        - An offset that is not a neighbour has no opposite:

            >>> from digitalrivers.core.directions import DIR_OFFSETS
            >>> from digitalrivers.dem._kernels.flowpaths import opposite_direction
            >>> opposite_direction(5, 5, DIR_OFFSETS) is None
            True
    """
    for d, (d_col, d_row) in dir_offsets.items():
        if d_row == -dr and d_col == -dc:
            return d
    return None


def accumulate_upstream(
    r: int, c: int, flow_dir: np.ndarray, acc: np.ndarray, dir_offsets: dict
) -> int:
    """Count the cells draining into `(r, c)`, memoising into `acc`.

    A depth-first traversal over an explicit stack rather than recursion — the grids this
    runs on are large enough that a recursive walk would need the interpreter's recursion
    limit raised, which is how this code used to work.

    `acc` doubles as the memo: any cell already holding a non-negative count is taken as
    final and not revisited, so the whole grid costs one pass no matter how the calls are
    ordered.

    Args:
        r: Row of the cell to count for.
        c: Column of the cell to count for.
        flow_dir: `(rows, cols)` direction-code grid.
        acc: `(rows, cols)` accumulator, negative where not yet computed. **Modified in
            place** — that is the memo.
        dir_offsets: The direction table, mapping code to `(col_offset, row_offset)`.

    Returns:
        The number of cells upstream of `(r, c)`, excluding the cell itself. Off-grid
        coordinates return 0.

    Examples:
        - A three-cell chain draining east accumulates along its length:

            >>> import numpy as np
            >>> from digitalrivers.core.directions import DIR_OFFSETS
            >>> from digitalrivers.dem._kernels.flowpaths import accumulate_upstream
            >>> fdir = np.array([[6, 6, -1]])
            >>> acc = np.full(fdir.shape, -1)
            >>> int(accumulate_upstream(0, 2, fdir, acc, DIR_OFFSETS))
            2

        - A cell with nothing draining into it counts zero:

            >>> import numpy as np
            >>> from digitalrivers.core.directions import DIR_OFFSETS
            >>> from digitalrivers.dem._kernels.flowpaths import accumulate_upstream
            >>> fdir = np.array([[6, 6, -1]])
            >>> int(accumulate_upstream(0, 0, fdir, np.full(fdir.shape, -1), DIR_OFFSETS))
            0

        - Coordinates off the grid are 0 rather than an error:

            >>> import numpy as np
            >>> from digitalrivers.core.directions import DIR_OFFSETS
            >>> from digitalrivers.dem._kernels.flowpaths import accumulate_upstream
            >>> fdir = np.array([[6, 6, -1]])
            >>> accumulate_upstream(-1, 0, fdir, np.full(fdir.shape, -1), DIR_OFFSETS)
            0
    """
    rows, cols = flow_dir.shape

    if not (0 <= r < rows and 0 <= c < cols):
        return 0
    if acc[r, c] >= 0:
        return acc[r, c]

    offsets_list = [
        (d_col, d_row, opposite_direction(d_row, d_col, dir_offsets))
        for d_col, d_row in dir_offsets.values()
    ]

    stack = [(r, c, 0, 0)]

    while stack:
        cr, cc, idx, total = stack[-1]

        if acc[cr, cc] >= 0:
            stack.pop()
            if stack:
                pr, pc, pidx, ptotal = stack[-1]
                stack[-1] = (pr, pc, pidx, ptotal + acc[cr, cc] + 1)
            continue

        # Advance through remaining neighbours.
        found_unprocessed = False
        while idx < len(offsets_list):
            d_col, d_row, opp = offsets_list[idx]
            idx += 1
            rr, rc = cr + d_row, cc + d_col
            if not (0 <= rr < rows and 0 <= rc < cols):
                continue
            if flow_dir[rr, rc] != opp:
                continue
            if opp is None:
                continue
            # Neighbour already computed — just add its count.
            if acc[rr, rc] >= 0:
                total += acc[rr, rc] + 1
                continue
            # Neighbour needs processing — save our state and push it.
            stack[-1] = (cr, cc, idx, total)
            stack.append((rr, rc, 0, 0))
            found_unprocessed = True
            break

        if not found_unprocessed:
            # All neighbours processed — finalise this cell.
            acc[cr, cc] = total
            stack.pop()
            if stack:
                pr, pc, pidx, ptotal = stack[-1]
                stack[-1] = (pr, pc, pidx, ptotal + total + 1)

    return acc[r, c]


def cell_indices_from_directions(flow_dir: np.ndarray, dir_offsets: dict) -> np.ndarray:
    """Resolve each cell's direction code into the `(row, col)` it drains to.

    Args:
        flow_dir: `(rows, cols)` float direction grid. `NaN` marks no-data; every finite
            value must be a code `0..7`.
        dir_offsets: The direction table, mapping code to `(col_offset, row_offset)`.

    Returns:
        `(rows, cols, 2)` float64 array. `[..., 0]` is the downstream **column** and
        `[..., 1]` the downstream **row** — that order, matching the reference fixture
        and `tests/conftest.py`. Both are `NaN` where the input was no-data. Float rather
        than integer precisely so the no-data cells can stay `NaN` rather than take a
        sentinel that could be mistaken for an index.

    Examples:
        - A cell at the origin flowing east drains to column 1, row 0 — reported in
          that order:

            >>> import numpy as np
            >>> from digitalrivers.core.directions import DIR_OFFSETS
            >>> from digitalrivers.dem._kernels.flowpaths import (
            ...     cell_indices_from_directions,
            ... )
            >>> out = cell_indices_from_directions(np.array([[6.0, 6.0]]), DIR_OFFSETS)
            >>> out[0, 0].tolist()
            [1.0, 0.0]

        - and one flowing south drains to column 0, row 1:

            >>> import numpy as np
            >>> from digitalrivers.core.directions import DIR_OFFSETS
            >>> from digitalrivers.dem._kernels.flowpaths import (
            ...     cell_indices_from_directions,
            ... )
            >>> out = cell_indices_from_directions(np.array([[0.0]]), DIR_OFFSETS)
            >>> out[0, 0].tolist()
            [0.0, 1.0]

        - No-data cells stay NaN rather than resolving to a neighbour:

            >>> import numpy as np
            >>> from digitalrivers.core.directions import DIR_OFFSETS
            >>> from digitalrivers.dem._kernels.flowpaths import (
            ...     cell_indices_from_directions,
            ... )
            >>> out = cell_indices_from_directions(np.array([[np.nan]]), DIR_OFFSETS)
            >>> bool(np.isnan(out[0, 0, 0]))
            True
    """
    rows, cols = flow_dir.shape
    valid = ~np.isnan(flow_dir)

    # DIR_OFFSETS is (col_offset, row_offset), and index 0 is added to the row index
    # while index 1 is added to the column index. That looks transposed and is: the net
    # effect is that layer 0 comes out as the downstream column and layer 1 as the row.
    # It is the long-standing behaviour the reference fixture encodes, so the axes are
    # named accordingly above rather than the arithmetic being "corrected" here.
    offset_0 = np.array([dir_offsets[d][0] for d in range(8)], dtype=np.float64)
    offset_1 = np.array([dir_offsets[d][1] for d in range(8)], dtype=np.float64)

    flow_direction_cell = np.full((rows, cols, 2), np.nan, dtype=np.float64)

    dir_idx = flow_dir[valid].astype(int)
    row_idx, col_idx = np.nonzero(valid)
    flow_direction_cell[valid, 0] = row_idx + offset_0[dir_idx]
    flow_direction_cell[valid, 1] = col_idx + offset_1[dir_idx]

    return flow_direction_cell
