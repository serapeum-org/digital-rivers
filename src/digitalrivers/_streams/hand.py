"""Height Above Nearest Drainage (HAND) — Rennó 2008 / Nobre 2011 (P11).

Given a filled DEM, a single-direction flow-direction raster (D8 / Rho8) and a
binary stream mask, HAND assigns every land cell its vertical drop along its
flow path to the nearest stream cell:

    HAND[cell] = elev[cell] - elev[drain_cell]

where `drain_cell` is the first stream cell encountered when following the
flow direction from `cell`. Stream cells themselves have HAND = 0; cells
whose flow path does not reach a stream (sinks, orphans) get NaN.

The implementation uses the pyflwdir telescoping-sum trick: rather than
remembering the target drain cell, accumulate `dz` along the flow path. By
the fundamental theorem of summation, `Σ dz = elev[cell] - elev[drain]`.
Memoised iteratively so each cell is visited at most twice.
"""

from __future__ import annotations

import numpy as np

_DIR_DR = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_DIR_DC = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)


def hand_d8(
    elev: np.ndarray,
    fdir: np.ndarray,
    stream_mask: np.ndarray,
) -> np.ndarray:
    """Compute HAND under D8 / Rho8 routing.

    Args:
        elev: `(rows, cols)` float DEM. NaN cells are treated as no-data.
        fdir: `(rows, cols)` int D8 direction-code raster. Codes outside
            `[0, 7]` are treated as sinks.
        stream_mask: `(rows, cols)` bool — True at stream cells (HAND = 0).

    Returns:
        `(rows, cols)` float64 HAND raster. Stream cells = 0; cells with no
        flow path to a stream (orphans, sinks, no-data) = NaN.

    Examples:
        - Two top-row cells drain south (direction code `0`) into the
          bottom row, which is the stream. HAND at each top-row cell is the
          drop from that cell to the stream cell directly below it
          (`10 - 2 = 8` and `8 - 1 = 7`); stream cells themselves are 0:

            >>> import numpy as np
            >>> elev = np.array([
            ...     [10., 8.],
            ...     [2., 1.],
            ... ])
            >>> fdir = np.array([[0, 0], [-1, -1]], dtype=np.int32)
            >>> stream_mask = np.array([[False, False], [True, True]])
            >>> hand = hand_d8(elev, fdir, stream_mask)
            >>> float(hand[0, 0]), float(hand[0, 1])
            (8.0, 7.0)
            >>> float(hand[1, 0])
            0.0

        - An east-flowing chain with no stream cells is an orphan; the entire
          chain returns NaN (and the unreachable memo keeps the walk linear):

            >>> import numpy as np
            >>> elev = np.array([[4.0, 3.0, 2.0, 1.0]])
            >>> fdir = np.array([[6, 6, 6, -1]], dtype=np.int32)
            >>> stream_mask = np.zeros((1, 4), dtype=bool)
            >>> out = hand_d8(elev, fdir, stream_mask)
            >>> bool(np.isnan(out).all())
            True
    """
    rows, cols = elev.shape
    hand = np.full((rows, cols), np.nan, dtype=np.float64)
    hand[stream_mask] = 0.0
    # Second memo: cells whose downstream walk has been shown not to reach a
    # stream. Without this, an orphan tree of length L makes the outer
    # double-for re-walk every prefix, giving O(L^2) worst-case.
    unreachable = np.zeros((rows, cols), dtype=bool)

    elev64 = elev.astype(np.float64, copy=False)

    for r0 in range(rows):
        for c0 in range(cols):
            if not np.isnan(hand[r0, c0]):
                continue
            if unreachable[r0, c0]:
                continue
            if np.isnan(elev64[r0, c0]):
                continue
            # Walk downstream until we hit a stream cell, a cell with known HAND,
            # or a cell already classified unreachable.
            path: list[tuple[int, int]] = []
            r, c = r0, c0
            reached = False
            while True:
                if not np.isnan(hand[r, c]):
                    reached = True
                    break
                if unreachable[r, c]:
                    break
                if np.isnan(elev64[r, c]):
                    break
                path.append((r, c))
                d = int(fdir[r, c])
                if d < 0 or d > 7:
                    break
                nr = r + int(_DIR_DR[d])
                nc = c + int(_DIR_DC[d])
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                r, c = nr, nc
            if not reached:
                # Memoise the entire failed path so we don't re-walk it.
                for pr, pc in path:
                    unreachable[pr, pc] = True
                continue
            # (r, c) is the drain (or a downstream cell with known HAND).
            # drain_elev = elev[r, c] - hand[r, c] — telescoping back to the
            # original stream cell.
            drain_elev = float(elev64[r, c]) - float(hand[r, c])
            for pr, pc in path:
                hand[pr, pc] = float(elev64[pr, pc]) - drain_elev
    return hand
