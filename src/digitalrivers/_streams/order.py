"""Stream-network ordering schemes (P10): Strahler, Shreve, Horton.

All three operate on a stream-cell mask plus a D8 flow-direction raster. Output is a
2-D `uint16` raster of orders / magnitudes; non-stream cells hold `0`.

* **Strahler (1957)** — Kahn BFS over stream cells. Heads get order 1; at each
  confluence the downstream order is `max_in + 1` iff at least two upstream
  tributaries arrive carrying the same `max_in`, else `max_in`.
* **Shreve (1966)** — additive magnitude. Heads = 1; downstream gets the sum
  of incoming magnitudes. Outlet equals the number of headwaters.
* **Horton (1945)** — Strahler with main-stem promotion. After Strahler runs,
  walk upstream from the outlet; at every confluence pick the tributary with the
  longer trace and re-stamp its entire path back to its head with the
  confluence's outgoing order. The shorter sibling keeps its local Strahler
  value. Ties broken by lower row-major linear index for determinism.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# DIR_OFFSETS-aligned offsets and inverse table (see dem.py / stream_raster.py).
_DIR_DR = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_DIR_DC = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
_INV_DIR = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int32)


def _build_topology(
    stream_mask: np.ndarray, fdir: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (indeg, downstream_idx) for every stream cell.

    Args:
        stream_mask: `(rows, cols)` bool.
        fdir: `(rows, cols)` int — D8 direction codes 0–7 (or any sentinel).

    Returns:
        indeg: `(rows, cols)` int32 of incoming-stream-neighbour counts.
        ds_idx: `(rows, cols, 2)` int32 of `(dr, dc)` to the downstream cell;
            `(-1, -1)` for cells with no valid downstream (sink, off-grid, or
            non-stream downstream).
    """
    rows, cols = stream_mask.shape
    indeg = np.zeros((rows, cols), dtype=np.int32)
    ds_idx = np.full((rows, cols, 2), -1, dtype=np.int32)

    for k in range(8):
        dr = int(_DIR_DR[k])
        dc = int(_DIR_DC[k])
        # Neighbour at offset (dr, dc) flowing INTO us has direction code inv[k].
        src_r = slice(max(0, dr), min(rows, rows + dr))
        src_c = slice(max(0, dc), min(cols, cols + dc))
        dst_r = slice(max(0, -dr), min(rows, rows - dr))
        dst_c = slice(max(0, -dc), min(cols, cols - dc))
        sm_src = stream_mask[src_r, src_c]
        fd_src = fdir[src_r, src_c]
        inflow = sm_src & (fd_src == _INV_DIR[k]) & stream_mask[dst_r, dst_c]
        indeg[dst_r, dst_c] += inflow.astype(np.int32)

    # Build downstream offsets.
    for r in range(rows):
        for c in range(cols):
            if not stream_mask[r, c]:
                continue
            d = int(fdir[r, c])
            if d < 0 or d > 7:
                continue
            nr = r + int(_DIR_DR[d])
            nc = c + int(_DIR_DC[d])
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not stream_mask[nr, nc]:
                continue
            ds_idx[r, c, 0] = nr - r
            ds_idx[r, c, 1] = nc - c
    return indeg, ds_idx


def strahler(stream_mask: np.ndarray, fdir: np.ndarray) -> np.ndarray:
    """Strahler stream order (Strahler 1957) via Kahn BFS over the stream graph.

    Args:
        stream_mask: `(rows, cols)` bool, True at stream cells.
        fdir: `(rows, cols)` int direction-code raster.

    Returns:
        `(rows, cols)` uint16 of Strahler orders. Non-stream cells hold `0`.

    Examples:
        - Two single-cell head tributaries meeting at a confluence promote the
          downstream trunk to order 2:

            >>> import numpy as np
            >>> sm = np.zeros((4, 3), dtype=bool)
            >>> sm[0, 0] = sm[0, 2] = True
            >>> sm[1, 1] = sm[2, 1] = sm[3, 1] = True
            >>> fd = np.array([
            ...     [7, -1,  1],
            ...     [-1, 0, -1],
            ...     [-1, 0, -1],
            ...     [-1, -1, -1],
            ... ], dtype=np.int32)
            >>> order = strahler(sm, fd)
            >>> int(order[0, 0])
            1
            >>> int(order[3, 1])
            2
    """
    rows, cols = stream_mask.shape
    out = np.zeros((rows, cols), dtype=np.uint16)
    indeg, ds_idx = _build_topology(stream_mask, fdir)

    # Per-cell running state: max incoming order and how many tributaries match it.
    max_in = np.zeros((rows, cols), dtype=np.uint16)
    cnt_max = np.zeros((rows, cols), dtype=np.uint16)

    queue: deque[tuple[int, int]] = deque()
    for r, c in zip(*np.where(stream_mask & (indeg == 0))):
        out[r, c] = 1
        queue.append((int(r), int(c)))

    while queue:
        r, c = queue.popleft()
        o = int(out[r, c])
        dr = int(ds_idx[r, c, 0])
        dc = int(ds_idx[r, c, 1])
        if dr == -1 and dc == -1:
            continue
        nr = r + dr
        nc = c + dc
        if o > max_in[nr, nc]:
            max_in[nr, nc] = o
            cnt_max[nr, nc] = 1
        elif o == max_in[nr, nc]:
            cnt_max[nr, nc] += 1
        indeg[nr, nc] -= 1
        if indeg[nr, nc] == 0:
            out[nr, nc] = (
                max_in[nr, nc] + 1
                if cnt_max[nr, nc] >= 2
                else max_in[nr, nc]
            )
            queue.append((nr, nc))
    return out


def shreve(stream_mask: np.ndarray, fdir: np.ndarray) -> np.ndarray:
    """Shreve magnitude (Shreve 1966): heads = 1, downstream = sum of incoming.

    Args:
        stream_mask: `(rows, cols)` bool.
        fdir: `(rows, cols)` int direction-code raster.

    Returns:
        `(rows, cols)` uint32 of Shreve magnitudes. Non-stream cells hold `0`.
        `uint32` rather than `uint16` because magnitudes are total head
        counts and easily exceed 65 535 on continental basins.

    Examples:
        - A 2-head Y-junction produces magnitude 1 at each head and 2 at the outlet:

            >>> import numpy as np
            >>> sm = np.zeros((4, 3), dtype=bool)
            >>> sm[0, 0] = sm[0, 2] = True
            >>> sm[1, 1] = sm[2, 1] = sm[3, 1] = True
            >>> fd = np.array([
            ...     [7, -1,  1],
            ...     [-1, 0, -1],
            ...     [-1, 0, -1],
            ...     [-1, -1, -1],
            ... ], dtype=np.int32)
            >>> mag = shreve(sm, fd)
            >>> int(mag[0, 0]), int(mag[0, 2])
            (1, 1)
            >>> int(mag[3, 1])
            2
    """
    rows, cols = stream_mask.shape
    out = np.zeros((rows, cols), dtype=np.uint32)
    indeg, ds_idx = _build_topology(stream_mask, fdir)

    queue: deque[tuple[int, int]] = deque()
    for r, c in zip(*np.where(stream_mask & (indeg == 0))):
        out[r, c] = 1
        queue.append((int(r), int(c)))

    while queue:
        r, c = queue.popleft()
        m = int(out[r, c])
        dr = int(ds_idx[r, c, 0])
        dc = int(ds_idx[r, c, 1])
        if dr == -1 and dc == -1:
            continue
        nr = r + dr
        nc = c + dc
        out[nr, nc] += m
        indeg[nr, nc] -= 1
        if indeg[nr, nc] == 0:
            queue.append((nr, nc))
    return out


def horton(stream_mask: np.ndarray, fdir: np.ndarray) -> np.ndarray:
    """Horton order (Horton 1945) — Strahler with main-stem promotion.

    Runs Strahler first, then for each confluence walks upstream and identifies the
    longest tributary (length measured in cell steps along the stream). The main
    tributary's entire trace back to its head is re-stamped with the confluence's
    outgoing order; the shorter sibling keeps its local Strahler value.

    Args:
        stream_mask: `(rows, cols)` bool.
        fdir: `(rows, cols)` int direction-code raster.

    Returns:
        `(rows, cols)` uint16 of Horton orders.
    """
    rows, cols = stream_mask.shape
    out = strahler(stream_mask, fdir).copy()
    indeg, _ds_idx = _build_topology(stream_mask, fdir)

    # Length-from-head (in cell steps) for every stream cell via topological sweep
    # in the same order as Strahler. Used as the main-stem tiebreaker.
    length_from_head = np.zeros((rows, cols), dtype=np.int32)
    indeg_copy = indeg.copy()
    queue: deque[tuple[int, int]] = deque()
    for r, c in zip(*np.where(stream_mask & (indeg_copy == 0))):
        queue.append((int(r), int(c)))

    while queue:
        r, c = queue.popleft()
        d = int(fdir[r, c])
        if d < 0 or d > 7:
            continue
        nr = r + int(_DIR_DR[d])
        nc = c + int(_DIR_DC[d])
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if not stream_mask[nr, nc]:
            continue
        # Main-stem length = max over incoming tributaries + 1.
        cand = length_from_head[r, c] + 1
        if cand > length_from_head[nr, nc]:
            length_from_head[nr, nc] = cand
        indeg_copy[nr, nc] -= 1
        if indeg_copy[nr, nc] == 0:
            queue.append((nr, nc))

    # For Horton main-stem promotion, walk upstream from every outlet. At each
    # confluence pick the tributary with the longest length_from_head trace; restamp
    # its whole path back to its head with the confluence's outgoing order.
    # Outlets = stream cells with no downstream stream neighbour.
    outlets: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if not stream_mask[r, c]:
                continue
            d = int(fdir[r, c])
            if d < 0 or d > 7:
                outlets.append((r, c))
                continue
            nr = r + int(_DIR_DR[d])
            nc = c + int(_DIR_DC[d])
            if not (0 <= nr < rows and 0 <= nc < cols) or not stream_mask[nr, nc]:
                outlets.append((r, c))

    # Reverse-walk via inverse-direction inflows. Process each outlet recursively
    # (iterative stack) and restamp.
    for out_r, out_c in outlets:
        stack: list[tuple[int, int, int]] = [(out_r, out_c, int(out[out_r, out_c]))]
        while stack:
            r, c, stem_order = stack.pop()
            # Re-stamp this cell with the stem order.
            if out[r, c] < stem_order:
                out[r, c] = stem_order
            # Find upstream inflowing tributaries. The cell at (r + dr, c + dc)
            # with direction code inv[k] flows into (r, c) — this mirrors the
            # topology builder's adjacency arithmetic.
            inflow_cells: list[tuple[int, int, int, int]] = []
            for k in range(8):
                dr = int(_DIR_DR[k])
                dc = int(_DIR_DC[k])
                ur = r + dr
                uc = c + dc
                if not (0 <= ur < rows and 0 <= uc < cols):
                    continue
                if not stream_mask[ur, uc]:
                    continue
                if int(fdir[ur, uc]) != int(_INV_DIR[k]):
                    continue
                inflow_cells.append((int(length_from_head[ur, uc]),
                                     ur * cols + uc, ur, uc))
            if not inflow_cells:
                continue
            # Main stem = longest length; tie-break by lower linear index.
            inflow_cells.sort(reverse=True)
            main_len, _main_lin, main_r, main_c = inflow_cells[0]
            stack.append((main_r, main_c, stem_order))
            # Siblings keep their existing (Strahler) order — walk them with their
            # own current order so the recursion preserves their main-stem labelling
            # within their subtree.
            for _l, _lin, sr, sc in inflow_cells[1:]:
                stack.append((sr, sc, int(out[sr, sc])))
    return out
