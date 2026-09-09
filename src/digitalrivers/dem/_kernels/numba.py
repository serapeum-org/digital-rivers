"""Numba pit-removal kernels: Priority-Flood fill, with and without spill labels.

Barnes, Lehman & Mulla (2014) Priority-Flood over a hand-rolled binary heap — no
`heapq` / `numba.typed.List` dependency, so the kernel compiles under `nopython` with
no reflected containers.

Two variants, differing only in what they carry along:

* :func:`priority_flood_numba` — the filled surface.
* :func:`priority_flood_labels_numba` — the filled surface plus, per cell, the id of the
  boundary cell its water spills to. The tiled fill needs those labels to stitch tiles
  into one global spill graph.

Both take the neighbour offsets as `int32[:]` arrays rather than a dict, which is what
keeps Numba's type inference happy. They come from `digitalrivers.core.directions`.

Imported lazily by :mod:`digitalrivers.dem._kernels.pitremoval` — this module pulls in
`numba` at import, and `import digitalrivers` must not.
"""

from __future__ import annotations

import numpy as np

from digitalrivers.core.numba import njit

__all__ = ["priority_flood_numba", "priority_flood_labels_numba"]


@njit(cache=True, inline="always")
def _heap_push(
    prio: np.ndarray, idx: np.ndarray, size: int, new_prio: float, new_idx: int
) -> int:
    """Push `(new_prio, new_idx)` onto a min-heap stored in
    `prio[:size]` / `idx[:size]`. Returns the new size. Tie-breaks via
    `idx` so the order is deterministic across runs."""
    i = size
    prio[i] = new_prio
    idx[i] = new_idx
    size += 1
    while i > 0:
        parent = (i - 1) >> 1
        if prio[parent] > prio[i] or (prio[parent] == prio[i] and idx[parent] > idx[i]):
            tmp_p = prio[parent]
            prio[parent] = prio[i]
            prio[i] = tmp_p
            tmp_i = idx[parent]
            idx[parent] = idx[i]
            idx[i] = tmp_i
            i = parent
        else:
            break
    return size


@njit(cache=True, inline="always")
def _heap_pop(prio: np.ndarray, idx: np.ndarray, size: int):
    """Pop the smallest (prio, idx). Returns (top_prio, top_idx, new_size)."""
    top_p = prio[0]
    top_i = idx[0]
    size -= 1
    prio[0] = prio[size]
    idx[0] = idx[size]
    i = 0
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < size and (
            prio[left] < prio[smallest]
            or (prio[left] == prio[smallest] and idx[left] < idx[smallest])
        ):
            smallest = left
        if right < size and (
            prio[right] < prio[smallest]
            or (prio[right] == prio[smallest] and idx[right] < idx[smallest])
        ):
            smallest = right
        if smallest == i:
            break
        tmp_p = prio[i]
        prio[i] = prio[smallest]
        prio[smallest] = tmp_p
        tmp_i = idx[i]
        idx[i] = idx[smallest]
        idx[smallest] = tmp_i
        i = smallest
    return top_p, top_i, size


@njit(cache=True)
def priority_flood_numba(
    elev: np.ndarray,
    nodata_mask: np.ndarray,
    epsilon: float,
    d_row: np.ndarray,
    d_col: np.ndarray,
) -> np.ndarray:
    """Barnes 2014 Priority-Flood depression fill, JIT-compiled.

    Output semantics match the pure-Python `_pitremoval._priority_flood`:
    cells along a depression path lift to the spill height (plus `epsilon`
    per step if `epsilon > 0`); cells already strictly higher keep their
    original elevation.

    Args:
        elev: `(rows, cols)` float64 elevation.
        nodata_mask: `(rows, cols)` bool — True at no-data cells.
        epsilon: per-step elevation lift inside depressions.
        d_row: `int32[8]` row offsets.
        d_col: `int32[8]` column offsets.

    Returns:
        `(rows, cols)` float64 filled elevation. No-data positions are NaN.
    """
    rows, cols = elev.shape
    out = elev.copy()
    closed = nodata_mask.copy()
    # Mark NaN positions explicitly.
    for r in range(rows):
        for c in range(cols):
            if nodata_mask[r, c]:
                out[r, c] = np.nan
                closed[r, c] = True

    capacity = rows * cols + 1
    heap_prio = np.empty(capacity, dtype=np.float64)
    heap_idx = np.empty(capacity, dtype=np.int64)
    heap_size = 0

    # FIFO pit queue (Barnes two-queue trick) — pre-allocated.
    pit_idx = np.empty(capacity, dtype=np.int64)
    pit_prio = np.empty(capacity, dtype=np.float64)
    pit_head = 0
    pit_tail = 0

    # Seed: array-boundary cells + cells adjacent to no-data. The hand-rolled
    # heap's tie-break is via `idx` — we store the row-major linear index
    # there, which is unique per cell and makes the pop order deterministic.
    for r in range(rows):
        for c in range(cols):
            if closed[r, c]:
                continue
            is_seed = r == 0 or r == rows - 1 or c == 0 or c == cols - 1
            if not is_seed:
                for k in range(8):
                    nr = r + d_row[k]
                    nc = c + d_col[k]
                    if 0 <= nr < rows and 0 <= nc < cols and nodata_mask[nr, nc]:
                        is_seed = True
                        break
            if is_seed:
                linear = r * cols + c
                heap_size = _heap_push(
                    heap_prio, heap_idx, heap_size, out[r, c], linear
                )
                closed[r, c] = True

    while heap_size > 0 or pit_head < pit_tail:
        if pit_head < pit_tail:
            e = pit_prio[pit_head]
            linear = pit_idx[pit_head]
            pit_head += 1
        else:
            e, linear, heap_size = _heap_pop(heap_prio, heap_idx, heap_size)

        r = linear // cols
        c = linear % cols

        for k in range(8):
            nr = r + d_row[k]
            nc = c + d_col[k]
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if closed[nr, nc]:
                continue
            closed[nr, nc] = True
            n_linear = nr * cols + nc
            if out[nr, nc] <= e:
                lifted = e + epsilon if epsilon > 0.0 else e
                out[nr, nc] = lifted
                pit_prio[pit_tail] = lifted
                pit_idx[pit_tail] = n_linear
                pit_tail += 1
            else:
                heap_size = _heap_push(
                    heap_prio, heap_idx, heap_size, out[nr, nc], n_linear
                )
    return out


@njit(cache=True)
def priority_flood_labels_numba(
    elev: np.ndarray,
    nodata_mask: np.ndarray,
    d_row: np.ndarray,
    d_col: np.ndarray,
):
    """Watershed-labelled Priority-Flood (Barnes 2016, Algorithm 1).

    Runs the same `epsilon = 0` Priority-Flood as :func:`priority_flood_numba` — so the returned ``filled`` is
    **bit-for-bit identical** to ``priority_flood_numba(elev, nodata_mask, 0.0, ...)`` — while additionally
    painting every data cell with a **watershed label**. A label identifies the set of cells that drain to one
    common outlet on the seed set (the domain edge / no-data boundary). The labels are the per-tile output the
    Barnes 2016 master-graph fill (B3) stitches across tile seams.

    Labelling rule: seed cells (domain edge + cells adjacent to no-data) are pushed unlabelled; when a cell is
    popped still unlabelled it mints a fresh label, and every cell it floods into inherits that label. No-data
    cells get label 0; every data cell ends with a label ``>= 1``.

    Args:
        elev: `(rows, cols)` float64 elevation.
        nodata_mask: `(rows, cols)` bool — True at no-data cells.
        d_row: `int32[8]` row offsets (`DIR_OFFSETS` convention).
        d_col: `int32[8]` column offsets.

    Returns:
        Tuple ``(filled, labels)``: ``filled`` is `(rows, cols)` float64 (no-data positions are NaN, identical to
        ``priority_flood_numba`` with ``epsilon = 0``); ``labels`` is `(rows, cols)` int32 (0 at no-data, ``>= 1``
        elsewhere).
    """
    rows, cols = elev.shape
    out = elev.copy()
    closed = nodata_mask.copy()
    labels = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            if nodata_mask[r, c]:
                out[r, c] = np.nan
                closed[r, c] = True

    capacity = rows * cols + 1
    heap_prio = np.empty(capacity, dtype=np.float64)
    heap_idx = np.empty(capacity, dtype=np.int64)
    heap_size = 0

    pit_idx = np.empty(capacity, dtype=np.int64)
    pit_prio = np.empty(capacity, dtype=np.float64)
    pit_head = 0
    pit_tail = 0

    for r in range(rows):
        for c in range(cols):
            if closed[r, c]:
                continue
            is_seed = r == 0 or r == rows - 1 or c == 0 or c == cols - 1
            if not is_seed:
                for k in range(8):
                    nr = r + d_row[k]
                    nc = c + d_col[k]
                    if 0 <= nr < rows and 0 <= nc < cols and nodata_mask[nr, nc]:
                        is_seed = True
                        break
            if is_seed:
                linear = r * cols + c
                heap_size = _heap_push(
                    heap_prio, heap_idx, heap_size, out[r, c], linear
                )
                closed[r, c] = True

    next_label = 1
    while heap_size > 0 or pit_head < pit_tail:
        if pit_head < pit_tail:
            e = pit_prio[pit_head]
            linear = pit_idx[pit_head]
            pit_head += 1
        else:
            e, linear, heap_size = _heap_pop(heap_prio, heap_idx, heap_size)

        r = linear // cols
        c = linear % cols
        if labels[r, c] == 0:
            labels[r, c] = next_label
            next_label += 1
        cur = labels[r, c]

        for k in range(8):
            nr = r + d_row[k]
            nc = c + d_col[k]
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if closed[nr, nc]:
                continue
            closed[nr, nc] = True
            labels[nr, nc] = cur
            n_linear = nr * cols + nc
            if out[nr, nc] <= e:
                out[nr, nc] = e
                pit_prio[pit_tail] = e
                pit_idx[pit_tail] = n_linear
                pit_tail += 1
            else:
                heap_size = _heap_push(
                    heap_prio, heap_idx, heap_size, out[nr, nc], n_linear
                )
    return out, labels
