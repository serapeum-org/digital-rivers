"""Numba-accelerated inner kernels for digital-rivers (P7).

Numba JIT-compiled implementations of the hottest loops in the hydro pre-processor:

* :func:`d8_flow_direction_numba` — steepest-descent D8 over the elevation grid.
* :func:`kahn_accumulate_d8_numba` — Kahn topological-sort flow accumulation for
  single-direction routings (D8 / Rho8).
* :func:`priority_flood_numba` — depression-fill via Barnes 2014 Priority-Flood
  with a hand-rolled binary heap (no `heapq.typed.List` dependency).

All kernels share a single direction-offset convention (`DIR_OFFSETS` from
`dem.py`: `0=S, 1=SW, 2=W, 3=NW, 4=N, 5=NE, 6=E, 7=SE`) passed as two
`int32[:]` arrays — never as a dict — to keep Numba's type inference happy.

Disabling Numba
---------------

Set the env var `DIGITALRIVERS_DISABLE_NUMBA=1` (or fail to install Numba) and
this module's decorators degrade to no-ops, producing identical bit-for-bit
output from the pure-Python branch. Used for debugging (step-through in an IDE)
and for CI on platforms without Numba wheels.
"""
from __future__ import annotations

import os

import numpy as np

_USE_NUMBA = os.environ.get("DIGITALRIVERS_DISABLE_NUMBA", "0") != "1"

if _USE_NUMBA:
    try:
        from numba import njit, prange  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — environment without numba
        _USE_NUMBA = False

if not _USE_NUMBA:
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """No-op decorator used when Numba is unavailable / disabled."""
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    def prange(*args, **kwargs):  # type: ignore[no-redef]
        return range(*args, **kwargs)


# DIR_OFFSETS as parallel int32 arrays — passable into @njit kernels.
_DIR_DR_I32 = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_DIR_DC_I32 = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)


def is_numba_enabled() -> bool:
    """Return True if Numba JIT is currently active for this process."""
    return _USE_NUMBA


def neighbour_offsets() -> tuple[np.ndarray, np.ndarray]:
    """Return the (dr, dc) offset arrays as a tuple of `int32` arrays.

    Convenience for callers that want to pass them into JIT kernels without
    importing the underscored module-level constants.
    """
    return _DIR_DR_I32.copy(), _DIR_DC_I32.copy()


# ----- D8 flow direction --------------------------------------------------------------------

@njit(cache=True)
def d8_flow_direction_numba(
    elev: np.ndarray,
    cell_size: float,
    nodata_out: np.int32,
    d_row: np.ndarray,
    d_col: np.ndarray,
) -> np.ndarray:
    """Steepest-descent D8 flow direction over `elev`.

    Cells whose max 8-neighbour slope is non-positive (no strictly downhill
    neighbour) are marked `nodata_out` — matches the P5 sink semantics. NaN
    cells in the input also receive `nodata_out`.

    Args:
        elev: `(rows, cols)` float32 elevation array; NaN = no-data.
        cell_size: square cell side length in map units.
        nodata_out: int32 sentinel for sinks / no-data.
        d_row: `int32[8]` row offset per direction (DIR_OFFSETS order).
        d_col: `int32[8]` column offset per direction.

    Returns:
        `(rows, cols)` int32 direction-code raster.
    """
    rows, cols = elev.shape
    out = np.full((rows, cols), nodata_out, dtype=np.int32)
    diag = cell_size * np.sqrt(2.0)
    # Per-direction distance (cardinal vs diagonal).
    dist = np.empty(8, dtype=np.float64)
    for k in range(8):
        if d_row[k] != 0 and d_col[k] != 0:
            dist[k] = diag
        else:
            dist[k] = cell_size

    for r in range(rows):
        for c in range(cols):
            z = elev[r, c]
            if np.isnan(z):
                continue
            best_slope = 0.0
            best_dir = -1
            for k in range(8):
                nr = r + d_row[k]
                nc = c + d_col[k]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                zn = elev[nr, nc]
                if np.isnan(zn):
                    continue
                slope = (z - zn) / dist[k]
                if slope > best_slope:
                    best_slope = slope
                    best_dir = k
            if best_dir >= 0:
                out[r, c] = best_dir
    return out


# ----- D8 / Rho8 accumulation (Kahn topo sort) ----------------------------------------------

@njit(cache=True)
def kahn_accumulate_d8_numba(
    fdir: np.ndarray,
    weights: np.ndarray,
    d_row: np.ndarray,
    d_col: np.ndarray,
) -> np.ndarray:
    """Kahn topological-sort accumulation for single-direction routing.

    Same semantics as the pure-Python `_accumulation.kahn_accumulate` with
    `K=1`: `out[cell] = sum of weights over strictly-upstream cells` (own
    weight is never counted at self).

    Args:
        fdir: `(rows, cols)` int32 direction-code raster. Values outside
            `[0, 7]` are sinks (no outgoing flow); they still accumulate
            inbound contributions.
        weights: `(rows, cols)` float64 per-cell weight.
        d_row: `int32[8]` row offsets (DIR_OFFSETS order).
        d_col: `int32[8]` column offsets.

    Returns:
        `(rows, cols)` float64 accumulation grid.
    """
    rows, cols = fdir.shape
    indeg = np.zeros((rows, cols), dtype=np.int32)

    # In-degree pass.
    for r in range(rows):
        for c in range(cols):
            d = fdir[r, c]
            if d < 0 or d > 7:
                continue
            nr = r + d_row[d]
            nc = c + d_col[d]
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            indeg[nr, nc] += 1

    out = np.zeros((rows, cols), dtype=np.float64)
    # Pre-allocated FIFO queue as two parallel arrays (rows, cols).
    total = rows * cols
    qr = np.empty(total, dtype=np.int32)
    qc = np.empty(total, dtype=np.int32)
    head = 0
    tail = 0
    for r in range(rows):
        for c in range(cols):
            if indeg[r, c] == 0:
                qr[tail] = r
                qc[tail] = c
                tail += 1

    while head < tail:
        r = qr[head]
        c = qc[head]
        head += 1
        contribution = weights[r, c] + out[r, c]
        d = fdir[r, c]
        if d < 0 or d > 7:
            continue
        nr = r + d_row[d]
        nc = c + d_col[d]
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue
        out[nr, nc] += contribution
        indeg[nr, nc] -= 1
        if indeg[nr, nc] == 0:
            qr[tail] = nr
            qc[tail] = nc
            tail += 1
    return out


# ----- Priority-flood fill (binary-heap) ----------------------------------------------------

@njit(cache=True, inline="always")
def _heap_push(prio: np.ndarray, idx: np.ndarray, size: int,
               new_prio: float, new_idx: int) -> int:
    """Push `(new_prio, new_idx)` onto a min-heap stored in
    `prio[:size]` / `idx[:size]`. Returns the new size. Tie-breaks via
    `idx` so the order is deterministic across runs."""
    i = size
    prio[i] = new_prio
    idx[i] = new_idx
    size += 1
    while i > 0:
        parent = (i - 1) >> 1
        if prio[parent] > prio[i] or (
            prio[parent] == prio[i] and idx[parent] > idx[i]
        ):
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
def cotat_upscale_numba(
    fdir: np.ndarray,
    acc: np.ndarray,
    scale_factor: int,
    d_row: np.ndarray,
    d_col: np.ndarray,
    nodata_out: np.int32,
) -> np.ndarray:
    """Native Numba COTAT upscaling kernel (P28 / Reed 2003).

    For each coarse cell (a `scale_factor` × `scale_factor` block of fine
    cells), finds the fine cell with the largest accumulation as the
    coarse-cell outlet, traces downstream along `fdir` until leaving the
    block, then assigns the coarse-cell direction by comparing the source
    and destination coarse-cell coordinates.

    Args:
        fdir: `(rows, cols)` int32 fine-resolution D8 directions.
        acc: `(rows, cols)` float64 fine-resolution accumulation.
        scale_factor: integer coarsening factor (>= 2).
        d_row, d_col: int32 DIR_OFFSETS neighbour offsets.
        nodata_out: int32 sentinel for coarse cells with no defined outlet.

    Returns:
        `(rows // scale_factor, cols // scale_factor)` int32 coarse-grid
        D8 raster.
    """
    rows, cols = fdir.shape
    out_rows = rows // scale_factor
    out_cols = cols // scale_factor
    coarse_fdir = np.full((out_rows, out_cols), nodata_out, dtype=np.int32)
    for br in range(out_rows):
        for bc in range(out_cols):
            r_lo = br * scale_factor
            r_hi = r_lo + scale_factor
            c_lo = bc * scale_factor
            c_hi = c_lo + scale_factor
            # Block argmax — explicit loop because numba doesn't support
            # np.argmax over a sliced 2-D array in all versions.
            best_v = -np.inf
            fr = r_lo
            fc = c_lo
            for r in range(r_lo, r_hi):
                for c in range(c_lo, c_hi):
                    v = acc[r, c]
                    if v > best_v:
                        best_v = v
                        fr = r
                        fc = c
            r = fr
            c = fc
            while True:
                d = fdir[r, c]
                if d < 0 or d > 7:
                    break
                nr = r + d_row[d]
                nc = c + d_col[d]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    break
                coarse_dr = (nr // scale_factor) - br
                coarse_dc = (nc // scale_factor) - bc
                if coarse_dr != 0 or coarse_dc != 0:
                    for k in range(8):
                        if d_row[k] == coarse_dr and d_col[k] == coarse_dc:
                            coarse_fdir[br, bc] = k
                            break
                    break
                r = nr
                c = nc
    return coarse_fdir


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
            is_seed = (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
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
