"""Numba flow kernels: D8 accumulation and COTAT upscaling.

* :func:`kahn_accumulate_d8_numba` — Kahn topological-sort accumulation for the
  single-direction routings (D8 / Rho8). Every cell is visited once, in an order where
  its upstream neighbours are already resolved, so no recursion and no revisits.
* :func:`cotat_upscale_numba` — Reed (2003) Cell Outlet Tracing with an Area Threshold,
  the fast path behind `FlowDirection.upscale(method="cotat")`. Bit-for-bit identical to
  the pure-Python loop it replaces.

Both take the neighbour offsets as `int32[:]` arrays from
`digitalrivers.core.directions`, never as a dict, which is what keeps Numba's type
inference happy.

Imported lazily by :mod:`digitalrivers.flow._kernels.accumulation`,
:mod:`digitalrivers.flow.upscale` and the out-of-core accumulators — this module pulls
in `numba` at import, and `import digitalrivers` must not.
"""

from __future__ import annotations

import numpy as np

from digitalrivers.core.numba import njit

__all__ = ["kahn_accumulate_d8_numba", "cotat_upscale_numba"]


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
