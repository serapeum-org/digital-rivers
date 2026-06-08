"""Exit-distance ramp epsilon-fill — the shared ``eps_fill="exact"`` / ``"monotone"`` definition (B6).

    fill_ramp = fill_0 + epsilon * g

where ``g`` is the graph distance, over the ``epsilon = 0`` filled surface (B3) stepping to lower-or-equal
neighbours, from each cell to the nearest **real-terrain exit** (a strictly-lower cell that was *not* raised by
the fill) or domain edge / no-data. Real (already-draining) terrain has ``g = 0`` and is left unchanged.

This is the **single epsilon>0 definition used by both engines**: the in-memory path
(``DEM.fill_depressions(eps_fill="exact")``) computes :func:`monotone_fill_reference` directly, and the tiled
path reproduces it tile-by-tile — so ``engine="in_memory"`` and ``engine="tiled"`` agree **bit-for-bit**, which
is what makes ``eps_fill="exact"`` exact (see ``docs/eps-fill-exact-feasibility.md`` and issue #69).

**What this guarantees (tested):**

* ``epsilon -> 0`` reduces to the exact ``fill_0``.
* The tiled result is **bit-for-bit identical to its whole-array reference** (graph distance is unique and
  seam-reconcilable), so tiling introduces no error and matches the in-memory engine exactly.

**What this does NOT guarantee — read before using:**

``g`` is a tile-reconstructible **min-distance**, not the in-memory Barnes step-count. Because it is deterministic
and order-independent it cannot, for *large* epsilon, replicate the classic kernel's universal flat-removal:
``epsilon * g`` can over-inflate a wide flat above adjacent lower terrain and leave a residual flat **when epsilon
is not small relative to the terrain's vertical steps**. This is a valid, flat-free fill for **small** epsilon
(``epsilon`` ≪ the smallest real elevation difference you care about — the normal regime, e.g. ``1e-3`` on
metre-scale DEMs). For guaranteed flat removal at *any* epsilon use ``eps_fill="barnes"`` (the classic
Priority-Flood step-count), which is in-memory only — it depends on the global traversal order and is provably
not tileable (``docs/eps-fill-exact-feasibility.md``).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from pyramids.dataset import Dataset

from digitalrivers._outofcore.fill import (
    _nodata_mask,
    fill_depressions_tiled,
    out_dtype,
)
from digitalrivers._outofcore.tiling import plan_tiles, read_tile, write_core

_BIG = 1 << 30  # "unset" / +inf sentinel for the integer distance field
_DIRS = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))


def _shift(arr: np.ndarray, dr: int, dc: int):
    """Return ``(neighbour_values, valid_mask)`` for the (dr, dc) shift; invalid outside the array."""
    rows, cols = arr.shape
    out = np.zeros_like(arr)
    valid = np.zeros(arr.shape, dtype=bool)
    r0s, r1s = max(0, -dr), rows - max(0, dr)
    c0s, c1s = max(0, -dc), cols - max(0, dc)
    out[r0s:r1s, c0s:c1s] = arr[r0s + dr : r1s + dr, c0s + dc : c1s + dc]
    valid[r0s:r1s, c0s:c1s] = True
    return out, valid


def _source_mask(orig, fill0, nodata, glob_r0, glob_c0, rows, cols) -> np.ndarray:
    """Cells with ``g = 0``: a strictly-lower **real-terrain** neighbour, or domain edge / no-data-adjacent."""
    nod = _nodata_mask(fill0, nodata)
    real = (fill0 == orig) & ~nod  # cell that was NOT raised by the epsilon=0 fill
    lower_real = np.zeros(fill0.shape, dtype=bool)
    nodata_adj = np.zeros(fill0.shape, dtype=bool)
    for dr, dc in _DIRS:
        nbr_f, valid = _shift(fill0, dr, dc)
        nbr_real, _ = _shift(real.astype(np.uint8), dr, dc)
        nbr_nod, _ = _shift(nod.astype(np.uint8), dr, dc)
        nbr_real = nbr_real.astype(bool) & valid
        nbr_nod = nbr_nod.astype(bool) & valid
        lower_real |= valid & nbr_real & (nbr_f < fill0)
        nodata_adj |= nbr_nod
    rr = glob_r0 + np.arange(fill0.shape[0])[:, None]
    cc = glob_c0 + np.arange(fill0.shape[1])[None, :]
    edge = (rr == 0) | (rr == rows - 1) | (cc == 0) | (cc == cols - 1)
    return (lower_real | nodata_adj | edge) & ~nod


def _relax(fill0, g, source, nodata) -> np.ndarray:
    """Relax ``g`` to local convergence: ``g[c] = min(g[c], 1 + g[n])`` over neighbours ``n`` with
    ``fill0[n] <= fill0[c]`` (downhill-or-flat); ``g = 0`` at sources."""
    nod = _nodata_mask(fill0, nodata)
    g = g.astype(np.int64, copy=True)
    g[source] = 0
    while True:
        best = g.copy()
        for dr, dc in _DIRS:
            nbr_g, valid = _shift(g, dr, dc)
            nbr_f, _ = _shift(fill0, dr, dc)
            nbr_nod, _ = _shift(nod.astype(np.uint8), dr, dc)
            step = valid & (nbr_f <= fill0) & ~nbr_nod.astype(bool)
            best = np.minimum(best, np.where(step, nbr_g + 1, _BIG))
        best[source] = 0
        best[nod] = g[nod]
        if np.array_equal(best, g):
            return best
        g = best


def monotone_fill_reference(orig, fill0, epsilon, nodata) -> np.ndarray:
    """Whole-array reference for the monotone fill (the tiled path must reproduce this)."""
    orig = np.asarray(orig, dtype=np.float64)
    fill0 = np.asarray(fill0, dtype=np.float64)
    rows, cols = fill0.shape
    src = _source_mask(orig, fill0, nodata, 0, 0, rows, cols)
    g = _relax(fill0, np.full(fill0.shape, _BIG, dtype=np.int64), src, nodata)
    g = np.where(g >= _BIG, 0, g)
    out = fill0 + epsilon * g
    nod = _nodata_mask(fill0, nodata)
    out[nod] = fill0[nod]
    return out


def fill_depressions_monotone_tiled(
    dem,
    out_path: str,
    *,
    epsilon: float,
    tile_rows: int = 2048,
    tile_cols: int = 2048,
    cache: str = "evict",
):
    """Tiled monotone terracing = tiled ``fill_0`` (B3) + ``epsilon`` * tiled exit-distance ``g``.

    Flat-free for small ``epsilon`` only; see the module docstring for the (important) caveats.
    """
    rows, cols = dem.rows, dem.columns
    nodata = dem.no_data_value[0] if dem.no_data_value else None
    dtype = out_dtype(dem)
    specs = plan_tiles(rows, cols, tile_rows, tile_cols, halo=1)

    out = Dataset.create_empty(
        rows,
        cols,
        dtype=dtype,
        geo=dem.geotransform,
        epsg=dem.epsg,
        no_data_value=-9999.0 if nodata is None else nodata,
        driver_type="GTiff",
        path=out_path,
    )
    scratch = tempfile.mkdtemp(prefix="dr_monotone_")
    fill0_ds = g_ds = None
    try:
        # fill_0 scratch is float64 (not the source dtype) so that ``fill_0 + epsilon * g`` matches the in-memory
        # engine bit-for-bit on float32 sources — the final ``out`` below is still cast to the source dtype.
        fill0_ds = fill_depressions_tiled(
            dem,
            os.path.join(scratch, "fill0.tif"),
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            epsilon=0.0,
            cache=cache,
            dtype="float64",
        )
        g_ds = Dataset.create_empty(
            rows,
            cols,
            dtype="int32",
            geo=dem.geotransform,
            epsg=dem.epsg,
            no_data_value=-1,
            driver_type="GTiff",
            path=os.path.join(scratch, "g.tif"),
        )
        for s in specs:
            write_core(g_ds, s, np.full((s.n_rows, s.n_cols), _BIG, dtype=np.int32))

        # seam-reconciled multi-source BFS: Gauss-Seidel sweeps until the field stops changing. The exit-distance
        # can span many tiles, so bound the sweeps by total perimeter (not tile count) and RAISE on
        # non-convergence rather than silently writing a partial distance field (cf. accumulate.py).
        perimeter_cells = sum(2 * (s.n_rows + s.n_cols) for s in specs)
        converged = False
        for _ in range(max(64, perimeter_cells + 2)):
            changed = False
            for s in specs:
                o_halo, core = read_tile(dem, s, rows, cols)
                f_halo, _ = read_tile(fill0_ds, s, rows, cols)
                g_halo, _ = read_tile(g_ds, s, rows, cols)
                r0 = s.row_off - core[0].start
                c0 = s.col_off - core[1].start
                orig = np.asarray(o_halo, dtype=np.float64)
                fill0 = np.asarray(f_halo, dtype=np.float64)
                src = _source_mask(orig, fill0, nodata, r0, c0, rows, cols)
                new_g = _relax(fill0, np.asarray(g_halo), src, nodata)
                core_new = new_g[core].astype(np.int32)
                if not np.array_equal(core_new, np.asarray(g_halo)[core]):
                    changed = True
                write_core(g_ds, s, core_new)
            if not changed:
                converged = True
                break
        if not converged:
            raise RuntimeError(
                "tiled monotone fill: the flat-distance BFS did not converge — please report with the input"
            )

        for s in specs:  # combine: fill_0 + epsilon * g
            f_halo, core = read_tile(fill0_ds, s, rows, cols)
            g_halo, _ = read_tile(g_ds, s, rows, cols)
            f_core = np.asarray(f_halo, dtype=np.float64)[core]
            g_core = np.asarray(g_halo)[core]
            g_core = np.where(g_core >= _BIG, 0, g_core)
            res = f_core + epsilon * g_core
            nod = _nodata_mask(f_core, nodata)
            out_nodata = -9999.0 if nodata is None else nodata
            write_core(out, s, np.where(nod, out_nodata, res).astype(dtype))
        return out
    finally:
        if fill0_ds is not None:
            fill0_ds.close()
        if g_ds is not None:
            g_ds.close()
