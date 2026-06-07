"""Tiled, out-of-core flow accumulation — D8 / Rho8 only (B4).

Single-receiver (non-divergent) flow means each cell sends its whole accumulated flow to exactly one neighbour, so
flow only crosses tile seams at boundary cells. We exploit the linearity of accumulation:

    acc_tile = kahn(fdir, weights + inflow) + inflow

where ``inflow`` is the external flow arriving at the tile's inlet cells. A tile's **export** at a boundary cell
``e`` whose receiver lies in a neighbour tile is ``acc[e] + weight[e]``, delivered to that receiver (an inlet).
Inlet inflows therefore depend on neighbour exports, which depend on inlet inflows — a fixed point that converges
because global D8 flow over a filled DEM is acyclic. We iterate the perimeter exchange to convergence (a simpler,
still-exact alternative to the Barnes 2017 single-pass FOLLOWPATH graph; same result).

D∞ / MFD are out of scope here (divergent flow has no fixed-halo closure); they raise.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from pyramids.dataset import Dataset

from digitalrivers._outofcore.tiling import plan_tiles, read_tile, write_core


def _read_core(dataset, spec, rows, cols, dtype):
    arr, core = read_tile(dataset, spec, rows, cols)
    return np.ascontiguousarray(arr[core], dtype=dtype)


def _tile_grid_index(r: int, c: int, tile_rows: int, tile_cols: int) -> tuple[int, int]:
    return r // tile_rows, c // tile_cols


def flow_accumulation_tiled(
    fdir,
    out_path: str,
    *,
    weights=None,
    tile_rows: int = 2048,
    tile_cols: int = 2048,
    cache: str = "evict",
    workers: int = 1,
    scratch_dir: str | None = None,
    scheduler: str = "threads",
    client=None,
):
    """Out-of-core D8/Rho8 flow accumulation by tiled perimeter-exchange to convergence.

    Args:
        fdir: A ``FlowDirection`` (or ``Dataset``) of D8 direction codes (``DIR_OFFSETS`` order; sinks / no-data
            are values outside ``[0, 7]``). Must be D8 or Rho8 routing.
        out_path: Path of the GeoTIFF to create and stream the accumulation into.
        weights: Optional per-cell weight ``Dataset``; defaults to 1.0 per cell (cell counts).
        tile_rows: Core tile height. Defaults to 2048.
        tile_cols: Core tile width. Defaults to 2048.
        cache: Reserved for parity with the fill API; the perimeter exchange re-reads tiles per round.
        workers: Reserved for the dask backend (B7).
        scratch_dir: Reserved.

    Returns:
        The accumulation `pyramids` ``Dataset`` opened on ``out_path``.

    Raises:
        NotImplementedError: If ``fdir.routing`` is not D8 / Rho8 (divergent flow has no fixed-halo closure).
    """
    routing = getattr(fdir, "routing", "d8")
    if routing not in ("d8", "rho8"):
        raise NotImplementedError(
            f"tiled accumulation is D8/rho8 only; got routing={routing!r} (divergent flow has no fixed-halo "
            "closure — keep it on the in-memory engine)"
        )
    if client is not None or workers > 1:
        from digitalrivers._outofcore.distributed import (
            flow_accumulation_dask,
        )  # noqa: PLC0415

        return flow_accumulation_dask(
            fdir,
            out_path,
            weights=weights,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            scheduler=scheduler,
            client=client,
        )
    from digitalrivers._numba import (  # noqa: PLC0415
        _DIR_DC_I32,
        _DIR_DR_I32,
        kahn_accumulate_d8_numba,
    )

    rows, cols = fdir.rows, fdir.columns
    specs = plan_tiles(rows, cols, tile_rows, tile_cols, halo=1)
    by_grid = {(s.row, s.col): s for s in specs}
    dr = _DIR_DR_I32
    dc = _DIR_DC_I32

    out = Dataset.create_empty(
        rows,
        cols,
        dtype="float32",
        geo=fdir.geotransform,
        epsg=fdir.epsg,
        no_data_value=-1.0,
        driver_type="GTiff",
        path=out_path,
    )

    def load_tile(spec):
        fd = _read_core(fdir, spec, rows, cols, np.int32)
        if weights is None:
            w = np.ones(fd.shape, dtype=np.float64)
        else:
            w = _read_core(weights, spec, rows, cols, np.float64)
        return fd, w

    def bucket_inflow(
        inflow: dict[int, float]
    ) -> dict[int, dict[tuple[int, int], float]]:
        buckets: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
        for cell_id, amount in inflow.items():
            r, c = divmod(cell_id, cols)
            spec = by_grid.get(_tile_grid_index(r, c, tile_rows, tile_cols))
            if spec is None:
                continue
            buckets[spec.tid][(r - spec.row_off, c - spec.col_off)] = amount
        return buckets

    def inflow_weight(spec, fd, buckets) -> np.ndarray:
        iw = np.zeros(fd.shape, dtype=np.float64)
        for (i, j), amount in buckets.get(spec.tid, {}).items():
            iw[i, j] += amount
        return iw

    def tile_exports(spec, fd, w, acc) -> dict[int, float]:
        """Flow leaving this tile at boundary cells, keyed by the receiver's global cell id."""
        exports: dict[int, float] = defaultdict(float)
        n_rows, n_cols = fd.shape
        r0, r1 = spec.row_off, spec.row_off + n_rows
        c0, c1 = spec.col_off, spec.col_off + n_cols
        cells = []
        for j in range(n_cols):
            cells.append((0, j))
            cells.append((n_rows - 1, j))
        for i in range(n_rows):
            cells.append((i, 0))
            cells.append((i, n_cols - 1))
        # dict.fromkeys dedups while preserving insertion order, so the export-sum order is deterministic
        # (unlike set() iteration) — float addition is not associative.
        for i, j in dict.fromkeys(cells):
            d = int(fd[i, j])
            if d < 0 or d > 7:
                continue
            gr = r0 + i + int(dr[d])
            gc = c0 + j + int(dc[d])
            if gr < 0 or gr >= rows or gc < 0 or gc >= cols:
                continue
            if r0 <= gr < r1 and c0 <= gc < c1:
                continue  # internal receiver
            exports[gr * cols + gc] += acc[i, j] + w[i, j]
        return exports

    # --- fixed-point perimeter exchange ---
    # Each round propagates external inflow one more tile-hop along the (acyclic) flow graph. A path can re-enter a
    # tile at progressively lower cells, so the number of hops is bounded by the total perimeter cell count, not by
    # len(specs); cap accordingly and RAISE on non-convergence rather than ever returning a silently-wrong result.
    inflow: dict[int, float] = {}
    perimeter_cells = sum(2 * (s.n_rows + s.n_cols) for s in specs)
    max_rounds = max(64, perimeter_cells + 2)
    converged = False
    for _ in range(max_rounds):
        buckets = bucket_inflow(inflow)
        next_inflow: dict[int, float] = defaultdict(float)
        for spec in specs:
            fd, w = load_tile(spec)
            iw = inflow_weight(spec, fd, buckets)
            acc = kahn_accumulate_d8_numba(fd, w + iw, dr, dc) + iw
            for cell_id, amount in tile_exports(spec, fd, w, acc).items():
                next_inflow[cell_id] += amount
        next_inflow = {k: v for k, v in next_inflow.items() if v != 0.0}
        if next_inflow == inflow:
            converged = True
            break
        inflow = next_inflow
    if not converged:
        raise RuntimeError(
            f"tiled accumulation did not converge within {max_rounds} perimeter-exchange rounds; "
            "this should not happen for acyclic D8 flow — please report with the input"
        )

    # --- finalize: write acc = kahn(weights + inflow) + inflow ---
    buckets = bucket_inflow(inflow)
    for spec in specs:
        fd, w = load_tile(spec)
        iw = inflow_weight(spec, fd, buckets)
        acc = kahn_accumulate_d8_numba(fd, w + iw, dr, dc) + iw
        write_core(out, spec, acc.astype(np.float32))
    return out
