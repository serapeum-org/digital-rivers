"""Dask-distributed execution for the tiled fill / accumulation (B7).

The Barnes design is "two MapReduce operations": the per-tile **map** passes are embarrassingly parallel and the
edge **reduce** is a cheap serial graph solve on the producer. This module runs the map passes through
``dask.delayed`` (any scheduler, or a ``distributed.Client``) while the reduce stays in-process — reusing the same
kernels and graph as the serial path, so the result is identical.

Design notes:

* Workers reopen the source **by path** (`Dataset.read_file`), so the source must be file-backed; pyramids'
  ``CachingFileManager`` makes repeated re-opens cheap and a GDAL handle is never pickled.
* Per-tile *interiors* never leave the worker: stage 1 returns only perimeter-sized payloads (label count, local
  spill edges, outlet edges, border strips). Stage 3 returns the finished **core tile array**, which the producer
  writes sequentially — avoiding concurrent writes to one GeoTIFF entirely.
* When a real ``distributed.Client`` is supplied, ``pyramids.configure(client=...)`` replays the GDAL/cloud env on
  every worker.
"""

from __future__ import annotations

import os

import numpy as np
from pyramids.dataset import Dataset

from digitalrivers._outofcore.fill import (
    _edge_strips,
    _flood_tile,
    collect_outlet_edges,
    out_dtype,
)
from digitalrivers._outofcore.spillgraph import GlobalSpillGraph
from digitalrivers._outofcore.tiling import plan_tiles, write_core


def _source_path(dataset) -> str:
    """Return a reopenable path for ``dataset``, or raise if it is not file-backed.

    Workers reopen the source by path, so an in-memory MEM dataset (empty description) or a description that is
    neither an on-disk file nor a GDAL ``/vsi`` virtual path cannot be used by the dask backend.
    """
    path = dataset.raster.GetDescription()
    if not path:
        raise ValueError(
            "the dask backend requires a file-backed source (an in-memory MEM dataset cannot be reopened on "
            "workers); write the source to disk first or use the serial engine"
        )
    if not os.path.exists(path) and not path.startswith("/vsi"):
        raise ValueError(
            f"the dask backend cannot reopen the source by path: {path!r} is not an on-disk file or a GDAL /vsi "
            "path; write the source to a GeoTIFF first or use the serial engine"
        )
    return path


def _compute(delayeds, scheduler, client):
    import dask  # noqa: PLC0415

    if client is not None:
        return client.gather(client.compute(list(delayeds)))
    return list(dask.compute(*delayeds, scheduler=scheduler))


# --- fill (Barnes 2016) workers ---------------------------------------------------------------


def _consume_fill_tile(path, spec, rows, cols, nodata):
    """Stage-1 map: flood one tile locally; return perimeter-sized payload in LOCAL labels."""
    ds = Dataset.read_file(path)
    try:
        filled, glabels, n_local, halo_arr, core = _flood_tile(
            ds, spec, rows, cols, nodata, 0
        )
        local = GlobalSpillGraph()
        local.add_adjacency(glabels, filled)
        outlets = collect_outlet_edges(
            spec, glabels, filled, halo_arr, core, nodata, rows, cols
        )
        strips = _edge_strips(glabels, filled)
        return spec.tid, n_local, local.edges, outlets, strips
    finally:
        ds.close()


def _finalize_fill_tile(path, spec, rows, cols, nodata, offset, drainvec, dtype):
    """Stage-3 map: recompute the flood, raise to drain levels, return the finished core tile array."""
    ds = Dataset.read_file(path)
    try:
        filled, glabels, _n, _h, _c = _flood_tile(ds, spec, rows, cols, nodata, offset)
        label_offset = len(drainvec) - 1
        levels = drainvec[np.clip(glabels, 0, label_offset)]
        raised = np.where(glabels >= 1, np.maximum(filled, levels), filled)
        out_nodata = -9999.0 if nodata is None else nodata
        return spec.tid, np.where(np.isnan(raised), out_nodata, raised).astype(dtype)
    finally:
        ds.close()


def fill_depressions_dask(
    dem,
    out_path: str,
    *,
    tile_rows: int = 2048,
    tile_cols: int = 2048,
    scheduler: str = "threads",
    client=None,
):
    """Dask-distributed Barnes-2016 tiled fill (epsilon=0). Result identical to the serial engine."""
    import dask  # noqa: PLC0415

    if client is not None:
        from pyramids import configure  # noqa: PLC0415

        configure(client=client)

    path = _source_path(dem)
    rows, cols = dem.rows, dem.columns
    nodata = dem.no_data_value[0] if dem.no_data_value else None
    dtype = out_dtype(dem)
    specs = plan_tiles(rows, cols, tile_rows, tile_cols, halo=1)
    by_grid = {(s.row, s.col): s for s in specs}

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

    # stage 1: parallel map -> local payloads
    consumed = _compute(
        (dask.delayed(_consume_fill_tile)(path, s, rows, cols, nodata) for s in specs),
        scheduler,
        client,
    )
    by_tid = {
        tid: (n_local, edges, outlets, strips)
        for tid, n_local, edges, outlets, strips in consumed
    }

    # producer: prefix-sum offsets, remap local -> global, build + solve graph
    offsets: dict[int, int] = {}
    label_offset = 0
    for s in specs:
        offsets[s.tid] = label_offset
        label_offset += by_tid[s.tid][0]

    graph = GlobalSpillGraph()
    strips_global: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for s in specs:
        off = offsets[s.tid]
        _n, edges, outlets, strips = by_tid[s.tid]
        for (la, lb), e in edges.items():
            graph.add_edge(la + off, lb + off, e)
        for la, e in outlets:
            graph.add_outlet(la + off, e)
        strips_global[s.tid] = {
            side: (np.where(lab >= 1, lab + off, 0), fil)
            for side, (lab, fil) in strips.items()
        }

    for s in specs:
        right = by_grid.get((s.row, s.col + 1))
        if right is not None:
            graph.join_strips(
                *strips_global[s.tid]["right"], *strips_global[right.tid]["left"]
            )
        below = by_grid.get((s.row + 1, s.col))
        if below is not None:
            graph.join_strips(
                *strips_global[s.tid]["bottom"], *strips_global[below.tid]["top"]
            )
        diag = by_grid.get((s.row + 1, s.col + 1))
        if diag is not None:
            a_lab, a_fil = strips_global[s.tid]["bottom"]
            d_lab, d_fil = strips_global[diag.tid]["top"]
            if a_lab[-1] >= 1 and d_lab[0] >= 1:
                graph.add_edge(
                    int(a_lab[-1]),
                    int(d_lab[0]),
                    max(float(a_fil[-1]), float(d_fil[0])),
                )
        anti = by_grid.get((s.row + 1, s.col - 1))
        if anti is not None:
            a_lab, a_fil = strips_global[s.tid]["bottom"]
            d_lab, d_fil = strips_global[anti.tid]["top"]
            if a_lab[0] >= 1 and d_lab[-1] >= 1:
                graph.add_edge(
                    int(a_lab[0]),
                    int(d_lab[-1]),
                    max(float(a_fil[0]), float(d_fil[-1])),
                )

    drain = graph.solve()
    drainvec = np.full(label_offset + 1, -np.inf, dtype=np.float64)
    for label, level in drain.items():
        if 1 <= label <= label_offset:
            drainvec[label] = level

    # stage 3: parallel map -> finished core tiles; producer writes sequentially
    finalized = _compute(
        (
            dask.delayed(_finalize_fill_tile)(
                path, s, rows, cols, nodata, offsets[s.tid], drainvec, dtype
            )
            for s in specs
        ),
        scheduler,
        client,
    )
    tiles = {tid: arr for tid, arr in finalized}
    for s in specs:
        write_core(out, s, tiles[s.tid])
    return out


# --- accumulation (Barnes 2017, D8/Rho8) workers ----------------------------------------------


def _accum_read_core(ds, spec, rows, cols, dtype):
    from digitalrivers._outofcore.tiling import read_tile  # noqa: PLC0415

    arr, core = read_tile(ds, spec, rows, cols)
    return np.ascontiguousarray(arr[core], dtype=dtype)


def _accum_load(path, weights_path, spec, rows, cols):
    ds = Dataset.read_file(path)
    fd = _accum_read_core(ds, spec, rows, cols, np.int32)
    ds.close()
    if weights_path is None:
        w = np.ones(fd.shape, dtype=np.float64)
    else:
        wds = Dataset.read_file(weights_path)
        w = _accum_read_core(wds, spec, rows, cols, np.float64)
        wds.close()
    return fd, w


def _accum_exports(spec, fd, w, acc, rows, cols, dr, dc):
    exports: dict[int, float] = {}
    n_rows, n_cols = fd.shape
    r0, c0, r1, c1 = (
        spec.row_off,
        spec.col_off,
        spec.row_off + n_rows,
        spec.col_off + n_cols,
    )
    cells = []
    for j in range(n_cols):
        cells.append((0, j))
        cells.append((n_rows - 1, j))
    for i in range(n_rows):
        cells.append((i, 0))
        cells.append((i, n_cols - 1))
    # dict.fromkeys dedups deterministically (order-stable) so the export-sum order is reproducible.
    for i, j in dict.fromkeys(cells):
        d = int(fd[i, j])
        if d < 0 or d > 7:
            continue
        gr = r0 + i + int(dr[d])
        gc = c0 + j + int(dc[d])
        if (
            gr < 0
            or gr >= rows
            or gc < 0
            or gc >= cols
            or (r0 <= gr < r1 and c0 <= gc < c1)
        ):
            continue
        gid = gr * cols + gc
        exports[gid] = exports.get(gid, 0.0) + float(acc[i, j] + w[i, j])
    return exports


def _accum_tile(path, weights_path, spec, rows, cols, inflow_items, finalize):
    """One tile's round: acc = kahn(w+inflow)+inflow; return exports (round) or the core acc (finalize)."""
    from digitalrivers._numba import (  # noqa: PLC0415
        _DIR_DC_I32,
        _DIR_DR_I32,
        kahn_accumulate_d8_numba,
    )

    fd, w = _accum_load(path, weights_path, spec, rows, cols)
    iw = np.zeros(fd.shape, dtype=np.float64)
    for (i, j), amount in inflow_items:
        iw[i, j] += amount
    acc = kahn_accumulate_d8_numba(fd, w + iw, _DIR_DR_I32, _DIR_DC_I32) + iw
    if finalize:
        return spec.tid, acc.astype(np.float32)
    return spec.tid, _accum_exports(
        spec, fd, w, acc, rows, cols, _DIR_DR_I32, _DIR_DC_I32
    )


def flow_accumulation_dask(
    fdir,
    out_path: str,
    *,
    weights=None,
    tile_rows: int = 2048,
    tile_cols: int = 2048,
    scheduler: str = "threads",
    client=None,
):
    """Dask-distributed D8/Rho8 tiled accumulation. Result identical to the serial engine."""
    import dask  # noqa: PLC0415

    routing = getattr(fdir, "routing", "d8")
    if routing not in ("d8", "rho8"):
        raise NotImplementedError(
            f"tiled accumulation is D8/rho8 only; got routing={routing!r}"
        )
    if client is not None:
        from pyramids import configure  # noqa: PLC0415

        configure(client=client)

    path = _source_path(fdir)
    weights_path = _source_path(weights) if weights is not None else None
    rows, cols = fdir.rows, fdir.columns
    specs = plan_tiles(rows, cols, tile_rows, tile_cols, halo=1)
    by_grid_tile = {(s.row, s.col): s for s in specs}

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

    def buckets_for(inflow):
        buckets: dict[int, list[tuple[tuple[int, int], float]]] = {}
        for cell_id, amount in inflow.items():
            r, c = divmod(cell_id, cols)
            spec = by_grid_tile.get((r // tile_rows, c // tile_cols))
            if spec is None:
                continue
            buckets.setdefault(spec.tid, []).append(
                ((r - spec.row_off, c - spec.col_off), amount)
            )
        return buckets

    inflow: dict[int, float] = {}
    perimeter_cells = sum(2 * (s.n_rows + s.n_cols) for s in specs)
    max_rounds = max(64, perimeter_cells + 2)
    converged = False
    for _ in range(max_rounds):
        buckets = buckets_for(inflow)
        results = _compute(
            (
                dask.delayed(_accum_tile)(
                    path, weights_path, s, rows, cols, buckets.get(s.tid, []), False
                )
                for s in specs
            ),
            scheduler,
            client,
        )
        next_inflow: dict[int, float] = {}
        for _tid, exports in results:
            for gid, amount in exports.items():
                next_inflow[gid] = next_inflow.get(gid, 0.0) + amount
        next_inflow = {k: v for k, v in next_inflow.items() if v != 0.0}
        if next_inflow == inflow:
            converged = True
            break
        inflow = next_inflow
    if not converged:
        raise RuntimeError(
            f"tiled accumulation did not converge within {max_rounds} rounds"
        )

    buckets = buckets_for(inflow)
    finalized = _compute(
        (
            dask.delayed(_accum_tile)(
                path, weights_path, s, rows, cols, buckets.get(s.tid, []), True
            )
            for s in specs
        ),
        scheduler,
        client,
    )
    tiles = {tid: arr for tid, arr in finalized}
    for s in specs:
        write_core(out, s, tiles[s.tid])
    return out
