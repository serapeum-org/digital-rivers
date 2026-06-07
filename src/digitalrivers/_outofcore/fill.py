"""Tiled, out-of-core depression fill — Barnes 2016 master-graph method (B3).

Two MapReduce passes over a larger-than-RAM DEM:

1. **map** — flood each tile independently with the watershed-labelled Priority-Flood (B2), treating the tile's
   own edges as local outlets.
2. **reduce** — stitch tile perimeters into one :class:`~digitalrivers._outofcore.spillgraph.GlobalSpillGraph`
   (intra-tile saddles, cross-seam saddles, and outlet connections for true domain-edge / no-data-adjacent
   cells), then solve it for each watershed's global drainage elevation.
3. **map** — raise every cell to ``max(local_filled, drain[label])`` and write the tile core to disk.

For ``epsilon = 0`` this is bit-for-bit identical to a whole-array Priority-Flood, because the fill is always a
selection among the original elevations and the master graph assigns one consistent spill level per watershed.
``epsilon > 0`` does **not** compose across seams and is rejected (see the out-of-core plan §2.3 / B6).
"""

from __future__ import annotations

import numpy as np
from pyramids.dataset import Dataset

from digitalrivers._outofcore.spillgraph import GlobalSpillGraph
from digitalrivers._outofcore.tiling import TileSpec, plan_tiles, read_tile, write_core


def _nodata_mask(elev: np.ndarray, nodata: float | None) -> np.ndarray:
    """Boolean no-data mask: NaN cells, plus cells equal to ``nodata`` if a sentinel is given."""
    mask = np.isnan(elev)
    if nodata is not None and not np.isnan(nodata):
        mask = mask | (elev == nodata)
    return mask


def _flood_tile(
    dem, spec: TileSpec, full_rows: int, full_cols: int, nodata, offset: int
):
    """Run the labelled flood on one tile's core; return ``(filled, glabels, halo_arr, core)``."""
    # Lazy import keeps `import digitalrivers` numba-free (CLAUDE.md rule).
    from digitalrivers._numba import (  # noqa: PLC0415
        _DIR_DC_I32,
        _DIR_DR_I32,
        priority_flood_labels_numba,
    )

    halo_arr, core = read_tile(dem, spec, full_rows, full_cols)
    core_elev = np.ascontiguousarray(halo_arr[core], dtype=np.float64)
    mask = _nodata_mask(core_elev, nodata)
    filled, llabels = priority_flood_labels_numba(
        core_elev, mask, _DIR_DR_I32, _DIR_DC_I32
    )
    n_local = int(llabels.max()) if llabels.size else 0
    glabels = np.where(llabels >= 1, llabels.astype(np.int64) + offset, 0)
    return filled, glabels, n_local, halo_arr, core


def _add_outlet_edges(
    graph, spec, glabels, filled, halo_arr, core, nodata, full_rows, full_cols
):
    """Connect true-outlet cells (domain edge or no-data-adjacent) to the OUTLET node at their filled elevation."""
    from digitalrivers._numba import _DIR_DC_I32, _DIR_DR_I32  # noqa: PLC0415

    halo_nodata = _nodata_mask(np.asarray(halo_arr, dtype=np.float64), nodata)
    rsl, csl = core
    hr0, hc0 = rsl.start, csl.start
    hrows, hcols = halo_nodata.shape
    n_rows, n_cols = glabels.shape
    for i in range(n_rows):
        gr = spec.row_off + i
        on_row_edge = gr == 0 or gr == full_rows - 1
        for j in range(n_cols):
            lab = int(glabels[i, j])
            if lab < 1:
                continue
            gc = spec.col_off + j
            outlet = on_row_edge or gc == 0 or gc == full_cols - 1
            if not outlet:
                hi, hj = hr0 + i, hc0 + j
                for k in range(8):
                    ni = hi + int(_DIR_DR_I32[k])
                    nj = hj + int(_DIR_DC_I32[k])
                    if 0 <= ni < hrows and 0 <= nj < hcols and halo_nodata[ni, nj]:
                        outlet = True
                        break
            if outlet:
                graph.add_outlet(lab, float(filled[i, j]))


def _edge_strips(
    glabels: np.ndarray, filled: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The four 1-cell-thick border strips (labels, filled) used to stitch adjacent tiles."""
    return {
        "top": (glabels[0, :].copy(), filled[0, :].copy()),
        "bottom": (glabels[-1, :].copy(), filled[-1, :].copy()),
        "left": (glabels[:, 0].copy(), filled[:, 0].copy()),
        "right": (glabels[:, -1].copy(), filled[:, -1].copy()),
    }


def fill_depressions_tiled(
    dem,
    out_path: str,
    *,
    tile_rows: int = 2048,
    tile_cols: int = 2048,
    epsilon: float = 0.0,
    cache: str = "evict",
    workers: int = 1,
    scratch_dir: str | None = None,
):
    """Out-of-core depression fill (Barnes 2016 master-graph, ``epsilon = 0`` only).

    Args:
        dem: Source `pyramids` ``Dataset`` (or ``DEM``) to fill.
        out_path: Path of the GeoTIFF to create and stream the filled result into.
        tile_rows: Core tile height in cells. Defaults to 2048.
        tile_cols: Core tile width in cells. Defaults to 2048.
        epsilon: Per-step lift. Only ``0.0`` is seam-correct; anything else raises ``NotImplementedError``.
        cache: ``TileStore`` mode — ``"evict"`` (default), ``"retain"``, or ``"cache"``.
        workers: Reserved for the dask backend (B7); currently the loop is serial.
        scratch_dir: Scratch directory for ``cache`` mode.

    Returns:
        The filled `pyramids` ``Dataset`` opened on ``out_path``.

    Raises:
        NotImplementedError: If ``epsilon != 0`` (see B6 / plan §2.3).
    """
    if epsilon != 0.0:
        raise NotImplementedError(
            "tiled fill is seam-correct for epsilon=0 only (monotonic eps-fill does not compose across "
            "tile seams; see B6 / the out-of-core plan §2.3)"
        )
    from digitalrivers._outofcore.cache import TileStore  # noqa: PLC0415

    rows, cols = dem.rows, dem.columns
    nodata = dem.no_data_value[0] if dem.no_data_value else None
    specs = plan_tiles(rows, cols, tile_rows, tile_cols, halo=1)
    by_grid = {(s.row, s.col): s for s in specs}

    out = Dataset.create_empty(
        rows,
        cols,
        dtype="float32",
        geo=dem.geotransform,
        epsg=dem.epsg,
        no_data_value=-9999.0 if nodata is None else nodata,
        driver_type="GTiff",
        path=out_path,
    )
    graph = GlobalSpillGraph()
    store = TileStore(cache, scratch_dir)
    strips: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    offsets: dict[int, int] = {}

    # --- stage 1: map (per-tile labelled flood + local graph contributions) ---
    label_offset = 0
    for s in specs:
        offsets[s.tid] = label_offset
        filled, glabels, n_local, halo_arr, core = _flood_tile(
            dem, s, rows, cols, nodata, label_offset
        )
        label_offset += n_local
        graph.add_adjacency(glabels, filled)
        _add_outlet_edges(graph, s, glabels, filled, halo_arr, core, nodata, rows, cols)
        strips[s.tid] = _edge_strips(glabels, filled)
        if not store.recompute:
            store.put(s.tid, filled=filled, glabels=glabels)

    # --- stage 2: reduce (stitch seams + global solve) ---
    for s in specs:
        right = by_grid.get((s.row, s.col + 1))
        if right is not None:
            a_lab, a_fil = strips[s.tid]["right"]
            b_lab, b_fil = strips[right.tid]["left"]
            graph.join_strips(a_lab, a_fil, b_lab, b_fil)
        below = by_grid.get((s.row + 1, s.col))
        if below is not None:
            a_lab, a_fil = strips[s.tid]["bottom"]
            b_lab, b_fil = strips[below.tid]["top"]
            graph.join_strips(a_lab, a_fil, b_lab, b_fil)
        # Tile-corner diagonals (where four tiles meet): single corner-to-corner adjacencies not covered by the
        # orthogonal-neighbour seam joins above.
        diag = by_grid.get((s.row + 1, s.col + 1))
        if diag is not None:
            a_lab, a_fil = strips[s.tid]["bottom"]
            d_lab, d_fil = strips[diag.tid]["top"]
            if a_lab[-1] >= 1 and d_lab[0] >= 1:
                graph.add_edge(
                    int(a_lab[-1]),
                    int(d_lab[0]),
                    max(float(a_fil[-1]), float(d_fil[0])),
                )
        anti = by_grid.get((s.row + 1, s.col - 1))
        if anti is not None:
            a_lab, a_fil = strips[s.tid]["bottom"]
            d_lab, d_fil = strips[anti.tid]["top"]
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

    # --- stage 3: map (raise + write) ---
    for s in specs:
        if store.recompute:
            filled, glabels, _, _, _ = _flood_tile(
                dem, s, rows, cols, nodata, offsets[s.tid]
            )
        else:
            data = store.get(s.tid)
            filled, glabels = data["filled"], data["glabels"]
        clipped = np.clip(glabels, 0, label_offset)
        levels = drainvec[clipped]
        raised = np.where(glabels >= 1, np.maximum(filled, levels), filled)
        out_nodata = -9999.0 if nodata is None else nodata
        raised = np.where(np.isnan(raised), out_nodata, raised).astype(np.float32)
        write_core(out, s, raised)
    return out
