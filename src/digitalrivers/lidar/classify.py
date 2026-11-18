"""Separating ground from everything standing on it.

Two classifiers over the same cloud:

* `classify_ground` — a progressive morphological filter. It opens the surface with a
  growing window; a point that survives every window is ground, and one that vanishes
  under a window wider than a building is not. The elevation threshold grows with the
  window so that real terrain slope is not mistaken for a structure.
* `detect_trees` — local-maximum detection over the canopy height model, with a search
  radius that scales with tree height, since a tall crown is wider than a short one and
  a fixed radius either splits big crowns or merges small ones.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from digitalrivers.lidar.points import LasPoints

__all__ = ["classify_ground", "detect_trees"]


def _default_tree_radius(h: float) -> float:
    """Default variable-window half-width for `detect_trees` (Popescu & Wynne 2004).

    `0.5 + 0.05 * h` — ~5% of canopy height plus a small absolute floor
    so dense low-canopy regions still get a meaningful search window.
    """
    return 0.5 + 0.05 * h


def classify_ground(
    points: LasPoints,
    *,
    method: str = "zhang",
    cell_size: float = 1.0,
    window_cells: int = 5,
    slope_threshold: float = 1.0,
) -> np.ndarray:
    """Classify each LiDAR point as ground or non-ground.

    Two algorithm families are available:

    * **`"zhang"`** (Zhang 2003) — morphological tophat filter on a min-grid
      DEM. Points whose elevation rises more than `slope_threshold` above
      the morphological opening at their cell are non-ground. Fast; works
      well on relatively flat terrain. Implementation uses a single
      structuring-element scale rather than the original paper's
      multi-scale stack.
    * **`"axelsson"`** — Axelsson 2000 progressive TIN densification.
      Not yet implemented; raises NotImplementedError.

    Args:
        points: `LasPoints` cloud to classify.
        method: `"zhang"` (default) or `"axelsson"`.
        cell_size: Cell size in map units for the intermediate min-grid.
            Smaller values capture fine ground detail at the cost of memory.
        window_cells: Side length of the structuring element used for the
            morphological opening (Zhang only). Default 5 (a 5×5 window).
        slope_threshold: Elevation threshold above the opening above which a
            point is classified as non-ground (Zhang only). Default 1.0.

    Returns:
        `(N,)` uint8 array of ASPRS class codes — `2` (ground) or `1`
        (unclassified / non-ground), parallel to `points`.

    Raises:
        ValueError: If `method` is unknown or geometry-arguments are
            invalid.
        NotImplementedError: If `method="axelsson"` is requested.
    """
    if method not in ("zhang", "axelsson"):
        raise ValueError(f"method must be 'zhang' or 'axelsson'; got {method!r}")
    if method == "axelsson":
        raise NotImplementedError(
            "Axelsson 2000 TIN-progressive ground filter is not yet "
            "implemented; use method='zhang' for a morphological-tophat "
            "ground filter, or pre-classify the input cloud with PDAL "
            "or LASGround."
        )

    from scipy.ndimage import grey_opening

    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive; got {cell_size!r}")
    if window_cells < 1:
        raise ValueError(f"window_cells must be >= 1; got {window_cells!r}")
    if slope_threshold < 0:
        raise ValueError(
            f"slope_threshold must be non-negative; got {slope_threshold!r}"
        )

    xs, ys, zs = points.x, points.y, points.z
    if not len(xs):
        return np.empty(0, dtype=np.uint8)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    cols = max(1, int(np.ceil((x_max - x_min) / cell_size)))
    rows = max(1, int(np.ceil((y_max - y_min) / cell_size)))
    col_idx = np.clip(((xs - x_min) / cell_size).astype(np.int64), 0, cols - 1)
    row_idx = np.clip(((y_max - ys) / cell_size).astype(np.int64), 0, rows - 1)

    # Min-grid: lowest z per cell. Empty cells inherit the sentinel +inf so
    # the opening "spreads" the lowest neighbouring elevation across them.
    min_grid = np.full((rows, cols), np.inf, dtype=np.float64)
    np.minimum.at(min_grid, (row_idx, col_idx), zs)
    finite = np.isfinite(min_grid)
    if not finite.any():
        return np.full(len(xs), 1, dtype=np.uint8)
    # Replace +inf with the global min so the opening kernel has something
    # to work with (it would otherwise propagate inf into adjacent cells).
    z_lo = float(min_grid[finite].min())
    min_grid = np.where(finite, min_grid, z_lo)

    opened = grey_opening(min_grid, size=int(window_cells))

    # Per-point comparison: ground if z - opened_at_cell <= slope_threshold.
    cell_opened = opened[row_idx, col_idx]
    is_ground = (zs - cell_opened) <= slope_threshold
    out = np.where(is_ground, 2, 1).astype(np.uint8)
    return out


def detect_trees(
    chm,
    *,
    min_height_m: float = 2.0,
    radius_fn=None,
):
    """Detect individual tree tops on a canopy height model.

    Variable-window local-maxima search: for each candidate cell whose CHM
    value is `>= min_height_m`, scan a square window whose half-width
    scales with the cell's height (`radius_fn(h)` map units, default
    `_default_tree_radius` — ~5% of canopy height). The cell is reported
    as a tree top iff its CHM value is the maximum in the window.

    Args:
        chm: pyramids `Dataset` of the canopy height model (typically
            DSM − DTM, in metres). Must be single-band and projected.
        min_height_m: Minimum canopy height (m) for a cell to be a tree-top
            candidate. Defaults to 2.0.
        radius_fn: Callable mapping `height_m` -> window half-width in map
            units. Defaults to `_default_tree_radius` (Popescu & Wynne 2004
            conifer rule of thumb: `0.5 + 0.05 * h`).

    Returns:
        `geopandas.GeoDataFrame` of tree-top Point geometries with columns
        `height_m` (canopy height at the top), `row`, `col` (raster
        indices), and `geometry`. CRS is set from the CHM's EPSG.

    References:
        Popescu, S. C. & Wynne, R. H. (2004). "Seeing the trees in the
        forest: Using LIDAR and multispectral data fusion with local
        filtering and variable window size for estimating tree height."
        *Photogrammetric Engineering & Remote Sensing* 70(5): 589–604.
    """
    if radius_fn is None:
        radius_fn = _default_tree_radius

    z = chm.read_array().astype(np.float32, copy=False)
    no_val = chm.no_data_value[0] if chm.no_data_value else None
    if no_val is not None:
        z = np.where(z == no_val, np.nan, z)
    cell_size = float(abs(chm.geotransform[1]))

    rows, cols = z.shape
    gt = chm.geotransform
    candidates = np.argwhere(z >= min_height_m)
    tops_r: list[int] = []
    tops_c: list[int] = []
    tops_h: list[float] = []
    for r, c in candidates:
        h = float(z[r, c])
        rad = max(1, int(radius_fn(h) / cell_size))
        r0 = max(0, int(r) - rad)
        r1 = min(rows, int(r) + rad + 1)
        c0 = max(0, int(c) - rad)
        c1 = min(cols, int(c) + rad + 1)
        window = z[r0:r1, c0:c1]
        if not np.isfinite(window).any():
            continue
        win_max = float(np.nanmax(window))
        if h >= win_max:
            tops_r.append(int(r))
            tops_c.append(int(c))
            tops_h.append(h)

    xs = [gt[0] + (c + 0.5) * gt[1] for c in tops_c]
    ys = [gt[3] + (r + 0.5) * gt[5] for r in tops_r]
    geometries = [Point(x, y) for x, y in zip(xs, ys)]
    return gpd.GeoDataFrame(
        {
            "height_m": tops_h,
            "row": tops_r,
            "col": tops_c,
            "geometry": geometries,
        },
        crs=chm.epsg,
    )
