"""LiDAR point-cloud I/O, gridding, ground filtering, and analysis.

* :class:`LasPoints` — in-memory point-cloud record (xyz + intensity +
  classification + return-number + CRS).
* :func:`read_las` / :func:`write_las` — LAS / LAZ I/O via `laspy`.
* :func:`grid_lidar_points` — bucket raw `(x, y, z)` arrays into a
  gridded DEM with min / max / mean / median aggregation.

Reading / writing LAS files requires `laspy`. Install with
`pip install laspy[lazrs]` to also handle LAZ compression. The
gridding helper does not require laspy and operates on raw arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_LASPY_HINT = (
    "laspy is required for LAS / LAZ I/O. Install with "
    "`pip install laspy[lazrs]`."
)


@dataclass
class LasPoints:
    """In-memory LiDAR point cloud.

    Numeric arrays are all parallel — index `i` selects the i-th point
    across every field. `classification` follows the ASPRS LAS standard
    (2 = ground, 5 = high vegetation, 6 = building, etc.).

    Attributes:
        x: `(N,)` float64 array of x-coordinates.
        y: `(N,)` float64 array of y-coordinates.
        z: `(N,)` float64 array of z-coordinates (elevation).
        intensity: `(N,)` uint16 array of return intensity, or empty.
        classification: `(N,)` uint8 array of ASPRS class codes, or empty.
        return_number: `(N,)` uint8 array of return-number-within-pulse,
            or empty.
        crs: Optional CRS object (whatever `laspy.LasHeader.parse_crs`
            returns; typically `pyproj.CRS`).
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    intensity: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint16),
    )
    classification: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8),
    )
    return_number: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8),
    )
    crs: object | None = None

    def __post_init__(self) -> None:
        n = len(self.x)
        if not (len(self.y) == n and len(self.z) == n):
            raise ValueError(
                f"x/y/z must have the same length; got {len(self.x)}, "
                f"{len(self.y)}, {len(self.z)}"
            )

    def __len__(self) -> int:
        """Number of points in the cloud."""
        return int(self.x.shape[0])

    def subset(self, mask: np.ndarray) -> "LasPoints":
        """Return a new `LasPoints` containing only the points where `mask` is True.

        Args:
            mask: `(N,)` bool array (same length as the point cloud).

        Returns:
            A new `LasPoints` with the selected subset across every field.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.x.shape:
            raise ValueError(
                f"mask shape {mask.shape} != points shape {self.x.shape}"
            )
        kw = {"x": self.x[mask], "y": self.y[mask], "z": self.z[mask]}
        if self.intensity.size:
            kw["intensity"] = self.intensity[mask]
        if self.classification.size:
            kw["classification"] = self.classification[mask]
        if self.return_number.size:
            kw["return_number"] = self.return_number[mask]
        return LasPoints(crs=self.crs, **kw)


def read_las(path: str) -> LasPoints:
    """Read a LAS or LAZ file into a `LasPoints` record.

    Args:
        path: Filesystem path to a `.las` / `.laz` file.

    Returns:
        `LasPoints` populated from the file. xyz are scaled+offset by
        `laspy` so values are in the file's CRS units.

    Raises:
        ImportError: If `laspy` is not installed.
    """
    import warnings
    try:
        import laspy  # type: ignore
    except ImportError as exc:  # pragma: no cover — environment-specific
        raise ImportError(_LASPY_HINT) from exc
    f = laspy.read(path)
    # CRS parsing can fail on older LAS headers that pre-date the WKT
    # / VLR conventions laspy expects. Surface that to the caller as a
    # `UserWarning` (not a silent `crs = None`) so a downstream gridding /
    # reprojection step has something to act on.
    try:
        crs = f.header.parse_crs()
    except (ValueError, KeyError, AttributeError) as exc:
        warnings.warn(
            f"Could not parse CRS from LAS header {path!r}: {exc!r}. "
            f"Returning `crs=None`.",
            UserWarning,
            stacklevel=2,
        )
        crs = None
    intensity = (
        np.asarray(f.intensity, dtype=np.uint16)
        if hasattr(f, "intensity")
        else np.empty(0, dtype=np.uint16)
    )
    classification = (
        np.asarray(f.classification, dtype=np.uint8)
        if hasattr(f, "classification")
        else np.empty(0, dtype=np.uint8)
    )
    return_number = (
        np.asarray(f.return_number, dtype=np.uint8)
        if hasattr(f, "return_number")
        else np.empty(0, dtype=np.uint8)
    )
    return LasPoints(
        x=np.asarray(f.x, dtype=np.float64),
        y=np.asarray(f.y, dtype=np.float64),
        z=np.asarray(f.z, dtype=np.float64),
        intensity=intensity,
        classification=classification,
        return_number=return_number,
        crs=crs,
    )


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
        raise ValueError(
            f"method must be 'zhang' or 'axelsson'; got {method!r}"
        )
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
        raise ValueError(
            f"window_cells must be >= 1; got {window_cells!r}"
        )
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
    col_idx = np.clip(
        ((xs - x_min) / cell_size).astype(np.int64), 0, cols - 1
    )
    row_idx = np.clip(
        ((y_max - ys) / cell_size).astype(np.int64), 0, rows - 1
    )

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


def _default_tree_radius(h: float) -> float:
    """Default variable-window half-width for `detect_trees` (Popescu & Wynne 2004).

    `0.5 + 0.05 * h` — ~5% of canopy height plus a small absolute floor
    so dense low-canopy regions still get a meaningful search window.
    """
    return 0.5 + 0.05 * h


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
    import geopandas as gpd
    from shapely.geometry import Point

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


def clip(
    points: LasPoints,
    polygon,
    *,
    inverse: bool = False,
) -> LasPoints:
    """Clip a `LasPoints` cloud to a polygon (or its complement).

    Args:
        points: Input `LasPoints`.
        polygon: Shapely Polygon or MultiPolygon in the same CRS as `points`.
            All other geometry types are rejected.
        inverse: If False (default), keep only points inside `polygon`. If
            True, keep only points OUTSIDE the polygon (i.e. erase).

    Returns:
        A new `LasPoints` containing the surviving subset.
    """
    import shapely
    inside = shapely.contains_xy(polygon, points.x, points.y)
    keep = inside if not inverse else ~inside
    return points.subset(keep)


def merge(*pointclouds: LasPoints) -> LasPoints:
    """Concatenate two or more `LasPoints` into a single cloud.

    Numeric arrays are stacked field-by-field. Optional fields (intensity /
    classification / return_number) are preserved only when every input
    carries them — otherwise the field on the output is empty (size 0).
    The CRS is taken from the first input.

    Args:
        *pointclouds: Two or more `LasPoints` to merge.

    Returns:
        A new `LasPoints` containing the concatenation.

    Raises:
        ValueError: If fewer than one cloud is supplied.
    """
    if not pointclouds:
        raise ValueError("merge requires at least one point cloud")
    x = np.concatenate([p.x for p in pointclouds])
    y = np.concatenate([p.y for p in pointclouds])
    z = np.concatenate([p.z for p in pointclouds])
    kw: dict = {}
    if all(p.intensity.size for p in pointclouds):
        kw["intensity"] = np.concatenate([p.intensity for p in pointclouds])
    if all(p.classification.size for p in pointclouds):
        kw["classification"] = np.concatenate(
            [p.classification for p in pointclouds]
        )
    if all(p.return_number.size for p in pointclouds):
        kw["return_number"] = np.concatenate(
            [p.return_number for p in pointclouds]
        )
    return LasPoints(x=x, y=y, z=z, crs=pointclouds[0].crs, **kw)


def filter_classes(
    points: LasPoints,
    classes: set[int] | list[int] | tuple[int, ...],
) -> LasPoints:
    """Keep only points whose ASPRS classification is in `classes`.

    Standard ASPRS codes include `2` (ground), `3` (low vegetation), `4`
    (medium vegetation), `5` (high vegetation), `6` (building), `9` (water).

    Args:
        points: Input `LasPoints`. Must carry a populated
            `classification` array (i.e. read from a LAS / LAZ file).
        classes: Iterable of integer class codes to keep.

    Returns:
        A new `LasPoints` containing only the matching subset.

    Raises:
        ValueError: If `points.classification` is empty (the cloud carries
            no class codes).
    """
    if not points.classification.size:
        raise ValueError(
            "points carries no classification data; nothing to filter"
        )
    keep = np.isin(points.classification, list(classes))
    return points.subset(keep)


def write_las(
    points: LasPoints,
    path: str,
    *,
    point_format: int = 6,
    version: str = "1.4",
) -> None:
    """Write a `LasPoints` cloud to a LAS or LAZ file.

    The file extension determines compression: `.laz` uses LAZ
    (requires `lazrs`), `.las` is uncompressed.

    Args:
        points: `LasPoints` to write.
        path: Output filesystem path.
        point_format: ASPRS LAS point format (default 6 — supports GPS time
            and high-precision returns; pick 0 for legacy compatibility).
        version: LAS version string (default `"1.4"`).

    Raises:
        ImportError: If `laspy` is not installed.
    """
    try:
        import laspy  # type: ignore
    except ImportError as exc:  # pragma: no cover — environment-specific
        raise ImportError(_LASPY_HINT) from exc
    header = laspy.LasHeader(point_format=point_format, version=version)
    out = laspy.LasData(header)
    out.x = points.x
    out.y = points.y
    out.z = points.z
    if points.intensity.size:
        out.intensity = points.intensity
    if points.classification.size:
        out.classification = points.classification
    if points.return_number.size:
        out.return_number = points.return_number
    out.write(path)


def grid_lidar_points(
    xs,
    ys,
    zs,
    cell_size: float,
    bounds=None,
    aggregate: str = "min",
    epsg: int = 4326,
    *,
    idw_k: int = 8,
    idw_power: float = 2.0,
    rbf_kernel: str = "thin_plate_spline",
    rbf_smoothing: float = 0.0,
):
    """Grid a LiDAR point cloud to a DEM.

    Pragmatic LiDAR-to-DEM step that operates on raw `(x, y, z)` arrays.
    Useful when the caller has read LAS / LAZ externally (via `read_las`)
    and wants a gridded surface.

    The `aggregate` parameter selects either a block-aggregation method or
    a spatial-interpolation method:

    **Block aggregation** (one value per cell, based on points that land
    inside the cell):

    * `"min"` (default) — canonical bare-earth choice for first-return LiDAR.
    * `"max"` — canopy / DSM choice.
    * `"mean"` / `"median"` — smoothed surfaces.
    * `"count"` — point-density raster.

    **Spatial interpolation** (one value per cell centre, computed from
    nearby points regardless of cell membership):

    * `"idw"` — inverse-distance-weighted mean of the K nearest points.
    * `"nn"` — nearest-neighbour assignment.
    * `"tin"` — barycentric interpolation on the Delaunay triangulation.
    * `"rbf"` — radial-basis-function interpolation (`scipy.interpolate.
      RBFInterpolator`).

    Args:
        xs / ys / zs: 1-D arrays of point coordinates.
        cell_size: output cell side length in map units (must match the
            CRS).
        bounds: `(x_min, y_min, x_max, y_max)` to clip the grid to. If
            `None`, the input points' bounding box is used.
        aggregate: One of `"min"`, `"max"`, `"mean"`, `"median"`,
            `"count"`, `"idw"`, `"nn"`, `"tin"`, `"rbf"`.
        epsg: EPSG code of the input coordinates.
        idw_k: Neighbour count for `aggregate="idw"`. Defaults to 8.
        idw_power: Distance exponent for IDW weights `(1 / d**power)`.
            Defaults to 2.0.
        rbf_kernel: Kernel name passed to `scipy.interpolate.RBFInterpolator`
            for `aggregate="rbf"`. Defaults to `"thin_plate_spline"`.
        rbf_smoothing: Smoothing parameter for the RBF kernel. Defaults
            to 0.0 (exact interpolation).

    Returns:
        A pyramids `Dataset` of the gridded surface.

    Raises:
        ValueError: For mismatched input lengths or unknown `aggregate`.

    Examples:
        - Bucket four points into a 2x1 grid with min aggregation:

            >>> import numpy as np
            >>> from digitalrivers.lidar import grid_lidar_points
            >>> xs = np.array([0.1, 0.2, 1.1])
            >>> ys = np.array([0.1, 0.2, 0.1])
            >>> zs = np.array([5.0, 2.0, 4.0])
            >>> ds = grid_lidar_points(
            ...     xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 1.0),
            ...     aggregate="min", epsg=3857,
            ... )
            >>> ds.read_array().tolist()
            [[2.0, 4.0]]

        - Mean aggregation averages every point that lands in a cell:

            >>> import numpy as np
            >>> from digitalrivers.lidar import grid_lidar_points
            >>> xs = np.array([0.1, 0.2])
            >>> ys = np.array([0.1, 0.2])
            >>> zs = np.array([4.0, 6.0])
            >>> ds = grid_lidar_points(
            ...     xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 1.0, 1.0),
            ...     aggregate="mean", epsg=3857,
            ... )
            >>> float(ds.read_array()[0, 0])
            5.0

        - Empty cells receive the dataset's no-data sentinel (-9999.0):

            >>> import numpy as np
            >>> from digitalrivers.lidar import grid_lidar_points
            >>> ds = grid_lidar_points(
            ...     np.array([0.5]), np.array([0.5]), np.array([3.0]),
            ...     cell_size=1.0, bounds=(0.0, 0.0, 2.0, 2.0),
            ...     aggregate="min",
            ... )
            >>> float(ds.no_data_value[0])
            -9999.0
            >>> int((ds.read_array() == -9999.0).sum())
            3
    """
    import numpy as np
    from pyramids.dataset import Dataset

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError(
            f"xs, ys, zs must have the same length; got {len(xs)}, "
            f"{len(ys)}, {len(zs)}"
        )
    valid_aggregates = {
        "min", "max", "mean", "median", "count",
        "idw", "nn", "tin", "rbf",
    }
    if aggregate not in valid_aggregates:
        raise ValueError(
            f"aggregate must be one of {sorted(valid_aggregates)}; "
            f"got {aggregate!r}"
        )
    if bounds is None:
        x_min, y_min = float(xs.min()), float(ys.min())
        x_max, y_max = float(xs.max()), float(ys.max())
    else:
        x_min, y_min, x_max, y_max = bounds

    cols = int(np.ceil((x_max - x_min) / cell_size))
    rows = int(np.ceil((y_max - y_min) / cell_size))

    nodata = -9999.0

    # Spatial-interpolation paths evaluate at cell centres and exit early —
    # they don't use the per-cell binning of the block-aggregation paths.
    if aggregate in ("idw", "nn", "tin", "rbf"):
        # Cell-centre coordinates in the output grid.
        cx = x_min + (np.arange(cols) + 0.5) * cell_size
        cy = y_max - (np.arange(rows) + 0.5) * cell_size
        grid_x, grid_y = np.meshgrid(cx, cy)
        xy_pts = np.column_stack([xs, ys])
        target = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        if aggregate == "idw":
            from scipy.spatial import cKDTree
            tree = cKDTree(xy_pts)
            k = min(int(idw_k), len(xs))
            dists, idxs = tree.query(target, k=k)
            if k == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]
            with np.errstate(divide="ignore"):
                # Exact-hit points pin the result; weight them as infinity
                # so they dominate the weighted average. Replace 1/0 below.
                weights = 1.0 / np.power(dists, idw_power)
            # Cells with an exact hit (distance 0) take that point's z.
            exact = (dists == 0).any(axis=1)
            with np.errstate(invalid="ignore"):
                z_interp = (weights * zs[idxs]).sum(axis=1) / weights.sum(axis=1)
            if exact.any():
                # For exact hits, take the z at the matching nearest point.
                z_interp[exact] = zs[idxs[exact, 0]]
            out = z_interp.reshape(rows, cols)
        elif aggregate == "nn":
            from scipy.spatial import cKDTree
            tree = cKDTree(xy_pts)
            _, idxs = tree.query(target, k=1)
            out = zs[idxs].reshape(rows, cols)
        elif aggregate == "tin":
            from scipy.interpolate import LinearNDInterpolator
            interp = LinearNDInterpolator(xy_pts, zs, fill_value=nodata)
            out = interp(target).reshape(rows, cols)
        else:  # rbf
            from scipy.interpolate import RBFInterpolator
            interp = RBFInterpolator(
                xy_pts, zs, kernel=rbf_kernel, smoothing=rbf_smoothing,
            )
            out = interp(target).reshape(rows, cols)
        out = np.where(np.isfinite(out), out, nodata).astype(
            np.float32, copy=False,
        )
        geo = (x_min, cell_size, 0.0, y_max, 0.0, -cell_size)
        return Dataset.create_from_array(
            out, geo=geo, epsg=epsg, no_data_value=nodata,
        )

    col_idx = np.clip(((xs - x_min) / cell_size).astype(np.int64), 0, cols - 1)
    row_idx = np.clip(
        ((y_max - ys) / cell_size).astype(np.int64), 0, rows - 1
    )

    # `min` / `max` use `np.minimum.at` / `np.maximum.at`;
    # `mean` uses `np.add.at` for an O(N_points) reduction;
    # `count` uses `np.add.at` with weight 1 — same kernel as the mean
    # denominator; `median` still requires per-cell bucketing because
    # there is no closed-form running median in NumPy.
    if aggregate == "min":
        out = np.full((rows, cols), np.inf, dtype=np.float64)
        np.minimum.at(out, (row_idx, col_idx), zs)
        out[~np.isfinite(out)] = nodata
    elif aggregate == "max":
        out = np.full((rows, cols), -np.inf, dtype=np.float64)
        np.maximum.at(out, (row_idx, col_idx), zs)
        out[~np.isfinite(out)] = nodata
    elif aggregate == "mean":
        sums = np.zeros((rows, cols), dtype=np.float64)
        counts = np.zeros((rows, cols), dtype=np.int64)
        np.add.at(sums, (row_idx, col_idx), zs)
        np.add.at(counts, (row_idx, col_idx), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(counts > 0, sums / counts, nodata)
    elif aggregate == "count":
        counts = np.zeros((rows, cols), dtype=np.int64)
        np.add.at(counts, (row_idx, col_idx), 1)
        out = counts.astype(np.float64)
    else:  # median — per-cell bucketing
        buckets: dict[tuple[int, int], list[float]] = {}
        for r, c, z in zip(row_idx, col_idx, zs):
            buckets.setdefault((int(r), int(c)), []).append(float(z))
        out = np.full((rows, cols), nodata, dtype=np.float64)
        for (r, c), vals in buckets.items():
            out[r, c] = float(np.median(np.asarray(vals, dtype=np.float64)))

    geo = (x_min, cell_size, 0.0, y_max, 0.0, -cell_size)
    return Dataset.create_from_array(
        out.astype(np.float32, copy=False),
        geo=geo, epsg=epsg, no_data_value=nodata,
    )


