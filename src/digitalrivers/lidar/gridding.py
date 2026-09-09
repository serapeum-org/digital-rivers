"""Turning a point cloud into a raster.

`grid_lidar_points` bins returns into cells and reduces each cell to one value. Which
reduction to pick is the whole decision: `min` gives a bare-earth surface from an
unclassified cloud, `max` gives canopy top, `mean` and `median` smooth noisy returns, and
`count` gives point density, which is how you find where the cloud is too sparse to trust
the other four.
"""

from __future__ import annotations

import numpy as np
from pyramids.dataset import Dataset, GeoReference

__all__ = ["grid_lidar_points"]


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
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError(
            f"xs, ys, zs must have the same length; got {len(xs)}, "
            f"{len(ys)}, {len(zs)}"
        )
    valid_aggregates = {
        "min",
        "max",
        "mean",
        "median",
        "count",
        "idw",
        "nn",
        "tin",
        "rbf",
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
                xy_pts,
                zs,
                kernel=rbf_kernel,
                smoothing=rbf_smoothing,
            )
            out = interp(target).reshape(rows, cols)
        out = np.where(np.isfinite(out), out, nodata).astype(
            np.float32,
            copy=False,
        )
        geo = (x_min, cell_size, 0.0, y_max, 0.0, -cell_size)
        return Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=geo, epsg=epsg),
            no_data_value=nodata,
        )

    col_idx = np.clip(((xs - x_min) / cell_size).astype(np.int64), 0, cols - 1)
    row_idx = np.clip(((y_max - ys) / cell_size).astype(np.int64), 0, rows - 1)

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
    return Dataset.from_array(
        out.astype(np.float32, copy=False),
        geo_ref=GeoReference(geo=geo, epsg=epsg),
        no_data_value=nodata,
    )
