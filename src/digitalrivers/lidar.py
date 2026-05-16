"""LiDAR point-cloud → DEM gridding.

Working half ships today; the full PDAL pipeline (LAS / LAZ read +
ground classification + filtering + DEM conditioning) remains deferred.

* :func:`grid_lidar_points` — bucket raw `(x, y, z)` arrays into a
  gridded DEM with min / max / mean / median aggregation.
* :func:`pdal_lidar_pipeline` — umbrella for the full PDAL pipeline;
  raises `NotImplementedError` and points at the gridding helper.
"""
from __future__ import annotations


def grid_lidar_points(
    xs,
    ys,
    zs,
    cell_size: float,
    bounds=None,
    aggregate: str = "min",
    epsg: int = 4326,
):
    """Grid a LiDAR point cloud to a DEM.

    Pragmatic LiDAR-to-DEM step that operates on raw `(x, y, z)` arrays
    — useful when the caller has read LAS / LAZ externally (via `laspy`,
    `pylas`, or PDAL) and wants a gridded surface. The full PDAL pipeline
    (read + classify + ground-filter + grid + condition) remains deferred.

    For each cell, aggregates the z values of every point that lands in
    it. `aggregate` controls the aggregation: `"min"` (default — the
    canonical bare-earth choice for first-return LiDAR), `"max"`,
    `"mean"`, or `"median"`. Cells with no points receive the dataset
    no-data sentinel.

    Args:
        xs / ys / zs: 1-D arrays of point coordinates.
        cell_size: output cell side length in map units (must match the
            CRS).
        bounds: `(x_min, y_min, x_max, y_max)` to clip the grid to. If
            `None`, the input points' bounding box is used.
        aggregate: `"min"` (default), `"max"`, `"mean"`, `"median"`.
        epsg: EPSG code of the input coordinates.

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
    if aggregate not in ("min", "max", "mean", "median"):
        raise ValueError(
            f"aggregate must be one of 'min','max','mean','median'; "
            f"got {aggregate!r}"
        )
    if bounds is None:
        x_min, y_min = float(xs.min()), float(ys.min())
        x_max, y_max = float(xs.max()), float(ys.max())
    else:
        x_min, y_min, x_max, y_max = bounds

    cols = int(np.ceil((x_max - x_min) / cell_size))
    rows = int(np.ceil((y_max - y_min) / cell_size))
    col_idx = np.clip(((xs - x_min) / cell_size).astype(np.int64), 0, cols - 1)
    row_idx = np.clip(
        ((y_max - ys) / cell_size).astype(np.int64), 0, rows - 1
    )

    nodata = -9999.0
    # `min` / `max` use `np.minimum.at` / `np.maximum.at`;
    # `mean` uses `np.add.at` for an O(N_points) reduction;
    # `median` still requires per-cell bucketing because there is no
    # closed-form running median in NumPy.
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


def pdal_lidar_pipeline(*args, **kwargs):
    """Full PDAL pipeline — umbrella stub.

    The point-cloud-to-DEM gridding half ships as :func:`grid_lidar_points`
    and works on raw `(x, y, z)` arrays. The full PDAL pipeline (LAS /
    LAZ read + ground classification + filtering + grid + Phase 1-3
    conditioning chain) remains deferred pending the PDAL conda-forge
    dependency.
    """
    raise NotImplementedError(
        "pdal_lidar_pipeline umbrella API deferred. The gridding "
        "half is available via digitalrivers.lidar.grid_lidar_points"
        "; read LAS files externally with laspy / pylas / PDAL and pass "
        "the resulting arrays into grid_lidar_points."
    )
