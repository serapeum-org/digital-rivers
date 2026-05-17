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
    try:
        import laspy  # type: ignore
    except ImportError as exc:  # pragma: no cover — environment-specific
        raise ImportError(_LASPY_HINT) from exc
    f = laspy.read(path)
    try:
        crs = f.header.parse_crs()
    except Exception:  # pragma: no cover — older LAS headers lack a CRS
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


