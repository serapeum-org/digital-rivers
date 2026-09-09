"""Reading and writing LAS / LAZ files.

`laspy` is an optional dependency: the rest of the LiDAR surface works on arrays that
came from anywhere, and only these two functions need a LAS reader. It is imported
inside the functions so `import digitalrivers` does not require it, and both raise a
message naming the extra to install rather than an ImportError from three frames down.
"""

from __future__ import annotations

import warnings

import numpy as np

from digitalrivers.lidar.points import LasPoints

__all__ = ["read_las", "write_las"]


from digitalrivers.lidar.points import _LASPY_HINT


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
