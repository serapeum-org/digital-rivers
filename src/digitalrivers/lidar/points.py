"""The `LasPoints` container and the operations that reshape a point cloud.

A LiDAR return is more than an `(x, y, z)` triple: it carries a classification code, an
intensity, and a return number, and the operations here have to keep those aligned with
the coordinates or the cloud silently corrupts. `LasPoints` holds them as parallel arrays
and `subset` is the one place indexing happens, so every filter, clip and merge goes
through a single implementation rather than re-deriving the alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely

__all__ = ["LasPoints", "clip", "merge", "filter_classes"]


_LASPY_HINT = (
    "laspy is required for LAS / LAZ I/O. Install with " "`pip install laspy[lazrs]`."
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
            raise ValueError(f"mask shape {mask.shape} != points shape {self.x.shape}")
        kw = {"x": self.x[mask], "y": self.y[mask], "z": self.z[mask]}
        if self.intensity.size:
            kw["intensity"] = self.intensity[mask]
        if self.classification.size:
            kw["classification"] = self.classification[mask]
        if self.return_number.size:
            kw["return_number"] = self.return_number[mask]
        return LasPoints(crs=self.crs, **kw)


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
        kw["classification"] = np.concatenate([p.classification for p in pointclouds])
    if all(p.return_number.size for p in pointclouds):
        kw["return_number"] = np.concatenate([p.return_number for p in pointclouds])
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
        raise ValueError("points carries no classification data; nothing to filter")
    keep = np.isin(points.classification, list(classes))
    return points.subset(keep)
