"""LiDAR point clouds: reading them, classifying them, and gridding them.

The path from a raw cloud to a surface digital-rivers can route over:

1. :mod:`io` — read the LAS / LAZ file into a `LasPoints`.
2. :mod:`points` — clip, merge or filter it down to the returns you want.
3. :mod:`classify` — separate ground from vegetation and structures, or find tree tops.
4. :mod:`gridding` — bin the ground returns into a raster, which is where `DEM` takes
   over.

`laspy` is an optional dependency and is imported only inside :mod:`io`, so the array
half of this package works on a cloud that came from anywhere.
"""

from digitalrivers.lidar.classify import classify_ground, detect_trees
from digitalrivers.lidar.gridding import grid_lidar_points
from digitalrivers.lidar.io import read_las, write_las
from digitalrivers.lidar.points import LasPoints, clip, filter_classes, merge

__all__ = [
    "LasPoints",
    "classify_ground",
    "clip",
    "detect_trees",
    "filter_classes",
    "grid_lidar_points",
    "merge",
    "read_las",
    "write_las",
]
