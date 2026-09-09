"""The elevation domain: the `DEM` class and the conditioning kernels behind it.

`DEM` is the entry point for everything that treats a raster as a surface — depression
filling and breaching, flat resolution, stream / building / breakline burning, the
morphometric indices, D8 routing, and the hydrodynamic-model exporters.

The array-level algorithms it delegates to live in the private `_kernels` subpackage:
Priority-Flood / Wang-Liu / Planchon-Darboux fill, Lindsay breaching, and Garbrecht &
Martz flat resolution. They take and return plain arrays, so they are testable without a
`Dataset`; callers outside this package should go through the `DEM` methods.

`DEM` and `DIR_OFFSETS` are re-exported here so `from digitalrivers.dem import DEM` keeps
working now that this module is a package rather than a single file.
"""

from digitalrivers.core.directions import DIR_OFFSETS
from digitalrivers.dem.conditioning import _reproject_if_needed
from digitalrivers.dem.dem import DEM

__all__ = ["DEM", "DIR_OFFSETS", "_reproject_if_needed"]
