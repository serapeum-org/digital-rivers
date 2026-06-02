"""digital-rivers — GIS utility package for DEM processing,
terrain analysis, and hydrologic modelling.

Public surface exposed at the package root:

* `DEM` — digital-elevation-model processor (filling, breaching, flow
  direction, accumulation, slope, exporters, …).
* `Terrain` — colour-relief raster generation.
* `FlowDirection` / `Accumulation` / `StreamRaster` /
  `WatershedRaster` — typed result classes with routing-scheme
  provenance tagging.
* `Mesh` — triangle-mesh container with Laplacian smoothing and
  aspect-ratio quality metrics (Phase 4 P33).
"""

# Import pyramids FIRST — before anything else. pyramids-gis >=0.20.0
# bundles GDAL's osgeo SWIG bindings + native libs inside its platform
# wheel and activates them (puts `_vendor/osgeo` on sys.path, sets
# GDAL_DATA / PROJ_DATA, registers the Windows DLL dir) on import. Every
# digitalrivers submodule does `from osgeo import gdal` at module load,
# so this import must run before them or those imports fail when no
# separately installed GDAL is present.
import pyramids  # noqa: F401

try:
    from importlib.metadata import PackageNotFoundError  # type: ignore
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError  # type: ignore
    from importlib_metadata import version


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


from digitalrivers.accumulation import Accumulation
from digitalrivers.dem import DEM
from digitalrivers.flow_direction import FlowDirection
from digitalrivers.mesh import Mesh
from digitalrivers.stream_raster import StreamRaster
from digitalrivers.terrain import Terrain
from digitalrivers.watershed_raster import WatershedRaster

__all__ = [
    "Accumulation",
    "DEM",
    "FlowDirection",
    "Mesh",
    "StreamRaster",
    "Terrain",
    "WatershedRaster",
]
