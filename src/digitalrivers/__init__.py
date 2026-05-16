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

# documentation format
__author__ = "Mostafa Farrag"
__email__ = "moah.farag@gmail.com"
__docformat__ = "restructuredtext"

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ()  # ("numpy", "pandas", "gdal")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)
        print(e)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(missing_dependencies))

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

__doc__ = """digital-rivers — GIS utility package for DEM processing,
terrain analysis, and hydrologic modelling.

Public surface exposed at the package root:

* ``DEM`` — digital-elevation-model processor (filling, breaching, flow
  direction, accumulation, slope, exporters, …).
* ``Terrain`` — colour-relief raster generation.
* ``FlowDirection`` / ``Accumulation`` / ``StreamRaster`` /
  ``WatershedRaster`` — typed result classes with routing-scheme
  provenance tagging.
* ``Mesh`` — triangle-mesh container with Laplacian smoothing and
  aspect-ratio quality metrics (Phase 4 P33).
"""
