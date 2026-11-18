"""Interop: handing the package's results to something outside it.

Everything here crosses the boundary out of digital-rivers, which is why it sits apart
from the domain packages rather than inside them:

* :mod:`export` — six writers for hydrodynamic models (LISFLOOD-FP, HEC-RAS, TUFLOW,
  SFINCS, Gmsh, Iber). Each takes an `ExportGrid` and returns the paths it wrote.
* :mod:`subgrid` — per-coarse-cell bathymetry tables for reduced-order 2D solvers.
* :mod:`anudem` — relaxation gap-fill for surfaces with holes.
* :mod:`mesh` — the `Mesh` container behind the triangle-mesh export path.
* :mod:`cloud_io` — COG writing and tiled window iteration. Transitional: this is
  generic raster I/O that belongs upstream in `pyramids`, and is tracked there as U1 and
  U2 in `planning/pyramids/pyramids-feat-generic-gis-extraction.md`. It lives here as a
  parking spot, not as an endorsement — do not build on it.
"""

from digitalrivers.interop.mesh import Mesh

__all__ = ["Mesh"]
