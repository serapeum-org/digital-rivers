"""GDAL-backed terrain rendering and derivatives.

`Terrain` wraps GDAL's `DEMProcessing` — colour relief, hillshade, slope, aspect,
roughness, TPI, TRI and viewshed. It is the visual and cartographic half of terrain
analysis, as distinct from the hydrological half in `digitalrivers.dem`.

Where the two overlap they are different calculations, not duplicates: `Terrain.slope`
is GDAL's Horn estimator over the full 3x3 window, `DEM.slope` is the maximum D8 descent
in numpy; `Terrain.tri` and `DEM.ruggedness` likewise differ in their neighbourhood.
Pick by which definition the downstream consumer expects.

This surface is a standing candidate to move upstream into `pyramids` — colour-ramp
rendering and hillshade are generic raster visualisation with nothing hydrological about
them. `Terrain.color_relief` is tracked there as U10 in
`planning/pyramids/pyramids-feat-generic-gis-extraction.md`. Keeping the class in its own
package means that move is a directory delete rather than a surgery.
"""

from digitalrivers.terrain.terrain import CREATION_OPTIONS, Terrain

__all__ = ["CREATION_OPTIONS", "Terrain"]
