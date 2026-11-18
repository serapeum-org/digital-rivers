"""Array-level stream kernels: ordering and height-above-nearest-drainage.

* :mod:`order` — Strahler, Shreve, Horton, Hack and topological ordering, over a shared
  link-topology builder.
* :mod:`hand` — the HAND walk under D8 / Rho8 routing, memoised so each cell is visited
  at most twice.

Package-private. Downstream callers should use `StreamRaster.order` and `DEM.hand`.
"""
