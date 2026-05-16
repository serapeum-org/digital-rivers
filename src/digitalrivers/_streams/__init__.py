"""Stream-derived metric kernels: HAND and stream ordering.

Two internal modules:

* :mod:`hand` — Renno 2008 / Nobre 2011 Height-Above-Nearest-Drainage
  walk under D8 / Rho8 routing.
* :mod:`order` — Strahler / Shreve / Horton stream-ordering kernels.

These kernels back :meth:`DEM.hand` and :meth:`StreamRaster.order`. They
are package-private; downstream callers should use the typed-class
methods rather than importing the kernels directly.
"""
