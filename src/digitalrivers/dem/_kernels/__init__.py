"""DEM-conditioning kernels: depression fill, breach, and flat resolution.

Three internal modules:

* :mod:`pitremoval` — Priority-Flood / Wang-Liu / Planchon-Darboux fill plus
  the shared `local_minima_8` detector.
* :mod:`breach` — Lindsay 2016 single-cell / least-cost / hybrid breach.
* :mod:`flats` — Garbrecht & Martz 1997 flat-area gradient resolution.

These kernels back :meth:`DEM.fill_depressions`, :meth:`DEM.breach_depressions`,
and :meth:`DEM.resolve_flats`. They take and return plain arrays, so they are
testable without a `Dataset`, but they are package-private: downstream callers
should use the `DEM.*` methods rather than importing the kernels directly.
"""
