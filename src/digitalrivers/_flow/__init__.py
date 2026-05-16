"""Flow-routing, accumulation, watershed-BFS, and IHU-upscaling kernels.

Four internal modules:

* :mod:`routing` — D8, D∞, MFD-Quinn, MFD-Holmgren, Rho8 direction kernels
  and the shared ``DIR_OFFSETS`` / ``_DIR_DR`` / ``_DIR_DC`` tables.
* :mod:`accumulation` — Kahn topological-sort accumulation that dispatches
  to all five routing schemes through a single ``(receivers, proportions,
  weights, valid_mask)`` representation.
* :mod:`watershed` — reverse-BFS watershed labelling (one basin per pour
  point or per terminal outlet).
* :mod:`ihu` — Eilander 2021 IHU hill-climbing engine on top of a COTAT
  seed network.

These kernels back the typed-result classes :class:`FlowDirection`,
:class:`Accumulation`, and :class:`WatershedRaster`. They are
package-private; downstream callers should use those typed-class methods
rather than importing the kernels directly.
"""
