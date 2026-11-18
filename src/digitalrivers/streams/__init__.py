"""The stream domain: the channel network extracted from a flow grid.

`StreamRaster` is a boolean / labelled channel mask that remembers the routing scheme
and the accumulation threshold it was cut at — provenance a bare mask cannot carry, and
which every downstream ordering or vectorisation step needs to be correct.

The array walks live in the private `_kernels` subpackage:

* :mod:`_kernels.order` — Strahler / Shreve / Horton / Hack ordering and the link
  topology they share.
* :mod:`_kernels.hand` — the Height-Above-Nearest-Drainage walk (Renno 2008 /
  Nobre 2011), which is stream-relative and so belongs here rather than with the DEM
  that supplies its elevations.
"""

from digitalrivers.streams.raster import StreamRaster

__all__ = ["StreamRaster"]
