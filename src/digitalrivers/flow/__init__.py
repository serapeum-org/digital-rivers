"""The flow domain: where water goes, and how much of it arrives.

Two typed rasters and the kernels behind them:

* `FlowDirection` — a direction-code raster that knows its own routing scheme (`d8`,
  `dinf`, `mfd_quinn`, `mfd_holmgren`, `rho8`) and cell-value encoding. The routing
  argument is required at construction, which is the safety property: a raster of
  unknown provenance cannot be silently reinterpreted as D8 downstream.
* `Accumulation` — an upstream-area raster carrying the routing scheme that produced it.

`FlowDirection` is assembled from mixins that keep one concern each: :mod:`upscale`
(COTAT / EAM / DMM / IHU coarsening) and :mod:`pfafstetter` (nested sub-basin coding).
The array algorithms live in the private :mod:`_kernels` subpackage.
"""

from digitalrivers.flow.accumulation import Accumulation
from digitalrivers.flow.direction import FlowDirection

__all__ = ["Accumulation", "FlowDirection"]
