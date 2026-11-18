"""The watershed domain: which cells drain to which outlet.

`WatershedRaster` is a labelled basin raster that remembers the routing scheme and the
outlets it was delineated from, so a basin map cannot be silently re-read against a
different flow grid than the one that produced it. It carries the per-basin statistics
and the polygonisation that turn labels into something a report can use.

`_kernels.watershed` holds the reverse-BFS labelling walk. It lived under the old
`_flow` package, which was a filing accident: its only consumer builds a
`WatershedRaster` out of it, so it belongs here.
"""

from digitalrivers.watershed.raster import WatershedRaster

__all__ = ["WatershedRaster"]
