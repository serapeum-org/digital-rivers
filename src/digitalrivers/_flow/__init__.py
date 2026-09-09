"""Reverse-BFS watershed labelling.

What is left of the old `_flow` package: the routing, accumulation and IHU kernels
have moved to :mod:`digitalrivers.flow._kernels`, next to the classes they back.
:mod:`watershed` follows to :mod:`digitalrivers.watershed._kernels` in the next step —
its only consumer builds a `WatershedRaster` out of it.
"""
