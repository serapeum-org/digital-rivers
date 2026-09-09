"""Array-level watershed kernels.

:mod:`watershed` walks upstream from each pour point under D8 / Rho8 routing, labelling
every contributing cell with the basin id of its downstream seed. Package-private;
callers should use `FlowDirection.watershed` / `.basins`.
"""
