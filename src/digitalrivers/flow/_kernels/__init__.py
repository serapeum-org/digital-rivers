"""Array-level flow-routing kernels behind `FlowDirection` and `Accumulation`.

Three modules, all taking and returning plain arrays so they are testable without a
`Dataset`:

* :mod:`routing` — the D∞, MFD-Quinn, MFD-Holmgren and Rho8 direction kernels, and the
  `int8` neighbour tables they broadcast against.
* :mod:`accumulation` — Kahn topological-sort accumulation, dispatching all five routing
  schemes through one `(receivers, proportions, weights, valid_mask)` representation.
* :mod:`ihu` — the Eilander 2021 IHU hill-climbing engine over a COTAT seed network.

Package-private. Downstream callers should use the `FlowDirection` / `Accumulation`
methods rather than importing these directly.
"""
