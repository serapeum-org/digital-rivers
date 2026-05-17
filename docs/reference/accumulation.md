# Accumulation

Typed flow-accumulation raster — produced by `FlowDirection.accumulate()`. Carries the source's
`routing` tag so callers know which routing scheme drove the kahn-sort pass. Output semantics:
`accumulation[cell] = sum of weights over upstream cells` (the cell's own weight is not included),
matching the long-standing `DEM.flow_accumulation` behaviour with `weights=1`.

Top-level surface:

* **`streams(threshold, units="cells", slope_dem=None, area_slope_exponent=None, envelope=None)`** —
  extract a `StreamRaster` via accumulation threshold or the Montgomery & Foufoula-Georgiou
  area-slope criterion.
* **`snap_pour_points(points, radius_cells=..., method="max_accumulation"/"jenson")`** — Jenson &
  Domingue 1988 or ArcGIS-style max-accumulation snap of pour-point geometries.

::: digitalrivers.accumulation.Accumulation
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
