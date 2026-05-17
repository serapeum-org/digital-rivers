# FlowDirection

Typed flow-direction raster — produced by `DEM.flow_direction(method=...)` for any of the five routing
schemes (`d8` / `dinf` / `mfd_quinn` / `mfd_holmgren` / `rho8`). Carries a `routing` tag persisted into
the raster's metadata (`DR_ROUTING`) so downstream consumers stay aware of which scheme produced the
grid.

Top-level surface:

* **`accumulate(weights=...)`** — Kahn topological-sort flow accumulation (single kernel for all five
  routings).
* **`watershed(points, ...)`** — pour-point watershed delineation under D8 / Rho8.
* **`basins(...)`** — partition the DEM into one basin per terminal outlet.
* **`subbasins_pfafstetter(max_level=N)`** — hierarchical Pfafstetter coding (level 1..N).
* **`isobasins(streams, accumulation, target_area_km2)`** — equal-area sub-basin partition (W-7).
* **`upslope_flowpath_length()`** — per-cell longest upslope flow path (W-9).
* **`upscale(method=...)` / `upscale_ihu(...)`** — COTAT / EAM / DMM / IHU (Eilander 2021) upscalers.

::: digitalrivers.flow_direction.FlowDirection
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
