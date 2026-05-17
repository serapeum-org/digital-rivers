# WatershedRaster

Typed watershed / basin / sub-basin raster — produced by `FlowDirection.watershed()`,
`FlowDirection.basins()`, `FlowDirection.subbasins_pfafstetter()`,
`FlowDirection.isobasins()`, and `StreamRaster.subbasins()`.

Top-level surface:

* **`basin_count`** (lazy property) — number of distinct non-zero basin labels.
* **`statistics(dem=None, slope=None, streams=None, flow_direction=None, accumulation=None,
  metrics=None)`** — per-basin descriptor table. Available columns:
    * `area_km2`, `centroid_x`, `centroid_y` (always).
    * `min_elev`, `max_elev`, `mean_elev`, `std_elev`, `hypsometric_integral` (with `dem`).
    * `mean_slope` (with `slope`).
    * `drainage_density_km_per_km2` (with `streams`, optionally `flow_direction` for diagonal
      length-weighting).
    * `longest_flow_path_m` (with `flow_direction`; W-8 — `accumulation` is no-op post-M1).
* **`to_polygons()`** — vectorise the labelled raster to per-basin polygons.

::: digitalrivers.watershed_raster.WatershedRaster
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
