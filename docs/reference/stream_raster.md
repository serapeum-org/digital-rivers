# StreamRaster

Typed stream-cell raster — produced by `Accumulation.streams(...)`. Carries the threshold and routing
tags of its source pipeline so consumers stay aware of how the network was extracted.

Top-level surface:

* **`order(method=...)`** — Strahler / Shreve / Horton / Hack (W-1) / Topological (W-2) ordering.
* **`to_vector(flow_direction, dem=None)`** — vectorise the network into a `GeoDataFrame` of
  LineString links with `link_id`, `from_node`, `to_node`, `length_m`, `drop_m`, `mean_slope`, and
  `sinuosity` (W-3) columns.
* **`main_stem(flow_direction, outlet=None)`** — binary mask of cells on the longest source-to-outlet
  path (W-4).
* **`prune_short(flow_direction, min_length_m)`** — drop headwater links below the threshold (W-5).
* **`subbasins(flow_direction, method="link")`** — partition the basin into one sub-basin per stream
  link.

::: digitalrivers.stream_raster.StreamRaster
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
