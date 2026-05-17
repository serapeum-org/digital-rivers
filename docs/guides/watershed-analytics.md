# Watershed analytics

digital-rivers ships five distinct watershed-related partitioning APIs, each producing a typed
`WatershedRaster`. Pick the one that matches the question you're asking.

## 1. Pour-point watershed — `FlowDirection.watershed(points, ...)`

Reverse-BFS from each pour point under D8 / Rho8 routing. Labels every contributing cell with the
basin ID of its downstream seed.

* `require_unique_basins=False` (default) — later seeds overwrite earlier ones along shared
  upstream paths. The mathematically-equivalent reverse-order pass keeps the kernel O(N) instead
  of O(B·N) when basins overlap heavily.
* `require_unique_basins=True` — first seed to claim a cell wins; later seeds get whatever's left.

## 2. Terminal-outlet basins — `FlowDirection.basins(...)`

Partitions the entire DEM into basins, one per terminal outlet (cell whose flow direction is the
no-data sentinel — either the data envelope or an unfilled internal sink).

`min_area_cells` / `min_area_km2` + `merge_small="drop"` / `"merge"` post-processes small basins.

## 3. Hierarchical Pfafstetter coding — `FlowDirection.subbasins_pfafstetter(accumulation, streams, level=N)`

Pfafstetter coding to level N. Each cell carries an N-digit integer:

* Codes 1, 3, 5, 7, 9 = inter-basins along the main stem.
* Codes 2, 4, 6, 8 = the four largest tributaries by accumulation, ordered by along-stem position
  (so code 2 is the downstream-most tributary).
* Recursion: level 2 applies the same scheme inside each level-1 sub-basin.

Output is a uint64 raster carrying the digit-encoded code.

## 4. Equal-area sub-basins — `FlowDirection.isobasins(streams, accumulation, target_area_km2)` (W-7)

Partition the catchment into sub-basins of approximately equal area. Walks the stream network from
heads to outlet via the accumulation raster; at every cell whose floor-divided accumulation
quantile (`acc // target_cells`) is strictly greater than its stream-upstream max, a virtual
sub-basin outlet is placed. Final sub-basin labels come from a first-claim-wins reverse-BFS
watershed delineation on those seeds.

Used in distributed-hydro modelling (HEC-HMS, SWAT) where each sub-basin must be ≤ a maximum
modelling unit.

## 5. Per-link sub-basins — `StreamRaster.subbasins(flow_direction, method="link")`

Partition each cell by the first downstream stream link it joins. Confluence cells belong to the
new downstream link (WBT / TauDEM convention). Off-stream cells inherit the link ID of the first
stream cell their flow path reaches.

## Per-basin metrics — `WatershedRaster.statistics(...)`

| Column | Trigger | What it measures |
|--------|---------|------------------|
| `area_km2`, `centroid_x`, `centroid_y` | always | Basin area and centroid in dataset CRS |
| `min_elev`, `max_elev`, `mean_elev`, `std_elev`, `hypsometric_integral` | `dem=` | Elevation stats (Strahler 1952 hypsometric integral) |
| `mean_slope` | `slope=` | Mean of the slope raster across the basin |
| `drainage_density_km_per_km2` | `streams=` (optionally `flow_direction=`) | Stream-length / area; diagonal-aware when `flow_direction` is supplied |
| `longest_flow_path_m` | `flow_direction=` (**W-8**) | Longest upstream-to-outlet flow path; uses the shared `kahn_max_upslope_length` Kahn pass |

## `WatershedRaster.to_polygons()`

Vectorise the labelled raster to per-basin Shapely polygons (or MultiPolygons for disconnected
basins). Returns a `GeoDataFrame` carrying `basin_id` plus the polygon geometry.

## See also

* [FlowDirection reference](../reference/flow_direction.md) — pour-point watershed, basins,
  Pfafstetter, isobasins APIs.
* [StreamRaster reference](../reference/stream_raster.md) — per-link sub-basins.
* [WatershedRaster reference](../reference/watershed_raster.md) — full statistics + polygons API.
