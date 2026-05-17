# Hydrology pipeline

End-to-end DEM-hydrology workflow. Every stage produces a typed result that flows into the next.

```text
DEM
  → fill_depressions       (Wang-Liu / Planchon-Darboux / Priority-Flood / breach / fill-burn)
  → resolve_flats          (Garbrecht-Martz random-tiebreak)
  → flow_direction         (D8 / D∞ / MFD-Quinn / MFD-Holmgren / Rho8)
  → accumulate             (Kahn-sort over the routing's receivers)
  → streams                (accumulation threshold or area-slope criterion)
  → order                  (Strahler / Shreve / Horton / Hack / Topological)
  → to_vector              (LineString links carrying length, drop, slope, sinuosity)
```

## 1. DEM conditioning

| Method | Reference | What it does |
|--------|-----------|--------------|
| `DEM.fill_depressions(method="priority_flood")` | Barnes 2014 | O(N log N) heap-driven hydrologic fill |
| `DEM.fill_depressions(method="wang_liu")` | Wang & Liu 2006 | Heap-driven flat-fill (no plateau lift) |
| `DEM.fill_depressions(method="planchon_darboux")` | Planchon & Darboux 2002 | Sweep-based iterative fill |
| `DEM.breach_depressions(method="least_cost")` | Lindsay 2016 | Dijkstra breach that finds the lowest cumulative breach path |
| `DEM.breach_depressions(method="hybrid")` | Lindsay 2016 | Breach where possible, fill where not |
| `DEM.breach_depressions(method="single_cell")` | — | One-pass single-cell pit breach |
| `DEM.burn_streams(method="fill_burn")` | Lindsay 2018 (= WBT `FillBurn`) | Burn a vector network into the DEM, then priority-flood |
| `DEM.burn_streams(method="topological_breach")` | Lindsay 2016 | Topology-aware breach + burn |
| `DEM.burn_streams(method="agree")` | Hellweger 1997 | Distance-weighted AGREE burn + buffer |
| `DEM.resolve_flats()` | Garbrecht & Martz 1997 (variant) | Random-tiebreak plateau resolution |
| `DEM.enforce_culverts(roads, streams)` | WBT `BurnStreamsAtRoads` | Drop a cell at every stream-road intersection |
| `DEM.enforce_breaklines(breaklines, lift)` | WBT `RaiseWalls` | Raise cells along linear barriers |
| `DEM.hydroflatten(water_polygons, method)` | — | Force lake / pond surfaces to a single elevation |
| `DEM.burn_buildings(building_polygons, lift)` | — | Lift building footprints above the DEM |
| `DEM.anudem_interpolate(...)` | Hutchinson 1989 | Biharmonic ANUDEM interpolation |
| `DEM.stochastic_depressions(sigma, n_runs)` | Lindsay & Creed 2006 | **W-11** — Monte-Carlo depression occurrence probability |

## 2. Flow direction

| Method | Receivers per cell | Reference |
|--------|--------------------|-----------|
| `"d8"` | 1 | O'Callaghan & Mark 1984 |
| `"rho8"` | 1 (stochastic) | Fairfield & Leymarie 1991 |
| `"dinf"` | up to 2 | Tarboton 1997 |
| `"mfd_quinn"` | up to 8 | Quinn 1991 |
| `"mfd_holmgren"` | up to 8 | Holmgren 1994 |

Every routing emits a typed `FlowDirection` carrying a `routing` tag that propagates through all
downstream stages.

## 3. Flow accumulation

`FlowDirection.accumulate(weights=None)` runs a single Kahn topological-sort kernel that handles all
five routings via the unified `(receivers, proportions)` representation. Returns a typed
`Accumulation` raster.

Build extensions:

* `FlowDirection.upslope_flowpath_length()` — **W-9** — per-cell longest upslope flow path.

## 4. Stream extraction

`Accumulation.streams(threshold, units="cells")` returns a typed `StreamRaster`. With a slope DEM,
the call switches to the Montgomery & Foufoula-Georgiou (1993) area-slope criterion
(`acc * slope**theta >= k`).

## 5. Stream ordering and cleanup

| Method | Reference | What it produces |
|--------|-----------|------------------|
| `order(method="strahler")` | Strahler 1957 | Topology-based order (Kahn BFS) |
| `order(method="shreve")` | Shreve 1966 | Additive magnitude (head count) |
| `order(method="horton")` | Horton 1945 | Strahler with main-stem promotion |
| `order(method="hack")` | Hack 1957 (**W-1**) | Main-stem-first ordering |
| `order(method="topological")` | Kahn 1962 (**W-2**) | Sequential Kahn-sort indices |
| `to_vector(...)` | — | `GeoDataFrame` of links with `length_m`, `drop_m`, `mean_slope`, **`sinuosity` (W-3)** |
| `main_stem(flow_direction, outlet=None)` | **W-4** | Binary mask of the longest source-to-outlet trace |
| `prune_short(flow_direction, min_length_m)` | WBT `RemoveShortStreams` (**W-5**) | Drop headwater links below threshold |

## 6. Watershed delineation

| Method | What it produces |
|--------|------------------|
| `FlowDirection.watershed(points, ...)` | One basin per pour point (D8 / Rho8) |
| `FlowDirection.basins(...)` | Partition into terminal-outlet basins |
| `FlowDirection.subbasins_pfafstetter(accumulation, streams, level=N)` | Pfafstetter coding to level N (1 or 2) |
| `FlowDirection.isobasins(streams, accumulation, target_area_km2)` | **W-7** — equal-area sub-basin partition |
| `StreamRaster.subbasins(flow_direction)` | Per-link sub-basins (head / confluence segments) |

`WatershedRaster.statistics(...)` ships per-basin metrics: area, elevation stats,
hypsometric integral, drainage density, **`longest_flow_path_m` (W-8)**, centroid.

## 7. Hydrologic indices

* `DEM.hand(streams, flow_direction, method="d8")` — Rennó 2008 / Nobre 2011 D8-traced HAND.
* `DEM.hand(streams, method="euclidean")` — **W-10** — Euclidean-nearest-stream variant.
* `DEM.twi(accumulation, slope_deg=None)` — **W-12** — Beven & Kirkby 1979 wetness index.
* `DEM.spi(accumulation, slope_deg=None)` — **W-13** — stream power index.
* `DEM.sti(accumulation, slope_deg=None)` — **W-14** — Moore & Burch 1986 sediment transport index.

## 8. Composite pipeline (W-20)

`DEM.full_hydro_pipeline(fill_method, flow_method, stream_threshold_cells)` chains fill →
flow_direction → accumulate → (optional streams) in one call, returning a dict of typed
results.

## See also

* [DEM reference](../reference/dem.md)
* [FlowDirection reference](../reference/flow_direction.md)
* [Accumulation reference](../reference/accumulation.md)
* [StreamRaster reference](../reference/stream_raster.md)
* [WatershedRaster reference](../reference/watershed_raster.md)
