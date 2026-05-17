# Change log

## Unreleased (`feat/phase-4`)

**Hydrology — new stream-network surface (W-1 → W-5)**

* `StreamRaster.order(method="hack")` — Hack 1957 main-stem-first ordering.
* `StreamRaster.order(method="topological")` — Kahn-sort sequential numbering.
* `StreamRaster.to_vector(...)` now carries a `sinuosity` column per link.
* `StreamRaster.main_stem(flow_direction, outlet=None)` — public API for the longest source-to-outlet
  trace.
* `StreamRaster.prune_short(flow_direction, min_length_m)` — drop headwater links below a threshold.

**Hydrology — watershed metrics (W-7 → W-9)**

* `FlowDirection.isobasins(streams, accumulation, target_area_km2)` — equal-area sub-basin
  partition.
* `WatershedRaster.statistics(...)` gained `longest_flow_path_m` (via the shared
  `kahn_max_upslope_length` Kahn pass). Triggers on `flow_direction` alone after the M1 review fix.
* `FlowDirection.upslope_flowpath_length()` — per-cell longest upslope flow path.

**Hydrology — indices and Monte-Carlo (W-10 → W-14, W-20)**

* `DEM.hand(..., method="euclidean")` — nearest-stream-by-2D-distance HAND variant; drops stream
  cells on no-data DEM positions with a `UserWarning` (M2 / L2 review fixes).
* `DEM.stochastic_depressions(sigma, n_runs, seed)` — Monte-Carlo per-cell depression-occurrence
  probability (Lindsay & Creed 2006).
* `DEM.twi(accumulation, slope_deg=None)` — Beven & Kirkby 1979 wetness index.
* `DEM.spi(accumulation, slope_deg=None)` — stream power index.
* `DEM.sti(accumulation, slope_deg=None)` — Moore & Burch 1986 sediment transport index.
* `DEM.full_hydro_pipeline(...)` — composite fill → flow_direction → accumulate (+ optional
  streams) returning a typed-results dict.

**LiDAR (W-15 → W-19)**

* `LasPoints` dataclass + `read_las` / `write_las` via soft-imported `laspy`.
* `classify_ground(method="zhang")` — Zhang 2003 morphological tophat ground filter.
* `grid_lidar_points(aggregate=...)` extended with `idw` / `nn` / `tin` (linear barycentric) /
  `rbf` interpolation plus a `count` density aggregator.
* `clip` / `merge` / `filter_classes` — vector clip + concatenation + ASPRS-class subset on
  `LasPoints`.
* `detect_trees(chm, min_height_m, radius_fn)` — variable-window local-maxima tree detection on a
  canopy height model.

**Terrain attributes (W-21 → W-28)**

* `DEM.tpi(window)` — Guisan 1999 Topographic Position Index.
* `DEM.deviation_from_mean(window)` — standardised TPI.
* `DEM.elev_std(window)` — focal-window standard deviation of elevation.
* `DEM.ruggedness(window)` — Riley et al. 1999 terrain ruggedness index.
* `DEM.curvature(kind="plan"/"profile"/"total"/"mean"/"gaussian")` — Zevenbergen-Thorne 1987
  curvature family.
* `DEM.normal_vector_deviation(window)` — surface-normal angular deviation roughness metric.
* `DEM.openness(search_radius, kind="positive"/"negative")` — Yokoyama 2002 topographic openness.
* `DEM.sky_view_factor(search_radius)` — Zakšek et al. 2011 sky-view factor.

Shared private helpers: `_focal_window_stats` (no-data-aware focal mean / SD) and
`_numba.horizon_walk_kernel` (numba-accelerated 8-azimuth horizon scan).

**Phase-4 backfill (P29 – P34)**

* `cloud_io.tile_windows` / `cloud_io.write_cog` — chunked-tile iteration and Cloud-Optimized
  GeoTIFF writes.
* `topobathy_fusion(method=...)` — 4-mode topobathy blend (`max` / `min` / `topo_above` /
  `bathy_below`).
* `Mesh` — triangle-mesh container with Laplacian smoothing and aspect-ratio quality metrics.
* `grid_lidar_points` (foundation that W-17 extended).
* `DEM.anudem_interpolate` — Hutchinson 1989 biharmonic ANUDEM.
* `FlowDirection.subbasins_pfafstetter(max_level=N)` — multi-level Pfafstetter coding.

**Dependencies**

* `cleopatra` removed (now sourced transitively through `pyramids-gis[viz]`).
* `pyramids-gis` bumped from `0.9.1` → `0.18.0`.
* `gdal >=3.10` and `numba >=0.60,<1` added as hard deps to keep PyPI and pixi installs in sync.

**Documentation**

* New guides: hydrology pipeline, watershed analytics, terrain indices, LiDAR pipeline.
* New reference pages: `FlowDirection`, `Accumulation`, `StreamRaster`, `WatershedRaster`,
  `LiDAR`, `Mesh`, `Cloud I/O`, `Topobathy fusion`.

## 0.1.0 (2025-09-**)

* First release on PyPI.
