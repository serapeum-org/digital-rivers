# digital-rivers

A GIS utility library for **DEM-hydrology**, **terrain analysis**, and **LiDAR processing**, built
on GDAL and the [`pyramids`](https://github.com/serapeum-org/pyramids) raster wrapper.

## What it does

**Hydrology** — full DEM-to-streams pipeline:

- **Depression handling** — Wang-Liu / Planchon-Darboux / Priority-Flood fill, Lindsay 2016 breach
  (least-cost / hybrid / single-cell), fill-burn (Lindsay 2018), AGREE, topological breach.
- **Flow routing** — D8, D∞, MFD-Quinn, MFD-Holmgren, Rho8.
- **Flow accumulation** — single Kahn-sort kernel for all five routings.
- **Stream extraction** — accumulation threshold or area-slope criterion (Montgomery &
  Foufoula-Georgiou 1993).
- **Stream ordering** — Strahler, Shreve, Horton, **Hack**, **Topological**.
- **Watersheds** — pour-point delineation, terminal-outlet basins, link-based sub-basins,
  multi-level Pfafstetter coding, **equal-area Isobasin** partitioning.
- **Hydro indices** — HAND (D8 + Euclidean variants), **TWI**, **SPI**, **STI**.

**Terrain analysis** — single-band raster attribute family:

- **Slope / aspect / hillshade / color-relief** (via GDAL `DEMProcessing` on `Terrain`, plus
  `DEM.slope`).
- **Focal-window indices** — **TPI**, **deviation-from-mean elevation**, **elevation SD**,
  **ruggedness index** (Riley et al. 1999).
- **Surface geometry** — **Zevenbergen-Thorne curvature family** (plan / profile / total / mean /
  gaussian), **normal-vector angular deviation**.
- **Visibility** — **topographic openness** (Yokoyama 2002), **sky-view factor** (Zakšek et al.
  2011) — shared horizon-walk kernel.

**LiDAR** — LAS/LAZ I/O through gridded DEMs:

- **`LasPoints`** record + **`read_las`** / **`write_las`** via `laspy` (soft-imported).
- **Ground filter** — Zhang 2003 morphological tophat.
- **Gridding** — block aggregation (`min` / `max` / `mean` / `median` / `count`) plus interpolation
  (`idw` / `nn` / `tin` / `rbf`).
- **Vector ops** — `clip`, `merge`, `filter_classes`.
- **Forestry** — `detect_trees` (variable-window local-maxima on a CHM).

**Composite / utilities**:

- **`DEM.full_hydro_pipeline(...)`** — one-call fill → flow_direction → accumulate (+ optional
  streams).
- **`topobathy_fusion`** — blend a topographic DEM with a bathymetric DEM (4 modes).
- **`Mesh`** — triangle-mesh container with Laplacian smoothing and aspect-ratio QC.
- **`cloud_io`** — `tile_windows` iterator and Cloud-Optimized GeoTIFF (COG) writer.

Every typed class subclasses `pyramids.dataset.Dataset`, so all pyramids methods (`crop`, `to_crs`,
`plot`, `stats`, …) are available alongside the digital-rivers additions.

## Quick start

```python
from osgeo import gdal
from digitalrivers import DEM, Terrain

dem = DEM(gdal.Open("dem.tif"))

# One-call hydro pipeline.
out = dem.full_hydro_pipeline(stream_threshold_cells=500)
filled = out["filled_dem"]
fdir = out["flow_direction"]
accumulation = out["accumulation"]
streams = out["streams"]

# Stream-network analysis.
ordered = streams.order(method="hack", flow_direction=fdir)
links = streams.to_vector(fdir, dem=filled)  # GeoDataFrame with sinuosity, length_m, drop_m
main_stem = streams.main_stem(fdir)           # binary mask along the longest path

# Watershed metrics.
basins = fdir.basins()
metrics = basins.statistics(dem=filled, flow_direction=fdir, streams=streams)
# columns: area_km2, mean_elev, hypsometric_integral, drainage_density_km_per_km2,
#          longest_flow_path_m, centroid_x, centroid_y

# Terrain indices.
tpi = dem.tpi(window=5)
curvature = dem.curvature(kind="profile")
svf = dem.sky_view_factor(search_radius=10)

# Visualisation.
terrain = Terrain("dem.tif")
hill_shade = terrain.hill_shade()
```

## Where to next

- [Installation](installation.md) — Pixi/conda or pip-from-source instructions.

**Guides** (concept overviews):

- [Hydrology pipeline](guides/hydrology-pipeline.md) — fill → flow → accumulate → streams →
  ordering → HAND.
- [Watershed analytics](guides/watershed-analytics.md) — the five partitioning APIs and per-basin
  metrics.
- [Terrain indices](guides/terrain-indices.md) — TPI / curvature / openness / sky-view factor.
- [LiDAR pipeline](guides/lidar.md) — LAS I/O → ground classification → gridding → tree detection.

**Reference** (per-module API):

- [DEM](reference/dem.md), [FlowDirection](reference/flow_direction.md),
  [Accumulation](reference/accumulation.md), [StreamRaster](reference/stream_raster.md),
  [WatershedRaster](reference/watershed_raster.md).
- [Terrain](reference/terrain.md), [LiDAR](reference/lidar.md), [Mesh](reference/mesh.md),
  [Cloud I/O](reference/cloud_io.md), [Topobathy fusion](reference/fusion.md).

**Other**:

- [Change log](change-log.md)
- [Architecture Decision Records](adr/index.md)

## Status

Pre-release (v0.1.0). Not yet on PyPI or conda-forge — install from source. Supported Python:
**3.11–3.13**.

Source: <https://github.com/serapeum-org/digital-rivers>
