# ADR-0003: I/O format choices

- Status: Accepted
- Date: 2025-08-16

## Context

`digital-rivers` operates exclusively on raster Digital Elevation Models. All raster I/O is delegated to GDAL via the `pyramids.dataset.Dataset` base class.

## Decision

- **Raster input**: any GDAL-readable single-band elevation raster. GeoTIFF is the primary supported format.
- **Raster output**: GeoTIFF, written through GDAL with DEFLATE + PREDICTOR=2 compression (`Terrain`) or via `pyramids` defaults (`DEM`).
- **Vector input**: GeoJSON / Shapefile via `geopandas` for the optional `forced_direction` outfall point in `DEM.flow_direction`.
- **NetCDF / multi-dimensional formats**: explicitly **out of scope**. If you need NetCDF support, work with `pyramids.netcdf.NetCDF` directly.

## Consequences

- The package surface stays small and focused on hydrological / terrain operations on a single elevation raster.
- Any GDAL-readable raster works as input without per-format code in this repo.
- Compression defaults for GeoTIFF outputs are tuned for elevation/byte rasters (`COMPRESS=DEFLATE`, `PREDICTOR=2`).
