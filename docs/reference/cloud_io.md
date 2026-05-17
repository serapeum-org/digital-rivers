# Cloud I/O

Tile-window iteration and Cloud-Optimized GeoTIFF (COG) write helpers. Introduced as Phase-4
backfill P29 to give digital-rivers a Dask-style chunked-streaming story without requiring a Dask
dependency outright.

## Module-level functions

::: digitalrivers.cloud_io
    options:
        show_root_heading: false
        show_source: true
        heading_level: 3
        members_order: source
        filters:
            - "!^_"

## Surface map

| Function | Purpose |
|----------|---------|
| `tile_windows(dataset, tile_size, overlap=0)` | Generator yielding `(row_slice, col_slice)` windows for chunked I/O |
| `write_cog(dataset, path, compress="deflate")` | Write a pyramids `Dataset` as a Cloud-Optimized GeoTIFF (overviews + tile layout) |
| `dask_backend(*args, **kwargs)` | Umbrella stub — raises `NotImplementedError` with a pointer to `tile_windows` for current Dask interop |
| `cloud_storage(*args, **kwargs)` | Umbrella stub — raises `NotImplementedError` for cloud-storage adapters (`s3://`, `gs://`, …) |
