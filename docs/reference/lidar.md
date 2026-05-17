# LiDAR

LiDAR point-cloud I/O, classification, gridding, and analysis. The LAS / LAZ I/O surface
soft-imports `laspy` — install with `pip install laspy[lazrs]` to enable file read/write.
Everything else (gridding, ground filtering, clipping, tree detection) operates on the
in-memory `LasPoints` record without any external dependency beyond NumPy / SciPy.

## Module-level functions

::: digitalrivers.lidar
    options:
        show_root_heading: false
        show_source: true
        heading_level: 3
        members_order: source
        filters:
            - "!^_"

## Surface map

| Section | Functions / classes |
|---------|---------------------|
| **Record class** | `LasPoints` (xyz + intensity + classification + return_number + crs) |
| **I/O — W-15** | `read_las(path)`, `write_las(points, path, point_format=6, version="1.4")` |
| **Ground filter — W-16** | `classify_ground(points, method="zhang"/"axelsson", ...)` (Zhang 2003 tophat ships; Axelsson 2000 TIN-progressive deferred) |
| **Gridding — W-17 / P34** | `grid_lidar_points(xs, ys, zs, cell_size, aggregate=...)` — `min` / `max` / `mean` / `median` / `count` block aggregation plus `idw` / `nn` / `tin` / `rbf` interpolation |
| **Vector ops — W-18** | `clip(points, polygon, inverse=False)`, `merge(*pointclouds)`, `filter_classes(points, classes)` |
| **Forestry — W-19** | `detect_trees(chm, min_height_m=2.0, radius_fn=...)` — variable-window local-maxima on a canopy height model |
