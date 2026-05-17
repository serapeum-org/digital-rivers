# LiDAR pipeline

The W-15 → W-19 ports built a complete LAS / LAZ → DEM workflow on top of the existing
`grid_lidar_points` block-aggregator (P34). The full chain:

```text
LAS / LAZ file
  → read_las                    # W-15
  → classify_ground             # W-16 (Zhang 2003 tophat)
  → filter_classes({2})         # W-18 — keep ground points
  → grid_lidar_points(method=)  # W-17 — interpolation (idw / nn / tin / rbf) or block aggregate
  → detect_trees                # W-19 — local-maxima on a CHM
```

The LAS I/O step **soft-imports** `laspy` — install with `pip install laspy[lazrs]` to enable
read/write. All other stages (classification, gridding, clipping, merging, tree detection)
run on the in-memory `LasPoints` record with no laspy dependency.

## 1. The `LasPoints` record (W-15)

Plain dataclass holding parallel NumPy arrays — index `i` selects the i-th point across every
field. ASPRS class codes follow the LAS 1.4 standard.

| Field | dtype | Meaning |
|-------|-------|---------|
| `x`, `y`, `z` | `float64` | World coordinates (scale + offset already applied by laspy) |
| `intensity` | `uint16` | Return intensity (empty array if not populated) |
| `classification` | `uint8` | ASPRS class code (2=ground, 5=high veg, 6=building, 9=water, ...) |
| `return_number` | `uint8` | Return-number-within-pulse |
| `crs` | `pyproj.CRS | None` | Parsed from the LAS header on read |

Construct from arrays directly, or:

* `lidar.read_las(path)` — read a LAS / LAZ file into `LasPoints` (laspy required).
* `lidar.write_las(points, path, point_format=6, version="1.4")` — write to disk (laspy required).

## 2. Ground classification (W-16)

`lidar.classify_ground(points, method="zhang", cell_size=1.0, window_cells=5, slope_threshold=1.0)`
implements the **Zhang 2003 morphological tophat** filter:

1. Build a min-grid DEM from the points.
2. Apply a morphological opening (single structuring-element scale —
   `scipy.ndimage.grey_opening`).
3. A point is ground iff `z - opening_at_cell <= slope_threshold`; non-ground otherwise.

Returns an `(N,)` `uint8` array of ASPRS codes (2 = ground, 1 = unclassified / non-ground).

The Axelsson 2000 TIN-progressive ground filter is documented in the API but raises
`NotImplementedError` — it's deferred subsystem-scale work. For Axelsson-style classification today
use PDAL's `filters.smrf` or `lastools` and pass the pre-classified points in.

## 3. Vector operations (W-18)

| Function | Purpose |
|----------|---------|
| `lidar.clip(points, polygon, inverse=False)` | Keep points inside (default) or outside the polygon. Uses `shapely.contains_xy` for vectorised point-in-polygon. |
| `lidar.merge(*pointclouds)` | Concatenate two or more clouds. Optional fields (intensity / classification / return_number) survive only when every input has them. |
| `lidar.filter_classes(points, classes)` | Keep only points whose classification is in the set. Requires populated classification. |

## 4. Gridding (W-17 + P34)

`lidar.grid_lidar_points(xs, ys, zs, cell_size, bounds=None, aggregate=...)` covers both **block
aggregation** (one value per cell from its contained points) and **spatial interpolation** (one
value per cell centre, computed from nearby points):

### Block aggregation (one-pass, fast)

| `aggregate` | Output |
|-------------|--------|
| `"min"` | Lowest z per cell — canonical bare-earth DEM choice for first-return LiDAR |
| `"max"` | Highest z per cell — DSM / canopy choice |
| `"mean"` | Per-cell average |
| `"median"` | Per-cell median (per-cell bucketing — slower than mean) |
| `"count"` | Per-cell point count (density raster) |

### Interpolation (cell-centre values)

| `aggregate` | Backend | Kwargs |
|-------------|---------|--------|
| `"idw"` | scipy `cKDTree` k-NN inverse-distance weighting | `idw_k=8`, `idw_power=2.0` |
| `"nn"` | scipy `cKDTree` 1-NN | — |
| `"tin"` | scipy `LinearNDInterpolator` (Delaunay barycentric) | — |
| `"rbf"` | scipy `RBFInterpolator` | `rbf_kernel="thin_plate_spline"`, `rbf_smoothing=0.0` |

## 5. Tree detection (W-19)

`lidar.detect_trees(chm, min_height_m=2.0, radius_fn=None)` runs a **variable-window local-maxima
search** on a canopy height model (CHM, typically `DSM - DTM`):

* Candidate cells: CHM ≥ `min_height_m`.
* For each candidate, the search window half-width scales with height via `radius_fn(h)`.
  Default: `_default_tree_radius(h) = 0.5 + 0.05 * h` (Popescu & Wynne 2004 conifer rule of thumb).
* Cell is a tree top iff its CHM value is the maximum in the window.

Returns a `geopandas.GeoDataFrame` of Point geometries with `height_m`, `row`, `col`, and
`geometry` columns.

## See also

* [LiDAR reference](../reference/lidar.md) — auto-generated function signatures and docstrings.
