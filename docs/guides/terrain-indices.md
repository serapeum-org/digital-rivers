# Terrain indices

W-21 through W-28 added a full set of terrain-attribute indices to the `DEM` class. Every method
returns a typed `Dataset` aligned with the DEM's geotransform and uses float32 storage with the
DEM's no-data sentinel.

## Focal-window stats (W-21 → W-24)

These four indices share a private `_focal_window_stats` kernel — a no-data-aware
`scipy.ndimage.uniform_filter` pass that divides by the per-window valid-neighbour count, so cells
near the data envelope are computed only from the valid cells inside their window (not
contaminated by no-data substitution).

| Method | Formula | What it measures |
|--------|---------|------------------|
| `DEM.tpi(window)` (**W-21**) | `z - focal_mean(z)` | Topographic Position Index — ridges > 0, valleys < 0 (Guisan 1999) |
| `DEM.deviation_from_mean(window)` (**W-22**) | `(z - focal_mean) / focal_sd` | Standardised TPI; dimensionless |
| `DEM.elev_std(window)` (**W-23**) | `focal_sd(z)` | Local elevation SD; a roughness proxy |
| `DEM.ruggedness(window)` (**W-24**) | mean(\|z - z_neighbour\|) | Riley et al. 1999 terrain ruggedness index |

## Surface geometry (W-25, W-26)

### Curvature family (W-25)

`DEM.curvature(kind=...)` fits a Zevenbergen-Thorne 1987 partial quartic polynomial to each cell's
3×3 neighbourhood and evaluates one of five variants from the coefficient grid:

| `kind` | Formula | Geomorphological meaning |
|--------|---------|--------------------------|
| `"plan"` | `-2(D·H² + E·G² - F·G·H) / (G² + H²)` | Curvature perpendicular to slope direction |
| `"profile"` | `2(D·G² + E·H² + F·G·H) / (G² + H²)` | Curvature parallel to slope direction |
| `"total"` | `2(D + E)` | Total relief curvature (sign-independent) |
| `"mean"` | `D + E` | Average of the two principal curvatures |
| `"gaussian"` | `4·D·E - F²` | Product of the two principal curvatures |

The polynomial coefficients are `D = ∂²z/∂x²/2`, `E = ∂²z/∂y²/2`, `F = ∂²z/∂x∂y`, `G = ∂z/∂x`,
`H = ∂z/∂y` (Zevenbergen-Thorne notation).

### Normal-vector angular deviation (W-26)

`DEM.normal_vector_deviation(window)` computes each cell's outward-pointing surface normal from
finite differences, focal-means the unit-normal components in a window, and returns the per-cell
angle (in radians) between the local normal and the focal-mean normal. A roughness metric that
grows with how strongly the surface bends inside the window.

## Visibility (W-27, W-28)

Both metrics share a numba-accelerated `horizon_walk_kernel` (in `digitalrivers._numba`) that walks
outward from every cell along 8 azimuths up to `search_radius` cells, recording the maximum
elevation angle (the "horizon") in each direction.

### Topographic openness (W-27)

`DEM.openness(search_radius, kind="positive"/"negative")` — Yokoyama 2002.

* **Positive openness** = mean of `(π/2 - horizon_angle)` across the 8 azimuths. High values mark
  exposed / high-relief locations (the cell sees nothing higher than itself nearby).
* **Negative openness** = same kernel applied to `-z`. High values mark deep depressions.

### Sky-view factor (W-28)

`DEM.sky_view_factor(search_radius)` — Zakšek et al. 2011. The fraction of the upper hemisphere
visible from each cell:

`SVF = mean over directions of (1 - sin(horizon_angle))`

Range `[0, 1]`; 1 on flat terrain, < 1 wherever surrounding terrain occludes part of the sky.

## See also

* [DEM reference](../reference/dem.md) — full API including all eight terrain-index methods.
* [WhiteboxTools deep-dive](https://github.com/serapeum-org/digital-rivers/blob/feat/phase-4/planning/feat/whiteboxtools-deep-dive.md)
  — port-task tracker covering every W-N item.
