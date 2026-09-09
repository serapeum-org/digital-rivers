## Unreleased

### Refactor

- **structure**: move the package from thirteen flat root modules into domain subpackages
  (`core/`, `dem/`, `flow/`, `streams/`, `watershed/`, `terrain/`, `lidar/`, `interop/`).

#### Moved import paths

The seven names re-exported from the package root are **unchanged** — `from digitalrivers import DEM,
FlowDirection, Accumulation, StreamRaster, WatershedRaster, Terrain, Mesh` works exactly as before, and is the
import worth using.

`digitalrivers.dem`, `digitalrivers.terrain` and `digitalrivers.lidar` also keep their paths: each became a
package of the same name.

Seven module paths moved. Each keeps a shim at the old location that re-exports from the new one and raises
`DeprecationWarning`; **the shims are removed in 0.6.0**.

| Old path | New path |
| --- | --- |
| `digitalrivers.flow_direction` | `digitalrivers.flow.direction` |
| `digitalrivers.accumulation` | `digitalrivers.flow.accumulation` |
| `digitalrivers.stream_raster` | `digitalrivers.streams.raster` |
| `digitalrivers.watershed_raster` | `digitalrivers.watershed.raster` |
| `digitalrivers.mesh` | `digitalrivers.interop.mesh` |
| `digitalrivers.cloud_io` | `digitalrivers.interop.cloud_io` |
| `digitalrivers.fusion` | `digitalrivers.dem.fusion` |

The shims carry the class (or, for `cloud_io`, its four functions) and nothing else. Names those modules exposed
incidentally — none of them declared `__all__`, so anything they imported was reachable through them — are not
carried over. The ones most likely to be in use:

| Name | New home |
| --- | --- |
| `VALID_ROUTING`, `VALID_ENCODING` | `digitalrivers.core.metadata` |
| `META_CLASS`, `META_ROUTING`, `META_ENCODING`, `META_THRESHOLD` | `digitalrivers.core.metadata` |
| `strahler`, `horton`, `shreve`, `hack`, `topological` | `digitalrivers.streams._kernels.order` |
| `kahn_max_upslope_length` | `digitalrivers.flow._kernels.accumulation` |
| `ihu_upscale` | `digitalrivers.flow._kernels.ihu` |
| `watershed_d8` | `digitalrivers.watershed._kernels.watershed` |
| `local_minima_8` | `digitalrivers.dem._kernels.pitremoval` |
| `hand_d8` | `digitalrivers.streams._kernels.hand` |

The private `digitalrivers._numba` module is gone; its kernels now sit with the domain that uses them, in
`digitalrivers.dem._kernels.numba`, `digitalrivers.dem._kernels.morphometry` and
`digitalrivers.flow._kernels.numba`.

## 0.1.0 (2026-05-18)

### Feat

- deliver Phase 1, 2, 3, and 4 of the digital-rivers roadmap (#13)

### Refactor

- **dem,terrain**: overhaul DEM/Terrain modules with bug fixes, tests, and CI (#2)
