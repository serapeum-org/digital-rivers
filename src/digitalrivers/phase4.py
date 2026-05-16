"""Public surface for Phase 4 helpers (P28-P35).

The actual implementations live in ``digitalrivers._phase4_stubs`` alongside
the umbrella NotImplementedError stubs for the genuinely-deferred halves
(Dask graph composition, Zarr / S3 / GCS factories, full PDAL pipeline,
multigrid ANUDEM, edge-flip mesh refinement). This module re-exports the
working halves under stable, non-underscore-prefixed names so downstream
callers don't have to import a private path.

* :func:`tile_windows` — chunked-iteration helper (P30 partial).
* :func:`write_cog` — Cloud-Optimised GeoTIFF writer (P31 partial).
* :func:`grid_lidar_points` — LiDAR point cloud → DEM gridding
  (P34 partial).
* :func:`topobathy_fusion` — bathymetric DEM fusion (P35).

The umbrella stubs (``dask_backend``, ``cloud_io``, ``anudem_solver``,
``mesh_quality_optimise``, ``pdal_lidar_pipeline``, ``native_cotat_upscale``,
``native_ihu_upscale``) remain at ``digitalrivers._phase4_stubs`` because
they raise ``NotImplementedError`` and are documented as deferred.
"""
from __future__ import annotations

from digitalrivers._phase4_stubs import (
    grid_lidar_points,
    tile_windows,
    topobathy_fusion,
    write_cog,
)

__all__ = [
    "grid_lidar_points",
    "tile_windows",
    "topobathy_fusion",
    "write_cog",
]
