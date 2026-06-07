"""Internal out-of-core (tiled, larger-than-RAM) processing primitives.

This package is **internal** (underscore-prefixed) and not part of the public API. It holds the shared plumbing
for the tiled DEM-hydrology algorithms (Barnes 2016/2017 master-graph fill and accumulation): a halo-aware tile
plan, halo-aware tile read / core write over `pyramids`, an overflow-safe global cell-id scheme, and the
Evict/Retain/Cache intermediate store.

The public hydrology entry points (``DEM.fill_depressions(engine="tiled")`` etc.) dispatch into the modules here;
callers never import this package directly.

These primitives were originally scoped to `pyramids` (Workstream A: ``map_overlap`` / ``store_windows`` /
``tiled_reduce``) but were declared out of scope upstream, so digital-rivers owns the minimal versions it needs
here. See ``planning/out-of-core/`` for the design.
"""

from __future__ import annotations

from digitalrivers._outofcore.cache import TileStore
from digitalrivers._outofcore.tiling import (
    TileSpec,
    gid,
    plan_tiles,
    read_tile,
    write_core,
)

__all__ = [
    "TileSpec",
    "TileStore",
    "gid",
    "plan_tiles",
    "read_tile",
    "write_core",
]
