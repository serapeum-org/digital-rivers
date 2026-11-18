"""Numba morphometry kernel: the per-cell horizon walk.

:func:`horizon_walk_kernel` backs both `DEM.openness` (Yokoyama 2002) and
`DEM.sky_view_factor`. They differ only in how the eight per-direction horizon angles
are aggregated, which is the `mode` argument, so one walk serves both.

The kernel walks its own azimuth ordering (E, NE, N, NW, W, SW, S, SE) rather than the
`DIR_OFFSETS` convention the rest of the package uses. That is deliberate and harmless:
it covers the same eight neighbours and the result is a mean over all of them, so the
labelling does not reach the output. It is kept local rather than pulled into
`core.directions`, which would imply the two orderings are interchangeable.

Imported lazily by :mod:`digitalrivers.dem.dem` — this module pulls in `numba` at
import, and `import digitalrivers` must not.
"""

from __future__ import annotations

import numpy as np

from digitalrivers.core.numba import njit

__all__ = ["horizon_walk_kernel"]


@njit(cache=True)
def horizon_walk_kernel(
    z: np.ndarray,
    cell_size: float,
    search_radius: int,
    mode: int,
) -> np.ndarray:
    """Per-cell horizon-angle aggregation along 8 azimuths.

    Walks outward from every cell up to `search_radius` cells along each of
    8 cardinal / diagonal directions, recording the maximum elevation angle
    (above the cell's plane) as the "horizon" for that direction. The
    per-cell output is then aggregated across the 8 directions according
    to `mode`:

    * `mode = 0` (openness, Yokoyama 2002) — mean of `(π/2 - horizon)`
      across directions. High values mark exposed / open positions.
    * `mode = 1` (sky-view factor) — mean of `(1 - sin(horizon))` across
      directions. High values mark cells with most of the sky visible.

    Args:
        z: `(rows, cols)` float64 elevation grid (NaN-free; callers should
            pre-substitute a finite sentinel for no-data).
        cell_size: Cell size in map units. Must be positive.
        search_radius: Maximum walk distance in cells (≥ 1).
        mode: 0 for openness, 1 for sky-view factor.

    Returns:
        `(rows, cols)` float64 raster of the aggregated metric.
    """
    rows, cols = z.shape
    out = np.zeros_like(z)
    # 8-direction offsets: E, NE, N, NW, W, SW, S, SE (azimuth order).
    dr = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
    dc = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
    n_dirs = 8
    # Serial outer loop (no `prange`/`parallel=True`): the numba parallel
    # threading layer segfaults on macOS-arm64 for this kernel, and the per-cell
    # work is cheap enough that single-threaded JIT is fine.
    for r in range(rows):
        for c in range(cols):
            z0 = z[r, c]
            total = 0.0
            for k in range(n_dirs):
                max_angle = 0.0
                for s in range(1, int(search_radius) + 1):
                    nr = r + int(dr[k]) * s
                    nc = c + int(dc[k]) * s
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        break
                    d_step = (
                        float(cell_size)
                        * np.sqrt(float(dr[k] * dr[k] + dc[k] * dc[k]))
                        * float(s)
                    )
                    a = np.arctan2(z[nr, nc] - z0, d_step)
                    if a > max_angle:
                        max_angle = a
                if mode == 0:
                    total += np.pi / 2.0 - max_angle
                else:
                    total += 1.0 - np.sin(max_angle)
            out[r, c] = total / n_dirs
    return out
