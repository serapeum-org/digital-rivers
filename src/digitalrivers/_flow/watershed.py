"""Watershed delineation kernels (P13 / P14).

Reverse-BFS upstream from each pour point under D8 / Rho8 routing, labelling
every contributing cell with the basin ID of its downstream seed. Used by
:class:`FlowDirection.watershed` for pour-point delineation (P13) and by the
no-pour-points variant in P14 (each terminal outlet becomes its own basin).
"""
from __future__ import annotations

from collections import deque

import numpy as np

_DIR_DR = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_DIR_DC = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
_INV_DIR = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int32)


def watershed_d8(
    fdir: np.ndarray,
    seeds: list[tuple[int, int]],
    basin_ids: list[int],
    require_unique_basins: bool = False,
) -> np.ndarray:
    """Reverse-BFS watershed labelling.

    For each `(seed, basin_id)` pair, walk upstream from the seed and label
    every contributing cell with that ID. Seeds are processed in the order given;
    when `require_unique_basins=False` an inner seed's basin overwrites the
    outer one's cells (so the outer basin contains a hole). When
    `require_unique_basins=True` the first seed to claim a cell wins.

    Args:
        fdir: `(rows, cols)` int32 direction-code raster (DIR_OFFSETS codes).
        seeds: list of `(row, col)` integer cell positions.
        basin_ids: list of integer IDs, parallel to `seeds`.
        require_unique_basins: if True, the first basin to claim a cell keeps
            it; if False (default), later basins overwrite earlier ones along
            shared upstream paths.

    Returns:
        `(rows, cols)` int32 basin-ID raster. Cells not in any basin are 0.

    Examples:
        - A single seed at the end of a westward chain captures every cell
          upstream of it:

            >>> import numpy as np
            >>> fdir = np.array([[6, 6, 6, -1]], dtype=np.int32)
            >>> basins = watershed_d8(fdir, [(0, 3)], [1])
            >>> [int(v) for v in basins[0]]
            [1, 1, 1, 1]

        - Two seeds with require_unique_basins=True keep first-claim:

            >>> import numpy as np
            >>> fdir = np.array([[6, 6, 6, -1]], dtype=np.int32)
            >>> basins = watershed_d8(
            ...     fdir, [(0, 3), (0, 1)], [1, 2],
            ...     require_unique_basins=True,
            ... )
            >>> [int(v) for v in basins[0]]
            [1, 1, 1, 1]
    """
    if len(seeds) != len(basin_ids):
        raise ValueError("seeds and basin_ids must have the same length")
    rows, cols = fdir.shape
    out = np.zeros((rows, cols), dtype=np.int32)

    # In `require_unique_basins=False` mode the contract is "later seeds
    # overwrite earlier seeds along shared upstream paths" — which is
    # equivalent to "process seeds in reverse order, first-claim wins".
    # The reversed form costs O(N) total because each cell is visited at
    # most once across all BFS sweeps, whereas the naive forward form is
    # O(B*N) when basins overlap heavily.
    if require_unique_basins:
        ordered = list(zip(seeds, basin_ids))
    else:
        ordered = list(zip(reversed(seeds), reversed(basin_ids)))

    for (sr, sc), bid in ordered:
        if not (0 <= sr < rows and 0 <= sc < cols):
            continue
        if out[sr, sc] != 0:
            continue
        queue: deque[tuple[int, int]] = deque([(sr, sc)])
        out[sr, sc] = bid
        while queue:
            r, c = queue.popleft()
            for k in range(8):
                ur = r + int(_DIR_DR[k])
                uc = c + int(_DIR_DC[k])
                if not (0 <= ur < rows and 0 <= uc < cols):
                    continue
                if int(fdir[ur, uc]) != int(_INV_DIR[k]):
                    continue
                if out[ur, uc] != 0:
                    continue
                out[ur, uc] = bid
                queue.append((ur, uc))
    return out
