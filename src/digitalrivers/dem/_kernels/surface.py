"""Pure-numpy surface math behind the morphometric indices.

These are the array kernels the `MorphometryMixin` methods wrap. Keeping them here rather
than inline in the mixin is what makes them testable: each takes an elevation array and
returns an array, so a 5x5 fixture written in a test exercises the same code a raster
does, with no `Dataset` and nothing on disk.

Two of them are shared rather than per-index, which is the main reason the split is worth
having:

* :func:`eight_direction_slopes` backs both `DEM.slope` and `DEM.flow_direction`, so the
  slope convention is stated once and the two cannot drift apart.
* :func:`focal_window_stats` backs `tpi`, `deviation_from_mean` and `elev_std`, so the
  window convention and the no-data handling are decided in one place.

This module is deliberately Numba-free. Its sibling `_kernels/morphometry.py` holds the
JIT horizon walk and imports `numba` at module scope, so it can only be imported lazily;
these helpers are needed at module scope by the mixin and must stay cheap to import.
"""

from __future__ import annotations

import numpy as np

__all__ = ["eight_direction_slopes", "focal_window_stats"]


def eight_direction_slopes(elev: np.ndarray, cell_size: float) -> np.ndarray:
    """Slope from each cell to each of its eight neighbours.

    Slope is rise over run, positive downhill: `(centre - neighbour) / distance`, so a
    cell higher than its neighbour has a positive slope in that direction. Diagonal
    neighbours use `cell_size * sqrt(2)` rather than `cell_size`, without which diagonal
    slopes read about 41% too steep and D8 routing biases toward the diagonals.

    The grid is padded with `NaN`, so every off-edge neighbour yields a `NaN` slope
    rather than wrapping or clamping. Callers reduce with `np.nanmax` and get the
    steepest *in-bounds* descent.

    Args:
        elev: `(rows, cols)` elevation array. `NaN` marks no-data.
        cell_size: Cell size in CRS units.

    Returns:
        `(rows, cols, 8)` float32 array of slopes, indexed in `DIR_OFFSETS` order
        (`0=S, 1=SW, 2=W, 3=NW, 4=N, 5=NE, 6=E, 7=SE`). Boundary and no-data neighbours
        are `NaN`.

    Examples:
        - A cell one metre above a flat surround slopes away from it equally in the four
          cardinal directions:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import eight_direction_slopes
            >>> z = np.zeros((3, 3), dtype=np.float32)
            >>> z[1, 1] = 1.0
            >>> s = eight_direction_slopes(z, 1.0)
            >>> [round(float(s[1, 1, k]), 4) for k in (0, 2, 4, 6)]
            [1.0, 1.0, 1.0, 1.0]

        - and less steeply along the diagonals, because the run is longer:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import eight_direction_slopes
            >>> z = np.zeros((3, 3), dtype=np.float32)
            >>> z[1, 1] = 1.0
            >>> round(float(eight_direction_slopes(z, 1.0)[1, 1, 1]), 4)
            0.7071

        - Off-grid neighbours are NaN, not zero:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import eight_direction_slopes
            >>> s = eight_direction_slopes(np.zeros((3, 3), dtype=np.float32), 1.0)
            >>> bool(np.isnan(s[0, 0, 4]))
            True
    """
    dist2 = cell_size * np.sqrt(2)
    distances = [
        cell_size,
        dist2,
        cell_size,
        dist2,
        cell_size,
        dist2,
        cell_size,
        dist2,
    ]
    rows, cols = elev.shape
    slopes = np.full((rows, cols, 8), np.nan, dtype=np.float32)

    padded_elev = np.full((rows + 2, cols + 2), np.nan, dtype=np.float32)
    padded_elev[1:-1, 1:-1] = elev

    diff_right = padded_elev[1:-1, 1:-1] - padded_elev[1:-1, 2:]
    diff_top_right = padded_elev[1:-1, 1:-1] - padded_elev[:-2, 2:]
    diff_top = padded_elev[1:-1, 1:-1] - padded_elev[:-2, 1:-1]
    diff_top_left = padded_elev[1:-1, 1:-1] - padded_elev[:-2, :-2]
    diff_left = padded_elev[1:-1, 1:-1] - padded_elev[1:-1, :-2]
    diff_bottom_left = padded_elev[1:-1, 1:-1] - padded_elev[2:, :-2]
    diff_bottom = padded_elev[1:-1, 1:-1] - padded_elev[2:, 1:-1]
    diff_bottom_right = padded_elev[1:-1, 1:-1] - padded_elev[2:, 2:]

    slopes[:, :, 0] = diff_bottom / distances[0]
    slopes[:, :, 1] = diff_bottom_left / distances[1]
    slopes[:, :, 2] = diff_left / distances[2]
    slopes[:, :, 3] = diff_top_left / distances[3]
    slopes[:, :, 4] = diff_top / distances[4]
    slopes[:, :, 5] = diff_top_right / distances[5]
    slopes[:, :, 6] = diff_right / distances[6]
    slopes[:, :, 7] = diff_bottom_right / distances[7]

    return slopes


def focal_window_stats(
    elev: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectangular focal mean and standard deviation, aware of no-data.

    The mean and SD at each cell are taken over the *valid* cells inside its window only,
    rather than over the window as a whole. That matters at a raster's no-data edges: a
    plain `uniform_filter` would average the no-data fill into the statistic and drag
    every cell near a hole toward it.

    The trick is to filter the zero-filled surface and the validity mask with the same
    kernel, then divide. Both sums then carry the same divisor — the count of valid cells
    in the window — so the no-data cells contribute nothing rather than contributing
    zero.

    Args:
        elev: `(rows, cols)` elevation array. `NaN` marks no-data.
        window: Side length of the focal window in cells. Must be >= 1.

    Returns:
        `(z, mean, sd)`, each `(rows, cols)` float64. `z` is `elev` as float64. Cells
        whose window holds no valid neighbour are `NaN` in `mean` and `sd`; callers
        convert those to the raster's sentinel.

    Raises:
        ValueError: If `window` is less than 1.

    Examples:
        - On a flat surface the mean is the surface and the deviation is zero:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import focal_window_stats
            >>> _z, mean, sd = focal_window_stats(np.full((5, 5), 7.0), 3)
            >>> float(mean[2, 2]), float(sd[2, 2])
            (7.0, 0.0)

        - A no-data cell does not drag its neighbours' mean down:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import focal_window_stats
            >>> z = np.full((5, 5), 10.0)
            >>> z[2, 2] = np.nan
            >>> _z, mean, _sd = focal_window_stats(z, 3)
            >>> float(mean[1, 1])
            10.0

        - A window with nothing valid in it yields NaN rather than zero:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import focal_window_stats
            >>> _z, mean, _sd = focal_window_stats(np.full((3, 3), np.nan), 3)
            >>> bool(np.isnan(mean[1, 1]))
            True
    """
    from scipy.ndimage import (
        uniform_filter,
    )  # lazy: scipy is an undeclared optional dep

    if window < 1:
        raise ValueError(f"window must be >= 1; got {window!r}")
    z = elev.astype(np.float64, copy=False)
    valid = ~np.isnan(z)
    valid_f = valid.astype(np.float64)
    z_filled = np.where(valid, z, 0.0)
    sum_z = uniform_filter(z_filled, size=int(window), mode="reflect")
    sum_zz = uniform_filter(z_filled * z_filled, size=int(window), mode="reflect")
    # `uniform_filter` averages by default — multiply by the window area to recover
    # unscaled sums, so the same divisor applies to both the mean and the second moment.
    n_window = float(int(window) * int(window))
    sum_z *= n_window
    sum_zz *= n_window
    count = uniform_filter(valid_f, size=int(window), mode="reflect") * n_window
    with np.errstate(invalid="ignore", divide="ignore"):
        m = sum_z / count
        ex2 = sum_zz / count
        # Clip before the root: catastrophic cancellation in E[x^2] - E[x]^2 can leave a
        # tiny negative variance on a flat window, and sqrt of that is NaN. Clipping ties
        # sd's NaN-ness to the mean's, which callers rely on.
        sd = np.sqrt(np.maximum(ex2 - m * m, 0.0))
    return z, m, sd
