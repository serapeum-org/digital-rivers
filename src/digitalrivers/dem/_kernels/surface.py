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

__all__ = [
    "eight_direction_slopes",
    "focal_window_stats",
    "curvature_surface",
    "ruggedness_surface",
    "normal_vector_deviation_surface",
    "area_slope_index",
]


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


def curvature_surface(elev: np.ndarray, cell_size: float, kind: str) -> np.ndarray:
    """Zevenbergen-Thorne curvature of a surface.

    Fits a local quadratic to each 3x3 neighbourhood and reads the curvature off its
    coefficients. The five kinds answer different questions: `profile` is curvature along
    the slope, which controls whether flow accelerates or decelerates; `plan` is curvature
    across it, which controls whether flow converges or diverges; `total`, `mean` and
    `gaussian` are the scale-free combinations.

    Args:
        elev: `(rows, cols)` elevation array. `NaN` marks no-data.
        cell_size: Cell size in CRS units.
        kind: One of `"plan"`, `"profile"`, `"total"`, `"mean"`, `"gaussian"`.

    Returns:
        `(rows, cols)` float32 curvature. Cells that were no-data, and cells where the
        fit is not finite, come back `NaN` for the caller to replace with its sentinel.

    Raises:
        ValueError: For an unrecognised `kind`.

    Examples:
        - A plane has no curvature of any kind:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import curvature_surface
            >>> plane = np.tile(np.arange(5, dtype=np.float64), (5, 1))
            >>> bool(abs(curvature_surface(plane, 1.0, "total")[2, 2]) < 1e-9)
            True

        - A bowl curves upward, so its total curvature is positive at the centre:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import curvature_surface
            >>> axis = (np.arange(5) - 2.0) ** 2
            >>> bowl = np.add.outer(axis, axis)
            >>> bool(curvature_surface(bowl, 1.0, "total")[2, 2] > 0)
            True
    """
    if kind not in ("plan", "profile", "total", "mean", "gaussian"):
        raise ValueError(
            f"kind must be one of 'plan', 'profile', 'total', 'mean', "
            f"'gaussian'; got {kind!r}"
        )
    z = elev.astype(np.float64, copy=False)
    length = float(abs(cell_size))
    # 3x3 stencil - np.pad(mode="edge") mirrors the boundary value so finite
    # differences stay defined at the raster edge.
    zp = np.pad(np.where(np.isnan(z), 0.0, z), 1, mode="edge")
    z1, z2, z3 = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    z4, z5, z6 = zp[1:-1, :-2], zp[1:-1, 1:-1], zp[1:-1, 2:]
    z7, z8, z9 = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    # Zevenbergen-Thorne polynomial coefficients.
    d = ((z4 + z6) / 2.0 - z5) / (length * length)
    e = ((z2 + z8) / 2.0 - z5) / (length * length)
    f = (-z1 + z3 + z7 - z9) / (4.0 * length * length)
    g = (-z4 + z6) / (2.0 * length)
    h = (z2 - z8) / (2.0 * length)
    denom = g * g + h * h + 1.0e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "plan":
            arr = -2.0 * (d * h * h + e * g * g - f * g * h) / denom
        elif kind == "profile":
            arr = 2.0 * (d * g * g + e * h * h + f * g * h) / denom
        elif kind == "total":
            arr = 2.0 * (d + e)
        elif kind == "mean":
            arr = d + e
        else:  # gaussian
            arr = 4.0 * d * e - f * f
    out = arr.astype(np.float32)
    return np.where(np.isnan(z) | ~np.isfinite(out), np.nan, out).astype(np.float32)


def ruggedness_surface(elev: np.ndarray, window: int) -> np.ndarray:
    """Mean absolute elevation difference between each cell and its neighbours.

    Riley's Terrain Ruggedness Index generalised to an arbitrary window. Unlike a
    standard deviation it does not care about the *shape* of the neighbourhood, only how
    far the surface moves across it, so a smooth steep slope and a rough flat one are
    told apart.

    Args:
        elev: `(rows, cols)` elevation array. `NaN` marks no-data.
        window: Side length of the neighbourhood in cells. Must be >= 1.

    Returns:
        `(rows, cols)` float32 ruggedness, `NaN` where the input was no-data.

    Raises:
        ValueError: If `window` is less than 1.

    Examples:
        - A flat surface is not rugged:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import ruggedness_surface
            >>> float(ruggedness_surface(np.full((5, 5), 4.0), 3)[2, 2])
            0.0

        - On a 0/10 checkerboard the four orthogonal neighbours each differ by 10 and
          the four diagonal ones by 0, so the mean is exactly half the amplitude:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import ruggedness_surface
            >>> z = np.indices((5, 5)).sum(axis=0) % 2 * 10.0
            >>> float(ruggedness_surface(z, 3)[2, 2])
            5.0
    """
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window!r}")
    z = elev.astype(np.float64, copy=False)
    z_filled = np.where(np.isnan(z), 0.0, z)
    total = np.zeros_like(z_filled)
    count = 0
    half = int(window) // 2
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(z_filled, shift=(dr, dc), axis=(0, 1))
            total += np.abs(z_filled - shifted)
            count += 1
    out = (total / float(count)).astype(np.float32)
    return np.where(np.isnan(z), np.nan, out).astype(np.float32)


def normal_vector_deviation_surface(
    elev: np.ndarray, cell_size: float, window: int
) -> np.ndarray:
    """Angle between each cell's surface normal and the mean normal around it.

    A roughness measure insensitive to overall tilt: a uniformly steep plane has every
    normal pointing the same way and scores zero, where a slope-based measure would call
    it rough. What it picks up is local disagreement in *orientation*.

    One blind spot worth knowing: the partials are central differences, comparing the
    cells two apart rather than the immediate neighbours. Detail that alternates every
    cell therefore aliases to a zero gradient and the metric reports a perfectly smooth
    surface — a 0/10 checkerboard scores the same as a plane. Anything with a period
    longer than two cells is seen normally.

    Args:
        elev: `(rows, cols)` elevation array. `NaN` marks no-data.
        cell_size: Cell size in CRS units.
        window: Side length of the averaging window in cells. Must be >= 1.

    Returns:
        `(rows, cols)` float32 angles in radians, `NaN` where the input was no-data.

    Raises:
        ValueError: If `window` is less than 1.

    Examples:
        - Every normal on a tilted plane points the same way, so the deviation is zero
          to within rounding. Not exactly zero: the angle comes from `arccos` of a dot
          product that is 1 to a part in 1e12, and `arccos` near 1 turns that into ~1e-6:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import (
            ...     normal_vector_deviation_surface,
            ... )
            >>> plane = np.tile(np.arange(7, dtype=np.float64), (7, 1))
            >>> bool(normal_vector_deviation_surface(plane, 1.0, 3)[3, 3] < 1e-5)
            True

        - A ridge makes the normals either side of it disagree sharply:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import (
            ...     normal_vector_deviation_surface,
            ... )
            >>> ridge = np.zeros((9, 9))
            >>> ridge[4, :] = 10.0
            >>> bool(normal_vector_deviation_surface(ridge, 1.0, 3)[3, 4] > 0.9)
            True

        - But a period-2 checkerboard is invisible to it, for the reason in the note
          above — the cells the central difference compares are equal:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import (
            ...     normal_vector_deviation_surface,
            ... )
            >>> z = np.indices((7, 7)).sum(axis=0) % 2 * 5.0
            >>> bool(normal_vector_deviation_surface(z, 1.0, 3)[3, 3] < 1e-5)
            True
    """
    # lazy: scipy is an undeclared optional dependency
    from scipy.ndimage import uniform_filter

    if window < 1:
        raise ValueError(f"window must be >= 1; got {window!r}")
    z = elev.astype(np.float64, copy=False)
    length = float(abs(cell_size))
    # Finite-difference partials with reflective edge padding.
    zp = np.pad(np.where(np.isnan(z), 0.0, z), 1, mode="edge")
    dzdx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2.0 * length)
    dzdy = (zp[2:, 1:-1] - zp[:-2, 1:-1]) / (2.0 * length)
    # Outward-pointing normal (-dz/dx, -dz/dy, 1); renormalise to unit.
    nx_raw = -dzdx
    ny_raw = -dzdy
    nz_raw = np.ones_like(z)
    nm = np.sqrt(nx_raw * nx_raw + ny_raw * ny_raw + nz_raw * nz_raw)
    nx = nx_raw / nm
    ny = ny_raw / nm
    nz = nz_raw / nm
    # Focal mean of each component, then renormalise to keep the mean vector unit-length.
    mnx = uniform_filter(nx, size=int(window), mode="reflect")
    mny = uniform_filter(ny, size=int(window), mode="reflect")
    mnz = uniform_filter(nz, size=int(window), mode="reflect")
    mm = np.sqrt(mnx * mnx + mny * mny + mnz * mnz) + 1.0e-12
    mnx /= mm
    mny /= mm
    mnz /= mm
    cos_theta = np.clip(nx * mnx + ny * mny + nz * mnz, -1.0, 1.0)
    out = np.arccos(cos_theta).astype(np.float32)
    return np.where(np.isnan(z), np.nan, out).astype(np.float32)


def area_slope_index(
    slope_deg: np.ndarray, accumulation: np.ndarray, cell_size: float, kind: str
) -> np.ndarray:
    """Wetness and sediment-transport indices from upstream area and slope.

    The three share a shape - a function of specific catchment area over a function of
    slope - which is why they share a kernel:

    * `twi` - Topographic Wetness Index, `ln(SCA / tan(slope))`. High where a lot drains
      in across little gradient, i.e. where water accumulates.
    * `spi` - Stream Power Index, `SCA * tan(slope)`. High where a lot drains in *and* the
      gradient is steep, i.e. where erosive energy concentrates.
    * `sti` - Sediment Transport Index, the RUSLE length-slope factor in the same terms.

    Slope is floored at 1e-3 rad (about 0.06 degrees) before the `tan` and `sin`, without
    which a flat cell divides by zero and TWI runs to infinity across every plain.

    Args:
        slope_deg: `(rows, cols)` slope in degrees. `NaN` marks no-data.
        accumulation: `(rows, cols)` upstream cell counts, same shape.
        cell_size: Cell size in CRS units, used to turn a cell count into an area.
        kind: One of `"twi"`, `"spi"`, `"sti"`.

    Returns:
        `(rows, cols)` float32 index values, `NaN` wherever the result is not finite.

    Raises:
        ValueError: For an unrecognised `kind`, or if the two inputs disagree in shape.

    Examples:
        - Wetness rises with upstream area at a fixed slope:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import area_slope_index
            >>> slope = np.full((2, 2), 5.0)
            >>> small = area_slope_index(slope, np.full((2, 2), 10.0), 30.0, "twi")
            >>> large = area_slope_index(slope, np.full((2, 2), 1000.0), 30.0, "twi")
            >>> bool(large[0, 0] > small[0, 0])
            True

        - Stream power rises with slope at a fixed area:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import area_slope_index
            >>> acc = np.full((2, 2), 100.0)
            >>> gentle = area_slope_index(np.full((2, 2), 1.0), acc, 30.0, "spi")
            >>> steep = area_slope_index(np.full((2, 2), 30.0), acc, 30.0, "spi")
            >>> bool(steep[0, 0] > gentle[0, 0])
            True

        - A flat cell is floored rather than dividing by zero:

            >>> import numpy as np
            >>> from digitalrivers.dem._kernels.surface import area_slope_index
            >>> out = area_slope_index(np.zeros((2, 2)), np.full((2, 2), 5.0), 1.0, "twi")
            >>> bool(np.isfinite(out[0, 0]))
            True
    """
    if kind not in ("twi", "spi", "sti"):
        raise ValueError(f"kind must be one of 'twi', 'spi', 'sti'; got {kind!r}")
    slope_deg_arr = np.asarray(slope_deg, dtype=np.float64)
    acc = np.asarray(accumulation, dtype=np.float64)
    if slope_deg_arr.shape != acc.shape:
        raise ValueError(
            f"slope shape {slope_deg_arr.shape} != accumulation shape {acc.shape}"
        )

    slope_rad = np.deg2rad(slope_deg_arr)
    # Floor at ~0.001 rad (about 0.06 degrees) to keep tan() / sin() bounded away from
    # zero on flats, where otherwise TWI runs to infinity across every plain.
    slope_rad = np.where(
        np.isfinite(slope_rad),
        np.maximum(slope_rad, 1.0e-3),
        np.nan,
    )

    cs = float(abs(cell_size))
    sca = acc * cs  # specific catchment area (m, since width ~ cell_size)

    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "twi":
            arr = np.log(sca / np.tan(slope_rad))
        elif kind == "spi":
            arr = sca * np.tan(slope_rad)
        else:  # sti
            arr = ((sca / 22.13) ** 0.6) * ((np.sin(slope_rad) / 0.0896) ** 1.3)

    return np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
