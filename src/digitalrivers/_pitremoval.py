"""Depression-fill algorithms for DEM hydro pre-processing (P2).

Three methods are exposed via :func:`fill_depressions`:

* ``"priority_flood"`` — Barnes, Lehman & Mulla (2014) Priority-Flood with the two-queue plateau
  optimisation. With ``epsilon > 0`` it produces a strictly monotonic surface; with ``epsilon == 0``
  it produces flat fills (mathematically equivalent to Wang & Liu, just with a faster plateau drain).
* ``"wang_liu"`` — Wang & Liu (2006). Flat fill, no epsilon. Implemented as Priority-Flood with the
  pit queue disabled, which produces the same output more transparently.
* ``"planchon_darboux"`` — Planchon & Darboux (2002). Iterative directional-sweep algorithm.
  Slower than Priority-Flood on large DEMs, kept for low-relief reference.

No-data handling is consistent across the three: cells flagged no-data are treated as outlets
(they cannot be filled, and data cells adjacent to them act as drainage seeds along with the
true raster boundary). The returned array preserves the no-data positions as NaN; the input is
not mutated.
"""
from __future__ import annotations

import heapq
from collections import deque
from itertools import count

import numpy as np

# 8-connectivity neighbour offsets (dr, dc).
_NEIGHBOURS_8: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

VALID_METHODS: frozenset[str] = frozenset({"priority_flood", "wang_liu", "planchon_darboux"})


def local_minima_8(z: np.ndarray, nodata_mask: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of cells strictly lower than all 8 valid neighbours.

    Boundary cells (first/last row and column) are excluded — they have fewer than 8
    neighbours and the comparison is ill-defined. NaN cells (and cells flagged by
    ``nodata_mask`` if provided) are excluded from the output AND ignored when computing
    a candidate cell's neighbour minimum.

    This is the generic local-minima detector that the breach algorithm uses to find pits
    and the depression-fill tests use as a "no internal sinks remain" assertion. It is
    earmarked for promotion to ``pyramids.morphology`` (see
    ``planning/pyramids/pyramids-feat-morphology-utils.md``).

    Args:
        z: 2-D float array. NaN marks no-data unless a separate ``nodata_mask`` is given.
        nodata_mask: Optional 2-D bool array, True at no-data cells. Combined with NaN
            positions in ``z`` if both are present.

    Returns:
        2-D bool array, True at strict 8-connected local minima.

    Examples:
        - Single pit at the centre of a 5×5 grid is flagged:

            >>> import numpy as np
            >>> z = np.array([
            ...     [5., 5., 5., 5., 5.],
            ...     [5., 4., 4., 4., 5.],
            ...     [5., 4., 1., 4., 5.],
            ...     [5., 4., 4., 4., 5.],
            ...     [5., 5., 5., 5., 5.],
            ... ])
            >>> mask = local_minima_8(z)
            >>> bool(mask[2, 2])
            True
            >>> int(mask.sum())
            1

        - A monotonic ramp has no internal local minima:

            >>> import numpy as np
            >>> z = np.arange(16, dtype=float).reshape(4, 4)
            >>> local_minima_8(z).any()
            np.False_
    """
    if z.ndim != 2:
        raise ValueError(f"local_minima_8 expects a 2-D array; got {z.ndim}-D")
    nan_mask = np.isnan(z)
    if nodata_mask is not None:
        nan_mask = nan_mask | nodata_mask.astype(bool, copy=False)
    rows, cols = z.shape
    out = np.zeros((rows, cols), dtype=bool)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if nan_mask[r, c]:
                continue
            window = z[r - 1 : r + 2, c - 1 : c + 2]
            others = np.delete(window.ravel(), 4)
            valid = ~np.isnan(others)
            if nodata_mask is not None:
                nm_window = nodata_mask[r - 1 : r + 2, c - 1 : c + 2]
                valid = valid & ~np.delete(nm_window.ravel(), 4)
            if not valid.any():
                continue
            if z[r, c] < others[valid].min():
                out[r, c] = True
    return out


def _nodata_adjacent(nodata_mask: np.ndarray) -> np.ndarray:
    """Boolean mask of data cells touching a no-data cell (8-connectivity).

    Pure-NumPy 8-connected binary dilation of ``nodata_mask`` minus the mask itself.
    Equivalent to ``scipy.ndimage.binary_dilation(nodata_mask) & ~nodata_mask`` but
    without the scipy dependency.

    Args:
        nodata_mask: 2-D bool array, True where the cell is no-data.

    Returns:
        2-D bool array, True at data cells with at least one no-data 8-neighbour.
    """
    if not nodata_mask.any():
        return np.zeros_like(nodata_mask)
    rows, cols = nodata_mask.shape
    dilated = nodata_mask.copy()
    for dr, dc in _NEIGHBOURS_8:
        src_r = slice(max(0, dr), min(rows, rows + dr))
        src_c = slice(max(0, dc), min(cols, cols + dc))
        dst_r = slice(max(0, -dr), min(rows, rows - dr))
        dst_c = slice(max(0, -dc), min(cols, cols - dc))
        dilated[dst_r, dst_c] |= nodata_mask[src_r, src_c]
    return dilated & ~nodata_mask


def _seed_mask(nodata_mask: np.ndarray) -> np.ndarray:
    """Boolean mask of drainage-seed cells.

    A seed cell is any data cell that is either on the raster's array boundary or adjacent to a
    no-data cell. These are the cells that get pushed onto the open queue at the start of a
    Priority-Flood sweep — they represent the "outside world" where water leaves the grid.

    Args:
        nodata_mask: 2-D bool array, True where the cell is no-data.

    Returns:
        2-D bool array of seed positions. Disjoint from ``nodata_mask``.
    """
    rows, cols = nodata_mask.shape
    seed = np.zeros((rows, cols), dtype=bool)
    seed[0, :] = ~nodata_mask[0, :]
    seed[-1, :] = ~nodata_mask[-1, :]
    seed[:, 0] = ~nodata_mask[:, 0]
    seed[:, -1] = ~nodata_mask[:, -1]
    seed |= _nodata_adjacent(nodata_mask)
    return seed


def _priority_flood(
    z: np.ndarray,
    nodata_mask: np.ndarray,
    *,
    epsilon: float,
    use_pit_queue: bool,
) -> np.ndarray:
    """Core Priority-Flood implementation (Barnes 2014).

    Args:
        z: 2-D float64 elevation array. No-data positions may hold any value; the algorithm
            never reads them.
        nodata_mask: 2-D bool array, True at no-data cells.
        epsilon: Per-step elevation lift inside depressions. ``0.0`` produces flat fills.
            ``epsilon > 0`` produces a strictly monotonic surface with cumulative lift
            proportional to plateau width.
        use_pit_queue: If True, route same-or-lower-elevation neighbours through a FIFO pit
            queue (Barnes 2014 two-queue optimisation — keeps plateau drain at O(1) per cell
            instead of O(log n)). If False, push everything onto the heap (Wang & Liu reference
            behaviour; same output, slower on plateaus).

    Returns:
        2-D float64 filled-elevation array. No-data positions are set to NaN.
    """
    rows, cols = z.shape
    z_fill = z.astype(np.float64, copy=True)
    z_fill[nodata_mask] = np.nan

    closed = np.zeros((rows, cols), dtype=bool)
    closed |= nodata_mask

    open_heap: list[tuple[float, int, int, int]] = []
    pit: deque[tuple[float, int, int, int]] = deque()
    counter = count()

    for r, c in zip(*np.where(_seed_mask(nodata_mask))):
        heapq.heappush(open_heap, (float(z_fill[r, c]), next(counter), int(r), int(c)))
        closed[r, c] = True

    while open_heap or pit:
        if pit:
            e, _, r, c = pit.popleft()
        else:
            e, _, r, c = heapq.heappop(open_heap)

        for dr, dc in _NEIGHBOURS_8:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if closed[nr, nc]:
                continue
            closed[nr, nc] = True

            if z_fill[nr, nc] <= e:
                lifted = e + epsilon if epsilon > 0 else e
                z_fill[nr, nc] = lifted
                if use_pit_queue:
                    pit.append((lifted, next(counter), nr, nc))
                else:
                    heapq.heappush(open_heap, (lifted, next(counter), nr, nc))
            else:
                heapq.heappush(open_heap, (float(z_fill[nr, nc]), next(counter), nr, nc))

    return z_fill


def _planchon_darboux(
    z: np.ndarray,
    nodata_mask: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Planchon & Darboux (2002) iterative directional-sweep depression fill.

    Initialises ``W = +inf`` interior and ``W = Z`` on seed cells (raster boundary plus data
    cells adjacent to no-data), then repeats four directional sweeps until no cell is updated.
    Each sweep visits cells in a different scan order so that updates propagate from every seed
    direction.

    Args:
        z: 2-D float64 elevation array.
        nodata_mask: 2-D bool array, True at no-data cells.
        epsilon: Per-step elevation lift. Must be strictly positive — Planchon-Darboux without
            epsilon does not converge to a unique solution on plateaus.

    Returns:
        2-D float64 filled-elevation array. No-data positions are set to NaN.

    Raises:
        ValueError: If ``epsilon`` is not strictly positive.
    """
    if not (epsilon > 0):
        raise ValueError(
            f"planchon_darboux requires epsilon > 0; got {epsilon}. "
            "Use method='wang_liu' for flat fills without an explicit gradient."
        )

    rows, cols = z.shape
    z64 = z.astype(np.float64, copy=False)

    w = np.full((rows, cols), np.inf, dtype=np.float64)
    seed = _seed_mask(nodata_mask)
    w[seed] = z64[seed]
    w[nodata_mask] = np.nan

    # Four sweep orders covering both rasterised diagonals.
    sweep_orders: tuple[tuple[range, range], ...] = (
        (range(0, rows, 1), range(0, cols, 1)),
        (range(rows - 1, -1, -1), range(cols - 1, -1, -1)),
        (range(0, rows, 1), range(cols - 1, -1, -1)),
        (range(rows - 1, -1, -1), range(0, cols, 1)),
    )

    something_done = True
    while something_done:
        something_done = False
        for r_range, c_range in sweep_orders:
            for r in r_range:
                for c in c_range:
                    if nodata_mask[r, c]:
                        continue
                    if w[r, c] == z64[r, c]:
                        continue
                    for dr, dc in _NEIGHBOURS_8:
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            continue
                        if nodata_mask[nr, nc]:
                            continue
                        wn = w[nr, nc] + epsilon
                        if z64[r, c] >= wn:
                            w[r, c] = z64[r, c]
                            something_done = True
                            break
                        if w[r, c] > wn:
                            w[r, c] = wn
                            something_done = True
    return w


def _priority_flood_with_numba(
    z: np.ndarray, nodata_mask: np.ndarray, epsilon: float
) -> np.ndarray:
    """Numba-accelerated entry point for the default ``priority_flood`` path.

    Falls back to the pure-Python :func:`_priority_flood` (with pit-queue) when
    Numba is disabled / unavailable via the ``DIGITALRIVERS_DISABLE_NUMBA`` env
    var. The Numba kernel is bit-for-bit identical on synthetic fixtures; on
    real DEMs it is ≥ 20× faster on cold runs and ≥ 50× faster warm.
    """
    from digitalrivers._numba import (
        _DIR_DR_I32,
        _DIR_DC_I32,
        is_numba_enabled,
        priority_flood_numba,
    )

    if is_numba_enabled():
        return priority_flood_numba(
            z.astype(np.float64, copy=False), nodata_mask, float(epsilon),
            _DIR_DR_I32, _DIR_DC_I32,
        )
    return _priority_flood(z, nodata_mask, epsilon=epsilon, use_pit_queue=True)


def fill_depressions(
    z: np.ndarray,
    nodata_mask: np.ndarray | None = None,
    *,
    method: str = "priority_flood",
    epsilon: float = 0.0,
) -> np.ndarray:
    """Fill depressions in an elevation surface.

    Args:
        z: 2-D elevation array (any float dtype; promoted to float64 internally).
        nodata_mask: 2-D bool mask, True at no-data cells. If ``None``, no cell is treated
            as no-data and only the array boundary acts as a drainage seed. ``NaN`` cells in
            ``z`` are added to the mask automatically.
        method: One of ``"priority_flood"`` (default), ``"wang_liu"``, ``"planchon_darboux"``.
        epsilon: Per-step elevation lift inside depressions. ``0.0`` produces flat fills
            (ignored entirely by ``wang_liu``; rejected by ``planchon_darboux`` which requires
            ``> 0``). Positive values guarantee a strict downhill path at the cost of slight
            elevation inflation proportional to plateau width.

    Returns:
        2-D float64 array. No-data positions hold NaN; all other cells satisfy
        ``z_fill >= z`` and every interior cell has at least one 8-neighbour with strictly
        lower (epsilon > 0) or equal-or-lower (epsilon == 0) elevation.

    Raises:
        ValueError: If ``method`` is unknown, or ``planchon_darboux`` is requested with
            ``epsilon <= 0``.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {sorted(VALID_METHODS)}; got {method!r}"
        )

    if nodata_mask is None:
        nodata_mask = np.zeros(z.shape, dtype=bool)
    else:
        nodata_mask = nodata_mask.astype(bool, copy=False)

    nan_mask = np.isnan(z)
    if nan_mask.any():
        nodata_mask = nodata_mask | nan_mask

    if method == "priority_flood":
        return _priority_flood_with_numba(z, nodata_mask, epsilon)
    if method == "wang_liu":
        # Wang & Liu = Priority-Flood with epsilon=0 and no pit queue.
        return _priority_flood(z, nodata_mask, epsilon=0.0, use_pit_queue=False)
    return _planchon_darboux(z, nodata_mask, epsilon=epsilon)
