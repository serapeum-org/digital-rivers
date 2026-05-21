"""Least-cost depression breaching (Lindsay 2016) for DEM hydro pre-processing (P3).

Breaching is the structural alternative to filling: instead of raising the pit floor, it
*cuts a channel through the lowest barrier* between the pit and an outlet. On LiDAR DEMs
this is usually more realistic — most internal pits are data artefacts (interpolation
shadows, vegetation gaps, road embankments) and the natural drainage path is preserved by
cutting the artefact away rather than inflating the surrounding terrain.

Three methods are exposed via :func:`breach_depressions`:

* `"single_cell"` — cheap O(n) pre-pass that resolves isolated 1-cell pits by lowering an
  intermediate first-order neighbour to the midpoint of the pit and a lower second-order
  cell. Does nothing if no such configuration exists. Run before any heavier method to
  dispose of speckle-noise pits without inflating the DEM.
* `"least_cost"` — Lindsay 2016 Dijkstra-from-each-pit over an elevation-cost surface.
  Carves a strictly monotonic channel from the pit to the nearest outlet (a data cell at
  or below the pit elevation, or a no-data cell). Optional `max_depth` and `max_length`
  constraints abort the breach if the channel would be too deep or too long; aborted pits
  are left unresolved.
* `"hybrid"` — try `least_cost` first; pits that fail their constraint fall back to the
  Priority-Flood depression fill from P2. The breach phase has already lowered parts of the
  DEM where partial breaching occurred, so the fill operates on a modified surface and
  produces less overall lift than fill-only.

No-data handling: cells flagged no-data are treated as free outlets (a Dijkstra path that
reaches a no-data cell terminates the search). Input is not mutated.
"""

from __future__ import annotations

import heapq
from itertools import count

import numpy as np

from digitalrivers._conditioning.pitremoval import (
    _NEIGHBOURS_8,
    fill_depressions,
    local_minima_8,
)

VALID_BREACH_METHODS: frozenset[str] = frozenset(
    {"least_cost", "hybrid", "single_cell"}
)


def _small_num(z: np.ndarray, nodata_mask: np.ndarray) -> float:
    """Whitebox-style data-range-scaled per-step elevation lift for the back-trace.

    Chosen so the cumulative lift over a typical breach path (≤ a few hundred cells) is
    numerically negligible against the DEM's elevation range but always strictly positive,
    so the back-trace produces a strictly monotonic channel. Float64 throughout — the
    formula gives ~1e-6 to 1e-12 m for plausible DEMs, well above the float64 noise floor.
    """
    valid = z[~nodata_mask]
    if valid.size == 0:
        return 1e-9
    z_range = float(valid.max() - valid.min())
    return max(z_range, 1.0) * 1e-9


def _backtrace(
    z: np.ndarray,
    end_r: int,
    end_c: int,
    pit_z: float,
    pathlen: np.ndarray,
    backlink: np.ndarray,
    small_num: float,
) -> None:
    """Walk from `(end_r, end_c)` back to the pit, lowering path cells to a downhill chain.

    Each cell at distance `k` from the pit is lowered to `pit_z - k * small_num` if its
    current elevation is higher. The pit itself is left alone (its `backlink` is -1, which
    terminates the walk).
    """
    cur_r, cur_c = end_r, end_c
    while backlink[cur_r, cur_c] != -1:
        z_target = pit_z - pathlen[cur_r, cur_c] * small_num
        if z[cur_r, cur_c] > z_target:
            z[cur_r, cur_c] = z_target
        d = backlink[cur_r, cur_c]
        dr, dc = _NEIGHBOURS_8[d]
        cur_r -= dr
        cur_c -= dc


def _candidate_intermediates(dr2: int, dc2: int) -> list[tuple[int, int]]:
    """First-order offsets `(dr1, dc1)` such that `(dr1, dc1) -> (dr2, dc2)` is a single
    8-connected step. Used by the single-cell pit pre-pass to identify which neighbour of
    the pit can be lowered as a channel to a lower second-order cell.
    """
    out: list[tuple[int, int]] = []
    for dr1, dc1 in _NEIGHBOURS_8:
        if abs(dr2 - dr1) <= 1 and abs(dc2 - dc1) <= 1:
            out.append((dr1, dc1))
    return out


def _breach_single_cell_pits(
    z: np.ndarray,
    nodata_mask: np.ndarray,
    pits: np.ndarray,
) -> list[tuple[int, int]]:
    """Resolve isolated single-cell pits by lowering an intermediate first-order neighbour.

    For each pit, scan the 16 second-order cells (Chebyshev distance 2). The first
    second-order cell strictly lower than the pit is selected; an intermediate first-order
    cell that connects the pit to that second-order cell is lowered to `(z_pit + z_low) /
    2`, breaching the pit without modifying any other cell.

    Args:
        z: 2-D float64 elevation array. Mutated in place.
        nodata_mask: 2-D bool array, True at no-data cells.
        pits: `(n, 2)` array of pit `(row, col)` coordinates.

    Returns:
        List of pits that could not be resolved by this pass (no lower second-order cell
        with a higher intermediate, or pit too close to the array edge).
    """
    rows, cols = z.shape
    unresolved: list[tuple[int, int]] = []

    # All 16 second-order offsets: max(|dr|, |dc|) == 2.
    second_order_offsets: list[tuple[int, int]] = [
        (dr, dc)
        for dr in (-2, -1, 0, 1, 2)
        for dc in (-2, -1, 0, 1, 2)
        if max(abs(dr), abs(dc)) == 2
    ]

    for pit_r, pit_c in pits:
        pit_r = int(pit_r)
        pit_c = int(pit_c)
        if nodata_mask[pit_r, pit_c]:
            continue
        pit_z = float(z[pit_r, pit_c])

        resolved = False
        for dr2, dc2 in second_order_offsets:
            r2 = pit_r + dr2
            c2 = pit_c + dc2
            if r2 < 0 or r2 >= rows or c2 < 0 or c2 >= cols:
                continue
            if nodata_mask[r2, c2]:
                # A second-order no-data cell is a free outlet through any intermediate.
                z_target = (
                    pit_z  # lower intermediate to pit elevation; nodata is the sink
                )
            elif z[r2, c2] < pit_z:
                z_target = (pit_z + float(z[r2, c2])) / 2.0
            else:
                continue

            for dr1, dc1 in _candidate_intermediates(dr2, dc2):
                r1 = pit_r + dr1
                c1 = pit_c + dc1
                if r1 < 0 or r1 >= rows or c1 < 0 or c1 >= cols:
                    continue
                if nodata_mask[r1, c1]:
                    continue
                if z[r1, c1] > z_target:
                    z[r1, c1] = z_target
                    resolved = True
                    break
            if resolved:
                break

        if not resolved:
            unresolved.append((pit_r, pit_c))

    return unresolved


def _breach_least_cost_one_pit(
    z: np.ndarray,
    nodata_mask: np.ndarray,
    pit_r: int,
    pit_c: int,
    max_depth: float | None,
    max_length: int | None,
    small_num: float,
    cost: np.ndarray,
    backlink: np.ndarray,
    pathlen: np.ndarray,
    visited: np.ndarray,
) -> bool:
    """Run Dijkstra from a single pit; on outlet, back-trace and mutate `z` to install the
    channel. The four state arrays (`cost`, `backlink`, `pathlen`, `visited`) are
    pre-allocated by the caller and reset for touched cells after this returns.

    Returns:
        True if an outlet was reached within the constraints; False if aborted by
        `max_depth` / `max_length` or if the reachable region exhausted without finding
        one.
    """
    rows, cols = z.shape
    pit_z = float(z[pit_r, pit_c])

    cost[pit_r, pit_c] = 0.0
    counter = count()
    heap: list[tuple[float, int, int, int]] = [(0.0, next(counter), pit_r, pit_c)]
    touched: list[tuple[int, int]] = [(pit_r, pit_c)]

    while heap:
        accum, _, r, c = heapq.heappop(heap)
        if visited[r, c]:
            continue
        visited[r, c] = True

        if max_depth is not None and accum > max_depth:
            # Caller will treat this as unresolved. Reset state before returning.
            for tr, tc in touched:
                cost[tr, tc] = np.inf
                backlink[tr, tc] = -1
                pathlen[tr, tc] = 0
                visited[tr, tc] = False
            return False

        for d_idx, (dr, dc) in enumerate(_NEIGHBOURS_8):
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if visited[nr, nc]:
                continue

            len_n = pathlen[r, c] + 1
            if max_length is not None and len_n > max_length:
                continue

            if nodata_mask[nr, nc]:
                # No-data neighbour is a free outlet; back-trace from (r, c) to the pit.
                _backtrace(z, r, c, pit_z, pathlen, backlink, small_num)
                for tr, tc in touched:
                    cost[tr, tc] = np.inf
                    backlink[tr, tc] = -1
                    pathlen[tr, tc] = 0
                    visited[tr, tc] = False
                return True

            z_n = float(z[nr, nc])
            zout_n = pit_z - len_n * small_num
            if z_n <= zout_n:
                # Found a data outlet at or below the running spill height. Back-trace from
                # (r, c) — the outlet itself is left untouched.
                _backtrace(z, r, c, pit_z, pathlen, backlink, small_num)
                for tr, tc in touched:
                    cost[tr, tc] = np.inf
                    backlink[tr, tc] = -1
                    pathlen[tr, tc] = 0
                    visited[tr, tc] = False
                return True

            cost2 = max(0.0, z_n - zout_n)
            new_cost = accum + cost2
            if new_cost < cost[nr, nc]:
                if not np.isfinite(cost[nr, nc]):
                    touched.append((int(nr), int(nc)))
                cost[nr, nc] = new_cost
                backlink[nr, nc] = d_idx
                pathlen[nr, nc] = len_n
                heapq.heappush(heap, (new_cost, next(counter), int(nr), int(nc)))

    # Heap exhausted without reaching an outlet.
    for tr, tc in touched:
        cost[tr, tc] = np.inf
        backlink[tr, tc] = -1
        pathlen[tr, tc] = 0
        visited[tr, tc] = False
    return False


def breach_depressions(
    z: np.ndarray,
    nodata_mask: np.ndarray | None = None,
    *,
    method: str = "least_cost",
    max_depth: float | None = None,
    max_length: int | None = None,
    fill_remaining: bool = True,
) -> np.ndarray:
    """Breach depressions in an elevation surface (Lindsay 2016 family).

    Args:
        z: 2-D elevation array (any float dtype; promoted to float64 internally).
        nodata_mask: 2-D bool mask, True at no-data cells. NaN cells in `z` are added
            automatically. If `None`, only the array boundary acts as an outlet.
        method: `"single_cell"`, `"least_cost"` (default), or `"hybrid"`.
        max_depth: Maximum cumulative `|Δz|` for a single breach path. Pits whose nearest
            outlet exceeds this are unresolved (and filled if `method="hybrid"` and
            `fill_remaining=True`). `None` disables the constraint.
        max_length: Maximum path length in cells. `None` disables.
        fill_remaining: Only meaningful when `method="hybrid"`. If True, unresolved pits
            are passed to Priority-Flood with `epsilon=0`. If False, they are left as
            pits in the output.

    Returns:
        2-D float64 array with breach channels installed. No-data positions hold NaN; all
        other cells satisfy `z_out <= z_in` on the breach path and `z_out == z_in`
        elsewhere.

    Raises:
        ValueError: If `method` is unknown.
    """
    if method not in VALID_BREACH_METHODS:
        raise ValueError(
            f"method must be one of {sorted(VALID_BREACH_METHODS)}; got {method!r}"
        )

    if nodata_mask is None:
        nodata_mask = np.zeros(z.shape, dtype=bool)
    else:
        nodata_mask = nodata_mask.astype(bool, copy=False)

    nan_mask = np.isnan(z)
    if nan_mask.any():
        nodata_mask = nodata_mask | nan_mask

    z_work = z.astype(np.float64, copy=True)
    z_work[nodata_mask] = np.nan

    pit_mask = local_minima_8(z_work, nodata_mask=nodata_mask)
    pit_rcs = np.argwhere(pit_mask)
    if pit_rcs.size == 0:
        return z_work

    # Process pits in ascending elevation so shallower pits don't re-breach a channel that
    # a deeper pit has already carved.
    pit_elevs = z_work[pit_rcs[:, 0], pit_rcs[:, 1]]
    order = np.argsort(pit_elevs, kind="stable")
    pit_rcs = pit_rcs[order]

    if method == "single_cell":
        _breach_single_cell_pits(z_work, nodata_mask, pit_rcs)
        return z_work

    # least_cost and hybrid both run the full Dijkstra. Cheap single-cell pre-pass first.
    remaining_after_singles = _breach_single_cell_pits(z_work, nodata_mask, pit_rcs)
    # Re-detect pits after the single-cell pass — some pits may have been resolved as a
    # side effect of lowering an intermediate that was also a neighbour of another pit.
    pit_mask = local_minima_8(z_work, nodata_mask=nodata_mask)
    pit_rcs = np.argwhere(pit_mask)
    if pit_rcs.size > 0:
        pit_elevs = z_work[pit_rcs[:, 0], pit_rcs[:, 1]]
        order = np.argsort(pit_elevs, kind="stable")
        pit_rcs = pit_rcs[order]

    small_num = _small_num(z_work, nodata_mask)
    rows, cols = z_work.shape
    cost = np.full((rows, cols), np.inf, dtype=np.float64)
    backlink = np.full((rows, cols), -1, dtype=np.int8)
    pathlen = np.zeros((rows, cols), dtype=np.int32)
    visited = np.zeros((rows, cols), dtype=bool)

    unresolved: list[tuple[int, int]] = []
    for pit_r, pit_c in pit_rcs:
        pit_r = int(pit_r)
        pit_c = int(pit_c)
        # A pit from the initial mask might have been incidentally resolved by an earlier
        # breach; re-check before running Dijkstra.
        if not _is_local_minimum(z_work, pit_r, pit_c, nodata_mask):
            continue
        ok = _breach_least_cost_one_pit(
            z_work,
            nodata_mask,
            pit_r,
            pit_c,
            max_depth=max_depth,
            max_length=max_length,
            small_num=small_num,
            cost=cost,
            backlink=backlink,
            pathlen=pathlen,
            visited=visited,
        )
        if not ok:
            unresolved.append((pit_r, pit_c))

    if method == "hybrid" and unresolved and fill_remaining:
        # Fall back to Priority-Flood on the (already breach-modified) surface. We fill the
        # whole DEM; the breach phase has lowered enough cells that this is cheaper than
        # fill-only on the original surface.
        z_work = fill_depressions(
            z_work, nodata_mask=nodata_mask, method="priority_flood", epsilon=0.0
        )
    return z_work


def _is_local_minimum(z: np.ndarray, r: int, c: int, nodata_mask: np.ndarray) -> bool:
    """Cheap point-wise local-minimum check used by the per-pit re-validation loop.

    Vectorised via a 3×3 window slice + masked comparison: pulls the 3-row × 3-col
    neighbourhood (clipped at array edges), masks the centre and any no-data cells
    out, then asks whether every remaining neighbour is strictly higher than the
    centre.
    """
    rows, cols = z.shape
    if nodata_mask[r, c]:
        return False
    r0, r1 = max(0, r - 1), min(rows, r + 2)
    c0, c1 = max(0, c - 1), min(cols, c + 2)
    window = z[r0:r1, c0:c1]
    nd_window = nodata_mask[r0:r1, c0:c1]
    centre_local = (r - r0, c - c0)
    valid_neighbour = ~nd_window
    # Exclude the centre itself from the neighbour mask.
    valid_neighbour[centre_local] = False
    if not valid_neighbour.any():
        return False
    return bool(np.all(window[valid_neighbour] > z[r, c]))
