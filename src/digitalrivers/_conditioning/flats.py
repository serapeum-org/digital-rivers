"""Flat-area resolution for DEM hydro pre-processing (P4).

After `fill_depressions(method="wang_liu")` (or `priority_flood` with `epsilon=0`),
every closed depression is filled to its spill elevation — but the interior of each
filled depression is now a *flat plateau* with no defined steepest descent. D8 flow
direction over the result has `NO_FLOW` cells across every plateau.

This module imposes an artificial gradient on each plateau so that flow has a unique
deterministic direction. The algorithm is Garbrecht & Martz (1997) with the Barnes
(2014) BFS optimisation:

1. Find connected components of equal-elevation cells with size > 1 (the *plateaus*).
2. For each plateau cell, classify it as:
   - **LEC** (low-edge cell) if at least one 8-neighbour has strictly lower elevation
     (the outlet of the depression).
   - **HEC** (high-edge cell) if at least one 8-neighbour has strictly higher elevation
     (the rim).
   A cell can be both.
3. BFS from LECs inward → `g_low` (distance to nearest outlet, level 1 at the outlet).
4. BFS from HECs inward → `g_high` (distance to nearest rim). Invert per-plateau so
   cells far from the rim get the smallest value.
5. Add `(2 * g_high_inverted + g_low) * epsilon` to plateau cells. The `2x` weighting
   makes "drain towards the outlet" dominate, with "drain away from higher terrain"
   acting as a deterministic tiebreaker for cells equidistant between two outlets.

No-data is honoured throughout — no-data cells are skipped as both plateau members and as
gradient seeds. Plateaus with no LEC (closed depressions that should not survive a fill)
are left unmodified.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# 8-connectivity neighbour offsets (dr, dc).
_NEIGHBOURS_8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# 4-connectivity neighbour offsets (dr, dc) — cardinal directions only.
_NEIGHBOURS_4: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 0),
)


def _label_plateaus(
    z: np.ndarray,
    nodata_mask: np.ndarray,
    neighbours: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, int]:
    """Connected-component label of equal-elevation cells with at least one equal neighbour.

    Returns an int32 label grid (0 = not in any plateau) and the total number of plateaus.
    Plateaus respect both elevation equality and the supplied connectivity.
    """
    rows, cols = z.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    current_label = 0

    for r in range(rows):
        for c in range(cols):
            if labels[r, c] != 0 or nodata_mask[r, c]:
                continue
            z_c = z[r, c]
            # Check if there is at least one equal-elevation neighbour. If not, this cell
            # is a singleton (not a plateau) and we skip the BFS.
            has_equal = False
            for dr, dc in neighbours:
                nr = r + dr
                nc = c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not nodata_mask[nr, nc]
                    and z[nr, nc] == z_c
                ):
                    has_equal = True
                    break
            if not has_equal:
                continue

            current_label += 1
            queue: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = current_label
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in neighbours:
                    nr = cr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and not nodata_mask[nr, nc]
                        and z[nr, nc] == z_c
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = current_label
                        queue.append((nr, nc))

    return labels, current_label


def _classify_lec_hec(
    z: np.ndarray,
    labels: np.ndarray,
    nodata_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of low-edge and high-edge plateau cells.

    LEC: plateau cell with at least one 8-neighbour at strictly lower elevation.
    HEC: plateau cell with at least one 8-neighbour at strictly higher elevation.
    A cell can be both. LEC/HEC classification always uses 8-connectivity regardless of the
    plateau-labelling connectivity — Garbrecht & Martz require it.
    """
    rows, cols = z.shape
    is_lec = np.zeros((rows, cols), dtype=bool)
    is_hec = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if labels[r, c] == 0:
                continue
            z_c = z[r, c]
            for dr, dc in _NEIGHBOURS_8:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if nodata_mask[nr, nc]:
                    continue
                if z[nr, nc] < z_c:
                    is_lec[r, c] = True
                elif z[nr, nc] > z_c:
                    is_hec[r, c] = True
    return is_lec, is_hec


def _bfs_levels(
    labels: np.ndarray,
    seeds: np.ndarray,
    neighbours: tuple[tuple[int, int], ...],
    max_iter: int,
) -> np.ndarray:
    """BFS-level grid: cell `c` gets level `k` if its shortest plateau-internal path to
    any seed has `k - 1` hops. Seeds themselves are level 1. Non-plateau cells stay at 0.

    BFS stays within a single plateau (a step to a neighbour with a different label is
    not taken).
    """
    rows, cols = labels.shape
    g = np.zeros((rows, cols), dtype=np.int32)

    frontier: deque[tuple[int, int]] = deque()
    for r, c in zip(*np.where(seeds)):
        g[r, c] = 1
        frontier.append((int(r), int(c)))

    level = 1
    while frontier and level < max_iter:
        next_frontier: deque[tuple[int, int]] = deque()
        while frontier:
            r, c = frontier.popleft()
            lbl = labels[r, c]
            for dr, dc in neighbours:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if labels[nr, nc] != lbl:
                    continue
                if g[nr, nc] != 0:
                    continue
                g[nr, nc] = level + 1
                next_frontier.append((nr, nc))
        frontier = next_frontier
        level += 1

    return g


def _invert_per_plateau(
    g: np.ndarray,
    labels: np.ndarray,
    num_plateaus: int,
) -> np.ndarray:
    """For each plateau, invert `g` so cells far from the original seeds get 0 and cells
    next to a seed get the per-plateau maximum minus 1.

    Used on the `g_high` grid so that "distance from HEC" becomes "elevation above the
    plateau interior" — cells next to higher terrain get the largest bump, draining
    flow away from the rim.
    """
    if num_plateaus == 0:
        return g
    g_out = np.zeros_like(g)
    for lbl in range(1, num_plateaus + 1):
        mask = labels == lbl
        if not mask.any():
            continue
        max_g = int(g[mask].max())
        if max_g == 0:
            continue
        # Cells with g == 0 inside the plateau (no seed reached them) stay at 0.
        # Other cells become max_g - g (so seed cells at g=1 become max_g-1).
        in_plateau_with_g = mask & (g > 0)
        g_out[in_plateau_with_g] = max_g - g[in_plateau_with_g]
    return g_out


def resolve_flats(
    z: np.ndarray,
    nodata_mask: np.ndarray | None = None,
    *,
    epsilon: float = 1e-5,
    connectivity: int = 8,
    max_iter: int = 1000,
) -> np.ndarray:
    """Impose a deterministic gradient on every flat plateau in `z`.

    Args:
        z: 2-D elevation array (any float dtype; promoted to float64 internally).
        nodata_mask: 2-D bool mask, True at no-data cells. NaN positions in `z` are
            added automatically.
        epsilon: Per-BFS-step elevation lift. Total lift over a plateau is at most
            `(2 * max_high_dist + max_low_dist) * epsilon`; pick small enough that this
            stays well below the minimum elevation step between adjacent non-plateau
            cells. Default `1e-5` is safe for ~1000-cell-wide plateaus on metre-precision
            DEMs.
        connectivity: 4 or 8. Controls both the plateau-labelling and the BFS step.
            LEC/HEC classification always uses 8-connectivity regardless (Garbrecht-Martz
            convention). Default is 8.
        max_iter: Safety cap on BFS levels per plateau. Real plateaus rarely exceed
            `max(rows, cols)`; the default `1000` is essentially unbounded.

    Returns:
        2-D float64 array with plateau cells nudged so each has a defined steepest
        descent. No-data positions hold NaN; the input is not mutated.

    Raises:
        ValueError: If `connectivity` is not 4 or 8.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8; got {connectivity}")

    if nodata_mask is None:
        nodata_mask = np.zeros(z.shape, dtype=bool)
    else:
        nodata_mask = nodata_mask.astype(bool, copy=False)

    nan_mask = np.isnan(z)
    if nan_mask.any():
        nodata_mask = nodata_mask | nan_mask

    z_out = z.astype(np.float64, copy=True)
    z_out[nodata_mask] = np.nan

    neighbours = _NEIGHBOURS_8 if connectivity == 8 else _NEIGHBOURS_4

    labels, num_plateaus = _label_plateaus(z_out, nodata_mask, neighbours)
    if num_plateaus == 0:
        return z_out

    is_lec, is_hec = _classify_lec_hec(z_out, labels, nodata_mask)

    g_low = _bfs_levels(labels, is_lec, neighbours, max_iter)
    g_high_raw = _bfs_levels(labels, is_hec, neighbours, max_iter)
    g_high = _invert_per_plateau(g_high_raw, labels, num_plateaus)

    # A plateau gets resolved only if it both:
    #   (1) contains at least one truly-flat cell (no strictly lower 8-neighbour), and
    #   (2) has at least one LEC to drain to.
    # Plateaus where every cell is an LEC (e.g. a 9-ring framing a lower-elevation
    # region) are mathematically plateaus but every cell already has flow direction —
    # lifting them is a needless modification of the input. Plateaus without an LEC
    # are closed depressions that fill should have already removed; leaving them
    # alone keeps the algorithm a pure no-op when there is nothing safe to do.
    plateau_needs_resolution = np.zeros(num_plateaus + 1, dtype=bool)
    has_lec_per_plateau = np.zeros(num_plateaus + 1, dtype=bool)
    for lbl in range(1, num_plateaus + 1):
        mask = labels == lbl
        if not mask.any():
            continue
        if is_lec[mask].any():
            has_lec_per_plateau[lbl] = True
        if (~is_lec[mask]).any():
            plateau_needs_resolution[lbl] = True

    resolvable_plateaus = plateau_needs_resolution & has_lec_per_plateau
    plateau_mask = labels > 0
    final_lift_mask = plateau_mask & resolvable_plateaus[labels]
    if final_lift_mask.any():
        # Apply lift only to genuinely-flat cells within resolvable plateaus. The LECs
        # themselves are left at the plateau's original elevation — they already have
        # flow direction toward the outlet, and lifting them would shrink the gradient
        # between the LEC and the cell it drains to.
        true_flat = final_lift_mask & ~is_lec
        z_out[true_flat] += (2 * g_high[true_flat] + g_low[true_flat]) * epsilon

    return z_out
