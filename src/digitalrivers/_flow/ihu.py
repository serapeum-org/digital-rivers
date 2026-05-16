"""Iterative Hydrography Upscaling (Eilander et al. 2021, HESS 25:5287–5313).

Greedy hill-climbing implementation of IHU. Starts from a COTAT initial
network, then iteratively swaps each coarse cell's outlet to whichever fine
candidate reduces a global drainage-area-error metric. Converges when no
single-cell swap improves the metric, or after ``max_iter`` sweeps.

References:
    Eilander D., van Verseveld W., Yamazaki D., Weerts A., Winsemius H. C.,
    Ward P. J. (2021). "A hydrography upscaling method for scale-invariant
    parametrization of distributed hydrological models." Hydrology and Earth
    System Sciences 25(9): 5287-5313. https://doi.org/10.5194/hess-25-5287-2021

Algorithm sketch:

1. For each fine cell, trace its downstream walk until it leaves the
   containing coarse cell. Record the last-in-block cell and the coarse-grid
   offset (``exit_dr``, ``exit_dc``) it exits to. Cells that never leave
   their block (sinks within the block) get no exit info and are not
   candidate outlets.
2. Per coarse cell, list candidate outlets — every fine cell whose
   downstream walk does exit the block. Sort by accumulation descending so
   the COTAT outlet (max-acc) is candidates[0].
3. Initialise current outlets = ``candidates[block][0]`` for every block.
4. Hill-climb: for each iteration, for each coarse cell, try every
   alternative outlet candidate in turn and accept the first swap that
   reduces the global error.
5. Global error: ``sum over coarse cells of |fine_acc[outlet] -
   coarse_acc[cell] * scale_factor^2|``. The coarse accumulation is
   recomputed via Kahn topological sweep (Phase 1 P6) for each trial.

The metric runs O(coarse_cells^2 * candidates_per_cell * max_iter). Pure
Python; Numba acceleration is a follow-up. Works on small/medium DEMs
(thousands of cells) within seconds; for continental DEMs use the pyflwdir
vendor path until a Numba IHU lands.
"""
from __future__ import annotations

import numpy as np

_DIR_DR = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_DIR_DC = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)


def _precompute_exit_info(fdir: np.ndarray, scale_factor: int) -> tuple:
    """For every fine cell, trace downstream until leaving its block.

    Returns four ``(rows, cols)`` int32 arrays:

    - ``has_exit``: 1 if the downstream walk exits the block, 0 otherwise.
    - ``exit_dr``, ``exit_dc``: coarse-grid offsets to the destination block
      (-1 where ``has_exit == 0``).
    """
    rows, cols = fdir.shape
    has_exit = np.zeros((rows, cols), dtype=np.int8)
    exit_dr = np.full((rows, cols), -1, dtype=np.int32)
    exit_dc = np.full((rows, cols), -1, dtype=np.int32)
    for fr in range(rows):
        for fc in range(cols):
            br0 = fr // scale_factor
            bc0 = fc // scale_factor
            r = fr
            c = fc
            while True:
                d = int(fdir[r, c])
                if d < 0 or d > 7:
                    break
                nr = r + int(_DIR_DR[d])
                nc = c + int(_DIR_DC[d])
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                nbr = nr // scale_factor
                nbc = nc // scale_factor
                if nbr != br0 or nbc != bc0:
                    has_exit[fr, fc] = 1
                    exit_dr[fr, fc] = nbr - br0
                    exit_dc[fr, fc] = nbc - bc0
                    break
                r = nr
                c = nc
    return has_exit, exit_dr, exit_dc


def _coarse_accumulation_from_outlets(
    outlets: dict, out_rows: int, out_cols: int,
) -> np.ndarray:
    """Coarse-grid Kahn topological-sort accumulation. Unit weights.

    Each cell's outgoing direction is derived from its outlet's
    ``(exit_dr, exit_dc)`` offset; cells without an outlet are sinks.
    Returns the float64 accumulation grid.
    """
    coarse_fdir = np.full((out_rows, out_cols), -1, dtype=np.int32)
    for (br, bc), out in outlets.items():
        edr = out[3]
        edc = out[4]
        for k in range(8):
            if int(_DIR_DR[k]) == edr and int(_DIR_DC[k]) == edc:
                coarse_fdir[br, bc] = k
                break
    # In-degree pass.
    indeg = np.zeros((out_rows, out_cols), dtype=np.int32)
    for br in range(out_rows):
        for bc in range(out_cols):
            d = int(coarse_fdir[br, bc])
            if d < 0 or d > 7:
                continue
            nr = br + int(_DIR_DR[d])
            nc = bc + int(_DIR_DC[d])
            if 0 <= nr < out_rows and 0 <= nc < out_cols:
                indeg[nr, nc] += 1
    from collections import deque
    queue: deque[tuple[int, int]] = deque()
    for br in range(out_rows):
        for bc in range(out_cols):
            if indeg[br, bc] == 0:
                queue.append((br, bc))
    out = np.zeros((out_rows, out_cols), dtype=np.float64)
    while queue:
        br, bc = queue.popleft()
        contrib = out[br, bc] + 1.0
        d = int(coarse_fdir[br, bc])
        if d < 0 or d > 7:
            continue
        nr = br + int(_DIR_DR[d])
        nc = bc + int(_DIR_DC[d])
        if not (0 <= nr < out_rows and 0 <= nc < out_cols):
            continue
        out[nr, nc] += contrib
        indeg[nr, nc] -= 1
        if indeg[nr, nc] == 0:
            queue.append((nr, nc))
    return out


def _global_error(
    outlets: dict, fine_acc: np.ndarray, out_rows: int, out_cols: int,
    scale_factor: int,
) -> float:
    """Sum over coarse cells of ``|fine_outlet_acc - coarse_acc * sf^2|``.

    This is the simplified axis-aligned form of the Eilander 2021 IHU
    drainage-area-error metric. The paper compares each coarse outlet's
    fine-grid upstream area against the cell's *true* upstream-subgrid
    count from the coarse topology; we approximate that with
    ``(coarse_acc + 1) * sf^2``, which assumes every upstream coarse cell
    contributes exactly ``sf^2`` fine cells.

    The approximation is exact for interior cells whose entire ``sf x sf``
    block sits inside the basin. It over-penalises cells near basin
    boundaries (where some of the ``sf x sf`` block falls outside the
    basin) — those cells' "true" subgrid contribution is less than
    ``sf^2``. Practical impact is small for ``sf <= 10`` on basins that
    are mostly interior; the convergence direction is preserved because
    the over-penalty is monotonic in coarse_acc.

    A faithful per-coarse-cell upstream-subgrid count would re-run the
    fine-grid BFS from every outlet on each swap trial — that is the
    follow-up referenced by the docstring note in :func:`ihu_upscale`.
    """
    coarse = _coarse_accumulation_from_outlets(outlets, out_rows, out_cols)
    sf2 = float(scale_factor ** 2)
    total = 0.0
    for (br, bc), out in outlets.items():
        a = float(out[0])  # fine accumulation at the outlet
        modeled = (coarse[br, bc] + 1.0) * sf2
        total += abs(a + 1.0 - modeled)
    return total


def ihu_upscale(
    fdir: np.ndarray,
    acc: np.ndarray,
    scale_factor: int,
    max_iter: int = 20,
) -> tuple[np.ndarray, dict, dict]:
    """Run hill-climbing IHU.

    Complexity note. Each candidate swap re-runs the global Kahn
    accumulation sweep over the coarse grid, so the run is
    O(N · K · max_iter) where N is the number of coarse cells and K is
    the average number of outlet candidates per coarse cell (bounded
    above by ``scale_factor ** 2``). On a 50×50 coarse output with
    ``max_iter=20`` the work fits in seconds; on a continental
    coarse grid prefer the pyflwdir vendor path.

    Args:
        fdir: ``(rows, cols)`` int32 fine-resolution D8 direction codes.
        acc: ``(rows, cols)`` float64 fine accumulation.
        scale_factor: integer aggregation factor.
        max_iter: hill-climbing sweep cap. Higher values run longer; the
            engine short-circuits once an iteration produces no swap.

    Returns:
        Tuple ``(coarse_fdir, metrics, outlets)``:
            coarse_fdir: ``(rows // sf, cols // sf)`` int32 with values 0-7
                or -1 for cells without an outlet candidate.
            metrics: dict carrying ``"final_error"``, ``"iterations"``,
                ``"swaps"``, ``"swaps_per_iteration"`` (per-iteration list),
                and ``"converged"`` (bool).
            outlets: dict mapping coarse-cell ``(br, bc)`` → outlet record
                ``(fine_acc, fine_row, fine_col, exit_dr, exit_dc)``.
                Useful for downstream callers that want to inspect the
                final outlet network.

    Examples:
        - Upscale a small east-flowing chain by factor 2 and inspect the
          full three-tuple return:

            >>> import numpy as np
            >>> from digitalrivers._flow.ihu import ihu_upscale
            >>> fdir = np.full((4, 4), 6, dtype=np.int32)
            >>> fdir[:, -1] = -1  # right-edge sinks
            >>> acc = np.tile(np.arange(4, dtype=np.float64), (4, 1))
            >>> coarse_fdir, metrics, outlets = ihu_upscale(
            ...     fdir, acc, scale_factor=2, max_iter=5,
            ... )
            >>> coarse_fdir.shape
            (2, 2)
            >>> sorted(metrics.keys())
            ['converged', 'final_error', 'iterations', 'swaps', 'swaps_per_iteration']
            >>> len(metrics["swaps_per_iteration"]) == metrics["iterations"]
            True
            >>> sum(metrics["swaps_per_iteration"]) == metrics["swaps"]
            True
            >>> len(outlets) > 0
            True
    """
    rows, cols = fdir.shape
    out_rows = rows // scale_factor
    out_cols = cols // scale_factor
    has_exit, exit_dr, exit_dc = _precompute_exit_info(fdir, scale_factor)

    # Per-block candidate list: sorted by acc descending. Each entry is
    # (acc, fr, fc, exit_dr, exit_dc).
    candidates: dict[tuple[int, int], list[tuple[float, int, int, int, int]]] = {}
    for fr in range(rows):
        for fc in range(cols):
            if not has_exit[fr, fc]:
                continue
            br = fr // scale_factor
            bc = fc // scale_factor
            candidates.setdefault((br, bc), []).append(
                (float(acc[fr, fc]), int(fr), int(fc),
                 int(exit_dr[fr, fc]), int(exit_dc[fr, fc]))
            )
    for k in candidates:
        candidates[k].sort(reverse=True)

    # Initial COTAT outlets (max-acc candidate per block).
    outlets = {k: v[0] for k, v in candidates.items()}

    base_error = _global_error(outlets, acc, out_rows, out_cols, scale_factor)
    iterations = 0
    total_swaps = 0
    swaps_per_iteration: list[int] = []
    converged = False
    for _ in range(max_iter):
        iterations += 1
        swaps_this_pass = 0
        for cell in list(outlets.keys()):
            alts = candidates[cell]
            if len(alts) <= 1:
                continue
            current = outlets[cell]
            for alt in alts:
                if alt == current:
                    continue
                trial = dict(outlets)
                trial[cell] = alt
                e = _global_error(trial, acc, out_rows, out_cols, scale_factor)
                if e < base_error - 1e-9:
                    outlets[cell] = alt
                    base_error = e
                    swaps_this_pass += 1
                    total_swaps += 1
                    break
        swaps_per_iteration.append(swaps_this_pass)
        if swaps_this_pass == 0:
            converged = True
            break

    # Build the final coarse fdir from accepted outlets.
    coarse_fdir = np.full((out_rows, out_cols), -1, dtype=np.int32)
    for (br, bc), out in outlets.items():
        edr = out[3]
        edc = out[4]
        for k in range(8):
            if int(_DIR_DR[k]) == edr and int(_DIR_DC[k]) == edc:
                coarse_fdir[br, bc] = k
                break
    metrics = {
        "final_error": base_error,
        "iterations": iterations,
        "swaps": total_swaps,
        "swaps_per_iteration": swaps_per_iteration,
        "converged": converged,
    }
    return coarse_fdir, metrics, outlets
