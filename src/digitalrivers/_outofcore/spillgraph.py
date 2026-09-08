"""Global spillover graph for the Barnes 2016 tiled depression fill (B3).

After every tile is flooded and watershed-labelled (B2), the only cross-tile information that matters is, for each
pair of labelled watersheds, the **lowest elevation at which they meet** — the saddle they would spill over. Those
saddles form a small graph whose nodes are global watershed labels (plus a special :data:`OUTLET` node for the
domain edge / no-data, the ultimate drain at ``-inf``) and whose edge weights are spill elevations.

The graph "is itself a DEM": solving it is a minimax (bottleneck) shortest-path from the outlet — exactly a
Priority-Flood on the label graph. The result, ``drain[label]``, is the elevation each watershed must be raised
to. The whole reconciliation is **max-of-the-two-cells then keep-the-minimum-over-touching-pairs**.

This module is pure Python/NumPy (no GDAL, no Numba) and is perimeter-sized, so it is cheap and unit-testable in
isolation.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

import numpy as np

#: Special graph node id for the domain edge / no-data "ocean" — the ultimate outlet (drains at ``-inf``).
OUTLET = 0

# 8-neighbour forward directions (right, down, down-right, down-left) — enough to visit every adjacency once.
_FORWARD = ((0, 1), (1, 0), (1, 1), (1, -1))


class GlobalSpillGraph:
    """Accumulates watershed spill elevations and solves for each label's global drainage elevation.

    Nodes are global watershed labels (``>= 1``); node :data:`OUTLET` (``0``) is the domain edge / no-data
    outlet. Edge weight ``(a, b) -> e`` is the lowest elevation at which watersheds ``a`` and ``b`` meet.
    """

    def __init__(self) -> None:
        self.edges: dict[tuple[int, int], float] = {}

    def add_edge(self, a: int, b: int, elevation: float) -> None:
        """Register a spill of ``elevation`` between labels ``a`` and ``b``, keeping the minimum seen.

        Self-edges (``a == b``) are ignored. The pair is stored order-independently (``min, max``).
        """
        a = int(a)
        b = int(b)
        if a == b:
            return
        if a > b:
            a, b = b, a
        key = (a, b)
        current = self.edges.get(key)
        if current is None or elevation < current:
            self.edges[key] = float(elevation)

    def add_outlet(self, label: int, elevation: float) -> None:
        """Register that ``label`` can spill out of the domain (to :data:`OUTLET`) at ``elevation``."""
        self.add_edge(OUTLET, label, elevation)

    def add_adjacency(self, labels: np.ndarray, filled: np.ndarray) -> None:
        """Add intra-region spill edges from a tile's ``labels`` + ``filled`` arrays.

        For every pair of 8-adjacent cells carrying different (``>= 1``) labels, the saddle elevation is
        ``max(filled_a, filled_b)``; :meth:`add_edge` keeps the minimum over all touching pairs.

        Args:
            labels: `(rows, cols)` integer label array (``0`` = no-data / outside, ``>= 1`` = watersheds).
            filled: `(rows, cols)` float filled-elevation array aligned with ``labels``.
        """
        rows, cols = labels.shape
        for dr, dc in _FORWARD:
            r0, r1 = (0, rows - dr) if dr >= 0 else (-dr, rows)
            c0, c1 = (0, cols - dc) if dc >= 0 else (-dc, cols)
            if r1 <= r0 or c1 <= c0:
                continue
            la = labels[r0:r1, c0:c1]
            lb = labels[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            fa = filled[r0:r1, c0:c1]
            fb = filled[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            mask = (la != lb) & (la >= 1) & (lb >= 1)
            if not mask.any():
                continue
            self._reduce_pairs(la[mask], lb[mask], np.maximum(fa[mask], fb[mask]))

    def join_strips(
        self,
        a_labels: np.ndarray,
        a_filled: np.ndarray,
        b_labels: np.ndarray,
        b_filled: np.ndarray,
    ) -> None:
        """Stitch two adjacent tiles along their shared seam (orthogonal + diagonal touches).

        ``a_*`` / ``b_*`` are aligned 1-D border strips (tile A's edge facing tile B and vice versa). Cell ``i``
        of A touches cells ``i-1, i, i+1`` of B. Spill = ``max(filled_a, filled_b)``; minimum kept.
        """
        n = len(a_labels)
        m = len(b_labels)
        for i in range(n):
            la = int(a_labels[i])
            if la < 1:
                continue
            fa = float(a_filled[i])
            for dj in (-1, 0, 1):
                j = i + dj
                if 0 <= j < m:
                    lb = int(b_labels[j])
                    if lb >= 1 and lb != la:
                        self.add_edge(la, lb, max(fa, float(b_filled[j])))

    def solve(self) -> dict[int, float]:
        """Solve for each label's drainage elevation by minimax Priority-Flood from :data:`OUTLET`.

        ``drain[label]`` is the minimum over all paths to the outlet of the maximum edge (saddle) on the path —
        the elevation that label's watershed must be raised to.

        Returns:
            Mapping ``label -> drain_elevation``. :data:`OUTLET` maps to ``-inf``; labels with no path to the
            outlet are absent.

        Examples:
            >>> g = GlobalSpillGraph()
            >>> g.add_outlet(1, 5.0)      # label 1 spills out at 5
            >>> g.add_edge(1, 2, 8.0)     # label 2 reaches label 1 over a saddle at 8
            >>> drain = g.solve()
            >>> drain[1], drain[2]
            (5.0, 8.0)
        """
        adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (a, b), e in self.edges.items():
            adjacency[a].append((b, e))
            adjacency[b].append((a, e))

        drain: dict[int, float] = {OUTLET: -math.inf}
        heap: list[tuple[float, int]] = [(-math.inf, OUTLET)]
        visited: set[int] = set()
        while heap:
            level, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            drain[node] = level
            for neighbour, weight in adjacency[node]:
                if neighbour in visited:
                    continue
                new_level = weight if weight > level else level
                if new_level < drain.get(neighbour, math.inf):
                    drain[neighbour] = new_level
                    heapq.heappush(heap, (new_level, neighbour))
        return drain

    def _reduce_pairs(
        self, a_arr: np.ndarray, b_arr: np.ndarray, e_arr: np.ndarray
    ) -> None:
        # Group by the unordered label pair and keep the minimum spill, so the Python-level work is per *unique
        # pair* (perimeter-sized) rather than per adjacent cell (area-sized).
        if a_arr.size == 0:
            return
        lo = np.minimum(a_arr, b_arr)
        hi = np.maximum(a_arr, b_arr)
        order = np.lexsort((hi, lo))
        lo, hi, e_sorted = lo[order], hi[order], e_arr[order]
        starts = np.flatnonzero(
            np.concatenate(([True], (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])))
        )
        mins = np.minimum.reduceat(e_sorted, starts)
        for k, s in enumerate(starts):
            self.add_edge(int(lo[s]), int(hi[s]), float(mins[k]))


def stitch_seams(specs, by_grid, strips, graph) -> None:
    """Join every tile's perimeter labels to its neighbours' in the spill graph.

    Each tile is flooded in isolation, so a depression spanning a seam appears as
    two unrelated labels until they are joined here. Orthogonal neighbours share a
    whole edge and go through `join_strips`; the two diagonals share exactly one
    corner cell, which no strip join covers, so they get a single explicit edge at
    the higher of the two fill levels.

    Only the down-right and down-left diagonals are walked: every tile pair is
    reached once from the upper tile, so adding the mirrored pair would double the
    edges without changing the solve.

    Args:
        specs: Every tile, in any order.
        by_grid: `(row, col)` -> `TileSpec`, for neighbour lookup.
        strips: `tile id` -> `{"top"|"bottom"|"left"|"right": (labels, filled)}`,
            with labels already offset into the global numbering.
        graph: The `GlobalSpillGraph` to add the seam edges to. Mutated in place.
    """
    for s in specs:
        right = by_grid.get((s.row, s.col + 1))
        if right is not None:
            graph.join_strips(*strips[s.tid]["right"], *strips[right.tid]["left"])
        below = by_grid.get((s.row + 1, s.col))
        if below is not None:
            graph.join_strips(*strips[s.tid]["bottom"], *strips[below.tid]["top"])
        diag = by_grid.get((s.row + 1, s.col + 1))
        if diag is not None:
            a_lab, a_fil = strips[s.tid]["bottom"]
            d_lab, d_fil = strips[diag.tid]["top"]
            if a_lab[-1] >= 1 and d_lab[0] >= 1:
                graph.add_edge(
                    int(a_lab[-1]),
                    int(d_lab[0]),
                    max(float(a_fil[-1]), float(d_fil[0])),
                )
        anti = by_grid.get((s.row + 1, s.col - 1))
        if anti is not None:
            a_lab, a_fil = strips[s.tid]["bottom"]
            d_lab, d_fil = strips[anti.tid]["top"]
            if a_lab[0] >= 1 and d_lab[-1] >= 1:
                graph.add_edge(
                    int(a_lab[0]),
                    int(d_lab[-1]),
                    max(float(a_fil[0]), float(d_fil[-1])),
                )


def solve_drain_levels(graph, label_offset: int) -> np.ndarray:
    """Solve the spill graph and index the drain level of each label.

    An array rather than the solver's dict because the per-tile raise pass looks
    up one level per cell; `-inf` marks a label the solve returned nothing for,
    which the raise reads as "do not lift this cell".

    Args:
        graph: The stitched `GlobalSpillGraph`.
        label_offset: One past the highest global label issued.

    Returns:
        `float64` array of length `label_offset + 1`, indexed by label.
    """
    drainvec = np.full(label_offset + 1, -np.inf, dtype=np.float64)
    for label, level in graph.solve().items():
        if 1 <= label <= label_offset:
            drainvec[label] = level
    return drainvec
