"""Lightweight Mesh class + Laplacian smoothing (P33).

A minimal triangle-mesh container plus quality-improvement operations:

* :class:`Mesh` — vertex array ``(N, 2)`` or ``(N, 3)`` plus triangle
  index array ``(M, 3)``. Read-only after construction (smoothing
  returns a new instance).
* :meth:`Mesh.laplacian_smooth` — iterative Laplacian smoothing
  (Persson & Strang 2004). Each interior vertex moves toward the
  centroid of its 1-ring neighbours. Boundary vertices are held fixed.
* :meth:`Mesh.aspect_ratios` — per-triangle quality metric
  ``circumradius / (2 * inradius)``. Equilateral triangles score 1.0;
  degenerate triangles score arbitrarily large.

Use cases: post-process meshes from Phase 3 P26 exporters before
handing them to HEC-RAS / TUFLOW / SFINCS. The full P33 quality
optimisation also covers edge flips and refinement around breaklines;
those remain deferred.
"""
from __future__ import annotations

import numpy as np


class Mesh:
    """A triangle mesh with vertex and triangle index arrays.

    Args:
        vertices: ``(N, 2)`` or ``(N, 3)`` float64 array of vertex
            coordinates. 3-D inputs are kept as 3-D; smoothing operates
            on the XY plane and leaves Z unchanged.
        triangles: ``(M, 3)`` int array of vertex indices, CCW order.

    Attributes:
        vertices: ``(N, 2 or 3)`` float64.
        triangles: ``(M, 3)`` int64.
        n_vertices, n_triangles: counts.
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] not in (2, 3):
            raise ValueError(
                f"vertices must be (N, 2) or (N, 3); got {self.vertices.shape}"
            )
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must be (M, 3); got {self.triangles.shape}"
            )
        self.n_vertices = int(self.vertices.shape[0])
        self.n_triangles = int(self.triangles.shape[0])

    def boundary_vertex_mask(self) -> np.ndarray:
        """Boolean ``(n_vertices,)`` mask of boundary vertices.

        A vertex is on the boundary iff at least one of its incident
        edges belongs to only one triangle (the canonical mesh-boundary
        criterion).
        """
        edge_count: dict[tuple[int, int], int] = {}
        for tri in self.triangles:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                edge_count[key] = edge_count.get(key, 0) + 1
        out = np.zeros(self.n_vertices, dtype=bool)
        for (u, v), n in edge_count.items():
            if n == 1:
                out[u] = True
                out[v] = True
        return out

    def neighbour_lists(self) -> list[list[int]]:
        """Per-vertex list of neighbour vertex indices (1-ring)."""
        adj: list[set[int]] = [set() for _ in range(self.n_vertices)]
        for tri in self.triangles:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            adj[a].update((b, c))
            adj[b].update((a, c))
            adj[c].update((a, b))
        return [sorted(s) for s in adj]

    def laplacian_smooth(
        self,
        n_iterations: int = 10,
        relaxation: float = 0.5,
        hold_boundary: bool = True,
    ) -> "Mesh":
        """Iterative Laplacian smoothing.

        Each iteration moves every non-boundary vertex toward the
        centroid of its 1-ring neighbours by ``relaxation`` of the
        full step:

            v_new = v + relaxation * (centroid(neighbours) - v)

        Args:
            n_iterations: Number of smoothing passes.
            relaxation: Step size in ``[0, 1]``. ``1.0`` snaps every
                vertex onto its neighbour centroid each iteration;
                smaller values relax more gradually and avoid
                oscillation.
            hold_boundary: If True (default), boundary vertices are
                fixed. Set False only when the mesh is closed (no
                boundary).

        Returns:
            A new ``Mesh`` with smoothed vertex positions. Triangle
            connectivity is unchanged.
        """
        if not (0.0 <= relaxation <= 1.0):
            raise ValueError(
                f"relaxation must be in [0, 1]; got {relaxation}"
            )
        v = self.vertices.copy()
        adj = self.neighbour_lists()
        if hold_boundary:
            boundary = self.boundary_vertex_mask()
        else:
            boundary = np.zeros(self.n_vertices, dtype=bool)
        for _ in range(n_iterations):
            new_v = v.copy()
            for i in range(self.n_vertices):
                if boundary[i]:
                    continue
                neigh = adj[i]
                if not neigh:
                    continue
                centroid = v[neigh].mean(axis=0)
                new_v[i] = v[i] + relaxation * (centroid - v[i])
            v = new_v
        return Mesh(v, self.triangles)

    def aspect_ratios(self) -> np.ndarray:
        """Per-triangle aspect ratio ``circumradius / (2 * inradius)``.

        Equilateral triangles score 1.0 (the optimum). Higher values
        indicate worse quality. Degenerate triangles (zero area) score
        ``+inf``.

        Returns:
            ``(n_triangles,)`` float64 array.
        """
        v = self.vertices[:, :2]
        out = np.empty(self.n_triangles, dtype=np.float64)
        for i, tri in enumerate(self.triangles):
            a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
            la = np.linalg.norm(b - c)
            lb = np.linalg.norm(a - c)
            lc = np.linalg.norm(a - b)
            s = (la + lb + lc) / 2.0
            area = float(np.abs(
                (b[0] - a[0]) * (c[1] - a[1])
                - (c[0] - a[0]) * (b[1] - a[1])
            )) / 2.0
            if area == 0.0:
                out[i] = np.inf
                continue
            inradius = area / s
            circumradius = (la * lb * lc) / (4.0 * area)
            out[i] = circumradius / (2.0 * inradius)
        return out

    def __repr__(self) -> str:
        return (
            f"<Mesh vertices={self.n_vertices} "
            f"triangles={self.n_triangles}>"
        )
