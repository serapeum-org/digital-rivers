"""Lightweight Mesh class + Laplacian smoothing (P33).

A minimal triangle-mesh container plus quality-improvement operations:

* :class:`Mesh` — vertex array `(N, 2)` or `(N, 3)` plus triangle
  index array `(M, 3)`. Read-only after construction (smoothing
  returns a new instance).
* :meth:`Mesh.laplacian_smooth` — iterative Laplacian smoothing
  (Persson & Strang 2004). Each interior vertex moves toward the
  centroid of its 1-ring neighbours. Boundary vertices are held fixed.
* :meth:`Mesh.aspect_ratios` — per-triangle quality metric
  `circumradius / (2 * inradius)`. Equilateral triangles score 1.0;
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

    Performance note. `boundary_vertex_mask`, `neighbour_lists` and
    `aspect_ratios` iterate triangles in pure Python — fine for
    small / medium meshes (<~50k triangles). Above that, prefer a vendor
    library (`meshio` / `pymesh`) or vectorise the kernels.

    Args:
        vertices: `(N, 2)` or `(N, 3)` float64 array of vertex
            coordinates. 3-D inputs are kept as 3-D, and `laplacian_smooth`
            moves points in every column it is given — including Z. Smooth a
            topobathy mesh and the elevations are smoothed with it.
        triangles: `(M, 3)` int array of vertex indices, CCW order.

    Attributes:
        vertices: `(N, 2 or 3)` float64.
        triangles: `(M, 3)` int64.
        n_vertices, n_triangles: counts.

    Examples:
        - Build a two-triangle quad and inspect its size:

            >>> import numpy as np
            >>> from digitalrivers.interop.mesh import Mesh
            >>> verts = np.array(
            ...     [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ...     dtype=np.float64,
            ... )
            >>> tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
            >>> mesh = Mesh(verts, tris)
            >>> mesh.n_vertices
            4
            >>> mesh.n_triangles
            2

        - 3-D input keeps Z untouched:

            >>> import numpy as np
            >>> from digitalrivers.interop.mesh import Mesh
            >>> v = np.array(
            ...     [[0.0, 0.0, 10.0], [1.0, 0.0, 11.0], [0.0, 1.0, 12.0]],
            ...     dtype=np.float64,
            ... )
            >>> t = np.array([[0, 1, 2]], dtype=np.int64)
            >>> Mesh(v, t).vertices.shape
            (3, 3)

    See Also:
        Mesh.laplacian_smooth: iterative quality-improvement smoothing.
        Mesh.aspect_ratios: per-triangle quality metric.
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        """Build a mesh from vertex coordinates and triangle indices.

        Vertices become `float64` and triangles `int64`, which is what the quality
        kernels index with, and a list of lists works as well as an ndarray.

        The conversion is `np.asarray`, which does **not** copy when the input already
        has the right dtype, so a `float64` array passed in is adopted rather than
        copied and `mesh.vertices is your_array` holds. Writing through either one is
        visible in the other. `laplacian_smooth` copies before it writes, so it is safe
        today, but it returns a mesh sharing your triangle array.

        Args:
            vertices: `(N, 2)` or `(N, 3)` array of vertex coordinates. A 3-D input
                stays 3-D, and smoothing then moves Z along with X and Y.
            triangles: `(M, 3)` array of vertex indices, counter-clockwise.

        Raises:
            ValueError: If `vertices` is not 2-D with 2 or 3 columns, or `triangles` is
                not 2-D with exactly 3 columns. The shape is named in the message.

        Examples:
            - Build a single triangle and read back its counts:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> mesh = Mesh(
                ...     np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                ...     np.array([[0, 1, 2]]),
                ... )
                >>> mesh.n_vertices, mesh.n_triangles
                (3, 1)

            - Coordinates are promoted to float64 whatever came in:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> mesh = Mesh(
                ...     np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32),
                ...     np.array([[0, 1, 2]]),
                ... )
                >>> mesh.vertices.dtype
                dtype('float64')

            - A quad of vertices is not a triangle array:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> Mesh(np.zeros((4, 2)), np.array([[0, 1, 2, 3]]))
                Traceback (most recent call last):
                    ...
                ValueError: triangles must be (M, 3); got (1, 4)
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] not in (2, 3):
            raise ValueError(
                f"vertices must be (N, 2) or (N, 3); got {self.vertices.shape}"
            )
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must be (M, 3); got {self.triangles.shape}")
        self.n_vertices = int(self.vertices.shape[0])
        self.n_triangles = int(self.triangles.shape[0])

    def boundary_vertex_mask(self) -> np.ndarray:
        """Boolean `(n_vertices,)` mask of boundary vertices.

        A vertex is on the boundary iff at least one of its incident
        edges belongs to only one triangle (the canonical mesh-boundary
        criterion).

        Examples:
            - Every vertex of a two-triangle quad sits on the boundary:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> v = np.array(
                ...     [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                ...     dtype=np.float64,
                ... )
                >>> t = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
                >>> mask = Mesh(v, t).boundary_vertex_mask()
                >>> bool(mask.all())
                True

            - Adding a centre vertex moves only the four corners to the
              boundary:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> v = np.array(
                ...     [
                ...         [0.0, 0.0], [2.0, 0.0], [2.0, 2.0],
                ...         [0.0, 2.0], [1.0, 1.0],
                ...     ],
                ...     dtype=np.float64,
                ... )
                >>> t = np.array(
                ...     [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
                ...     dtype=np.int64,
                ... )
                >>> mask = Mesh(v, t).boundary_vertex_mask()
                >>> mask.tolist()
                [True, True, True, True, False]
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
        """Per-vertex list of neighbour vertex indices (1-ring).

        Examples:
            - Inspect the shared diagonal of a two-triangle quad:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> v = np.array(
                ...     [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                ...     dtype=np.float64,
                ... )
                >>> t = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
                >>> adj = Mesh(v, t).neighbour_lists()
                >>> adj[0]
                [1, 2, 3]
                >>> adj[1]
                [0, 2]
        """
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
        centroid of its 1-ring neighbours by `relaxation` of the
        full step:

            v_new = v + relaxation * (centroid(neighbours) - v)

        The centroid is taken over every column of the vertex array, so on a 3-D mesh
        this smooths Z as well as X and Y. That is ordinary 3-D Laplacian smoothing, but
        it means the method is not safe to run on a mesh whose Z carries elevations you
        need preserved — it will flatten them toward their neighbourhood mean.

        Args:
            n_iterations: Number of smoothing passes.
            relaxation: Step size in `[0, 1]`. `1.0` snaps every
                vertex onto its neighbour centroid each iteration;
                smaller values relax more gradually and avoid
                oscillation.
            hold_boundary: If True (default), boundary vertices are
                fixed. Set False only when the mesh is closed (no
                boundary).

        Returns:
            A new `Mesh` with smoothed vertex positions. Triangle
            connectivity is unchanged.

        Raises:
            ValueError: If `relaxation` is not in `[0, 1]`.

        Examples:
            - Smooth an off-centre interior vertex toward the centroid of
              its four corner neighbours; the corners stay pinned:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> v = np.array(
                ...     [
                ...         [0.0, 0.0], [2.0, 0.0], [2.0, 2.0],
                ...         [0.0, 2.0], [1.5, 1.5],
                ...     ],
                ...     dtype=np.float64,
                ... )
                >>> t = np.array(
                ...     [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
                ...     dtype=np.int64,
                ... )
                >>> smoothed = Mesh(v, t).laplacian_smooth(
                ...     n_iterations=20, relaxation=1.0,
                ... )
                >>> [round(float(c), 6) for c in smoothed.vertices[4]]
                [1.0, 1.0]
                >>> smoothed.vertices[0].tolist()
                [0.0, 0.0]
        """
        if not (0.0 <= relaxation <= 1.0):
            raise ValueError(f"relaxation must be in [0, 1]; got {relaxation}")
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
        """Per-triangle aspect ratio `circumradius / (2 * inradius)`.

        Equilateral triangles score 1.0 (the optimum). Higher values
        indicate worse quality. Degenerate triangles (zero area) score
        `+inf`.

        Returns:
            `(n_triangles,)` float64 array.

        Examples:
            - An equilateral triangle scores exactly 1.0:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> h = np.sqrt(3.0) / 2.0
                >>> v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]], dtype=np.float64)
                >>> t = np.array([[0, 1, 2]], dtype=np.int64)
                >>> round(float(Mesh(v, t).aspect_ratios()[0]), 6)
                1.0

            - A 3-4-5 right triangle has aspect ratio 1.25:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> v = np.array(
                ...     [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float64,
                ... )
                >>> t = np.array([[0, 1, 2]], dtype=np.int64)
                >>> round(float(Mesh(v, t).aspect_ratios()[0]), 6)
                1.25

            - Three collinear points give a degenerate (infinite) ratio:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> v = np.array(
                ...     [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64,
                ... )
                >>> t = np.array([[0, 1, 2]], dtype=np.int64)
                >>> float(Mesh(v, t).aspect_ratios()[0])
                inf
        """
        v = self.vertices[:, :2]
        out = np.empty(self.n_triangles, dtype=np.float64)
        for i, tri in enumerate(self.triangles):
            a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
            la = np.linalg.norm(b - c)
            lb = np.linalg.norm(a - c)
            lc = np.linalg.norm(a - b)
            s = (la + lb + lc) / 2.0
            area = (
                float(
                    np.abs(
                        (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
                    )
                )
                / 2.0
            )
            # `area` is an absolute value, so `<= 0` is the degenerate case and
            # avoids an equality test on a computed float.
            if area <= 0.0:
                out[i] = np.inf
                continue
            inradius = area / s
            circumradius = (la * lb * lc) / (4.0 * area)
            out[i] = circumradius / (2.0 * inradius)
        return out

    def __repr__(self) -> str:
        """Return a one-line summary naming the mesh's size.

        Returns:
            A string of the form `<Mesh vertices=N triangles=M>`.

        Examples:
            - The repr reports both counts:

                >>> import numpy as np
                >>> from digitalrivers.interop.mesh import Mesh
                >>> Mesh(
                ...     np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                ...     np.array([[0, 1, 2]]),
                ... )
                <Mesh vertices=3 triangles=1>
        """
        return f"<Mesh vertices={self.n_vertices} " f"triangles={self.n_triangles}>"
