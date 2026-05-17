# Mesh

Triangle-mesh container with Laplacian smoothing and aspect-ratio quality metrics. Introduced as
Phase-4 backfill P33 to support mesh-output workflows (TIN exports, gmsh `.geo` round-trips, FEM
preprocessing).

Top-level surface:

* **`boundary_vertex_mask`** — bool mask flagging boundary vertices (used as fixed anchors in
  smoothing).
* **`neighbour_lists`** — per-vertex adjacency from the triangle index list.
* **`laplacian_smooth(n_iterations=..., relaxation=..., hold_boundary=True)`** — Persson & Strang
  2004 Laplacian smoothing; boundary vertices are pinned by default.
* **`aspect_ratios()`** — per-triangle aspect-ratio quality metric.

::: digitalrivers.mesh.Mesh
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
