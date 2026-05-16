"""Tests for ``digitalrivers.mesh.Mesh`` (P33 backfill)."""
from __future__ import annotations

import numpy as np
import pytest

from digitalrivers.mesh import Mesh


@pytest.fixture
def two_triangle_quad() -> Mesh:
    """A unit square made of two triangles (4 vertices, 2 triangles)."""
    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return Mesh(vertices, triangles)


def test_init_records_counts(two_triangle_quad):
    assert two_triangle_quad.n_vertices == 4
    assert two_triangle_quad.n_triangles == 2


def test_init_rejects_bad_vertices_shape():
    bad = np.zeros((3, 4), dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="vertices must be"):
        Mesh(bad, tris)


def test_init_rejects_bad_triangles_shape():
    verts = np.zeros((3, 2), dtype=np.float64)
    bad = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="triangles must be"):
        Mesh(verts, bad)


def test_boundary_vertex_mask_all_corners(two_triangle_quad):
    mask = two_triangle_quad.boundary_vertex_mask()
    # In a 2-triangle quad, every vertex is on the boundary.
    assert mask.all()


def test_neighbour_lists_share_diagonal(two_triangle_quad):
    adj = two_triangle_quad.neighbour_lists()
    # Diagonal (0-2) is shared by both triangles.
    assert 2 in adj[0]
    assert 0 in adj[2]
    # Outer corners are connected to their two edge neighbours.
    assert set(adj[1]) == {0, 2}
    assert set(adj[3]) == {0, 2}


def test_laplacian_smooth_holds_boundary(two_triangle_quad):
    smoothed = two_triangle_quad.laplacian_smooth(
        n_iterations=5, relaxation=0.5, hold_boundary=True
    )
    # All vertices are boundary in this quad → unchanged.
    np.testing.assert_allclose(smoothed.vertices, two_triangle_quad.vertices)


def test_laplacian_smooth_moves_interior():
    # Build a mesh where vertex 4 is interior, surrounded by 4 corner verts.
    vertices = np.array(
        [
            [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0],
            [1.5, 1.5],  # interior — off-centre
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int64
    )
    mesh = Mesh(vertices, triangles)
    smoothed = mesh.laplacian_smooth(
        n_iterations=20, relaxation=1.0, hold_boundary=True
    )
    # Interior vertex should converge toward the centroid of its neighbours
    # (1.0, 1.0).
    np.testing.assert_allclose(smoothed.vertices[4], [1.0, 1.0], atol=1e-6)
    # Corners unchanged.
    np.testing.assert_allclose(smoothed.vertices[:4], vertices[:4])


def test_laplacian_smooth_relaxation_validated(two_triangle_quad):
    with pytest.raises(ValueError, match="relaxation"):
        two_triangle_quad.laplacian_smooth(relaxation=-0.1)
    with pytest.raises(ValueError, match="relaxation"):
        two_triangle_quad.laplacian_smooth(relaxation=1.5)


def test_aspect_ratios_equilateral_is_unity():
    # Build a perfect equilateral triangle.
    h = np.sqrt(3.0) / 2.0
    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, h]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = Mesh(vertices, triangles)
    ratios = mesh.aspect_ratios()
    np.testing.assert_allclose(ratios, [1.0], atol=1e-9)


def test_aspect_ratios_degenerate_is_infinite():
    # Three collinear points form a zero-area triangle.
    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = Mesh(vertices, triangles)
    ratios = mesh.aspect_ratios()
    assert np.isinf(ratios[0])


def test_repr_summarises(two_triangle_quad):
    rep = repr(two_triangle_quad)
    assert "vertices=4" in rep
    assert "triangles=2" in rep
