"""Tests for `digitalrivers.interop.anudem.relax_gaps`.

Relaxation gap-fill has two properties worth pinning and one trap.

The properties: the known cells must come back untouched — a filler that drifts its own
anchors has corrupted the survey it was given — and the filled values must lie inside the
range of what surrounds them, since a membrane stretched across a hole cannot rise above
its rim.

The trap is the boundary condition. An earlier version used `np.roll`, which wraps the
array into a torus and feeds the far edge's elevations into the near edge, quietly
corrupting anchors along the border. The current code pads with edge replication, a
Neumann boundary. `test_boundary_does_not_wrap_around_the_raster` is what stops that
regressing.
"""

from __future__ import annotations

import numpy as np
import pytest

from digitalrivers.interop.anudem import relax_gaps


@pytest.fixture
def hole_5x5() -> np.ndarray:
    """A flat surface at 10.0 with a single `NaN` at the centre.

    Returns:
        np.ndarray: 5x5 float64 array, one hole at `(2, 2)`.
    """
    z = np.full((5, 5), 10.0)
    z[2, 2] = np.nan
    return z


class TestRelaxGaps:
    """Tests for relax_gaps."""

    def test_fills_a_hole_in_a_flat_surface_with_the_surrounding_value(self, hole_5x5):
        """A hole in a constant surface relaxes to that constant."""
        out = relax_gaps(hole_5x5)
        assert np.isclose(
            out[2, 2], 10.0
        ), f"Expected the hole to fill to 10.0, got {out[2, 2]}"

    def test_leaves_no_nan_behind(self, hole_5x5):
        """Every unknown cell is resolved to a finite value."""
        out = relax_gaps(hole_5x5)
        assert np.isfinite(out).all(), f"NaN survived the fill:\n{out}"

    def test_known_cells_are_returned_untouched(self):
        """Anchors do not drift. A filler that moves them has corrupted the survey."""
        z = np.arange(25, dtype=np.float64).reshape(5, 5)
        z[2, 2] = np.nan
        known = np.isfinite(z)
        out = relax_gaps(z)
        assert np.array_equal(
            out[known], z[known]
        ), "Relaxation altered cells that were supposed to be fixed"

    def test_filled_value_lies_within_the_surrounding_range(self):
        """A membrane across a hole cannot rise above its rim or sink below it."""
        z = np.full((5, 5), 5.0)
        z[0, :] = 20.0
        z[2, 2] = np.nan
        out = relax_gaps(z)
        assert (
            5.0 <= out[2, 2] <= 20.0
        ), f"Filled value {out[2, 2]} is outside the surrounding range [5, 20]"

    def test_boundary_does_not_wrap_around_the_raster(self):
        """Edge replication, not periodic wrap: far-side values must not leak in.

        Test scenario:
            The left column is 0 and the right column is 1000, with a hole beside the
            left edge. Under a periodic boundary the 1000s would wrap into the hole and
            pull it far above its neighbours. Under edge replication it stays near 0.
        """
        z = np.zeros((5, 5))
        z[:, -1] = 1000.0
        z[2, 0] = np.nan
        out = relax_gaps(z, max_iter=500, tol=1e-9)
        assert (
            out[2, 0] < 1.0
        ), f"Hole at the left edge reached {out[2, 0]}; the right edge wrapped in"

    def test_returns_float64_regardless_of_input_dtype(self):
        """Relaxation runs in float64 even when handed float32."""
        z = np.full((4, 4), 3.0, dtype=np.float32)
        z[1, 1] = np.nan
        out = relax_gaps(z)
        assert out.dtype == np.float64, f"Expected float64, got {out.dtype}"

    def test_shape_is_preserved(self):
        """The output grid matches the input grid cell for cell."""
        z = np.full((3, 7), 1.0)
        z[1, 3] = np.nan
        assert relax_gaps(z).shape == (3, 7), "Shape changed"

    def test_does_not_mutate_its_input(self):
        """The kernel copies before relaxing; the caller's array is untouched.

        Test scenario:
            Uses a surface whose values all differ, and compares the values as well as
            the NaN pattern. On a constant fixture, or comparing only where the NaNs
            are, a kernel that overwrote every cell in place would still pass.
        """
        z = np.arange(25, dtype=np.float64).reshape(5, 5)
        z[2, 2] = np.nan
        before = z.copy()
        relax_gaps(z)
        assert np.array_equal(
            np.isnan(z), np.isnan(before)
        ), "The NaN pattern of the input changed"
        assert np.array_equal(
            z[~np.isnan(z)], before[~np.isnan(before)]
        ), "Input values were modified in place"

    def test_surface_with_no_holes_is_returned_unchanged(self):
        """Nothing to fill means nothing to change."""
        z = np.arange(9, dtype=np.float64).reshape(3, 3)
        assert np.allclose(relax_gaps(z), z), "A complete surface was altered"

    @pytest.mark.parametrize("method", ["laplacian", "biharmonic"])
    def test_both_solvers_fill_the_hole(self, hole_5x5, method):
        """Both solvers resolve the hole; they differ in smoothness, not in coverage.

        Args:
            method: The solver under test.
        """
        out = relax_gaps(hole_5x5, method=method)
        assert np.isfinite(out).all(), f"{method} left a NaN behind"
        assert np.isclose(
            out[2, 2], 10.0, atol=1e-6
        ), f"{method} filled to {out[2, 2]}, expected 10.0"

    def test_biharmonic_differs_from_laplacian_on_a_curved_surface(self):
        """The two solvers are genuinely different, not aliases of one another.

        Test scenario:
            They agree on any surface a plane can describe, because a plane satisfies
            both the Laplace and the biharmonic equation — a linear ramp is not a
            discriminating fixture. Curvature is what separates them: the biharmonic
            solver matches the slope at the hole edge, the Laplacian only the value.
        """
        z = np.add.outer((np.arange(9) - 4.0) ** 2, (np.arange(9) - 4.0) ** 2)
        z[4, 4] = np.nan
        lap = relax_gaps(z, method="laplacian", max_iter=300, tol=1e-12)
        bih = relax_gaps(z, method="biharmonic", max_iter=300, tol=1e-12)
        assert not np.isclose(
            lap[4, 4], bih[4, 4]
        ), f"Both solvers filled the hole with {lap[4, 4]}; they should differ on a bowl"

    def test_both_solvers_agree_on_a_plane(self):
        """A plane satisfies both equations, so the two solvers must coincide there.

        Test scenario:
            This is the control for the test above: it shows that the difference seen on
            a curved surface comes from curvature and not from one solver being broken.
        """
        z = np.tile(np.arange(9, dtype=np.float64), (9, 1))
        z[4, 4] = np.nan
        lap = relax_gaps(z, method="laplacian", max_iter=300, tol=1e-12)
        bih = relax_gaps(z, method="biharmonic", max_iter=300, tol=1e-12)
        assert np.allclose(
            lap, bih
        ), "Solvers disagree on a plane, where both are exact"

    @pytest.mark.parametrize("hole_size", [1, 2, 3, 5, 9])
    def test_biharmonic_stays_within_the_surrounding_range(self, hole_size):
        """The biharmonic solver never leaves the range of its own boundary.

        Args:
            hole_size: Edge length of the square hole under test.

        Test scenario:
            This is the regression guard for a solver that used to diverge. The previous
            two-Laplacian approximation recomputed the Laplacian from its own diverging
            output each sweep, and for holes of 3x3 and up returned values around 1e68 —
            reaching the public API as ~2.8e38 through float32. Every size from 1 to 9 is
            checked, because the failure was size-dependent and 1x1 and 2x2 passed
            throughout.
        """
        z = np.tile(np.arange(13, dtype=np.float64), (13, 1))
        lo = 6 - hole_size // 2
        z[lo : lo + hole_size, lo : lo + hole_size] = np.nan
        out = relax_gaps(z, method="biharmonic", max_iter=2000, tol=1e-13)
        assert np.isfinite(
            out
        ).all(), f"{hole_size}x{hole_size} produced non-finite cells"
        assert (
            -1e-6 <= out.min() and out.max() <= 12.0 + 1e-6
        ), f"{hole_size}x{hole_size} left 0..12: [{out.min():.4g}, {out.max():.4g}]"

    @pytest.mark.parametrize("hole_size", [1, 3, 5])
    def test_biharmonic_reproduces_a_plane_exactly(self, hole_size):
        """A plane satisfies the biharmonic equation, so the fill has a known answer.

        Args:
            hole_size: Edge length of the square hole under test.

        Test scenario:
            Stronger than a range check: it pins accuracy, not just stability. Any surface
            a plane can describe must come back cell-for-cell, so a solver that is stable
            but wrong still fails here.
        """
        truth = np.tile(np.arange(13, dtype=np.float64), (13, 1))
        z = truth.copy()
        lo = 6 - hole_size // 2
        z[lo : lo + hole_size, lo : lo + hole_size] = np.nan
        out = relax_gaps(z, method="biharmonic", max_iter=2000, tol=1e-13)
        assert np.allclose(
            out, truth, atol=1e-6
        ), f"max error {np.max(np.abs(out - truth)):.4g} recovering a plane"

    def test_biharmonic_reproduces_a_quadratic_bowl(self):
        """A quadratic has a constant Laplacian, so it too is an exact solution."""
        axis = (np.arange(13) - 6.0) ** 2
        truth = np.add.outer(axis, axis)
        z = truth.copy()
        z[5:8, 5:8] = np.nan
        out = relax_gaps(z, method="biharmonic", max_iter=2000, tol=1e-13)
        assert np.allclose(
            out, truth, atol=1e-6
        ), f"max error {np.max(np.abs(out - truth)):.4g} recovering a bowl"

    def test_mask_marks_cells_to_preserve_not_cells_to_fill(self):
        """`mask=True` holds a cell fixed, so a masked `NaN` stays `NaN`.

        Test scenario:
            This is the opposite of the reading the name first suggests, and is what
            `DEM.anudem_interpolate` documents: the mask adds to the set of fixed cells
            rather than to the set of unknown ones.
        """
        z = np.array([[1.0, np.nan, 3.0]])
        out = relax_gaps(z, mask=np.array([[False, True, False]]))
        assert np.isnan(
            out[0, 1]
        ), f"A masked NaN should be held fixed, got {out[0, 1]}"

    def test_without_a_mask_the_same_hole_is_filled(self):
        """The contrast that makes the mask's meaning visible."""
        z = np.array([[1.0, np.nan, 3.0]])
        out = relax_gaps(z)
        assert np.isclose(out[0, 1], 2.0), f"Expected 2.0 by interpolation, got {out}"

    def test_max_iter_bounds_the_work(self):
        """A single sweep does not converge, which is how the cap is observable."""
        z = np.zeros((9, 9))
        z[:, 0] = 100.0
        z[1:-1, 1:-1] = np.nan
        one = relax_gaps(z, max_iter=1, tol=0.0)
        many = relax_gaps(z, max_iter=400, tol=1e-12)
        assert not np.allclose(one, many), "max_iter had no effect on the result"

    def test_loose_tolerance_stops_earlier_than_a_tight_one(self):
        """`tol` is the convergence gate; a loose one leaves the surface less settled."""
        z = np.zeros((9, 9))
        z[:, 0] = 100.0
        z[1:-1, 1:-1] = np.nan
        loose = relax_gaps(z, max_iter=500, tol=1.0)
        tight = relax_gaps(z, max_iter=500, tol=1e-10)
        assert not np.allclose(loose, tight), "tol had no effect on the result"

    def test_unknown_method_raises_value_error(self, hole_5x5):
        """An unrecognised solver fails immediately, naming what was given."""
        with pytest.raises(ValueError, match="method must be") as exc_info:
            relax_gaps(hole_5x5, method="bogus")
        assert "'bogus'" in str(
            exc_info.value
        ), f"Message omits the bad value: {exc_info.value}"

    def test_all_nan_input_raises_value_error(self):
        """With no anchor there is nothing to interpolate from."""
        with pytest.raises(ValueError, match="at least one finite anchor") as exc_info:
            relax_gaps(np.full((3, 3), np.nan))
        assert "anchor" in str(exc_info.value), str(exc_info.value)

    def test_mismatched_mask_shape_raises_value_error(self, hole_5x5):
        """A mask of the wrong shape fails on the broadcast, not silently.

        Test scenario:
            There is no explicit shape check; numpy raises `ValueError` from the
            broadcast. The type is what callers can rely on, so it is what is asserted.
        """
        with pytest.raises(ValueError):
            relax_gaps(hole_5x5, mask=np.zeros((2, 2), dtype=bool))

    def test_is_deterministic(self, hole_5x5):
        """Two runs over the same input produce the same surface."""
        a = relax_gaps(hole_5x5)
        b = relax_gaps(hole_5x5)
        assert np.array_equal(a, b), "Repeated calls disagree"
