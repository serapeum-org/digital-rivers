"""ANUDEM-lite: relaxation gap-fill for a surface with holes.

A pragmatic subset of Hutchinson (1989) covering the case that actually comes up — a DEM
with `NaN` holes from cloud shadow, survey gaps or vegetation occlusion — rather than
the full drainage-enforcing thin-plate spline.

Two solvers, both Gauss-Seidel sweeps holding the known cells fixed:

* `"laplacian"` solves the 4-neighbour mean, the discrete form of a membrane stretched
  across the hole. Fast, and it cannot overshoot, but it meets the known surface with a
  visible kink: it matches values at the boundary without matching slopes.
* `"biharmonic"` solves the plate equation instead, matching slope as well as value at
  the hole edge, so the seam disappears. Slower, and it can overshoot slightly.
  Implemented as damped Jacobi on the 13-point stencil; an earlier two-Laplacian
  approximation diverged for holes of 3x3 and larger.

`relax_gaps` is the array kernel; `DEM.anudem_interpolate` wraps it and records the
operation in the DEM's conditioning history.
"""

from __future__ import annotations

import numpy as np

__all__ = ["relax_gaps"]


#: Under-relaxation for the biharmonic sweep. The 13-point stencil is not diagonally
#: dominant, so plain Jacobi (1.0) diverges; measured against a plane, anything above
#: 0.6 blows up on a hole of 5x5 or larger and 0.5 is stable to at least 15x15.
_BIHARMONIC_RELAXATION = 0.5


def _biharmonic_neighbours(arr: np.ndarray):
    """Return the twelve neighbours the 13-point biharmonic stencil reads.

    Padded by two with edge replication, matching the Neumann boundary the Laplacian
    branch uses — a periodic wrap here would fold the far edge of the raster into cells
    near the near edge.

    Args:
        arr: 2-D array to take neighbourhoods of.

    Returns:
        `(n, s, e, w, ne, nw, se, sw, nn, ss, ee, ww)`, each the same shape as `arr`.
    """
    p = np.pad(arr, 2, mode="edge")
    return (
        p[1:-3, 2:-2],
        p[3:-1, 2:-2],
        p[2:-2, 3:-1],
        p[2:-2, 1:-3],
        p[1:-3, 3:-1],
        p[1:-3, 1:-3],
        p[3:-1, 3:-1],
        p[3:-1, 1:-3],
        p[0:-4, 2:-2],
        p[4:, 2:-2],
        p[2:-2, 4:],
        p[2:-2, 0:-4],
    )


def relax_gaps(
    elev: np.ndarray,
    mask=None,
    max_iter: int = 200,
    tol: float = 1e-3,
    method: str = "laplacian",
) -> np.ndarray:
    """Fill the unknown cells of `elev` by relaxation, holding the known ones fixed.

    Args:
        elev: 2-D elevation array; `NaN` marks the cells to fill.
        mask: Optional bool array of the same shape. `True` marks a cell as one to
            *preserve*, in addition to the finite cells that are held fixed anyway. A
            `NaN` cell marked `True` is therefore left `NaN` rather than filled.
            Defaults to `None`, which holds every finite cell fixed and fills the rest.
        max_iter: Maximum relaxation sweeps. Defaults to 200.
        tol: Stop once the largest change in a sweep falls below this. Defaults to 1e-3.
        method: `"laplacian"` (default) or `"biharmonic"`.

    Returns:
        `float64` array of the same shape, with the unknown cells filled.

    Raises:
        ValueError: For an unknown `method`, or an input with no cell held fixed to
            interpolate from. A `mask` whose shape does not match also raises
            `ValueError`, from the broadcast rather than from an explicit check.
    """
    if method not in ("laplacian", "biharmonic"):
        raise ValueError(f"method must be 'laplacian' or 'biharmonic'; got {method!r}")

    z = elev.astype(np.float64, copy=True)
    fixed = np.isfinite(z)
    if mask is not None:
        fixed = fixed | mask.astype(bool, copy=False)
    if not fixed.any():
        raise ValueError("anudem_interpolate needs at least one finite anchor cell")
    # Seed unknown cells to the mean of known values to speed convergence.
    z = np.where(fixed, z, z[fixed].mean())

    def _edge_shifts(arr):
        """Return (north, south, east, west) views of `arr` with
        edge-replication boundary handling (no periodic wrap).

        Using `np.pad(..., mode="edge")` matches a Neumann (zero
        normal-derivative) boundary, which is the natural choice for
        an interpolation kernel — the original `np.roll` formed a
        torus and injected far-edge values into near-edge cells,
        corrupting anchors near the DEM boundary.
        """
        padded = np.pad(arr, 1, mode="edge")
        return (
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, 2:],
            padded[1:-1, :-2],
        )

    if method == "laplacian":
        for _ in range(max_iter):
            north, south, east, west = _edge_shifts(z)
            new_z = (north + south + east + west) / 4.0
            new_z[fixed] = z[fixed]
            diff = float(np.max(np.abs(new_z - z)))
            z = new_z
            if diff < tol:
                break
    else:  # biharmonic
        # Damped Jacobi on the 13-point biharmonic stencil, the standard
        # discretisation of the plate equation:
        #
        #     20 z - 8(N+S+E+W) + 2(NE+NW+SE+SW) + (NN+SS+EE+WW) = 0
        #
        # The stencil is not diagonally dominant (20 against 44), so undamped
        # Jacobi does not converge on it. `_BIHARMONIC_RELAXATION` is what makes
        # the iteration contract; it was chosen by sweeping the factor against a
        # plane, which is an exact biharmonic solution and therefore has a known
        # answer. Anything above 0.6 diverges on a hole of 5x5 or larger.
        for _ in range(max_iter):
            n, s_, e, w, ne, nw, se, sw, nn, ss, ee, ww = _biharmonic_neighbours(z)
            target = (
                8.0 * (n + s_ + e + w) - 2.0 * (ne + nw + se + sw) - (nn + ss + ee + ww)
            ) / 20.0
            new_z = z + _BIHARMONIC_RELAXATION * (target - z)
            new_z[fixed] = z[fixed]
            diff = float(np.max(np.abs(new_z - z)))
            z = new_z
            if diff < tol:
                break

    return z
