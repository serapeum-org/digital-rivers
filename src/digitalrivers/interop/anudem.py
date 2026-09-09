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

`relax_gaps` is the array kernel; `DEM.anudem_interpolate` wraps it and records the
operation in the DEM's conditioning history.
"""

from __future__ import annotations

import numpy as np

__all__ = ["relax_gaps"]


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

    rows, cols = elev.shape
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
        # Alternate two Laplacian sweeps to approximate Δ²z = 0.
        # Step A: compute u = Δz on the current z.
        # Step B: relax z so Δz ≈ smoothed u (mean of neighbour u's).
        # Composed, this approximates a biharmonic relaxation with C¹
        # continuity at the anchors.
        for _ in range(max_iter):
            north, south, east, west = _edge_shifts(z)
            u = north + south + east + west - 4.0 * z
            un, us, ue, uw = _edge_shifts(u)
            u_smooth = (un + us + ue + uw) / 4.0
            # Solve Δz = u_smooth → new z[i,j] =
            # (sum of neighbours - u_smooth) / 4.
            new_z = (north + south + east + west - u_smooth) / 4.0
            new_z[fixed] = z[fixed]
            diff = float(np.max(np.abs(new_z - z)))
            z = new_z
            if diff < tol:
                break

    return z
