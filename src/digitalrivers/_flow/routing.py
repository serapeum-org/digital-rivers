"""Generalised flow-direction algorithms for DEM hydro pre-processing (P5).

Four routing schemes alongside the existing D8:

* `"d8"` (already in dem.py) — single-direction steepest descent.
* `"dinf"` — Tarboton (1997). 8 triangular facets per cell; output is a 2-band raster
  (angle in radians CCW from east, slope magnitude). Aspect is split between two
  neighbours proportional to the within-facet angle.
* `"mfd_quinn"` — Quinn et al. (1991). Multi-direction; distributes flow to every
  downslope neighbour proportional to `s_k * L_k` where `L_k` is the contour-length
  factor (0.5 for cardinals, 0.354 for diagonals). Output: 8-band float32 stack.
* `"mfd_holmgren"` — Holmgren (1994). Same family, no contour-length weighting,
  tunable exponent `p`; high `p` (4–6) mimics D8, `p=1` mimics Quinn.
* `"rho8"` — Fairfield & Leymarie (1991). Stochastic single-direction; cardinal slopes
  are perturbed by `/(2 - U)` where `U ~ Uniform(0, 1)`, then steepest is picked.

Output direction codes follow the `DIR_OFFSETS` convention from `dem.py`:
`0=S, 1=SW, 2=W, 3=NW, 4=N, 5=NE, 6=E, 7=SE`. All multi-band outputs index axis 0
(bands) in that order for MFD, and `(angle, magnitude)` for D∞.
"""

from __future__ import annotations

import numpy as np

# DIR_OFFSETS-aligned (dr, dc) for each direction index.
# Direction codes (matching dem.py's DIR_OFFSETS):
#   0=S, 1=SW, 2=W, 3=NW, 4=N, 5=NE, 6=E, 7=SE
_DIR_DR = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int8)
_DIR_DC = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int8)

# Cardinal indices (S=0, W=2, N=4, E=6) — slope divisor is cell_size.
# Diagonal indices (SW=1, NW=3, NE=5, SE=7) — slope divisor is cell_size * sqrt(2).
_IS_CARDINAL = np.array([True, False, True, False, True, False, True, False])


def _dinf_facet_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tarboton facet adjacency tables for the 8 D∞ facets.

    Each facet K is defined by a (cardinal_dir_idx, diagonal_dir_idx) pair and a
    `(ac, af)` rule for converting the within-facet angle `r ∈ [0, π/4]` into the
    global aspect angle `ac * π/2 + af * r` (radians CCW from east, mod 2π).

    Returns:
        Tuple of four (8,) arrays:
            cardinal_idx[K]: direction code (0–7) of the facet's cardinal neighbour
            diagonal_idx[K]: direction code (0–7) of the facet's diagonal neighbour
            ac[K]: axis index 0–4 (E=0, N=1, W=2, S=3, E_wrap=4)
            af[K]: rotation sign ±1
    """
    # Facet K -> (cardinal, diagonal, ac, af) — matches TauDEM's 1-indexed tables.
    # Facet 1: E (cardinal=6) + NE (diagonal=5),  ac=0, af=+1
    # Facet 2: N (cardinal=4) + NE (diagonal=5),  ac=1, af=-1
    # Facet 3: N (cardinal=4) + NW (diagonal=3),  ac=1, af=+1
    # Facet 4: W (cardinal=2) + NW (diagonal=3),  ac=2, af=-1
    # Facet 5: W (cardinal=2) + SW (diagonal=1),  ac=2, af=+1
    # Facet 6: S (cardinal=0) + SW (diagonal=1),  ac=3, af=-1
    # Facet 7: S (cardinal=0) + SE (diagonal=7),  ac=3, af=+1
    # Facet 8: E (cardinal=6) + SE (diagonal=7),  ac=4, af=-1
    cardinal_idx = np.array([6, 4, 4, 2, 2, 0, 0, 6], dtype=np.int8)
    diagonal_idx = np.array([5, 5, 3, 3, 1, 1, 7, 7], dtype=np.int8)
    ac = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int8)
    af = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)
    return cardinal_idx, diagonal_idx, ac, af


def _padded_elevation(z: np.ndarray) -> np.ndarray:
    """Return a NaN-padded copy of `z` so 8-neighbour lookups never go out of bounds.

    Padding rows and columns hold `NaN`; any computation using a NaN neighbour
    propagates `NaN` through arithmetic and is filtered out by `nanargmax` /
    `np.where` checks downstream.
    """
    rows, cols = z.shape
    padded = np.full((rows + 2, cols + 2), np.nan, dtype=np.float64)
    padded[1:-1, 1:-1] = z
    return padded


def dinf_flow_direction(
    z: np.ndarray, cell_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Tarboton D∞ flow direction.

    Args:
        z: 2-D float elevation array. NaN cells are treated as no-data.
        cell_size: square cell side length in map units.

    Returns:
        Tuple `(angle, magnitude)`:
            angle: `(rows, cols)` float32, aspect in radians CCW from east in
                `[0, 2π)`. `-1.0` marks cells with no downhill flow (sinks,
                no-data, all-flat neighbourhoods).
            magnitude: `(rows, cols)` float32, slope magnitude along the chosen
                facet. `0.0` where `angle == -1`.

    Examples:
        - A planar surface tilted so water flows east (`Z = -x`) gives an
          aspect angle ≈ 0 (CCW from east) on interior cells:

            >>> import numpy as np
            >>> z = -np.arange(5, dtype=float)[None, :].repeat(5, axis=0)
            >>> angle, magnitude = dinf_flow_direction(z, cell_size=1.0)
            >>> bool(abs(angle[2, 2]) < 0.05)
            True
            >>> bool(magnitude[2, 2] > 0.9)
            True
    """
    rows, cols = z.shape
    padded = _padded_elevation(z)
    e0 = padded[1:-1, 1:-1]

    d1 = cell_size
    d2 = cell_size  # cardinal-to-adjacent-diagonal distance is one cell side
    diag_dist = cell_size * np.sqrt(2)

    # Slice-shift to get cardinal and diagonal neighbour elevation arrays per direction.
    # Indexing follows the DIR_OFFSETS convention.
    e_neigh: list[np.ndarray] = []
    for d in range(8):
        dr = int(_DIR_DR[d])
        dc = int(_DIR_DC[d])
        e_neigh.append(padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols])

    cardinal_idx, diagonal_idx, ac, af = _dinf_facet_tables()

    facet_s = np.full((rows, cols, 8), -np.inf, dtype=np.float64)
    facet_r = np.zeros((rows, cols, 8), dtype=np.float64)

    pi_over_4 = np.pi / 4.0

    for k in range(8):
        e1 = e_neigh[cardinal_idx[k]]
        e2 = e_neigh[diagonal_idx[k]]
        s1 = (e0 - e1) / d1
        s2 = (e1 - e2) / d2
        # Within-facet angle in radians.
        r = np.arctan2(s2, s1)
        s = np.sqrt(s1 * s1 + s2 * s2)
        # Clamp to [0, π/4]; out-of-range falls back to facet-boundary slopes.
        clamp_low = r < 0
        r = np.where(clamp_low, 0.0, r)
        s = np.where(clamp_low, s1, s)
        clamp_high = r > pi_over_4
        r = np.where(clamp_high, pi_over_4, r)
        s_high = (e0 - e2) / diag_dist
        s = np.where(clamp_high, s_high, s)
        # Cells whose facet has non-positive max slope are not downhill on this facet.
        s = np.where(s > 0, s, -np.inf)
        facet_s[:, :, k] = s
        facet_r[:, :, k] = r

    # NaN handling: any facet whose s came out NaN (because e0 or e1 or e2 was NaN)
    # is excluded from the argmax via the -inf substitution above only if isnan was
    # tested; arctan2/sqrt propagate NaN, so we explicitly filter.
    facet_s = np.where(np.isnan(facet_s), -np.inf, facet_s)

    # Best facet per cell.
    best_facet = np.argmax(facet_s, axis=2)
    rr, cc = np.indices((rows, cols))
    best_s = facet_s[rr, cc, best_facet]
    best_r = facet_r[rr, cc, best_facet]

    # Reconstruct global angle: ac * π/2 + af * r.
    global_angle = ac[best_facet] * (np.pi / 2.0) + af[best_facet] * best_r
    global_angle = np.mod(global_angle, 2 * np.pi)

    valid = np.isfinite(best_s) & (best_s > 0) & ~np.isnan(e0)
    angle_out = np.where(valid, global_angle, -1.0).astype(np.float32)
    magnitude_out = np.where(valid, best_s, 0.0).astype(np.float32)
    return angle_out, magnitude_out


def mfd_flow_direction(
    slopes: np.ndarray,
    elev_mask: np.ndarray,
    *,
    weighting: str,
    exponent: float,
) -> np.ndarray:
    """Multi-flow direction (Quinn 1991 / Holmgren 1994 / Freeman 1991).

    Args:
        slopes: `(rows, cols, 8)` float32 slopes-to-neighbour, ordered by
            `DIR_OFFSETS`. NaN slopes represent boundary/no-data neighbours.
        elev_mask: `(rows, cols)` bool — True where the centre cell has a valid
            elevation (not NaN / not no-data).
        weighting: `"quinn"` applies contour-length weights (0.5 cardinal,
            0.354 diagonal). `"holmgren"` uses raw exponent (no length factor).
        exponent: `p` in the weight formula. Quinn defaults to 1.0; Holmgren
            typical 4–10 (high p mimics D8, p=1 mimics Quinn).

    Returns:
        `(rows, cols, 8)` float32 fraction stack. Each cell's fractions sum to 1.0
        if any downslope neighbour exists, else all zero (sink / no-flow cell).
    """
    rows, cols, _ = slopes.shape
    # Mask: only positive (downslope) slopes contribute.
    pos = np.where(np.isnan(slopes), 0.0, slopes)
    pos = np.where(pos > 0, pos, 0.0)

    if weighting == "quinn":
        # Cardinal contour length = 0.5; diagonal ≈ 1/(2√2) ≈ 0.3535.
        contour = np.where(
            _IS_CARDINAL[np.newaxis, np.newaxis, :], 0.5, 1.0 / (2.0 * np.sqrt(2.0))
        )
        weights = (pos**exponent) * contour
    elif weighting == "holmgren":
        weights = pos**exponent
    else:
        raise ValueError(f"weighting must be 'quinn' or 'holmgren'; got {weighting!r}")

    denom = weights.sum(axis=2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        fractions = np.where(denom > 0, weights / denom, 0.0)
    # Cells whose centre is invalid get all-zero fractions.
    fractions = np.where(elev_mask[:, :, np.newaxis], fractions, 0.0)
    return fractions.astype(np.float32)


def rho8_flow_direction(
    slopes: np.ndarray,
    elev_mask: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rho8 stochastic single-direction (Fairfield & Leymarie 1991).

    Cardinal slopes are divided by `2 - U` where `U ~ Uniform(0, 1)` per cell; this
    randomly weighs cardinal vs diagonal preference so that, integrated over many
    realisations, the expected flow direction matches the gradient aspect.

    Args:
        slopes: `(rows, cols, 8)` float32 slopes-to-neighbour.
        elev_mask: `(rows, cols)` bool, True for valid-elevation cells.
        rng: NumPy `Generator`; pass `np.random.default_rng(seed)` for
            reproducibility.

    Returns:
        `(rows, cols)` int32 direction code (0–7) or the no-data sentinel for cells
        with no downslope neighbour / invalid elevation.
    """
    rows, cols, _ = slopes.shape
    # Generate U for cardinal directions only; diagonals stay unscaled.
    u = rng.uniform(0.0, 1.0, size=(rows, cols, 8)).astype(np.float32)
    divisors = np.where(_IS_CARDINAL[np.newaxis, np.newaxis, :], 2.0 - u, 1.0)
    perturbed = slopes / divisors
    # Mask invalid (NaN or <= 0) so they cannot be argmax.
    perturbed = np.where(np.isnan(perturbed), -np.inf, perturbed)
    perturbed = np.where(perturbed > 0, perturbed, -np.inf)
    best = np.argmax(perturbed, axis=2)
    best_value = np.take_along_axis(perturbed, best[..., np.newaxis], axis=2)[..., 0]
    valid = elev_mask & np.isfinite(best_value)
    return np.where(valid, best, -1).astype(np.int32)
