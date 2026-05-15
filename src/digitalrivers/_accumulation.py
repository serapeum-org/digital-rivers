"""Generalised flow-accumulation algorithm for DEM hydro pre-processing (P6).

Single Kahn-style topological-sort sweep handles all five routing schemes. The
per-routing differences live in how a ``FlowDirection`` is decoded into a uniform
``(receivers, proportions)`` representation:

* **D8 / Rho8** — one receiver per cell, proportion ``1.0``.
* **D∞** — two receivers per cell, proportions derived from the within-sector angle
  fraction. Cells outside the 45° sectors collapse to a single receiver.
* **MFD-Quinn / MFD-Holmgren** — up to eight receivers per cell, proportions from
  the 8-band fraction stack on disk.

Output semantics: ``out[cell] = sum of weights[upstream_cells]`` — i.e. the cell's
own weight does **not** contribute to its own accumulation. This matches the legacy
``DEM.flow_accumulation`` behaviour (count of upstream cells when ``weights=1``).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from digitalrivers._flow_routing import _DIR_DR, _DIR_DC


def _receivers_d8(
    fd_array: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a D8/Rho8 single-band direction-code raster into the canonical
    ``(receivers, proportions)`` form.

    Returns:
        receivers: ``(rows, cols, 1)`` int8 of direction codes (0–7) or ``-1`` for
            sinks / no-data.
        proportions: ``(rows, cols, 1)`` float32 of ``1.0`` for valid receivers,
            else ``0.0``.
    """
    rows, cols = fd_array.shape
    receivers = np.full((rows, cols, 1), -1, dtype=np.int8)
    proportions = np.zeros((rows, cols, 1), dtype=np.float32)
    valid_dir = valid_mask & (fd_array >= 0) & (fd_array <= 7)
    receivers[valid_dir, 0] = fd_array[valid_dir].astype(np.int8)
    proportions[valid_dir, 0] = 1.0
    return receivers, proportions


def _receivers_dinf(
    angle: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a D∞ angle raster into ``(receivers, proportions)`` with two receivers
    per cell. Angle convention: radians CCW from east in ``[0, 2π)``; ``-1`` marks
    sinks / no-data.

    Returns:
        receivers: ``(rows, cols, 2)`` int8.
        proportions: ``(rows, cols, 2)`` float32; rows sum to 1 for valid cells.
    """
    rows, cols = angle.shape
    receivers = np.full((rows, cols, 2), -1, dtype=np.int8)
    proportions = np.zeros((rows, cols, 2), dtype=np.float32)
    pi_over_4 = np.pi / 4.0
    valid = valid_mask & (angle >= 0)
    rs, cs = np.where(valid)
    a = angle[rs, cs]
    sector = np.floor(a / pi_over_4).astype(np.int32) % 8
    frac2 = a / pi_over_4 - sector
    frac1 = 1.0 - frac2
    # DIR_OFFSETS direction codes for each sector boundary.
    # sector 0: E(6) and NE(5);  sector 1: NE(5) and N(4); ...; sector 7: SE(7) and E(6).
    code1 = (6 - sector) % 8
    code2 = (5 - sector) % 8
    receivers[rs, cs, 0] = code1.astype(np.int8)
    receivers[rs, cs, 1] = code2.astype(np.int8)
    proportions[rs, cs, 0] = frac1.astype(np.float32)
    proportions[rs, cs, 1] = frac2.astype(np.float32)
    return receivers, proportions


def _receivers_mfd(
    fractions: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an MFD 8-band fraction stack into ``(receivers, proportions)`` with
    up to eight receivers per cell.

    Args:
        fractions: ``(rows, cols, 8)`` float — fraction sent to each DIR_OFFSETS
            direction. Rows sum to 1.0 for non-sink cells, 0.0 for sinks.

    Returns:
        receivers: ``(rows, cols, 8)`` int8; ``-1`` for cells that should not
            propagate at all.
        proportions: ``(rows, cols, 8)`` float32 — direct copy of input.
    """
    rows, cols, _ = fractions.shape
    receivers = np.tile(np.arange(8, dtype=np.int8), (rows, cols, 1))
    receivers[~valid_mask] = -1
    proportions = fractions.astype(np.float32, copy=False)
    return receivers, proportions


def kahn_accumulate(
    receivers: np.ndarray,
    proportions: np.ndarray,
    weights: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Kahn-style topological-sort flow accumulation.

    Args:
        receivers: ``(rows, cols, K)`` int8 of DIR_OFFSETS direction codes 0–7 (or
            ``-1`` for "no receiver in this slot"). ``K=1`` for D8/Rho8,
            ``K=2`` for D∞, ``K=8`` for MFD.
        proportions: ``(rows, cols, K)`` float32 of partition fractions matching
            ``receivers``. Rows summing to 1.0 (or 0.0 for sinks).
        weights: ``(rows, cols)`` float of per-cell input weights (e.g. rainfall,
            runoff coefficient × area, uniform 1.0 for cell counts).
        valid_mask: ``(rows, cols)`` bool — True for valid-data cells. Invalid
            cells are excluded from both the in-degree count and the propagation.

    Returns:
        ``(rows, cols)`` float64 accumulation grid. ``out[cell]`` is the sum of
        ``weights`` over all strictly-upstream cells (not including ``cell``
        itself), weighted by propagation fractions along each path.
    """
    rows, cols, K = receivers.shape

    # In-degree: for every (r, c, k) with a valid receiver and positive proportion,
    # increment the receiver cell's indeg.
    indeg = np.zeros((rows, cols), dtype=np.int32)
    rr, cc = np.indices((rows, cols))
    for k in range(K):
        d = receivers[:, :, k]
        p = proportions[:, :, k]
        live = valid_mask & (d >= 0) & (p > 0)
        if not live.any():
            continue
        nr = rr + _DIR_DR[d.clip(min=0)]
        nc = cc + _DIR_DC[d.clip(min=0)]
        in_bounds = (
            live & (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
        )
        # Mask non-live before lookup to keep indices in range.
        nr_safe = np.where(in_bounds, nr, 0)
        nc_safe = np.where(in_bounds, nc, 0)
        valid_target = in_bounds & valid_mask[nr_safe, nc_safe]
        # Accumulate in-degree contributions.
        np.add.at(indeg, (nr_safe[valid_target], nc_safe[valid_target]), 1)

    out = np.zeros((rows, cols), dtype=np.float64)
    weights_f = weights.astype(np.float64, copy=False)

    # Seed the queue with all valid cells that have no upstream contributors.
    queue: deque[tuple[int, int]] = deque()
    indeg_zero_valid = valid_mask & (indeg == 0)
    for r, c in zip(*np.where(indeg_zero_valid)):
        queue.append((int(r), int(c)))

    while queue:
        r, c = queue.popleft()
        # Cell's outgoing contribution = own weight + everything that flowed in.
        contribution = weights_f[r, c] + out[r, c]
        for k in range(K):
            d = int(receivers[r, c, k])
            if d < 0:
                continue
            p = float(proportions[r, c, k])
            if p <= 0:
                continue
            nr = r + int(_DIR_DR[d])
            nc = c + int(_DIR_DC[d])
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not valid_mask[nr, nc]:
                continue
            out[nr, nc] += p * contribution
            indeg[nr, nc] -= 1
            if indeg[nr, nc] == 0:
                queue.append((nr, nc))
    return out


def accumulate(
    flow_direction_array: np.ndarray,
    routing: str,
    valid_mask: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """High-level dispatcher: decode ``flow_direction_array`` per ``routing``, then
    run :func:`kahn_accumulate`.

    Args:
        flow_direction_array: For ``d8``/``rho8``, a 2-D int direction-code raster.
            For ``dinf``, a 2-band float32 array shape ``(2, rows, cols)`` with
            angle on band 0. For MFD, an 8-band float32 array shape ``(8, rows, cols)``.
        routing: ``"d8" | "rho8" | "dinf" | "mfd_quinn" | "mfd_holmgren"``.
        valid_mask: ``(rows, cols)`` bool of valid-data cells.
        weights: ``(rows, cols)`` float of per-cell weights, or ``None`` for unit
            weights (cell-count accumulation).

    Returns:
        ``(rows, cols)`` float64 accumulation.
    """
    if routing in ("d8", "rho8"):
        if flow_direction_array.ndim != 2:
            raise ValueError(
                f"{routing!r} expects a 2-D direction-code array; got shape "
                f"{flow_direction_array.shape}"
            )
        receivers, proportions = _receivers_d8(flow_direction_array, valid_mask)
    elif routing == "dinf":
        if flow_direction_array.ndim != 3 or flow_direction_array.shape[0] != 2:
            raise ValueError(
                f"dinf expects a (2, rows, cols) angle/magnitude stack; got "
                f"shape {flow_direction_array.shape}"
            )
        angle = flow_direction_array[0]
        receivers, proportions = _receivers_dinf(angle, valid_mask)
    elif routing in ("mfd_quinn", "mfd_holmgren"):
        if flow_direction_array.ndim != 3 or flow_direction_array.shape[0] != 8:
            raise ValueError(
                f"{routing!r} expects an (8, rows, cols) fraction stack; got "
                f"shape {flow_direction_array.shape}"
            )
        # Reshape to (rows, cols, 8) for the helpers.
        fractions = np.transpose(flow_direction_array, (1, 2, 0))
        receivers, proportions = _receivers_mfd(fractions, valid_mask)
    else:
        raise ValueError(
            f"routing must be one of d8/rho8/dinf/mfd_quinn/mfd_holmgren; got {routing!r}"
        )

    rows, cols = valid_mask.shape
    if weights is None:
        weights = np.ones((rows, cols), dtype=np.float64)
    else:
        weights = weights.astype(np.float64, copy=False)
    # Zero out weights at invalid cells.
    weights = np.where(valid_mask, weights, 0.0)
    return kahn_accumulate(receivers, proportions, weights, valid_mask)
