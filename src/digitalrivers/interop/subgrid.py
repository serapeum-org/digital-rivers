"""Sub-grid bathymetry tables for reduced-order 2D models.

SFINCS and models like it run on a coarse grid but still want the small-scale topography
that grid cannot resolve. The compromise is a per-coarse-cell lookup table: for each
block of fine cells, what fraction of the block lies below each of `n_bins` depth
levels. The solver reads that table instead of resolving the topography, recovering most
of the storage and conveyance behaviour at coarse-grid cost.

`subgrid_table` is the array kernel; `DEM.subgrid_bathymetry` wraps it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["subgrid_table"]


def subgrid_table(elev: np.ndarray, scale_factor: int, n_bins: int) -> pd.DataFrame:
    """Build the per-coarse-cell depth / wetted-fraction table.

    Args:
        elev: 2-D elevation array with `NaN` in no-data cells.
        scale_factor: Fine cells per coarse cell along each axis. Must be >= 2.
        n_bins: Number of depth bins per coarse cell. Must be >= 1.

    Returns:
        `pandas.DataFrame` indexed by the coarse cell's `(row, col)`, with `z_min`,
        `z_max` and `frac_below_1` .. `frac_below_<n_bins>`. Blocks with no finite cell
        are omitted; a flat block reports `1.0` in every fraction column, since every
        bin edge sits at or above its single elevation.

    Raises:
        ValueError: For `scale_factor < 2` or `n_bins < 1`.
    """
    if scale_factor < 2:
        raise ValueError(f"scale_factor must be >= 2; got {scale_factor}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")

    rows, cols = elev.shape
    out_rows = rows // scale_factor
    out_cols = cols // scale_factor

    records: list[dict] = []
    for br in range(out_rows):
        for bc in range(out_cols):
            block = elev[
                br * scale_factor : (br + 1) * scale_factor,
                bc * scale_factor : (bc + 1) * scale_factor,
            ].ravel()
            valid = block[np.isfinite(block)]
            if valid.size == 0:
                continue
            z_min = float(valid.min())
            z_max = float(valid.max())
            rec = {"row": br, "col": bc, "z_min": z_min, "z_max": z_max}
            if z_max == z_min:
                # Flat block: every bin is "below" the single value, so
                # frac_below_k == 1.0 for every k. (The B1 review found
                # that the original code computed this list but never
                # wrote it into the record, dropping the frac columns
                # entirely when every block was flat.)
                for k in range(1, n_bins + 1):
                    rec[f"frac_below_{k}"] = 1.0
            else:
                bin_edges = np.linspace(z_min, z_max, n_bins + 1)
                for k, edge in enumerate(bin_edges[1:], start=1):
                    rec[f"frac_below_{k}"] = float((valid <= edge).sum() / valid.size)
            records.append(rec)
    df = pd.DataFrame(records).set_index(["row", "col"])
    return df
