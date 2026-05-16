"""Bathymetric DEM fusion (topo + bathy → single hydrographic surface).

* :func:`topobathy_fusion` — per-cell fuse of a topographic and a
  bathymetric DEM into a single surface. Four blend modes:
  `"max"`, `"min"`, `"topo_above"`, `"bathy_below"`.
"""
from __future__ import annotations


def topobathy_fusion(
    topo,
    bathy,
    shoreline_elev: float = 0.0,
    blend: str = "max",
):
    """Fuse a topographic DEM and a bathymetric DEM into a single hydrographic surface.

    Both inputs must be aligned (same shape, geotransform, CRS). The
    shoreline is the contour at `shoreline_elev` (default 0 — mean sea
    level).

    Blend modes:

    * `"max"` (default): per-cell maximum of the two DEMs. Topo wins
      above the shoreline, bathy below — the canonical conservative
      choice when the two DEMs disagree across the shoreline (NOAA
      ETOPO uses this).
    * `"min"`: per-cell minimum — the pessimistic-bathymetry choice
      for flood inundation studies where you want to assume the deeper
      of two conflicting surveys.
    * `"topo_above"`: pick topo where `topo >= shoreline_elev`,
      bathy elsewhere. Sharp transition at the shoreline; preferred when
      the topo DEM is known accurate at the coastline.
    * `"bathy_below"`: pick bathy where `bathy <= shoreline_elev`,
      topo elsewhere. Mirror of the above.

    Args:
        topo: Topographic `Dataset` (DEM subclass acceptable).
        bathy: Bathymetric `Dataset` aligned to `topo`.
        shoreline_elev: Elevation defining the shoreline contour.
            Default 0.0 (MSL).
        blend: `"max"` (default), `"min"`, `"topo_above"`, or
            `"bathy_below"`.

    Returns:
        `Dataset` of the fused surface.

    Raises:
        ValueError: If shapes mismatch or `blend` is unknown.

    Examples:
        - The `"min"` blend picks the cell-by-cell minimum — useful as a
          pessimistic-bathymetry baseline:

            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from digitalrivers.fusion import topobathy_fusion
            >>> topo = Dataset.create_from_array(
            ...     np.array([[5.0, -1.0]], dtype=np.float32),
            ...     top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
            ... )
            >>> bathy = Dataset.create_from_array(
            ...     np.array([[-3.0, -5.0]], dtype=np.float32),
            ...     top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
            ... )
            >>> fused = topobathy_fusion(topo, bathy, blend="min")
            >>> fused.read_array().tolist()
            [[-3.0, -5.0]]

    References:
        Eakins B. W., Grothe P. R. (2014). "Challenges in building coastal
        digital elevation models." Journal of Coastal Research 30(5).
    """
    import numpy as np
    from pyramids.dataset import Dataset

    if blend not in ("max", "min", "topo_above", "bathy_below"):
        raise ValueError(
            f"blend must be one of 'max', 'min', 'topo_above', "
            f"'bathy_below'; got {blend!r}"
        )

    topo_arr = topo.read_array().astype(np.float64, copy=False)
    bathy_arr = bathy.read_array().astype(np.float64, copy=False)
    if topo_arr.shape != bathy_arr.shape:
        raise ValueError(
            f"topo shape {topo_arr.shape} != bathy shape {bathy_arr.shape}"
        )

    topo_no_val = topo.no_data_value[0] if topo.no_data_value else None
    bathy_no_val = bathy.no_data_value[0] if bathy.no_data_value else None
    if topo_no_val is not None:
        topo_arr = np.where(topo_arr == topo_no_val, np.nan, topo_arr)
    if bathy_no_val is not None:
        bathy_arr = np.where(bathy_arr == bathy_no_val, np.nan, bathy_arr)

    if blend == "max":
        fused = np.fmax(topo_arr, bathy_arr)
    elif blend == "min":
        fused = np.fmin(topo_arr, bathy_arr)
    elif blend == "topo_above":
        fused = np.where(topo_arr >= shoreline_elev, topo_arr, bathy_arr)
    else:  # bathy_below
        fused = np.where(bathy_arr <= shoreline_elev, bathy_arr, topo_arr)

    out_no_val = topo_no_val if topo_no_val is not None else -9999.0
    fused = np.where(np.isnan(fused), out_no_val, fused)
    return Dataset.create_from_array(
        fused.astype(np.float32, copy=False),
        geo=topo.geotransform, epsg=topo.epsg,
        no_data_value=out_no_val,
    )
