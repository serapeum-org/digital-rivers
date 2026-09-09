"""Writers that hand a DEM to a hydrodynamic model.

Six targets, each wanting a different shape of the same surface:

| target | files written | format |
| --- | --- | --- |
| `lisflood_fp` | the path as given | Arc ASCII grid, six-line header |
| `hec_ras` | the path as given | single-band float32 GeoTIFF in the dataset CRS |
| `tuflow` | `.flt` + `.hdr` | ESRI float grid, little-endian, with a text header |
| `sfincs` | `.dep` + `.msk` | headerless float32, plus a 0/1 validity mask |
| `gmsh` | `.geo` | a bounds rectangle with a uniform characteristic length |
| `iber` | `.dat` | ASCII boundary file for the Iber pre-processor |

Every writer takes an :class:`ExportGrid` — the surface plus the georeferencing the
headers need — and returns a `dict` mapping an artefact label to the path written, so a
caller that does not know which target produces two files still learns both names.

Splitting these out of `DEM.export` is what makes them testable: a writer can be handed a
synthetic 3x3 grid and its bytes checked, with no raster on disk and no `DEM` instance.
`DEM.export` validates the target and dispatches through :func:`write`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyramids.dataset import Dataset, GeoReference

__all__ = ["ExportGrid", "TARGETS", "write"]


@dataclass(frozen=True)
class ExportGrid:
    """The surface and georeferencing every writer needs.

    Built once by `DEM.export` so the no-data substitution and the corner arithmetic
    happen in one place rather than in each of six writers.

    Attributes:
        values: 2-D elevation array with the no-data sentinel already substituted for
            `NaN`, so writers never handle `NaN` themselves.
        nan_mask: `True` where the source array was `NaN`. SFINCS needs it for its
            `.msk` companion; the others ignore it.
        geotransform: The six-element GDAL geotransform.
        epsg: EPSG code of the dataset CRS.
        no_data: The no-data sentinel, written into the ASCII headers.
    """

    values: np.ndarray
    nan_mask: np.ndarray
    geotransform: tuple
    epsg: int
    no_data: float

    @property
    def shape(self) -> tuple[int, int]:
        """`(rows, columns)` of the grid."""
        return self.values.shape

    @property
    def cell_size(self) -> float:
        """Cell size in CRS units, from the geotransform's x pitch."""
        return abs(self.geotransform[1])

    @property
    def xllcorner(self) -> float:
        """X of the lower-left corner, the origin Arc-style headers count from."""
        return self.geotransform[0]

    @property
    def yllcorner(self) -> float:
        """Y of the lower-left corner.

        The geotransform's y origin is the *top* edge and its y pitch is negative, so
        the bottom edge is `y0 + rows * dy`.
        """
        rows = self.values.shape[0]
        return self.geotransform[3] + rows * self.geotransform[5]


def _suffixed(path: str, ext: str) -> str:
    """Return `path` with `ext` appended unless it already ends with it."""
    return path if path.endswith(ext) else path + ext


def _arc_header(grid: ExportGrid) -> str:
    """Return the six-line Arc ASCII header that LISFLOOD-FP and TUFLOW share."""
    rows, cols = grid.shape
    return (
        f"ncols {cols}\n"
        f"nrows {rows}\n"
        f"xllcorner {grid.xllcorner}\n"
        f"yllcorner {grid.yllcorner}\n"
        f"cellsize {grid.cell_size}\n"
        f"NODATA_value {grid.no_data}\n"
    )


def _write_rows(fh, values: np.ndarray) -> None:
    """Write an elevation array as space-separated rows at six decimals."""
    rows, cols = values.shape
    for r in range(rows):
        fh.write(" ".join(f"{values[r, c]:.6f}" for c in range(cols)))
        fh.write("\n")


def write_lisflood_fp(grid: ExportGrid, path: str, **kwargs) -> dict:
    """Write an Arc ASCII grid, the canonical LISFLOOD-FP elevation input.

    Args:
        grid: The surface to write.
        path: Output path, used exactly as given.
        **kwargs: Ignored; accepted so every writer shares one signature.

    Returns:
        `{"dem_asc": path}`.
    """
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(_arc_header(grid))
        _write_rows(fh, grid.values)
    return {"dem_asc": path}


def write_hec_ras(grid: ExportGrid, path: str, **kwargs) -> dict:
    """Write a single-band float32 GeoTIFF for HEC-RAS Mapper.

    Mapper wants the dataset CRS and a consistent geotransform, which is exactly what
    `Dataset.from_array(path=...)` writes — the driver comes from the `.tif` extension.

    Args:
        grid: The surface to write.
        path: Output path, `.tif`.
        **kwargs: Ignored; accepted so every writer shares one signature.

    Returns:
        `{"dem_tif": path}`.
    """
    Dataset.from_array(
        grid.values.astype(np.float32, copy=False),
        geo_ref=GeoReference(geo=grid.geotransform, epsg=grid.epsg),
        no_data_value=grid.no_data,
        path=path,
    )
    return {"dem_tif": path}


def write_tuflow(grid: ExportGrid, path: str, **kwargs) -> dict:
    """Write an ESRI floating-point grid: `.flt` binary plus `.hdr` text header.

    The binary is row-major little-endian float32, top-left cell first.

    Args:
        grid: The surface to write.
        path: Output path; `.flt` is appended when missing.
        **kwargs: Ignored; accepted so every writer shares one signature.

    Returns:
        `{"dem_flt": ..., "dem_hdr": ...}`.
    """
    flt_path = _suffixed(path, ".flt")
    hdr_path = flt_path[:-4] + ".hdr"
    grid.values.astype(np.float32, copy=False).tofile(flt_path)
    with open(hdr_path, "w") as fh:
        fh.write(_arc_header(grid))
        fh.write("byteorder LSBFIRST\n")
    return {"dem_flt": flt_path, "dem_hdr": hdr_path}


def write_sfincs(grid: ExportGrid, path: str, **kwargs) -> dict:
    """Write the SFINCS pair: a headerless `.dep` and its `.msk` companion.

    `.dep` is row-major little-endian float32 with no header at all — SFINCS takes the
    shape from its own control file. `.msk` is `uint8`, `0` where the source was
    no-data and `1` elsewhere.

    Args:
        grid: The surface to write.
        path: Output path; `.dep` is appended when missing.
        **kwargs: Ignored; accepted so every writer shares one signature.

    Returns:
        `{"dem_dep": ..., "dem_msk": ...}`.
    """
    dep_path = _suffixed(path, ".dep")
    msk_path = dep_path[:-4] + ".msk"
    grid.values.astype(np.float32, copy=False).tofile(dep_path)
    np.where(grid.nan_mask, 0, 1).astype(np.uint8).tofile(msk_path)
    return {"dem_dep": dep_path, "dem_msk": msk_path}


def write_gmsh(grid: ExportGrid, path: str, **kwargs) -> dict:
    """Write a minimal Gmsh `.geo` script describing the DEM's bounds.

    The script is the grid extent as a rectangle with a uniform characteristic length
    equal to the cell size — enough for `gmsh -2 <path>` to produce a mesh over the
    footprint. It carries no elevations; the surface itself is not exported here.

    Args:
        grid: The surface whose bounds are written.
        path: Output path; `.geo` is appended when missing.
        **kwargs: Ignored; accepted so every writer shares one signature.

    Returns:
        `{"geo": path}`.
    """
    geo_path = _suffixed(path, ".geo")
    rows, cols = grid.shape
    x0, dx, _, y0, _, dy = grid.geotransform
    ext_x_lo = x0
    ext_x_hi = x0 + cols * dx
    ext_y_hi = y0
    ext_y_lo = y0 + rows * dy
    cl = grid.cell_size
    with open(geo_path, "w") as fh:
        fh.write(f"cl = {cl};\n")
        fh.write(f"Point(1) = {{{ext_x_lo}, {ext_y_lo}, 0, cl}};\n")
        fh.write(f"Point(2) = {{{ext_x_hi}, {ext_y_lo}, 0, cl}};\n")
        fh.write(f"Point(3) = {{{ext_x_hi}, {ext_y_hi}, 0, cl}};\n")
        fh.write(f"Point(4) = {{{ext_x_lo}, {ext_y_hi}, 0, cl}};\n")
        fh.write("Line(1) = {1, 2};\n")
        fh.write("Line(2) = {2, 3};\n")
        fh.write("Line(3) = {3, 4};\n")
        fh.write("Line(4) = {4, 1};\n")
        fh.write("Line Loop(1) = {1, 2, 3, 4};\n")
        fh.write("Plane Surface(1) = {1};\n")
    return {"geo": geo_path}


def write_iber(grid: ExportGrid, path: str, **kwargs) -> dict:
    """Write an ASCII boundary file for Iber's pre-processor.

    Iber ultimately wants a `.dat` mesh. Until mesh generation lands (P33) this writes
    the grid as an uppercase-header ASCII file the pre-processor can import and refine.

    Args:
        grid: The surface to write.
        path: Output path; `.dat` is appended when missing.
        **kwargs: Ignored; accepted so every writer shares one signature.

    Returns:
        `{"dem_dat": ...}`.
    """
    dat_path = _suffixed(path, ".dat")
    rows, cols = grid.shape
    with open(dat_path, "w") as fh:
        fh.write("# Iber mesh boundary (auto-generated)\n")
        fh.write(f"NCOLS {cols}\nNROWS {rows}\n")
        fh.write(f"XLLCORNER {grid.xllcorner}\nYLLCORNER {grid.yllcorner}\n")
        fh.write(f"CELLSIZE {grid.cell_size}\nNODATA {grid.no_data}\n")
        _write_rows(fh, grid.values)
    return {"dem_dat": dat_path}


#: Target name -> writer. `DEM.export` validates against the keys, so adding a target
#: here is the only edit a new format needs.
TARGETS = {
    "lisflood_fp": write_lisflood_fp,
    "hec_ras": write_hec_ras,
    "tuflow": write_tuflow,
    "sfincs": write_sfincs,
    "gmsh": write_gmsh,
    "iber": write_iber,
}


def write(target: str, grid: ExportGrid, path: str, **kwargs) -> dict:
    """Dispatch to the writer for `target`.

    Args:
        target: One of the keys of :data:`TARGETS`.
        grid: The surface to write.
        path: Output path; each writer appends its own extension when missing.
        **kwargs: Passed through to the writer.

    Returns:
        `dict` mapping artefact label to the path written.

    Raises:
        ValueError: For an unknown `target`.
    """
    try:
        writer = TARGETS[target]
    except KeyError:
        raise ValueError(
            f"target must be one of {sorted(TARGETS)}; got {target!r}"
        ) from None
    return writer(grid, path, **kwargs)
