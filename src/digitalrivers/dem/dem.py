"""The `DEM` class: a raster read as a surface water moves across.

`DEM` is a `pyramids.dataset.Dataset` that knows its band holds elevation, and everything
that follows from that. The method surface is assembled from four mixins, one per
concern, so this module holds the class itself rather than three thousand lines of
method bodies:

* :mod:`digitalrivers.dem.conditioning` -- edits to the surface that make it routable:
  depression fill and breach, flat resolution, stream / culvert / building / breakline
  enforcement.
* :mod:`digitalrivers.dem.morphometry` -- shape descriptors: slope, curvature, the
  focal statistics, the wetness and transport indices, openness and sky-view factor.
* :mod:`digitalrivers.dem.routing` -- flow direction under five routing schemes, and
  accumulation over the result.
* :mod:`digitalrivers.dem.hydrology` -- products that need a stream network as well as a
  surface: HAND, and the end-to-end pipeline.

What stays here is what is genuinely about being a DEM rather than about one of those
concerns: construction, the `values` view that swaps the no-data sentinel for `NaN`, and
the exporters that hand the surface to something outside the package.

The mixins carry no state. Everything they touch -- `values`, `geotransform`, `epsg`,
`no_data_value`, `raster` -- comes from `Dataset` or from this class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from osgeo import gdal
from pyramids.dataset import Dataset, GeoReference

from digitalrivers.dem._kernels.pitremoval import local_minima_8
from digitalrivers.dem.conditioning import ConditioningMixin
from digitalrivers.dem.hydrology import HydrologyMixin
from digitalrivers.dem.morphometry import MorphometryMixin
from digitalrivers.dem.routing import RoutingMixin

__all__ = ["DEM"]


class DEM(
    ConditioningMixin,
    MorphometryMixin,
    RoutingMixin,
    HydrologyMixin,
    Dataset,
):
    """Digital Elevation Model processor.

    Wraps a GDAL raster dataset and adds hydrological analysis methods:
    sink filling, D8 flow direction, flow accumulation, and slope
    computation.

    Args:
        src: GDAL dataset containing a single-band elevation raster. To open a
            DEM from a file path, use the inherited `DEM.read_file(path)`
            classmethod — it returns a `DEM`, not a plain
            `pyramids.dataset.Dataset`.
        access: `"read_only"` (default) or `"write"`.
        gdal_env: GDAL config (cloud credentials, HTTP knobs) captured on
            the dataset and re-installed around its reads, so the paths that
            reopen the file authenticate the same way. Default `None`.
        open_options: GDAL open options captured on the dataset and reapplied
            when it is reopened. Default `None`.
    """

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        *,
        gdal_env: dict[str, str] | None = None,
        open_options: tuple[str, ...] | list[str] | None = None,
    ):
        """Wrap a GDAL dataset as a digital elevation model.

        Args:
            src: Open GDAL dataset to wrap. The handle is adopted, not copied.
                To open a DEM from a file path use the inherited
                `DEM.read_file(path)` classmethod instead; this constructor
                accepts a `gdal.Dataset` only.
            access: `"read_only"` (default) or `"write"`.
            gdal_env: GDAL config (cloud credentials, HTTP knobs) captured on the
                dataset and re-installed around its reads. Default `None`.
            open_options: GDAL open options captured on the dataset and reapplied
                when it is reopened. Default `None`.
        """
        super().__init__(src, access, gdal_env=gdal_env, open_options=open_options)

    @property
    def values(self):
        """Elevation array with no-data cells replaced by `np.nan`.

        Reads band 0 as `float32` and masks every cell whose value is
        close to the raster's no-data value (relative tolerance 1e-5).

        Returns:
            np.ndarray: 2-D `float32` array of shape `(rows, columns)`.
        """
        values = self.read_array(band=0).astype(np.float32)
        # get the value stores in no data value cells
        no_val = self.no_data_value[0]
        values[np.isclose(values, no_val, rtol=0.00001)] = np.nan
        return values

    def subgrid_bathymetry(
        self,
        scale_factor: int,
        n_bins: int = 10,
    ) -> "pd.DataFrame":
        """Build per-coarse-cell bathymetry tables (SFINCS-style).

        For each coarse cell (`scale_factor × scale_factor` block of fine
        cells), compute a histogram-like table mapping a coarsened water-
        depth level to the wetted area within the block. This is the
        sub-grid representation SFINCS and similar reduced-order 2D models
        use to recover small-scale topography without resolving it on the
        coarse grid.

        Args:
            scale_factor: Integer aggregation factor (>= 2).
            n_bins: Number of depth bins per coarse cell.

        Returns:
            `pandas.DataFrame` indexed by coarse-cell `(row, col)` with
            `n_bins + 2` columns: `z_min`, `z_max`, plus
            `frac_below_<k>` for `k` in `[1, n_bins]` giving the
            fraction of fine cells at or below the `k`-th depth bin.
            For flat blocks (`z_max == z_min`) every `frac_below_<k>`
            is `1.0`.

        Raises:
            ValueError: For `scale_factor < 2` or `n_bins < 1`.

        Examples:
            - A flat block produces `frac_below_<k> == 1.0` for every bin
              (B1 regression — the columns are always present):

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> ds = Dataset.from_array(
                ...     np.full((4, 4), 5.0, dtype=np.float32),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> df = DEM(ds.raster).subgrid_bathymetry(scale_factor=2, n_bins=3)
                >>> sorted(df.columns.tolist())
                ['frac_below_1', 'frac_below_2', 'frac_below_3', 'z_max', 'z_min']
                >>> float(df["frac_below_1"].iloc[0])
                1.0
        """
        if scale_factor < 2:
            raise ValueError(f"scale_factor must be >= 2; got {scale_factor}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1; got {n_bins}")

        elev = self.values
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
                        rec[f"frac_below_{k}"] = float(
                            (valid <= edge).sum() / valid.size
                        )
                records.append(rec)
        df = pd.DataFrame(records).set_index(["row", "col"])
        return df

    def export(
        self,
        path: str,
        target: str,
        *,
        breaklines=None,
        walls=None,
        buildings=None,
        manning_n=None,
        boundary_conditions=None,
        validate: bool = True,
        **kwargs,
    ) -> dict:
        """Export the DEM to a hydrodynamic-model format.

        Every target listed below ships with a working writer. Most were
        backfilled after the initial Phase-3 cut; `lisflood_fp` is the
        canonical Arc-ASCII writer and remains the only target that
        actually requires a sinks-free input.

        Args:
            path: Output file path.
            target: One of `"hec_ras"`, `"tuflow"`, `"sfincs"`,
                `"lisflood_fp"`, `"iber"`, `"gmsh"`.
            breaklines / walls / buildings / manning_n / boundary_conditions:
                Reserved for target-specific bundles. Currently ignored by
                the LISFLOOD-FP writer.
            validate: When `True` (default), refuse to export a DEM with
                internal sinks. **Only applied for** `target="lisflood_fp"`
                — the Arc-ASCII writer is the only target where downstream
                tooling actually requires sinks-free input. Other writers
                skip the sink scan even when `validate=True` so the
                `local_minima_8` pass does not run unnecessarily (I4
                fixup). Pass `validate=False` to also disable the
                lisflood_fp guard.
            **kwargs: Target-specific options.

        Returns:
            `dict` mapping artefact label → file path written.

        Raises:
            ValueError: For unknown `target`.
            RuntimeError: When `target == "lisflood_fp"`, `validate=True`,
                and the DEM has internal sinks.
        """
        valid_targets = {
            "hec_ras",
            "tuflow",
            "sfincs",
            "lisflood_fp",
            "iber",
            "gmsh",
        }
        if target not in valid_targets:
            raise ValueError(
                f"target must be one of {sorted(valid_targets)}; got {target!r}"
            )

        # Only run the (expensive) sink scan when we'll actually export to
        # an implemented target. The unimplemented targets raise
        # NotImplementedError further down and would otherwise pay the full
        # validation cost for nothing.
        if validate and target == "lisflood_fp":
            sinks = local_minima_8(self.values)
            if int(sinks.sum()) > 0:
                raise RuntimeError(
                    f"DEM has {int(sinks.sum())} internal sinks; either fix "
                    "them (DEM.fill_depressions) or pass validate=False"
                )

        elev = self.values
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt
        rows, cols = elev.shape
        nodata = float(self.no_data_value[0])
        out = np.where(np.isnan(elev), nodata, elev)
        cell_size = abs(dx)
        yllcorner = y0 + rows * dy

        if target == "lisflood_fp":
            with open(path, "w", encoding="ascii", newline="\n") as fh:
                fh.write(f"ncols {cols}\n")
                fh.write(f"nrows {rows}\n")
                fh.write(f"xllcorner {x0}\n")
                fh.write(f"yllcorner {yllcorner}\n")
                fh.write(f"cellsize {cell_size}\n")
                fh.write(f"NODATA_value {nodata}\n")
                for r in range(rows):
                    fh.write(" ".join(f"{out[r, c]:.6f}" for c in range(cols)))
                    fh.write("\n")
            return {"dem_asc": path}

        if target == "hec_ras":
            # HEC-RAS Mapper expects a single-band float32 GeoTIFF in the
            # dataset CRS with consistent geotransform — exactly what
            # Dataset.from_array(path=...) writes (the driver comes from the
            # `.tif` extension).
            Dataset.from_array(
                out.astype(np.float32, copy=False),
                geo_ref=GeoReference(geo=gt, epsg=self.epsg),
                no_data_value=nodata,
                path=path,
            )
            return {"dem_tif": path}

        if target == "tuflow":
            # ESRI floating-point grid (.flt binary, row-major little-endian
            # float32, top-left first) + .hdr text header.
            flt_path = path if path.endswith(".flt") else path + ".flt"
            hdr_path = flt_path[:-4] + ".hdr"
            out.astype(np.float32, copy=False).tofile(flt_path)
            with open(hdr_path, "w") as fh:
                fh.write(f"ncols {cols}\n")
                fh.write(f"nrows {rows}\n")
                fh.write(f"xllcorner {x0}\n")
                fh.write(f"yllcorner {yllcorner}\n")
                fh.write(f"cellsize {cell_size}\n")
                fh.write(f"NODATA_value {nodata}\n")
                fh.write("byteorder LSBFIRST\n")
            return {"dem_flt": flt_path, "dem_hdr": hdr_path}

        if target == "sfincs":
            # SFINCS .dep: row-major little-endian float32, no header.
            # Companion .msk: 0 where no-data, 1 elsewhere.
            dep_path = path if path.endswith(".dep") else path + ".dep"
            msk_path = dep_path[:-4] + ".msk"
            out.astype(np.float32, copy=False).tofile(dep_path)
            mask = np.where(np.isnan(elev), 0, 1).astype(np.uint8)
            mask.tofile(msk_path)
            return {"dem_dep": dep_path, "dem_msk": msk_path}

        if target == "gmsh":
            # Minimal .geo script: define the DEM bounds as a rectangle
            # with a uniform characteristic length. Downstream meshers can
            # be run via `gmsh -2 <path>`.
            geo_path = path if path.endswith(".geo") else path + ".geo"
            ext_x_lo = x0
            ext_x_hi = x0 + cols * dx
            ext_y_hi = y0
            ext_y_lo = y0 + rows * dy
            cl = cell_size
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

        if target == "iber":
            # Iber expects a .dat ascii mesh; pending mesh generation
            # (Phase 4 P33) we write a placeholder boundary file that the
            # user can refine in Iber's pre-processor.
            dat_path = path if path.endswith(".dat") else path + ".dat"
            with open(dat_path, "w") as fh:
                fh.write("# Iber mesh boundary (auto-generated)\n")
                fh.write(f"NCOLS {cols}\nNROWS {rows}\n")
                fh.write(f"XLLCORNER {x0}\nYLLCORNER {yllcorner}\n")
                fh.write(f"CELLSIZE {cell_size}\nNODATA {nodata}\n")
                for r in range(rows):
                    fh.write(" ".join(f"{out[r, c]:.6f}" for c in range(cols)))
                    fh.write("\n")
            return {"dem_dat": dat_path}

        # Unreachable — guarded by valid_targets check above.
        raise NotImplementedError(target)

    def anudem_interpolate(
        self,
        mask=None,
        max_iter: int = 200,
        tol: float = 1e-3,
        method: str = "laplacian",
        inplace: bool = False,
    ) -> DEM | None:
        """ANUDEM-lite: Laplacian-relaxation gap fill (P25).

        A pragmatic subset of Hutchinson 1989 ANUDEM that handles the
        common gap-filling case: a DEM with NaN holes (cloud shadows,
        survey gaps, vegetation occlusion) is filled by Gauss-Seidel
        Laplacian relaxation, holding the known cells fixed. Each
        iteration replaces every unknown cell with the mean of its four
        4-connected neighbours; iteration stops when the maximum change
        in a sweep drops below `tol` or after `max_iter` sweeps.

        Two solver methods are available:

        - `"laplacian"` (default): solves Δz = 0 via the 4-neighbour
          mean iteration. Fast, smooth interior, but only C⁰ continuity
          at the anchor cells — the surface has visible "kinks" at
          known points.
        - `"biharmonic"`: solves Δ²z = 0 by alternating two Laplacian
          sweeps (relax `u = Δz`, then relax `z` so `Δz = u`).
          C¹ continuity at anchors; closer to Hutchinson 1989 ANUDEM's
          tension-spline objective but still without multigrid
          acceleration or drainage enforcement.

        Limitations vs full ANUDEM:

        - No multigrid acceleration; iteration cost is O(N · max_iter).
        - No drainage enforcement. For stream-conditioned DEMs, combine
          with :meth:`burn_streams` before or after.
        - No tension parameter (Hutchinson's λ); the biharmonic mode
          is a fixed-λ approximation.

        Args:
            mask: Optional bool array same shape as the DEM. `True`
                marks cells whose values must be preserved (in addition
                to the existing finite cells). `None` keeps every
                finite cell fixed.
            max_iter: Maximum relaxation sweeps.
            tol: Convergence tolerance — stop when `max |Δz| < tol`.
            method: `"laplacian"` (default) or `"biharmonic"`.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: Filled DEM, or None when `inplace=True`.

        Raises:
            ValueError: If the input DEM has no finite cells or `method`
                is not `"laplacian"` / `"biharmonic"`.

        Examples:
            - Fill a single-cell NaN hole with the default Laplacian solver;
              the filled value sits inside the bracketing range:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
                ...     dtype=np.float32,
                ... )
                >>> ds = Dataset.from_array(
                ...     np.where(np.isnan(z), -9999.0, z),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> filled = DEM(ds.raster).anudem_interpolate(
                ...     method="laplacian", max_iter=200, tol=1e-6,
                ... )
                >>> out = filled.values
                >>> bool(1.0 <= out[1, 1] <= 9.0)
                True

            - The biharmonic mode (P32 backfill) approximates Hutchinson
              1989's Delta^2 z = 0 by alternating Laplacian sweeps:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
                ...     dtype=np.float32,
                ... )
                >>> ds = Dataset.from_array(
                ...     np.where(np.isnan(z), -9999.0, z),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> filled = DEM(ds.raster).anudem_interpolate(
                ...     method="biharmonic", max_iter=200, tol=1e-5,
                ... )
                >>> bool(np.isfinite(filled.values).all())
                True

            - Unknown method raises ValueError:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> ds = Dataset.from_array(
                ...     np.ones((2, 2), dtype=np.float32),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> DEM(ds.raster).anudem_interpolate(method="bogus")
                Traceback (most recent call last):
                    ...
                ValueError: method must be 'laplacian' or 'biharmonic'; got 'bogus'

            - The 5-point stencil uses edge-replication (Neumann) boundary
              padding, so a NaN cell adjacent to the raster boundary is
              filled from its in-bounds neighbours only — no periodic-wrap
              contamination from the opposite edge:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> # Top-left NaN with very different anchor at bottom-right.
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> z[-1, -1] = -100.0
                >>> z[0, 0] = np.nan
                >>> ds = Dataset.from_array(
                ...     np.where(np.isnan(z), -9999.0, z),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> filled = DEM(ds.raster).anudem_interpolate(
                ...     method="laplacian", max_iter=300, tol=1e-6,
                ... )
                >>> bool(abs(float(filled.values[0, 0]) - 10.0) < 5.0)
                True

        See Also:
            DEM.fill_depressions: hydrologic conditioning that removes sinks.
            DEM.burn_streams: stream-network drainage enforcement.
        """
        if method not in ("laplacian", "biharmonic"):
            raise ValueError(
                f"method must be 'laplacian' or 'biharmonic'; got {method!r}"
            )

        elev = self.values
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

        return self._conditioned_result(elev, z, inplace, gaps=~np.isfinite(z))
