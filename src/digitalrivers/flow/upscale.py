"""Coarse-grid upscaling of flow-direction rasters (COTAT, EAM, DMM, IHU).

Upscaling turns a fine flow-direction grid into a coarse one whose cells still drain the
way the fine network did. Four schemes are offered, differing in how they pick the coarse
cell's outlet and therefore in how faithfully the coarse network reproduces the fine
one's basin boundaries:

* **COTAT** (Reed 2003) — Cell Outlet Tracing with an Area Threshold. Traces the fine
  flow path out of each coarse block and takes the first crossing that carries enough
  upstream area.
* **EAM** (Fekete 2001) — Effective Area Method. Ranks candidate outlets by accumulated
  area rather than by trace order.
* **DMM** — Dominant River Tracing on the same machinery as EAM, without the weights.
* **IHU** (Eilander 2021) — Iterative Hydrography Upscaling, a hill-climbing refinement
  over a COTAT seed network. The most faithful and the slowest.

`UpscaleMixin` carries the `Dataset`-level plumbing — reading arrays, building the
coarse geotransform, wrapping results back into typed classes. The array algorithms live
in `flow._kernels`.
"""

from __future__ import annotations

import math

import numpy as np
from pyramids.dataset import Dataset, GeoReference

from digitalrivers.core.directions import (
    DIR_DC_I32 as _DIR_DC,
    DIR_DR_I32 as _DIR_DR,
)
from digitalrivers.flow._kernels.ihu import ihu_upscale

__all__ = ["UpscaleMixin"]


def _flow_direction_cls():
    """Return the `FlowDirection` class.

    Circular import: `FlowDirection` mixes this class in, so it cannot be named at
    module scope. Every upscaler builds its coarse result as a `FlowDirection`, so the
    lookup is deferred to call time rather than repeated as a local import in each of
    the four methods that need it.
    """
    from digitalrivers.flow.direction import FlowDirection

    return FlowDirection


def _pick_coarse_elev(z_fine, fr: int, fc: int) -> float:
    """Lift the fine-grid elevation at `(fr, fc)` to a coarse-cell value.

    Returns `z_fine[fr, fc]` as a float, or
    `Dataset.default_no_data_value` when the fine value is NaN / non-
    finite. Used by every upscaler (COTAT, EAM/DMM, IHU) to project the
    chosen-outlet fine elevation into the coarse DEM raster.
    """
    zv = z_fine[fr, fc]
    return float(zv) if math.isfinite(float(zv)) else Dataset.default_no_data_value


class UpscaleMixin:
    """Coarse-grid upscaling methods, mixed into :class:`FlowDirection`.

    The mixin declares no state: everything it touches (`routing`, `encoding`,
    `geotransform`, `epsg`, `raster`, `read_array`) comes from `Dataset` and from
    `FlowDirection`'s own constructor.
    """

    def upscale_ihu(
        self,
        scale_factor: int,
        accumulation,
        dem,
        max_iter: int = 20,
        report: bool = False,
    ) -> tuple:
        """Iterative Hydrography Upscaling (Eilander 2021).

        The state-of-the-art D8 upscaling method that builds an initial
        coarse network with COTAT-style outlet selection and then refines
        boundary mismatches by swapping outlets between adjacent coarse
        cells until convergence.

        v1 status: `scale_factor=1` is a no-op (passes through the input
        unchanged). All other `scale_factor` values raise
        `NotImplementedError`. The roadmap recommends vendoring pyflwdir
        as the first-release backend; a native swap-search implementation
        is deferred to Phase 4.

        Args:
            scale_factor: Integer aggregation factor (>= 1).
            accumulation: `Accumulation` aligned to this FlowDirection.
            dem: `DEM` aligned to this FlowDirection.
            max_iter: Maximum refinement iterations.
            report: When True, the third return slot carries Eilander 2021
                validation metrics (`area_error_pct`, `hit_rate`,
                `network_shift_km`). Currently always returns an empty
                dict.

        Returns:
            Tuple `(upscaled_dem, upscaled_fdir, metrics)`.

        Raises:
            NotImplementedError: For `scale_factor > 1` — the iterative
                core is deferred.
        """
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1; got {scale_factor}")
        if scale_factor == 1:
            up_dem, up_fdir = self.upscale(
                scale_factor=1,
                method="cotat",
                accumulation=accumulation,
                dem=dem,
            )
            return up_dem, up_fdir, {}

        fdir_arr = self.read_array().astype(np.int32, copy=False)
        acc_arr = accumulation.read_array().astype(np.float64, copy=False)
        if fdir_arr.shape != acc_arr.shape:
            raise ValueError(
                f"accumulation shape {acc_arr.shape} != flow_direction "
                f"shape {fdir_arr.shape}"
            )

        coarse_fdir, metrics, outlets = ihu_upscale(
            fdir_arr,
            acc_arr,
            scale_factor,
            max_iter=max_iter,
        )
        # Replace -1 with the dataset no-data sentinel for on-disk consistency.
        coarse_fdir = np.where(
            coarse_fdir < 0,
            np.int32(Dataset.default_no_data_value),
            coarse_fdir,
        )

        gt = self.geotransform
        coarse_gt = (
            gt[0],
            gt[1] * scale_factor,
            gt[2],
            gt[3],
            gt[4],
            gt[5] * scale_factor,
        )
        plain_fdir = Dataset.from_array(
            coarse_fdir,
            geo_ref=GeoReference(geo=coarse_gt, epsg=self.epsg),
            no_data_value=Dataset.default_no_data_value,
        )
        upscaled_fdir = _flow_direction_cls().from_dataset(
            plain_fdir,
            routing="d8",
            encoding=self.encoding,
        )

        upscaled_dem = None
        if dem is not None:
            from digitalrivers.dem import DEM as _DEM

            out_rows = fdir_arr.shape[0] // scale_factor
            out_cols = fdir_arr.shape[1] // scale_factor
            coarse_z = np.full(
                (out_rows, out_cols),
                Dataset.default_no_data_value,
                dtype=np.float32,
            )
            z_arr = dem.read_array().astype(np.float64, copy=False)
            for (br, bc), out in outlets.items():
                fr = int(out[1])
                fc = int(out[2])
                coarse_z[br, bc] = _pick_coarse_elev(z_arr, fr, fc)
            plain_dem = Dataset.from_array(
                coarse_z,
                geo_ref=GeoReference(geo=coarse_gt, epsg=self.epsg),
                no_data_value=Dataset.default_no_data_value,
            )
            upscaled_dem = _DEM(plain_dem.raster)

        return upscaled_dem, upscaled_fdir, (metrics if report else {})

    def upscale(
        self,
        scale_factor: int,
        method: str = "cotat",
        accumulation=None,
        dem=None,
        area_threshold_cells: int | None = None,
    ) -> tuple:
        """Upscale the flow-direction raster by an integer factor.

        Three classical methods are specified by P18; this initial
        implementation ships `"cotat"` (Reed 2003 — Cell Outlet Tracing
        with an Area Threshold). EAM (Olivera 2002) and DMM raise
        `NotImplementedError` pending a follow-up.

        COTAT algorithm (per coarse cell):

        1. Find the fine cell with the highest accumulation in the
           scale_factor × scale_factor block. This is the coarse cell's
           outlet.
        2. Trace downstream from that fine outlet along the fine
           `fdir` until leaving the block.
        3. The direction from the source coarse cell to the destination
           coarse cell becomes the coarse cell's D8 flow direction.

        Args:
            scale_factor: Integer aggregation factor (>= 1).
            method: `"cotat"` (default); `"eam"` / `"dmm"` raise
                `NotImplementedError`.
            accumulation: `Accumulation` aligned to this FlowDirection;
                required for COTAT (used to pick the per-block outlet).
            dem: Optional `DEM` aligned to this FlowDirection — when
                supplied, the returned `upscaled_dem` reports the
                elevation of each coarse cell's outlet (Reed 2003).
            area_threshold_cells: Reserved for COTAT+ branch-cutoff
                refinement; currently ignored.

        Returns:
            Tuple `(upscaled_dem, upscaled_fdir)`. If `dem` is
            `None` the first element is `None` and the caller is
            expected to recompute elevations from a coarsened DEM.

        Raises:
            NotImplementedError: For methods other than `"cotat"`.
            ValueError: If `scale_factor < 1` or `accumulation` is
                missing for COTAT.
        """
        from digitalrivers.flow.accumulation import Accumulation

        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1; got {scale_factor}")
        if scale_factor == 1:
            return (
                dem,
                _flow_direction_cls().from_dataset(
                    Dataset(self.raster),
                    routing=self.routing,
                    encoding=self.encoding,
                ),
            )
        if method == "ihu":
            return self.upscale_ihu(scale_factor, accumulation, dem)[:2]
        if method in ("eam", "dmm"):
            return self._upscale_eam_or_dmm(
                scale_factor,
                method=method,
                accumulation=accumulation,
                dem=dem,
            )
        if method != "cotat":
            raise NotImplementedError(
                f"method={method!r} not yet implemented " "(only 'cotat', 'eam', 'dmm')"
            )
        if not isinstance(accumulation, Accumulation):
            raise ValueError("COTAT requires an Accumulation input")

        fdir = self.read_array().astype(np.int32, copy=False)
        acc = accumulation.read_array().astype(np.float64, copy=False)
        if fdir.shape != acc.shape:
            raise ValueError(
                f"accumulation shape {acc.shape} != flow_direction shape "
                f"{fdir.shape}"
            )

        d_row = _DIR_DR
        d_col = _DIR_DC
        rows, cols = fdir.shape
        out_rows = rows // scale_factor
        out_cols = cols // scale_factor

        # Native Numba COTAT fast path — bit-for-bit identical to the
        # pure-Python loop below; ~30-50x faster on continental DEMs.
        from digitalrivers.flow._kernels.numba import cotat_upscale_numba
        from digitalrivers.core.numba import is_numba_enabled

        if is_numba_enabled():
            coarse_fdir = cotat_upscale_numba(
                fdir,
                acc,
                scale_factor,
                d_row,
                d_col,
                np.int32(Dataset.default_no_data_value),
            )
            z = None
            coarse_z = None
            if dem is not None:
                z = dem.read_array().astype(np.float64, copy=False)
                coarse_z = np.full(
                    (out_rows, out_cols),
                    Dataset.default_no_data_value,
                    dtype=np.float32,
                )
                for br in range(out_rows):
                    for bc in range(out_cols):
                        block_acc = acc[
                            br * scale_factor : (br + 1) * scale_factor,
                            bc * scale_factor : (bc + 1) * scale_factor,
                        ]
                        idx = int(np.argmax(block_acc))
                        fr = br * scale_factor + idx // scale_factor
                        fc = bc * scale_factor + idx % scale_factor
                        coarse_z[br, bc] = _pick_coarse_elev(z, fr, fc)
            return self._wrap_upscaled(coarse_fdir, scale_factor, coarse_z)

        coarse_fdir = np.full(
            (out_rows, out_cols),
            Dataset.default_no_data_value,
            dtype=np.int32,
        )

        z = None
        coarse_z = None
        if dem is not None:
            z = dem.read_array().astype(np.float64, copy=False)
            coarse_z = np.full(
                (out_rows, out_cols),
                Dataset.default_no_data_value,
                dtype=np.float32,
            )

        for br in range(out_rows):
            for bc in range(out_cols):
                r_lo = br * scale_factor
                r_hi = r_lo + scale_factor
                c_lo = bc * scale_factor
                c_hi = c_lo + scale_factor
                block = acc[r_lo:r_hi, c_lo:c_hi]
                best = np.unravel_index(int(np.argmax(block)), block.shape)
                fr = r_lo + int(best[0])
                fc = c_lo + int(best[1])
                if z is not None:
                    coarse_z[br, bc] = _pick_coarse_elev(z, fr, fc)
                r, c = fr, fc
                # Trace downstream until exiting the block.
                while True:
                    d = int(fdir[r, c])
                    if d < 0 or d > 7:
                        break
                    nr = r + int(d_row[d])
                    nc = c + int(d_col[d])
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        break
                    coarse_dr = (nr // scale_factor) - br
                    coarse_dc = (nc // scale_factor) - bc
                    if coarse_dr != 0 or coarse_dc != 0:
                        # Single fine D8 step crosses at most one coarse cell
                        # boundary, so coarse_dr and coarse_dc are each in
                        # {-1, 0, 1} and the lookup is guaranteed to hit. The
                        # guard below catches any future regression that
                        # widens fine-grid steps (e.g. a non-D8 routing) and
                        # would otherwise silently leave the coarse cell at
                        # no-data.
                        matched = False
                        for k in range(8):
                            if (
                                int(d_row[k]) == coarse_dr
                                and int(d_col[k]) == coarse_dc
                            ):
                                coarse_fdir[br, bc] = k
                                matched = True
                                break
                        if not matched:
                            raise RuntimeError(
                                f"COTAT offset lookup failed at coarse cell "
                                f"({br}, {bc}) for fine step "
                                f"({coarse_dr}, {coarse_dc}); the routing "
                                f"produced a multi-coarse-cell step which "
                                f"COTAT cannot encode."
                            )
                        break
                    r, c = nr, nc

        return self._wrap_upscaled(coarse_fdir, scale_factor, coarse_z)

    def _wrap_upscaled(
        self,
        coarse_fdir: np.ndarray,
        scale_factor: int,
        coarse_z: np.ndarray | None = None,
    ):
        """Wrap the coarse arrays as typed rasters on the upscaled grid.

        The tail every upscaling method shares: derive the coarse geotransform by
        stretching the source cell size, then wrap the flow-direction array — and
        the coarse elevation array when there is one — as typed results.

        Args:
            coarse_fdir: Upscaled D8 codes, shaped to the coarse grid.
            scale_factor: Cells of the source grid per coarse cell.
            coarse_z: Upscaled elevations, or `None` when the caller upscaled no
                DEM. Decides whether a `DEM` comes back.

        Returns:
            `(upscaled_dem, upscaled_fdir)`, the first `None` when `coarse_z` is.
        """
        gt = self.geotransform
        geo_ref = GeoReference(
            geo=(
                gt[0],
                gt[1] * scale_factor,
                gt[2],
                gt[3],
                gt[4],
                gt[5] * scale_factor,
            ),
            epsg=self.epsg,
        )
        upscaled_fdir = _flow_direction_cls().from_dataset(
            Dataset.from_array(
                coarse_fdir,
                geo_ref=geo_ref,
                no_data_value=Dataset.default_no_data_value,
            ),
            routing="d8",
            encoding=self.encoding,
        )
        if coarse_z is None:
            return None, upscaled_fdir
        # Circular-import break: dem imports flow_direction at module load.
        from digitalrivers.dem import DEM as _DEM

        plain_dem = Dataset.from_array(
            coarse_z,
            geo_ref=geo_ref,
            no_data_value=Dataset.default_no_data_value,
        )
        return _DEM(plain_dem.raster), upscaled_fdir

    def _upscale_eam_or_dmm(
        self,
        scale_factor: int,
        method: str,
        accumulation=None,
        dem=None,
    ) -> tuple:
        """EAM (Olivera 2002) / DMM upscalers — voting-based variants.

        For each coarse cell, every fine cell inside the block traces
        downstream until it exits the block; the exit direction (in coarse
        coordinates) is the fine cell's vote. The winning direction:

        - `"dmm"`: most-voted direction. Each fine cell votes with
          weight 1.
        - `"eam"`: most accumulation-weighted direction. Each fine cell
          votes with weight = its accumulation (so high-accumulation cells
          dominate the choice).

        Args:
            scale_factor: Integer coarsening factor.
            method: `"eam"` or `"dmm"`.
            accumulation: Required for `"eam"` (provides per-cell vote
                weight). Ignored for `"dmm"`.
            dem: Optional input DEM; when supplied, the coarse-grid DEM
                reports the elevation of each coarse cell's COTAT-style
                outlet (highest-accumulation fine cell in the block).

        Returns:
            Tuple `(upscaled_dem, upscaled_fdir)`.
        """
        from digitalrivers.flow.accumulation import Accumulation

        fdir = self.read_array().astype(np.int32, copy=False)
        rows, cols = fdir.shape
        gt = self.geotransform
        out_rows = rows // scale_factor
        out_cols = cols // scale_factor

        if method == "eam":
            if not isinstance(accumulation, Accumulation):
                raise ValueError("EAM upscaling requires an Accumulation input")
            weights = accumulation.read_array().astype(np.float64, copy=False)
        else:  # dmm
            weights = None

        d_row = _DIR_DR
        d_col = _DIR_DC
        coarse_fdir = np.full(
            (out_rows, out_cols),
            Dataset.default_no_data_value,
            dtype=np.int32,
        )

        for br in range(out_rows):
            for bc in range(out_cols):
                r_lo = br * scale_factor
                r_hi = r_lo + scale_factor
                c_lo = bc * scale_factor
                c_hi = c_lo + scale_factor
                votes = np.zeros(8, dtype=np.float64)
                for fr in range(r_lo, r_hi):
                    for fc in range(c_lo, c_hi):
                        r = fr
                        c = fc
                        w = 1.0 if weights is None else float(weights[fr, fc])
                        while True:
                            d = int(fdir[r, c])
                            if d < 0 or d > 7:
                                break
                            nr = r + int(d_row[d])
                            nc = c + int(d_col[d])
                            if not (0 <= nr < rows and 0 <= nc < cols):
                                break
                            if not (r_lo <= nr < r_hi and c_lo <= nc < c_hi):
                                coarse_dr = (nr // scale_factor) - br
                                coarse_dc = (nc // scale_factor) - bc
                                for k in range(8):
                                    if (
                                        int(d_row[k]) == coarse_dr
                                        and int(d_col[k]) == coarse_dc
                                    ):
                                        votes[k] += w
                                        break
                                break
                            r = nr
                            c = nc
                if votes.max() > 0:
                    coarse_fdir[br, bc] = int(np.argmax(votes))

        coarse_gt = (
            gt[0],
            gt[1] * scale_factor,
            gt[2],
            gt[3],
            gt[4],
            gt[5] * scale_factor,
        )
        plain_fdir = Dataset.from_array(
            coarse_fdir,
            geo_ref=GeoReference(geo=coarse_gt, epsg=self.epsg),
            no_data_value=Dataset.default_no_data_value,
        )
        upscaled_fdir = _flow_direction_cls().from_dataset(
            plain_fdir,
            routing="d8",
            encoding=self.encoding,
        )
        upscaled_dem = None
        if dem is not None and weights is not None:
            from digitalrivers.dem import DEM as _DEM

            coarse_z = np.full(
                (out_rows, out_cols),
                Dataset.default_no_data_value,
                dtype=np.float32,
            )
            z = dem.read_array().astype(np.float64, copy=False)
            for br in range(out_rows):
                for bc in range(out_cols):
                    block_acc = weights[
                        br * scale_factor : (br + 1) * scale_factor,
                        bc * scale_factor : (bc + 1) * scale_factor,
                    ]
                    idx = int(np.argmax(block_acc))
                    fr = br * scale_factor + idx // scale_factor
                    fc = bc * scale_factor + idx % scale_factor
                    coarse_z[br, bc] = _pick_coarse_elev(z, fr, fc)
            plain_dem = Dataset.from_array(
                coarse_z,
                geo_ref=GeoReference(geo=coarse_gt, epsg=self.epsg),
                no_data_value=Dataset.default_no_data_value,
            )
            upscaled_dem = _DEM(plain_dem.raster)
        return upscaled_dem, upscaled_fdir
