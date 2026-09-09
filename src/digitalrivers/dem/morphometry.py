"""Morphometric indices: what the surface looks like, cell by cell.

These describe terrain shape rather than routing. They fall into three families:

* **Slope and its compounds** -- `slope` (steepest D8 descent), and the wetness and
  transport indices built on it: `twi`, `spi`, `sti`. Each combines slope with upstream
  area, so they take an `Accumulation` alongside the DEM.
* **Focal statistics** -- `tpi`, `deviation_from_mean`, `elev_std`, `ruggedness` and
  `normal_vector_deviation` compare each cell to a moving window around it. They share
  one windowing helper, so the window convention and the no-data handling are stated
  once rather than five times.
* **Visibility** -- `openness` (Yokoyama 2002) and `sky_view_factor` walk outward along
  eight azimuths to find the horizon in each direction. Both run over the same Numba
  kernel and differ only in how the eight angles are aggregated.

`curvature` stands slightly apart: it fits a local quadratic and reports profile, plan
or total curvature from its coefficients.

`DEM.slope` and `Terrain.slope` are different calculations rather than duplicates: this
one is the maximum D8 descent computed in numpy, while `Terrain.slope` is GDAL's Horn
estimator over the full 3x3 window.
"""

from __future__ import annotations

import numpy as np
from pyramids.dataset import Dataset, GeoReference

from digitalrivers.dem._kernels.surface import (
    eight_direction_slopes,
    focal_window_stats,
)


__all__ = ["MorphometryMixin"]


class MorphometryMixin:
    """Morphometric index methods, mixed into :class:`DEM`."""

    def _get_8_direction_slopes(self) -> np.ndarray:
        """Compute slopes to all eight neighbours for every cell.

        Uses a padded elevation array and vectorised NumPy slicing to
        calculate the elevation difference divided by the inter-cell
        distance (cell size for cardinal, cell size × √2 for diagonal)
        in each of the eight D8 directions.

        Returns:
            np.ndarray: 3-D `float32` array of shape
                `(rows, columns, 8)` where the third axis corresponds
                to the direction indices defined in `DIR_OFFSETS`.
        """
        return eight_direction_slopes(self.values, self.cell_size)

    def twi(
        self,
        accumulation,
        slope_deg: Dataset | None = None,
    ) -> Dataset:
        """Topographic Wetness Index (Beven & Kirkby 1979).

        TWI = ln(SCA / tan(slope)) where SCA = specific catchment area =
        `accumulation_cells * cell_size`. High TWI values mark cells likely
        to be wet (low slope and / or large upstream area). Slope is treated
        in radians internally; values below a small floor (≈ 0.06°) are
        clamped to avoid log-singularity at flat cells.

        Args:
            accumulation: `Accumulation` raster aligned to this DEM.
            slope_deg: Optional pre-computed slope raster in degrees. If
                None, `Terrain.slope`-equivalent is computed on the fly via
                `arctan` of `DEM.slope()`'s rise/run output.

        Returns:
            `Dataset` of float32 TWI values. No-data sentinel `-9999.0`.

        References:
            Beven, K. J. & Kirkby, M. J. (1979). "A physically based,
            variable contributing area model of basin hydrology."
            *Hydrological Sciences Bulletin* 24(1): 43-69.
        """
        return self._area_slope_index(accumulation, slope_deg, kind="twi")

    def spi(
        self,
        accumulation,
        slope_deg: Dataset | None = None,
    ) -> Dataset:
        """Stream Power Index (Moore et al. 1991).

        SPI = SCA * tan(slope). Proportional to the rate at which overland
        flow does work at a cell; useful as a proxy for erosion risk.

        Args:
            accumulation: `Accumulation` raster aligned to this DEM.
            slope_deg: Optional pre-computed slope raster in degrees. If
                None, computed on the fly.

        Returns:
            `Dataset` of float32 SPI values. No-data sentinel `-9999.0`.
        """
        return self._area_slope_index(accumulation, slope_deg, kind="spi")

    def sti(
        self,
        accumulation,
        slope_deg: Dataset | None = None,
    ) -> Dataset:
        """Sediment Transport Index (Moore & Burch 1986).

        STI = (SCA / 22.13)^0.6 * (sin(slope) / 0.0896)^1.3. The 22.13 m and
        0.0896 m/m constants come from the original USLE plot length and
        slope; STI predicts where overland flow will transport sediment
        rather than deposit it.

        Args:
            accumulation: `Accumulation` raster aligned to this DEM.
            slope_deg: Optional pre-computed slope raster in degrees.

        Returns:
            `Dataset` of float32 STI values. No-data sentinel `-9999.0`.
        """
        return self._area_slope_index(accumulation, slope_deg, kind="sti")

    def _area_slope_index(
        self,
        accumulation,
        slope_deg,
        *,
        kind: str,
    ) -> Dataset:
        """Shared kernel for the TWI / SPI / STI family.

        All three indices need `(SCA, slope_rad)`; the difference is the
        functional form applied to them.
        """
        if slope_deg is None:
            slope_ratio = self.slope().read_array().astype(np.float64, copy=False)
            slope_deg_arr = np.degrees(
                np.arctan(np.where(np.isfinite(slope_ratio), slope_ratio, 0.0))
            )
        else:
            slope_deg_arr = slope_deg.read_array().astype(np.float64, copy=False)
            no_val = slope_deg.no_data_value[0] if slope_deg.no_data_value else None
            if no_val is not None:
                slope_deg_arr = np.where(slope_deg_arr == no_val, np.nan, slope_deg_arr)

        acc = accumulation.read_array().astype(np.float64, copy=False)
        if slope_deg_arr.shape != acc.shape:
            raise ValueError(
                f"slope shape {slope_deg_arr.shape} != accumulation shape "
                f"{acc.shape}"
            )

        slope_rad = np.deg2rad(slope_deg_arr)
        # Floor at ~0.001 rad (≈ 0.06°) to keep tan() / sin() bounded away
        # from zero on flats.
        slope_rad = np.where(
            np.isfinite(slope_rad),
            np.maximum(slope_rad, 1.0e-3),
            np.nan,
        )

        cs = float(abs(self.geotransform[1]))
        sca = acc * cs  # specific catchment area (m, since width ≈ cell_size)

        with np.errstate(divide="ignore", invalid="ignore"):
            if kind == "twi":
                arr = np.log(sca / np.tan(slope_rad))
            elif kind == "spi":
                arr = sca * np.tan(slope_rad)
            elif kind == "sti":
                arr = ((sca / 22.13) ** 0.6) * ((np.sin(slope_rad) / 0.0896) ** 1.3)
            else:  # pragma: no cover — defensive
                raise ValueError(f"unknown kind {kind!r}")

        no_val = -9999.0
        arr = np.where(np.isfinite(arr), arr, no_val).astype(np.float32)
        return Dataset.from_array(
            arr,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def _focal_window_stats(
        self,
        window: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared focal-window kernel for the terrain-index family.

        Returns `(z, focal_mean, focal_sd)` where the focal stats are
        rectangular `window×window` reductions, **no-data-aware**: the mean
        and SD at each cell are computed only over the valid (non-NaN)
        cells inside its window. Cells whose window contains zero valid
        neighbours receive NaN — the per-caller wrapping converts those to
        the DEM's no-data sentinel.

        Used by `tpi`, `deviation_from_mean`, and `elev_std`. `ruggedness`
        has its own per-shift kernel and does not call this helper.
        """
        return focal_window_stats(self.values, window)

    def tpi(self, window: int = 3) -> Dataset:
        """Topographic Position Index (Guisan 1999).

        TPI = `z - focal_mean(z, window)`. Positive values mark ridges and
        upland positions; negative values mark valleys and depressions.

        Args:
            window: Side length of the focal window in cells (must be ≥ 1).
                Defaults to 3 (a 3×3 neighbourhood). Larger windows pick up
                regional / catchment-scale topography; smaller windows pick
                up local relief.

        Returns:
            `Dataset` of float32 TPI values. No-data cells use this DEM's
            no-data sentinel.

        References:
            Guisan, A., Weiss, S. B., & Weiss, A. D. (1999). "GLM versus
            CCA spatial modeling of plant species distribution." *Plant
            Ecology* 143(1): 107-122.

        Examples:
            - A flat surface has every cell at its focal mean → TPI = 0:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> tpi = DEM(ds.raster).tpi(window=3).read_array()
                >>> bool(np.allclose(tpi, 0.0))
                True

            - A single ridge cell on flat terrain reports positive TPI; a
              pit reports negative:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[2, 2] = 9.0
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> bool(DEM(ds.raster).tpi(window=3).read_array()[2, 2] > 0)
                True
        """
        z, focal_mean, _focal_sd = self._focal_window_stats(window)
        out = (z - focal_mean).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(out), no_val, out)
        return Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def deviation_from_mean(self, window: int = 3) -> Dataset:
        """Deviation from mean elevation — standardised TPI.

        `(z - focal_mean) / focal_sd`. Dimensionless ridge / valley index;
        because it normalises by local roughness it allows comparing
        positions across regimes with very different relief.

        Args:
            window: Side length of the focal window in cells (≥ 1).

        Returns:
            `Dataset` of float32 deviation values. Flat cells (focal_sd ≈ 0)
            yield 0.0 by definition. No-data cells use this DEM's no-data
            sentinel.

        Examples:
            - Flat terrain yields zero deviation everywhere:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((4, 4), 5.0, dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).deviation_from_mean(window=3).read_array(), 0.0
                ... ))
                True

            - A peak reports positive standardised deviation:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[2, 2] = 10.0
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> out = DEM(ds.raster).deviation_from_mean(window=3).read_array()
                >>> bool(out[2, 2] > 0)
                True
        """
        z, focal_mean, focal_sd = self._focal_window_stats(window)
        # A standard deviation is never negative, so `> 0` picks out exactly the
        # degenerate flat-window case without an equality test on a computed float.
        out = (z - focal_mean) / np.where(focal_sd > 0.0, focal_sd, 1.0)
        out = out.astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(out), no_val, out)
        return Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def elev_std(self, window: int = 3) -> Dataset:
        """Standard deviation of elevation in a focal window.

        Pure focal-window SD on the elevation raster. A roughness proxy:
        high values mark varied terrain, low values mark smooth terrain.

        Args:
            window: Side length of the focal window in cells (≥ 1).

        Returns:
            `Dataset` of float32 SD values. No-data cells use this DEM's
            no-data sentinel.

        Examples:
            - Flat terrain reports zero SD everywhere:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((4, 4), 5.0, dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).elev_std(window=3).read_array(), 0.0
                ... ))
                True

            - A step in elevation produces non-zero SD along the boundary:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[:, 3:] = 10.0
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> sd = DEM(ds.raster).elev_std(window=3).read_array()
                >>> bool((sd[:, 2] > 0).all())
                True
        """
        _z, _focal_mean, focal_sd = self._focal_window_stats(window)
        out = focal_sd.astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(out), no_val, out)
        return Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def curvature(self, kind: str = "profile") -> Dataset:
        """Surface curvature (Zevenbergen & Thorne 1987).

        Fits a partial quartic polynomial `z = Ax²y² + Bx²y + Cxy² + Dx² +
        Ey² + Fxy + Gx + Hy + I` to the 3×3 neighbourhood of each cell and
        evaluates one of the five canonical curvature variants from the
        coefficient grid:

        * `"plan"` — curvature perpendicular to the slope direction;
          positive on diverging slopes, negative on converging ones.
        * `"profile"` — curvature parallel to the slope direction; positive
          on convex (decelerating) slopes, negative on concave (accelerating)
          ones.
        * `"total"` — `2 * (D + E)`; sign-independent total relief.
        * `"mean"` — average of the two principal curvatures.
        * `"gaussian"` — product of the two principal curvatures.

        Args:
            kind: One of `"plan"`, `"profile"`, `"total"`, `"mean"`,
                `"gaussian"`. Defaults to `"profile"`.

        Returns:
            `Dataset` of float32 curvature values. No-data cells use this
            DEM's no-data sentinel.

        Raises:
            ValueError: If `kind` is not one of the five recognised
                variants.

        References:
            Zevenbergen, L. W. & Thorne, C. R. (1987). "Quantitative
            analysis of land surface topography." *Earth Surface Processes
            and Landforms* 12(1): 47-56.

        Examples:
            - Every curvature variant is zero on a flat DEM:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).curvature(kind="total").read_array(), 0.0
                ... ))
                True

            - Mean curvature equals total / 2 on a paraboloid (interior):

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
                >>> z = (-(x * x + y * y)).astype(np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> total = dem.curvature(kind="total").read_array()[2:-2, 2:-2]
                >>> mean = dem.curvature(kind="mean").read_array()[2:-2, 2:-2]
                >>> bool(np.allclose(mean, total / 2.0, atol=1e-5))
                True
        """
        if kind not in ("plan", "profile", "total", "mean", "gaussian"):
            raise ValueError(
                f"kind must be one of 'plan', 'profile', 'total', 'mean', "
                f"'gaussian'; got {kind!r}"
            )
        z = self.values.astype(np.float64, copy=False)
        L = float(abs(self.geotransform[1]))
        # 3×3 stencil — `np.pad(mode="edge")` mirrors the boundary value so
        # finite differences stay defined at the raster edge.
        zp = np.pad(np.where(np.isnan(z), 0.0, z), 1, mode="edge")
        z1, z2, z3 = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
        z4, z5, z6 = zp[1:-1, :-2], zp[1:-1, 1:-1], zp[1:-1, 2:]
        z7, z8, z9 = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
        # Zevenbergen-Thorne polynomial coefficients.
        D = ((z4 + z6) / 2.0 - z5) / (L * L)
        E = ((z2 + z8) / 2.0 - z5) / (L * L)
        F = (-z1 + z3 + z7 - z9) / (4.0 * L * L)
        G = (-z4 + z6) / (2.0 * L)
        H = (z2 - z8) / (2.0 * L)
        denom = G * G + H * H + 1.0e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "plan":
                arr = -2.0 * (D * H * H + E * G * G - F * G * H) / denom
            elif kind == "profile":
                arr = 2.0 * (D * G * G + E * H * H + F * G * H) / denom
            elif kind == "total":
                arr = 2.0 * (D + E)
            elif kind == "mean":
                arr = D + E
            else:  # gaussian
                arr = 4.0 * D * E - F * F
        out = arr.astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z) | ~np.isfinite(out), no_val, out)
        return Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def normal_vector_deviation(self, window: int = 3) -> Dataset:
        """Per-cell angular deviation of the surface normal from its focal mean.

        Computes each cell's outward-pointing surface normal from finite
        differences of the elevation grid, then takes the focal-mean of the
        unit-normal components in a `window×window` neighbourhood. The
        result at each cell is the angle (in radians) between the local
        normal and the focal-mean normal — a roughness metric that grows
        with how strongly the surface bends within the window.

        Args:
            window: Side length of the focal window in cells (≥ 1).

        Returns:
            `Dataset` of float32 angular deviations in radians,
            `[0, π/2]`. No-data cells use this DEM's no-data sentinel.

        Examples:
            - Flat terrain yields zero angular deviation:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).normal_vector_deviation(window=3).read_array(),
                ...     0.0, atol=1e-5,
                ... ))
                True

            - A constant-slope plane has identical normals in its deep
              interior, so deviation there is ~0:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> x, y = np.meshgrid(np.arange(7), np.arange(7))
                >>> z = (2.0 * x + y).astype(np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).normal_vector_deviation(window=3).read_array()
                >>> bool(np.allclose(arr[2:-2, 2:-2], 0.0, atol=1e-4))
                True
        """
        from scipy.ndimage import uniform_filter

        if window < 1:
            raise ValueError(f"window must be >= 1; got {window!r}")
        z = self.values.astype(np.float64, copy=False)
        L = float(abs(self.geotransform[1]))
        # Finite-difference partials with reflective edge padding.
        zp = np.pad(np.where(np.isnan(z), 0.0, z), 1, mode="edge")
        dzdx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2.0 * L)
        dzdy = (zp[2:, 1:-1] - zp[:-2, 1:-1]) / (2.0 * L)
        # Outward-pointing normal `(-dz/dx, -dz/dy, 1)`; renormalise to unit.
        nx_raw = -dzdx
        ny_raw = -dzdy
        nz_raw = np.ones_like(z)
        nm = np.sqrt(nx_raw * nx_raw + ny_raw * ny_raw + nz_raw * nz_raw)
        nx = nx_raw / nm
        ny = ny_raw / nm
        nz = nz_raw / nm
        # Focal mean of each component, then renormalise to keep the mean
        # vector unit-length.
        mnx = uniform_filter(nx, size=int(window), mode="reflect")
        mny = uniform_filter(ny, size=int(window), mode="reflect")
        mnz = uniform_filter(nz, size=int(window), mode="reflect")
        mm = np.sqrt(mnx * mnx + mny * mny + mnz * mnz) + 1.0e-12
        mnx /= mm
        mny /= mm
        mnz /= mm
        cos_theta = np.clip(nx * mnx + ny * mny + nz * mnz, -1.0, 1.0)
        out = np.arccos(cos_theta).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z), no_val, out)
        return Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def openness(
        self,
        *,
        search_radius: int = 10,
        kind: str = "positive",
    ) -> Dataset:
        """Topographic openness (Yokoyama 2002).

        For each cell, walks outward along 8 azimuths up to `search_radius`
        cells and records the maximum elevation angle (positive openness)
        or the minimum (negative openness) along each walk. The per-cell
        output is the mean of `(π/2 - horizon_angle)` across the 8
        directions, in radians.

        High positive openness marks exposed / high-relief locations; high
        negative openness marks deep depressions / valley floors.

        Args:
            search_radius: Maximum walk distance in cells. Must be ≥ 1.
                Defaults to 10.
            kind: `"positive"` (default) or `"negative"`. Negative openness
                flips the sign of the elevation difference internally —
                effectively measuring the local pit / depression depth.

        Returns:
            `Dataset` of float32 openness values in radians. No-data cells
            use this DEM's no-data sentinel.

        Raises:
            ValueError: If `kind` is not one of `"positive"` / `"negative"`
                or `search_radius < 1`.

        References:
            Yokoyama, R., Shirasawa, M., & Pike, R. J. (2002). "Visualizing
            topography by openness: A new application of image processing
            to digital elevation models." *Photogrammetric Engineering and
            Remote Sensing* 68(3): 257-265.

        Examples:
            - Flat terrain: every horizon angle is 0, so positive openness
              is `π/2` at every cell:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).openness(search_radius=2).read_array()
                >>> bool(np.allclose(arr, np.pi / 2.0, atol=1e-5))
                True

            - A peak on flat terrain has strictly larger positive openness
              than its neighbours (which look up at it):

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[2, 2] = 10.0
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).openness(search_radius=3).read_array()
                >>> bool(arr[2, 2] > arr[1, 2])
                True
        """
        from digitalrivers.dem._kernels.morphometry import horizon_walk_kernel

        if kind not in ("positive", "negative"):
            raise ValueError(f"kind must be 'positive' or 'negative'; got {kind!r}")
        if search_radius < 1:
            raise ValueError(f"search_radius must be >= 1; got {search_radius!r}")
        z = self.values.astype(np.float64, copy=False)
        # For negative openness, flip the elevation so the kernel's
        # "maximum upward angle" becomes "maximum downward angle" relative
        # to the original surface.
        z_in = (-z if kind == "negative" else z).astype(np.float64, copy=False)
        z_filled = np.where(np.isnan(z_in), 0.0, z_in)
        out = horizon_walk_kernel(
            z_filled,
            float(abs(self.geotransform[1])),
            int(search_radius),
            0,
        ).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z), no_val, out)
        return Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def sky_view_factor(
        self,
        *,
        search_radius: int = 10,
    ) -> Dataset:
        """Sky-view factor (Zakšek et al. 2011).

        For each cell, the fraction of the upper hemisphere that is visible
        from that cell. Walks along 8 azimuths up to `search_radius`,
        records the maximum elevation angle along each walk, and returns
        the mean of `(1 - sin(horizon_angle))` across the 8 directions —
        equivalent to the fraction of an isotropic sky dome not occluded
        by surrounding terrain.

        Shares the horizon-walk kernel with `openness` (W-27); the two
        differ only in the per-direction aggregation function.

        Args:
            search_radius: Maximum walk distance in cells. Must be ≥ 1.
                Defaults to 10.

        Returns:
            `Dataset` of float32 SVF values in `[0, 1]`. No-data cells use
            this DEM's no-data sentinel.

        Raises:
            ValueError: If `search_radius < 1`.

        References:
            Zakšek, K., Oštir, K., & Kokalj, Ž. (2011). "Sky-view factor as
            a relief visualization technique." *Remote Sensing* 3(2):
            398-415.

        Examples:
            - Flat terrain: nothing occludes the sky, SVF = 1 everywhere:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).sky_view_factor(search_radius=2).read_array()
                >>> bool(np.allclose(arr, 1.0, atol=1e-5))
                True

            - A pit surrounded by higher cells reports SVF strictly less
              than 1 (the walls occlude part of the sky):

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> z[2, 2] = 0.0
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).sky_view_factor(search_radius=2).read_array()
                >>> bool(arr[2, 2] < 1.0)
                True
        """
        from digitalrivers.dem._kernels.morphometry import horizon_walk_kernel

        if search_radius < 1:
            raise ValueError(f"search_radius must be >= 1; got {search_radius!r}")
        z = self.values.astype(np.float64, copy=False)
        z_filled = np.where(np.isnan(z), 0.0, z)
        out = horizon_walk_kernel(
            z_filled,
            float(abs(self.geotransform[1])),
            int(search_radius),
            1,
        ).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z), no_val, out)
        return Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def ruggedness(self, window: int = 3) -> Dataset:
        """Terrain Ruggedness Index — Wilson (2007) mean-absolute-difference form.

        Per-cell mean of absolute elevation differences to every other cell
        in a `window×window` neighbourhood. Output unit is the DEM elevation
        unit (metres). Higher values mark rougher terrain; flat terrain is
        zero.

        Note:
            This implements the **Wilson et al. (2007)** mean-absolute-
            difference variant, *not* Riley's original root-sum-square TRI.
            For the Riley form use `Terrain.tri(algorithm="Riley")` (the
            GDAL default), which equals `sqrt(Σ(z_c − z_i)²)` over the 3x3
            neighbourhood; `Terrain.tri(algorithm="Wilson")` reproduces
            this method's formula on a fixed 3x3 window.

        Args:
            window: Side length of the focal window in cells (≥ 1).
                Defaults to 3 (the original 3×3 neighbourhood).

        Returns:
            `Dataset` of float32 ruggedness values. No-data cells use this
            DEM's no-data sentinel.

        References:
            Wilson, M. F. J., O'Connell, B., Brown, C., Guinan, J. C., &
            Grehan, A. J. (2007). "Multiscale terrain analysis of
            multibeam bathymetry data for habitat mapping on the
            continental slope." *Marine Geodesy* 30(1-2): 3-35.

            Riley, S. J., DeGloria, S. D., & Elliot, R. (1999). "A terrain
            ruggedness index that quantifies topographic heterogeneity."
            *Intermountain Journal of Sciences* 5(1-4): 23-27.

        Examples:
            - Flat terrain has zero ruggedness everywhere:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((4, 4), 5.0, dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).ruggedness(window=3).read_array(), 0.0
                ... ))
                True

            - A peak surrounded by flat terrain contributes positive
              ruggedness at the peak and its 8-neighbours:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[2, 2] = 9.0
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> tri = DEM(ds.raster).ruggedness(window=3).read_array()
                >>> bool(tri[2, 2] > 0 and tri[1, 2] > 0 and tri[3, 2] > 0)
                True
        """
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window!r}")
        z = self.values.astype(np.float64, copy=False)
        z_filled = np.where(np.isnan(z), 0.0, z)
        total = np.zeros_like(z_filled)
        count = 0
        half = int(window) // 2
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(z_filled, shift=(dr, dc), axis=(0, 1))
                total += np.abs(z_filled - shifted)
                count += 1
        out = (total / float(count)).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z), no_val, out)
        return Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def slope(
        self,
        *,
        engine: str = "auto",
        out_path: str | None = None,
        tile_size: int | tuple[int, int] = 2048,
    ) -> Dataset:
        """Compute the maximum downhill slope at every cell.

        Calculates slopes in all eight D8 directions via
        `_get_8_direction_slopes` and returns a raster whose cell
        values are the maximum slope across the eight neighbours.

        Args:
            engine: `"auto"` (default) picks the whole-array path and switches
                to the tiled one only for rasters large enough to risk
                exhausting RAM; `"tiled"` forces the streaming path and
                requires `out_path`; `"in_memory"` forces the whole-array one.
            out_path: Destination for the tiled path, which writes as it
                streams rather than materialising the result. Required when
                the tiled engine runs, ignored otherwise.
            tile_size: Tile shape for the tiled path, as an edge length or an
                explicit `(rows, columns)`. Defaults to 2048.

        Returns:
            Dataset: Single-band raster with the same geometry as the
                DEM, containing the maximum slope value per cell.

        Out-of-core:
            Slope is a local 3×3 stencil, so `engine="tiled"` streams it tile-by-tile to `out_path` with
            constant memory (bit-for-bit identical to the whole-array result). `engine="auto"` (default) only
            switches to it for rasters large enough to risk exhausting RAM; `engine="tiled"` requires `out_path`.

        See Also:
            Terrain.slope: GDAL-based slope using Horn or
                Zevenbergen-Thorne algorithms.
        """
        from digitalrivers._outofcore.engine import (  # lazy
            require_out_path,
            resolve_engine,
        )

        resolved = resolve_engine(engine, self.rows, self.columns, k=3)
        if resolved == "tiled":
            require_out_path(engine, out_path)
            # lazy import keeps `import digitalrivers` light
            from digitalrivers._outofcore.local import max_slope_2d, tiled_stencil

            cell_size = self.cell_size
            nodata = self.no_data_value[0] if self.no_data_value else None
            return tiled_stencil(
                self,
                lambda block: max_slope_2d(block, cell_size),
                out_path,
                depth=1,
                tile_size=tile_size,
                input_nodata=nodata,
            )

        slope = self._get_8_direction_slopes()
        max_slope = np.nanmax(slope, axis=2)

        src = self.dataset_like(self, max_slope)
        return src
