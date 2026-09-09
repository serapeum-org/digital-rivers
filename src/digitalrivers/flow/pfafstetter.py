"""Pfafstetter sub-basin coding for a flow-direction raster.

Pfafstetter (1989) codes nest by construction: a basin's four largest tributaries take
the even digits 2, 4, 6, 8 and the main-stem interbasins between them take the odd digits
1, 3, 5, 7, 9, counting upstream from the outlet. Appending a digit per level means a
code carries its own topology — 8421 is inside 842, which is inside 84 — so upstream /
downstream questions are answered by string prefix rather than by re-walking the network.

`PfafstetterMixin` carries the `Dataset`-level plumbing; the array walks that assign the
digits are the two private kernels at the bottom of the class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from pyramids.dataset import Dataset, GeoReference

from digitalrivers._flow.watershed import watershed_d8
from digitalrivers.core.directions import (
    DIR_DC_I32 as _DIR_DC,
    DIR_DR_I32 as _DIR_DR,
    INV_DIR as _INV_DIR,
)

if TYPE_CHECKING:
    from digitalrivers.watershed_raster import WatershedRaster

__all__ = ["PfafstetterMixin"]


class PfafstetterMixin:
    """Pfafstetter sub-basin coding, mixed into :class:`FlowDirection`.

    The mixin declares no state: `routing`, `geotransform` and `epsg` come from
    `Dataset` and from `FlowDirection`'s own constructor.
    """

    def subbasins_pfafstetter(
        self,
        accumulation,
        streams,
        level: int = 1,
        encoding: str = "packed_int",
    ) -> WatershedRaster:
        """Compute Pfafstetter (Verdin & Verdin 1999) hierarchical codes.

        Single-basin level-1 implementation: identifies the main stem (the
        path with the largest downstream-accumulation), finds the four
        tributaries with the largest accumulation at confluence with the main
        stem, and labels every cell with one of the nine Pfafstetter codes:
        `2/4/6/8` for the four main tributaries (downstream order) and
        `1/3/5/7/9` for the inter-basin segments between them.

        The multi-level recursive descent (`level > 1`) and the HydroBASINS
        iso-basin pre-split are out of scope for this initial implementation.

        Args:
            accumulation: `Accumulation` raster aligned to this
                FlowDirection. Used for ranking tributaries by area.
            streams: `StreamRaster` aligned to this FlowDirection. Defines
                the channel network the Pfafstetter scheme walks.
            level: Hierarchy depth. Only `level=1` is implemented; higher
                levels raise `NotImplementedError`.
            encoding: `"packed_int"` (default) writes codes as int32 cell
                values. `"string"` is not yet implemented.

        Returns:
            :class:`WatershedRaster` with int32 Pfafstetter codes. Level-1
            codes are in `[1, 9]`; level-N codes are N-digit concatenations
            `parent * 10 + child` (e.g. level-2 ⇒ `[11, 99]`, level-3 ⇒
            `[111, 999]`). Cells outside the basin envelope are 0.

        Raises:
            ValueError: If `level < 1` or non-D8 routing is used or an
                argument has the wrong type.
            NotImplementedError: If `encoding != "packed_int"`.

        Examples:
            - Level-1 coding on a small east-flowing DEM yields codes within
              the canonical `[1, 9]` Pfafstetter range:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [
                ...         [9, 9, 9, 9, 9, 9],
                ...         [9, 5, 4, 3, 2, 1],
                ...         [9, 9, 9, 9, 9, 9],
                ...     ],
                ...     dtype=np.float32,
                ... )
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
                >>> fd = dem.flow_direction(method="d8")
                >>> acc = fd.accumulate()
                >>> sr = acc.streams(threshold=1)
                >>> ws = fd.subbasins_pfafstetter(acc, sr, level=1)
                >>> arr = ws.read_array()
                >>> codes = sorted({int(v) for v in np.unique(arr) if v != 0})
                >>> bool(set(codes).issubset(set(range(1, 10))))
                True

            - Level-2 coding produces two-digit `parent*10 + child` codes
              (P16 multi-level backfill):

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [
                ...         [9, 9, 9, 9, 9, 9],
                ...         [9, 5, 4, 3, 2, 1],
                ...         [9, 9, 9, 9, 9, 9],
                ...     ],
                ...     dtype=np.float32,
                ... )
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
                >>> fd = dem.flow_direction(method="d8")
                >>> acc = fd.accumulate()
                >>> sr = acc.streams(threshold=1)
                >>> ws = fd.subbasins_pfafstetter(acc, sr, level=2)
                >>> arr = ws.read_array()
                >>> codes = sorted({int(v) for v in np.unique(arr) if v != 0})
                >>> bool(all(11 <= c <= 99 for c in codes))
                True

            - `level < 1` is rejected:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [[9, 9, 9], [9, 5, 9], [9, 9, 9]], dtype=np.float32
                ... )
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
                >>> fd = dem.flow_direction(method="d8")
                >>> acc = fd.accumulate()
                >>> sr = acc.streams(threshold=1)
                >>> fd.subbasins_pfafstetter(acc, sr, level=0)
                Traceback (most recent call last):
                    ...
                ValueError: level must be >= 1; got 0

        See Also:
            FlowDirection.basins: terminal-outlet partitioning of the DEM.
            FlowDirection.accumulate: upstream-area accumulation needed for
                tributary ranking.
        """
        from digitalrivers.flow.accumulation import Accumulation
        from digitalrivers.stream_raster import StreamRaster
        from digitalrivers.watershed_raster import WatershedRaster

        if level < 1:
            raise ValueError(f"level must be >= 1; got {level}")
        if encoding != "packed_int":
            raise NotImplementedError(
                f"encoding={encoding!r} not yet implemented " f"(only 'packed_int')"
            )
        if self.routing not in ("d8", "rho8"):
            raise ValueError(
                f"subbasins_pfafstetter currently supports single-direction "
                f"routing only; got {self.routing!r}"
            )
        if not isinstance(accumulation, Accumulation):
            raise ValueError("accumulation must be an Accumulation instance")
        if not isinstance(streams, StreamRaster):
            raise ValueError("streams must be a StreamRaster instance")

        fdir = self.read_array().astype(np.int32, copy=False)
        acc = accumulation.read_array().astype(np.float64, copy=False)
        stream_mask = streams.read_array().astype(bool, copy=False)
        if not (fdir.shape == acc.shape == stream_mask.shape):
            raise ValueError(
                f"Shape mismatch: fdir={fdir.shape}, "
                f"accumulation={acc.shape}, streams={stream_mask.shape}"
            )
        out = self._pfafstetter_kernel(
            fdir=fdir,
            acc=acc,
            stream_mask=stream_mask,
            basin_mask=None,
            level=level,
        )
        plain = Dataset.from_array(
            out,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=0,
        )
        ids = sorted({int(v) for v in np.unique(out) if v != 0})
        # Per-basin outlet = the cell with the highest accumulation in that
        # basin. Locate it via masked-argmax so the resulting GeoDataFrame
        # carries real coordinates rather than placeholders.
        x0, dx, _, y0, _, dy = self.geotransform
        xs: list[float] = []
        ys: list[float] = []
        for basin_id in ids:
            basin_acc = np.where(out == basin_id, acc, -np.inf)
            idx = np.unravel_index(int(np.argmax(basin_acc)), basin_acc.shape)
            xs.append(float(x0 + (int(idx[1]) + 0.5) * dx))
            ys.append(float(y0 + (int(idx[0]) + 0.5) * dy))
        outlets_gdf = gpd.GeoDataFrame(
            {"basin_id": ids},
            geometry=gpd.points_from_xy(xs, ys),
            crs=self.epsg,
        )
        return WatershedRaster.from_dataset(
            plain,
            routing=self.routing,
            outlets=outlets_gdf,
        )

    def _pfafstetter_kernel(
        self,
        fdir,
        acc,
        stream_mask,
        basin_mask,
        level: int,
    ):
        """Recursive Pfafstetter kernel.

        Computes Pfafstetter codes for the cells in `basin_mask` (or
        every cell if `basin_mask` is `None`). At `level == 1`
        returns codes in `[1, 9]`; at `level > 1` recursively
        subdivides each level-N basin into nine level-(N-1) sub-basins
        and concatenates the codes as decimal digits
        (`parent * 10 + sub`).

        Args:
            fdir / acc / stream_mask: aligned input arrays.
            basin_mask: `(rows, cols)` bool; True = cell is part of
                this basin. `None` means the whole raster.
            level: hierarchy depth (1 = single pass).

        Returns:
            `(rows, cols)` int32 of Pfafstetter codes. Cells outside
            `basin_mask` (or sub-basins with no stream cells) are 0.
        """
        out_level_1 = self._pfafstetter_level1(fdir, acc, stream_mask, basin_mask)
        if level == 1:
            return out_level_1
        out = np.zeros_like(out_level_1)
        sub_codes = [c for c in np.unique(out_level_1) if c != 0]
        for c in sub_codes:
            sub_mask = out_level_1 == c
            if not sub_mask.any():
                continue
            sub_out = self._pfafstetter_kernel(
                fdir,
                acc,
                stream_mask,
                sub_mask,
                level - 1,
            )
            # Combine: parent code shifted left + sub-code.
            shift = 10 ** (level - 1)
            sub_nonzero = sub_out != 0
            out[sub_nonzero] = int(c) * shift + sub_out[sub_nonzero]
            # Cells in sub_mask without a sub-code keep just the parent.
            untouched = sub_mask & ~sub_nonzero
            out[untouched] = int(c) * shift
        return out

    def _pfafstetter_level1(self, fdir, acc, stream_mask, basin_mask):
        """Compute level-1 Pfafstetter codes (1-9) on the cells in
        `basin_mask` (or every cell if `basin_mask` is `None`).

        Returns:
            `(rows, cols)` int32 array with codes `1..9` inside the
            basin and `0` everywhere else.
        """
        if basin_mask is None:
            basin_mask = np.ones(fdir.shape, dtype=bool)

        d_row = _DIR_DR
        d_col = _DIR_DC
        inv_dir = _INV_DIR
        rows, cols = fdir.shape

        local_stream = stream_mask & basin_mask
        masked_acc = np.where(local_stream, acc, -np.inf)
        if not np.any(np.isfinite(masked_acc)):
            return np.where(basin_mask, 1, 0).astype(np.int32)

        outlet_idx = np.unravel_index(int(np.argmax(masked_acc)), acc.shape)
        outlet_r, outlet_c = int(outlet_idx[0]), int(outlet_idx[1])

        main_stem: set[tuple[int, int]] = {(outlet_r, outlet_c)}
        # Each tributary head carries `(stem_position, accumulation, r, c)`
        # where `stem_position` is the step index along the main stem at
        # which the tributary joins (0 = outlet, increases upstream).
        # Canonical Pfafstetter ordering numbers tributaries downstream-
        # first (lowest stem_position → code 2, next → 4, ...).
        tributary_heads: list[tuple[int, float, int, int]] = []
        r, c = outlet_r, outlet_c
        stem_position = 0
        while True:
            best_in_acc = -np.inf
            best_in: tuple[int, int] | None = None
            inflows: list[tuple[float, int, int]] = []
            for k in range(8):
                ur = r + int(d_row[k])
                uc = c + int(d_col[k])
                if not (0 <= ur < rows and 0 <= uc < cols):
                    continue
                if not local_stream[ur, uc]:
                    continue
                if int(fdir[ur, uc]) != int(inv_dir[k]):
                    continue
                v = float(acc[ur, uc])
                inflows.append((v, ur, uc))
                if v > best_in_acc:
                    best_in_acc = v
                    best_in = (ur, uc)
            if best_in is None:
                break
            main_stem.add(best_in)
            for v, ur, uc in inflows:
                if (ur, uc) != best_in:
                    tributary_heads.append((stem_position, v, ur, uc))
            r, c = best_in
            stem_position += 1

        # Pick the four highest-accumulation tributaries (volume rank), then
        # order *those four* by stem position so the downstream-most one
        # gets code 2, the next upstream code 4, etc.
        tributary_heads.sort(key=lambda t: t[1], reverse=True)
        top4 = sorted(tributary_heads[:4], key=lambda t: t[0])

        out = np.zeros((rows, cols), dtype=np.int32)
        for r0, c0 in main_stem:
            out[r0, c0] = 5

        codes = [2, 4, 6, 8]
        seeds = [(int(uh[2]), int(uh[3])) for uh in top4]
        ids = codes[: len(seeds)]
        if seeds:
            sub = watershed_d8(fdir, seeds, ids, require_unique_basins=True)
            mask = (sub != 0) & basin_mask
            out[mask] = sub[mask]

        unlabelled = (out == 0) & basin_mask
        for r0 in range(rows):
            for c0 in range(cols):
                if not unlabelled[r0, c0]:
                    continue
                path: list[tuple[int, int]] = []
                rr, cc = r0, c0
                tail = 0
                while True:
                    if not basin_mask[rr, cc]:
                        break
                    if out[rr, cc] != 0:
                        tail = int(out[rr, cc])
                        break
                    path.append((rr, cc))
                    d = int(fdir[rr, cc])
                    if d < 0 or d > 7:
                        break
                    nr = rr + int(d_row[d])
                    nc = cc + int(d_col[d])
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        break
                    rr, cc = nr, nc
                if tail != 0:
                    for pr, pc in path:
                        out[pr, pc] = tail

        out[~basin_mask] = 0
        return out
