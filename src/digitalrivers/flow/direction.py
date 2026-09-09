"""Typed flow-direction raster carrying routing-scheme metadata.

The `FlowDirection` class subclasses `pyramids.dataset.Dataset` and tags the
wrapped raster with the routing scheme (`d8` / `dinf` / `mfd_quinn` /
`mfd_holmgren` / `rho8`) and the cell-value encoding convention. Its method
surface comes from two mixins as well as this module: `UpscaleMixin`
(:mod:`digitalrivers.flow.upscale`) and `PfafstetterMixin`
(:mod:`digitalrivers.flow.pfafstetter`). The `routing` argument is required at construction; there is no
default. That is the safety property: it prevents a flow-direction raster of
unknown provenance from being silently reinterpreted as D8 by a downstream
consumer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from osgeo import gdal
from pyramids.dataset import Dataset, GeoReference
from shapely.geometry import Point

from digitalrivers.watershed._kernels.watershed import watershed_d8
from digitalrivers.streams._kernels.order import _stream_outlets
from digitalrivers.core.directions import (
    DIR_DC_I32 as _DIR_DC,
    DIR_DR_I32 as _DIR_DR,
    INV_DIR as _INV_DIR,
)
from digitalrivers.core.metadata import (
    META_CLASS,
    META_ENCODING,
    META_ROUTING,
    VALID_ENCODING,
    VALID_ROUTING,
)
from digitalrivers.flow._kernels.accumulation import (
    accumulate as _accumulate_array,
    kahn_max_upslope_length,
)
from digitalrivers.flow.pfafstetter import PfafstetterMixin
from digitalrivers.flow.upscale import UpscaleMixin

if TYPE_CHECKING:
    from digitalrivers.flow.accumulation import Accumulation
    from digitalrivers.watershed.raster import WatershedRaster


class FlowDirection(UpscaleMixin, PfafstetterMixin, Dataset):
    """Flow-direction raster with routing-scheme metadata.

    Args:
        src: GDAL dataset wrapping a flow-direction raster.
        access: `"read_only"` (default) or `"write"`.
        routing: Routing scheme used to produce this raster. One of
            `"d8"`, `"dinf"`, `"mfd_quinn"`, `"mfd_holmgren"`,
            `"rho8"`. Required keyword-only argument — no default.
        encoding: Cell-value encoding convention. One of `"digitalrivers"`,
            `"taudem"`, `"esri"`, `"whitebox"`. Defaults to
            `"digitalrivers"` (the convention defined by `DIR_OFFSETS` in
            `core.directions`).
        gdal_env: GDAL config (cloud credentials, HTTP knobs) captured on
            the dataset and re-installed around its reads, so the paths that
            reopen the file authenticate the same way. Default `None`.
        open_options: GDAL open options captured on the dataset and reapplied
            when it is reopened. Default `None`.

    Raises:
        ValueError: If `routing` or `encoding` is not a recognised value.
    """

    routing: str
    encoding: str

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        *,
        routing: str,
        encoding: str = "digitalrivers",
        gdal_env: dict[str, str] | None = None,
        open_options: tuple[str, ...] | list[str] | None = None,
    ):
        """Wrap a GDAL dataset as a flow-direction raster.

        Args:
            src: Open GDAL dataset to wrap. The handle is adopted, not copied.
            access: `"read_only"` (default) or `"write"`.
            gdal_env: GDAL config (cloud credentials, HTTP knobs) captured on the
                dataset and re-installed around its reads. Default `None`.
            open_options: GDAL open options captured on the dataset and reapplied
                when it is reopened. Default `None`.
            routing: Routing scheme the raster encodes. Required keyword-only —
                there is no default, because reading a D-infinity raster as D8
                silently corrupts every derivative.
            encoding: Cell-value convention. Defaults to `"digitalrivers"`.

        Raises:
            ValueError: If `routing` or `encoding` is not a recognised value.
        """
        super().__init__(src, access, gdal_env=gdal_env, open_options=open_options)
        if routing not in VALID_ROUTING:
            raise ValueError(
                f"routing must be one of {sorted(VALID_ROUTING)}; got {routing!r}"
            )
        if encoding not in VALID_ENCODING:
            raise ValueError(
                f"encoding must be one of {sorted(VALID_ENCODING)}; got {encoding!r}"
            )
        self.routing = routing
        self.encoding = encoding

    @classmethod
    def from_dataset(
        cls,
        ds: Dataset,
        *,
        routing: str,
        encoding: str = "digitalrivers",
    ) -> FlowDirection:
        """Promote a plain `Dataset` into a `FlowDirection`.

        The source's access mode, `gdal_env` and `open_options` are carried onto
        the wrapper. Dropping them left a promoted file-backed raster unable to
        write its own metadata tags, and stripped the credentials a signed remote
        raster needs when pyramids reopens it.

        Args:
            ds: Dataset wrapping the flow-direction raster. Its raster handle is
                reused, not copied.
            routing: Routing scheme. Required keyword-only.
            encoding: Cell-value encoding convention.

        Returns:
            A `FlowDirection` over the same raster, with `ds`'s handle
            configuration.

        Examples:
            - Promote an in-memory raster and read the provenance back, and confirm the handle carries through:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import FlowDirection
                >>> plain = Dataset.from_array(
                ...     np.array([[0, 1], [2, 3]], dtype=np.int32),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
                ...     ),
                ... )
                >>> wrapped = FlowDirection.from_dataset(plain, routing="d8")
                >>> wrapped.routing, wrapped.encoding
                ('d8', 'digitalrivers')
                >>> FlowDirection.from_dataset(
                ...     plain, routing="d8"
                ... ).access == plain.access
                True

                ```
        """
        return cls(
            ds.raster,
            ds.access,
            routing=routing,
            encoding=encoding,
            gdal_env=ds.gdal_env or None,
            open_options=ds.open_options or None,
        )

    def to_dataset(self) -> Dataset:
        """Drop the typed wrapper and return the underlying `Dataset`.

        Symmetric with `from_dataset`: the access mode, `gdal_env` and
        `open_options` come back out with the raster, so a round trip does not
        silently downgrade a writable handle to a read-only one.

        Returns:
            A plain `Dataset` over the same raster and handle configuration.

        Examples:
            - Unwrap and read the grid straight off the plain `Dataset`, and confirm the handle carries through:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import FlowDirection
                >>> plain = Dataset.from_array(
                ...     np.array([[1, 2], [3, 4]], dtype=np.float32),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
                ...     ),
                ... )
                >>> wrapped = FlowDirection.from_dataset(plain, routing="d8")
                >>> plain_again = wrapped.to_dataset()
                >>> plain_again.read_array().tolist()
                [[1.0, 2.0], [3.0, 4.0]]
                >>> wrapped.to_dataset().access == wrapped.access
                True

                ```
        """
        return Dataset(
            self.raster,
            self.access,
            gdal_env=self.gdal_env or None,
            open_options=self.open_options or None,
        )

    def persist_metadata(self) -> None:
        """Write `routing` and `encoding` to the underlying raster tags.

        Stored under `DR_CLASS` / `DR_ROUTING` / `DR_ENCODING` GeoTIFF
        metadata keys so `FlowDirection.open(path)` can recover them.
        """
        self.meta_data = {
            META_CLASS: type(self).__name__,
            META_ROUTING: self.routing,
            META_ENCODING: self.encoding,
        }

    @classmethod
    def open(
        cls,
        path: str,
        *,
        routing: str | None = None,
        encoding: str | None = None,
        read_only: bool = True,
        gdal_env: dict[str, str] | None = None,
        open_options: tuple[str, ...] | list[str] | None = None,
    ) -> FlowDirection:
        """Open a `FlowDirection` GeoTIFF.

        Resolution order for the routing/encoding tags:

        1. Explicit `routing=` / `encoding=` kwargs win unconditionally
           (caller knows what the file is).
        2. Otherwise, `DR_ROUTING` / `DR_ENCODING` metadata tags are used
           if present.
        3. Otherwise, raise `ValueError`. There is no silent fallback to
           `"d8"` — a D∞ raster on disk is float32 in `[0, 2π]` and
           reinterpreting it as int D8 codes silently corrupts every
           downstream computation.

        Args:
            path: Path to the GeoTIFF.
            routing: Explicit routing override. If `None`, falls back to
                the `DR_ROUTING` tag.
            encoding: Explicit encoding override. If `None`, falls back to
                the `DR_ENCODING` tag, then to `"digitalrivers"`.
            read_only: Open the file read-only (default). Pass `False` to
                get a writable handle — required before `persist_metadata()`
                can stamp the `DR_*` tags onto an existing file.
            gdal_env: GDAL config (cloud credentials, HTTP knobs) installed
                for the open and captured on the result, so pyramids' reopen
                paths re-authenticate. Default `None`.
            open_options: GDAL open options forwarded to the driver and
                captured on the result. Default `None`.

        Returns:
            A `FlowDirection` wrapping the opened raster.

        Raises:
            ValueError: If neither `routing=` nor a `DR_ROUTING` tag is
                available.
        """
        ds = Dataset.read_file(
            path, read_only, gdal_env=gdal_env, open_options=open_options
        )
        md = ds.meta_data or {}
        resolved_routing = routing or md.get(META_ROUTING)
        resolved_encoding = encoding or md.get(META_ENCODING) or "digitalrivers"
        if resolved_routing is None:
            raise ValueError(
                f"{path!r} carries no DR_ROUTING tag and no routing= was passed. "
                f"Pass routing= explicitly (one of {sorted(VALID_ROUTING)}) to "
                f"avoid silent misinterpretation of cell values."
            )
        return cls(
            ds.raster,
            ds.access,
            routing=resolved_routing,
            encoding=resolved_encoding,
            gdal_env=ds.gdal_env or None,
            open_options=ds.open_options or None,
        )

    def accumulate(
        self,
        weights: Dataset | None = None,
        *,
        engine: str = "auto",
        out_path: str | None = None,
        tile_size: int | tuple[int, int] = 2048,
        cache: str = "evict",
        workers: int = 1,
        scratch_dir: str | None = None,
        scheduler: str = "threads",
        client=None,
    ) -> Accumulation:
        """Run flow accumulation over this raster's routing scheme.

        Implements a Kahn topological-sort sweep that handles all five routing
        schemes (D8, Rho8, D∞, MFD-Quinn, MFD-Holmgren) via a single algorithm,
        dispatched by `self.routing`.

        Output semantics: `out[cell] = sum of weights over strictly-upstream
        cells` — the cell's own weight does not contribute to its own count.
        This matches the legacy `DEM.flow_accumulation` convention.

        Args:
            weights: Per-cell weight raster (rainfall, runoff coefficient,
                whatever). Must align with this FlowDirection's shape. `None`
                means unit weights (cell-count accumulation).

        Returns:
            Accumulation carrying this object's `routing` for provenance.

        Out-of-core:
            `engine="tiled"` streams a Barnes-2017 tiled accumulation to `out_path` (D8/Rho8 only). `engine="auto"`
            (default) only switches to it when the raster is large enough to risk exhausting RAM.
        """
        from digitalrivers._outofcore.engine import (  # lazy
            require_out_path,
            resolve_engine,
        )

        resolved = resolve_engine(engine, self.rows, self.columns, k=6)
        if resolved == "tiled":
            require_out_path(engine, out_path)
            # lazy import: pulls the numba kernels only when the tiled path is actually used
            from digitalrivers._outofcore.accumulate import flow_accumulation_tiled

            tile_rows, tile_cols = (
                (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
            )
            out = flow_accumulation_tiled(
                self,
                out_path,
                weights=weights,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                cache=cache,
                workers=workers,
                scratch_dir=scratch_dir,
                scheduler=scheduler,
                client=client,
            )
            from digitalrivers.flow.accumulation import Accumulation

            return Accumulation.from_dataset(out, routing=self.routing)

        from digitalrivers.flow.accumulation import Accumulation

        fd_arr = self.read_array()
        valid_mask = self._valid_mask_from_array(fd_arr)
        if weights is not None:
            w_arr = weights.read_array()
            if w_arr.shape != valid_mask.shape:
                raise ValueError(
                    f"weights shape {w_arr.shape} does not match flow_direction "
                    f"shape {valid_mask.shape}"
                )
        else:
            w_arr = None
        acc = _accumulate_array(fd_arr, self.routing, valid_mask, weights=w_arr)
        acc_f32 = acc.astype(np.float32, copy=False)
        plain = Dataset.from_array(
            acc_f32,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=self.default_no_data_value,
        )
        return Accumulation.from_dataset(plain, routing=self.routing)

    def _valid_mask_from_array(self, arr) -> np.ndarray:
        """Compute the (rows, cols) bool mask of valid-data cells from the raster.

        For accumulation purposes `valid` means "this cell can hold and receive a
        contribution". For D8/Rho8 we cannot distinguish a sink (cell with no
        outgoing direction but still in the data envelope) from a truly-outside
        cell at the flow-direction level — both share the no-data sentinel. We
        treat all in-bounds cells as valid; truly-outside cells naturally end up
        with accumulation 0 because no valid direction points at them, and
        callers that want to mask them in the output do so against the original
        DEM (this is what `DEM.flow_accumulation` does).

        Multi-band MFD/D∞ rasters use band 0 as the routing-specific validity
        indicator (angle `>= 0` for D∞, any non-zero fraction for MFD).
        """
        if arr.ndim == 2:
            # D8 / Rho8: treat every in-bounds cell as a valid receiver. Sinks
            # (direction == no_data) are kept in the graph so they accumulate.
            return np.ones(arr.shape, dtype=bool)
        # Multi-band routings.
        band0 = arr[0]
        if self.routing == "dinf":
            return band0 >= 0
        no_val = self.no_data_value[0] if self.no_data_value else None
        if no_val is None:
            return np.ones(band0.shape, dtype=bool)
        return band0 != no_val

    def upslope_flowpath_length(self) -> Dataset:
        """Return a per-cell raster of the longest upslope flow path.

        For every cell in the raster, the returned value is the longest
        planimetric flow path from any upstream source to that cell —
        following D8 / Rho8 receivers, with cardinal steps contributing
        `cell_size` and diagonal steps contributing `cell_size * sqrt(2)`.
        Source cells (no upstream neighbour pointing at them) hold `0.0`.

        Only single-direction routings (`d8` / `rho8`) are supported.

        Returns:
            `Dataset` of float32 lengths in map units, aligned to this
            `FlowDirection` raster. Uses `-9999.0` as the on-disk no-data
            sentinel.

        Raises:
            ValueError: If `self.routing` is not single-direction.
        """
        if self.routing not in ("d8", "rho8"):
            raise ValueError(
                f"upslope_flowpath_length supports single-direction routing "
                f"only; got {self.routing!r}"
            )
        fdir = self.read_array().astype(np.int32, copy=False)
        lengths = kahn_max_upslope_length(fdir, float(abs(self.geotransform[1])))
        return Dataset.from_array(
            lengths.astype(np.float32),
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=-9999.0,
        )

    def isobasins(
        self,
        streams,
        accumulation,
        target_area_km2: float,
    ) -> WatershedRaster:
        """Partition the catchment into sub-basins of approximately equal area.

        Walks the stream network from heads to outlet via the accumulation
        raster. At every stream cell whose floor-divided accumulation quantile
        (`accumulation // target_cells`) is strictly greater than the maximum
        quantile of its stream-upstream neighbours, a virtual sub-basin
        outlet is placed. The final sub-basin labels are produced by
        first-claim-wins reverse-BFS watershed delineation on those seeds.

        Args:
            streams: `StreamRaster` aligned to this flow-direction raster.
            accumulation: `Accumulation` raster aligned to this flow-direction
                raster (units: cells).
            target_area_km2: Target sub-basin area in km². Converted to
                target cell count via the dataset's cell size; must be
                positive.

        Returns:
            `WatershedRaster` whose cells carry a `1..N` sub-basin label
            (or 0 for cells outside any sub-basin) and whose `outlets`
            mapping carries `{basin_id: (row, col)}`.

        Raises:
            ValueError: If routing is multi-direction, shapes mismatch, or
                `target_area_km2` is not positive.
        """
        from digitalrivers.watershed.raster import WatershedRaster

        if self.routing not in ("d8", "rho8"):
            raise ValueError(
                f"isobasins supports single-direction routing only; got "
                f"{self.routing!r}"
            )
        if target_area_km2 <= 0:
            raise ValueError(
                f"target_area_km2 must be positive; got {target_area_km2!r}"
            )

        sm = streams.read_array().astype(bool, copy=False)
        acc_arr = accumulation.read_array().astype(np.int64, copy=False)
        fdir = self.read_array().astype(np.int32, copy=False)
        if sm.shape != fdir.shape or acc_arr.shape != fdir.shape:
            raise ValueError(
                f"shape mismatch: fdir={fdir.shape}, streams={sm.shape}, "
                f"accumulation={acc_arr.shape}"
            )

        gt = self.geotransform
        cell_area_km2 = abs(gt[1] * gt[5]) / 1.0e6
        target_cells = max(1, int(round(target_area_km2 / cell_area_km2)))

        # Quantile bucket of each stream cell, -1 elsewhere.
        quantile = np.where(sm, acc_arr // target_cells, -1).astype(np.int64)

        # For each stream cell, find the max quantile across its stream-upstream
        # neighbours. A cell with strictly larger quantile than that max marks a
        # bucket-boundary — place a seed there.
        rows, cols = sm.shape
        up_max = np.full((rows, cols), -1, dtype=np.int64)
        for k in range(8):
            dr = int(_DIR_DR[k])
            dc = int(_DIR_DC[k])
            src_r = slice(max(0, dr), min(rows, rows + dr))
            src_c = slice(max(0, dc), min(cols, cols + dc))
            dst_r = slice(max(0, -dr), min(rows, rows - dr))
            dst_c = slice(max(0, -dc), min(cols, cols - dc))
            sm_src = sm[src_r, src_c]
            fd_src = fdir[src_r, src_c]
            inflow = sm_src & (fd_src == int(_INV_DIR[k])) & sm[dst_r, dst_c]
            cand = np.where(inflow, quantile[src_r, src_c], -1).astype(np.int64)
            block = up_max[dst_r, dst_c]
            np.maximum(block, cand, out=block)
            up_max[dst_r, dst_c] = block

        seed_mask = sm & (quantile > up_max)
        seed_rcs: list[tuple[int, int]] = [
            (int(r), int(c)) for r, c in zip(*np.nonzero(seed_mask))
        ]
        if not seed_rcs:
            # Catchment smaller than target → fall back to a single basin at
            # the outlet (consistent with WBT's behaviour at extreme inputs).
            outlets = _stream_outlets(sm, fdir)
            if not outlets:
                # No stream cells at all — emit an all-zero basin raster.
                labels = np.zeros((rows, cols), dtype=np.int32)
                plain = Dataset.from_array(
                    labels,
                    geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
                    no_data_value=0,
                )
                return WatershedRaster.from_dataset(
                    plain,
                    routing=self.routing,
                    outlets={},
                )
            seed_rcs = [outlets[0]]

        basin_ids = list(range(1, len(seed_rcs) + 1))
        labels = watershed_d8(
            fdir,
            seed_rcs,
            basin_ids,
            require_unique_basins=True,
        )
        plain = Dataset.from_array(
            labels.astype(np.int32),
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=0,
        )
        outlets_dict = dict(zip(basin_ids, seed_rcs))
        return WatershedRaster.from_dataset(
            plain,
            routing=self.routing,
            outlets=outlets_dict,
        )

    def basins(
        self,
        *,
        min_area_cells: int | None = None,
        min_area_km2: float | None = None,
        merge_small: str = "drop",
    ) -> WatershedRaster:
        """Partition the entire DEM into basins, one label per terminal outlet.

        Detects every cell whose flow direction is the no-data sentinel
        (cells with no defined downstream — either at the data envelope or
        at internal sinks that survived the fill phase) and seeds a reverse
        BFS from each. The result labels every valid cell with the ID of
        the outlet it drains to.

        Args:
            min_area_cells: Optional minimum basin area in cells; basins
                smaller than this are post-processed via `merge_small`.
            min_area_km2: Same threshold expressed in map km². Mutually
                exclusive with `min_area_cells`.
            merge_small: `"drop"` (default) sets undersized basins to 0;
                `"merge_to_neighbour"` dilates the small basin's mask by
                one cell, collects the labels of every basin it touches,
                and relabels the small basin with the largest of those
                8-neighbour labels. Returns `0` for basins whose entire
                8-neighbourhood is either background or other small
                basins (no qualifying survivor).

        Returns:
            :class:`WatershedRaster` tagged with this FlowDirection's
            routing. The `outlets` GeoDataFrame has one row per surviving
            basin with the outlet `row`/`col`/`x`/`y` and
            `cell_count`.

        Raises:
            ValueError: If both area kwargs are supplied or
                `merge_small` is unknown.
        """
        from digitalrivers.watershed.raster import WatershedRaster

        if self.routing not in ("d8", "rho8"):
            raise ValueError(
                f"basins currently supports single-direction routing only; "
                f"got {self.routing!r}"
            )
        if min_area_cells is not None and min_area_km2 is not None:
            raise ValueError("Pass at most one of min_area_cells / min_area_km2")
        if merge_small not in ("drop", "merge_to_neighbour"):
            raise ValueError(
                f"merge_small must be 'drop' or 'merge_to_neighbour'; "
                f"got {merge_small!r}"
            )

        fdir = self.read_array().astype(np.int32, copy=False)
        rows, cols = fdir.shape
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt

        no_val = self.no_data_value[0] if self.no_data_value else None
        # Outlet = cell whose direction code is not in [0, 7] (sink) but the
        # cell itself is in the data envelope.
        if no_val is None:
            no_val = -9999
        is_outlet = (fdir < 0) | (fdir > 7)

        if min_area_km2 is not None:
            cell_area_m2 = abs(dx * dy)
            min_area_cells = int(round(min_area_km2 * 1.0e6 / cell_area_m2))

        seeds: list[tuple[int, int]] = []
        basin_ids: list[int] = []
        outlet_records: list[dict] = []
        bid = 1
        for r, c in zip(*np.nonzero(is_outlet)):
            r = int(r)
            c = int(c)
            seeds.append((r, c))
            basin_ids.append(bid)
            outlet_records.append(
                {
                    "basin_id": bid,
                    "row": r,
                    "col": c,
                    "x": x0 + (c + 0.5) * dx,
                    "y": y0 + (r + 0.5) * dy,
                }
            )
            bid += 1

        basins = watershed_d8(fdir, seeds, basin_ids, require_unique_basins=True)

        # Area filter.
        if min_area_cells is not None and min_area_cells > 1:
            unique, counts = np.unique(basins, return_counts=True)
            sizes = dict(zip(unique.tolist(), counts.tolist()))
            small_ids = {b for b, n in sizes.items() if b != 0 and n < min_area_cells}
            if merge_small == "drop":
                for b in small_ids:
                    basins[basins == b] = 0
            else:  # merge_to_neighbour
                # 8-connected adjacency: shift the small-basin mask in each of
                # the 8 directions and collect any non-self, non-small basin
                # labels that touch its boundary. Pick the largest of those.
                rows, cols = basins.shape
                for b in small_ids:
                    mask = basins == b
                    if not mask.any():
                        continue
                    # Build the 1-cell-dilated border of the small basin.
                    border_labels: set[int] = set()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            r0, r1 = max(0, dr), rows + min(0, dr)
                            c0, c1 = max(0, dc), cols + min(0, dc)
                            src_r0, src_r1 = max(0, -dr), rows + min(0, -dr)
                            src_c0, src_c1 = max(0, -dc), cols + min(0, -dc)
                            mask_dst = mask[r0:r1, c0:c1]
                            labels_src = basins[src_r0:src_r1, src_c0:src_c1]
                            touched = labels_src[mask_dst]
                            border_labels.update(int(v) for v in np.unique(touched))
                    # Drop self, background, and other small basins.
                    candidates = [
                        lbl
                        for lbl in border_labels
                        if lbl != 0 and lbl != b and lbl not in small_ids
                    ]
                    if not candidates:
                        neighbour_id = 0
                    else:
                        neighbour_id = max(candidates, key=lambda lbl: sizes[lbl])
                    basins[mask] = neighbour_id
            # Trim outlet records.
            outlet_records = [
                rec for rec in outlet_records if rec["basin_id"] not in small_ids
            ]
            for rec in outlet_records:
                rec["cell_count"] = int(sizes.get(rec["basin_id"], 0))
        else:
            unique, counts = np.unique(basins, return_counts=True)
            sizes = dict(zip(unique.tolist(), counts.tolist()))
            for rec in outlet_records:
                rec["cell_count"] = int(sizes.get(rec["basin_id"], 0))

        plain = Dataset.from_array(
            basins,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=0,
        )

        outlets_gdf = gpd.GeoDataFrame(
            outlet_records,
            geometry=[Point(rec["x"], rec["y"]) for rec in outlet_records],
            crs=self.epsg,
        )
        return WatershedRaster.from_dataset(
            plain,
            routing=self.routing,
            outlets=outlets_gdf,
        )

    def watershed(
        self,
        pour_points,
        require_unique_basins: bool = False,
    ) -> WatershedRaster:
        """Delineate the upstream watershed of each pour point.

        Reverse-BFS from every pour-point cell, labelling every contributing
        cell with the pour point's 1-based basin ID. Multi-point inputs
        produce a labelled raster (one ID per pour point).

        Args:
            pour_points: `GeoDataFrame` of Point geometries — one row per
                desired basin. Points outside the raster envelope are skipped
                with a NaN entry in the returned `outlets` GeoDataFrame.
            require_unique_basins: If False (default), inner pour points
                overwrite the outer basin's cells along shared upstream
                paths — the outer basin contains a hole around the inner
                basin. If True, the first seed to claim a cell keeps it; the
                outer basin contains no inner-basin cells.

        Returns:
            :class:`WatershedRaster` tagged with this FlowDirection's routing.
            The `outlets` attribute is a GeoDataFrame parallel to the input
            `pour_points`.
        """
        from digitalrivers.watershed.raster import WatershedRaster

        if self.routing not in ("d8", "rho8"):
            raise ValueError(
                f"watershed currently supports single-direction routing only; "
                f"got {self.routing!r}"
            )

        target_epsg = self.epsg
        if (
            getattr(pour_points, "crs", None) is not None
            and target_epsg is not None
            and pour_points.crs.to_epsg() != target_epsg
        ):
            pour_points = pour_points.to_crs(target_epsg)

        fdir = self.read_array().astype(np.int32, copy=False)
        rows, cols = fdir.shape
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt

        seeds: list[tuple[int, int]] = []
        basin_ids: list[int] = []
        outlet_records: list[dict] = []
        for i, pt in enumerate(pour_points.geometry):
            px, py = float(pt.x), float(pt.y)
            col = int((px - x0) / dx)
            row = int((py - y0) / dy)
            bid = i + 1
            if 0 <= row < rows and 0 <= col < cols:
                seeds.append((row, col))
                basin_ids.append(bid)
                outlet_records.append(
                    {
                        "basin_id": bid,
                        "row": row,
                        "col": col,
                        "x": x0 + (col + 0.5) * dx,
                        "y": y0 + (row + 0.5) * dy,
                    }
                )
            else:
                outlet_records.append(
                    {
                        "basin_id": bid,
                        "row": -1,
                        "col": -1,
                        "x": float("nan"),
                        "y": float("nan"),
                    }
                )

        basins = watershed_d8(
            fdir, seeds, basin_ids, require_unique_basins=require_unique_basins
        )
        plain = Dataset.from_array(
            basins,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=0,
        )

        outlets_gdf = gpd.GeoDataFrame(
            outlet_records,
            geometry=[
                Point(rec["x"], rec["y"]) if not (rec["row"] < 0) else None
                for rec in outlet_records
            ],
            crs=target_epsg,
        )
        return WatershedRaster.from_dataset(
            plain,
            routing=self.routing,
            outlets=outlets_gdf,
        )

    def __repr__(self) -> str:
        """Return a one-line summary naming the raster's shape, routing scheme and cell-value encoding.

        Both tags are shown, because a direction grid is only interpretable
              given the pair — the same cell value means different neighbours
              under different encodings.

        Returns:
            A string of the form `<FlowDirection rows=R cols=C routing='...' encoding='...'>`.
        """
        return (
            f"<FlowDirection rows={self.rows} cols={self.columns} "
            f"routing={self.routing!r} encoding={self.encoding!r}>"
        )
