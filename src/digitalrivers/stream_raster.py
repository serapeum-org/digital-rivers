"""Typed stream-network raster.

``StreamRaster.__init__`` enforces the *ismulti guard* from TopoToolbox
MATLAB ``@STREAMobj/STREAMobj.m:36`` — stream extraction from a
multi-direction flow scheme is not well-defined, so the constructor rejects
any ``routing`` outside the supported single-direction set up front.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers._metadata import (
    META_CLASS,
    META_ROUTING,
    META_THRESHOLD,
    VALID_ROUTING,
)

if TYPE_CHECKING:
    from digitalrivers.watershed_raster import WatershedRaster


class StreamRaster(Dataset):
    """Boolean/int stream-network raster tagged with extraction threshold.

    Args:
        src: GDAL dataset wrapping the stream raster.
        access: ``"read_only"`` (default) or ``"write"``.
        threshold: Accumulation threshold used to extract this stream
            network. Stored for provenance and round-trip persistence.
            Required keyword-only.
        routing: Routing scheme of the ``FlowDirection`` that produced the
            upstream accumulation. Required keyword-only. Must be in
            ``_SUPPORTED_ROUTING``.

    Raises:
        ValueError: If ``routing`` is not a recognised value at all.
        TypeError: If ``routing`` is a multi-direction scheme. Convert the
            ``FlowDirection`` to D8 first.
    """

    threshold: float | int
    routing: str

    _SUPPORTED_ROUTING: frozenset[str] = frozenset({"d8", "rho8"})

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        *,
        threshold: float | int,
        routing: str,
    ):
        if routing not in VALID_ROUTING:
            raise ValueError(
                f"routing must be one of {sorted(VALID_ROUTING)}; got {routing!r}"
            )
        if routing not in self._SUPPORTED_ROUTING:
            raise TypeError(
                f"StreamRaster currently supports only single-direction routing "
                f"({sorted(self._SUPPORTED_ROUTING)}); got {routing!r}. "
                f"Convert the FlowDirection to D8 first."
            )
        super().__init__(src, access)
        self.threshold = threshold
        self.routing = routing

    @classmethod
    def from_dataset(
        cls,
        ds: Dataset,
        *,
        threshold: float | int,
        routing: str,
    ) -> StreamRaster:
        """Promote a plain ``Dataset`` into a ``StreamRaster``."""
        return cls(ds.raster, threshold=threshold, routing=routing)

    def to_dataset(self) -> Dataset:
        """Drop the typed wrapper and return the underlying ``Dataset``."""
        return Dataset(self.raster)

    def persist_metadata(self) -> None:
        """Write ``routing`` and ``threshold`` to the raster's metadata tags."""
        self.meta_data = {
            META_CLASS: type(self).__name__,
            META_ROUTING: self.routing,
            META_THRESHOLD: str(self.threshold),
        }

    @classmethod
    def open(
        cls,
        path: str,
        *,
        threshold: float | int | None = None,
        routing: str | None = None,
    ) -> StreamRaster:
        """Open a ``StreamRaster`` GeoTIFF.

        Resolution order: explicit kwargs > ``DR_*`` metadata tags > raise.
        ``threshold`` is parsed from the tag as a float (it was written via
        ``str(self.threshold)``).

        Raises:
            ValueError: If either ``routing`` or ``threshold`` cannot be
                resolved from kwargs or metadata tags.
        """
        ds = Dataset.read_file(path)
        md = ds.meta_data or {}
        resolved_routing = routing or md.get(META_ROUTING)
        if resolved_routing is None:
            raise ValueError(
                f"{path!r} carries no DR_ROUTING tag and no routing= was passed. "
                f"Pass routing= explicitly (one of {sorted(VALID_ROUTING)})."
            )
        if threshold is None:
            tag = md.get(META_THRESHOLD)
            if tag is None:
                raise ValueError(
                    f"{path!r} carries no DR_THRESHOLD tag and no threshold= was "
                    f"passed."
                )
            threshold = float(tag)
        return cls(ds.raster, threshold=threshold, routing=resolved_routing)

    def subbasins(
        self,
        flow_direction,
        method: str = "link",
    ) -> WatershedRaster:
        """Partition the basin into one sub-basin per stream link.

        Each cell is labelled with the ID of the first downstream stream
        link it joins. Confluence cells belong to the new downstream link
        (WhiteboxTools / TauDEM convention). Off-stream cells inherit the
        link ID of the first stream cell their flow path reaches.

        Args:
            flow_direction: Single-direction ``FlowDirection`` aligned to this
                stream raster.
            method: ``"link"`` (default) — one sub-basin per link. The
                ``"min_order"`` and ``"isobasin"`` modes from the spec are
                deferred.

        Returns:
            :class:`WatershedRaster` tagged with this stream raster's
            ``routing`` (via the FlowDirection). Background cells (those that
            never reach a stream) are 0.

        Raises:
            ValueError: If ``method`` is not ``"link"`` or
                ``flow_direction`` is multi-direction.
        """
        import numpy as np

        from digitalrivers.flow_direction import FlowDirection
        from digitalrivers.watershed_raster import WatershedRaster

        if method != "link":
            raise ValueError(
                f"method must be 'link' (other modes deferred); got {method!r}"
            )
        if not isinstance(flow_direction, FlowDirection):
            raise ValueError("flow_direction must be a FlowDirection instance")
        if flow_direction.routing not in ("d8", "rho8"):
            raise ValueError(
                f"subbasins currently supports single-direction routing only; "
                f"got {flow_direction.routing!r}"
            )

        stream_mask = self.read_array().astype(bool, copy=False)
        fdir = flow_direction.read_array().astype(np.int32, copy=False)
        if stream_mask.shape != fdir.shape:
            raise ValueError(
                f"flow_direction shape {fdir.shape} != stream raster shape "
                f"{stream_mask.shape}"
            )

        d_row = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
        d_col = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
        inv_dir = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int32)
        rows, cols = stream_mask.shape

        # Incoming-stream count per stream cell (for confluence detection).
        nup = np.zeros(stream_mask.shape, dtype=np.int8)
        for k in range(8):
            dr = int(d_row[k])
            dc = int(d_col[k])
            src_r = slice(max(0, dr), min(rows, rows + dr))
            src_c = slice(max(0, dc), min(cols, cols + dc))
            dst_r = slice(max(0, -dr), min(rows, rows - dr))
            dst_c = slice(max(0, -dc), min(cols, cols - dc))
            sm_src = stream_mask[src_r, src_c]
            fd_src = fdir[src_r, src_c]
            inflow = sm_src & (fd_src == inv_dir[k]) & stream_mask[dst_r, dst_c]
            nup[dst_r, dst_c] += inflow.astype(np.int8)

        link_id = np.zeros((rows, cols), dtype=np.int32)
        next_id = 1
        # Heads and downstream-of-confluence cells start new links.
        # Build link IDs by walking from every head/confluence-downstream entry.
        starts = stream_mask & ((nup == 0) | (nup >= 2))
        for r0, c0 in np.argwhere(starts):
            r0 = int(r0)
            c0 = int(c0)
            if link_id[r0, c0] != 0:
                continue
            current = next_id
            next_id += 1
            link_id[r0, c0] = current
            r, c = r0, c0
            while True:
                d = int(fdir[r, c])
                if d < 0 or d > 7:
                    break
                nr = r + int(d_row[d])
                nc = c + int(d_col[d])
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if not stream_mask[nr, nc]:
                    break
                if nup[nr, nc] >= 2:
                    break  # confluence — handled by its own iteration
                if link_id[nr, nc] != 0:
                    break  # already assigned by an earlier walk
                link_id[nr, nc] = current
                r, c = nr, nc

        # Off-stream cells: walk downstream until hitting a labelled cell.
        out = link_id.copy()
        for r0 in range(rows):
            for c0 in range(cols):
                if out[r0, c0] != 0:
                    continue
                path: list[tuple[int, int]] = []
                r, c = r0, c0
                tail_id = 0
                while True:
                    if out[r, c] != 0:
                        tail_id = int(out[r, c])
                        break
                    path.append((r, c))
                    d = int(fdir[r, c])
                    if d < 0 or d > 7:
                        break
                    nr = r + int(d_row[d])
                    nc = c + int(d_col[d])
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        break
                    r, c = nr, nc
                if tail_id != 0:
                    for pr, pc in path:
                        out[pr, pc] = tail_id

        plain = Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=0,
        )

        import geopandas as gpd
        unique_ids = sorted({int(v) for v in np.unique(out) if v != 0})
        # For each link, the outlet is the cell whose D8 successor either
        # belongs to a different basin or falls off the grid — i.e., the
        # most-downstream cell of the link. Pick the first such cell we
        # find; ties don't matter because all such cells map to the same
        # outlet point cluster on a well-formed flow graph.
        x0, dx, _, y0, _, dy = self.geotransform
        outlet_xs: list[float] = []
        outlet_ys: list[float] = []
        for bid in unique_ids:
            rs, cs = np.where(out == bid)
            chosen_r, chosen_c = int(rs[0]), int(cs[0])
            for r, c in zip(rs.tolist(), cs.tolist()):
                d = int(fdir[r, c])
                if d < 0 or d > 7:
                    chosen_r, chosen_c = r, c
                    break
                nr = r + int(d_row[d])
                nc = c + int(d_col[d])
                if not (0 <= nr < rows and 0 <= nc < cols):
                    chosen_r, chosen_c = r, c
                    break
                if int(out[nr, nc]) != bid:
                    chosen_r, chosen_c = r, c
                    break
            outlet_xs.append(float(x0 + (chosen_c + 0.5) * dx))
            outlet_ys.append(float(y0 + (chosen_r + 0.5) * dy))
        outlets_gdf = gpd.GeoDataFrame(
            {"basin_id": unique_ids,
             "cell_count": [int((out == bid).sum()) for bid in unique_ids]},
            geometry=gpd.points_from_xy(outlet_xs, outlet_ys),
            crs=self.epsg,
        )
        return WatershedRaster.from_dataset(
            plain, routing=flow_direction.routing, outlets=outlets_gdf,
        )

    def order(
        self,
        method: str = "strahler",
        flow_direction=None,
    ) -> "StreamRaster":
        """Compute Strahler / Shreve / Horton stream order on this raster.

        Args:
            method: ``"strahler"`` (default), ``"shreve"``, or ``"horton"``.
            flow_direction: Single-direction (``d8`` / ``rho8``) FlowDirection
                aligned to this stream raster. Required — the topology walks
                the flow-direction edges.

        Returns:
            A new ``StreamRaster`` whose underlying raster is uint16 (uint32
            for ``shreve``) and holds the stream order; non-stream cells hold
            ``0``. The returned object preserves this raster's ``threshold``
            and ``routing`` tags for downstream consumers.

        Raises:
            ValueError: If ``method`` is unknown or ``flow_direction`` is
                missing / multi-direction.
        """
        import numpy as np

        from digitalrivers._stream_order import horton, shreve, strahler
        from digitalrivers.flow_direction import FlowDirection

        if method not in ("strahler", "shreve", "horton"):
            raise ValueError(
                f"method must be one of 'strahler', 'shreve', 'horton'; "
                f"got {method!r}"
            )
        if not isinstance(flow_direction, FlowDirection):
            raise ValueError(
                "flow_direction is required and must be a FlowDirection"
            )
        if flow_direction.routing not in ("d8", "rho8"):
            raise ValueError(
                f"order currently supports single-direction routing only; got "
                f"{flow_direction.routing!r}"
            )
        stream_mask = self.read_array().astype(bool, copy=False)
        fdir = flow_direction.read_array().astype(np.int32, copy=False)
        if stream_mask.shape != fdir.shape:
            raise ValueError(
                f"flow_direction shape {fdir.shape} != stream raster shape "
                f"{stream_mask.shape}"
            )
        if method == "strahler":
            arr = strahler(stream_mask, fdir)
        elif method == "shreve":
            arr = shreve(stream_mask, fdir)
        else:
            arr = horton(stream_mask, fdir)
        plain = Dataset.create_from_array(
            arr, geo=self.geotransform, epsg=self.epsg, no_data_value=0,
        )
        return StreamRaster.from_dataset(
            plain, threshold=self.threshold, routing=self.routing
        )

    def to_vector(
        self,
        flow_direction,
        dem=None,
        single_direction: str = "max",
    ):
        """Vectorise the stream raster into a ``GeoDataFrame`` of LineString links.

        Walks the flow-direction raster from every head and every cell
        downstream of a confluence until it reaches a confluence or an outlet.
        Each resulting LineString is one stream link.

        Args:
            flow_direction: ``FlowDirection`` raster aligned to this stream
                raster. Must be a single-direction routing (``d8`` or ``rho8``)
                — multi-direction inputs raise. (D∞ / MFD inputs would need a
                separate dominant-direction collapse, deferred.)
            dem: Optional ``DEM`` aligned to the stream raster. When supplied,
                the link attributes include ``drop_m`` and ``mean_slope``.
            single_direction: Reserved for future multi-direction collapse
                (``"max"`` for argmax-of-fractions; ``"weighted"`` for
                weighted-mean direction). Ignored when ``flow_direction`` is
                already single-direction.

        Returns:
            ``geopandas.GeoDataFrame`` with columns:
              - ``link_id`` (int64): 0-based sequential link identifier.
              - ``from_node`` (int64): node ID at the upstream end (head /
                confluence).
              - ``to_node`` (int64): node ID at the downstream end.
              - ``length_m`` (float64): sum of per-step distances using the
                D8 grid-lengths lookup (``cell_size`` cardinal,
                ``cell_size * sqrt(2)`` diagonal).
              - ``drop_m`` (float64): ``z[from] - z[to]`` (positive if the
                link descends; clamped to 0 otherwise). NaN if ``dem`` is
                None.
              - ``mean_slope`` (float64): ``drop_m / length_m`` (m/m). NaN if
                ``dem`` is None or ``length_m == 0``.
              - ``geometry``: shapely ``LineString`` in the dataset's CRS,
                vertices at cell centres.

        Raises:
            ValueError: If ``flow_direction`` is multi-direction.
            ValueError: If shapes do not match.
        """
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import LineString

        from digitalrivers.flow_direction import FlowDirection  # for type-narrow

        if not isinstance(flow_direction, FlowDirection):
            raise TypeError(
                f"flow_direction must be a FlowDirection; got {type(flow_direction).__name__}"
            )
        if flow_direction.routing not in ("d8", "rho8"):
            raise ValueError(
                f"to_vector currently supports single-direction routing only; "
                f"got {flow_direction.routing!r}. Collapse the flow direction to "
                f"D8 first."
            )

        fdir = flow_direction.read_array().astype(np.int32, copy=False)
        stream_mask = self.read_array().astype(bool, copy=False)
        if fdir.shape != stream_mask.shape:
            raise ValueError(
                f"flow_direction shape {fdir.shape} != stream raster shape "
                f"{stream_mask.shape}"
            )

        if dem is not None:
            z = dem.read_array().astype(np.float64, copy=False)
            no_val = dem.no_data_value[0] if dem.no_data_value else None
            if no_val is not None:
                z = np.where(z == no_val, np.nan, z)
            if z.shape != stream_mask.shape:
                raise ValueError(
                    f"dem shape {z.shape} != stream raster shape {stream_mask.shape}"
                )
        else:
            z = None

        # 8-direction offsets matching DIR_OFFSETS (0=S, 1=SW, ..., 7=SE).
        d_row = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
        d_col = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
        # Inverse direction: if cell at offset (dr, dc) has direction = inv[k],
        # it is flowing INTO us.
        inv_dir = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int32)
        grid_lengths = np.array(
            [1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0),
             1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)],
            dtype=np.float64,
        ) * float(self.cell_size)

        rows, cols = stream_mask.shape

        # Step 1 — incoming-stream count per stream cell.
        nup = np.zeros(stream_mask.shape, dtype=np.int8)
        for k in range(8):
            dr = int(d_row[k])
            dc = int(d_col[k])
            src_r = slice(max(0, dr), min(rows, rows + dr))
            src_c = slice(max(0, dc), min(cols, cols + dc))
            dst_r = slice(max(0, -dr), min(rows, rows - dr))
            dst_c = slice(max(0, -dc), min(cols, cols - dc))
            sm_src = stream_mask[src_r, src_c]
            fd_src = fdir[src_r, src_c]
            # A neighbour at (src) points into (dst) iff its direction equals inv[k].
            inflow = sm_src & (fd_src == inv_dir[k]) & stream_mask[dst_r, dst_c]
            nup[dst_r, dst_c] += inflow.astype(np.int8)

        # Step 2 — find link starts.
        heads_or_confluences_mask = stream_mask & ((nup == 0) | (nup >= 2))

        # Step 3 — walk each link.
        def _trace(start_r: int, start_c: int):
            path = [(start_r, start_c)]
            r, c = start_r, start_c
            length = 0.0
            while True:
                d = int(fdir[r, c])
                if d < 0 or d > 7:
                    break  # sink / outlet
                nr = r + int(d_row[d])
                nc = c + int(d_col[d])
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if not stream_mask[nr, nc]:
                    break
                length += float(grid_lengths[d])
                path.append((nr, nc))
                if nup[nr, nc] >= 2:
                    break
                r, c = nr, nc
            return path, length

        # Assign node IDs: every distinct head / confluence / link-end is a node.
        node_id_grid = np.full(stream_mask.shape, -1, dtype=np.int64)
        next_node_id = 0

        def _get_node_id(r: int, c: int) -> int:
            nonlocal next_node_id
            if node_id_grid[r, c] < 0:
                node_id_grid[r, c] = next_node_id
                next_node_id += 1
            return int(node_id_grid[r, c])

        gt = self.geotransform
        records: list[dict] = []
        link_id = 0
        starts = np.argwhere(heads_or_confluences_mask)
        for r0, c0 in starts:
            r0 = int(r0)
            c0 = int(c0)
            # Skip a confluence cell if it has no outgoing direction (it's an outlet
            # confluence — handled when its upstream link arrives, no link begins here).
            d_start = int(fdir[r0, c0])
            if d_start < 0 or d_start > 7:
                continue
            path, length_m = _trace(r0, c0)
            if len(path) < 2:
                continue
            from_node = _get_node_id(r0, c0)
            r_end, c_end = path[-1]
            to_node = _get_node_id(r_end, c_end)

            xs = [gt[0] + (c + 0.5) * gt[1] + (r + 0.5) * gt[2] for r, c in path]
            ys = [gt[3] + (c + 0.5) * gt[4] + (r + 0.5) * gt[5] for r, c in path]
            geom = LineString(zip(xs, ys))

            if z is not None:
                z_from = float(z[r0, c0])
                z_to = float(z[r_end, c_end])
                drop_m = max(0.0, z_from - z_to)
                mean_slope = drop_m / length_m if length_m > 0 else np.nan
            else:
                drop_m = np.nan
                mean_slope = np.nan

            records.append({
                "link_id": link_id,
                "from_node": from_node,
                "to_node": to_node,
                "length_m": length_m,
                "drop_m": drop_m,
                "mean_slope": mean_slope,
                "geometry": geom,
            })
            link_id += 1

        crs = self.epsg
        return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    def __repr__(self) -> str:
        return (
            f"<StreamRaster rows={self.rows} cols={self.columns} "
            f"threshold={self.threshold!r} routing={self.routing!r}>"
        )
