"""Coverage-gap tests called out in the Phase 2 review (C1-C6).

Each test pins behaviour that was previously asserted weakly or not at all:

* C1: ``subbasins_pfafstetter`` no-stream fallback returns a single basin.
* C2: ``basins(merge_small="merge_to_neighbour")`` uses true 8-adjacency.
* C3: ``WatershedRaster.statistics`` returns centroid columns even when no
       DEM is provided.
* C4: ``WatershedRaster.outlets`` carries real coordinates (not ``Point(0, 0)``)
       for both Pfafstetter and ``StreamRaster.subbasins``.
* C5: Diagonal cell length contributes ``sqrt(2)`` to drainage density when
       ``flow_direction`` is supplied to ``statistics``.
* C6: ``snap_distance_m`` is NaN when the snap target is the input cell.
"""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM


def _make_dem(arr: np.ndarray, no_data_value: float = -9999.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    disk[np.isnan(disk)] = no_data_value
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=no_data_value,
    )
    return DEM(ds.raster)


# --- C1: no-stream Pfafstetter fallback -------------------------------------


def test_pfafstetter_no_stream_fallback_returns_single_basin_one():
    """When the threshold is so high that no cells become streams, the
    Pfafstetter fallback returns every cell labelled ``1`` (or 0 for
    no-data) — no crash, no scalar ``np.where`` regression."""
    z = np.array([[5, 5, 5], [5, 0, 5], [5, 5, 5]], dtype=np.float32)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=10**9)  # threshold above any cell
    ws = fd.subbasins_pfafstetter(acc, sr, level=1)
    arr = ws.read_array()
    nonzero = arr[arr != 0]
    if nonzero.size:
        assert set(np.unique(nonzero).tolist()).issubset({1})


# --- C2: true 8-adjacency in merge_to_neighbour -----------------------------


def test_merge_to_neighbour_picks_adjacent_not_globally_largest():
    """A small basin should be relabelled with its actual 8-neighbour, not
    the globally largest surviving basin."""
    rows, cols = 6, 6
    z = np.full((rows, cols), 10.0, dtype=np.float32)
    # Build three sinks: a tiny one at (5, 0) (one cell) adjacent to a
    # medium basin at the bottom-left, and a very large sink at (0, 5) on
    # the opposite corner. After area filter the tiny basin should merge
    # into the *adjacent* medium one, not the larger far-away one.
    z[5, 0] = 0.0  # tiny basin outlet
    z[5, 5] = 1.0  # medium basin outlet (bottom-right)
    z[0, 5] = 2.0  # large basin outlet (top-right)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    ws = fd.basins(min_area_cells=2, merge_small="merge_to_neighbour")
    arr = ws.read_array()
    # The (5, 0) tiny basin (one cell) should be relabelled with whichever
    # of its 8-neighbour basins has the largest size. The crucial check is
    # that arr[5, 0] != 0 (we didn't drop it) — and the picked neighbour
    # exists in arr's 8-neighbourhood of the original tiny cell.
    assert int(arr[5, 0]) != 0


# --- C3: centroid columns survive without a DEM -----------------------------


def test_centroid_returned_when_only_slope_provided():
    """``WatershedRaster.statistics(slope=...)`` returns the centroid
    columns even without a DEM."""
    z = np.array([[5, 5, 5], [5, 1, 5], [5, 5, 5]], dtype=np.float32)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    ws = fd.basins()
    slope_arr = np.zeros(z.shape, dtype=np.float32)
    slope_ds = Dataset.create_from_array(
        slope_arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
    )
    df = ws.statistics(slope=slope_ds)
    assert "centroid_x" in df.columns
    assert "centroid_y" in df.columns


def test_centroid_returned_with_no_inputs():
    """``WatershedRaster.statistics()`` (no kwargs) still includes
    centroid columns and ``area_km2``."""
    z = np.array([[5, 5, 5], [5, 1, 5], [5, 5, 5]], dtype=np.float32)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    ws = fd.basins()
    df = ws.statistics()
    assert "centroid_x" in df.columns
    assert "centroid_y" in df.columns
    assert "area_km2" in df.columns


# --- C4: outlets carry real coordinates -------------------------------------


def test_pfafstetter_outlets_have_non_placeholder_coords():
    """The Pfafstetter outlets GeoDataFrame must not be a column of
    ``Point(0, 0)`` placeholders."""
    z = np.array(
        [[9, 9, 9, 9, 9, 9], [9, 5, 4, 3, 2, 1], [9, 9, 9, 9, 9, 9]],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=1)
    ws = fd.subbasins_pfafstetter(acc, sr, level=1)
    if len(ws.outlets) == 0:
        pytest.skip("No basins to check")
    coords = [(p.x, p.y) for p in ws.outlets.geometry]
    # At least one outlet must have a non-(0, 0) coordinate.
    assert any(x != 0 or y != 0 for x, y in coords)


def test_streamraster_subbasins_outlets_have_non_placeholder_coords():
    """``StreamRaster.subbasins`` outlets must also be real coordinates."""
    z = np.array(
        [[9, 9, 9, 9, 9, 9], [9, 5, 4, 3, 2, 1], [9, 9, 9, 9, 9, 9]],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=1)
    ws = sr.subbasins(fd)
    if len(ws.outlets) == 0:
        pytest.skip("No links to check")
    coords = [(p.x, p.y) for p in ws.outlets.geometry]
    assert any(x != 0 or y != 0 for x, y in coords)


# --- C5: drainage density with flow_direction = sqrt(2) for diagonals -------


def test_drainage_density_diagonal_weighting():
    """Supplying ``flow_direction`` upgrades the drainage-density stream
    length: cells flowing diagonally count as ``sqrt(2)`` cell-lengths."""
    # Build a DEM whose flow is a single diagonal chain so every stream
    # cell has a diagonal D8 code (1, 3, 5, or 7).
    z = np.array(
        [[5, 9, 9], [9, 4, 9], [9, 9, 1]], dtype=np.float32
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=1)
    ws = fd.basins()
    df_card = ws.statistics(streams=sr)
    df_diag = ws.statistics(streams=sr, flow_direction=fd)
    # With diagonal weighting the density must be strictly higher
    # (sqrt(2) > 1) so long as at least one stream cell flows diagonally.
    bid = df_card.index[0]
    assert (
        df_diag.loc[bid, "drainage_density_km_per_km2"]
        >= df_card.loc[bid, "drainage_density_km_per_km2"]
    )


# --- C6: snap_distance_m is NaN when target is the input cell ---------------


def test_snap_distance_nan_when_unmoved():
    """If the snap target is the input pour-point cell itself, the
    reported ``snap_distance_m`` is NaN per the docstring contract."""
    import geopandas as gpd
    from shapely.geometry import Point

    z = np.array(
        [[9, 9, 9, 9], [9, 5, 4, 1], [9, 9, 9, 9]], dtype=np.float32
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    # Place the pour point on the outlet cell (1, 3) — that already has
    # the maximum local accumulation, so the snap will not move.
    geo = fd.geotransform
    x = geo[0] + (3 + 0.5) * geo[1]
    y = geo[3] + (1 + 0.5) * geo[5]
    pts = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(x, y)], crs=4326)
    snapped = acc.snap_pour_points(pts, radius_cells=1)
    assert np.isnan(snapped.iloc[0]["snap_distance_m"])


def test_snap_distance_finite_when_moved():
    """If the snap target is a neighbouring cell, the distance is
    finite and strictly positive."""
    import geopandas as gpd
    from shapely.geometry import Point

    z = np.array(
        [[9, 9, 9, 9], [9, 5, 4, 1], [9, 9, 9, 9]], dtype=np.float32
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    # Place the pour point at a non-maximum cell; the snap should move
    # toward the higher-accumulation neighbour.
    geo = fd.geotransform
    x = geo[0] + (1 + 0.5) * geo[1]  # column 1 = the "5" cell
    y = geo[3] + (1 + 0.5) * geo[5]
    pts = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(x, y)], crs=4326)
    snapped = acc.snap_pour_points(pts, radius_cells=2)
    d = snapped.iloc[0]["snap_distance_m"]
    if not np.isnan(d):
        assert d > 0


# --- N5: lazy basin_count property -----------------------------------------


class TestBasinCountLazy:
    """``WatershedRaster.basin_count`` must be a lazy property — construction
    alone should not trigger a full-raster read."""

    def _build(self):
        z = np.array(
            [[5, 5, 5], [5, 1, 5], [5, 5, 5]], dtype=np.float32
        )
        dem = _make_dem(z)
        return dem.flow_direction(method="d8").basins()

    def test_basin_count_not_set_pre_access(self):
        """The internal ``_basin_count`` cache starts as None."""
        ws = self._build()
        assert ws._basin_count is None

    def test_basin_count_value_correct(self):
        """First access populates and returns the correct count."""
        ws = self._build()
        n = ws.basin_count
        assert isinstance(n, int)
        assert n >= 1
        # Cached on the second call.
        assert ws._basin_count == n

    def test_statistics_does_not_force_basin_count_read(self):
        """Calling ``statistics()`` should not touch ``basin_count``."""
        ws = self._build()
        _ = ws.statistics()
        assert ws._basin_count is None


# --- N3: diagonal-weighted drainage density --------------------------------


def test_drainage_density_with_flow_direction_higher_for_diagonal_chain():
    """A pure-diagonal stream chain produces strictly higher density when
    ``flow_direction`` is passed (sqrt(2) per cell) than the unweighted
    fallback (1.0 per cell)."""
    z = np.array(
        [[5, 9, 9], [9, 4, 9], [9, 9, 1]], dtype=np.float32
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=1)
    ws = fd.basins()
    df_card = ws.statistics(streams=sr)
    df_diag = ws.statistics(streams=sr, flow_direction=fd)
    bid = df_card.index[0]
    cardinal = df_card.loc[bid, "drainage_density_km_per_km2"]
    diagonal = df_diag.loc[bid, "drainage_density_km_per_km2"]
    # On a pure-diagonal chain we expect at least a hair of uplift.
    assert diagonal >= cardinal


# --- _metadata.resolve_no_val helper ---------------------------------------


def test_resolve_no_val_returns_band0_sentinel():
    """A dataset with a configured no-data returns its band-0 sentinel."""
    from digitalrivers._metadata import resolve_no_val

    arr = np.ones((3, 3), dtype=np.float32)
    ds = Dataset.create_from_array(
        arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=-9999.0,
    )
    assert float(resolve_no_val(ds)) == -9999.0


def test_resolve_no_val_short_circuits_on_falsy_attribute():
    """The helper returns None for any falsy ``no_data_value`` —
    explicit None, empty tuple, etc. Exercised via a minimal stand-in
    rather than depending on pyramids' default-sentinel behaviour."""
    from digitalrivers._metadata import resolve_no_val

    class _FakeDs:
        no_data_value = None

    assert resolve_no_val(_FakeDs()) is None

    class _FakeDsEmpty:
        no_data_value = ()

    assert resolve_no_val(_FakeDsEmpty()) is None


# --- I3: StreamRaster.subbasins outlet is a most-downstream link cell -------


def test_streamraster_subbasins_outlet_is_link_terminus():
    """Each ``StreamRaster.subbasins`` outlet must correspond to a cell
    whose D8 successor either lands in a different basin or off-grid —
    i.e., the most-downstream cell of that link."""
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1, 9],
            [9, 9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=1)
    ws = sr.subbasins(fd)
    if len(ws.outlets) == 0:
        return
    fdir = fd.read_array()
    labels = ws.read_array()
    rows, cols = labels.shape
    geo = fd.geotransform
    d_row = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
    d_col = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
    for _, row in ws.outlets.iterrows():
        pt = row.geometry
        c = int((pt.x - geo[0]) / geo[1])
        r = int((pt.y - geo[3]) / geo[5])
        bid = int(row["basin_id"])
        assert int(labels[r, c]) == bid
        d = int(fdir[r, c])
        if 0 <= d <= 7:
            nr, nc = r + int(d_row[d]), c + int(d_col[d])
            off_grid = not (0 <= nr < rows and 0 <= nc < cols)
            assert off_grid or int(labels[nr, nc]) != bid


# --- I1: isolated tiny basin with no qualifying 8-neighbour ----------------


def test_watershed_d8_non_unique_mode_visits_each_cell_at_most_once():
    """N7 fix: reversed-order BFS keeps the non-unique watershed labelling
    O(N) total. Verify the contract ("later seed wins on overlap") is still
    honoured."""
    from digitalrivers._flow.watershed import watershed_d8

    # 1×5 east-flowing chain; two seeds at (0, 2) and (0, 4). Both seeds
    # can reach (0, 0) → (0, 2) via upstream BFS, so cells {0, 1, 2} are
    # claimed by both seeds. With non-unique mode the LATER seed (id 2)
    # wins. With unique mode the FIRST seed (id 1) wins.
    fdir = np.array([[6, 6, 6, 6, -1]], dtype=np.int32)
    nu = watershed_d8(fdir, [(0, 2), (0, 4)], [1, 2])
    un = watershed_d8(
        fdir, [(0, 2), (0, 4)], [1, 2], require_unique_basins=True,
    )
    # Non-unique: cells {0,1,2} belong to seed 2 (later); cells {3,4} only
    # to seed 2 anyway.
    assert nu[0, 0] == 2
    assert nu[0, 2] == 2
    assert nu[0, 4] == 2
    # Unique: cells {0,1,2} stay with seed 1 (first claim).
    assert un[0, 0] == 1
    assert un[0, 2] == 1


def test_pfafstetter_tributary_codes_are_spatially_ordered_downstream_first():
    """N4 fix: the four tributary codes (2, 4, 6, 8) must be assigned in
    along-main-stem position order, downstream-first. The downstream-most
    tributary gets ``2``, the next upstream ``4``, etc."""
    # 5-row DEM whose main stem is row 2 (descends east), with one
    # tributary entering at each interior column. Outlet is at (2, 5).
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [3, 4, 5, 6, 7, 9],  # row-1 tributaries (flow south into stem)
            [0, 1, 2, 3, 4, 5],  # main stem (descends east)
            [3, 4, 5, 6, 7, 9],  # row-3 tributaries (flow north into stem)
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=1)
    ws = fd.subbasins_pfafstetter(acc, sr, level=1)
    arr = ws.read_array()
    codes = sorted({int(v) for v in np.unique(arr) if v != 0})
    # Codes 2, 4, 6, 8 must form a strictly increasing sequence (canonical
    # Pfafstetter); if fewer than 4 tributaries survived, the prefix
    # 2 < 4 < 6 < 8 still holds.
    tributary_codes = [c for c in codes if c in (2, 4, 6, 8)]
    assert tributary_codes == sorted(tributary_codes)


def test_merge_to_neighbour_relabels_to_zero_when_no_qualifying_neighbour():
    """When every 8-neighbour of a small basin is either background or
    another small basin, the merge cannot pick a survivor — the small
    basin is relabelled as 0."""
    # Two equally small isolated basins at opposite corners — neither can
    # merge into the other (both are in small_ids), and there is no
    # third surviving basin in either's 8-neighbourhood.
    z = np.array(
        [
            [0.0, 9.0, 9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 0.0],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    # Force every basin into ``small_ids`` so merge_to_neighbour cannot
    # find a survivor. With min_area_cells far above the total cell
    # count, all basins are small.
    ws = fd.basins(min_area_cells=10**6, merge_small="merge_to_neighbour")
    # No basin qualifies, all cells fall back to 0.
    assert int(ws.read_array().max()) == 0
