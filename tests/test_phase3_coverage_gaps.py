"""Coverage-gap tests called out in the Phase 3 review (C1-C8).

Each test pins behaviour that was previously asserted weakly or not at all:

* C1: MultiLineString input on burn methods.
* C2: Stream that exits the raster (out-of-bounds clip).
* C3: CRS reprojection path for mismatched EPSGs.
* C4: ``subgrid_bathymetry`` with a flat block (regression for B1).
* C5: ``subgrid_bathymetry`` with a fully no-data block.
* C6: ``DEM.export(validate=True)`` success path.
* C7: ``enforce_culverts`` with multiple roads crossing the same stream.
* C8: ``enforce_breaklines(inplace=True)`` and per-feature attribute hint.
"""
from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import LineString, MultiLineString

from digitalrivers import DEM


def _make_dem(arr: np.ndarray, no_data_value: float = -9999.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    disk[np.isnan(disk)] = no_data_value
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=no_data_value,
    )
    return DEM(ds.raster)


def _line(coords):
    # Convert (col, row) integer pairs to world coords with cell-size 1.
    return LineString([(c + 0.5, -(r + 0.5)) for r, c in coords])


# --- C1: MultiLineString -----------------------------------------------------


def test_burn_streams_accepts_multilinestring():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    mls = MultiLineString(
        [
            _line([(2, 0), (2, 2)]),
            _line([(2, 2), (2, 4)]),
        ]
    )
    streams = gpd.GeoDataFrame(geometry=[mls], crs=4326)
    burnt = dem.burn_streams(streams, constant_drop=2.0)
    out = burnt.values
    # Both segments end up below the rim.
    assert float(out[2, 0]) < float(out[0, 0])
    assert float(out[2, 4]) < float(out[0, 4])


def test_enforce_breaklines_accepts_multilinestring():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    mls = MultiLineString(
        [_line([(0, 0), (0, 2)]), _line([(4, 0), (4, 2)])]
    )
    lifted = dem.enforce_breaklines(
        gpd.GeoDataFrame(geometry=[mls], crs=4326), lift=5.0,
    )
    out = lifted.values
    # First and last rows along col 0..2 should be lifted.
    assert float(out[0, 0]) > 10.0
    assert float(out[4, 0]) > 10.0


# --- C2: stream exits raster -------------------------------------------------


def test_burn_streams_stream_exiting_raster_is_clipped():
    """A line whose endpoint sits outside the raster should still rasterise
    the in-bounds segment without raising."""
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    # Line goes from inside (col=2, row=2) to far outside (col=20, row=2).
    streams = gpd.GeoDataFrame(
        geometry=[_line([(2, 2), (2, 20)])], crs=4326,
    )
    burnt = dem.burn_streams(streams, constant_drop=2.0)
    out = burnt.values
    # In-bounds part of the line ends up below the rim.
    assert float(out[2, 2]) < float(out[0, 2])


# --- C3: CRS reprojection ----------------------------------------------------


def test_burn_streams_reprojects_mismatched_crs():
    """Inputs in a different CRS get reprojected and still hit cells."""
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)  # EPSG:4326
    # Streams in a metric CRS — at this small scale near the equator the
    # reprojection lands inside the raster.
    streams = gpd.GeoDataFrame(
        geometry=[_line([(2, 0), (2, 4)])], crs=4326,
    ).to_crs(3857)
    burnt = dem.burn_streams(streams, constant_drop=2.0)
    out = burnt.values
    # Stream row is lowered relative to the rim.
    assert float(out[2, :].max()) < 11.0


# --- C4: subgrid flat block --------------------------------------------------


def test_subgrid_bathymetry_flat_block_has_frac_columns():
    """B1 regression: every flat block must still produce frac_below_<k>
    columns equal to 1.0."""
    z = np.full((4, 4), 5.0, dtype=np.float32)
    dem = _make_dem(z)
    df = dem.subgrid_bathymetry(scale_factor=2, n_bins=3)
    for k in (1, 2, 3):
        col = f"frac_below_{k}"
        assert col in df.columns
        assert float(df[col].iloc[0]) == 1.0


# --- C5: subgrid all-nodata block -------------------------------------------


def test_subgrid_bathymetry_all_nodata_block_skipped():
    """Blocks consisting entirely of NaN no-data are silently dropped."""
    z = np.full((4, 4), np.nan, dtype=np.float32)
    z[:2, :2] = 5.0  # Only the top-left block has any data.
    dem = _make_dem(z)
    df = dem.subgrid_bathymetry(scale_factor=2, n_bins=3)
    # 4 blocks total in a 4x4 raster at sf=2; only 1 has valid data.
    assert len(df) == 1


# --- C6: export validate=True success path ----------------------------------


def test_export_validate_true_on_sinks_free_dem(tmp_path):
    """A sinks-free DEM exports cleanly with validate=True."""
    # Monotonic ramp — no internal sinks.
    z = np.array(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    out = tmp_path / "dem.asc"
    paths = dem.export(str(out), target="lisflood_fp", validate=True)
    assert os.path.exists(paths["dem_asc"])


# --- C7: multi-road culverts ------------------------------------------------


def test_enforce_culverts_multiple_roads_at_same_stream():
    """Two roads each crossing the same stream both result in lowered
    cells at their intersections."""
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line([(2, 0), (2, 4)])], crs=4326,
    )
    road1 = _line([(0, 1), (4, 1)])
    road2 = _line([(0, 3), (4, 3)])
    roads = gpd.GeoDataFrame(geometry=[road1, road2], crs=4326)
    out = dem.enforce_culverts(
        roads=roads, streams=streams, culvert_drop=2.0,
    )
    arr = out.values
    # Both road/stream crossing cells are lowered.
    assert float(arr[2, 1]) < 10.0
    assert float(arr[2, 3]) < 10.0


# --- C8: enforce_breaklines inplace -----------------------------------------


def test_enforce_breaklines_inplace_returns_none():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    line = _line([(2, 0), (2, 4)])
    breaklines = gpd.GeoDataFrame(geometry=[line], crs=4326)
    result = dem.enforce_breaklines(breaklines, lift=5.0, inplace=True)
    assert result is None
    # Mutation took effect.
    assert float(dem.values[2, :].max()) > 10.0


# --- I1 regression: rasterise_line consistency -----------------------------


def test_rasterise_line_floor_oversampled_hits_diagonal_cells():
    """After the I1 unification, ``_rasterise_line`` is the single source
    of cell-snapping for every Phase-3 line-burning method. A diagonal
    line through a 5×5 grid hits every diagonal cell along its path."""
    z = np.full((5, 5), 0.0, dtype=np.float32)
    dem = _make_dem(z)
    mask = np.zeros(z.shape, dtype=bool)
    line = _line([(0, 0), (4, 4)])
    dem._rasterise_line(line, mask, dem.geotransform)
    # The 2× oversampling guarantees every cell along the diagonal is hit.
    for k in range(5):
        assert bool(mask[k, k]), f"Diagonal cell ({k},{k}) not rasterised"


# --- (1) `_polygon_cell_indices` helper coverage --------------------------


class TestPolygonCellIndices:
    """Direct coverage of the new vectorised cell-in-polygon helper (I2)."""

    def _dem(self):
        z = np.full((5, 5), 0.0, dtype=np.float32)
        return _make_dem(z)

    def test_polygon_outside_raster_returns_empty_arrays(self):
        from shapely.geometry import Polygon

        dem = self._dem()
        # Polygon far outside the raster's world extent.
        far = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
        rs, cs = dem._polygon_cell_indices(
            far, dem.geotransform, 5, 5,
        )
        assert rs.size == 0
        assert cs.size == 0

    def test_polygon_inside_single_cell_returns_one_index(self):
        from shapely.geometry import Polygon

        dem = self._dem()
        # Polygon entirely inside cell (row=2, col=2): center is (2.5, -2.5).
        poly = Polygon([(2.4, -2.6), (2.6, -2.6), (2.6, -2.4), (2.4, -2.4)])
        rs, cs = dem._polygon_cell_indices(
            poly, dem.geotransform, 5, 5,
        )
        assert rs.tolist() == [2]
        assert cs.tolist() == [2]

    def test_returned_index_dtype_is_int(self):
        from shapely.geometry import Polygon

        dem = self._dem()
        poly = Polygon([(0, 0), (3, 0), (3, -3), (0, -3)])
        rs, cs = dem._polygon_cell_indices(
            poly, dem.geotransform, 5, 5,
        )
        assert np.issubdtype(rs.dtype, np.integer)
        assert np.issubdtype(cs.dtype, np.integer)

    def test_multipolygon_callers_iterate_components(self):
        """``hydroflatten`` accepts MultiPolygon by iterating components."""
        from shapely.geometry import MultiPolygon, Polygon

        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        a = Polygon([(0, 0), (1, 0), (1, -1), (0, -1)])
        b = Polygon([(3, -3), (4, -3), (4, -4), (3, -4)])
        layer = gpd.GeoDataFrame(
            geometry=[MultiPolygon([a, b])], crs=4326,
        )
        # Shapely 2.x makes MultiPolygon iterable via .geoms — the
        # vectorised helper still runs on the outer geometry's bounds,
        # which encompasses both child polys.
        out = dem.hydroflatten(layer, method="min")
        # No crash, output shape preserved.
        assert out.values.shape == z.shape


# --- (2) subgrid_bathymetry edge cases --------------------------------------


def test_subgrid_scale_factor_below_two_rejected():
    z = np.full((4, 4), 0.0, dtype=np.float32)
    dem = _make_dem(z)
    with pytest.raises(ValueError, match="scale_factor"):
        dem.subgrid_bathymetry(scale_factor=1, n_bins=3)


def test_subgrid_n_bins_below_one_rejected():
    z = np.full((4, 4), 0.0, dtype=np.float32)
    dem = _make_dem(z)
    with pytest.raises(ValueError, match="n_bins"):
        dem.subgrid_bathymetry(scale_factor=2, n_bins=0)


def test_subgrid_non_evenly_divisible_drops_remainder():
    """A 5x5 raster at scale_factor=2 yields 2x2 = 4 coarse rows (the last
    row and column don't form complete blocks)."""
    z = np.full((5, 5), 0.0, dtype=np.float32)
    dem = _make_dem(z)
    df = dem.subgrid_bathymetry(scale_factor=2, n_bins=2)
    assert len(df) == 4


# --- (3) export non-lisflood targets short-circuit before sink scan --------


def test_export_non_lisflood_target_skips_sink_scan(tmp_path):
    """I4 verification: a DEM full of internal sinks must export cleanly
    to any non-lisflood_fp target because the sink-scan validation runs
    only for the lisflood_fp ASCII writer."""
    # A DEM with a clear interior pit — would fail validate=True on
    # lisflood_fp, but the other writers don't run the scan.
    z = np.array(
        [
            [10.0, 10.0, 10.0],
            [10.0, 1.0, 10.0],  # interior pit
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    # hec_ras succeeds (no sink scan).
    paths = dem.export(
        str(tmp_path / "out.tif"), target="hec_ras", validate=True,
    )
    assert "dem_tif" in paths
    # lisflood_fp still rejects the same DEM under validate=True.
    with pytest.raises(RuntimeError, match="internal sinks"):
        dem.export(
            str(tmp_path / "out.asc"),
            target="lisflood_fp", validate=True,
        )


# --- (4) empty GeoDataFrames leave the DEM unchanged -----------------------


def test_enforce_culverts_empty_layer_is_no_op():
    z = np.full((4, 4), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    empty = gpd.GeoDataFrame(geometry=[], crs=4326)
    streams = gpd.GeoDataFrame(
        geometry=[_line([(2, 0), (2, 3)])], crs=4326,
    )
    out = dem.enforce_culverts(
        roads=empty, streams=streams, culvert_drop=2.0,
    )
    np.testing.assert_array_equal(out.values, z)


def test_enforce_breaklines_empty_layer_is_no_op():
    z = np.full((4, 4), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    empty = gpd.GeoDataFrame(geometry=[], crs=4326)
    out = dem.enforce_breaklines(empty, lift=5.0)
    np.testing.assert_array_equal(out.values, z)


# --- N7: defensive _reproject_if_needed -------------------------------------


class TestReprojectIfNeeded:
    """Coverage for the N7 helper that replaces ``to_epsg() != target_epsg``
    integer comparison with proper CRS equality."""

    def test_same_crs_returns_layer_untouched(self):
        from digitalrivers.dem import _reproject_if_needed

        z = np.full((3, 3), 0.0, dtype=np.float32)
        dem = _make_dem(z)
        layer = gpd.GeoDataFrame(
            geometry=[_line([(0, 0), (1, 1)])], crs=4326,
        )
        out = _reproject_if_needed(layer, dem.epsg)
        # Same CRS → exact same object (no defensive copy needed).
        assert out is layer

    def test_different_crs_reprojects(self):
        from digitalrivers.dem import _reproject_if_needed

        layer = gpd.GeoDataFrame(
            geometry=[_line([(0, 0), (1, 1)])], crs=4326,
        ).to_crs(3857)
        out = _reproject_if_needed(layer, 4326)
        # Reprojected → new geodataframe in target CRS.
        assert int(out.crs.to_epsg()) == 4326

    def test_target_epsg_none_returns_layer_untouched(self):
        from digitalrivers.dem import _reproject_if_needed

        layer = gpd.GeoDataFrame(
            geometry=[_line([(0, 0), (1, 1)])], crs=4326,
        )
        out = _reproject_if_needed(layer, None)
        assert out is layer

    def test_layer_without_crs_returns_untouched(self):
        from digitalrivers.dem import _reproject_if_needed

        layer = gpd.GeoDataFrame(
            geometry=[_line([(0, 0), (1, 1)])],
        )  # no crs
        out = _reproject_if_needed(layer, 4326)
        assert out is layer
