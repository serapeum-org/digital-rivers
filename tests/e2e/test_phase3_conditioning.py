"""Tests for the DEM conditioning operations added in Phase 3 (P21–P25)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import LineString, Polygon

from digitalrivers import DEM


def _make_dem(arr: np.ndarray) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


# ----- P21: culverts ---------------------------------------------------------

def test_enforce_culverts_lowers_intersection_cells():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    # Stream runs east-west at row 2; road runs north-south at col 2.
    streams = gpd.GeoDataFrame(
        geometry=[LineString([(0.5, -2.5), (4.5, -2.5)])], crs=4326,
    )
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(2.5, -0.5), (2.5, -4.5)])], crs=4326,
    )
    out = dem.enforce_culverts(roads, streams, culvert_drop=2.0)
    vals = out.values
    # Intersection cell (2, 2) is lowered.
    assert vals[2, 2] < 10.0


def test_enforce_culverts_no_intersection_leaves_dem_unchanged():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[LineString([(0.5, -2.5), (4.5, -2.5)])], crs=4326,
    )
    # Road parallel to stream, not crossing.
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0.5, -0.5), (4.5, -0.5)])], crs=4326,
    )
    out = dem.enforce_culverts(roads, streams, culvert_drop=2.0)
    np.testing.assert_array_equal(out.values, dem.values)


# ----- P22: hydroflatten -----------------------------------------------------

def test_hydroflatten_min_assigns_lowest_value_to_polygon_cells():
    z = np.array(
        [
            [10, 10, 10, 10],
            [10, 5, 6, 10],
            [10, 4, 7, 10],
            [10, 10, 10, 10],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    # Polygon covering cells (1,1), (1,2), (2,1), (2,2).
    poly = Polygon([(1, -1), (3, -1), (3, -3), (1, -3)])
    waters = gpd.GeoDataFrame(geometry=[poly], crs=4326)
    out = dem.hydroflatten(waters, method="min")
    vals = out.values
    # All four polygon cells should equal min(5, 6, 4, 7) = 4.
    inside = [vals[1, 1], vals[1, 2], vals[2, 1], vals[2, 2]]
    assert all(v == pytest.approx(4.0) for v in inside)


def test_hydroflatten_invalid_method_raises():
    z = np.array([[1, 2], [3, 4]], dtype=np.float32)
    dem = _make_dem(z)
    waters = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, -1), (0, -1)])], crs=4326,
    )
    with pytest.raises(ValueError, match="method must be"):
        dem.hydroflatten(waters, method="bogus")


# ----- P23: building footprint burning ---------------------------------------

def test_burn_buildings_lifts_polygon_cells():
    z = np.full((4, 4), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    poly = Polygon([(1, -1), (3, -1), (3, -3), (1, -3)])
    bld = gpd.GeoDataFrame(geometry=[poly], crs=4326)
    out = dem.burn_buildings(bld, lift=20.0)
    vals = out.values
    inside = [vals[1, 1], vals[1, 2], vals[2, 1], vals[2, 2]]
    assert all(v == pytest.approx(30.0) for v in inside)


# ----- P24: breaklines -------------------------------------------------------

def test_enforce_breaklines_raises_line_cells():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    line = LineString([(0.5, -2.5), (4.5, -2.5)])
    bl = gpd.GeoDataFrame(geometry=[line], crs=4326)
    out = dem.enforce_breaklines(bl, lift=5.0)
    vals = out.values
    # The line cells along row 2 are raised.
    assert (vals[2, :] >= 14.5).all()


# ----- P25: ANUDEM-lite is now implemented (Laplacian relaxation) -----------

def test_anudem_lite_now_implemented():
    """P25 ANUDEM-lite was shipped in a backfill commit — verify it
    returns a typed DEM with finite values."""
    z = np.array(
        [[10, np.nan, 5], [np.nan, np.nan, np.nan], [10, np.nan, 5]],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    out = dem.anudem_interpolate()
    assert isinstance(out, DEM)
    assert np.all(np.isfinite(out.values))
