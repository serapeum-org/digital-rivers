"""Tests for `DEM.burn_streams` (P20)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import LineString

from digitalrivers import DEM


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def _line_world(rows_cols: list[tuple[int, int]]) -> LineString:
    """Convert a list of (row, col) into world coords for cell_size=1, top=(0,0)."""
    return LineString([(c + 0.5, -(r + 0.5)) for r, c in rows_cols])


def test_fill_burn_lowers_stream_cells():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(2, 0), (2, 4)])], crs=4326,
    )
    burnt = dem.burn_streams(streams, method="fill_burn", constant_drop=2.0)
    out = burnt.values
    # Stream cells (along row 2) must be at-or-below the original surface,
    # and strictly lower than the rim (row 0) — the burn actually changed
    # something. The non-stream row should be unchanged for a flat-fill
    # method when no depression exists.
    assert float(out[2, :].max()) <= 10.0
    assert float(out[2, 2]) < float(out[0, 0])


def test_fill_burn_returns_dem():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(2, 0), (2, 4)])], crs=4326,
    )
    burnt = dem.burn_streams(streams)
    assert isinstance(burnt, DEM)


def test_fill_burn_inplace_returns_none():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(2, 0), (2, 4)])], crs=4326,
    )
    result = dem.burn_streams(streams, inplace=True)
    assert result is None


def test_agree_now_implemented_and_returns_dem():
    """AGREE shipped in the backfill commit; verify it produces a DEM."""
    z = np.full((4, 4), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(1, 0), (1, 3)])], crs=4326,
    )
    out = dem.burn_streams(streams, method="agree")
    assert isinstance(out, DEM)


def test_topological_breach_now_implemented():
    """topological_breach now ships as rasterise-streams + P3 breach
    composition; verify the DEM is produced."""
    z = np.full((4, 4), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(1, 0), (1, 3)])], crs=4326,
    )
    out = dem.burn_streams(streams, method="topological_breach")
    assert isinstance(out, DEM)


def test_empty_streams_returns_filled_dem():
    z = np.full((4, 4), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(geometry=[], crs=4326)
    burnt = dem.burn_streams(streams)
    assert isinstance(burnt, DEM)
