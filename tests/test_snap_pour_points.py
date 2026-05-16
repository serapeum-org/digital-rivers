"""Tests for `Accumulation.snap_pour_points` (P12)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import Point

from digitalrivers import DEM, Accumulation


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def _build_acc(z: np.ndarray) -> tuple[DEM, Accumulation]:
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    return dem, fd.accumulate()


def test_snap_max_accumulation_moves_point_to_thalweg():
    # 9x9 grid with a thalweg in column 4. Cells in column 4 have lower elevation.
    rows, cols = 9, 9
    z = np.full((rows, cols), 10.0, dtype=np.float32)
    z[:, 4] = 1.0  # thalweg
    dem, acc = _build_acc(z)
    # Pour point at (col=6, row=4). World coords: (6.5, -4.5) since cell_size=1, top_left=(0,0).
    px = 6.5
    py = -4.5
    pts = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(px, py)], crs=4326)
    out = acc.snap_pour_points(pts, radius_cells=3)
    # Should have moved toward column 4.
    snapped_col = int((out.iloc[0]["snapped_x"] - 0.5) / 1.0)
    assert snapped_col == 4
    assert out.iloc[0]["snap_distance_m"] > 0


def test_snap_radius_one_leaves_point_unchanged_if_no_higher_acc_nearby():
    z = np.full((9, 9), 10.0, dtype=np.float32)
    z[:, 4] = 1.0
    dem, acc = _build_acc(z)
    px = 6.5
    py = -4.5
    pts = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(px, py)], crs=4326)
    out = acc.snap_pour_points(pts, radius_cells=1)
    # 1-cell radius cannot reach column 4 from column 6.
    snapped_col = int((out.iloc[0]["snapped_x"] - 0.5) / 1.0)
    assert snapped_col != 4


def test_snap_radius_m_matches_radius_cells_at_unit_cell_size():
    z = np.full((9, 9), 10.0, dtype=np.float32)
    z[:, 4] = 1.0
    dem, acc = _build_acc(z, )
    pts = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(6.5, -4.5)], crs=4326)
    out_cells = acc.snap_pour_points(pts, radius_cells=3)
    out_m = acc.snap_pour_points(pts, radius_m=3.0)
    assert out_cells.iloc[0]["snapped_x"] == out_m.iloc[0]["snapped_x"]
    assert out_cells.iloc[0]["snapped_y"] == out_m.iloc[0]["snapped_y"]


def test_jenson_snaps_to_nearest_stream_cell():
    z = np.full((9, 9), 10.0, dtype=np.float32)
    z[:, 4] = 1.0
    dem, acc = _build_acc(z)
    sr = acc.streams(threshold=1)
    pts = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(6.5, -4.5)], crs=4326)
    out = acc.snap_pour_points(pts, radius_cells=5, method="jenson", streams=sr)
    # Should snap to column 4 (the only stream column).
    snapped_col = int((out.iloc[0]["snapped_x"] - 0.5) / 1.0)
    assert snapped_col == 4


def test_min_acc_filter_excludes_low_accumulation_candidates():
    z = np.full((9, 9), 10.0, dtype=np.float32)
    z[:, 4] = 1.0
    dem, acc = _build_acc(z)
    pts = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(6.5, -4.5)], crs=4326)
    # Set min_acc above any reachable accumulation; point should not move.
    huge = float(acc.read_array().max()) + 100.0
    out = acc.snap_pour_points(pts, radius_cells=3, min_acc=huge)
    snapped_col = int((out.iloc[0]["snapped_x"] - 0.5) / 1.0)
    assert snapped_col == 6  # unchanged


def test_both_radius_args_raises():
    z = np.full((4, 4), 5.0, dtype=np.float32)
    dem, acc = _build_acc(z)
    pts = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(0.5, -0.5)], crs=4326)
    with pytest.raises(ValueError, match="Exactly one"):
        acc.snap_pour_points(pts, radius_cells=1, radius_m=1.0)


def test_neither_radius_arg_raises():
    z = np.full((4, 4), 5.0, dtype=np.float32)
    dem, acc = _build_acc(z)
    pts = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(0.5, -0.5)], crs=4326)
    with pytest.raises(ValueError, match="Exactly one"):
        acc.snap_pour_points(pts)


def test_jenson_without_streams_raises():
    z = np.full((4, 4), 5.0, dtype=np.float32)
    dem, acc = _build_acc(z)
    pts = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(0.5, -0.5)], crs=4326)
    with pytest.raises(ValueError, match="method='jenson'"):
        acc.snap_pour_points(pts, radius_cells=1, method="jenson")


def test_unknown_method_raises():
    z = np.full((4, 4), 5.0, dtype=np.float32)
    dem, acc = _build_acc(z)
    pts = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(0.5, -0.5)], crs=4326)
    with pytest.raises(ValueError, match="method must be"):
        acc.snap_pour_points(pts, radius_cells=1, method="bogus")


def test_point_outside_envelope_returns_nan():
    z = np.full((4, 4), 5.0, dtype=np.float32)
    dem, acc = _build_acc(z)
    pts = gpd.GeoDataFrame(
        {"id": [0]}, geometry=[Point(1000.0, -1000.0)], crs=4326
    )
    out = acc.snap_pour_points(pts, radius_cells=1)
    assert np.isnan(out.iloc[0]["snap_distance_m"])


def test_jenson_method_nan_when_snap_target_is_input_cell():
    """I2 regression for the `method='jenson'` path: an unmoved snap
    also reports `snap_distance_m == NaN`."""
    z = np.array(
        [[9, 9, 9, 9], [9, 5, 4, 1], [9, 9, 9, 9]], dtype=np.float32
    )
    dem, acc = _build_acc(z)
    fd = dem.flow_direction(method="d8")
    sr = acc.streams(threshold=1)
    # Place the point on a stream cell already; jenson should not move it.
    geo = acc.geotransform
    x = geo[0] + (3 + 0.5) * geo[1]
    y = geo[3] + (1 + 0.5) * geo[5]
    pts = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(x, y)], crs=4326)
    out = acc.snap_pour_points(
        pts, radius_cells=1, method="jenson", streams=sr,
    )
    assert np.isnan(out.iloc[0]["snap_distance_m"])
