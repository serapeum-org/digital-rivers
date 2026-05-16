"""Tests for ``DEM.export`` (P26) and ``DEM.subgrid_bathymetry`` (P27)."""
from __future__ import annotations

import os

import numpy as np
import pytest
from pyramids.dataset import Dataset

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


# ----- P26 export ------------------------------------------------------------

def test_lisflood_fp_export_writes_arc_ascii_header(tmp_path):
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    dem = _make_dem(z)
    path = tmp_path / "dem.asc"
    paths = dem.export(str(path), target="lisflood_fp", validate=False)
    assert "dem_asc" in paths
    text = path.read_text()
    assert text.startswith("ncols 3")
    # Header has 6 lines.
    header_lines = text.splitlines()[:6]
    assert any("nrows" in line for line in header_lines)
    assert any("cellsize" in line for line in header_lines)


def test_lisflood_fp_round_trip_with_numpy_loadtxt(tmp_path):
    z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dem = _make_dem(z)
    path = tmp_path / "dem.asc"
    dem.export(str(path), target="lisflood_fp", validate=False)
    back = np.loadtxt(str(path), skiprows=6)
    np.testing.assert_allclose(back, z, atol=1e-3)


def test_hec_ras_export_now_writes_geotiff(tmp_path):
    """HEC-RAS GeoTIFF writer shipped in the backfill commit; the test
    that previously asserted NotImplementedError is updated to verify
    the writer succeeds and the path is returned."""
    z = np.arange(4, dtype=np.float32).reshape(2, 2)
    dem = _make_dem(z)
    paths = dem.export(str(tmp_path / "out.tif"), target="hec_ras",
                       validate=False)
    assert "dem_tif" in paths


def test_unknown_target_raises(tmp_path):
    z = np.array([[1.0, 2.0]], dtype=np.float32)
    dem = _make_dem(z)
    with pytest.raises(ValueError, match="target must be"):
        dem.export(str(tmp_path / "out"), target="bogus", validate=False)


def test_validate_rejects_dem_with_sinks(tmp_path):
    """A DEM with an internal sink fails export under validate=True."""
    z = np.array(
        [
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 1, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    with pytest.raises(RuntimeError, match="internal sinks"):
        dem.export(str(tmp_path / "dem.asc"), target="lisflood_fp",
                   validate=True)


# ----- P27 sub-grid bathymetry ----------------------------------------------

def test_subgrid_returns_one_row_per_coarse_cell():
    z = np.arange(16, dtype=np.float32).reshape(4, 4)
    dem = _make_dem(z)
    df = dem.subgrid_bathymetry(scale_factor=2, n_bins=3)
    # 4x4 with scale_factor=2 → 2x2 coarse grid = 4 rows.
    assert len(df) == 4


def test_subgrid_columns_include_z_min_z_max_and_fracs():
    z = np.arange(16, dtype=np.float32).reshape(4, 4)
    dem = _make_dem(z)
    df = dem.subgrid_bathymetry(scale_factor=2, n_bins=3)
    assert "z_min" in df.columns
    assert "z_max" in df.columns
    for k in range(1, 4):
        assert f"frac_below_{k}" in df.columns


def test_subgrid_invalid_scale_raises():
    z = np.arange(4, dtype=np.float32).reshape(2, 2)
    dem = _make_dem(z)
    with pytest.raises(ValueError, match="scale_factor"):
        dem.subgrid_bathymetry(scale_factor=1)


def test_subgrid_invalid_n_bins_raises():
    z = np.arange(16, dtype=np.float32).reshape(4, 4)
    dem = _make_dem(z)
    with pytest.raises(ValueError, match="n_bins"):
        dem.subgrid_bathymetry(scale_factor=2, n_bins=0)
