"""Tests for the deferred-item backfill: additional P26 export targets and the
P20 AGREE stream-burn method."""
from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import LineString

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


def _line_world(rows_cols: list[tuple[int, int]]) -> LineString:
    return LineString([(c + 0.5, -(r + 0.5)) for r, c in rows_cols])


# ----- HEC-RAS GeoTIFF export -----------------------------------------------

def test_hec_ras_export_writes_geotiff(tmp_path):
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    dem = _make_dem(z)
    path = str(tmp_path / "ras.tif")
    paths = dem.export(path, target="hec_ras", validate=False)
    assert "dem_tif" in paths
    assert os.path.exists(path)
    # Re-read via pyramids to verify the GeoTIFF round-trips.
    back = Dataset.read_file(path)
    arr = back.read_array().astype(np.float32)
    np.testing.assert_allclose(arr, z, atol=1e-3)


# ----- TUFLOW .flt + .hdr ---------------------------------------------------

def test_tuflow_export_writes_flt_and_hdr(tmp_path):
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    dem = _make_dem(z)
    path = str(tmp_path / "tflow")
    paths = dem.export(path, target="tuflow", validate=False)
    assert paths["dem_flt"].endswith(".flt")
    assert paths["dem_hdr"].endswith(".hdr")
    back = np.fromfile(paths["dem_flt"], dtype=np.float32).reshape(3, 3)
    np.testing.assert_allclose(back, z, atol=1e-3)
    hdr = open(paths["dem_hdr"]).read()
    assert "byteorder LSBFIRST" in hdr


# ----- SFINCS .dep + .msk ---------------------------------------------------

def test_sfincs_export_writes_dep_and_msk(tmp_path):
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    dem = _make_dem(z)
    path = str(tmp_path / "sfincs")
    paths = dem.export(path, target="sfincs", validate=False)
    assert paths["dem_dep"].endswith(".dep")
    assert paths["dem_msk"].endswith(".msk")
    back = np.fromfile(paths["dem_dep"], dtype=np.float32).reshape(3, 3)
    np.testing.assert_allclose(back, z, atol=1e-3)
    msk = np.fromfile(paths["dem_msk"], dtype=np.uint8).reshape(3, 3)
    # No NaN in this fixture, every cell is valid (msk = 1).
    assert (msk == 1).all()


def test_sfincs_msk_marks_nodata(tmp_path):
    z = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    dem = _make_dem(z)
    path = str(tmp_path / "sf")
    paths = dem.export(path, target="sfincs", validate=False)
    msk = np.fromfile(paths["dem_msk"], dtype=np.uint8).reshape(2, 2)
    assert msk[0, 1] == 0
    assert (msk[~np.isnan(z)] == 1).all()


# ----- Gmsh .geo ------------------------------------------------------------

def test_gmsh_export_writes_geo_script(tmp_path):
    z = np.zeros((4, 4), dtype=np.float32)
    dem = _make_dem(z)
    path = str(tmp_path / "mesh")
    paths = dem.export(path, target="gmsh", validate=False)
    assert paths["geo"].endswith(".geo")
    text = open(paths["geo"]).read()
    # Four corner points, four lines, one loop, one surface.
    for needle in ("Point(1)", "Point(4)", "Line(1)",
                   "Line Loop(1)", "Plane Surface(1)"):
        assert needle in text


# ----- Iber .dat ------------------------------------------------------------

def test_iber_export_writes_dat(tmp_path):
    z = np.zeros((3, 3), dtype=np.float32)
    dem = _make_dem(z)
    path = str(tmp_path / "iber")
    paths = dem.export(path, target="iber", validate=False)
    assert paths["dem_dat"].endswith(".dat")
    text = open(paths["dem_dat"]).read()
    assert "NCOLS 3" in text


# ----- AGREE stream burning --------------------------------------------------

def test_agree_lowers_buffer_cells_around_stream():
    z = np.full((7, 7), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(3, 0), (3, 6)])], crs=4326,
    )
    out = dem.burn_streams(
        streams, method="agree", sharp=5.0, smooth=0.0, buffer_cells=2,
    )
    vals = out.values
    # Stream cells (row 3) drop by `sharp` = 5: 10 - 5 = 5.
    np.testing.assert_allclose(vals[3, :], 5.0, atol=1e-3)
    # Buffer cells one step away (rows 2 and 4): drop by sharp * (1 - 1/2) = 2.5.
    assert vals[2, 3] == pytest.approx(7.5, abs=1e-3)
    # Cells outside the buffer (row 0 or 6): unchanged.
    assert vals[0, 0] == pytest.approx(10.0)


def test_agree_returns_dem():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(2, 0), (2, 4)])], crs=4326,
    )
    out = dem.burn_streams(streams, method="agree")
    assert isinstance(out, DEM)


def test_topological_breach_still_not_implemented():
    z = np.full((4, 4), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(1, 0), (1, 3)])], crs=4326,
    )
    with pytest.raises(NotImplementedError, match="topological_breach"):
        dem.burn_streams(streams, method="topological_breach")
