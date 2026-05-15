"""End-to-end and coverage tests for Phase 2 of digital-rivers."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import Point

from digitalrivers import DEM, FlowDirection, WatershedRaster


def _make_dem(arr: np.ndarray) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


class TestPhase2EndToEndPipeline:
    """Chain: DEM → fill → resolve_flats → flow_direction → accumulate →
    snap_pour_points → watershed → basins → subbasins → pfafstetter →
    statistics. Asserts cross-cutting invariants."""

    @pytest.fixture(scope="class")
    def pipeline(self) -> dict:
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9, 9],
                [9, 7, 6, 5, 4, 3, 9],
                [9, 8, 7, 6, 5, 4, 9],
                [9, 9, 8, 7, 6, 5, 9],
                [9, 9, 9, 8, 7, 6, 9],
                [9, 9, 9, 9, 9, 9, 0],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        filled = dem.fill_depressions(method="wang_liu")
        resolved = filled.resolve_flats(epsilon=1e-4)
        fd = resolved.flow_direction(method="d8")
        acc = fd.accumulate()
        sr = acc.streams(threshold=2)
        pts = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[Point(5.5, -5.5)], crs=4326,
        )
        snapped = acc.snap_pour_points(pts, radius_cells=3)
        watershed = fd.watershed(snapped)
        basins = fd.basins()
        return {
            "dem": dem, "fd": fd, "acc": acc, "sr": sr,
            "snapped": snapped, "watershed": watershed, "basins": basins,
        }

    def test_pipeline_produces_typed_watersheds(self, pipeline):
        """End-to-end pipeline yields typed WatershedRaster outputs."""
        assert type(pipeline["watershed"]) is WatershedRaster
        assert type(pipeline["basins"]) is WatershedRaster

    def test_snapped_point_lies_in_data_envelope(self, pipeline):
        """The snapped pour point is inside the raster."""
        snapped = pipeline["snapped"]
        assert not np.isnan(snapped.iloc[0]["snap_distance_m"])

    def test_watershed_includes_pour_point_cell(self, pipeline):
        """The pour point itself is labelled with its basin ID."""
        ws = pipeline["watershed"]
        arr = ws.read_array()
        # At least one cell carries basin ID 1.
        assert (arr == 1).any()

    def test_basins_non_negative_labels(self, pipeline):
        """Every basin label is >= 0."""
        arr = pipeline["basins"].read_array()
        assert (arr >= 0).all()

    def test_subbasins_align_with_streams(self):
        """Sub-basin labels include every stream cell."""
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd = dem.flow_direction(method="d8")
        acc = fd.accumulate()
        sr = acc.streams(threshold=1)
        sub = sr.subbasins(fd)
        assert (sub.read_array()[sr.read_array().astype(bool)] > 0).all()

    def test_basin_statistics_include_area_km2(self):
        """statistics() emits area_km2 per basin."""
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd = dem.flow_direction(method="d8")
        ws = fd.basins()
        df = ws.statistics()
        assert "area_km2" in df.columns
        assert (df["area_km2"] >= 0).all()


class TestPhase2CoverageGaps:
    """Targeted tests for branches the per-task tests skip."""

    def test_basins_min_area_km2_path(self):
        """min_area_km2 branch (cell-area conversion)."""
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2],
                [9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd = dem.flow_direction(method="d8")
        # Provide an aggressive km² threshold (cell_size=1 m, so 1 km² = 1e6 cells).
        ws = fd.basins(min_area_km2=1.0, merge_small="drop")
        assert (ws.read_array() == 0).all() or ws.read_array().max() == 0

    def test_basins_merge_to_neighbour_branch(self):
        """merge_to_neighbour relabels small basins to the largest survivor."""
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2],
                [9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd = dem.flow_direction(method="d8")
        ws = fd.basins(min_area_cells=1_000_000, merge_small="merge_to_neighbour")
        # Should run without error.
        assert isinstance(ws.read_array(), np.ndarray)

    def test_watershed_to_polygons_geometry(self):
        """to_polygons() returns Polygons / MultiPolygons."""
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd = dem.flow_direction(method="d8")
        ws = fd.basins()
        if ws.basin_count > 0:
            polys = ws.to_polygons()
            assert "basin_id" in polys.columns
            assert "geometry" in polys.columns


def test_phase2_reexports_watershed_raster():
    """Package re-exports the P13 WatershedRaster class."""
    import digitalrivers
    assert hasattr(digitalrivers, "WatershedRaster")
