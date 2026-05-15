"""End-to-end pipeline + coverage gap tests for Phase 3."""
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


class TestPhase3ConditioningPipeline:
    """Chain: DEM → burn_streams → enforce_culverts → hydroflatten →
    burn_buildings → enforce_breaklines → fill → export. Asserts that
    each conditioning step composes cleanly."""

    @pytest.fixture(scope="class")
    def pipeline(self, tmp_path_factory) -> dict:
        z = np.full((10, 10), 10.0, dtype=np.float32)
        # Add a pit so fill is non-trivial.
        z[5, 5] = 0.0
        dem = _make_dem(z)
        streams = gpd.GeoDataFrame(
            geometry=[LineString([(0.5, -5.5), (9.5, -5.5)])], crs=4326,
        )
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(5.5, -0.5), (5.5, -9.5)])], crs=4326,
        )
        lakes = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (3, 0), (3, -3), (0, -3)])], crs=4326,
        )
        bld = gpd.GeoDataFrame(
            geometry=[Polygon([(7, -7), (9, -7), (9, -9), (7, -9)])], crs=4326,
        )
        bl = gpd.GeoDataFrame(
            geometry=[LineString([(0.5, -1.5), (4.5, -1.5)])], crs=4326,
        )

        d1 = dem.burn_streams(streams)
        d2 = d1.enforce_culverts(roads, streams, culvert_drop=1.0)
        d3 = d2.hydroflatten(lakes, method="min")
        d4 = d3.burn_buildings(bld, lift=20.0)
        d5 = d4.enforce_breaklines(bl, lift=3.0)
        filled = d5.fill_depressions(method="priority_flood", epsilon=0.1)

        tmp = tmp_path_factory.mktemp("phase3_e2e")
        out_path = str(tmp / "conditioned.asc")
        # validate may fail if sinks remain after sequential conditioning;
        # validate=False to focus on the export I/O path.
        paths = filled.export(out_path, target="lisflood_fp", validate=False)
        return {
            "input": dem, "conditioned": filled, "out_paths": paths,
        }

    def test_pipeline_runs_to_completion(self, pipeline):
        """Sequential conditioning ops + export produce at least one file."""
        assert pipeline["out_paths"]["dem_asc"]

    def test_export_round_trips_via_loadtxt(self, pipeline):
        """The Arc-ASCII output reloads as a NumPy array."""
        back = np.loadtxt(pipeline["out_paths"]["dem_asc"], skiprows=6)
        assert back.shape == pipeline["conditioned"].shape[1:]

    def test_conditioned_dem_has_no_sinks(self, pipeline):
        """fill_depressions(method='priority_flood', epsilon=0.1) leaves
        the conditioned DEM sinks-free for downstream flow routing."""
        from digitalrivers._pitremoval import local_minima_8
        sinks = local_minima_8(pipeline["conditioned"].values)
        assert int(sinks.sum()) == 0
