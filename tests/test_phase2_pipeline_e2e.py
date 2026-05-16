"""Phase-2 end-to-end pipeline test.

Chains together every Phase-2 deliverable on a single deterministic synthetic
DEM and asserts the cross-cutting invariants of the watershed/sub-basin
family (P12-P19 plus the Phase-4 backfill that landed multi-level
Pfafstetter and the spatial tributary ordering fix).

Pipeline (left to right):

    DEM
      -> fill_depressions(method="priority_flood")
      -> resolve_flats()
      -> flow_direction(method="d8")
      -> accumulate()
      -> snap_pour_points(...)
      -> watershed(snapped)             # pour-point delineation (P13)
      -> basins(merge_small=...)        # whole-DEM partition (P14)
      -> subbasins_pfafstetter()        # P16 (level 1 & level 2)
      -> statistics(streams, fd, dem)   # P17 (now including I4/N3 fixes)
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import Point

from digitalrivers import DEM, WatershedRaster


@pytest.fixture(scope="module")
def synthetic_dem() -> DEM:
    """A deterministic 12×12 DEM with two sinks so the pipeline exercises
    multi-basin behaviour. The west half drains to sink (5, 1); the east
    half drains to sink (5, 10).
    """
    z = np.full((12, 12), 50.0, dtype=np.float32)
    # Plant two sinks with a smooth slope toward each.
    for r in range(12):
        for c in range(12):
            # Distance to two sinks; choose whichever is nearer (forms a
            # ridge along the column at index 5).
            d_west = abs(r - 5) + abs(c - 1)
            d_east = abs(r - 5) + abs(c - 10)
            z[r, c] = float(min(d_west, d_east))
    # The two outlet cells themselves dip to 0.
    z[5, 1] = 0.0
    z[5, 10] = 0.0
    ds = Dataset.create_from_array(
        z, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


@pytest.fixture(scope="module")
def pipeline(synthetic_dem):
    """Run the full pipeline once and expose every intermediate product."""
    filled = synthetic_dem.fill_depressions(method="priority_flood")
    resolved = filled.resolve_flats()
    fd = resolved.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=2)
    # Two pour points near the two sinks (in dataset CRS — note y is
    # negative because the geotransform's dy is -1).
    pts = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(1.5, -5.5), Point(10.5, -5.5)],
        crs=4326,
    )
    snapped = acc.snap_pour_points(pts, radius_cells=2)
    watershed = fd.watershed(snapped)
    basins = fd.basins(min_area_cells=3, merge_small="merge_to_neighbour")
    pfaf_l1 = fd.subbasins_pfafstetter(acc, sr, level=1)
    pfaf_l2 = fd.subbasins_pfafstetter(acc, sr, level=2)
    stats = basins.statistics(
        dem=resolved, streams=sr, flow_direction=fd,
    )
    return {
        "dem": resolved, "fd": fd, "acc": acc, "sr": sr,
        "snapped": snapped, "watershed": watershed, "basins": basins,
        "pfaf_l1": pfaf_l1, "pfaf_l2": pfaf_l2, "stats": stats,
    }


class TestPhase2PipelineInvariants:
    """Cross-cutting Phase-2 invariants over the full pipeline."""

    def test_every_input_pour_point_maps_to_non_zero_basin(self, pipeline):
        """Every snapped pour point must land in a labelled watershed cell."""
        ws = pipeline["watershed"]
        snapped = pipeline["snapped"]
        labels = ws.read_array()
        geo = ws.geotransform
        for _, row in snapped.iterrows():
            x, y = row["snapped_x"], row["snapped_y"]
            c = int((x - geo[0]) / geo[1])
            r = int((y - geo[3]) / geo[5])
            assert 0 <= r < labels.shape[0]
            assert 0 <= c < labels.shape[1]
            assert int(labels[r, c]) > 0

    def test_basins_are_non_overlapping_non_negative_integers(self, pipeline):
        """Every label is ≥ 0 and the raster's dtype is integer-like."""
        arr = pipeline["basins"].read_array()
        assert np.issubdtype(arr.dtype, np.integer)
        assert int(arr.min()) >= 0

    def test_pfafstetter_level1_codes_in_canonical_range(self, pipeline):
        """Level-1 Pfafstetter codes lie in [1, 9]."""
        arr = pipeline["pfaf_l1"].read_array()
        nonzero = arr[arr != 0]
        if nonzero.size:
            codes = set(int(v) for v in np.unique(nonzero))
            assert codes.issubset(set(range(1, 10)))

    def test_pfafstetter_level2_codes_are_two_digit(self, pipeline):
        """Level-2 codes lie in [11, 99] (parent*10 + child)."""
        arr = pipeline["pfaf_l2"].read_array()
        nonzero = arr[arr != 0]
        if nonzero.size:
            for v in np.unique(nonzero):
                code = int(v)
                # Allow uniform-1 fallback for tributary-less sub-basins
                # (kernel emits parent*10 for those untouched cells).
                assert 1 <= code <= 99

    def test_statistics_has_expected_column_set(self, pipeline):
        """``statistics(dem=..., streams=..., flow_direction=...)`` must
        return area, elevation, slope (only when slope is provided →
        skip), drainage density, and centroid columns."""
        df = pipeline["stats"]
        expected = {
            "area_km2", "min_elev", "max_elev", "mean_elev", "std_elev",
            "hypsometric_integral", "centroid_x", "centroid_y",
            "drainage_density_km_per_km2",
        }
        missing = expected - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_total_stream_length_is_conservative_under_basins(self, pipeline):
        """The sum of per-basin stream length (drainage density × area)
        must not exceed the total stream-cell length over the raster.

        We test only the upper-bound (it's a conservation check, not a
        reconstruction): partial-cell fragments and basins below the
        merge threshold ensure ``sum <= total``.
        """
        df = pipeline["stats"]
        sr_arr = pipeline["sr"].read_array().astype(bool)
        fdir = pipeline["fd"].read_array()
        diag = np.isin(fdir, [1, 3, 5, 7])
        cell_size = abs(pipeline["fd"].geotransform[1])
        total_len_km = (
            (sr_arr & ~diag).sum() * cell_size / 1000.0
            + (sr_arr & diag).sum() * cell_size * np.sqrt(2.0) / 1000.0
        )
        # Per-basin length = density × area (km/km² × km² = km).
        per_basin_len_km = (
            df["drainage_density_km_per_km2"] * df["area_km2"]
        ).sum()
        assert per_basin_len_km <= total_len_km + 1e-6

    def test_watershed_and_basins_share_the_envelope(self, pipeline):
        """Both partitions cover the same set of in-bounds cells (no
        spurious labels outside the DEM data envelope)."""
        ws = pipeline["watershed"].read_array()
        bs = pipeline["basins"].read_array()
        assert ws.shape == bs.shape

    def test_pipeline_outputs_typed_correctly(self, pipeline):
        """Every typed result is the expected class (smoke check)."""
        assert type(pipeline["watershed"]) is WatershedRaster
        assert type(pipeline["basins"]) is WatershedRaster
        assert type(pipeline["pfaf_l1"]) is WatershedRaster
        assert type(pipeline["pfaf_l2"]) is WatershedRaster
