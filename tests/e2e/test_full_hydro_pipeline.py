"""Tests for `DEM.full_hydro_pipeline` (W-20)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import (
    DEM,
    Accumulation,
    FlowDirection,
    StreamRaster,
)


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


class TestFullHydroPipeline:
    """Tests for `DEM.full_hydro_pipeline`."""

    def test_default_returns_three_typed_results(self):
        """Test the default call returns DEM, FlowDirection, Accumulation.

        Test scenario:
            Without `stream_threshold_cells`, the result dict carries exactly
            three keys, each pointing at the matching typed class.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        out = dem.full_hydro_pipeline()
        assert set(out.keys()) == {"filled_dem", "flow_direction", "accumulation"}
        assert isinstance(out["filled_dem"], DEM)
        assert isinstance(out["flow_direction"], FlowDirection)
        assert isinstance(out["accumulation"], Accumulation)

    def test_with_threshold_includes_streams(self):
        """Test passing a threshold also produces a StreamRaster.

        Test scenario:
            With `stream_threshold_cells=1`, the result dict carries a
            `"streams"` key pointing at a StreamRaster.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        out = dem.full_hydro_pipeline(stream_threshold_cells=1)
        assert "streams" in out
        assert isinstance(out["streams"], StreamRaster)

    def test_results_align_with_step_by_step_call(self):
        """Test pipeline outputs match a manually-chained pipeline cell-by-cell.

        Test scenario:
            Calling fill → flow_direction → accumulate manually with the same
            arguments produces identical rasters.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        bundle = dem.full_hydro_pipeline()
        # Manual chain.
        filled = dem.fill_depressions(method="priority_flood")
        fdir = filled.flow_direction(method="d8")
        acc = fdir.accumulate()
        np.testing.assert_array_equal(
            bundle["accumulation"].read_array(), acc.read_array()
        )
        np.testing.assert_array_equal(
            bundle["flow_direction"].read_array(), fdir.read_array()
        )

    def test_alternative_methods_forwarded(self):
        """Test fill_method / flow_method kwargs are honoured.

        Test scenario:
            Passing `fill_method="wang_liu"` and `flow_method="rho8"` returns
            objects tagged with the corresponding routing scheme.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        out = dem.full_hydro_pipeline(
            fill_method="wang_liu", flow_method="rho8",
        )
        assert out["flow_direction"].routing == "rho8"
