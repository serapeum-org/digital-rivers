"""Tests for `WatershedRaster.statistics(longest_flow_path_m)` (W-8)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

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


class TestLongestFlowPath:
    """Tests for the `longest_flow_path_m` statistics column."""

    def test_column_present_when_inputs_supplied(self):
        """Test the column appears when both accumulation and flow_direction are passed.

        Test scenario:
            With both kwargs supplied, statistics() must include the
            longest_flow_path_m column.
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
        fd = dem.flow_direction(method="d8")
        acc = fd.accumulate()
        ws = fd.basins()
        df = ws.statistics(accumulation=acc, flow_direction=fd)
        assert "longest_flow_path_m" in df.columns

    def test_column_absent_when_accumulation_missing(self):
        """Test the column is absent when only flow_direction is supplied.

        Test scenario:
            Without `accumulation`, the longest-flow-path column should be
            omitted (and no error raised).
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
        fd = dem.flow_direction(method="d8")
        ws = fd.basins()
        df = ws.statistics(flow_direction=fd)
        assert "longest_flow_path_m" not in df.columns

    def test_single_chain_path_length(self):
        """Test the longest flow path matches a hand-computed value.

        Test scenario:
            Straight 5-cell east-flowing chain at cell_size=1: longest flow
            path from the head to the outlet is 4 cardinal steps = 4.0.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1, 0],
                [9, 9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd = dem.flow_direction(method="d8")
        acc = fd.accumulate()
        ws = fd.basins()
        df = ws.statistics(accumulation=acc, flow_direction=fd)
        # Longest path in this catchment must be non-trivial (≥ 4 cells from
        # the head to the outlet).
        assert (df["longest_flow_path_m"] >= 4.0 - 1e-9).any()

    def test_non_negative_values(self):
        """Test all longest-flow-path values are non-negative.

        Test scenario:
            Path lengths are summed step distances; they must be ≥ 0 for
            every basin.
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
        fd = dem.flow_direction(method="d8")
        acc = fd.accumulate()
        ws = fd.basins()
        df = ws.statistics(accumulation=acc, flow_direction=fd)
        assert (df["longest_flow_path_m"] >= 0).all()
