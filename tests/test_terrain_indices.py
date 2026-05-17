"""Tests for `DEM.tpi` / `.deviation_from_mean` / `.elev_std` / `.ruggedness`
(W-21 / W-22 / W-23 / W-24)."""
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


class TestTPI:
    """Tests for `DEM.tpi`."""

    def test_flat_terrain_zero_everywhere(self):
        """Test TPI on a constant-elevation DEM is zero everywhere.

        Test scenario:
            A flat 5×5 DEM at z=10 has focal_mean = 10 everywhere, so
            TPI = z - focal_mean = 0 at every cell.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        tpi = dem.tpi(window=3).read_array()
        assert np.allclose(tpi, 0.0)

    def test_ridge_cell_positive_tpi(self):
        """Test a single elevated cell on flat terrain produces positive TPI.

        Test scenario:
            A 5×5 flat DEM at z=0 with a peak at (2, 2)=9 produces TPI > 0
            at that cell (the cell sits above its focal mean).
        """
        z = np.zeros((5, 5), dtype=np.float32)
        z[2, 2] = 9.0
        dem = _make_dem(z)
        tpi = dem.tpi(window=3).read_array()
        assert tpi[2, 2] > 0

    def test_valley_cell_negative_tpi(self):
        """Test a single low cell on flat terrain produces negative TPI.

        Test scenario:
            A 5×5 flat DEM at z=10 with a pit at (2, 2)=0 produces TPI < 0
            at that cell.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        z[2, 2] = 0.0
        dem = _make_dem(z)
        tpi = dem.tpi(window=3).read_array()
        assert tpi[2, 2] < 0

    def test_invalid_window_rejected(self):
        """Test window < 1 raises ValueError.

        Test scenario:
            A zero or negative window is not a meaningful focal radius.
        """
        z = np.zeros((3, 3), dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="window"):
            dem.tpi(window=0)


class TestDeviationFromMean:
    """Tests for `DEM.deviation_from_mean`."""

    def test_flat_terrain_zero_everywhere(self):
        """Test deviation_from_mean on a flat DEM is zero everywhere.

        Test scenario:
            Constant elevation → focal_mean equals z, focal_sd is zero
            (treated as 1 to avoid /0) — result is zero everywhere.
        """
        z = np.full((5, 5), 10.0, dtype=np.float32)
        dem = _make_dem(z)
        dev = dem.deviation_from_mean(window=3).read_array()
        assert np.allclose(dev, 0.0)

    def test_peak_cell_positive_normalised_deviation(self):
        """Test a peak cell yields a positive normalised deviation.

        Test scenario:
            A flat DEM with a single elevated cell has dev > 0 at the peak.
        """
        z = np.zeros((5, 5), dtype=np.float32)
        z[2, 2] = 10.0
        dem = _make_dem(z)
        dev = dem.deviation_from_mean(window=3).read_array()
        assert dev[2, 2] > 0
