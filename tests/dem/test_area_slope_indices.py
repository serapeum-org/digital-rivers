"""Tests for `DEM.twi` / `DEM.spi` / `DEM.sti` (W-12 / W-13 / W-14)."""
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


def _build_pipeline(z: np.ndarray):
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    return dem, fd, acc


class TestTWI:
    """Tests for `DEM.twi`."""

    def test_returns_float32_dataset(self):
        """Test TWI output is a float32 Dataset.

        Test scenario:
            On a simple east-flowing chain DEM, TWI returns a float32 raster
            of the same shape as the DEM.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc = _build_pipeline(z)
        out = dem.twi(acc)
        arr = out.read_array()
        assert arr.dtype == np.float32
        assert arr.shape == z.shape

    def test_higher_acc_lower_slope_gives_higher_twi(self):
        """Test TWI grows with accumulation and falls with slope.

        Test scenario:
            On a single chain, the downstream cells with higher accumulation
            should carry larger TWI than upstream cells (at equal slope).
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc = _build_pipeline(z)
        out = dem.twi(acc)
        arr = out.read_array()
        # TWI at the downstream end exceeds TWI at the upstream end on the
        # chain at row 1.
        chain_twi = arr[1, 1:6]
        # Most-downstream TWI must be >= most-upstream TWI on the chain.
        finite = chain_twi[np.isfinite(chain_twi) & (chain_twi != -9999.0)]
        if finite.size >= 2:
            assert finite[-1] >= finite[0]


class TestSPI:
    """Tests for `DEM.spi`."""

    def test_returns_float32_dataset(self):
        """Test SPI output is float32 and same shape as the DEM.

        Test scenario:
            On a simple east-flowing chain DEM, SPI returns a float32 raster
            of the same shape as the DEM.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc = _build_pipeline(z)
        out = dem.spi(acc)
        arr = out.read_array()
        assert arr.dtype == np.float32
        assert arr.shape == z.shape

    def test_spi_non_negative_on_data(self):
        """Test SPI is non-negative wherever it is defined.

        Test scenario:
            SCA * tan(slope) is non-negative for any non-negative SCA and
            positive slope, so finite SPI values must be ≥ 0.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc = _build_pipeline(z)
        out = dem.spi(acc)
        arr = out.read_array()
        finite = arr[(arr != -9999.0) & np.isfinite(arr)]
        assert (finite >= 0).all()


class TestSTI:
    """Tests for `DEM.sti`."""

    def test_returns_float32_dataset(self):
        """Test STI output is float32 and same shape as the DEM.

        Test scenario:
            On a simple east-flowing chain DEM, STI returns a float32 raster
            of the same shape as the DEM.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc = _build_pipeline(z)
        out = dem.sti(acc)
        arr = out.read_array()
        assert arr.dtype == np.float32
        assert arr.shape == z.shape

    def test_sti_non_negative_on_data(self):
        """Test STI is non-negative wherever defined.

        Test scenario:
            (SCA/22.13)^0.6 * (sin(slope)/0.0896)^1.3 is non-negative for
            any non-negative SCA and slope, so finite STI values must be ≥ 0.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc = _build_pipeline(z)
        out = dem.sti(acc)
        arr = out.read_array()
        finite = arr[(arr != -9999.0) & np.isfinite(arr)]
        assert (finite >= 0).all()


class TestSlopeShapeMismatch:
    """Cross-cutting shape-validation tests."""

    def test_explicit_slope_shape_mismatch_raises(self):
        """Test passing a wrong-shape slope raster raises ValueError.

        Test scenario:
            slope_deg with a shape that doesn't match the DEM/accumulation
            must be rejected.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 4, 3, 2, 1],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem, fd, acc = _build_pipeline(z)
        bad_slope_arr = np.zeros((2, 2), dtype=np.float32)
        bad_slope = Dataset.create_from_array(
            bad_slope_arr, top_left_corner=(0.0, 0.0), cell_size=1.0,
            epsg=4326, no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="shape"):
            dem.twi(acc, slope_deg=bad_slope)
