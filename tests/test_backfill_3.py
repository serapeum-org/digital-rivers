"""Third backfill pass: P25 ANUDEM-lite Laplacian relaxation."""
from __future__ import annotations

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


def test_anudem_fills_nan_holes():
    """Unknown cells get a finite value after Laplacian relaxation."""
    z = np.array(
        [
            [10, 9, 8, 7, 6],
            [10, np.nan, np.nan, np.nan, 6],
            [10, np.nan, np.nan, np.nan, 6],
            [10, np.nan, np.nan, np.nan, 6],
            [10, 9, 8, 7, 6],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    filled = dem.anudem_interpolate(max_iter=500)
    vals = filled.values
    # All cells are now finite.
    assert np.all(np.isfinite(vals))
    # Fixed cells are preserved.
    assert vals[0, 0] == pytest.approx(10.0)
    assert vals[0, 4] == pytest.approx(6.0)
    # Interior cells fall between the boundary extremes.
    interior = vals[1:4, 1:4]
    assert (interior >= 6.0 - 0.01).all()
    assert (interior <= 10.0 + 0.01).all()


def test_anudem_with_explicit_mask_pins_extra_cells():
    """Cells flagged in ``mask=`` stay at their input value."""
    z = np.full((5, 5), np.nan, dtype=np.float32)
    z[0, 0] = 0.0
    z[4, 4] = 100.0
    # Pin an interior anchor too.
    mask = np.zeros((5, 5), dtype=bool)
    z[2, 2] = 50.0
    mask[2, 2] = True
    dem = _make_dem(z)
    filled = dem.anudem_interpolate(mask=mask, max_iter=500)
    assert filled.values[0, 0] == pytest.approx(0.0)
    assert filled.values[4, 4] == pytest.approx(100.0)
    assert filled.values[2, 2] == pytest.approx(50.0)


def test_anudem_no_anchor_raises():
    """An all-NaN DEM raises ValueError — there is nothing to anchor on."""
    z = np.full((4, 4), np.nan, dtype=np.float32)
    dem = _make_dem(z)
    with pytest.raises(ValueError, match="finite anchor"):
        dem.anudem_interpolate()


def test_anudem_returns_dem():
    z = np.array([[10.0, np.nan], [np.nan, 5.0]], dtype=np.float32)
    dem = _make_dem(z)
    out = dem.anudem_interpolate()
    assert isinstance(out, DEM)


def test_anudem_inplace_returns_none():
    z = np.array([[10.0, np.nan], [np.nan, 5.0]], dtype=np.float32)
    dem = _make_dem(z)
    result = dem.anudem_interpolate(inplace=True)
    assert result is None
