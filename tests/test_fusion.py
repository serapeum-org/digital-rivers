"""Tests for ``digitalrivers.fusion.topobathy_fusion``."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers.fusion import topobathy_fusion


def _make_ds(arr: np.ndarray) -> Dataset:
    return Dataset.create_from_array(
        arr.astype(np.float32, copy=False),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )


def test_max_blend_picks_higher_per_cell():
    """The ``"max"`` mode returns ``np.fmax(topo, bathy)`` cell-by-cell."""
    topo = _make_ds(np.array([[5.0, -1.0], [3.0, -2.0]]))
    bathy = _make_ds(np.array([[-3.0, -5.0], [-4.0, -6.0]]))
    fused = topobathy_fusion(topo, bathy, blend="max")
    expected = np.array([[5.0, -1.0], [3.0, -2.0]], dtype=np.float32)
    np.testing.assert_allclose(fused.read_array(), expected, atol=1e-3)


def test_min_blend_picks_lower_per_cell():
    """The ``"min"`` mode returns ``np.fmin(topo, bathy)`` cell-by-cell."""
    topo = _make_ds(np.array([[5.0, -1.0], [3.0, -2.0]]))
    bathy = _make_ds(np.array([[-3.0, -5.0], [-4.0, -6.0]]))
    fused = topobathy_fusion(topo, bathy, blend="min")
    expected = np.array([[-3.0, -5.0], [-4.0, -6.0]], dtype=np.float32)
    np.testing.assert_allclose(fused.read_array(), expected, atol=1e-3)


def test_topo_above_branch():
    """``topo_above`` pulls topo above the shoreline, bathy below."""
    topo = _make_ds(np.array([[5.0, -1.0]]))
    bathy = _make_ds(np.array([[-3.0, -5.0]]))
    fused = topobathy_fusion(
        topo, bathy, blend="topo_above", shoreline_elev=0.0,
    )
    # Cell 0: topo=5 >= 0 → topo wins (5). Cell 1: topo=-1 < 0 → bathy (-5).
    np.testing.assert_allclose(fused.read_array()[0], [5.0, -5.0], atol=1e-3)


def test_invalid_blend_rejected():
    z = _make_ds(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="blend must be"):
        topobathy_fusion(z, z, blend="bogus")


def test_min_blend_nan_picks_other_operand():
    """When one operand is NaN, ``np.fmin`` returns the non-NaN value."""
    topo = _make_ds(np.array([[np.nan, 5.0]]))
    bathy = _make_ds(np.array([[-3.0, np.nan]]))
    fused = topobathy_fusion(topo, bathy, blend="min")
    arr = fused.read_array()
    assert float(arr[0, 0]) == pytest.approx(-3.0)
    assert float(arr[0, 1]) == pytest.approx(5.0)
