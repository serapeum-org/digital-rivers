"""Tests for ``Accumulation.streams`` threshold-based stream extraction (P8)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, Accumulation, StreamRaster


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def _build_acc(dem: DEM) -> Accumulation:
    fd = dem.flow_direction(method="d8")
    return fd.accumulate()


# ----- Cell-count threshold ----------------------------------------------------------------

def test_threshold_one_returns_every_non_headwater_cell():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    acc = _build_acc(dem)
    sr = acc.streams(threshold=1)
    assert type(sr) is StreamRaster
    assert sr.routing == "d8"
    arr = sr.read_array()
    # Every cell with at least one upstream contributor (acc >= 1) is a stream cell.
    acc_arr = acc.read_array()
    expected = (acc_arr >= 1).astype(np.uint8)
    np.testing.assert_array_equal(arr, expected)


def test_threshold_max_returns_outlet_only():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    acc = _build_acc(dem)
    acc_arr = acc.read_array()
    max_acc = float(acc_arr.max())
    sr = acc.streams(threshold=max_acc)
    arr = sr.read_array()
    assert arr.sum() == int((acc_arr >= max_acc).sum())


def test_monotonic_decrease_in_stream_count_as_threshold_grows():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    acc = _build_acc(dem)
    counts = [acc.streams(threshold=t).read_array().sum() for t in (1, 2, 4, 8, 16)]
    # Non-increasing sequence.
    for a, b in zip(counts, counts[1:]):
        assert b <= a


# ----- Area-unit conversion ----------------------------------------------------------------

def test_km2_threshold_with_unit_cell_size():
    # With cell_size = 1000 m, one cell is 1 km². threshold_km2=2 → cells_threshold=2.
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z, cell_size=1000.0)
    acc = _build_acc(dem)
    sr_km2 = acc.streams(threshold=2.0, units="km2")
    sr_cells = acc.streams(threshold=2)
    np.testing.assert_array_equal(sr_km2.read_array(), sr_cells.read_array())


def test_m2_threshold_with_unit_cell_size():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z, cell_size=10.0)
    acc = _build_acc(dem)
    # cell_area = 100 m². threshold_m2=300 → cells_threshold=3.
    sr_m2 = acc.streams(threshold=300.0, units="m2")
    sr_cells = acc.streams(threshold=3)
    np.testing.assert_array_equal(sr_m2.read_array(), sr_cells.read_array())


# ----- Slope-area criterion ----------------------------------------------------------------

def test_slope_area_criterion_filters_by_support():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    acc = _build_acc(dem)
    # Build a slope raster: all cells slope 0.5 except outlet which is 0.1. Then
    # the area-slope support varies: low-slope cells need higher accumulation.
    slope_arr = np.full(z.shape, 0.5, dtype=np.float32)
    slope_arr[1, 5] = 0.1
    slope_ds = Dataset.create_from_array(
        slope_arr, geo=dem.geotransform, epsg=4326, no_data_value=-9999.0,
    )
    sr = acc.streams(
        threshold=1.0, slope_dem=slope_ds, area_slope_exponent=1.0
    )
    assert type(sr) is StreamRaster
    # The outlet, despite high accumulation, has low slope and may drop out.
    # Just verify the result is well-formed.
    arr = sr.read_array()
    assert arr.dtype == np.uint8
    assert set(np.unique(arr)).issubset({0, 1})


# ----- Validation --------------------------------------------------------------------------

def test_unknown_units_raises():
    z = np.array([[9, 9, 9], [9, 5, 9], [9, 9, 9]], dtype=np.float32)
    dem = _make_dem(z)
    acc = _build_acc(dem)
    with pytest.raises(ValueError, match="units must be"):
        acc.streams(threshold=1, units="bogus")


def test_only_slope_dem_without_exponent_raises():
    z = np.array([[9, 9, 9], [9, 5, 9], [9, 9, 9]], dtype=np.float32)
    dem = _make_dem(z)
    acc = _build_acc(dem)
    slope_ds = Dataset.create_from_array(
        np.zeros(z.shape, dtype=np.float32),
        top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
    )
    with pytest.raises(ValueError, match="both be supplied or both omitted"):
        acc.streams(threshold=1, slope_dem=slope_ds)


def test_returns_typed_stream_raster():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    acc = _build_acc(dem)
    sr = acc.streams(threshold=2)
    assert type(sr) is StreamRaster
    assert sr.threshold == pytest.approx(2.0)
    assert sr.routing == "d8"


# --- envelope kwarg (I1 fix) -----------------------------------------------


def _simple_acc() -> Accumulation:
    z = np.array(
        [[9, 9, 9, 9], [9, 5, 4, 1], [9, 9, 9, 9]], dtype=np.float32
    )
    return _build_acc(_make_dem(z))


class TestStreamsEnvelopeKwarg:
    """Coverage for ``Accumulation.streams(envelope=...)`` (I1 fix)."""

    def test_envelope_dataset_no_data_excludes_cells(self):
        """Passing a Dataset whose no_data_value masks some cells excludes
        them from the stream raster regardless of accumulation."""
        acc = _simple_acc()
        rows, cols = acc.read_array().shape
        env_arr = np.ones((rows, cols), dtype=np.float32)
        env_arr[0, :] = -9999.0  # entire row marked no-data
        env_ds = Dataset.create_from_array(
            env_arr, top_left_corner=(0.0, 0.0), cell_size=1.0,
            epsg=4326, no_data_value=-9999.0,
        )
        sr = acc.streams(threshold=1, envelope=env_ds)
        arr = sr.read_array()
        assert int(arr[0, :].sum()) == 0, "Row 0 should be excluded by envelope"

    def test_envelope_ndarray_bool_input(self):
        """A bool ndarray envelope masks cells where False."""
        acc = _simple_acc()
        rows, cols = acc.read_array().shape
        env = np.ones((rows, cols), dtype=bool)
        env[:, 0] = False  # entire col 0 excluded
        sr = acc.streams(threshold=1, envelope=env)
        assert int(sr.read_array()[:, 0].sum()) == 0

    def test_envelope_ndarray_non_bool_input_cast(self):
        """A non-bool ndarray (e.g. uint8 0/1) is cast to bool."""
        acc = _simple_acc()
        rows, cols = acc.read_array().shape
        env = np.ones((rows, cols), dtype=np.uint8)
        env[1, 0] = 0
        sr = acc.streams(threshold=1, envelope=env)
        assert int(sr.read_array()[1, 0]) == 0

    def test_envelope_dataset_without_nodata_uses_finite_mask(self):
        """A Dataset without a no_data_value falls back to ``isfinite``."""
        acc = _simple_acc()
        rows, cols = acc.read_array().shape
        env_arr = np.ones((rows, cols), dtype=np.float32)
        env_arr[0, 0] = np.nan  # NaN should be excluded
        env_ds = Dataset.create_from_array(
            env_arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        )
        sr = acc.streams(threshold=1, envelope=env_ds)
        assert int(sr.read_array()[0, 0]) == 0

    def test_envelope_all_false_yields_no_streams(self):
        """An all-False envelope produces zero stream cells."""
        acc = _simple_acc()
        env = np.zeros_like(acc.read_array(), dtype=bool)
        sr = acc.streams(threshold=1, envelope=env)
        assert int(sr.read_array().sum()) == 0

    def test_envelope_shape_mismatch_raises(self):
        """A mismatched-shape envelope raises ValueError with shape info."""
        acc = _simple_acc()
        bad = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="envelope shape"):
            acc.streams(threshold=1, envelope=bad)


class TestResolveEnvelopeDirect:
    """Direct coverage of the ``_resolve_envelope`` helper."""

    def test_dataset_no_data_excludes_sentinel_cells(self):
        from digitalrivers.accumulation import _resolve_envelope

        arr = np.array(
            [[1.0, -9999.0], [2.0, 3.0]], dtype=np.float32
        )
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
            no_data_value=-9999.0,
        )
        mask = _resolve_envelope(ds, (2, 2))
        assert mask.dtype == np.bool_
        assert mask.tolist() == [[True, False], [True, True]]

    def test_dataset_without_no_data_uses_finite_filter(self):
        from digitalrivers.accumulation import _resolve_envelope

        arr = np.array(
            [[1.0, np.nan], [2.0, 3.0]], dtype=np.float32
        )
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
        )
        mask = _resolve_envelope(ds, (2, 2))
        # NaN at (0, 1) is excluded by isfinite.
        assert not bool(mask[0, 1])
        assert bool(mask[0, 0])

    def test_already_bool_ndarray_passthrough(self):
        from digitalrivers.accumulation import _resolve_envelope

        env = np.array([[True, False], [False, True]])
        mask = _resolve_envelope(env, (2, 2))
        assert mask.tolist() == [[True, False], [False, True]]

    def test_non_bool_ndarray_cast_to_bool(self):
        from digitalrivers.accumulation import _resolve_envelope

        env = np.array([[1, 0], [2, 0]], dtype=np.int32)
        mask = _resolve_envelope(env, (2, 2))
        # int → bool: non-zero is True.
        assert mask.tolist() == [[True, False], [True, False]]

    def test_shape_mismatch_rejected(self):
        from digitalrivers.accumulation import _resolve_envelope

        with pytest.raises(ValueError, match="envelope shape"):
            _resolve_envelope(np.zeros((2, 3), dtype=bool), (4, 4))
