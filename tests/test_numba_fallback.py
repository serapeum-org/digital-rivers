"""Tests for the Numba acceleration layer and its pure-Python fallback (P7).

Verifies:

* The fast-path produces bit-for-bit identical output to the pure-Python branch
  on the affected algorithms (priority-flood fill, D8 accumulation).
* The ``DIGITALRIVERS_DISABLE_NUMBA=1`` env var cleanly disables the JIT path
  (requires re-importing the kernel module after setting the env var).
* The kernel module exposes a public ``is_numba_enabled`` predicate.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

import digitalrivers._numba as _numba
from digitalrivers._pitremoval import _priority_flood, fill_depressions
from digitalrivers._accumulation import _receivers_d8, kahn_accumulate


# ----- Toggle / availability ----------------------------------------------------------------

def test_is_numba_enabled_predicate_exists():
    assert isinstance(_numba.is_numba_enabled(), bool)


def test_env_var_disables_numba_on_reimport(monkeypatch):
    """Setting DIGITALRIVERS_DISABLE_NUMBA=1 and re-importing the module must
    return ``is_numba_enabled() is False``. This is how CI exercises the fallback
    path without needing a Numba-free environment."""
    monkeypatch.setenv("DIGITALRIVERS_DISABLE_NUMBA", "1")
    # Re-import the module under the new env.
    sys.modules.pop("digitalrivers._numba", None)
    reloaded = importlib.import_module("digitalrivers._numba")
    try:
        assert reloaded.is_numba_enabled() is False
    finally:
        # Restore the original module so other tests use the JIT path.
        sys.modules.pop("digitalrivers._numba", None)
        importlib.import_module("digitalrivers._numba")


# ----- Priority-flood parity -----------------------------------------------------------------

def _single_pit_5x5() -> np.ndarray:
    return np.array(
        [
            [5, 5, 5, 5, 5],
            [5, 4, 4, 4, 5],
            [5, 4, 1, 4, 5],
            [5, 4, 4, 4, 5],
            [5, 5, 5, 5, 5],
        ],
        dtype=np.float64,
    )


def test_priority_flood_numba_matches_pure_python_single_pit():
    z = _single_pit_5x5()
    nodata = np.zeros(z.shape, dtype=bool)
    numba_out = fill_depressions(z.copy(), method="priority_flood", epsilon=0.0)
    py_out = _priority_flood(z.copy(), nodata, epsilon=0.0, use_pit_queue=True)
    np.testing.assert_allclose(numba_out, py_out, rtol=0, atol=0)


def test_priority_flood_numba_with_epsilon_matches():
    z = _single_pit_5x5()
    nodata = np.zeros(z.shape, dtype=bool)
    eps = 0.01
    numba_out = fill_depressions(z.copy(), method="priority_flood", epsilon=eps)
    py_out = _priority_flood(z.copy(), nodata, epsilon=eps, use_pit_queue=True)
    np.testing.assert_allclose(numba_out, py_out, rtol=0, atol=1e-12)


def test_priority_flood_numba_handles_nodata():
    z = _single_pit_5x5()
    z[0, 0] = np.nan
    numba_out = fill_depressions(z.copy(), method="priority_flood", epsilon=0.0)
    nodata = np.isnan(z)
    py_out = _priority_flood(z.copy(), nodata, epsilon=0.0, use_pit_queue=True)
    # NaN cells stay NaN in both, others must match.
    nan_numba = np.isnan(numba_out)
    nan_py = np.isnan(py_out)
    assert np.array_equal(nan_numba, nan_py)
    np.testing.assert_allclose(numba_out[~nan_numba], py_out[~nan_py])


# ----- D8 accumulation parity ---------------------------------------------------------------

def test_kahn_accumulate_d8_numba_matches_pure_python():
    # Hand-crafted 3-row strip; the central row chains east into a sink.
    fdir = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [6, 6, 6, 6, 6, -9999],
            [4, 4, 4, 4, 4, 4],
        ],
        dtype=np.int32,
    )
    weights = np.ones(fdir.shape, dtype=np.float64)
    valid = np.ones(fdir.shape, dtype=bool)

    numba_out = _numba.kahn_accumulate_d8_numba(
        fdir, weights, _numba._DIR_DR_I32, _numba._DIR_DC_I32
    )
    receivers, proportions = _receivers_d8(fdir, valid)
    py_out = kahn_accumulate(receivers, proportions, weights, valid)
    np.testing.assert_allclose(numba_out, py_out)


def test_d8_kernel_handles_sinks():
    """A cell whose direction code is the no-data sentinel must still receive
    accumulation from upstream — the sink-routing fix from P6 must hold in the
    JIT path too."""
    fdir = np.array(
        [
            [6, 6, 6, -9999],
        ],
        dtype=np.int32,
    )
    weights = np.ones(fdir.shape, dtype=np.float64)
    out = _numba.kahn_accumulate_d8_numba(
        fdir, weights, _numba._DIR_DR_I32, _numba._DIR_DC_I32
    )
    # The sink at (0, 3) collects the three upstream cells.
    assert out[0, 3] == pytest.approx(3.0)


# ----- D8 flow-direction kernel -------------------------------------------------------------

def test_d8_flow_direction_numba_matches_steepest_descent():
    # Simple 3x3 hilltop: centre is highest, all neighbours slope away.
    z = np.array(
        [
            [3.0, 3.0, 3.0],
            [3.0, 5.0, 3.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )
    out = _numba.d8_flow_direction_numba(
        z, 1.0, np.int32(-9999), _numba._DIR_DR_I32, _numba._DIR_DC_I32
    )
    # Centre cell has 8 equally downhill neighbours; the kernel breaks ties by
    # the first direction it scans (index 0 = S) with strictly-greater slope.
    assert out[1, 1] in {0, 1, 2, 3, 4, 5, 6, 7}
    # Corners are at z=3 and the centre at z=5; corners have NO downhill
    # neighbour, so they are sinks under the P5 strict-D8 rule.
    assert out[0, 0] == -9999
    assert out[2, 2] == -9999
