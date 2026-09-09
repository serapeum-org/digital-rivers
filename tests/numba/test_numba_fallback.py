"""Tests for the Numba acceleration layer and its pure-Python fallback (P7).

Verifies:

* The fast-path produces bit-for-bit identical output to the pure-Python branch
  on the affected algorithms (priority-flood fill, D8 accumulation).
* The `DIGITALRIVERS_DISABLE_NUMBA=1` env var cleanly disables the JIT path
  (requires re-importing `core.numba`, which is what reads the flag).
* `digitalrivers.core.numba` exposes a public `is_numba_enabled` predicate.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

import digitalrivers.core

from digitalrivers.core.directions import DIR_DC_I32, DIR_DR_I32
from digitalrivers.core.numba import is_numba_enabled
from digitalrivers.dem._kernels.pitremoval import _priority_flood, fill_depressions
from digitalrivers.flow._kernels.accumulation import _receivers_d8, kahn_accumulate
from digitalrivers.flow._kernels.numba import kahn_accumulate_d8_numba
from tests.helpers import d8_flow_direction_numba


# ----- Toggle / availability ----------------------------------------------------------------


def test_is_numba_enabled_predicate_exists():
    assert isinstance(is_numba_enabled(), bool)


def test_env_var_disables_numba_on_reimport(monkeypatch):
    """Setting DIGITALRIVERS_DISABLE_NUMBA=1 and re-importing the module must
    return `is_numba_enabled() is False`. This is how CI exercises the fallback
    path without needing a Numba-free environment.

    The teardown puts back the *original module object* rather than re-importing.
    Re-importing here is a trap: monkeypatch undoes `setenv` in its fixture finalizer,
    which runs after this function's `finally`, so a re-import at this point still sees
    the flag set and installs a JIT-disabled module for the rest of the session. Every
    later test then silently takes the pure-Python branch.
    """
    original = sys.modules.get("digitalrivers.core.numba")
    monkeypatch.setenv("DIGITALRIVERS_DISABLE_NUMBA", "1")
    sys.modules.pop("digitalrivers.core.numba", None)
    try:
        reloaded = importlib.import_module("digitalrivers.core.numba")
        assert reloaded.is_numba_enabled() is False
    finally:
        sys.modules.pop("digitalrivers.core.numba", None)
        if original is not None:
            sys.modules["digitalrivers.core.numba"] = original
            digitalrivers.core.numba = original


def test_the_disable_test_left_the_jit_on():
    """The JIT is live again after the test above, resolved at call time.

    pytest runs tests in definition order within a file, so this sits immediately after
    the only test that turns the JIT off. It resolves the module through `sys.modules`
    rather than importing the name at module scope, because a module-scope import binds
    at collection time — before the leak could happen — and would pass regardless.
    """
    mod = sys.modules["digitalrivers.core.numba"]
    assert mod.is_numba_enabled() is True, (
        "DIGITALRIVERS_DISABLE_NUMBA leaked out of the test above; every later test in "
        "the session would silently run the pure-Python branch"
    )
    assert (
        digitalrivers.core.numba is mod
    ), "digitalrivers.core.numba still points at the disabled module object"


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
    # The numba Barnes step-count kernel must match the pure-Python one; select it explicitly with
    # eps_fill="barnes" (the default eps_fill="exact" uses the order-independent exit-distance ramp instead).
    z = _single_pit_5x5()
    nodata = np.zeros(z.shape, dtype=bool)
    eps = 0.01
    numba_out = fill_depressions(
        z.copy(), method="priority_flood", epsilon=eps, eps_fill="barnes"
    )
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

    numba_out = kahn_accumulate_d8_numba(fdir, weights, DIR_DR_I32, DIR_DC_I32)
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
    out = kahn_accumulate_d8_numba(fdir, weights, DIR_DR_I32, DIR_DC_I32)
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
    out = d8_flow_direction_numba(z, 1.0, np.int32(-9999), DIR_DR_I32, DIR_DC_I32)
    # Centre cell has 8 equally downhill neighbours; the kernel breaks ties by
    # the first direction it scans (index 0 = S) with strictly-greater slope.
    assert out[1, 1] in {0, 1, 2, 3, 4, 5, 6, 7}
    # Corners are at z=3 and the centre at z=5; corners have NO downhill
    # neighbour, so they are sinks under the P5 strict-D8 rule.
    assert out[0, 0] == -9999
    assert out[2, 2] == -9999
