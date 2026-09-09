"""Tests for `digitalrivers.core.numba`, the JIT availability shim.

Two contracts matter here, and both are the kind that break silently.

The first is that `import digitalrivers` must not pull Numba into the process. Numba is a
heavy import and a hard dependency of nothing at the top level; every kernel that needs it
imports this module from inside the function that runs. A stray module-level import
anywhere in the package would undo that without any test noticing, so it is checked in a
fresh interpreter rather than in-process, where an earlier test may already have imported
Numba for its own reasons.

The second is that `DIGITALRIVERS_DISABLE_NUMBA=1` genuinely disables the JIT. The flag is
read once at import, so flipping the variable in a live process does nothing — which is
exactly how a test can appear to exercise the fallback while running the fast path. These
tests re-import in a subprocess so the flag is read fresh.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from digitalrivers.core import numba as core_numba


def _run(code: str, env_flag: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter, optionally with the disable flag set.

    A subprocess is the only honest way to test an import-time decision: within this
    process the module is already imported and its flag already read.

    Args:
        code: Python source to execute.
        env_flag: Value for `DIGITALRIVERS_DISABLE_NUMBA`, or `None` to leave it unset.

    Returns:
        subprocess.CompletedProcess: The completed run, with output captured.
    """
    import os

    env = dict(os.environ)
    env.pop("DIGITALRIVERS_DISABLE_NUMBA", None)
    if env_flag is not None:
        env["DIGITALRIVERS_DISABLE_NUMBA"] = env_flag
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=180,
    )


class TestIsNumbaEnabled:
    """Tests for is_numba_enabled."""

    def test_returns_a_bool(self):
        """The predicate is a plain bool, safe to use in an `if`."""
        result = core_numba.is_numba_enabled()
        assert isinstance(result, bool), f"Expected bool, got {type(result).__name__}"

    def test_agrees_with_the_module_flag(self):
        """The predicate reports the flag rather than re-deriving it."""
        assert (
            core_numba.is_numba_enabled() is core_numba._USE_NUMBA
        ), "Predicate and flag disagree"

    def test_reports_false_in_a_fresh_interpreter_with_the_flag_set(self):
        """`DIGITALRIVERS_DISABLE_NUMBA=1` turns the JIT off at import.

        Test scenario:
            Run in a subprocess, because the flag is read once at import: setting it in
            this process would change nothing and the test would pass vacuously.
        """
        proc = _run(
            "from digitalrivers.core.numba import is_numba_enabled;"
            "print(is_numba_enabled())",
            env_flag="1",
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        assert (
            proc.stdout.strip() == "False"
        ), f"Expected False with the flag set, got {proc.stdout.strip()!r}"

    @pytest.mark.parametrize("value", ["0", "", "true", "yes", "2"])
    def test_only_the_exact_string_one_disables_the_jit(self, value):
        """Anything other than `"1"` leaves the JIT on.

        Args:
            value: The environment value under test.

        Test scenario:
            The check is `!= "1"`, so `"true"` and `"yes"` do *not* disable Numba. That
            is surprising enough to be worth pinning: someone setting
            `DIGITALRIVERS_DISABLE_NUMBA=true` gets the fast path.
        """
        proc = _run(
            "from digitalrivers.core.numba import is_numba_enabled;"
            "print(is_numba_enabled())",
            env_flag=value,
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        assert (
            proc.stdout.strip() == "True"
        ), f"{value!r} should not disable the JIT, got {proc.stdout.strip()!r}"


class TestNjitFallback:
    """Tests for the no-op njit decorator used when Numba is off."""

    def test_decorated_function_still_runs_with_the_jit_disabled(self):
        """The fallback decorator returns a callable that computes the same answer."""
        proc = _run(
            "from digitalrivers.core.numba import njit\n"
            "@njit(cache=True)\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "print(add(2, 3))",
            env_flag="1",
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        assert proc.stdout.strip() == "5", f"Expected 5, got {proc.stdout.strip()!r}"

    def test_bare_decorator_form_works_with_the_jit_disabled(self):
        """`@njit` without parentheses is handled too, not just `@njit(...)`.

        Test scenario:
            The fallback branches on whether its first argument is callable. Both call
            shapes appear in the kernels, so both are exercised.
        """
        proc = _run(
            "from digitalrivers.core.numba import njit\n"
            "@njit\n"
            "def double(x):\n"
            "    return x * 2\n"
            "print(double(21))",
            env_flag="1",
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        assert proc.stdout.strip() == "42", f"Expected 42, got {proc.stdout.strip()!r}"

    def test_prange_falls_back_to_range(self):
        """The fallback `prange` iterates like `range`, so parallel loops still run."""
        proc = _run(
            "from digitalrivers.core.numba import prange;print(list(prange(3)))",
            env_flag="1",
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        assert (
            proc.stdout.strip() == "[0, 1, 2]"
        ), f"Expected [0, 1, 2], got {proc.stdout.strip()!r}"

    def test_prange_accepts_start_and_stop(self):
        """`prange(a, b)` forwards both arguments, as `range` would."""
        proc = _run(
            "from digitalrivers.core.numba import prange;print(list(prange(2, 5)))",
            env_flag="1",
        )
        assert (
            proc.stdout.strip() == "[2, 3, 4]"
        ), f"Expected [2, 3, 4], got {proc.stdout.strip()!r}"


class TestLazyImportContract:
    """`import digitalrivers` must not drag Numba into the process."""

    def test_importing_the_package_does_not_import_numba(self):
        """Checked in a fresh interpreter, where nothing else has imported Numba.

        Test scenario:
            In-process this would be meaningless — another test may already have run a
            JIT kernel and put `numba` in `sys.modules`. The subprocess starts clean, so
            a positive result means the package really did not reach for it.
        """
        proc = _run("import sys, digitalrivers;print('numba' in sys.modules)")
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        assert proc.stdout.strip() == "False", (
            "importing digitalrivers pulled numba into the process; some module gained "
            "a top-level import of a kernel module"
        )

    def test_importing_core_directions_does_not_import_numba(self):
        """The shared direction tables are usable without the JIT stack."""
        proc = _run(
            "import sys, digitalrivers.core.directions;print('numba' in sys.modules)"
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        assert (
            proc.stdout.strip() == "False"
        ), "digitalrivers.core.directions pulled numba in"

    def test_importing_the_shim_itself_does_import_numba(self):
        """The contrast that gives the two tests above their meaning.

        Test scenario:
            `core.numba` is *supposed* to import Numba — it is the module that decides
            whether the JIT is live. If this ever reported False with the flag unset, the
            two assertions above would be passing for the wrong reason.
        """
        proc = _run(
            "import sys, digitalrivers.core.numba;print('numba' in sys.modules)"
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        assert (
            proc.stdout.strip() == "True"
        ), "core.numba did not import numba, so the lazy-import tests prove nothing"


class TestModuleSurface:
    """The shim exports exactly what the kernels import from it."""

    def test_all_names_are_importable(self):
        """`__all__` is accurate; every name it lists actually exists."""
        for name in core_numba.__all__:
            assert hasattr(core_numba, name), f"__all__ lists a missing name: {name}"

    def test_exports_the_three_names_the_kernels_use(self):
        """`njit`, `prange` and `is_numba_enabled` are the whole contract."""
        assert sorted(core_numba.__all__) == [
            "is_numba_enabled",
            "njit",
            "prange",
        ], f"Shim surface changed: {sorted(core_numba.__all__)}"


class TestFallbackBodiesInProcess:
    """Exercises the fallback decorator bodies where coverage can see them.

    `TestNjitFallback` proves the fallbacks behave correctly, but it does so in
    subprocesses, which coverage does not trace. These two re-import the shim in-process
    under the disable flag and restore it afterwards, so the no-op bodies are measured.
    """

    @pytest.fixture
    def disabled_shim(self, monkeypatch):
        """Import a fresh `core.numba` with the JIT disabled, then put the real one back.

        Yields:
            module: The shim as imported with `DIGITALRIVERS_DISABLE_NUMBA=1`.
        """
        import importlib

        original = sys.modules.get("digitalrivers.core.numba")
        monkeypatch.setenv("DIGITALRIVERS_DISABLE_NUMBA", "1")
        sys.modules.pop("digitalrivers.core.numba", None)
        try:
            yield importlib.import_module("digitalrivers.core.numba")
        finally:
            sys.modules.pop("digitalrivers.core.numba", None)
            if original is not None:
                sys.modules["digitalrivers.core.numba"] = original

    def test_fallback_njit_is_transparent_in_both_call_shapes(self, disabled_shim):
        """Both `@njit` and `@njit(...)` return a function that computes the same answer.

        Args:
            disabled_shim: The JIT-disabled shim fixture.
        """
        assert (
            disabled_shim.is_numba_enabled() is False
        ), "Fixture did not disable the JIT"

        @disabled_shim.njit
        def bare(x):
            return x + 1

        @disabled_shim.njit(cache=True, nogil=True)
        def parenthesised(x):
            return x + 2

        assert bare(1) == 2, f"Bare form returned {bare(1)}"
        assert parenthesised(1) == 3, f"Parenthesised form returned {parenthesised(1)}"

    def test_fallback_prange_is_range(self, disabled_shim):
        """The fallback `prange` forwards to `range` for every argument arity.

        Args:
            disabled_shim: The JIT-disabled shim fixture.
        """
        assert list(disabled_shim.prange(3)) == [0, 1, 2], "One-argument form wrong"
        assert list(disabled_shim.prange(1, 4)) == [1, 2, 3], "Two-argument form wrong"
        assert list(disabled_shim.prange(0, 6, 2)) == [
            0,
            2,
            4,
        ], "Three-argument form wrong"
