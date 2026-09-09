"""Pins the vendored-osgeo bootstrap the whole test suite rests on.

The `osgeo` bindings are not installed at the top level any more: they ship inside the
`pyramids` wheel under `_vendor/osgeo/`, and `pyramids/__init__.py` is what puts that
directory on `sys.path`. Most test modules here do `from osgeo import gdal` before they
import `digitalrivers`, so the `import pyramids` on the first line of `tests/conftest.py`
is what makes their imports resolve.

These tests assert the mechanism, in fresh subprocesses: osgeo is importable after
pyramids, it resolves inside the wheel, and it is not importable without pyramids. That
is the part worth pinning, because it can break silently — a free-threaded or
ABI-mismatched solve leaves `activate_vendored_osgeo` warning and returning `False`
rather than raising, and there is no conda GDAL underneath to absorb it any more.

What these tests deliberately do **not** try to do is police the import order inside
`tests/conftest.py`. A test living under `tests/` is collected through that conftest, so
if the pyramids import is deleted or sorted below the osgeo import, conftest itself fails
to import and nothing here runs — the failure surfaces as a collection error naming
`from osgeo import gdal` in `conftest.py`, which no assertion in this file can improve on.
An earlier version of this file asserted that order anyway and could never have fired.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

_SITE_DIRS = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}

# Whether a top-level `osgeo` package (conda-forge's gdal bindings, say) is installed
# alongside pyramids. When one is, `osgeo` resolves without the vendor bootstrap and the
# two wheel-shape assertions below do not apply to that installation. These directories
# are the whole of what the subprocesses below search outside the stdlib, which is what
# makes this probe an answer about them rather than about the machine.
TOP_LEVEL_OSGEO_INSTALLED = any((Path(d) / "osgeo").is_dir() for d in _SITE_DIRS)

# Subprocess timeout. Generous: importing gdal pulls in a large native stack, and a cold
# filesystem on CI is slower than anything seen locally.
_TIMEOUT = 120


def _clean_env() -> dict[str, str]:
    """Return the parent environment minus the variables that could inject another osgeo.

    `PYTHONPATH` and `PYTHONHOME` are dropped for the same reason `_run` passes `-P` and
    `-s`: without them, a tree that merely happens to contain an `osgeo/` directory makes
    the negative test below fail for a reason that has nothing to do with this package.
    Everything else is inherited, because the environment's own activation variables are
    what make GDAL work at all.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def _run(code: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter whose import roots are the environment's site dirs alone."""
    # `-P` keeps the working directory off the child's sys.path — with `-c` Python puts it
    # first — and `-s` keeps the per-user site directory off it. With PYTHONPATH and
    # PYTHONHOME stripped too, the child searches exactly the directories
    # TOP_LEVEL_OSGEO_INSTALLED probes, so the skip guard and the child agree on what
    # "installed" means. Neither flag touches the activation variables GDAL needs.
    return subprocess.run(
        [sys.executable, "-P", "-s", "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=_clean_env(),
        timeout=_TIMEOUT,
    )


@pytest.fixture(scope="module")
def bare_osgeo_import() -> subprocess.CompletedProcess[str]:
    """Result of importing `osgeo` in a fresh interpreter with no prior `import pyramids`."""
    return _run("import osgeo; print(osgeo.__file__)")


@pytest.fixture(scope="module")
def bootstrapped_osgeo() -> dict[str, str]:
    """Where `osgeo` resolves, and which GDAL it is, once `pyramids` has run its vendor bootstrap."""
    code = (
        "import json, pyramids, osgeo\n"
        "from osgeo import gdal\n"
        "print(json.dumps({'osgeo': osgeo.__file__, 'pyramids': pyramids.__path__[0], "
        "'gdal': gdal.__version__}))"
    )
    proc = _run(code)
    if proc.returncode != 0:
        pytest.fail(
            "`import pyramids` then `import osgeo` failed in a fresh interpreter, so the "
            f"vendored bootstrap is not working:\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestVendoredOsgeoBootstrap:
    """Importing `pyramids` is what makes `osgeo` importable, and it works."""

    def test_osgeo_is_importable_after_pyramids(self, bootstrapped_osgeo):
        """A fresh interpreter that imports pyramids can then import a working osgeo/gdal."""
        assert bootstrapped_osgeo[
            "gdal"
        ], "gdal.__version__ came back empty after the bootstrap"

    @pytest.mark.skipif(
        TOP_LEVEL_OSGEO_INSTALLED,
        reason="a top-level osgeo is installed alongside pyramids, so it resolves without the wheel",
    )
    def test_osgeo_resolves_inside_the_pyramids_wheel(self, bootstrapped_osgeo):
        """With no top-level osgeo installed, the imported osgeo is the one vendored in the wheel."""
        vendor = Path(bootstrapped_osgeo["pyramids"]) / "_vendor" / "osgeo"
        if not vendor.is_dir():
            pytest.skip("pyramids is an editable install with an unpopulated _vendor/")
        resolved = Path(bootstrapped_osgeo["osgeo"])
        assert (
            vendor in resolved.parents
        ), f"osgeo resolved to {resolved}, expected it under the wheel's vendor dir {vendor}"

    @pytest.mark.skipif(
        TOP_LEVEL_OSGEO_INSTALLED,
        reason="a top-level osgeo is installed alongside pyramids, so it resolves without the wheel",
    )
    def test_osgeo_is_not_importable_without_pyramids(self, bare_osgeo_import):
        """Importing osgeo with no prior `import pyramids` fails, which is why conftest imports it first."""
        assert bare_osgeo_import.returncode != 0, (
            "`import osgeo` succeeded with no prior `import pyramids`, resolving to "
            f"{bare_osgeo_import.stdout.strip()}; the conftest import order is no longer load-bearing "
            "and this test's premise needs revisiting"
        )
        assert (
            "No module named 'osgeo'" in bare_osgeo_import.stderr
        ), f"expected a ModuleNotFoundError for osgeo, got: {bare_osgeo_import.stderr}"


class TestSubprocessIsolation:
    """The probes above answer a question about the environment, not about where pytest was run."""

    def test_the_working_directory_is_not_on_the_child_import_path(self, tmp_path):
        """An `osgeo/` sitting beside the child's working directory stays invisible to it."""
        decoy = tmp_path / "osgeo"
        decoy.mkdir()
        (decoy / "__init__.py").write_text("", encoding="utf-8")
        proc = _run("import osgeo; print(osgeo.__file__)", cwd=tmp_path)
        assert str(tmp_path) not in proc.stdout, (
            f"the child imported the decoy osgeo at {proc.stdout.strip()}, so its working directory "
            "is on sys.path and the directory pytest was invoked from can decide the result of the "
            "assertions above"
        )
