"""Pins the osgeo import-order contract the whole test suite rests on.

The `osgeo` bindings are not installed at the top level any more: they ship inside the
`pyramids` wheel under `_vendor/osgeo/`, and `pyramids/__init__.py` is what puts that
directory on `sys.path`. Ten test modules do `from osgeo import gdal` before they import
`digitalrivers`, so the `import pyramids` on the first line of `tests/conftest.py` is what
makes their imports resolve.

Deleting that line — or letting isort under the configured black profile sort it below the
`osgeo` import, which is where that profile puts it — breaks collection in every one of
those modules at once, with nothing naming the cause. These tests name it.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")

_SITE_DIRS = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}

# Whether a top-level `osgeo` package (conda-forge's gdal bindings, say) is installed
# alongside pyramids. When one is, `osgeo` resolves without the vendor bootstrap and the
# two wheel-shape assertions below do not apply to that installation.
TOP_LEVEL_OSGEO_INSTALLED = any((Path(d) / "osgeo").is_dir() for d in _SITE_DIRS)


def _module_imports(path: Path = CONFTEST) -> list[tuple[int, str]]:
    """Return `(lineno, root module name)` for every import in `path`, in file order."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                (node.lineno, alias.name.split(".")[0]) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module.split(".")[0]))
    return sorted(found)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter that has imported nothing beforehand."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
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
    assert (
        proc.returncode == 0
    ), f"`import pyramids` then `import osgeo` failed: {proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestConftestImportOrder:
    """The static half of the contract: `tests/conftest.py` keeps `import pyramids` first."""

    def test_pyramids_is_the_first_import(self):
        """The very first import executed by conftest is `pyramids`, not anything else."""
        imports = _module_imports()
        assert imports, f"no imports parsed out of {CONFTEST}"
        lineno, name = imports[0]
        assert name == "pyramids", (
            f"the first import in {CONFTEST.name} is `{name}` at line {lineno}; it must be "
            "`import pyramids`, which is what puts the vendored osgeo on sys.path"
        )

    def test_pyramids_import_precedes_every_osgeo_import(self):
        """Every `osgeo` import in conftest sits below the `pyramids` import that enables it."""
        imports = _module_imports()
        pyramids_lines = [line for line, name in imports if name == "pyramids"]
        osgeo_lines = [line for line, name in imports if name == "osgeo"]
        assert pyramids_lines, (
            f"{CONFTEST.name} no longer imports pyramids at all; every test module that does "
            "`from osgeo import gdal` will fail to import"
        )
        assert (
            osgeo_lines
        ), "conftest no longer imports osgeo; this contract test needs revisiting"
        pyramids_line = min(pyramids_lines)
        assert pyramids_line < min(osgeo_lines), (
            f"`import pyramids` is at line {pyramids_line} but osgeo is imported at "
            f"{osgeo_lines}; osgeo cannot resolve before the pyramids bootstrap runs"
        )

    def test_the_pyramids_import_is_pinned_against_isort(self):
        """The `import pyramids` line carries `isort:skip` so a sorter cannot move it below osgeo."""
        line = CONFTEST.read_text(encoding="utf-8").splitlines()[
            _module_imports()[0][0] - 1
        ]
        assert (
            "isort:skip" in line
        ), f"the pyramids import must be pinned with `# isort:skip`, got: {line!r}"


class TestVendoredOsgeoBootstrap:
    """The runtime half: importing `pyramids` is what makes `osgeo` importable, and it works."""

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
