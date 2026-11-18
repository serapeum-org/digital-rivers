"""Tests for the seven deprecated import paths left behind by the restructure.

These modules exist only to keep older code importing, so they are exactly the files
nobody exercises — and a wrong re-export target, a symbol renamed in its new home, or a
plain syntax error in one would ship undetected. The restructure's own acceptance
criterion was that each shim "emits `DeprecationWarning` once and returns the same object
as the new path"; this file is that criterion, executed.

Each test pops the shim from `sys.modules` first, because the warning fires at import and
a module already imported by an earlier test would not warn again.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

#: old module path -> (new module path, names the shim re-exports)
SHIMS = {
    "digitalrivers.flow_direction": ("digitalrivers.flow.direction", ["FlowDirection"]),
    "digitalrivers.accumulation": ("digitalrivers.flow.accumulation", ["Accumulation"]),
    "digitalrivers.stream_raster": ("digitalrivers.streams.raster", ["StreamRaster"]),
    "digitalrivers.watershed_raster": (
        "digitalrivers.watershed.raster",
        ["WatershedRaster"],
    ),
    "digitalrivers.mesh": ("digitalrivers.interop.mesh", ["Mesh"]),
    "digitalrivers.cloud_io": (
        "digitalrivers.interop.cloud_io",
        ["tile_windows", "write_cog", "dask_backend", "cloud_storage"],
    ),
    "digitalrivers.fusion": ("digitalrivers.dem.fusion", ["topobathy_fusion"]),
}

REMOVAL_VERSION = "0.6.0"


def _fresh_import(module_path: str):
    """Import `module_path` with a clean slate so its import-time warning fires.

    Args:
        module_path: Dotted path of the module to import.

    Returns:
        The imported module.
    """
    sys.modules.pop(module_path, None)
    return importlib.import_module(module_path)


@pytest.mark.parametrize("old_path", sorted(SHIMS))
def test_shim_warns_on_import(old_path):
    """Importing a deprecated path raises `DeprecationWarning`.

    Args:
        old_path: The deprecated module path under test.
    """
    sys.modules.pop(old_path, None)
    with pytest.warns(DeprecationWarning) as record:
        importlib.import_module(old_path)
    assert len(record) >= 1, f"{old_path} imported without warning"


@pytest.mark.parametrize("old_path", sorted(SHIMS))
def test_warning_names_the_new_path_and_the_removal_version(old_path):
    """The warning tells the reader where to go and when the shim disappears.

    Args:
        old_path: The deprecated module path under test.

    Test scenario:
        A deprecation the user cannot act on is noise, so the message has to carry both
        halves: the replacement path and the version that drops the old one.
    """
    new_path = SHIMS[old_path][0]
    sys.modules.pop(old_path, None)
    with pytest.warns(DeprecationWarning) as record:
        importlib.import_module(old_path)
    message = str(record[0].message)
    assert new_path in message, f"Warning omits the new path {new_path!r}: {message}"
    assert REMOVAL_VERSION in message, f"Warning omits the removal version: {message}"


@pytest.mark.parametrize(
    ("old_path", "name"),
    [(old, n) for old, (_new, names) in sorted(SHIMS.items()) for n in names],
)
def test_shim_hands_back_the_identical_object(old_path, name):
    """`old.X is new.X` — the shim re-exports, it does not redefine.

    Args:
        old_path: The deprecated module path.
        name: The symbol re-exported through it.

    Test scenario:
        Identity rather than equality, because a shim that rebuilt the class would give
        an object that fails `isinstance` against the canonical one and would break every
        downstream type check in a way equality would not reveal.
    """
    new_path = SHIMS[old_path][0]
    old_module = _fresh_import(old_path)
    new_module = importlib.import_module(new_path)
    assert hasattr(old_module, name), f"{old_path} no longer exports {name}"
    assert getattr(old_module, name) is getattr(
        new_module, name
    ), f"{old_path}.{name} is not the same object as {new_path}.{name}"


@pytest.mark.parametrize("old_path", sorted(SHIMS))
def test_shim_all_matches_what_it_exports(old_path):
    """Every name in the shim's `__all__` actually resolves.

    Args:
        old_path: The deprecated module path under test.
    """
    module = _fresh_import(old_path)
    for name in module.__all__:
        assert hasattr(module, name), f"{old_path}.__all__ lists a missing {name!r}"
    assert sorted(module.__all__) == sorted(
        SHIMS[old_path][1]
    ), f"{old_path}.__all__ drifted from what this test pins: {module.__all__}"


def test_importing_the_package_does_not_warn():
    """`import digitalrivers` must not fire any shim's deprecation warning.

    Test scenario:
        The package root re-exports from the new locations, so a user who never touches a
        legacy path should never see one of these warnings. If `__init__.py` ever starts
        importing a shim, every downstream user gets a warning they cannot act on.
    """
    for old_path in SHIMS:
        sys.modules.pop(old_path, None)
    sys.modules.pop("digitalrivers", None)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        importlib.import_module("digitalrivers")
    shim_warnings = [
        w
        for w in record
        if issubclass(w.category, DeprecationWarning)
        and "has moved to" in str(w.message)
    ]
    assert (
        not shim_warnings
    ), f"import digitalrivers fired shim warnings: {shim_warnings}"


@pytest.mark.parametrize(
    ("old_path", "name"),
    [
        ("digitalrivers.flow_direction", "FlowDirection"),
        ("digitalrivers.accumulation", "Accumulation"),
        ("digitalrivers.stream_raster", "StreamRaster"),
        ("digitalrivers.watershed_raster", "WatershedRaster"),
        ("digitalrivers.mesh", "Mesh"),
    ],
)
def test_class_shims_agree_with_the_package_root(old_path, name):
    """The five class shims hand back the same object the package root exports.

    Args:
        old_path: The deprecated module path.
        name: The class re-exported through it.

    Test scenario:
        `from digitalrivers import FlowDirection` never moved, so it and the shim must
        resolve to one object. Two live classes for one name is the failure mode that
        breaks `isinstance` at a distance.
    """
    import digitalrivers

    old_module = _fresh_import(old_path)
    assert getattr(old_module, name) is getattr(
        digitalrivers, name
    ), f"{old_path}.{name} and digitalrivers.{name} are different objects"
