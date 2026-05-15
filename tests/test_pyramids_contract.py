"""Pyramids inheritance-contract smoke test.

The typed result classes in P1 (``FlowDirection`` / ``Accumulation`` /
``StreamRaster``) rely on three specific properties of the installed
``pyramids.dataset.Dataset``:

1. The three classmethods ``read_file`` / ``create_from_array`` /
   ``dataset_like`` exist and end with ``cls(...)`` (so subclass identity
   is preserved when called on a typed subclass).
2. The ``meta_data`` property has a setter and writes per-key via
   ``SetMetadataItem`` (the channel for ``DR_ROUTING`` etc.).
3. The same classmethods, when called from a subclass with a single-arg
   ``__init__`` (like ``DEM``), return an instance of the subclass — not a
   plain ``Dataset``. This is the regression guard for the
   ``Dataset(...)``-returning paths at pyramids ``dataset.py`` lines 1152 /
   1539 / 2076 / 3472; if any release routes ``dataset_like`` through one of
   those paths, this test fails first.

``DEM`` is used rather than ``FlowDirection`` in test (3) because the typed
subclasses require a ``routing`` kwarg that pyramids' inner
``cls(dst, access="write")`` cannot supply — they would raise ``TypeError``
before we could inspect the returned type. ``DEM`` keeps a single-arg
``__init__``, so it isolates the regression we actually want to detect.
"""
from __future__ import annotations

import inspect

import numpy as np
from pyramids.dataset import Dataset

from digitalrivers import DEM


def test_dataset_has_classmethod_read_file():
    assert callable(getattr(Dataset, "read_file", None))
    assert inspect.ismethod(Dataset.read_file) or isinstance(
        inspect.getattr_static(Dataset, "read_file"), classmethod
    )


def test_dataset_has_classmethod_create_from_array():
    assert callable(getattr(Dataset, "create_from_array", None))
    assert isinstance(
        inspect.getattr_static(Dataset, "create_from_array"), classmethod
    )


def test_dataset_has_classmethod_dataset_like():
    assert callable(getattr(Dataset, "dataset_like", None))
    assert isinstance(inspect.getattr_static(Dataset, "dataset_like"), classmethod)


def test_dataset_has_meta_data_setter():
    prop = inspect.getattr_static(Dataset, "meta_data")
    assert isinstance(prop, property)
    assert prop.fset is not None, (
        "pyramids.Dataset.meta_data must have a setter — that is the channel "
        "we use to write DR_ROUTING / DR_ENCODING / DR_THRESHOLD tags."
    )


def test_dem_dataset_like_preserves_subclass():
    """Regression guard for the pyramids ``Dataset(...)``-returning paths.

    If a future pyramids release routes ``dataset_like`` through one of the
    four ``return Dataset(...)`` call sites instead of ``cls(...)``, the
    returned object will be a plain ``Dataset`` rather than a ``DEM`` and
    this test fires before any of the typed-result tests do.
    """
    arr = np.array([[100.0, 200.0], [150.0, 250.0]], dtype=np.float32)
    src = Dataset.create_from_array(
        arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326, no_data_value=-9999
    )
    dem = DEM(src.raster)
    new_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = DEM.dataset_like(dem, new_arr)
    assert type(out) is DEM, (
        f"DEM.dataset_like returned {type(out).__name__}; expected DEM. "
        f"This is the canary for pyramids losing the cls(...) contract — "
        f"the typed-result classes in P1 will silently degrade if this passes "
        f"silently."
    )


def test_dem_create_from_array_preserves_subclass():
    """Same regression guard, for ``create_from_array`` (pyramids line 2280)."""
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    out = DEM.create_from_array(
        arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326, no_data_value=-9999
    )
    assert type(out) is DEM
