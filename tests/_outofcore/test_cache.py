"""Tests for digitalrivers._outofcore.cache.TileStore."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from digitalrivers._outofcore.cache import TileStore


class TestTileStore:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            TileStore("bogus")

    def test_cache_requires_scratch_dir(self):
        with pytest.raises(ValueError):
            TileStore("cache")

    def test_evict_stores_nothing_and_signals_recompute(self):
        store = TileStore("evict")
        store.put(0, filled=np.ones((2, 2)))
        assert store.get(0) is None
        assert store.recompute is True

    def test_retain_roundtrips_arrays(self):
        store = TileStore("retain")
        filled = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([[1, 1], [2, 2]], dtype=np.int32)
        store.put(0, filled=filled, labels=labels)
        got = store.get(0)
        assert sorted(got) == ["filled", "labels"]
        np.testing.assert_array_equal(got["filled"], filled)
        np.testing.assert_array_equal(got["labels"], labels)
        assert store.recompute is False

    def test_retain_missing_tile_returns_none(self):
        assert TileStore("retain").get(99) is None

    def test_cache_roundtrips_via_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TileStore("cache", scratch_dir=tmp)
            acc = np.arange(9, dtype=np.float64).reshape(3, 3)
            store.put(7, acc=acc)
            got = store.get(7)
            assert list(got) == ["acc"]
            np.testing.assert_array_equal(got["acc"], acc)
            assert store.recompute is False

    def test_cache_missing_tile_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert TileStore("cache", scratch_dir=tmp).get(5) is None
