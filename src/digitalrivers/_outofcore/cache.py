"""Per-tile intermediate store with three memory modes (Barnes 2016 §4.2).

A tiled algorithm runs in three stages — per-tile pass, global edge reduction, per-tile finalize. The middle
stage only needs tile *perimeters*; whether the per-tile *interiors* survive between stage 1 and stage 3 is a
RAM-vs-recompute trade-off, captured here as three modes:

* ``"retain"`` — keep every tile's intermediates in an in-RAM dict (1 read + 1 write per cell). Only viable when
  the dataset fits in RAM; also the natural fast path for the in-memory equivalence harness.
* ``"cache"`` — spill intermediates to ``scratch_dir`` as ``.npy`` files and reload them in the finalize stage
  (≈ 3 reads + 3 writes per cell). Bounded RAM at the cost of scratch disk.
* ``"evict"`` — store nothing (smallest footprint, the safe out-of-core default); the orchestrator recomputes the
  per-tile pass in stage 3. :attr:`TileStore.recompute` signals this.
"""

from __future__ import annotations

import os

import numpy as np


class TileStore:
    """Stage-1 → stage-3 intermediate store with ``retain`` / ``cache`` / ``evict`` modes.

    Args:
        mode: One of ``"retain"``, ``"cache"``, ``"evict"``. Defaults to ``"evict"``.
        scratch_dir: Directory for spilled ``.npy`` tiles; required (and created) for ``"cache"`` mode.

    Raises:
        ValueError: If ``mode`` is unknown, or ``"cache"`` is requested without ``scratch_dir``.

    Examples:
        >>> store = TileStore("retain")
        >>> import numpy as np
        >>> store.put(0, filled=np.array([[1.0, 2.0]]), labels=np.array([[1, 1]]))
        >>> got = store.get(0)
        >>> sorted(got)
        ['filled', 'labels']
        >>> got["filled"].tolist()
        [[1.0, 2.0]]
        >>> TileStore("evict").get(0) is None
        True
        >>> TileStore("evict").recompute
        True
    """

    VALID_MODES = ("retain", "cache", "evict")

    def __init__(self, mode: str = "evict", scratch_dir: str | None = None):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        if mode == "cache" and scratch_dir is None:
            raise ValueError("cache mode requires scratch_dir")
        self.mode = mode
        self.scratch_dir = scratch_dir
        self._mem: dict[int, dict[str, np.ndarray]] = {}
        self._keys: dict[int, set[str]] = {}
        if mode == "cache":
            os.makedirs(scratch_dir, exist_ok=True)

    @property
    def recompute(self) -> bool:
        """True when stage 3 must recompute per-tile intermediates (i.e. ``mode == "evict"``)."""
        return self.mode == "evict"

    def put(self, tid: int, **arrays: np.ndarray) -> None:
        """Store named intermediate arrays for tile ``tid``.

        A no-op in ``evict`` mode. In ``retain`` mode arrays are held in RAM; in ``cache`` mode they are written
        to ``scratch_dir``.
        """
        if self.mode == "evict":
            return
        if self.mode == "retain":
            self._mem.setdefault(tid, {}).update(
                {k: np.asarray(v) for k, v in arrays.items()}
            )
            return
        # cache
        self._keys.setdefault(tid, set())
        for key, value in arrays.items():
            np.save(self._path(tid, key), np.asarray(value))
            self._keys[tid].add(key)

    def get(self, tid: int) -> dict[str, np.ndarray] | None:
        """Return the stored arrays for tile ``tid``, or ``None`` if nothing is stored (``evict``, or unseen)."""
        if self.mode == "evict":
            return None
        if self.mode == "retain":
            return self._mem.get(tid)
        keys = self._keys.get(tid)
        if not keys:
            return None
        return {key: np.load(self._path(tid, key)) for key in keys}

    def _path(self, tid: int, key: str) -> str:
        return os.path.join(self.scratch_dir, f"tile_{tid}_{key}.npy")
