"""Engine selection for the in-memory vs tiled (out-of-core) execution paths (B5).

The public hydrology methods take ``engine="auto" | "in_memory" | "tiled"``. ``"auto"`` estimates the per-stage
peak memory of the whole-array path (≈ ``k · rows · cols · 8`` bytes, with the surveyed ~5–10× multiplier) and
flips to the tiled engine only when that would exceed a safety fraction of available RAM — so ordinary-sized DEMs
keep the exact in-memory behaviour and only genuinely-too-large ones stream to disk.
"""

from __future__ import annotations

_VALID_ENGINES = ("auto", "in_memory", "tiled")


def resolve_engine(
    engine: str,
    rows: int,
    cols: int,
    *,
    k: int = 8,
    safety: float = 0.5,
    available_bytes: int | None = None,
    threshold_cells: int = 50_000_000,
) -> str:
    """Resolve ``engine`` to a concrete ``"in_memory"`` or ``"tiled"`` choice.

    Args:
        engine: ``"auto"``, ``"in_memory"`` or ``"tiled"``.
        rows: Raster height in cells.
        cols: Raster width in cells.
        k: Per-stage peak-memory multiplier (≈ number of full-size float64 auxiliaries). ``8`` for fill, ``6``
            for accumulation.
        safety: Fraction of available RAM the estimate may use before switching to tiled. Defaults to 0.5.
        available_bytes: Override for available RAM (mainly for tests). When ``None``, queried via ``psutil`` if
            installed, else the ``threshold_cells`` fallback is used.
        threshold_cells: Cell-count threshold used only when available RAM is unknown (no ``psutil``).

    Returns:
        ``"in_memory"`` or ``"tiled"``.

    Raises:
        ValueError: If ``engine`` is not one of the three valid values.

    Examples:
        >>> resolve_engine("in_memory", 10, 10)
        'in_memory'
        >>> resolve_engine("auto", 100, 100, available_bytes=10**12)
        'in_memory'
        >>> resolve_engine("auto", 60000, 60000, available_bytes=8 * 10**9)
        'tiled'
    """
    if engine not in _VALID_ENGINES:
        raise ValueError(f"engine must be one of {_VALID_ENGINES}, got {engine!r}")
    if engine != "auto":
        return engine

    cells = int(rows) * int(cols)
    estimate = k * cells * 8  # bytes

    if available_bytes is None:
        try:
            import psutil  # defensive optional dependency

            available_bytes = int(psutil.virtual_memory().available)
        except Exception:  # pragma: no cover - psutil absent / unavailable
            available_bytes = None

    if available_bytes is not None:
        return "tiled" if estimate > safety * available_bytes else "in_memory"
    return "tiled" if cells > threshold_cells else "in_memory"
