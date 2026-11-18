# Core primitives

Shared, hydrology-agnostic building blocks that every domain package rests on. Nothing here imports from the rest
of `digitalrivers`, which is what makes it safe for everything else to import from here.

## Direction convention

The 8-neighbour convention every routing, ordering and walk kernel indexes: `0=S, 1=SW, 2=W, 3=NW, 4=N, 5=NE,
6=E, 7=SE`. It is exported in two shapes — a `(col, row)` dict and parallel `(row, col)` arrays — and the arrays
come in both `int8` and `int32`, because Numba compiles a separate specialisation per argument dtype and the two
halves of the package use different ones.

::: digitalrivers.core.directions
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

## Result metadata

The `DR_*` GeoTIFF tag keys and valid-value sets the typed rasters write, so a raster carries the routing scheme
and encoding that produced it and a downstream reader cannot silently misinterpret it.

::: digitalrivers.core.metadata
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

## Numba availability

The single place that decides whether the JIT path is live. Every kernel decorates with the `njit` exported here
rather than importing `numba` directly, so `DIGITALRIVERS_DISABLE_NUMBA=1` turns the whole package's fast path off
in one step.

::: digitalrivers.core.numba
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
