# Interop

Everything that hands a result to something outside the package.

## Hydrodynamic-model export

Six writers, each expecting a different shape of the same surface. Each takes an `ExportGrid` — the elevation
array plus the georeferencing the headers need — and returns a `dict` mapping an artefact label to the path it
wrote, so a caller that does not know which target produces two files still learns both names.

`DEM.export` validates the target and dispatches through `write`.

::: digitalrivers.interop.export
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

## Sub-grid bathymetry

Per-coarse-cell depth / wetted-fraction tables, the representation SFINCS and similar reduced-order solvers use
to recover small-scale topography they do not resolve.

::: digitalrivers.interop.subgrid
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

## ANUDEM-lite gap fill

Relaxation gap-fill for a surface with holes — cloud shadow, survey gaps, vegetation occlusion. Two solvers: a
Laplacian membrane, and a biharmonic plate that matches slope as well as value at the hole edge.

::: digitalrivers.interop.anudem
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
