[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/serapeum-org/digital-rivers/branch/main/graph/badge.svg)](https://codecov.io/gh/serapeum-org/digital-rivers)

# digital-rivers

**digital-rivers** is a small GIS utility library for Digital Elevation Model (DEM) processing and terrain analysis. It builds on [GDAL](https://gdal.org) and the [`pyramids`](https://github.com/serapeum-org/pyramids) raster wrapper to provide:

- **DEM processing** — sink filling, D8 flow direction, flow accumulation, slope (stack-based DFS, no recursion-limit hacks).
- **Terrain visualisation** — color relief, hill shade, slope, and aspect via GDAL's `DEMProcessing`.

The package exposes two classes: `DEM` and `Terrain`. Both subclass `pyramids.dataset.Dataset`, so any pyramids method works on them.

> **Naming note** — the distribution name on PyPI is `digital-rivers` (with hyphen), the Python import name is `digitalrivers` (no separator).

## Installation

The package is **not yet published to conda-forge or PyPI**. Install from source for now:

### With Pixi (recommended — provides GDAL via conda-forge)

```bash
git clone https://github.com/serapeum-org/digital-rivers.git
cd digital-rivers
pixi install -e dev      # creates the dev environment
pixi shell -e dev
```

### With pip

GDAL must already be importable. If you don't have it from conda-forge:

```bash
pip install git+https://github.com/serapeum-org/digital-rivers.git
```

Optional plotting extras (pulls `cleopatra` via pyramids' `[viz]` extra):

```bash
pip install "digital-rivers[viz] @ git+https://github.com/serapeum-org/digital-rivers.git"
```

Supported Python: **3.11–3.13**.

## Quick start

### DEM processing

```python
from osgeo import gdal
from digitalrivers.dem import DEM

dem = DEM(gdal.Open("path/to/dem.tif"))

filled = dem.fill_sinks()                  # remove single-cell sinks
slope = dem.slope()                        # max downhill slope (D8)
fd = dem.flow_direction()                  # 0–7 D8 codes
acc = dem.flow_accumulation(fd)            # upstream cell counts
```

You can pin the basin outfall direction via `flow_direction(forced_direction=gdf)` where `gdf` is a `GeoDataFrame` with `geometry` (point) and `direction` (int 0–7) columns.

### Terrain visualisation

```python
import pandas as pd
from digitalrivers.terrain import Terrain

terrain = Terrain("path/to/dem.tif")

# Hill shade
hs = terrain.hill_shade(azimuth=315, altitude=45)

# Color relief from a hex palette
palette = pd.DataFrame({
    "values": [0, 500, 1500, 3000],
    "color":  ["#3a7d44", "#f2cb05", "#bc4b51", "#8c8c8c"],
})
relief = terrain.color_relief(band=0, color_table=palette)

# GDAL-based slope and aspect
slope = terrain.slope(slope_format="degree", algorithm="Horn")
aspect = terrain.aspect(zero_flat_surface=True)
```

## Project layout

```
src/digitalrivers/
  dem.py        — DEM class (hydrological analysis)
  terrain.py    — Terrain class (color relief, hill shade, slope, aspect)
tests/          — pytest suite + Coello river basin fixtures
examples/       — runnable scripts and notebooks
docs/           — MkDocs sources (MkDocs Material + mkdocstrings)
```

## Documentation

Full API reference is built with MkDocs Material:

- Live site: <https://serapeum-org.github.io/digital-rivers/latest/>
- Local preview: `pixi run -e docs mkdocs serve`

## Development

This repository uses [Pixi](https://pixi.sh/) for environment management.

```bash
pixi run main          # run main test suite (excludes plot tests)
pixi run plot          # run plot/visualization tests
pixi run notebooks     # validate example notebooks
pre-commit run --all-files
```

See [`CLAUDE.md`](./CLAUDE.md) for more development notes.

## License

GNU General Public License v3 — see [`LICENSE.md`](./LICENSE.md).
