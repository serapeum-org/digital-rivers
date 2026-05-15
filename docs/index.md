# digital-rivers

A small GIS utility library for **DEM processing** and **terrain analysis**, built on GDAL and the [`pyramids`](https://github.com/serapeum-org/pyramids) raster wrapper.

## What it does

- **`DEM`** — sink filling, D8 flow direction, flow accumulation, slope (max-downhill across 8 neighbours), and basin filtering.
- **`Terrain`** — color relief, hill shade, slope, and aspect via GDAL's `DEMProcessing` utility.

Both classes subclass `pyramids.dataset.Dataset`, so every pyramids method (`crop`, `to_crs`, `plot`, `stats`, …) is available alongside the digital-rivers additions.

## Quick start

```python
from osgeo import gdal
from digitalrivers.dem import DEM
from digitalrivers.terrain import Terrain

dem = DEM(gdal.Open("dem.tif"))
fd = dem.flow_direction()
acc = dem.flow_accumulation(fd)

terrain = Terrain("dem.tif")
hill_shade = terrain.hill_shade()
```

## Where to next

- [Installation](installation.md) — Pixi/conda or pip-from-source instructions.
- [DEM reference](reference/dem.md) — full API for hydrological analysis.
- [Terrain reference](reference/terrain.md) — full API for visualisation.
- [Change log](change-log.md)
- [Architecture Decision Records](adr/index.md)

## Status

Pre-release (v0.1.0). Not yet on PyPI or conda-forge — install from source. Supported Python: **3.11–3.13**.

Source: <https://github.com/serapeum-org/digital-rivers>
