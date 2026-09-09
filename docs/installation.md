# Installation

This page covers installation of **digital-rivers** and its native dependencies.

| Item | Value |
|---|---|
| Distribution name (PyPI / conda-forge) | `digital-rivers` |
| Python import name | `digitalrivers` |
| Current version | `0.4.0` |
| Supported Python | **3.11 – 3.14** |
| License | GPL v3 |

> **conda-forge carries the current release; PyPI lags.** conda-forge has `0.4.0`, PyPI's newest is
> `0.1.0`. Install from conda-forge, or from source for anything newer than what PyPI offers.

## Dependencies

### Runtime
- `numpy >= 2.0.0`
- `geopandas >= 1.0.0`
- `pyramids-gis >= 0.60.0` (provides the `pyramids` import; pulled from PyPI)
GDAL is deliberately absent from that list. It is not a dependency of this package: the
`pyramids-gis` wheel vendors its own osgeo bindings, and `import pyramids` is what puts them
on `sys.path`. Nothing here installs or pins a separate GDAL.

### Optional extras
| Extra | Purpose | Pulls |
|---|---|---|
| `viz` | plotting / color tables | `pyramids-gis[viz]`, `cleopatra` |
| `distributed` | out-of-core / Dask backend | `pyramids-gis[lazy]` |
| `all` | both of the above | — |

### Development dependency groups
`dev`, `docs`, `lazy` and `notebook` are [PEP 735](https://peps.python.org/pep-0735/)
dependency groups, not extras — they are local tooling and are deliberately not published in
the package metadata. Pixi resolves a group by name wherever an environment lists it, so an
environment is a set of groups and extras rather than a single same-named group: `dev` is
`py314` + `dev` + `viz` + `lazy`, `docs` is `py314` + `docs`. `lazy` carries dask and
distributed for the out-of-core path; `notebook` is currently referenced by no environment.

```bash
pixi install -e dev      # tests, linting, build tooling
pixi install -e docs     # mkdocs toolchain
```

## Recommended: Pixi

This repository ships a [Pixi](https://pixi.sh/) configuration that resolves every dependency from
PyPI, GDAL included — the `pyramids-gis` wheel vendors its own, so no conda channel is involved and
the usual GDAL-wheel headaches do not arise.

Prerequisites: install [Pixi](https://pixi.sh/latest/#installation).

```bash
git clone https://github.com/serapeum-org/digital-rivers.git
cd digital-rivers

# Solve and install the dev environment
pixi install -e dev

# Drop into a shell with everything available
pixi shell -e dev

# Or run a task directly
pixi run main          # main test suite
pixi run plot          # plot/visualization tests
pixi run notebooks     # validate example notebooks
```

### Available Pixi environments

| Environment | Features | Purpose |
|---|---|---|
| `default` | `py314` | minimal runtime |
| `dev` | `py314`, `dev`, `viz`, `lazy` | tests, linting, build tooling |
| `docs` | `py314`, `docs` | docs site (`mkdocs serve`) |
| `py311` | `py311`, `dev`, `viz`, `lazy` | pinned Python 3.11 |
| `py312` | `py312`, `dev`, `viz`, `lazy` | pinned Python 3.12 |
| `py313` | `py313`, `dev`, `viz`, `lazy` | pinned Python 3.13 |
| `py314` | `py314`, `dev`, `viz`, `lazy` | pinned Python 3.14 |

`default`, `dev` and `docs` are not unpinned — they reuse the `py314` feature, so they
resolve Python 3.14 like the matrix environment of that name. `viz` is a PEP 621 extra;
`dev`, `docs` and `lazy` are PEP 735 dependency groups. The definitions live in
`pyproject.toml` under `[tool.pixi.environments]`.

## Alternative: conda

If you'd rather manage the environment yourself, install the native stack from
conda-forge and add the package from source:

```bash
mamba create -n digital-rivers -c conda-forge python=3.12 gdal libgdal-netcdf libgdal-hdf4
mamba activate digital-rivers
```

The GDAL version is deliberately unpinned here. `import pyramids` prepends its vendored
`_vendor/osgeo` to `sys.path`, so that copy is the one every import resolves to and a conda
GDAL alongside it is never imported. It is in this recipe only for `libgdal-netcdf` and
`libgdal-hdf4`, which need the C library; any version new enough for those will do.

The repository's own environments are defined in `pyproject.toml` and resolved by
pixi; `pixi install -e dev` is the supported way to reproduce them exactly, and the
only one that honours `pixi.lock`.

## Editable / development install

```bash
git clone https://github.com/serapeum-org/digital-rivers.git
cd digital-rivers
pixi install -e dev
pixi run -e dev pre-commit install
```

The package itself is registered as an editable pixi pypi-dependency, so
`pixi install -e dev` already puts your checkout on the path — there is no separate
editable-install step.

## Quick check

```python
>>> import digitalrivers
>>> digitalrivers.__version__
'0.4.0'
>>> from digitalrivers import DEM, Terrain
```

## Notes

- `pyramids` (conda-forge name) and `pyramids-gis` (PyPI name) are the **same package**. digital-rivers
  depends on the PyPI distribution name (`pyramids-gis`) so it works regardless of how pyramids itself was
  installed.
- The conda-forge ↔ PyPI hash-mapping lag that pixi used to hit here cannot occur any more: it applies
  only to conda packages, and this workspace resolves everything from PyPI.
- Documentation: <https://serapeum-org.github.io/digital-rivers/latest>
- Source repository: <https://github.com/serapeum-org/digital-rivers>
