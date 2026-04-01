[![Documentation Status](https://readthedocs.org/projects/digital-rivers/badge/?version=latest)](https://digital-rivers.readthedocs.io/en/latest/?badge=latest)
[![Python Versions](https://img.shields.io/pypi/pyversions/digitalrivers.png)](https://img.shields.io/pypi/pyversions/digitalrivers)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MAfarrag/Hapi.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MAfarrag/Hapi/context:python)

![GitHub last commit](https://img.shields.io/github/last-commit/MAfarrag/digitalrivers)
![GitHub forks](https://img.shields.io/github/forks/MAfarrag/digitalrivers?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/MAfarrag/digitalrivers?style=social)
[![codecov](https://codecov.io/gh/serapeum-org/digitalrivers/branch/main/graph/badge.svg?token=g0DV4dCa8N)](https://codecov.io/gh/serapeum-org/digitalrivers)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5e3aa4d0acc843d1a91caf33545ecf03)](https://www.codacy.com/gh/serapeum-org/digitalrivers/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=serapeum-org/digitalrivers&amp;utm_campaign=Badge_Grade)

![GitHub commits since latest release (by SemVer including pre-releases)](https://img.shields.io/github/commits-since/mafarrag/digitalrivers/0.1.0?include_prereleases&style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/mafarrag/digitalrivers)

Current release info
====================

| Name                                                                                                                 | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Version                                                                                                                                                                                                                     | Platforms                                                                                                                                                                                                                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Conda Recipe](https://img.shields.io/badge/recipe-digitalrivers-green.svg)](https://anaconda.org/conda-forge/digitalrivers) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/digitalrivers.svg)](https://anaconda.org/conda-forge/digitalrivers) [![Downloads](https://pepy.tech/badge/digitalrivers)](https://pepy.tech/project/digitalrivers) [![Downloads](https://pepy.tech/badge/digitalrivers/month)](https://pepy.tech/project/digitalrivers)  [![Downloads](https://pepy.tech/badge/digitalrivers/week)](https://pepy.tech/project/digitalrivers)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/digitalrivers?color=blue&style=flat-square) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/digitalrivers.svg)](https://anaconda.org/conda-forge/digitalrivers) [![PyPI version](https://badge.fury.io/py/digitalrivers.svg)](https://badge.fury.io/py/digitalrivers) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/digitalrivers.svg)](https://anaconda.org/conda-forge/digitalrivers) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

digitalrivers - GIS utility package
=====================================================================
**digitalrivers** is a GIS utility package using gdal, ....

digitalrivers

![1](/docs/images/package-work-flow/overall.png)

Main Features
-------------

- GIS modules to enable the modeler to fully prepare the meteorological inputs and do all the preprocessing
  needed to build the model (align rasters with the DEM), in addition to various methods to manipulate and
  convert different forms of distributed data (rasters, NetCDF, shapefiles)

Future work
-------------

- Developing a DEM processing module for generating the river network at different DEM spatial resolutions.

Installing digitalrivers
===============

Installing `digitalrivers` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge digitalrivers=0.1.0
```

It is possible to list all the versions of `digitalrivers` available on your platform with:

```
conda search digitalrivers --channel conda-forge
```

## Install from GitHub

to install the last development to time, you can install the library from GitHub

```
pip install git+https://github.com/serapeum-org/digitalrivers
```

## pip

to install the last release, you can easily use pip

```
pip install digitalrivers==0.1.0
```

Quick start
===========

```
  >>> import digitalrivers
```

[other code samples](https://digitalrivers.readthedocs.io/en/latest/?badge=latest)

## Coverage

[![codecov](https://codecov.io/gh/serapeum-org/digitalrivers/branch/main/graphs/sunburst.svg?token=g0DV4dCa8N)](https://codecov.io/gh/serapeum-org/digitalrivers)
