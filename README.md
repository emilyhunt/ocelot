[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://ocelot-docs.org)
[![Build Docs](https://github.com/emilyhunt/ocelot/actions/workflows/build-docs.yml/badge.svg)](https://ocelot-docs.org)

# ocelot

A toolbox for working with observations of star clusters. 

In the [long-running tradition](https://arxiv.org/abs/1903.12180) of astronomy software, `ocelot` is _not_ a good acronym for this project. It's the **O**pen-source star **C**lust**E**r mu**L**ti-purp**O**se **T**oolkit. (We hope the results you get from this package are better than this acronym)

## Current package status

⚠️ ocelot is currently in **alpha** and is in active development. **Expect breaking API changes** ⚠️

For the time being, `ocelot` is a collection of code that [emilyhunt](https://github.com/emilyhunt) wrote during her PhD, but the eventual goal will be to make a package usable by the entire star cluster community. If you'd like to see a feature added, then please consider opening an issue and proposing it!

## Installation

Install from PyPI with:

```
pip install ocelot
```

## Development

If you'd like to contribute to the package, we recommend setting up a new virtual environment of your choice. Then, you can install the latest commit on the main branch in edit mode (`-e`) with all development dependencies (`[dev]`) with:

```
pip install -e git+https://github.com/emilyhunt/ocelot[dev]
```

After installing development dependencies, you can also make and view edits to the package's documentation. To view a local copy of the documentation, do `mkdocs serve`. You can do a test build with `mkdocs build`.
