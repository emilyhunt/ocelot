[![docs](https://img.shields.io/badge/docs-latest-orange.svg)](https://ocelot-docs.org)
[![PyPI](https://img.shields.io/badge/PyPI-package-blue.svg)](https://pypi.org/project/ocelot/)
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

If you'd like to contribute to the package, we recommend setting up a new virtual environment with a tool of your choice. Then, you can install the latest commit on the main branch in edit mode (`-e`) with all development dependencies (`[dev]`) with:

```
pip install -e git+https://github.com/emilyhunt/ocelot[dev]
```

After installing development dependencies, you can also make and view edits to the package's documentation. To view a local copy of the documentation, do `mkdocs serve`. You can do a test build with `mkdocs build`.


## Citation

There is currently no paper associated with `ocelot`. For now, please at least mention the package and add a footnote to your mention, linking to this repository - in LaTeX, that would be:

```
\footnote{\url{https://github.com/emilyhunt/ocelot}}
```


For now, you can also cite [Hunt & Reffert 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A.104H/abstract), which was the paper for which development of this module began:

```
@ARTICLE{2021A&A...646A.104H,
       author = {{Hunt}, Emily L. and {Reffert}, Sabine},
        title = "{Improving the open cluster census. I. Comparison of clustering algorithms applied to Gaia DR2 data}",
      journal = {\aap},
     keywords = {methods: data analysis, open clusters and associations: general, astrometry, Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = feb,
       volume = {646},
          eid = {A104},
        pages = {A104},
          doi = {10.1051/0004-6361/202039341},
archivePrefix = {arXiv},
       eprint = {2012.04267},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021A&A...646A.104H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
