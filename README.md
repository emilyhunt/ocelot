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

Currently, using `ocelot.simulate` also requires manually downloading data from [here](https://drive.google.com/file/d/1wMXymFHo-K5jdIGoJi5oGuHeXSa3JVmu/view?usp=sharing). Place it at a directory of your choosing, and set the environment variable `OCELOT_DATA` to this location. 

If you're just working with a local dev copy of ocelot (i.e. you installed it via git clone), then you could put the data at the default location - /data in this folder.

## Development

We recommend using [uv](https://docs.astral.sh/uv/) to manage Python dependencies when developing a local copy of the project. Here's everything you need to do:

1. Clone the repo:

```
git clone https://github.com/emilyhunt/ocelot
```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), if you haven't already. (This won't mess with any of your other Python installations.)

3. Navigate to the new ocelot directory, and sync the project dependences _including dev and docs ones_ with:

```
uv sync --all-extras
```

After installing development dependencies, you can also locally view edits to the package's documentation. To view a local copy of the documentation, do `mkdocs serve`. You can do a test build with `mkdocs build`.


## Citation

There is currently no paper associated with `ocelot`. For now, please at least mention the package and add a footnote to your mention, linking to this repository - in LaTeX, that would be:

```
\footnote{\url{https://github.com/emilyhunt/ocelot}}
```


You can also cite [Hunt & Reffert 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A.104H/abstract), which was the paper for which development of this module began:

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
