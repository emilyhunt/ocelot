Installing ocelot is (relatively) easy!

Firstly, install from pip with:

```bash
pip install ocelot
```

to get started, which will install the latest release of the code.

To install dependencies required for cluster simulation, you'll need to do

```bash
pip install ocelot[simulate]
```

due to a separate issue that's currently being resolved in one of our dependencies.


## Getting additional data

To use ocelot.simulate, you will also need to install some additional data (namely, isochrones). Download [this folder](https://drive.google.com/file/d/1wMXymFHo-K5jdIGoJi5oGuHeXSa3JVmu/view?usp=sharing) of isochrones, extract it, and place it anywhere you'd like.

When running ocelot, you'll need to set the environment variable `OCELOT_DATA` to point towards your data.

... and that's it! Next off, check out [ocelot's best bits](features.md) to see what ocelot can do.