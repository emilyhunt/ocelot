## v0.4.0 - Addition of cluster simulation code and models

This release brings a brand new, sophisticated API for cluster simulation, in addition to optimized models covering a number of different aspects of star cluster science. Additions include:

- The new `ocelot.simulate` submodule for simulating clusters, including:
    - The flexible and hackable `SimulatedCluster` class. Extensive code optimizations allow for most open clusters to have full simulations (even randomly sampling the orbits of their binary stars to see if they would or wouldn't be resolved!) in less than a second
    - New `SimulatedClusterParamters`, `SimulatedClusterModels`, and `SimulatedClusterFeatures` dataclasses for controlling `SimulatedCluster` functionality
    - Ability to simulate observations of simulated clusters, allowing the same simulated cluster to be 'observed' by many different telescopes
- The new `ocelot.models` submodule for star cluster models, including:
    - A model of King 1962 empirical star clusters. Currently limited to sampling clusters in 3D
    - A bespoke model of differential extinction that approximates dust structure with fractal noise
    - A heavily optimized implementation of the Moe & DiStefano 2017 binary star models
    - Models for selection functions due to observations, such as an optimized implementation of the Castro-Ginard 2023 subsample selection function that can be used to model any generic subsample of stars
    - A model for star cluster observations with Gaia
- Extensive unit tests to assure the reliability and accuracy of simulated clusters and their models
- Documentation improvements, including the first tutorial in the module
- Refactoring of some old aspects of the module

The APIs of `ocelot.simulate` and `ocelot.models` are now the first stable APIs of ocelot, and are ready to use in production code.
