"""A package for simulating star clusters."""

from .cluster import (
    SimulatedCluster,  # noqa: F401
    SimulatedClusterParameters,  # noqa: F401
    SimulatedClusterModels,  # noqa: F401
)
import warnings
from ocelot import DATA_PATH


if not DATA_PATH.exists():
    warnings.warn(
        f"Path to data at {DATA_PATH} does not exist, and ocelot.simulate will not "
        "work. Did you manually download data for the package? (Sorry - this will be "
        "automated in the future!)"
    )
