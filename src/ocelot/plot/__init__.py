from . import axis
from . import process
from . import utilities
from .plot_figure import clustering_result
from .plot_figure import location
from .plot_figure import nearest_neighbor_distances
from .gaia_explorer import GaiaExplorer, ion, ioff
import warnings
warnings.warn(
    "ocelot.plot API will change soon. Most objects will move, change, or be "
    "deprecated.",
    DeprecationWarning,
)
