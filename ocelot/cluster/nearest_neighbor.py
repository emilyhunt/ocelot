"""A set of functions for nearest neighbor analysis of a clustering field. Especially useful for calculating optimum
DBSCAN/OPTICS epsilon parameters.
"""

def precalculate_nn_distances():
    """Pre-calculates nearest neighbor (nn) distances for direct plugging into a sklearn clustering algorithm with
    metric=pre-computed."""
    pass


def calculate_epsilon():
    """A method for calculating a number of different optimum epsilon values for a nearest neighbor field. Can also
    produce nearest neighbor plots if desired, calling functionality from ocelot.plot"""
    pass
