"""Metrics for use with Gaia data in computing distance metrics between stars in more sophisticated ways."""

import numpy as np


def five_parameter_error_aware_distance(x, y):
    """Data features should be in the order:

    0: ra
    1: dec
    2: pmra
    3: pmra_error
    4: pmdec
    5: pmdec_error
    6: distance
    7: distance_error

    """
    # Grab things we need
    mean_distance = (x[6] + y[6]) / 2

    pmra_error = np.sqrt(x[3]**2 + y[3]**2)
    pmdec_error = np.sqrt(x[5]**2 + y[5]**2)
    distance_error = np.sqrt(x[7]**2 + y[7]**2)

    # Calculate constants array and normalise it
    constants = np.asarray([1 / mean_distance,
                            1 / mean_distance,
                            1 / mean_distance / pmra_error,
                            1 / mean_distance / pmdec_error,
                            1 / distance_error])

    constants /= np.sum(constants)

    # Return time!
    return np.sqrt((x[0] - y[0])**2 * constants[0]
                   + (x[1] - y[1])**2 * constants[1]
                   + (x[2] - y[2])**2 * constants[2]
                   + (x[4] - y[4])**2 * constants[3]
                   + (x[6] - y[6])**2 * constants[4])
