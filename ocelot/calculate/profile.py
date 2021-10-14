from typing import Union

import numpy as np


def king_surface_density(r_values: Union[float, np.ndarray], r_core: float, r_tidal: float, normalise: bool = False):
    """Computes the King surface density (King 1962, equation 14) given the three parameters. Can take vectorised input.
    Will return the surface density per square unit expected at a distance r_values from the core of the cluster.

    Valid only for:
        0 <= r < r_tidal (0 elsewhere - this is checked internally)
        0 < r_core < r_tidal (raises an error if not the case)

    Args:
        r_values (float or np.ndarray): r values to compute the surface density at.
        r_core (float): the core radius of the cluster.
        r_tidal (float): the tidal radius of the cluster.
        normalise (bool): whether or not to return the normalised King surface density profile, calculated
            numerically.
            Default: False

    Returns:
        a float or array of floats of the surface density for the cluster.

    """
    _check_core_and_tidal_radii(r_core, r_tidal)
    r_values = _check_r_values(r_values)

    # Constants
    rt_rc = r_tidal / r_core
    a = (1 + rt_rc ** 2) ** -0.5

    # Compute normalisation constant if desired
    if normalise:
        term_1 = r_core * np.arctan(rt_rc)
        term_2 = -2 * a * r_core * np.log(rt_rc + 1 / a)
        term_3 = r_tidal * a ** 2
        normalisation_constant = 1 / (term_1 + term_2 + term_3)
    else:
        normalisation_constant = 1.

    # Work out where  0 <= r < r_tidal
    r_valid = np.logical_and(r_values >= 0, r_values < r_tidal)
    r_invalid = np.invert(r_valid)

    # Compute result
    reduced_r_values = (1 + (r_values[r_valid] / r_core) ** 2) ** (-0.5)
    result = np.empty(r_values.shape)
    result[r_valid] = normalisation_constant * (reduced_r_values - a) ** 2
    result[r_invalid] = 0

    return result


def _check_r_values(r_values):
    # Convert to a np array that's at least 1d, and check that it isn't bad
    r_values = np.atleast_1d(r_values)
    if not np.isfinite(r_values).all():
        raise ValueError("invalid r_values (e.g. nan or inf) are not allowed!")
    if np.any(r_values < 0):
        raise ValueError("all r_values must be positive or zero.")
    return r_values


def _check_core_and_tidal_radii(r_core, r_tidal):
    if not np.isfinite(r_core).all() or not np.isfinite(r_tidal).all():
        raise ValueError("input parameters must be finite!")
    if r_tidal < r_core or r_core < 0 or r_tidal < 0:
        raise ValueError("parameters must satisfy 0 < r_core < r_tidal")


def king_surface_density_fast(r_values: np.ndarray, r_core: float, r_tidal: float):
    """Computes the King surface density (King 1962, equation 14) given the three parameters. Can take vectorised input.
    Fast version intended for use with random sampling - it has no checks and cannot be normalised! Be careful!

    Valid only for:
        0 <= r < r_tidal (NOT CHECKED in this function!)
        0 < r_core < r_tidal (NOT CHECKED in this function!)

    Args:
        r_values (np.ndarray): r values to compute the surface density at. Must be a numpy array!
        r_core (float): the core radius of the cluster.
        r_tidal (float): the tidal radius of the cluster.

    Returns:
        a float or array of floats of the surface density for the cluster.

    """
    # Constants
    rt_rc = r_tidal / r_core
    a = (1 + rt_rc ** 2) ** -0.5

    # Compute result
    return ((1 + (r_values / r_core) ** 2) ** (-0.5) - a) ** 2


def sample_king_profile(r_core: float, r_tidal: float, n_samples: int):
    """Samples a 2D King profile to return n_samples sample radii."""
    # Fuck I realise sampling the 2D profile isn't actually useful for why I'm writing this code right now, so I'll
    # leave it for now lol (I need to sample just 1D distances from the core...)
    pass
