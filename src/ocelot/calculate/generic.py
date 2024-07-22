"""Generic calculation utilities used across the module."""
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Union
from ocelot.util.check import _check_matching_lengths_of_non_nones
from scipy.stats import directional_stats


def _weighted_standard_deviation(x: ArrayLike, weights: Optional[ArrayLike] = None):
    """Computes weighted standard deviation. Uses method from
    https://stackoverflow.com/a/52655244/12709989.
    """
    # Todo: not sure that this deals with small numbers of points correctly!
    #   See: unit test fails when v. few points used
    return np.sqrt(np.cov(x, aweights=weights))


def standard_error(
    standard_deviation: Union[ArrayLike, float, int],
    number_of_measurements: Union[ArrayLike, int],
) -> float:
    """Calculates the standard error on the mean of some parameter given the standard
    deviation.

    Parameters
    ----------
    standard_deviation : array-like, float, or int
        Standard deviation(s)
    number_of_measurements : array-like of ints, int
        Number of measurements used to find standard deviation.

    Returns
    -------
    standard_error : float
    """
    return standard_deviation / np.sqrt(number_of_measurements)


def mean_and_deviation(
    values: ArrayLike,
    weights: Optional[ArrayLike] = None,
) -> tuple[float]:
    """Calculates the mean and standard deviation of some set of values.

    Parameters
    ----------
    values : array-like
        Values to calculate mean and standard deviation of.
    weights : array-like, optional
        Array of weights to use to compute a weighted mean and average.

    Returns
    -------
    mean : float
        Mean of values.
    std : float
        Standard deviation of values.
    """
    _check_matching_lengths_of_non_nones(values, weights)

    return (
        np.average(values, weights=weights),
        _weighted_standard_deviation(values, weights),
    )


def lonlat_to_unitvec(longitudes: ArrayLike, latitudes: ArrayLike):
    """Converts longitudes and latitudes to unit vectors on a unit sphere. Uses method 
    at https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates.
    Assumes that latitudes is in the range [-pi / 2, pi / 2] as is common in 
    astronomical unit systems.
    """
    x = np.cos(latitudes) * np.cos(longitudes)
    y = np.cos(latitudes) * np.sin(longitudes)
    z = np.sin(latitudes)
    return np.column_stack((x, y, z))


def unitvec_to_lonlat(unit_vectors: ArrayLike):
    """Converts unit vectors on a unit sphere to longitudes and latitudes. See 
    `lonlat_to_unitvec` for more details.
    """
    x, y, z = [column.ravel() for column in np.hsplit(unit_vectors, 3)]
    longitudes = np.arctan2(y, x)
    latitudes  = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
    return longitudes, latitudes


def spherical_mean(longitudes: ArrayLike, latitudes: ArrayLike):
    """Calculates the spherical mean of angular positions."""
    longitudes = np.asarray_chkfinite(longitudes)
    latitudes = np.asarray_chkfinite(latitudes)

    unit_vectors = lonlat_to_unitvec(longitudes, latitudes)
    mean_unit_vector = directional_stats(unit_vectors).mean_direction
    mean_lon, mean_lat = unitvec_to_lonlat(mean_unit_vector)
    return mean_lon[0], mean_lat[0]
