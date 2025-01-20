"""This file contains an implementation of the King+1962 empirical star cluster
model: https://ui.adsabs.harvard.edu/abs/1962AJ.....67..471K/abstract

In addition, some of the equations from the paper are also implemented directly as
methods in this file.
"""

import numpy as np
from numba import jit

from ._base import (
    BaseClusterDistributionModel,
    # Implements1DMethods,
    # Implements2DMethods,
    Implements3DMethods,
)
from astropy import units as u
from astropy.units import Quantity
from scipy.optimize import minimize
from scipy.interpolate import interp1d


class King62(
    BaseClusterDistributionModel,
    # Implements1DMethods,
    # Implements2DMethods,
    Implements3DMethods,
):
    def __init__(self, r_core: Quantity, r_tidal: Quantity, dimensions: int = 3):
        """Implementation of the King+1962 [1] empirical star cluster model.

        Methods are available in 1D, 2D, and 3D form.

        Parameters
        ----------
        r_core : Quantity
            Core radius of the cluster. Must be defined with astropy units. May be in
            angular units when dimensions is 1 or 2.
        r_tidal : Quantity
            Tidal radius of the cluster. Must be defined with astropy units. May be in
            angular units when dimensions is 1 or 2.
        dimensions : int
            Number of dimensions of the model - must be 1, 2, or 3.
            In 1D mode, radii from the centre of the cluster are returned.
            In 2D mode, X/Y coordinates to the centre of the cluster are returned.
            In 3D mode, X/Y/Z coordinates to the centre of the cluster are returned.
            Angles are not supported in 3D mode.
            Default: 3

        References
        ----------
        [1] https://ui.adsabs.harvard.edu/abs/1962AJ.....67..471K/abstract
        """
        self._check_input(r_core, r_tidal, dimensions)
        # Todo implement 2D and 1D versions (easy)
        # Todo maybe I should have 2D spherical? And maybe even 3D spherical? And not just 2D/3D cartesian.
        if dimensions != 3:
            raise NotImplementedError(
                "Other levels of dimensionality are coming soon. Sorry!"
            )
        self.r_core: Quantity = r_core
        self.r_tidal: Quantity = r_tidal

        # Dimensionless versions - important for numpy interface
        self._r_core: float = self.r_core.to(u.pc).value
        self._r_tidal: float = self.r_tidal.to(u.pc).value
        self._r_50: float | None = None
        self.dimensions: int = dimensions

    def _check_input(self, r_core: Quantity, r_tidal: Quantity, dimensions: int):
        """Checks validity of input parameters."""
        if not isinstance(r_core, Quantity) or not isinstance(r_tidal, Quantity):
            raise ValueError(
                "Core and tidal radius must be specified as astropy Quantity objects!"
            )
        if r_core.unit != r_tidal.unit:
            raise ValueError(
                "Unable to pick best unit: r_core and r_tidal must have same input "
                f"unit, but instead have units {r_core.unit} and {r_tidal.unit}."
            )
        if r_core >= r_tidal:
            raise ValueError("r_core may not be greater than r_tidal!")
        if r_core < 0:
            raise ValueError("r_core must be positive!")
        if r_tidal < 0:
            raise ValueError("r_tidal must be positive!")
        if dimensions not in {1, 2, 3}:
            raise ValueError("dimensionality of the model must be 1, 2, or 3.")

    @property
    def r_50(self):
        """Median radius of the cluster (equivalent to half-light and half-mass for this
        model), *in on-sky units*.
        """
        # Todo: add option to get 3D r_50.
        if self._r_50 is None:
            self._calculate_r_50()
        return self._r_50 * self.base_unit

    @property
    def base_unit(self):
        return self.r_tidal.unit

    def _calculate_r_50(self):
        """Calculates & returns the half-light radius of the cluster using method from
        Hunt+23 - i.e., it finds the radius r_50 at which the King number density is
        half that of r_tidal.
        """
        total_value = king_number_density(self._r_tidal, self._r_core, self._r_tidal)
        target_value = total_value / 2

        def func_to_minimise(r):
            return (
                target_value - king_number_density(r, self._r_core, self._r_tidal)
            ) ** 2

        result = minimize(
            func_to_minimise,
            np.atleast_1d([self._r_core]),
            method="Nelder-Mead",
            bounds=((0.0, self._r_tidal),),
        )

        if not result.success:
            raise RuntimeError(
                f"unable to find an r_50 value given r_core={self.r_core} and "
                f"r_tidal={self.r_tidal}"
            )

        self._r_50 = result.x[0]

    def _pdf(self, x: np.ndarray[float]) -> np.ndarray[float]:
        # Todo implement PDF
        raise NotImplementedError("Sorry! This class is a work in progress.")

    def _cdf(self, x: np.ndarray[float]) -> np.ndarray[float]:
        # Todo implement CDF
        raise NotImplementedError("Sorry! This class is a work in progress.")

    def _rvs(
        self, size: int, seed: None | int | np.random.Generator = None
    ) -> np.ndarray[float]:
        match self.dimensions:
            case 3:
                return sample_3d_king_profile(
                    self._r_core, self._r_tidal, size, seed=seed
                )
        raise ValueError("Number of dimensions not supported!")


def king_surface_density(
    radius: float | np.ndarray,
    r_core: float | np.ndarray,
    r_tidal: float | np.ndarray,
    k: float | np.ndarray = 1.0,
    normalize: bool = False,
) -> np.ndarray:
    """Compute the King1962 cluster surface density, i.e. the density of stars per unit
    area in 2D on-sky coordinates.

    Parameters
    ----------
    radius : float | np.ndarray
        Radius values to compute the surface density at.
    r_core : float | np.ndarray
        Cluster core radius. Must be positive and less than r_tidal.
    r_tidal : float | np.ndarray
        Cluster tidal radius. Must be positive and greater than r_core.
    k : float | np.ndarray, optional
        Normalization constant. Default: 1.0
    normalize : bool, optional
        Whether or not to return the normalized surface density, turning this into an
        estimate of the 2D PDF of the cluster on-sky. Default: False

    Returns
    -------
    np.ndarray
        Surface density values at the requested radius/r_core/r_tidal values.
    """
    _check_inputs(radius, r_core, r_tidal, k)
    radius = np.atleast_1d(radius)

    # Constants
    rt_rc = r_tidal / r_core
    a = (1 + rt_rc**2) ** -0.5

    # Compute normalisation constant if desired
    # Todo this math is unpublished but should be, also pretty sure I lost the derivation...
    if normalize:
        term_1 = r_core * np.arctan(rt_rc)
        term_2 = -2 * a * r_core * np.log(rt_rc + 1 / a)
        term_3 = r_tidal * a**2
        normalisation_constant = 1 / (term_1 + term_2 + term_3)
    else:
        normalisation_constant = k

    # Work out where  0 <= r < r_tidal
    r_valid = np.logical_and(radius >= 0, radius < r_tidal)
    r_invalid = np.invert(r_valid)

    # Compute result
    reduced_r_values = (1 + (radius[r_valid] / r_core) ** 2) ** (-0.5)
    result = np.empty(radius.shape)
    result[r_valid] = normalisation_constant * (reduced_r_values - a) ** 2
    result[r_invalid] = 0

    return result


def king_number_density(
    radius: float | np.ndarray,
    r_core: float | np.ndarray,
    r_tidal: float | np.ndarray,
    k: float | np.ndarray = 1,
    cumulative=False,
) -> np.ndarray:
    """Calculates the King1962 number density (Eqn. 18 in the paper.)

    Unnormalised by default (i.e. k=1.)

    Returns cumulative distribution function for cumulative=True.


    Parameters
    ----------
    radius : float | np.ndarray
        Radius values to compute the surface density at.
    r_core : float | np.ndarray
        Cluster core radius. Must be positive and less than r_tidal.
    r_tidal : float | np.ndarray
        Cluster tidal radius. Must be positive and greater than r_core.
    k : float | np.ndarray, optional
        Normalization constant. Default: 1.0
    cumulative : bool, optional
        Whether or not to return the cumulative number density. Doing so means that
        this function will instead estimate the cumulative density function of the
        King number density. Default: False.

    Returns
    -------
    np.ndarray
        Number density values at the requested radius/r_core/r_tidal values. If
        cumulative is True, this will instead be an array of estimates of the CDF of the
        cluster's number density at a given radius.
    """
    _check_inputs(radius, r_core, r_tidal, k)
    radius = np.atleast_1d(radius)
    x = (radius / r_core) ** 2
    x_t = (r_tidal / r_core) ** 2

    term_1 = np.log(1 + x)
    term_2 = -4 * ((1 + x) ** (0.5) - 1) / (1 + x_t) ** (0.5)
    term_3 = x / (1 + x_t)

    result = np.pi * r_core**2 * k * (term_1 + term_2 + term_3)
    if cumulative:
        result = result / king_number_density(r_tidal, r_core, r_tidal)
    return result


def _check_inputs(radius: float, r_core: float, r_tidal: float, k: float | None = None):
    """Checks that inputs to standard King functions are correct."""
    _check_core_and_tidal_radii(r_core, r_tidal)
    _check_values(radius)
    if k is not None:
        _check_values(k, name="k")


def _check_values(values, name="radius"):
    # Convert to a np array that's at least 1d, and check that it isn't bad
    values = np.atleast_1d(values)
    if not np.isfinite(values).all():
        raise ValueError(f"invalid {name} values (e.g. nan or inf) are not allowed!")
    if np.any(values < 0):
        raise ValueError(f"all {name} values must be positive or zero.")
    return values


def _check_core_and_tidal_radii(r_core, r_tidal):
    if not np.isfinite(r_core).all() or not np.isfinite(r_tidal).all():
        raise ValueError("input parameters must be finite!")
    if r_tidal < r_core or r_core < 0 or r_tidal < 0:
        raise ValueError("parameters must satisfy 0 < r_core < r_tidal")


def sample_2d_king_profile(
    r_core: float,
    r_tidal: float,
    n_samples: int,
    seed: None | int | np.random.Generator = None,
    return_generator: bool = False,
    resolution: int = 500,
) -> np.ndarray:
    """Samples a 2D King profile to return n_samples sample radii.

    Parameters
    ----------
    r_core : float | np.ndarray
        Cluster core radius. Must be positive and less than r_tidal.
    r_tidal : float | np.ndarray
        Cluster tidal radius. Must be positive and greater than r_core.
    n_samples : int
        Number of samples to return.
    seed : None | int | np.random.Generator, optional
        Seed for the random generator, or a np.random.generator to use for sampling.
        Default: None
    resolution : int, optional
        Resolution of the internal estimate of the percentile point function of the
        cluster's number density function. (Wow - that's a mouthful.). Default: 500,
        which should be high enough to be accurate in all cases.

    Returns
    -------
    np.ndarray
        Radius samples.

    Notes
    -----
    To turn these radii into 2D random coordinates, assign each radius a random angle
    between 0 and 2 pi.
    """
    generator = np.random.default_rng(seed=seed)

    r_values = np.linspace(0, r_tidal, num=resolution)
    cumulative_density_function = king_number_density(
        r_values, r_core, r_tidal, cumulative=True
    )
    percentile_point_function = interp1d(cumulative_density_function, r_values)

    r_samples = percentile_point_function(generator.uniform(size=n_samples))

    return r_samples


def king_spatial_density(
    radius: float | np.ndarray,
    r_core: float | np.ndarray,
    r_tidal: float | np.ndarray,
    k: float | np.ndarray = 1,
):
    _check_values(radius)
    _check_core_and_tidal_radii(r_core, r_tidal)
    out = np.zeros_like(radius)
    good_radius = radius <= r_tidal
    out[good_radius] = _king_spatial_density_inner(
        radius[good_radius], r_core, r_tidal, k
    )
    return out


@jit(nopython=True, cache=True)
def _king_spatial_density_inner(
    radius: float | np.ndarray,
    r_core: float | np.ndarray,
    r_tidal: float | np.ndarray,
    k: float | np.ndarray = 1.0,
):
    rt_rc = 1 + (r_tidal / r_core) ** 2
    z = ((1 + (radius / r_core) ** 2) / rt_rc) ** (1 / 2)

    part_1 = k / (np.pi * r_core * rt_rc ** (3 / 2))
    part_2 = 1 / z**2
    part_3 = 1 / z * np.arccos(z) - (1 - z**2) ** (1 / 2)

    return part_1 * part_2 * part_3


@jit(nopython=True, cache=True)
def _king_spatial_density_one_val_numba(
    radius: float, r_core: float, r_tidal: float, k: float = 1.0
):
    if radius > r_tidal:
        return 0.0
    return _king_spatial_density_inner(radius, r_core, r_tidal, k=k)


def sample_3d_king_profile(
    r_core: float,
    r_tidal: float,
    n_samples: int,
    seed: None | int | np.random.Generator = None,
):
    """Sample random star locations in a King1962 model and return them in 3D Cartesian
    coordinates.

    This method uses rejection sampling, and is a bit slower than sampling in 2D. If you
    only need 2D positions, then it's easier to use sample_2d_king_profile.

    Parameters
    ----------
    r_core : float | np.ndarray
        Cluster core radius. Must be positive and less than r_tidal.
    r_tidal : float | np.ndarray
        Cluster tidal radius. Must be positive and greater than r_core.
    n_samples : int
        Number of samples to return.
    seed : None | int | np.random.Generator, optional
        Seed for the random generator, or a np.random.generator to use for sampling.
        Default: None

    Returns
    -------
    np.ndarray
        An array of cluster location samples with shape (n_samples, 3).
    """
    _check_core_and_tidal_radii(r_core, r_tidal)

    if seed is None:
        seed = np.random.default_rng().integers(2**32 - 1)
    if isinstance(seed, np.random.Generator):
        seed = seed.integers(2**32 - 1)

    return _sample_king_spatial_density_numba(r_core, r_tidal, n_samples, seed)


@jit(nopython=True, cache=True)
def _sample_king_spatial_density_numba(
    r_core: float, r_tidal: float, n_samples: int, seed: int
):
    """Optimized rejection sampling to sample 3D spatial coordiantes of a King62 model."""
    # Set seed (this is numba-only)
    np.random.seed(seed)

    # Calculate max value
    max_value = _king_spatial_density_one_val_numba(0.0, r_core, r_tidal)

    # Loop, each time doing rejection sampling with some random coordinates
    out = np.empty((n_samples, 3))
    i = 0
    while i < n_samples:
        random_coordinates = np.random.uniform(low=-r_tidal, high=r_tidal, size=3)
        random_radius = (
            random_coordinates[0] ** 2
            + random_coordinates[1] ** 2
            + random_coordinates[2] ** 2
        ) ** (1 / 2)
        trial_value = np.random.uniform(low=0.0, high=max_value)
        king_value = _king_spatial_density_one_val_numba(random_radius, r_core, r_tidal)

        if trial_value < king_value:
            out[i] = random_coordinates
            i += 1

    return out
