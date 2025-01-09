"""This file contains an implementation of the King+1962 empirical star cluster
model: https://ui.adsabs.harvard.edu/abs/1962AJ.....67..471K/abstract

In addition, some of the equations from the paper are also implemented directly as
methods in this file.
"""
# Todo tidy old methods

import numpy as np

from ocelot.model.distribution._common import spherical_to_cartesian
from ._base import (
    BaseClusterDistributionModel,
    Implements1DMethods,
    Implements2DMethods,
    Implements3DMethods,
)
from ocelot.util.random import points_on_sphere
from astropy import units as u
from astropy.units import Quantity
from scipy.optimize import minimize


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
        if dimensions != 3:
            raise NotImplementedError(
                "Other levels of dimensionality are coming soon. Sorry!"
            )
        self.r_core = r_core
        self.r_tidal = r_tidal
        self._r_50 = None
        self.dimensions = dimensions

    def _check_input(self, r_core: Quantity, r_tidal: Quantity, dimensions: int):
        """Checks validity of input parameters."""
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
        model).
        """
        if self._r_50 is None:
            self._calculate_r_50()
        return self._r_50

    def _calculate_r_50(self):
        """Calculates & returns the half-light radius of the cluster using method from
        Hunt+23 - i.e., it finds the radius r_50 at which the King number density is
        half that of r_tidal.
        """
        total_value = king_number_density(self.r_tidal, self.r_core, self.r_tidal)
        target_value = total_value / 2

        def func_to_minimise(r):
            return (
                target_value - king_number_density(r, self.r_core, self.r_tidal)
            ) ** 2

        result = minimize(
            func_to_minimise,
            np.atleast_1d([self.r_core]),
            method="Nelder-Mead",
            bounds=((0.0, self.r_tidal),),
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

    def _rvs(self, size: int, seed=None) -> np.ndarray[float]:
        match self.dimensions:
            case 3:
                return np.vstack(
                    sample_3d_king_profile(self.r_core, self.r_tidal, size, seed=seed)
                ).T
        raise ValueError("Number of dimensions not supported!")


def king_surface_density(
    r_values: float | np.ndarray,
    r_core: float,
    r_tidal: float,
    normalize: bool = False,
):
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
    a = (1 + rt_rc**2) ** -0.5

    # Compute normalisation constant if desired
    if normalize:
        term_1 = r_core * np.arctan(rt_rc)
        term_2 = -2 * a * r_core * np.log(rt_rc + 1 / a)
        term_3 = r_tidal * a**2
        normalisation_constant = 1 / (term_1 + term_2 + term_3)
    else:
        normalisation_constant = 1.0

    # Work out where  0 <= r < r_tidal
    r_valid = np.logical_and(r_values >= 0, r_values < r_tidal)
    r_invalid = np.invert(r_valid)

    # Compute result
    reduced_r_values = (1 + (r_values[r_valid] / r_core) ** 2) ** (-0.5)
    result = np.empty(r_values.shape)
    result[r_valid] = normalisation_constant * (reduced_r_values - a) ** 2
    result[r_invalid] = 0

    return result


def king_number_density(r, r_core, r_tidal, k=1):
    """Calculates the King1962 number density (eqn 18 in the paper.)

    Unnormalised by default (i.e. k=1.)
    """
    x = (r / r_core) ** 2
    x_t = (r_tidal / r_core) ** 2

    term_1 = np.log(1 + x)
    term_2 = -4 * ((1 + x) ** (0.5) - 1) / (1 + x_t) ** (0.5)
    term_3 = x / (1 + x_t)

    return np.pi * r_core**2 * k * (term_1 + term_2 + term_3)


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
    a = (1 + rt_rc**2) ** -0.5

    # Compute result
    return ((1 + (r_values / r_core) ** 2) ** (-0.5) - a) ** 2


def sample_2d_king_profile(
    r_core: float,
    r_tidal: float,
    n_samples: int,
    seed=None,
    oversampling_factor: float = 10,
    return_generator: bool = False,
):
    """Samples a 2D King profile to return n_samples sample radii.

    Valid only for:
        0 < r_core < r_tidal

    Args:
        r_core (float): the core radius of the cluster.
        r_tidal (float): the tidal radius of the cluster.
        n_samples (int): the number of samples to generate.
        seed (int, optional): the seed of the random number generator. Default: None.
        oversampling_factor (float): how many times n_samples to generate each step, which helps to make sure that
            enough samples are quickly generated in just one or two loops. Default: 10.
        return_generator (bool): whether or not to return the numpy random number generator created. Default: False

    Returns:
        an array of sample radii of size n_samples, plus the random generator if return_generator==True.
    """
    _check_core_and_tidal_radii(r_core, r_tidal)

    r_samples = np.empty(n_samples, dtype=float)
    generator = np.random.default_rng(seed=seed)
    completed_samples = 0

    max_value = king_surface_density_fast(np.zeros(1), r_core, r_tidal)[0]

    while completed_samples < n_samples:
        remaining_samples = n_samples - completed_samples
        samples_to_generate = int(
            np.clip(remaining_samples * oversampling_factor, 10, np.inf)
        )

        # Generate some initial radius samples
        test_r_values = generator.uniform(high=r_tidal, size=samples_to_generate)
        test_king_values = king_surface_density_fast(test_r_values, r_core, r_tidal)
        test_mcmc_values = generator.uniform(size=samples_to_generate, high=max_value)

        # See which & how many are valid and save them!
        valid_test_samples = test_king_values < test_mcmc_values
        n_valid_test_samples = np.count_nonzero(valid_test_samples)

        if n_valid_test_samples > remaining_samples:
            r_samples[completed_samples:] = test_r_values[valid_test_samples][
                :remaining_samples
            ]
            break

        r_samples[completed_samples : completed_samples + n_valid_test_samples] = (
            test_r_values[valid_test_samples]
        )
        completed_samples += n_valid_test_samples

    if return_generator:
        return r_samples, generator
    return r_samples


def sample_1d_king_profile(
    r_core: float,
    r_tidal: float,
    n_samples: int,
    seed: int = None,
    oversampling_factor: float = 10,
    return_generator: bool = False,
):
    """Samples a 1D King profile (e.g. useful to get line of sight distances from the center of a cluster.) Uses a
    little trick - assumes that we're looking at the cluster side-on and removes the not-line-of-sight axis as if we
    were looking at it from the front. (This is because I cba to work out a 1D profile and more to the point, I (Emily Hunt) couldn't
    find one...)

    Valid only when:
        0 < r_core < r_tidal

    Args:
        r_core (float): the core radius of the cluster.
        r_tidal (float): the tidal radius of the cluster.
        n_samples (int): the number of samples to generate.
        seed (int, optional): the seed of the random number generator. Default: None.
        oversampling_factor (float): how many times n_samples to generate each step, which helps to make sure that
            enough samples are quickly generated in just one or two loops. Default: 10.

    Returns:
        an array of sample radii of size n_samples
    """
    # Todo: change this to using the strip density g(x), which I *think* could do this better...
    r_samples, generator = sample_2d_king_profile(
        r_core,
        r_tidal,
        n_samples,
        seed=seed,
        oversampling_factor=oversampling_factor,
        return_generator=True,
    )

    # Now, deproject this to 1D by giving every value an angle and then finding the 1D radius with the cosine
    random_angles = generator.uniform(high=2 * np.pi, size=n_samples)
    r_samples = r_samples * np.cos(random_angles)

    if return_generator:
        return r_samples, generator
    return r_samples


def sample_3d_king_profile(
    r_core: float,
    r_tidal: float,
    n_samples: int,
    seed: int = None,
    oversampling_factor: float = 10,
):
    """Samples a 1D King profile (e.g. useful to get line of sight distances from the center of a cluster.) Uses a
    little trick - assumes that we're looking at the cluster side-on and removes the not-line-of-sight axis as if we
    were looking at it from the front. (This is because I cba to work out a 1D profile and more to the point, I couldn't
    fucking find one)

    Todo: change this to using the strip density g(x), which I *think* could do this better...

    Valid only when:
        0 < r_core < r_tidal

    Args:
        r_core (float): the core radius of the cluster.
        r_tidal (float): the tidal radius of the cluster.
        n_samples (int): the number of samples to generate.
        seed (int, optional): the seed of the random number generator. Default: None.
        oversampling_factor (float): how many times n_samples to generate each step, which helps to make sure that
            enough samples are quickly generated in just one or two loops. Default: 10.

    Returns:
        an array of sample radii of size (n_samples, 3)
    """
    r_samples, generator = sample_1d_king_profile(
        r_core,
        r_tidal,
        n_samples,
        seed=seed,
        oversampling_factor=oversampling_factor,
        return_generator=True,
    )
    phis, thetas = points_on_sphere(len(r_samples), phi_symmetric=False, seed=seed)
    x_values, y_values, z_values = spherical_to_cartesian(r_samples, thetas, phis)
    return x_values, y_values, z_values
