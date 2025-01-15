import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ocelot.model.binaries._base import BaseBinaryStarModelWithEccentricities
from imf.distributions import BrokenPowerLaw, PowerLaw


class MoeDiStefanoMultiplicityRelation(BaseBinaryStarModelWithEccentricities):
    def __init__(self) -> None:
        """An interpolated implementation of the MoeDiStefano17 multiplicity relations,
        plus DucheneKraus+13 for stars below 1 MSun.
        """
        pass

    def multiplicity_fraction(self, masses: np.ndarray) -> np.ndarray:
        return np.interp(masses, mass, multiplicity_fraction)

    def companion_star_frequency(self, masses: np.ndarray) -> np.ndarray:
        return np.interp(masses, mass, companion_star_frequency)

    def random_mass_ratio(self, masses: np.ndarray, seed=None) -> np.ndarray:
        return self.random_binary(masses, seed=seed)[0]

    def random_binary(
        self, masses: np.ndarray, seed=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mass_ratios, log_periods, eccentricities = _sample_binary(masses, seed=seed)
        return mass_ratios, 10**log_periods, eccentricities


# The below values are hard-coded directly from the tables of the following papers.
# Stellar masses 1+ Msun: Moe & DiStefano 2017
# Below this: Duchene & Kraus 2013
mass = np.asarray([0.10, 0.30, 1.00, 3.50, 7.00, 12.5, 16.0])
multiplicity_fraction = np.asarray([0.22, 0.26, 0.40, 0.59, 0.76, 0.84, 0.94])
companion_star_frequency = np.asarray([0.22, 0.33, 0.50, 0.84, 1.30, 1.60, 2.10])


# Now, let's interpolate all of the shit for periods. Note that masses below 1 MSun
# should use the first bin, as this info is only available down to solar-like stars.
# N.B.: this is specified in terms of mass on first axis and period on the second.
mass_ratio_periods = np.asarray([1.0, 3.0, 5.0, 7.0])  # Set to bounds outside this
mass_ratio_periods_with_bounds = np.hstack([0.2, mass_ratio_periods, 8.0])

period_frequencies = np.asarray(
    [
        [0.027, 0.057, 0.095, 0.075],
        [0.07, 0.12, 0.13, 0.09],
        [0.14, 0.22, 0.20, 0.11],
        [0.19, 0.26, 0.23, 0.13],
        [0.29, 0.32, 0.30, 0.18],
    ]
)

twin_fraction = np.asarray(
    [
        [
            0.30,
            0.20,
            0.10,
            0.015,
        ],  # n.b. the first lower lim is 0.015 - feels a bit more realistic
        [0.22, 0.10, 0.015, 0.0],
        [0.17, 0.015, 0.0, 0.0],
        [0.14, 0.015, 0.0, 0.0],
        [0.08, 0.015, 0.0, 0.0],
    ]
)
gamma_large = np.asarray(
    [
        [-0.5, -0.5, -0.5, -1.1],
        [-0.5, -0.9, -1.4, -2.0],
        [-0.5, -1.7, -2.0, -2.0],
        [-0.5, -1.7, -2.0, -2.0],
        [-0.5, -1.7, -2.0, -2.0],
    ]
)
gamma_small = np.asarray(
    [
        [0.3, 0.3, 0.3, 0.3],
        [0.2, 0.1, -0.5, -1.0],
        [0.1, -0.2, -1.2, -1.5],
        [0.1, -0.2, -1.2, -1.5],
        [0.1, -0.2, -1.2, -1.5],
    ]
)
eccentricity_periods = np.asarray([2.0, 4.0])
eccentricities = np.asarray(
    [
        [0.1, 0.4],
        [0.3, 0.5],
        [0.6, 0.7],
        [0.7, 0.8],
        [0.7, 0.8],
    ]
)


_eccentricity_interpolator = RegularGridInterpolator(
    (mass[2:], eccentricity_periods), eccentricities
)


def _calculate_max_eccentricity(log_period: np.ndarray):
    period = 10**log_period
    return np.where(period > 2, 1 - (period / 2) ** (-2 / 3), 0.0)


def _get_eccentricity_parameters(
    primary_mass: np.ndarray, log_period: np.ndarray
) -> tuple[np.ndarray]:
    """Calculate the eccentricity distribution parameters from Moe & DiStefano 2017.

    Parameters
    ----------
    primary_mass : np.ndarray
        Mass ratios of binaries.
    log_period : np.ndarray
        Log of the period of binaries in days.

    Returns
    -------
    tuple of np.ndarray
        The power law slope value eta and the maximum eccentricity emax.
    """
    primary_mass_clipped = np.clip(primary_mass, 1.0, 16.0)
    log_period_clipped = np.clip(log_period, 2.0, 4.0)
    points = np.vstack([primary_mass_clipped, log_period_clipped]).T
    return _eccentricity_interpolator(points), _calculate_max_eccentricity(log_period)


_gamma_large_interpolator = RegularGridInterpolator(
    (mass[2:], mass_ratio_periods), gamma_large
)
_gamma_small_interpolator = RegularGridInterpolator(
    (mass[2:], mass_ratio_periods), gamma_small
)
_twin_fraction_interpolator = RegularGridInterpolator(
    (mass[2:], mass_ratio_periods), twin_fraction
)


def _get_mass_ratio_distribution_parameters(
    primary_mass: np.ndarray, log_period: np.ndarray
):
    """Calculate the mass ratio distribution parameters from Moe & DiStefano 2017.

    Parameters
    ----------
    primary_mass : np.ndarray
        Mass ratios of binaries.
    log_period : np.ndarray
        Log of the period of binaries in days.

    Returns
    -------
    tuple of np.ndarray
        Values of gamma large, gamma small, and the twin fraction.
    """
    primary_mass_clipped = np.clip(primary_mass, 1.0, 16.0)
    log_period_clipped = np.clip(log_period, 1.0, 7.0)
    points = np.vstack([primary_mass_clipped, log_period_clipped]).T
    return (
        _gamma_large_interpolator(points),
        _gamma_small_interpolator(points),
        _twin_fraction_interpolator(points),
    )


# The period interpolator is set up so that it has extra flat regions from 0.2 to 1.0 and from 7.0 to 8.0 (as in the original paper)
_period_interpolator = RegularGridInterpolator(
    (mass[2:], mass_ratio_periods_with_bounds),
    np.hstack(
        (period_frequencies[:, :1], period_frequencies, period_frequencies[:, -1:])
    ),
    bounds_error=False,
    fill_value=0.0,
)


def _get_period_pdf(
    primary_mass: np.ndarray, resolution: int = 100
) -> tuple[np.ndarray]:
    """Calculate the period pdf distribution from Moe & DiStefano 2017.

    Parameters
    ----------
    primary_mass : np.ndarray
        Mass ratios of binaries.

    Returns
    -------
    np.ndarray
        The period PDF for each primary mass, defined for periods from 0.2 to 8.0.
    """
    # periods = mass_ratio_periods_with_bounds
    periods = np.linspace(0.2, 8.0, num=resolution)
    primary_mass_clipped = np.clip(primary_mass, 1.0, 16.0)
    n_masses, n_periods = len(primary_mass_clipped), len(periods)

    points = np.vstack(
        [
            np.repeat(primary_mass_clipped, n_periods),
            np.tile(periods, n_masses),
        ]
    ).T
    values = _period_interpolator(points).reshape(n_masses, n_periods)

    # Normalize to unit area
    values = values / np.trapz(values, np.tile(periods, (n_masses, 1)), axis=1).reshape(
        n_masses, 1
    )

    # Ensure that the bounds are properly defined (i.e. values from 0.2 to 8.0 allowed)
    # values = np.hstack((values[:1], values, values[-1:]))
    return periods, values


def _pdf_to_cdf(x: np.ndarray, pdf: np.ndarray):
    """Converts a piecewise PDF to a CDF. Assumes that the starting value is the
    lower bound, i.e. the CDF below x[0] should be zero.
    """
    # Deal with 1D input
    if len(pdf.shape) == 1:
        pdf = pdf.reshape(1, -1)

    # Calculate CDF
    delta_x = np.diff(x).reshape(1, -1)
    cdf = np.cumsum((pdf[:, :-1] + pdf[:, 1:]) / 2 * delta_x, axis=1)

    # Ensure lower bound defined, i.e. we start at zero
    cdf = np.hstack((np.zeros((pdf.shape[0], 1)), cdf))

    # Set to range [0, 1]
    cdf = cdf / cdf[:, -1].reshape(-1, 1)

    return cdf


def _get_period_percentile_point_function(
    primary_mass: np.ndarray, resolution: int = 100
) -> tuple[np.ndarray]:
    x_periods, period_pdf = _get_period_pdf(primary_mass, resolution=resolution)
    period_cdf = _pdf_to_cdf(x_periods, period_pdf)
    return period_cdf, x_periods


def _sample_period(
    primary_mass: np.ndarray[float | int] | float | int, seed=None
) -> np.ndarray:
    """Sample log period as a function of primary star mass."""
    rng = np.random.default_rng(seed)
    primary_mass = np.atleast_1d(primary_mass).astype(float)
    period_ppf, period_values = _get_period_percentile_point_function(primary_mass)
    uniform_deviates = rng.uniform(size=len(primary_mass))
    periods = np.zeros_like(primary_mass)
    for i in range(len(periods)):
        periods[i] = np.interp(
            uniform_deviates[i], period_ppf[i], period_values, 0.2, 8.0
        )
    return periods


def _sample_mass_ratio(
    primary_mass: np.ndarray[float | int],
    log_period: np.ndarray[float | int],
    seed=None,
):
    """Samples binary star mass ratios."""
    gamma_large, gamma_small, twin_fraction = _get_mass_ratio_distribution_parameters(
        primary_mass, log_period
    )

    n_stars = len(gamma_large)
    if n_stars == 0:
        return np.atleast_1d([])
    rng = np.random.default_rng(seed)
    mass_ratios = np.zeros_like(gamma_large)

    # Assign some as twins
    is_twin = rng.uniform(size=n_stars) < twin_fraction
    not_twin = np.invert(is_twin)
    n_twins = is_twin.sum()
    mass_ratios[is_twin] = rng.uniform(low=0.95, high=1.0, size=n_twins)
    if n_twins == n_stars:
        return mass_ratios

    # Assign the rest from the power law
    # Todo seeds are ignored by imf.distribution
    gamma_large, gamma_small = gamma_large[not_twin], gamma_small[not_twin]
    power_laws = [
        BrokenPowerLaw([gamma_small[i], gamma_large[i]], [0.1, 0.3, 1.0])
        for i in range(len(gamma_large))
    ]
    mass_ratios[not_twin] = np.hstack([dist.rvs(1) for dist in power_laws])
    return mass_ratios


def _sample_eccentricity(
    primary_mass: np.ndarray[float | int],
    log_period: np.ndarray[float | int],
    seed=None,
):
    power_law_slopes, max_eccentricity = _get_eccentricity_parameters(
        primary_mass, log_period
    )

    # Define where we'll be sampling. We set an eccentricity_min that's non-zero as
    # power laws are undefined at zero. Whenever the max_eccentricity is less than
    # this minimum value, we just assume zero eccentricity (it basically is anyway)
    eccentricity_min = 1e-10
    good_max = max_eccentricity > eccentricity_min

    # Sample some eccentricities!
    # Todo seeds are ignored by imf.distribution
    eccentricities = np.zeros_like(primary_mass)
    power_laws = [
        PowerLaw(a_slope, eccentricity_min, a_max)
        for a_slope, a_max in zip(
            power_law_slopes[good_max], max_eccentricity[good_max]
        )
    ]

    # Handle case where we have no objects at all
    if len(power_laws) == 0:
        return np.atleast_1d([])

    eccentricities[good_max] = np.hstack([dist.rvs(1) for dist in power_laws])
    return eccentricities


def _sample_binary(primary_mass: np.ndarray[float | int] | float | int, seed=None):
    """Returns binary star mass ratios and periods."""
    primary_mass = np.atleast_1d(primary_mass).astype(float)
    log_period = _sample_period(primary_mass, seed=seed)
    mass_ratios = _sample_mass_ratio(primary_mass, log_period, seed=seed)
    eccentricities = _sample_eccentricity(primary_mass, log_period)
    return mass_ratios, log_period, eccentricities
