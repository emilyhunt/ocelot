import numpy as np
from scipy.interpolate import RegularGridInterpolator


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

_period_interpolator = RegularGridInterpolator(
    (mass[2:], mass_ratio_periods), period_frequencies
)
_gamma_large_interpolator = RegularGridInterpolator(
    (mass[2:], mass_ratio_periods), gamma_large
)
_gamma_small_interpolator = RegularGridInterpolator(
    (mass[2:], mass_ratio_periods), gamma_small
)
_twin_fraction_interpolator = RegularGridInterpolator(
    (mass[2:], mass_ratio_periods), twin_fraction
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


def _get_period_pdf(primary_mass: np.ndarray) -> np.ndarray:
    """Calculate the period pdf distribution from Moe & DiStefano 2017.

    Parameters
    ----------
    primary_mass : np.ndarray
        Mass ratios of binaries.

    Returns
    -------
    np.ndarray
        The period PDF for each primary mass, defined for periods from 0.5 to 8.0.
    """
    primary_mass_clipped = np.clip(primary_mass, 1.0, 16.0)
    points = np.vstack([primary_mass_clipped, mass_ratio_periods])
    values = _period_interpolator(points).reshape(points.shape)

    # Ensure that the bounds are properly defined (i.e. values from 0.5 to 8.0 allowed)
    values = np.hstack((values[:1], values, values[-1:]))
    return values


# def _get_period_percentile_point_function(primary_mass: np.ndarray) -> np.ndarray:
#     period_pdf = _get_period_pdf(primary_mass)
#     period_ecdf = 


# def _sample_period(seed=None):
#     rng = np.random.default_rng(seed)


# class MoeDiStefanoMultiplicityRelation:
#     def __init__(
#         self,
#         make_resolved_stars=True,
#         distance=None,
#         separation=0.6,
#         interpolated_q=True,
#     ) -> None:
#         """An interpolated implementation of the MoeDiStefano17 multiplicity relations,
#         plus DucheneKraus+13 for stars below 1 MSun.
#         """
#         self.make_resolved_stars = make_resolved_stars
#         self.distance = distance
#         self.separation = separation

#         if not interpolated_q:
#             raise ValueError(
#                 "interpolated_q may only be set to True. This implementation only "
#                 "supports using pre-computed data from Hunt & Reffert 2024."
#             )

#         # If no distance specified in interpolation mode, then set to max distance
#         # (where everything is unresolved anyway)
#         if self.distance is None or make_resolved_stars is False:
#             self.distance = np.max(self.interpolator.grid[1])

#     def companion_star_fraction(self, masses):
#         return CSF_INTERPOLATOR(masses)

#     def multiplicity_fraction(self, masses):
#         return MF_INTERPOLATOR(masses)
