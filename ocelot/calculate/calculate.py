"""A set of functions for calculating typical cluster parameters.

Todo: error treatment here could be made more bayesian
"""

from typing import Optional

import numpy as np
import pandas as pd

from .constants import mas_per_yr_to_rad_per_s, pc_to_m, deg_to_rad


def _change_none_into_one(membership_probabilities):
    if membership_probabilities is None:
        return 1.
    else:
        return membership_probabilities


def _weighted_standard_deviation(x, weights):
    # See https://stackoverflow.com/a/52655244/12709989 for how the standard deviation is calculated in a weighted way
    # Todo: not sure that this deals with small numbers of points correctly!
    #   See: unit test fails when v. few points used
    return np.sqrt(np.cov(x, aweights=weights))


def king_surface_density(r_values, r_core, r_tidal, normalise=False):
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
    if r_tidal < r_core or r_core < 0 or r_tidal < 0:
        raise ValueError("parameters must satisfy 0 < r_core < r_tidal")

    # Convert to a np array that's at least 1d, and check that it isn't bad
    r_values = np.atleast_1d(r_values)

    if np.any(r_values < 0):
        raise ValueError("all r_values must be positive or zero.")

    # Constants
    rt_rc = r_tidal / r_core
    a = (1 + rt_rc ** 2) ** -0.5

    # Compute normalisation constant if desired
    if normalise:
        term_1 = r_core * np.arctan(rt_rc)
        term_2 = -2 * a * r_core * np.log(rt_rc + 1 / a)
        term_3 = r_tidal * a ** 2
        normalisation_constant = 1 / (term_1 + term_2 + term_3)

    # Work out where  0 <= r < r_tidal
    r_valid = np.logical_and(r_values >= 0, r_values < r_tidal)
    r_invalid = np.invert(r_valid)

    # Compute result
    reduced_r_values = (1 + (r_values[r_valid] / r_core) ** 2) ** (-0.5)
    result = np.empty(r_values.shape)
    result[r_valid] = normalisation_constant * (reduced_r_values - a) ** 2
    result[r_invalid] = 0

    return result


def mean_distance(data_gaia: pd.DataFrame,
                  membership_probabilities: Optional[np.ndarray] = None,
                  key_parallax: str = "parallax",
                  key_parallax_error: str = "parallax_error",
                  key_r_est: str = "r_est",
                  key_r_low: str = "r_lo",
                  key_r_high: str = "r_hi",
                  key_result_flag: str = "result_flag",
                  **kwargs):
    """Produces mean parallax and distance statistics, including a basic handling of error.

    Done in a very basic, frequentist way, whereby means are weighted based on the magnitude of the inverse errors on
    parameters and the membership probabilities (if specified).

    Args:
        data_gaia (pd.DataFrame): Gaia data for the cluster in the standard format (e.g. as in DR2.)
        membership_probabilities (optional, np.ndarray): membership probabilities for the cluster. When specified,
            they can increase or decrease the effect of certain stars on the mean.
        key_parallax (str): Gaia parameter name.
        key_parallax_error (str): Gaia parameter name.
        key_r_est (str): Gaia parameter name. Corresponds to Bailer-Jones+18 distance column names.
        key_r_low (str): Gaia parameter name. Corresponds to Bailer-Jones+18 distance column names.
        key_r_high (str): Gaia parameter name. Corresponds to Bailer-Jones+18 distance column names.
        key_result_flag (str): Gaia parameter name. Corresponds to Bailer-Jones+18 distance column names.

    Returns:
        a dict, formatted with:
            {
            "parallax": weighted mean parallax
            "parallax_error": error on the above
            "inverse_parallax": inverse of the weighted mean parallax, a naive distance estimate
            "distance": weighted mean distance
            "distance_error": (naive, non-Bayesian) weighted error on mean_distance
            }

    """
    membership_probabilities = _change_none_into_one(membership_probabilities)
    inferred_parameters = {}

    # Mean parallax
    parallax_weights = membership_probabilities / np.abs(data_gaia[key_parallax_error])
    inferred_parameters["parallax"] = np.average(data_gaia[key_parallax], weights=parallax_weights)
    inferred_parameters["parallax_error"] = _weighted_standard_deviation(data_gaia[key_parallax], parallax_weights)

    # The inverse too (which is a shitty proxy for distance when you're feeling too lazy to be a Bayesian)
    inferred_parameters["inverse_parallax"] = 1000 / inferred_parameters["parallax"]
    inferred_parameters["inverse_parallax_l68"] = (
        1000 / (inferred_parameters["parallax"] + inferred_parameters["parallax_error"]))
    inferred_parameters["inverse_parallax_u68"] = (
        1000 / (inferred_parameters["parallax"] - inferred_parameters["parallax_error"]))

    # Mean distance, but a bit shit for now lol
    # Todo this could infer a mean/MAP value in a Bayesian way
    # We only want to work on stars with a result
    good_stars = data_gaia[key_result_flag] == 1
    r_est = data_gaia.loc[good_stars, key_r_est].values
    r_low = data_gaia.loc[good_stars, key_r_low].values
    r_high = data_gaia.loc[good_stars, key_r_high].values

    # Deal with the fact that dropping stars without an inferred distance means membership_probabilities might not have
    # the same length as our r_est
    if type(membership_probabilities) is not float:
        membership_probabilities = membership_probabilities[good_stars]

    # Calculate low and high snr estimates and take the mean of them to get the overall snr
    r_est_weight_low = 1 / np.abs(r_est - r_low)
    r_est_weight_high = 1 / np.abs(r_est - r_high)
    r_est_weight = np.mean(np.vstack((r_est_weight_low, r_est_weight_high)), axis=0)
    r_est_weight = r_est_weight * membership_probabilities

    inferred_parameters["distance"] = np.average(r_est, weights=r_est_weight)
    inferred_parameters["distance_error"] = _weighted_standard_deviation(r_est, r_est_weight)

    return inferred_parameters


def mean_proper_motion(data_gaia: pd.DataFrame,
                       membership_probabilities: Optional[np.ndarray] = None,
                       key_pmra: str = "pmra",
                       key_pmra_error: str = "pmra_error",
                       key_pmdec: str = "pmdec",
                       key_pmdec_error: str = "pmdec_error",
                       **kwargs):
    """Calculates the weighted mean proper motion of a cluster, including error.

    Done in a very basic, frequentist way, whereby means are weighted based on the magnitude of the inverse errors on
    parameters and the membership probabilities (if specified).

    Args:
        data_gaia (pd.DataFrame): Gaia data for the cluster in the standard format (e.g. as in DR2.)
        membership_probabilities (optional, np.ndarray): membership probabilities for the cluster. When specified,
            they can increase or decrease the effect of certain stars on the mean.
        key_pmra (str): Gaia parameter name.
        key_pmra_error (str): Gaia parameter name.
        key_pmdec (str): Gaia parameter name.
        key_pmdec_error (str): Gaia parameter name.

    Returns:
        a dict, formatted with:
            {
            "pmra": weighted mean proper motion in the right ascension direction * cos declination
            "pmra_error": error on the above
            "pmdec": weighted mean proper motion in the declination direction
            "pmdec_error": error on the above
            }

    """
    membership_probabilities = _change_none_into_one(membership_probabilities)
    inferred_parameters = {}

    # Calculate the weights
    pmra_weights = membership_probabilities / data_gaia[key_pmra_error]
    pmdec_weights = membership_probabilities / data_gaia[key_pmdec_error]

    # Mean proper motion time!
    inferred_parameters["pmra"] = np.average(data_gaia[key_pmra], weights=pmra_weights)
    inferred_parameters["pmra_error"] = _weighted_standard_deviation(data_gaia[key_pmra], pmra_weights)

    inferred_parameters["pmdec"] = np.average(data_gaia[key_pmdec], weights=pmdec_weights)
    inferred_parameters["pmdec_error"] = _weighted_standard_deviation(data_gaia[key_pmdec], pmdec_weights)

    return inferred_parameters


def mean_radius(data_gaia: pd.DataFrame,
                membership_probabilities: Optional[np.ndarray] = None,
                already_inferred_parameters: Optional[dict] = None,
                key_ra: str = "ra",
                key_ra_error: str = "ra_error",
                key_dec: str = "dec",
                key_dec_error: str = "dec_error",
                distance_to_use: str = "inverse_parallax",
                **kwargs):
    """Produces various radius statistics on a given cluster, finding its sky location and three radii: the core, tidal
    and 50% radius.

    Done in a very basic, frequentist way, whereby means are weighted based on the membership probabilities (if
    specified).

    N.B. unlike the above functions, errors do *not change the mean* as this would potentially bias the
    estimates towards being dominated by large, centrally-located stars within clusters (that have generally lower
    velocities.) Hence, estimates here will be less precise but hopefully more accurate.

    Todo: add error estimation to this function (hard)

    Todo: add galactic l, b to the output of this function  

    Args:
        data_gaia (pd.DataFrame): Gaia data for the cluster in the standard format (e.g. as in DR2.)
        membership_probabilities (optional, np.ndarray): membership probabilities for the cluster. When specified,
            they can increase or decrease the effect of certain stars on the mean.
        already_inferred_parameters (optional, dict): a parameter dictionary of the mean distance and proper motion.
            Otherwise, this function calculates a version.
        key_ra (str): Gaia parameter name.
        key_ra_error (str): Gaia parameter name.
        key_dec (str): Gaia parameter name.
        key_dec_error (str): Gaia parameter name.
        distance_to_use (str): which already inferred distance to use to convert angular radii to parsecs.
            Default: "inverse_parallax"

    Returns:
        a dict, formatted with:
            {
            # Position
            "ra": ra of the cluster
            "ra_error": error on the above
            "dec": dec of the cluster
            "dec_error": error on the above

            # Angular radii
            "ang_radius_50": median ang. distance from the centre, i.e.angular radius of the cluster with 50% of members
            "ang_radius_50_error": error on the above
            "ang_radius_c": angular King's core radius of the cluster
            "ang_radius_c_error": error on the above
            "ang_radius_t": maximum angular distance from the centre, i.e. angular King's tidal radius of the cluster
            "ang_radius_t_error": error on the above

            # Parsec radii
            "radius_50": median distance from the centre, i.e.radius of the cluster with 50% of members
            "radius_50_error": error on the above
            "radius_c": King's core radius of the cluster
            "radius_c_error": error on the above
            "radius_t": maximum distance from the centre, i.e. King's tidal radius of the cluster
            "radius_t_error": error on the above
            }

    """
    inferred_parameters = {}

    # Grab the distances if they aren't specified - we'll need them in a moment!
    if already_inferred_parameters is None:
        already_inferred_parameters = mean_distance(data_gaia, membership_probabilities)

    # Estimate the ra, dec of the cluster as the weighted mean
    ra = np.average(data_gaia[key_ra], weights=membership_probabilities)
    dec = np.average(data_gaia[key_dec], weights=membership_probabilities)

    # Calculate how far every star in the cluster is from the central point
    distances_from_center = np.sqrt((data_gaia[key_ra] - ra)**2 + (data_gaia[key_dec] - dec)**2)
    ang_radius_50 = np.median(distances_from_center)
    ang_radius_t = np.max(distances_from_center)

    # Convert the angular distances into parsecs
    radius_50 = np.tan(ang_radius_50 * deg_to_rad) * already_inferred_parameters[distance_to_use]
    radius_t = np.tan(ang_radius_t * deg_to_rad) * already_inferred_parameters[distance_to_use]

    return {# Sky position
            "ra": ra,
            "ra_error": np.nan,
            "dec": dec,
            "dec_error": np.nan,

            # Angular radii
            "ang_radius_50": ang_radius_50,
            "ang_radius_50_error": np.nan,
            "ang_radius_c": np.nan,
            "ang_radius_c_error": np.nan,
            "ang_radius_t": ang_radius_t,
            "ang_radius_t_error": np.nan,

            # Parsec radii
            "radius_50": radius_50,
            "radius_50_error": np.nan,
            "radius_c": np.nan,
            "radius_c_error": np.nan,
            "radius_t": radius_t,
            "radius_t_error": np.nan}


def mean_internal_velocity_dispersion(data_gaia: pd.DataFrame,
                                      membership_probabilities: Optional[np.ndarray] = None,
                                      already_inferred_parameters: Optional[dict] = None,
                                      key_pmra: str = "ra",
                                      key_pmra_error: str = "ra_error",
                                      key_pmdec: str = "dec",
                                      key_pmdec_error: str = "dec_error",
                                      distance_to_use: str = "inverse_parallax",
                                      **kwargs):
    """Estimates the internal velocity dispersion of a cluster.

    Done in a very basic, frequentist way, whereby means are weighted based on the membership probabilities (if
    specified).

    N.B. unlike the above functions, errors do *not change the mean* as this would potentially bias the
    estimates towards being dominated by large, centrally-located stars within clusters (that have generally lower
    velocities.) Hence, estimates here will be less precise but hopefully more accurate.

    Todo: add error on velocity dispersion estimation to this function (hard)

    Args:
        data_gaia (pd.DataFrame): Gaia data for the cluster in the standard format (e.g. as in DR2.)
        membership_probabilities (optional, np.ndarray): membership probabilities for the cluster. When specified,
            they can increase or decrease the effect of certain stars on the mean.
        already_inferred_parameters (optional, dict): a parameter dictionary of the mean distance and proper motion.
            Otherwise, this function calculates a version.
        key_pmra (str): Gaia parameter name.
        key_pmra_error (str): Gaia parameter name.
        key_pmdec (str): Gaia parameter name.
        key_pmdec_error (str): Gaia parameter name.
        distance_to_use (str): which already inferred distance to use to convert proper motions to m/s velocities.
            Default: "inverse_parallax"

    Returns:
        a dict, formatted with:
            {
            "v_ra_dec": mean velocity dispersion of the cluster
            "v_ra_dec_error": error on the above
            }

    """
    inferred_parameters = {}

    # Grab the distances and proper motions if they aren't specified - we'll need them in a moment!
    if already_inferred_parameters is None:
        already_inferred_parameters = {**mean_distance(data_gaia, membership_probabilities),
                                       **mean_proper_motion(data_gaia, membership_probabilities)}

    # Center the proper motions on the cluster
    pmra = data_gaia[key_pmra] - already_inferred_parameters["pmra"]
    pmdec = data_gaia[key_pmdec] - already_inferred_parameters["pmdec"]

    # Velocity dispersion time
    pm_magnitude = np.sqrt(pmra ** 2 + pmdec ** 2)
    velocity_dispersion = (np.tan(pm_magnitude * mas_per_yr_to_rad_per_s)
                           * already_inferred_parameters[distance_to_use] * pc_to_m)

    # Save the standard deviations of the sum of the squares of parameters as our velocity dispersions
    inferred_parameters["v_internal_tangential"] = _weighted_standard_deviation(velocity_dispersion,
                                                                                membership_probabilities)
    inferred_parameters["v_internal_tangential_error"] = np.nan

    return inferred_parameters


def all_statistics(data_gaia: pd.DataFrame,
                   membership_probabilities: Optional[np.ndarray] = None,
                   parameter_inference_mode: str = "mean",
                   **kwargs):
    """Wraps all subfunctions in ocelot.calculate and calculates all the stats you could possibly ever want.

    Args:
        data_gaia (pd.DataFrame): Gaia data for the cluster in the standard format (e.g. as in DR2.)
        membership_probabilities (optional, np.ndarray): membership probabilities for the cluster. When specified,
            they can increase or decrease the effect of certain stars on the mean.
        parameter_inference_mode (str): mode to use when inferring parameters.
        **kwargs: keyword arguments to pass to the calculation functions this one calls.


    Returns:
        a dict, containing the following parameters for the cluster:
            'ra', 'ra_error', 'dec', 'dec_error', 'ang_radius_50', 'ang_radius_50_error', 'ang_radius_c',
            'ang_radius_c_error', 'ang_radius_t', 'ang_radius_t_error', 'radius_50', 'radius_50_error', 'radius_c',
            'radius_c_error', 'radius_t', 'radius_t_error', 'parallax', 'parallax_error', 'inverse_parallax',
            'inverse_parallax_l68', 'inverse_parallax_u68', 'distance', 'distance_error', 'pmra', 'pmra_error', 'pmdec',
            'pmdec_error', 'v_internal_tangential', 'v_internal_tangential_error', 'parameter_inference_mode'
        where parameter_inference_mode is the only one added by this function itself, and is a copy of the keyword arg
        this function holds.

    """
    if parameter_inference_mode == "mean":
        # Calculate all parameters! We incrementally add to the dictionary as some of the later functions require
        # parameters that we've already calculated. This is done in a weird order to make sure that the final dict is in
        # the right order.
        inferred_parameters = {}

        inferred_parameters.update(
            mean_distance(data_gaia, membership_probabilities=membership_probabilities, **kwargs))

        inferred_parameters = {
            **mean_radius(data_gaia, membership_probabilities=membership_probabilities,
                          already_inferred_parameters=inferred_parameters, **kwargs),
            **inferred_parameters}

        inferred_parameters.update(
            mean_proper_motion(data_gaia, membership_probabilities=membership_probabilities, **kwargs))

        inferred_parameters.update(
            mean_internal_velocity_dispersion(data_gaia, membership_probabilities=membership_probabilities,
                                              already_inferred_parameters=inferred_parameters, **kwargs))

        inferred_parameters.update({"parameter_inference_mode": "mean"})

    else:
        raise ValueError("the only currently implemented parameter_inference_mode for this function is 'mean'")

    # Return it all as one big dict
    return inferred_parameters
