"""A set of functions for calculating typical cluster parameters.

Todo: error treatment here could be made more bayesian
"""

import numpy as np
import pandas as pd

from astropy import units as u
from typing import Optional


def _handle_membership_probabilities(membership_probabilities):
    if membership_probabilities is None:
        return 1.
    else:
        return membership_probabilities


def _weighted_standard_deviation(x, weights):
    # See https://stackoverflow.com/a/52655244/12709989 for how the standard deviation is calculated in a weighted way
    return np.sqrt(np.cov(x, aweights=weights))


def mean_distance(data_gaia: pd.DataFrame,
                  membership_probabilities: Optional[np.ndarray] = None,
                  key_parallax: str = "parallax",
                  key_parallax_error: str = "parallax_error",
                  key_r_est: str = "r_est",
                  key_r_low: str = "r_lo",
                  key_r_high: str = "r_hi",
                  key_result_flag: str = "result_flag"):
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
            "mean_parallax": weighted mean parallax
            "mean_parallax_error": error on the above
            "mean_inverse_parallax": inverse of the weighted mean parallax, a naive distance estimate
            "mean_distance": weighted mean distance
            "mean_distance_error": (naive, non-Bayesian) weighted error on mean_distance
            }

    """
    membership_probabilities = _handle_membership_probabilities(membership_probabilities)
    inferred_parameters = {}

    # Mean parallax
    parallax_weights = membership_probabilities / np.abs(data_gaia[key_parallax_error])
    inferred_parameters["mean_parallax"] = np.average(data_gaia[key_parallax], weights=parallax_weights)
    inferred_parameters["mean_parallax_error"] = _weighted_standard_deviation(data_gaia[key_parallax], parallax_weights)

    inferred_parameters["mean_inverse_parallax"] = 1 / inferred_parameters["mean_parallax"]

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

    inferred_parameters["mean_distance"] = np.average(r_est, weights=r_est_weight)
    inferred_parameters["mean_distance_error"] = _weighted_standard_deviation(r_est, r_est_weight)

    return inferred_parameters


def mean_proper_motion(data_gaia: pd.DataFrame,
                       membership_probabilities: Optional[np.ndarray] = None,
                       key_pmra: str = "ra",
                       key_pmra_error: str = "ra_error",
                       key_pmdec: str = "dec",
                       key_pmdec_error: str = "dec_error"):
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
            "mean_pmra": weighted mean proper motion in the right ascension direction * cos declination
            "mean_pmra_error": error on the above
            "mean_pmdec": weighted mean proper motion in the declination direction
            "mean_pmdec_error": error on the above
            }

    """
    membership_probabilities = _handle_membership_probabilities(membership_probabilities)
    inferred_parameters = {}

    # Calculate the weights
    pmra_weights = membership_probabilities / data_gaia[key_pmra_error]
    pmdec_weights = membership_probabilities / data_gaia[key_pmdec_error]

    # Mean proper motion time!
    inferred_parameters["mean_pmra"] = np.average(data_gaia[key_pmra], weights=pmra_weights)
    inferred_parameters["mean_pmra_error"] = _weighted_standard_deviation(data_gaia[key_pmra], pmra_weights)

    inferred_parameters["mean_pmdec"] = np.average(data_gaia[key_pmdec], weights=pmdec_weights)
    inferred_parameters["mean_pmdec_error"] = _weighted_standard_deviation(data_gaia[key_pmdec], pmdec_weights)

    return inferred_parameters


def radius(data_gaia: pd.DataFrame = None,
           membership_probabilities: Optional[np.ndarray] = None,
           already_inferred_parameters: Optional[dict] = None,
           key_ra: str = "ra",
           key_ra_error: str = "ra_error",
           key_dec: str = "dec",
           key_dec_error: str = "dec_error"):
    """Produces various radius statistics on a given cluster, finding its sky location and three radii: the core, tidal
    and 50% radius.

    todo

    """

    return {"radius_50": np.nan,
            "radius_50_error": np.nan,
            "radius_c": np.nan,
            "radius_c_error": np.nan,
            "radius_t": np.nan,
            "radius_t_error": np.nan}


def internal_velocity_dispersion(data_gaia: pd.DataFrame,
                                 membership_probabilities: Optional[np.ndarray] = None,
                                 already_inferred_parameters: Optional[dict] = None,
                                 key_pmra: str = "ra",
                                 key_pmra_error: str = "ra_error",
                                 key_pmdec: str = "dec",
                                 key_pmdec_error: str = "dec_error",
                                 distance_to_use: str = "mean_inverse_parallax"):
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
            Default: "mean_inverse_parallax"

    Returns:
        a dict, formatted with:
            {
            "v_ra_dec": mean velocity dispersion of the cluster
            "v_ra_dec_error": error on the above
            }

    """
    membership_probabilities = _handle_membership_probabilities(membership_probabilities)
    inferred_parameters = {}

    # Grab the distances and proper motions if they aren't specified - we'll need them in a moment!
    if already_inferred_parameters is None:
        already_inferred_parameters = {**mean_distance(data_gaia, membership_probabilities),
                                       **mean_proper_motion(data_gaia, membership_probabilities)}

    # Grab constants we'll need
    masyr_to_rads = (u.mas / u.yr).to(u.rad / u.s)
    pc_to_m = u.parsec.to(u.meter)

    # Center the proper motions on the cluster
    pmra = data_gaia[key_pmra] - already_inferred_parameters["mean_pmra"]
    pmdec = data_gaia[key_pmdec] - already_inferred_parameters["mean_pmdec"]

    # Velocity dispersion time
    pm_magnitude = np.sqrt(pmra ** 2 + pmdec ** 2)
    velocity_dispersion = np.tan(pm_magnitude * masyr_to_rads) / already_inferred_parameters[distance_to_use] * pc_to_m

    # Save the standard deviations of the sum of the squares of parameters as our velocity dispersions
    inferred_parameters["v_ra_dec"] = _weighted_standard_deviation(velocity_dispersion, membership_probabilities)
    inferred_parameters["v_ra_dec_error"] = np.nan

    return inferred_parameters


def all_statistics(data_gaia: pd.DataFrame,
                   membership_probabilities: Optional[np.ndarray] = None,
                   **kwargs):
    """Wraps all subfunctions in ocelot.calculate and calculates all the stats you could possibly ever want.

    todo

    """
    # If membership probabilities is None, this turns it into a one.
    membership_probabilities = _handle_membership_probabilities(membership_probabilities)

    # Calculate all parameters! We incrementally add to the dictionary as some of the later functions require parameters
    # that we've already calculated.
    inferred_parameters = {}

    inferred_parameters.update(
        mean_distance(data_gaia, membership_probabilities=membership_probabilities, **kwargs))

    inferred_parameters.update(
        mean_proper_motion(data_gaia, membership_probabilities=membership_probabilities, **kwargs))

    inferred_parameters.update(
        radius(data_gaia, membership_probabilities=membership_probabilities,
               already_inferred_parameters=inferred_parameters, **kwargs))

    inferred_parameters.update(
        internal_velocity_dispersion(data_gaia, membership_probabilities=membership_probabilities,
                                     already_inferred_parameters=inferred_parameters, **kwargs))

    # Return it all as one big dict
    return inferred_parameters
