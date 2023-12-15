"""A set of functions for calculating typical cluster parameters.

Todo: error treatment here could be made more bayesian
"""

from typing import Optional
from astropy.coordinates import SkyCoord

import numpy as np
import pandas as pd

from .constants import mas_per_yr_to_rad_per_s, pc_to_m, deg_to_rad


def _weighted_standard_deviation(x, weights):
    # See https://stackoverflow.com/a/52655244/12709989 for how the standard deviation is calculated in a weighted way
    # Todo: not sure that this deals with small numbers of points correctly!
    #   See: unit test fails when v. few points used
    return np.sqrt(np.cov(x, aweights=weights))


def _handle_ra_discontinuity(ra_data, middle_ras_raise_error=True):
    """Tries to detect when the ras in a field cross the (0, 360) ra discontinuity and returns corrected results. Will
    raise an error if ras are all over the place (which will happen e.g. at very high declinations) in which you
    ought to instead switch to a method free of spherical distortions.

    Args:
        ra_data (pd.Series or np.ndarray): data on ras.
        middle_ras_raise_error (bool): whether or not a cluster having right ascensions in all ranges [0, 90), [90, 270]
            and (270, 360] raises an error. The error here indicates that this cluster has extreme spherical
            discontinuities (e.g. it's near a coordinate pole) and that the mean ra and mean dec will be inaccurate.
            Default: True

    Returns:
        ra_data but corrected for distortions. If values are both <90 and >270, the new ra data will be in the range
            (-90, 90).

    """
    # Firstly, check that the ras are valid ras
    if np.any(ra_data >= 360) or np.any(ra_data < 0):
        raise ValueError(
            "at least one input ra value was invalid! Ras must be in the range [0, 360)."
        )

    # Next, grab all the locations of dodgy friends and check that all three aren't ever in use at the same time
    low_ra = ra_data < 90
    high_ra = ra_data > 270
    middle_ra = np.logical_not(np.logical_or(low_ra, high_ra))

    # Proceed if we have both low and high ras
    if np.any(low_ra) and np.any(high_ra):
        # Stop if we have middle too (would imply stars everywhere or an extreme dec value)
        if np.any(middle_ra) and middle_ras_raise_error:
            raise ValueError(
                "ra values are in all three ranges: [0, 90), [90, 270] and (270, 360). This cluster can't "
                "be processed by this function! Spherical distortions must be removed first."
            )

        # Otherwise, apply the discontinuity removal
        else:
            # Make a copy so nothing weird happens
            ra_data = ra_data.copy()

            # And remove the distortion for all high numbers
            ra_data[high_ra] = ra_data[high_ra] - 360

    return ra_data


def mean_distance(
    data_gaia: pd.DataFrame,
    membership_probabilities: Optional[np.ndarray] = None,
    key_parallax: str = "parallax",
    key_parallax_error: str = "parallax_error",
    key_r_est: str = "r_est",
    key_r_low: str = "r_lo",
    key_r_high: str = "r_hi",
    key_result_flag: str = "result_flag",
    calculate_cbj_mean_distance: bool = False,
    **kwargs,
):
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
        calculate_cbj_mean_distance (bool): whether or not to even bother calculating a mean CBJ distance.
            Default: False

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
    inferred_parameters = {}
    sqrt_n_stars = np.sqrt(data_gaia.shape[0])

    # Mean parallax
    inferred_parameters["parallax"] = np.average(
        data_gaia[key_parallax], weights=membership_probabilities
    )
    inferred_parameters["parallax_std"] = _weighted_standard_deviation(
        data_gaia[key_parallax], membership_probabilities
    )
    inferred_parameters["parallax_error"] = (
        inferred_parameters["parallax_std"] / sqrt_n_stars
    )

    # The inverse too (which is a shitty proxy for distance when you're feeling too lazy to be a Bayesian)
    inferred_parameters["inverse_parallax"] = 1000 / inferred_parameters["parallax"]
    inferred_parameters["inverse_parallax_l68"] = 1000 / (
        inferred_parameters["parallax"] + inferred_parameters["parallax_error"]
    )
    inferred_parameters["inverse_parallax_u68"] = 1000 / (
        inferred_parameters["parallax"] - inferred_parameters["parallax_error"]
    )

    # Mean distance, but a bit shit for now lol
    if calculate_cbj_mean_distance:
        # Todo this could infer a mean/MAP value in a Bayesian way
        # We only want to work on stars with a result
        good_stars = data_gaia[key_result_flag] == 1
        r_est = data_gaia.loc[good_stars, key_r_est].values

        # Deal with the fact that dropping stars without an inferred distance means membership_probabilities might not
        # have the same length as our r_est
        if (
            type(membership_probabilities) is not float
            and membership_probabilities is not None
        ):
            membership_probabilities = membership_probabilities[good_stars]

        inferred_parameters["distance"] = np.average(
            r_est, weights=membership_probabilities
        )
        inferred_parameters["distance_std"] = _weighted_standard_deviation(
            r_est, membership_probabilities
        )
        inferred_parameters["distance_error"] = (
            inferred_parameters["distance_std"] / sqrt_n_stars
        )

    return inferred_parameters


def mean_proper_motion(
    data_gaia: pd.DataFrame,
    membership_probabilities: Optional[np.ndarray] = None,
    key_pmra: str = "pmra",
    key_pmra_error: str = "pmra_error",
    key_pmdec: str = "pmdec",
    key_pmdec_error: str = "pmdec_error",
    **kwargs,
):
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
    inferred_parameters = {}
    sqrt_n_stars = np.sqrt(data_gaia.shape[0])

    # Mean proper motion time!
    inferred_parameters["pmra"] = np.average(
        data_gaia[key_pmra], weights=membership_probabilities
    )
    inferred_parameters["pmra_std"] = _weighted_standard_deviation(
        data_gaia[key_pmra], membership_probabilities
    )
    inferred_parameters["pmra_error"] = inferred_parameters["pmra_std"] / sqrt_n_stars

    inferred_parameters["pmdec"] = np.average(
        data_gaia[key_pmdec], weights=membership_probabilities
    )
    inferred_parameters["pmdec_std"] = _weighted_standard_deviation(
        data_gaia[key_pmdec], membership_probabilities
    )
    inferred_parameters["pmdec_error"] = inferred_parameters["pmdec_std"] / sqrt_n_stars

    inferred_parameters["pm_dispersion"] = np.sqrt(
        inferred_parameters["pmra_std"] ** 2 + inferred_parameters["pmdec_std"] ** 2
    )

    return inferred_parameters


def mean_radius(
    data_gaia: pd.DataFrame,
    membership_probabilities: Optional[np.ndarray] = None,
    already_inferred_parameters: Optional[dict] = None,
    key_ra: str = "ra",
    key_ra_error: str = "ra_error",
    key_dec: str = "dec",
    key_dec_error: str = "dec_error",
    distance_to_use: str = "inverse_parallax",
    middle_ras_raise_error: bool = True,
    **kwargs,
):
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
        middle_ras_raise_error (bool): whether or not a cluster having right ascensions in all ranges [0, 90), [90, 270]
            and (270, 360] raises an error. The error here indicates that this cluster has extreme spherical
            discontinuities (e.g. it's near a coordinate pole) and that the mean ra and mean dec will be inaccurate.
            Default: True

    Returns:
        a dict, formatted with:
            {
            # Position
            "ra": ra of the cluster
            "ra_error": error on the above
            "dec": dec of the cluster
            "dec_error": error on the above

            # Angular radii
            "ang_radius_50": median ang. distance from the center, i.e.angular radius of the cluster with 50% of members
            "ang_radius_50_error": error on the above
            "ang_radius_c": angular King's core radius of the cluster
            "ang_radius_c_error": error on the above
            "ang_radius_t": maximum angular distance from the center, i.e. angular King's tidal radius of the cluster
            "ang_radius_t_error": error on the above

            # Parsec radii
            "radius_50": median distance from the center, i.e.radius of the cluster with 50% of members
            "radius_50_error": error on the above
            "radius_c": King's core radius of the cluster
            "radius_c_error": error on the above
            "radius_t": maximum distance from the center, i.e. King's tidal radius of the cluster
            "radius_t_error": error on the above
            }

    """
    inferred_parameters = {}
    sqrt_n_stars = np.sqrt(data_gaia.shape[0])

    # Grab the distances if they aren't specified - we'll need them in a moment!
    if already_inferred_parameters is None:
        already_inferred_parameters = mean_distance(data_gaia, membership_probabilities)

    # Estimate the ra, dec of the cluster as the weighted mean
    ra_data = _handle_ra_discontinuity(
        data_gaia[key_ra], middle_ras_raise_error=middle_ras_raise_error
    )

    inferred_parameters["ra"] = np.average(ra_data, weights=membership_probabilities)
    inferred_parameters["ra_std"] = _weighted_standard_deviation(
        ra_data, membership_probabilities
    )
    inferred_parameters["ra_error"] = inferred_parameters["ra_std"] / sqrt_n_stars

    if inferred_parameters["ra"] < 0:
        inferred_parameters["ra"] += 360

    inferred_parameters["dec"] = np.average(
        data_gaia[key_dec], weights=membership_probabilities
    )
    inferred_parameters["dec_std"] = _weighted_standard_deviation(
        data_gaia[key_dec], membership_probabilities
    )
    inferred_parameters["dec_error"] = inferred_parameters["dec_std"] / sqrt_n_stars

    inferred_parameters["ang_dispersion"] = np.sqrt(
        inferred_parameters["ra_std"] ** 2 + inferred_parameters["dec_std"] ** 2
    )

    # Calculate how far every star in the cluster is from the central point
    center_skycoord = SkyCoord(
        ra=inferred_parameters["ra"], dec=inferred_parameters["dec"], unit="deg"
    )
    star_skycoords = SkyCoord(
        ra=data_gaia[key_ra].to_numpy(), dec=data_gaia[key_dec].to_numpy(), unit="deg"
    )

    distances_from_center = center_skycoord.separation(star_skycoords).degree

    # And say something about the radii in this case
    inferred_parameters["ang_radius_50"] = np.median(distances_from_center)
    inferred_parameters["ang_radius_50_error"] = np.nan

    inferred_parameters["ang_radius_c"] = np.nan
    inferred_parameters["ang_radius_c_error"] = np.nan

    inferred_parameters["ang_radius_t"] = np.max(distances_from_center)
    inferred_parameters["ang_radius_t_error"] = np.nan

    # Convert the angular distances into parsecs
    inferred_parameters["radius_50"] = (
        np.tan(inferred_parameters["ang_radius_50"] * deg_to_rad)
        * already_inferred_parameters[distance_to_use]
    )
    inferred_parameters["radius_50_error"] = np.nan
    inferred_parameters["radius_c"] = np.nan
    inferred_parameters["radius_c_error"] = np.nan
    inferred_parameters["radius_t"] = (
        np.tan(inferred_parameters["ang_radius_t"] * deg_to_rad)
        * already_inferred_parameters[distance_to_use]
    )
    inferred_parameters["radius_t_error"] = np.nan

    return inferred_parameters


def mean_internal_velocity_dispersion(
    data_gaia: pd.DataFrame,
    membership_probabilities: Optional[np.ndarray] = None,
    already_inferred_parameters: Optional[dict] = None,
    key_pmra: str = "pmra",
    key_pmra_error: str = "pmra_error",
    key_pmdec: str = "pmdec",
    key_pmdec_error: str = "pmdec_error",
    distance_to_use: str = "inverse_parallax",
    **kwargs,
):
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
        already_inferred_parameters = {
            **mean_distance(data_gaia, membership_probabilities),
            **mean_proper_motion(data_gaia, membership_probabilities),
        }

    # Center the proper motions on the cluster
    pmra = data_gaia[key_pmra] - already_inferred_parameters["pmra"]
    pmdec = data_gaia[key_pmdec] - already_inferred_parameters["pmdec"]

    # Velocity dispersion time
    pm_magnitude = np.sqrt(pmra**2 + pmdec**2)
    velocity_dispersion = (
        np.tan(pm_magnitude * mas_per_yr_to_rad_per_s)
        * already_inferred_parameters[distance_to_use]
        * pc_to_m
    )

    # Save the standard deviations of the sum of the squares of parameters as our velocity dispersions
    inferred_parameters["v_internal_tangential"] = _weighted_standard_deviation(
        velocity_dispersion, membership_probabilities
    )
    inferred_parameters["v_internal_tangential_error"] = np.nan

    return inferred_parameters


def all_statistics(
    data_gaia: pd.DataFrame,
    membership_probabilities: Optional[np.ndarray] = None,
    parameter_inference_mode: str = "mean",
    override_membership_probabilities_being_off: bool = False,
    middle_ras_raise_error: bool = True,
    **kwargs,
):
    """Wraps all subfunctions in ocelot.calculate and calculates all the stats you could possibly ever want.

    Args:
        data_gaia (pd.DataFrame): Gaia data for the cluster in the standard format (e.g. as in DR2.)
        membership_probabilities (optional, np.ndarray): membership probabilities for the cluster. When specified,
            they can increase or decrease the effect of certain stars on the mean.
        parameter_inference_mode (str): mode to use when inferring parameters.
        override_membership_probabilities_being_off (bool): little check to stop membership probabilities from being
            used for now, as these actually mess up the
        middle_ras_raise_error (bool): whether or not a cluster having right ascensions in all ranges [0, 90), [90, 270]
            and (270, 360] raises an error. The error here indicates that this cluster has extreme spherical
            discontinuities (e.g. it's near a coordinate pole) and that the mean ra and mean dec will be inaccurate.
            Default: True
        **kwargs: keyword arguments to pass to the calculation functions this one calls.


    Returns:
        a dict, containing the following parameters for the cluster:
            'n_stars', 'ra', 'ra_error', 'dec', 'dec_error', 'ang_radius_50', 'ang_radius_50_error', 'ang_radius_c',
            'ang_radius_c_error', 'ang_radius_t', 'ang_radius_t_error', 'radius_50', 'radius_50_error', 'radius_c',
            'radius_c_error', 'radius_t', 'radius_t_error', 'parallax', 'parallax_error', 'inverse_parallax',
            'inverse_parallax_l68', 'inverse_parallax_u68', 'distance', 'distance_error', 'pmra', 'pmra_error', 'pmdec',
            'pmdec_error', 'v_internal_tangential', 'v_internal_tangential_error', 'parameter_inference_mode'
        where parameter_inference_mode is the only one added by this function itself, and is a copy of the keyword arg
        this function holds.

    """
    if not override_membership_probabilities_being_off:
        membership_probabilities = None

    if parameter_inference_mode == "mean":
        # Calculate all parameters! We incrementally add to the dictionary as some of the later functions require
        # parameters that we've already calculated. This is done in a weird order to make sure that the final dict is in
        # the right order.
        inferred_parameters = {}

        inferred_parameters.update(
            mean_distance(
                data_gaia, membership_probabilities=membership_probabilities, **kwargs
            )
        )

        # Todo: could also add the number of 1sigma+ member stars.

        inferred_parameters = {
            "n_stars": data_gaia.shape[0],
            **mean_radius(
                data_gaia,
                membership_probabilities=membership_probabilities,
                already_inferred_parameters=inferred_parameters,
                middle_ras_raise_error=middle_ras_raise_error,
                **kwargs,
            ),
            **inferred_parameters,
        }

        inferred_parameters.update(
            mean_proper_motion(
                data_gaia, membership_probabilities=membership_probabilities, **kwargs
            )
        )

        inferred_parameters.update(
            mean_internal_velocity_dispersion(
                data_gaia,
                membership_probabilities=membership_probabilities,
                already_inferred_parameters=inferred_parameters,
                **kwargs,
            )
        )

        inferred_parameters.update({"parameter_inference_mode": "mean"})

    else:
        raise ValueError(
            "the only currently implemented parameter_inference_mode for this function is 'mean'"
        )

    # Return it all as one big dict
    return inferred_parameters
