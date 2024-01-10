"""A set of functions for calculating typical cluster parameters.

Todo: error treatment here could be made more bayesian
"""

from typing import Optional
from astropy.coordinates import SkyCoord

import numpy as np
import pandas as pd

from .constants import mas_per_yr_to_rad_per_s, pc_to_m, deg_to_rad


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


def all_statistics():
    """
    """
    # Todo refactor this (and other high-level calculation methods)
    pass
