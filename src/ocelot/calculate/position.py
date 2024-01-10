"""Different methods for calculating the center of a cluster in spherical coordinates. 
(This is actually oddly difficult, thanks to how spheres work. Damn spheres.)
"""
import numpy as np
from numpy.typing import ArrayLike

from ocelot.calculate.generic import spherical_mean


# def shift_longitudinal_coordinates(longitudes: ArrayLike, middle_ras_raise_error=True):
#     """Helper function that tries to move longitudinal coordinates

#     Args:
#         ra_data (pd.Series or np.ndarray): data on ras.
#         middle_ras_raise_error (bool): whether or not a cluster having right ascensions in all ranges [0, 90), [90, 270]
#             and (270, 360] raises an error. The error here indicates that this cluster has extreme spherical
#             discontinuities (e.g. it's near a coordinate pole) and that the mean ra and mean dec will be inaccurate.
#             Default: True

#     Returns:
#         ra_data but corrected for distortions. If values are both <90 and >270, the new ra data will be in the range
#             (-90, 90).

#     """
#     # Firstly, check that the ras are valid ras
#     if np.any(longitudes >= 360) or np.any(longitudes < 0):
#         raise ValueError(
#             "at least one input ra value was invalid! Ras must be in the range [0, 360)."
#         )

#     # Next, grab all the locations of dodgy friends and check that all three aren't ever in use at the same time
#     low_ra = longitudes < 90
#     high_ra = longitudes > 270
#     middle_ra = np.logical_not(np.logical_or(low_ra, high_ra))

#     # Proceed if we have both low and high ras
#     if np.any(low_ra) and np.any(high_ra):
#         # Stop if we have middle too (would imply stars everywhere or an extreme dec value)
#         if np.any(middle_ra) and middle_ras_raise_error:
#             raise ValueError(
#                 "ra values are in all three ranges: [0, 90), [90, 270] and (270, 360). This cluster can't "
#                 "be processed by this function! Spherical distortions must be removed first."
#             )

#         # Otherwise, apply the discontinuity removal
#         else:
#             # Make a copy so nothing weird happens
#             longitudes = longitudes.copy()

#             # And remove the distortion for all high numbers
#             longitudes[high_ra] = longitudes[high_ra] - 360

#     return longitudes


def mean_position(
    longitudes: ArrayLike, latitudes: ArrayLike, degrees=True
) -> tuple[float]:
    """Calculates the spherical mean of angular positions, specified as longitudes and
    latitudes. This uses directional statistics to do so in a way that is aware of
    discontinuities, such as the fact that 0° = 360°.

    Parameters
    ----------
    longitudes : array-like
        Array of longitudinal positions of stars in your cluster (e.g. right ascensions
        or galactic longitudes.) Assumed to be in the range [0°, 360°].
    latitudes : array-like
        Array of latitudinal positions of stars in your cluster (e.g. declinations or 
        galactic latitudes.) Assumed to be in the range [-90°, 90°].
    degrees : bool
        Whether longitudes and latitudes are in degrees, and whether to return an answer
        in degrees. Defaults to True. If False, longitudes and latitudes are assumed to
        be in radians, with ranges [0, 2π] and [-π/2, π/2] respectively.

    Returns
    -------
    mean_longitude : float
    mean_latitude : float

    Notes
    -----
    This function explicitly assumes that your star cluster *has* a well-defined mean
    position. Some configurations (such as points uniformly distributed in at least one
    axis of a sphere) will not have a meaningful mean position.

    Internally, this function uses `scipy.stats.directional_stats`, with a definition
    taken from [1]. See [2] for more background.

    References
    ----------
    [1] Mardia, Jupp. (2000). Directional Statistics (p. 163). Wiley.
    [2] https://en.wikipedia.org/wiki/Directional_statistics
    """
    if degrees:
        longitudes, latitudes = np.radians(longitudes), np.radians(latitudes)
    mean_lon, mean_lat = spherical_mean(longitudes, latitudes)
    if degrees:
        return np.degrees(mean_lon), np.degrees(mean_lat)
    return mean_lon, mean_lat
