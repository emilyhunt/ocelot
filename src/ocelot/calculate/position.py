"""Different methods for calculating the center of a cluster in spherical coordinates. 
(This is actually oddly difficult, thanks to how spheres work. Damn spheres.)
"""
import numpy as np
from numpy.typing import ArrayLike

from ocelot.calculate.generic import spherical_mean


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

    # Convert back to correct interval
    mean_lon = np.where(mean_lon < 0, mean_lon + 2 * np.pi, mean_lon)

    if degrees:
        return np.degrees(mean_lon), np.degrees(mean_lat)
    return mean_lon, mean_lat


def mode_position():
    """Attempts to find the mode of a star cluster's 2D on-sky distribution. This is a 
    better estimator than the mean position for clusters that are assymmetric, which is
    often the case for clusters with assymetric tidal tails (e.g. due to one side being
    more easily detected than the other.)
    """
    # Todo
    pass
