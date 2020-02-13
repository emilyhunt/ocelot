"""Special use case functions for random sampling."""

import numpy as np

from typing import Tuple


# def random_spherical_points(num_points, b_range=(0.0, 90.0)):
#     """Helper function to draw random points from a sphere. See
#     http://mathworld.wolfram.com/SpherePointPicking.html for explanation!
#
#     For u, v random variates in the range (0, 1), and w a random sign, theta = 2*pi*u and phi = arcsin(v) * w
#
#     We draw v in the range (0, 0.5) and multiply by random signs to allow b_range to work with absolute values.
#
#     Args:
#         num_points (int): number of points to draw.
#         b_range (list-like): minimum and maximum absolute values of galactic latitude.
#             Default: (0.0, 90.0)
#
#     Returns:
#         a pd.DataFrame of randomly drawn locations, both with l, b coords, ra, dec coords.
#
#     """
#     # Convert the b values into minimum/maximum v.
#     v_range = np.flip(np.sin(np.asarray(b_range) * np.pi / 180))
#
#     # Longitude is easy
#     longitudes = 360 * np.random.rand(num_points)
#
#     # Latitude requires getting v, some signs, then multiplying all together
#     v = (v_range[1] - v_range[0]) * np.random.rand(num_points) + v_range[0]
#     signs = np.random.choice([-1, 1], num_points)
#     latitudes = 180 / np.pi * np.arcsin(v) * signs
#
#     # Convert these into ra, dec
#     coords_galactic = SkyCoord(longitudes << u.deg, latitudes << u.deg, frame='galactic')
#
#     coords_icrs = coords_galactic.transform_to('icrs')
#
#     # Return a dataframe
#     return pd.DataFrame({'ra': coords_icrs.ra.degree, 'dec': coords_icrs.dec.degree, 'l': longitudes, 'b': latitudes})


def points_on_sphere(shape: int, radians: bool = True, phi_symmetric=True) -> Tuple[np.ndarray, np.ndarray]:
    """To draw random points on a sphere, one cannot simply use uniform deviates based on theta and phi! For a good
    explanation, see: http://mathworld.wolfram.com/SpherePointPicking.html

    Args:
        shape (int): required number of deviates. Should be acceptable by np.random.rand().
        radians (bool): whether or not to return in radians.
            Default: True
        phi_symmetric (bool): whether or not to return a distribution for phi symmetric about zero (so,
            phi ~ [-pi/2, pi/2]) or whether to return one defined in the mathsy (but less useful for astronomy) way
            phi ~ [0, pi].
            Default: True, i.e. phi is immediately usable as galactic latitude b or declination.

    Returns:
        two np.ndarray of random spherical deviates, for theta ~ [0, 2pi) and phi ~ [-pi/2, pi/2] (or in degrees if
            radians=False)

    """
    theta = 2 * np.pi * np.random.rand(shape)
    phi = np.arccos(2 * np.random.rand(shape) - 1)

    if phi_symmetric:
        phi -= np.pi / 2

    if radians:
        return theta, phi
    else:
        return theta * 180 / np.pi, phi * 180 / np.pi
