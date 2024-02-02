"""Functions for doing various random sampling problems that aren't otherwise defined.
"""
import numpy as np
from typing import Tuple


def points_on_sphere(
    shape: int, radians: bool = True, phi_symmetric=True, seed=None
) -> Tuple[np.ndarray, np.ndarray]:
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
    rng = np.random.default_rng(seed=seed)
    theta = 2 * np.pi * rng.uniform(size=shape)
    phi = np.arccos(2 * rng.uniform(size=shape) - 1)

    if phi_symmetric:
        phi -= np.pi / 2

    if radians:
        return theta, phi
    else:
        return theta * 180 / np.pi, phi * 180 / np.pi
