"""Special use case functions for random sampling."""

import numpy as np

from typing import Union, Tuple


def random_spherical(shape: int, radians: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """To draw random points on a sphere, one cannot simply use uniform deviates based on theta and phi! For a good
    explanation, see: http://mathworld.wolfram.com/SpherePointPicking.html

    Args:
        shape (int): required number of deviates. Should be acceptable by np.random.rand().
        radians (bool): whether or not to return in radians.
            Default: True

    Returns:
        two np.ndarray of random spherical deviates, for theta ~ [0, 2pi) and phi ~ [0, pi].

    """
    theta = 2 * np.pi * np.random.rand(shape)
    phi = np.arccos(2 * np.random.rand(shape) - 1)

    if radians:
        return theta, phi
    else:
        return theta * 180 / np.pi, phi * 180 / np.pi
