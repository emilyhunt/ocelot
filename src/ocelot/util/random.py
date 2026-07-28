"""Functions for doing various random sampling problems that aren't otherwise defined."""

import numpy as np
from typing import Tuple
from ocelot.util.coordinates import spherical_to_cartesian


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


def unit_vectors(size: int, seed=None):
    """Returns an array of random unit vectors on the surface of a 3D unit sphere.

    Parameters
    ----------
    size : int
        Number of unit vectors to return.
    seed : _type_, optional
        Seed for the numpy random generator. Default: None

    Returns
    -------
    An array of shape (size, 3) of unit vectors.
    """
    theta, phi = points_on_sphere(size, phi_symmetric=False, seed=seed)
    x, y, z = spherical_to_cartesian(
        1.0, phi, theta
    )  # N.B. unhelpfully this function has different argument names
    return np.vstack((x, y, z)).T


def rotation_matrix(number: int, seed=None):
    """Creates random 3D rotation matrices. This is actually quite hard in practice,
    but is solved here with a QR decomposition of matrices of normal vectors.

    Parameters
    ----------
    size : int
            Number of rotation matrices to return.
    seed : _type_, optional
        Seed for the numpy random generator. Default: None

    Returns
    -------
        An array of shape (size, 3, 3) of rotation matrices.

    Notes
    -----
    Taken from the following: https://math.stackexchange.com/a/4832876

    See also:
    https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/1602779#1602779
    http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
    https://github.com/alecjacobson/gptoolbox/blob/master/matrix/rand_rotation.m
    """
    # TODO requires unit test
    rng = np.random.default_rng(seed=seed)

    z = rng.normal(size=(number, 3, 3))
    q, r = np.linalg.qr(z)
    sign = 2 * (np.diagonal(r, axis1=-2, axis2=-1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[:, 0, :] *= np.linalg.det(rot)[..., None]
    return rot


def fractal_noise_2d(resolution: int, seed: None):
    """Creates 2d fractal (pink) noise.

    Implemented for use in making synthetic (correlated) differential reddening.

    This function is completely beyond me. Thanks, StackOverflow!
    https://stackoverflow.com/a/76605642
    """
    rng = np.random.default_rng(seed)

    # Create white noise
    whitenoise = rng.uniform(0, 1, (resolution, resolution))

    # Generate frequency matrix
    ft_arr = np.fft.fftshift(np.fft.fft2(whitenoise))
    _x, _y = np.mgrid[0 : ft_arr.shape[0], 0 : ft_arr.shape[1]]
    f = np.hypot(_x - ft_arr.shape[0] / 2, _y - ft_arr.shape[1] / 2)

    # Convert to fractal noise
    fractal_fourier_transform = np.zeros_like(ft_arr)
    nonzero_denominator = (f != 0).nonzero()
    fractal_fourier_transform[nonzero_denominator] = (
        ft_arr[nonzero_denominator] / f[nonzero_denominator]
    )
    fractal_noise = np.fft.ifft2(np.fft.ifftshift(fractal_fourier_transform)).real

    # Rescale to have unit variance
    fractal_noise = fractal_noise / np.std(fractal_noise)
    return fractal_noise
