"""A set of tools for post-processing data ready for plotting."""

from typing import Optional, Union

import numpy as np
from sklearn.neighbors import KernelDensity


def kde_fit_2d(xy: np.ndarray,
               kde_bandwidth: float,
               x_limits: Optional[Union[list, np.ndarray]] = None,
               y_limits: Optional[Union[list, np.ndarray]] = None,
               kde_resolution: int = 50):
    """Fits a Kernel Density Estimator (KDE) in 2 dimensions with a Gaussian
    kernel of specified bandwidth.

    # Todo: This could easily be generalised to be multi-dimensional, simply by using the dimensions of xy as a hint.

    Notes:
        Do not use x and y limits to restrict the data set - only to reduce the
        area for plotting output. A fit will still be performed across the
        entire data range, which could get very unoptimised if you only need to
        fit a small segment of the dataset!

        KDE runtime scales as O(nm), where n is the number of samples and m is
        the number of points to evaluate at. Reducing the dataset size or the
        resolution of scoring will result in faster running.

    Args:
        xy (np.ndarray): the x and y data, of shape (n_samples, n_features).
        kde_bandwidth: the bandwidth to be used by the KDE. Recommended: about 0.08.
        x_limits (list-like[floats], optional): x limits of the final KDE scoring grid.
        y_limits (list-like[floats], optional): y limits of the final KDE scoring grid.
        kde_resolution (int): number of points to sample the KDE at when scoring
            (across a resolution x resolution-sized grid.) Default: 50.

    Returns:
        [2D array of x values of scores,
         2D array of y values of scores,
         2D array of scores]

    """
    # Use minimum and maximum values if limits haven't been specified.
    if x_limits is None:
        x_limits = np.array([xy[:, 0].min(), xy[:, 0].max()])

    if y_limits is None:
        y_limits = np.array([xy[:, 1].min(), xy[:, 1].max()])

    # Make up a mesh grid based on the limits of the dataset, so we can do some
    # density plotting in a moment
    grid_x = np.linspace(x_limits[0], x_limits[1], kde_resolution)
    grid_y = np.linspace(y_limits[0], y_limits[1], kde_resolution)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

    # Fit the KDE
    xy_kde = KernelDensity(bandwidth=kde_bandwidth,
                           metric='euclidean',
                           kernel='gaussian',
                           algorithm='ball_tree')
    xy_kde.fit(xy)

    # Evaluate the KDE on the meshes, inputting an argument of shape
    # (n_samples, n_features) to the sklearn kde
    z = xy_kde.score_samples(np.array([mesh_x.flatten(), mesh_y.flatten()]).T)

    # Re-normalise as the scores returned are logarithms
    z = np.exp(z)

    # Re-shape the scores to the right shape again
    z = z.reshape(mesh_x.shape)

    return [mesh_x, mesh_y, z]
