"""A set of functions for nearest neighbor analysis of a clustering field. Especially useful for calculating optimum
DBSCAN/OPTICS epsilon parameters.
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors
from typing import Union, Optional, List


def precalculate_nn_distances(data: np.ndarray, n_neighbors: int = 10, n_jobs: int = -1,
                              return_knn_distance_array: bool = False, **kwargs):
    """Pre-calculates nearest neighbor (nn) distances for direct plugging into a sklearn clustering algorithm with
    metric=pre-computed.

    Args:
        data (np.ndarray): dataset of shape (n_samples, n_features) to calculate nearest neighbor distances for.
        n_neighbors (int): the number of neighbors to calculate upto. Should be as high as the highest min_samples you
            wish to use in your clustering algorithm.
            Default: 10
        n_jobs (int): number of jobs to use to calculate nearest neighbor distances.
            Default: -1 (uses all CPUs)
        return_knn_distance_array (bool): whether or not to also return an (n_samples, n_neighbors) distance array,
            useful for determining epsilon for DBSCAN or making nearest neighbor plots.
            Default: False

    Returns:
        a sparse matrix of nearest neighbor distances,
        (a np.ndarray distance array iff return_knn_distance_array is True)

    """
    # Initialise & fit
    neighbor_calculator = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs, **kwargs)
    neighbor_calculator.fit(data)

    # Create a sparse matrix of distances to the nearest neighbors that can be fed to clustering algorithms
    sparse_matrix = neighbor_calculator.kneighbors_graph(mode='distance')

    # If requested, also return the distances to the nearest points but in sorted order. This is useful for making plots
    # of nearest neighbor distances for a given field, e.g. for DBSCAN epsilon determination.
    if return_knn_distance_array:
        distances, indices = neighbor_calculator.kneighbors()
        return sparse_matrix, distances
    else:
        return sparse_matrix


def calculate_epsilon():
    """A method for calculating a number of different optimum epsilon values for a nearest neighbor field. Can also
    produce nearest neighbor plots if desired, calling functionality from ocelot.plot"""
    pass
