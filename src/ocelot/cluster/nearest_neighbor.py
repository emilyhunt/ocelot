"""A set of functions for nearest neighbor analysis of a clustering field. Especially useful for calculating optimum
DBSCAN/OPTICS epsilon parameters.
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors


def precalculate_nn_distances(
    data: np.ndarray,
    n_neighbors: int = 10,
    n_jobs: int = -1,
    return_sparse_matrix: bool = True,
    return_knn_distance_array: bool = False,
    **kwargs,
):
    """Pre-calculates nearest neighbor (nn) distances for direct plugging into a sklearn clustering algorithm with
    metric=pre-computed. Basically just a wrapper for sklearn.neighbors.NearestNeighbors.

    Args:
        data (np.ndarray): dataset of shape (n_samples, n_features) to calculate nearest neighbor distances for.
        n_neighbors (int): the number of neighbors to calculate upto. Should be as high as the highest min_samples you
            wish to use in your clustering algorithm.
            Default: 10
        n_jobs (int): number of jobs to use to calculate nearest neighbor distances.
            Default: -1 (uses all CPUs)
        return_sparse_matrix (bool): whether or not to return an (n_samples, n_samples) sparse matrix of nearest
            neighbor distances for feeding into a clustering algorithm (e.g. DBSCAN, HDBSCAN*, OPTICS...)
            Default: True
        return_knn_distance_array (bool): whether or not to also return an (n_samples, n_neighbors) distance array,
            useful for determining epsilon for DBSCAN or making nearest neighbor plots.
            Default: False
        **kwargs: any keyword arguments to pass to the sklearn.neighbors.NearestNeighbors constructor.

    Returns:
        a sparse matrix of nearest neighbor distances,
        (a np.ndarray distance array iff return_knn_distance_array is True)

    todo add a way to save output

    """
    # Initialise & fit
    neighbor_calculator = NearestNeighbors(
        n_neighbors=n_neighbors, n_jobs=n_jobs, **kwargs
    )
    neighbor_calculator.fit(data)

    # Return as requested
    if return_knn_distance_array:
        distances, indices = neighbor_calculator.kneighbors()

        if return_sparse_matrix:
            sparse_matrix = neighbor_calculator.kneighbors_graph(mode="distance")
            return sparse_matrix, distances
        else:
            return distances

    elif return_sparse_matrix:
        sparse_matrix = neighbor_calculator.kneighbors_graph(mode="distance")
        return sparse_matrix

    else:
        raise ValueError(
            "Nothing was specified for return. That's probably not intentional!"
        )
