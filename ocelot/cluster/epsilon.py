"""A set of functions for calculating optimum DBSCAN/OPTICS epsilon parameters of a field."""

import gc
from typing import Union

import numpy as np

from .nearest_neighbor import precalculate_nn_distances


def acg18_epsilon(data_clustering: np.ndarray, nn_distances: np.ndarray, n_repeats: int = 10,
                  min_samples: Union[str, int] = 10, return_last_random_distance: bool = False):
    """A method for calculating an optimal epsilon value as in Alfredo Castro-Ginard's 2018 paper (hence the acronym
    acg18.)

    Args:
        data_clustering (np.ndarray): clustering data for the field in shape (n_samples, n_features).
        nn_distances (np.ndarray): nearest neighbor distances for the field, in shape
            (n_samples, max_neighbors_to_calculate).
        n_repeats (int): number of random repeats to perform.
            Default: 10
        min_samples (int, str): number of minimum samples to find the acg18 epsilon for (aka the kth nearest neighbor).
            May be an integer or 'all'.
            Default: 10
        return_last_random_distance (bool): whether or not to return the final random distance set made by the function.
            Useful for making plots & understanding what's going on internally.
            Default: False

    Returns:
        - single float or np.ndarray of optimal epsilon values
        - if return_last_random_distance: an array of shape nn_distances.shape of the distances between stars in the
            last random distances simulation.

    """
    if n_repeats < 1:
        raise ValueError("A positive number of repeats must be specified.")

    # Infer how many neighbors we need to calculate
    max_neighbors_to_calculate = nn_distances.shape[1]

    # Calculate nearest neighbor distances if they haven't been passed to the function already
    # REMOVED as this fucks with inferring max_neighbors_to_calculate and it should always be calculated by the user
    # already anyway.
    # if nn_distances is None:
    #     nn_distances = precalculate_nn_distances(data_clustering, n_neighbors=max_neighbors_to_calculate,
    #                                              return_sparse_matrix=False, return_knn_distance_array=True)

    # Grab the minimum epsilon in the unperturbed field & do some error checking of min_samples in the process.
    if min_samples == 'all':
        epsilon_minimum = np.min(nn_distances, axis=0)

    elif type(min_samples) is not int:
        raise ValueError("Incompatible number or string of min_samples specified.\n"
                         "Allowed values:\n"
                         "- integer less than max_neighbors_to_calculate and greater than zero\n"
                         "- 'all', which calculates all values upto max_neighbors_to_calculate\n")

    elif min_samples > max_neighbors_to_calculate or min_samples < 1:
        raise ValueError("min_samples may not be larger than max_neighbors_to_calculate (aka nn_distances.shape[1]) "
                         "and must be a positive integer.")

    else:
        epsilon_minimum = np.min(nn_distances[:, min_samples - 1])

    # Semi-paranoid memory management (lol)
    del nn_distances
    gc.collect()

    # Cycle over the required number of random repeats
    random_epsilons = np.zeros((n_repeats, max_neighbors_to_calculate))
    random_nn_distances = None  # Done solely to shut up my fucking linter
    i = 0
    while i < n_repeats:

        # Shuffle all values in the dataset column-wise, which is annoying to do but I guess I'll manage :L
        current_axis = 0
        while current_axis < data_clustering.shape[1]:
            data_clustering[:, current_axis] = np.random.permutation(data_clustering[:, current_axis])
            current_axis += 1

        # Get some nn distances
        random_nn_distances = precalculate_nn_distances(data_clustering, n_neighbors=max_neighbors_to_calculate,
                                                        return_sparse_matrix=False, return_knn_distance_array=True)

        random_epsilons[i, :] = np.min(random_nn_distances, axis=0)

        i += 1

        # Semi-paranoid memory management (lol)
        if return_last_random_distance is False or i != n_repeats:
            del random_nn_distances
            gc.collect()

    # Ignore random epsilons we don't need if the user doesn't want them all
    if min_samples != 'all':
        random_epsilons = random_epsilons[:, min_samples - 1]

    # Find the mean random epsilon & acg 18 epsilon
    mean_random_epsilons = np.mean(random_epsilons, axis=0)
    acg_epsilon = (mean_random_epsilons - epsilon_minimum) / 2

    if return_last_random_distance:
        return acg_epsilon, random_nn_distances
    else:
        return acg_epsilon


def maximum_curvature_epsilon():
    """Attempts to find optimum epsilon estimates with a numerical second derivative of a point number vs. epsilon
    plot. The main challenge is finding said numerical derivative in a stable way, so a number of different methods
    are available.

    todo


    """
    pass


def calculate_epsilon(distances, ):
    """The main method for calculating a number of different optimum epsilon values for a nearest neighbor field.
    Can also produce nearest neighbor plots if desired, calling functionality from ocelot.plot.

    todo

    """
    pass
