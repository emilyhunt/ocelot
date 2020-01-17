"""A set of functions for calculating optimum DBSCAN/OPTICS epsilon parameters of a field."""

import gc
from typing import Union

import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize
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


def _kth_nn_distribution(r_range, a, dimension, k):
    """Returns the kth nearest neighbor distribution for a multi-dimensional ideal gas. Not normalised!

    f = r_range^(dimension + k - 1) / a^dimension * exp(-(r_range/a)^dimension)

    Args:
        r_range: radius values away from the centre to evaluate at.
        k: the kth nearest neighbor moment of the distribution
        a: the fitting constant
        dimension: the assumed dimensionality of the distribution

    Returns:
        np.ndarray of the distribution evaluated at r_range

    """
    return r_range ** (dimension + k - 1) / a ** dimension * np.exp(-(r_range / a) ** dimension)


def _summed_kth_nn_distribution_one_cluster(parameters: np.ndarray, k: int, r_range: np.ndarray,
                                            y_range: np.ndarray = None, minimisation_mode: bool = True):
    """Returns the summer kth nearest neighbor distribution, assuming the field contains at most one cluster.

    Args:
        parameters (np.ndarray): parameters of the model of length 5, in the form:
            0: field_constant (also known as a)
            1: field_dimension
            2: cluster_constant (also known as a)
            3: cluster_dimension
            4: cluster_fraction
        r_range (np.ndarray): radius values away from the centre to evaluate at.
        y_range (np.ndarray): log10 points values to compare the model to. Must be specified if minimisation_mode=True.
            Default: None
        k (int): the kth nearest neighbor moment of the distribution
        minimisation_mode (bool): whether or not to just return a single residual value. Otherwise, returns an array
            of y_field, y_cluster and y_total.
            Default: True

    Returns:
        minimisation_mode =
            True: a single residual value
            False: an array of y_field, y_cluster and y_total.

    """
    # Calculate cumulatively summed (and normalised) distributions for both the field and the cluster
    # Todo negative areas here will fuck things up - need a check
    y_field = np.cumsum(_kth_nn_distribution(r_range, parameters[0], parameters[1], k))
    y_field /= np.trapz(y_field, x=r_range) / (1 - parameters[4])

    y_cluster = np.cumsum(_kth_nn_distribution(r_range, parameters[2], parameters[3], k))
    y_cluster /= np.trapz(y_cluster, x=r_range) / parameters[4]

    y_total = y_field + y_cluster

    # If we're minimising, we want to decide whether or not to take logs _fast_
    if minimisation_mode:
        if np.any(y_total <= 0):
            return np.inf
        else:
            return np.sum((np.log10(y_total) - y_range)**2)

    # Otherwise, we'll return raw values to be used by a plotter (slow due to the initialisation process, amongst other
    # things)
    else:
        # Make a big array to work on
        log_array = np.vstack([y_field, y_cluster, y_total])
        good_values = log_array > 0
        bad_values = np.invert(good_values)

        # Take logs only where log() is defined
        log_array[good_values] = np.log10(log_array, where=good_values)
        log_array[bad_values] = np.inf

        return log_array


def field_model_epsilon(nn_distances: np.ndarray, min_samples: int = 10, min_cluster_size: int = 1,
                        resolution: int = 500, point_fraction_to_keep: float = 0.95, max_iterations: int = 2000,
                        optimiser='BFGS'):
    """Attempts to find an optimum value for epsilon by modelling the field of the cluster and the cluster itself.
    Leverages scipy minimisation to find optimum model values, and can even report on the approximate estimated size
    of a cluster in the given field. Will fail if the signature of the cluster is extremely weak.

    Args:
        nn_distances (np.ndarray): nearest neighbor distances for the field, in shape
            (n_samples, max_neighbors_to_calculate).
        min_samples (int, str): number of minimum samples to find the acg18 epsilon for (aka the kth nearest neighbor).
            May be an integer or 'all'.
            Default: 10
        min_cluster_size (int): minimum allowed size of a cluster, based on the value of cluster_fraction derived in
            the fitting procedure. Setting this larger can help to avoid high epsilons that return noise clusters.
            Default: 1 (virtually equivalent to setting this to off)
        resolution (int): resolution to re-sample the data to. Should be high enough that all detail is kept, but not
            so high as to drastically slow down the program.
            Default: 500
        point_fraction_to_keep (float): for efficiency reasons, points with a very high epsilon should be dropped. This
            makes re-sampling the data require far fewer points and ensures the minimiser will focus more on the cluster
            (at low epsilon.) The bottom point_fraction_to_keep fraction of points is kept.
            Default: 0.95 (i.e. 5% of points with the highest epsilon are removed, a good general value)
        max_iterations (int): maximum number of iterations to run the optimiser for before quitting. Especially for slow
            algorithms this shouldn't be too high.
        optimiser (string): optimiser to be used by scipy.optimize.minimize. Must be an unconstrained, no gradient
            required option. BFGS is faster, while Powell and Nelder-Mead tend to be more reliable.
            Default: 'BFGS'

    """
    # -- Pre-processing
    # Grab the correct neighbor distances, sort them and drop stuff we don't want
    distances = np.sort(nn_distances[:min_samples - 1])
    distances = distances[:int(point_fraction_to_keep * distances.shape[0])]

    # Create a normalised log number of points array
    points = np.arange(distances.shape[0]) + 1
    points = points / np.trapz(points, x=distances)
    points = np.log10(points)

    # Interpolate it to ensure the points are linearly sampled and reduce noise
    interpolator = interp1d(distances, points, kind='linear')
    distances_interpolated = np.linspace(distances.min(), distances.max(), num=resolution)
    points_interpolated = interpolator(distances_interpolated)

    # -- Fitting
    # Grab an initial guess
    field_constant = 0.3
    field_dimension = 5
    cluster_constant = 0.05
    cluster_dimension = 3
    cluster_fraction = 0.01

    # Minimisation time!
    parameters = np.asarray(field_constant, field_dimension, cluster_constant, cluster_dimension, cluster_fraction)
    arguments = [min_samples, distances_interpolated, points_interpolated, True]
    result = minimize(_summed_kth_nn_distribution_one_cluster,
                      parameters,
                      args=arguments)


    # Todo - stop point on the Friday
    pass


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
