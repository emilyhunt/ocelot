"""A set of functions for calculating optimum DBSCAN/OPTICS epsilon parameters of a field."""

import gc
from typing import Union

import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize, curve_fit
from .nearest_neighbor import precalculate_nn_distances
from ..plot import nearest_neighbor_distances


def acg18(data_clustering: np.ndarray, nn_distances: np.ndarray, n_repeats: int = 10,
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
    acg_epsilon = (mean_random_epsilons + epsilon_minimum) / 2

    if return_last_random_distance:
        return acg_epsilon, random_nn_distances
    else:
        return acg_epsilon


def _kth_nn_distribution(r_range, a, dimension, k):
    """Returns the kth nearest neighbor distribution for a multi-dimensional ideal gas. Not normalised!

    f = r_range^(dimension + k - 1) / a^dimension * exp(-(r_range/a)^dimension)

    Args:
        r_range: radius values away from the centre to evaluate at.
        a: the fitting constant
        k: the kth nearest neighbor moment of the distribution
        dimension: the assumed dimensionality of the distribution

    Returns:
        np.ndarray of the distribution evaluated at r_range

    """
    return r_range ** (dimension + k - 1) / a ** dimension * np.exp(-(r_range / a) ** dimension)


def _summed_kth_nn_distribution_one_cluster_curve_fit(r_range, *parameters, k=10):

    # Calculate cumulatively summed (and normalised) distributions for both the field and the cluster
    y_field = np.cumsum(_kth_nn_distribution(r_range, parameters[0], parameters[1], k))
    normalisation_field = np.trapz(y_field, x=r_range) / (1 - parameters[4])

    y_cluster = np.cumsum(_kth_nn_distribution(r_range, parameters[2], parameters[3], k))
    normalisation_cluster = np.trapz(y_cluster, x=r_range) / parameters[4]

    y_field /= normalisation_field
    y_cluster /= normalisation_cluster
    y_total = y_field + y_cluster

    return np.log10(y_total)


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
    # Return inf if the parameters are wrong
    if np.any(parameters[:4] <= 0) or parameters[4] < 0:
        return np.inf
    if parameters[2] >= parameters[0]:
        return np.inf

    # Calculate cumulatively summed (and normalised) distributions for both the field and the cluster
    y_field = np.cumsum(_kth_nn_distribution(r_range, parameters[0], parameters[1], k))
    normalisation_field = np.trapz(y_field, x=r_range) / (1 - parameters[4])

    y_cluster = np.cumsum(_kth_nn_distribution(r_range, parameters[2], parameters[3], k))
    normalisation_cluster = np.trapz(y_cluster, x=r_range) / parameters[4]

    # Stop and return inf if the areas aren't valid
    if normalisation_field <= 0 or normalisation_cluster <= 0 \
            or np.all(np.isfinite(y_field)) is False or np.all(np.isfinite(y_cluster)) is False:
        return np.inf

    y_field /= normalisation_field
    y_cluster /= normalisation_cluster
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

        # Take logs only where log() is defined, otherwise replace with np.inf
        log_array = np.where(good_values, np.log10(log_array, where=good_values), np.inf)

        return log_array


def _get_epsilon_plotting_styles(epsilon_values):
    """Quick function to neaten my code and get default plotting styles for epsilon values for a diagnostic plot.

    Args:
        epsilon_values (list-like): epsilon values to plot, in the usual order (0, 1, 2)

    Returns:
        a new functions_to_overplot list

    """
    return [
            {'label': f'eps0: {epsilon_values[0]:.4f}',
             'style': 'k:',
             'x': [epsilon_values[0]] * 2,
             'y': [1e-300, 1e300],
             'differentiate': False},

            {'label': f'eps1: {epsilon_values[1]:.4f}',
             'style': 'k-.',
             'x': [epsilon_values[1]] * 2,
             'y': [1e-300, 1e300],
             'differentiate': False},

            {'label': f'eps2: {epsilon_values[2]:.4f}',
             'style': 'k:',
             'x': [epsilon_values[2]] * 2,
             'y': [1e-300, 1e300],
             'differentiate': False}
            ]


def _get_model_plotting_styles(x_range, y_field, y_cluster, y_total):
    """Quick function to neaten my code and get default plotting styles for epsilon values for a diagnostic plot.

    Args:
        x_range, y_field, y_cluster, y_total: self-explanatory tbh. Everyone shares an x_range!

    Returns:
        a new functions_to_overplot list

    """
    return [
            {'label': f'field model',
             'style': 'm--',
             'x': x_range,
             'y': 10**y_field,
             'differentiate': True},

            {'label': f'cluster model',
             'style': 'c--',
             'x': x_range,
             'y': 10**y_cluster,
             'differentiate': True},

            {'label': f'total model',
             'style': 'r-',
             'x': x_range,
             'y': 10**y_total,
             'differentiate': True},
            ]


def _find_sign_change_epsilons(x_range: np.ndarray, y_range: np.ndarray, return_all_sign_changes: bool = False):
    """Takes derivatives of a numerical cluster model to find the beginning, middle and end of the area with the
    steepest change in gradient, which typically corresponds to the point at which field stars begin to dominate.

    log point number
        ^            ________
        |           /
        |    ______/
        |   /     ^^^
        |  / eps: 012
        |____________________>
    kth nearest neighbour distance

    Notes:
        - DO NOT CALL THIS FUNCTION if cluster_fraction is extremely low: it may or may not find an epsilon value (from
            floating point errors) and that value may or may not be completely stupid. Be careful!
        - In initial development, I raised x_range to the power of ten to weight peaks further from epsilon=0 more
            highly. Once I switched to using models, this wasn't necessary, as the derivative was comparatively much
            more stable. However, that might just be a function of later testing. Worth bearing in mind if it suddenly
            doesn't work well. I think this was especially an issue if the first few points were noisy (and hence had an
            anomalously high area under their 2nd derivative.)

    Args:
        x_range (np.ndarray): x values of the calculated fit.
        y_range (np.ndarray): y values of the calculated fit.
        return_all_sign_changes (bool): whether or not to also return a list of all sign changes found. Useful for
            debugging.
            Default: False

    Returns:
        - a list containing epsilon 0, 1 and 2
        - return_all_sign_changes=True: also returns an array of all sign changes found

    """
    # Take the second derivative
    d2_y_range = np.gradient(np.gradient(y_range, x_range), x_range)

    # Find & output some cool values
    # Firstly, find all candidate 2nd derivative = 0 points (epsilon 0 or 2) by looking for sign changes, based on
    # https://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-for-elements-in-a-numpy-array
    sign_change_ids = np.asarray(np.sign(d2_y_range[:-1]) != np.sign(d2_y_range[1:])).nonzero()[0] + 1
    sign_change_x_values = x_range[sign_change_ids]

    # We make a clipped d2_y_range so that anything under the curve is ignored
    clipped_d2_y_range = np.clip(d2_y_range, 0, np.inf)

    # Cycle over pairs and record which one has the highest area below it
    i = 0
    epsilon_0_and_2_ids = np.zeros(2, dtype=int)
    epsilon_0_and_2_x_values = np.zeros(2)
    max_area = -1.

    while i < sign_change_x_values.shape[0] - 1:
        id_0 = sign_change_ids[i]
        id_1 = sign_change_ids[i + 1]

        area = np.trapz(clipped_d2_y_range[id_0:id_1 + 1], x=x_range[id_0:id_1 + 1])

        # Save this pair if it's the best
        if area > max_area:
            epsilon_0_and_2_ids = np.array([id_0, id_1])
            epsilon_0_and_2_x_values = sign_change_x_values[[i, i + 1]]
            max_area = area

        i += 1

    # Next, find the point of maximum curvature between eps0 and eps2 by finding the maximum value of the 2nd derivative
    # But we only do that if there were even any sign changes ( => i == 0 since no incrementation has happened)
    if i == 0:
        epsilon_1_x_value = 0.0
    else:
        epsilon_1_id = np.argmax(d2_y_range[epsilon_0_and_2_ids[0]:epsilon_0_and_2_ids[1] + 1])
        epsilon_1_x_value = (x_range[epsilon_0_and_2_ids[0]:epsilon_0_and_2_ids[1] + 1])[epsilon_1_id]

    # Happy return time =)
    if return_all_sign_changes:
        return [epsilon_0_and_2_x_values[0], epsilon_1_x_value, epsilon_0_and_2_x_values[1]], sign_change_x_values
    else:
        return [epsilon_0_and_2_x_values[0], epsilon_1_x_value, epsilon_0_and_2_x_values[1]]


def field_model(nn_distances: np.ndarray, min_samples: int = 10, min_cluster_size: int = 1,
                resolution: int = 500, point_fraction_to_keep: float = 0.95, max_iterations: int = 2000,
                optimiser: str ='Powell', print_convergence_messages: bool = False,
                make_diagnostic_plot: bool = False, **kwargs) -> Union[bool, tuple, tuple]:
    """Attempts to find an optimum value for epsilon by modelling the field of the cluster and the cluster itself.
    Leverages scipy minimisation to find optimum model values, and can even report on the approximate estimated size
    of a cluster in the given field. Will fail if the signature of the cluster is extremely weak.

    Args:
        nn_distances (np.ndarray): nearest neighbor distances for the field, in shape
            (n_samples, max_neighbors_to_calculate).
        min_samples (int): number of minimum samples to find the epsilon for (aka the kth nearest neighbor).
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
            required option. BFGS is faster, while Powell and Nelder-Mead tend to be more reliable (tested for
            min_samples=10). Powell is the best at finding a global minimum, & BFGS and NM often have the same answer.
            Default: 'Powell'
        print_convergence_messages (bool): whether or not to ask scipy.optimize.minimize to print convergence messages.
            Default: False
        make_diagnostic_plot (bool): whether or not to make a diagnostic plot with
            ocelot.plot.nearest_neighbor_distances of the results we've got.
            Default: False
        **kwargs: keyword arguments to pass to ocelot.plot.nearest_neighbor_distances

    Returns:
        - bool for whether or not a cluster was found
        - a tuple containing epsilon 0, 1 and 2 estimates
        - a tuple of the fitting parameters found
        - the expected number of cluster members, n_cluster_members

    """
    # -- Pre-processing
    # Grab the correct neighbor distances, sort them and drop stuff we don't want
    distances = np.sort(nn_distances[:, min_samples - 1])
    distances = distances[:int(point_fraction_to_keep * distances.shape[0])]

    # Create a normalised log number of points array
    points = np.arange(1, distances.shape[0] + 1)
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

    # Minimisation time! Parameters is the stuff to minimise, arguments is the stuff we pass to the function to use it
    parameters = np.asarray([field_constant, field_dimension, cluster_constant, cluster_dimension, cluster_fraction])
    arguments = (min_samples, distances_interpolated, points_interpolated, True)

    result = minimize(_summed_kth_nn_distribution_one_cluster,
                      parameters,
                      args=arguments,
                      method=optimiser,
                      options={"disp": print_convergence_messages,
                               "maxiter": max_iterations})

    # Currently just raises an error if we had any issues
    if result.success is False:
        raise RuntimeError("Minimisation was unsuccessful! scipy.optimize.minimize did not achieve convergence "
                           f"after {result.nit} steps.\n  Termination message:{result.message}")

    # -- Calculation and grabbing of epsilon values
    # We'll evaluate the function at the results that were grabbed, for plotting and epsilon purposes
    points_field, points_cluster, points_total = _summed_kth_nn_distribution_one_cluster(
        result.x, min_samples, distances_interpolated, minimisation_mode=False)

    # Calculate the estimated number of cluster members and calculate epsilon if it's equal or above the minimum size
    n_cluster_members = int(np.round(distances.shape[0] * result.x[-1]))
    if n_cluster_members >= min_cluster_size:
        epsilon_values = _find_sign_change_epsilons(distances_interpolated, points_total)
        functions_to_overplot_epsilon_values = _get_epsilon_plotting_styles(epsilon_values)
    else:
        epsilon_values = None
        functions_to_overplot_epsilon_values = []

    # -- Diagnostic plotting
    # Diagnostic plot if we're asked nicely
    if make_diagnostic_plot:
        functions_to_overplot = _get_model_plotting_styles(
            distances_interpolated, points_field, points_cluster, points_total)
        functions_to_overplot += functions_to_overplot_epsilon_values

        # We'll only want to normalise the raw data - everything else is already done
        normalisation_constants = [1.] + [0] * len(functions_to_overplot)

        # Plot time! We make a cheeky distance array so that the stuff we pass to the plotting function is the right
        # shape.
        distances_to_pass = np.zeros((distances.shape[0], min_samples))
        distances_to_pass[:, min_samples - 1] = distances
        nearest_neighbor_distances(distances_to_pass,
                                   neighbor_to_plot=min_samples,
                                   normalisation_constants=normalisation_constants,
                                   functions_to_overplot=functions_to_overplot,
                                   **kwargs)

    # Return nada if no cluster was found
    if n_cluster_members < min_cluster_size:
        return False, (0., 0., 0.), result.x, n_cluster_members
    else:
        return True, epsilon_values, result.x, n_cluster_members


def field_model_cf(nn_distances: np.ndarray, min_samples: int = 10, min_cluster_size: int = 1,
                   resolution: int = 500, point_fraction_to_keep: float = 0.95, max_iterations: int = 2000,
                   optimiser: str ='trf', print_convergence_messages: bool = False,
                   make_diagnostic_plot: bool = False, **kwargs) -> Union[bool, tuple, tuple]:
    """Attempts to find an optimum value for epsilon by modelling the field of the cluster and the cluster itself.
    Leverages scipy minimisation to find optimum model values, and can even report on the approximate estimated size
    of a cluster in the given field. Will fail if the signature of the cluster is extremely weak.

    Args:
        nn_distances (np.ndarray): nearest neighbor distances for the field, in shape
            (n_samples, max_neighbors_to_calculate).
        min_samples (int): number of minimum samples to find the epsilon for (aka the kth nearest neighbor).
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
            required option. BFGS is faster, while Powell and Nelder-Mead tend to be more reliable (tested for
            min_samples=10). Powell is the best at finding a global minimum, & BFGS and NM often have the same answer.
            Default: 'Powell'
        print_convergence_messages (bool): whether or not to ask scipy.optimize.minimize to print convergence messages.
            Default: False
        make_diagnostic_plot (bool): whether or not to make a diagnostic plot with
            ocelot.plot.nearest_neighbor_distances of the results we've got.
            Default: False
        **kwargs: keyword arguments to pass to ocelot.plot.nearest_neighbor_distances

    Returns:
        - bool for whether or not a cluster was found
        - a tuple containing epsilon 0, 1 and 2 estimates
        - a tuple of the fitting parameters found
        - the expected number of cluster members, n_cluster_members

    """
    # -- Pre-processing
    # Grab the correct neighbor distances, sort them and drop stuff we don't want
    distances = np.sort(nn_distances[:, min_samples - 1])
    distances = distances[:int(point_fraction_to_keep * distances.shape[0])]

    # Create a normalised log number of points array
    points = np.arange(1, distances.shape[0] + 1)
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

    # Minimisation time! Parameters is the stuff to minimise, arguments is the stuff we pass to the function to use it
    parameters = np.asarray([field_constant, field_dimension, cluster_constant, cluster_dimension, cluster_fraction])
    bounds = (np.asarray([0.0, 0.0, 0.0, 0.0, 0.0]),
              np.asarray([np.inf, np.inf, np.inf, np.inf, 1.0]))

    #arguments = (min_samples, distances_interpolated, points_interpolated, True)

    result, covariance = curve_fit(_summed_kth_nn_distribution_one_cluster_curve_fit,
                                   distances_interpolated,
                                   points_interpolated,
                                   p0=parameters,
                                   bounds=bounds,
                                   method=optimiser,
                                   verbose=print_convergence_messages)

    # -- Calculation and grabbing of epsilon values
    # We'll evaluate the function at the results that were grabbed, for plotting and epsilon purposes
    points_field, points_cluster, points_total = _summed_kth_nn_distribution_one_cluster(
        result, min_samples, distances_interpolated, minimisation_mode=False)

    # Calculate the estimated number of cluster members and calculate epsilon if it's equal or above the minimum size
    n_cluster_members = int(np.round(distances.shape[0] * result[-1]))
    if n_cluster_members >= min_cluster_size:
        epsilon_values = _find_sign_change_epsilons(distances_interpolated, points_total)
        functions_to_overplot_epsilon_values = _get_epsilon_plotting_styles(epsilon_values)
    else:
        epsilon_values = None
        functions_to_overplot_epsilon_values = []

    # -- Diagnostic plotting
    # Diagnostic plot if we're asked nicely
    if make_diagnostic_plot:
        functions_to_overplot = _get_model_plotting_styles(
            distances_interpolated, points_field, points_cluster, points_total)
        functions_to_overplot += functions_to_overplot_epsilon_values

        # We'll only want to normalise the raw data - everything else is already done
        normalisation_constants = [1.] + [0] * len(functions_to_overplot)

        # Plot time! We make a cheeky distance array so that the stuff we pass to the plotting function is the right
        # shape.
        distances_to_pass = np.zeros((distances.shape[0], min_samples))
        distances_to_pass[:, min_samples - 1] = distances
        nearest_neighbor_distances(distances_to_pass,
                                   neighbor_to_plot=min_samples,
                                   normalisation_constants=normalisation_constants,
                                   functions_to_overplot=functions_to_overplot,
                                   **kwargs)

    # Return nada if no cluster was found
    if n_cluster_members < min_cluster_size:
        return False, (0., 0., 0.), result, n_cluster_members
    else:
        return True, epsilon_values, result, n_cluster_members


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
