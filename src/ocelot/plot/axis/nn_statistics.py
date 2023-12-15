"""Various possible axes for use in the analysis of nearest neighbor distances (e.g. when running DBSCAN.)"""

from typing import Optional, Union

import numpy as np

from ..utilities import normalise_a_curve, percentile_based_plot_limits


def point_number_vs_nn_distance(axes, distances: np.ndarray,
                                neighbor_to_plot: int = 1,
                                normalisation_constants: Union[list, tuple, np.ndarray] = (1.,),
                                functions_to_overplot: Optional[list] = None,
                                show_numerical_derivatives: bool = False,
                                y_percentile_limits: Optional[np.ndarray] = None, ):
    """Plots point number vs nearest neighbor distance, with the option to add multiple regions to the plot (like
    highlighting an area of maximum curvature.)

    Notes:
        - to overplot a fitting function, all of fitting_functions_x, fitting_functions_y and fitting_function_styles
            must be specified.

    Args:
        axes (list of matplotlib axes): the axes to plot on. Each additional axis will have another level of
            derivative plotted on it.
        distances (np.ndarray): the kth nearest neighbor distance array, of shape (n_samples, n_neighbors_calculated.)
        neighbor_to_plot (int): kth nearest neighbor to plot. Must be lower than n_neighbors_calculated.
            Default: 1
        normalisation_constants (list-like of floats): normalisation constants to use for the curves. If zero,
            no normalisation will be performed. If a single float, all curves will be normalised to the same area. If a
            list of floats, all curves will have a different normalisation constant applied (extremely useful for
            plotting multiple components of a single model curve.) Normalisation is performed by the trapezium rule.
            Default: 1., i.e. all curves normalised to area 1
        functions_to_overplot (optional, list of dictionaries): list of fitting function information to overplot. Each
            dictionary should have:
            'x': x values (not logged)
            'y': y values (not logged)
            'style': matplotlib style, e.g. 'r--'
            'label': label for the curve or None
            'differentiate': whether to differentiate this each time (True/False). You probably want True, unless this
                is a specific value.
            Default: None
        show_numerical_derivatives (bool): whether or not to show the numerical derivatives of the field curve. They are
            typically very noisy so this isn't desired behaviour.
            Default: False
        y_percentile_limits (optional, list-like): percentile limits to apply to each derivative axis. Should be a
            np.ndarray of shape (n_extra_features - 1, 2), containing floats in the range [0, 100] and strictly ascending. Will be
            ignored for all axes that contain real points.

    Returns:
         the modified matplotlib axis.

    """
    # Is the user a fuckwit? Let's find out!
    if neighbor_to_plot > distances.shape[1]:
        raise ValueError("neighbor_to_plot cannot be higher than the number of calculated nearest neighbors!")

    # Count the number of fitting functions
    if functions_to_overplot is not None:
        n_fitting_functions = len(functions_to_overplot)
    else:
        n_fitting_functions = 0

    # Make sure that y_percentile_limits exists if it needs to if the user hasn't already specified limits
    if show_numerical_derivatives is False and n_fitting_functions > 0:
        infer_percentile_limits = True
        if y_percentile_limits is None:
            y_percentile_limits = np.tile(np.asarray([0, 100]), (n_fitting_functions, 1))
    else:
        infer_percentile_limits = False

    # Make sure the axes are a list and count the number of axes
    if type(axes) is not np.ndarray:
        axes = np.asarray([axes])
    n_axes = len(axes)

    # Grab some useful values
    nearest_neighbor_distances = np.sort(distances[:, neighbor_to_plot - 1])
    point_numbers = np.arange(nearest_neighbor_distances.shape[0]) + 1

    # Normalisation & log time
    point_numbers = normalise_a_curve(nearest_neighbor_distances, point_numbers, normalisation_constants[0])
    nearest_neighbor_distances = np.log10(nearest_neighbor_distances)
    point_numbers = np.log10(point_numbers)

    # Normalise and log the fitting functions too! We do this safely as there are more ways in which this could be
    # passed an invalid value expected (instead of the actually-physical real distances).
    i = 0
    while i < n_fitting_functions:
        # We'll only want to work on good stars
        good_stars = np.logical_and(
            np.logical_and(np.isfinite(functions_to_overplot[i]['x']), np.asarray(functions_to_overplot[i]['x']) > 0),
            np.logical_and(np.isfinite(functions_to_overplot[i]['y']), np.asarray(functions_to_overplot[i]['y']) > 0))

        functions_to_overplot[i]['x'] = np.asarray(functions_to_overplot[i]['x'])[good_stars]
        functions_to_overplot[i]['y'] = np.asarray(functions_to_overplot[i]['y'])[good_stars]

        # Normalise the y values
        functions_to_overplot[i]['y'] = np.asarray(normalise_a_curve(
            functions_to_overplot[i]['x'], functions_to_overplot[i]['y'], normalisation_constants[i + 1]))
        functions_to_overplot[i]['x'] = np.asarray(functions_to_overplot[i]['x'])

        # We'll only want to work on good stars (checking again just to be super safe lol)
        good_y_values = functions_to_overplot[i]['y'] > 0
        good_x_values = functions_to_overplot[i]['x'] > 0

        # Take logs only where log() is defined, otherwise replace with -np.inf
        functions_to_overplot[i]['y'] = np.where(
            good_y_values, np.log10(functions_to_overplot[i]['y'], where=good_y_values), -np.inf)

        functions_to_overplot[i]['x'] = np.where(
            good_x_values, np.log10(functions_to_overplot[i]['x'], where=good_x_values), -np.inf)

        i += 1

    # Cycle over axes, making my way downtown (to the-plot-is-done-land)
    x_limits = None  # Just here to shut the fucking LINTER UP
    for i_derivative, an_axis in enumerate(axes):
        # We keep hold of all the x and y data in a single array each round so that the limits can be correctly
        # calculated in a moment
        all_x_data = []
        all_y_data = []

        # We only need to take derivatives if we aren't on the final axis
        do_we_need_to_take_derivatives = i_derivative < n_axes - 1

        # Plot the field stars
        if i_derivative == 0 or show_numerical_derivatives:
            axes[i_derivative].plot(nearest_neighbor_distances, point_numbers, 'k-', label='Field stars')

            # Take derivatives while we're here, if necessary
            if do_we_need_to_take_derivatives and show_numerical_derivatives:
                point_numbers = np.gradient(point_numbers, nearest_neighbor_distances)

        # Plot the fitting functions
        i = 0
        while i < n_fitting_functions:
            axes[i_derivative].plot(functions_to_overplot[i]['x'], functions_to_overplot[i]['y'],
                                    functions_to_overplot[i]['style'], label=functions_to_overplot[i]['label'])

            # Only base the limits off of the stuff we're diffr'ing
            if functions_to_overplot[i]['differentiate']:
                all_x_data.append(functions_to_overplot[i]['x'].flatten())
                all_y_data.append(functions_to_overplot[i]['y'].flatten())

                # Take derivatives while we're here, if necessary (aka if this isn't the last axis) and remove any
                # gradient values that will have become inf
                if do_we_need_to_take_derivatives:
                    good_stars = np.logical_and(np.isfinite(functions_to_overplot[i]['x']),
                                                np.isfinite(functions_to_overplot[i]['y']))
                    functions_to_overplot[i]['x'] = (functions_to_overplot[i]['x'])[good_stars]
                    functions_to_overplot[i]['y'] = (functions_to_overplot[i]['y'])[good_stars]

                    functions_to_overplot[i]['y'] = np.gradient(functions_to_overplot[i]['y'],
                                                                functions_to_overplot[i]['x'])
            i += 1

        # Set y and x limits based on the points if they're in the plot (always the case in the first one,) or based on
        # the fitting functions if not
        if i_derivative == 0 or i == 0:
            x_limits = [np.min(nearest_neighbor_distances)-0.1, np.max(nearest_neighbor_distances)+0.1]
            y_limits = [np.min(point_numbers) - 1, np.max(point_numbers) + 1]
        elif infer_percentile_limits is False:
            y_limits = [np.min(point_numbers) - 1, np.max(point_numbers) + 1]
        else:
            all_x_data = np.concatenate(all_x_data)
            all_y_data = np.concatenate(all_y_data)
            x_limits_from_percentiles, y_limits = percentile_based_plot_limits(
                all_x_data, all_y_data, y_percentiles=y_percentile_limits[i_derivative - 1])

        axes[i_derivative].set_xlim(x_limits)
        axes[i_derivative].set_ylim(y_limits)

        # Axis beautification
        axes[i_derivative].minorticks_on()  # Makes it easier to read!
        axes[i_derivative].set_xlabel(f"log {neighbor_to_plot}th nearest neighbour distance")

        if i_derivative == 0:
            axes[i_derivative].set_ylabel(f"log point number")
        else:
            axes[i_derivative].set_ylabel(f"log point number\n{i_derivative}th derivative")

    axes[0].legend(fontsize=8, fancybox=True, edgecolor='black', loc='lower right')
    return axes
