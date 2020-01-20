"""Various possible axes for use in the analysis of nearest neighbor distances (e.g. when running DBSCAN.)"""

from typing import Optional, Union

import numpy as np

from ..utilities import normalise_a_curve


def point_number_vs_nn_distance(axes, distances: np.ndarray,
                                neighbor_to_plot: int = 1,
                                normalisation_constants: Union[list, tuple, np.ndarray] = (1.,),
                                functions_to_overplot: Optional[list] = None,
                                show_numerical_derivatives: bool = False):
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

    # Normalise and log the fitting functions too!
    i = 0
    while i < n_fitting_functions:
        functions_to_overplot[i]['y'] = np.log10(normalise_a_curve(
            functions_to_overplot[i]['x'], functions_to_overplot[i]['y'], normalisation_constants[i + 1]))
        functions_to_overplot[i]['x'] = np.log10(functions_to_overplot[i]['x'])
        i += 1

    # Cycle over axes, making my way downtown (to the-plot-is-done-land)
    for i_derivative, an_axis in enumerate(axes):

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
        fit_minimum = 0
        fit_maximum = 0
        while i < n_fitting_functions:
            axes[i_derivative].plot(functions_to_overplot[i]['x'], functions_to_overplot[i]['y'],
                                    functions_to_overplot[i]['style'], label=functions_to_overplot[i]['label'])

            # Only base the limits off of the stuff we're diffr'ing
            if functions_to_overplot[i]['differentiate']:
                fit_minimum = np.min([fit_minimum, functions_to_overplot[i]['y'].min()])
                fit_maximum = np.max([fit_maximum, functions_to_overplot[i]['y'].max()])

                # Take derivatives while we're here, if necessary
                if do_we_need_to_take_derivatives:
                    functions_to_overplot[i]['y'] = np.gradient(functions_to_overplot[i]['y'],
                                                                functions_to_overplot[i]['x'])
            i += 1

        print(fit_minimum, fit_maximum)

        # Set y limits based on the points for 0 derivative/no fits, or based on the fitting functions if not
        if i_derivative == 0 or i == 0:
            axes[i_derivative].set_ylim(np.min(point_numbers) - 1, np.max(point_numbers) + 1)
        else:
            axes[i_derivative].set_ylim((fit_minimum - 1, fit_maximum + 1))

        # Axis beautification
        axes[i_derivative].minorticks_on()  # Makes it easier to read!
        axes[i_derivative].set_xlabel(f"log {neighbor_to_plot}th nearest neighbour distance")

        if i_derivative == 0:
            axes[i_derivative].set_ylabel(f"log point number")
        else:
            axes[i_derivative].set_ylabel(f"log point number, {i_derivative}th derivative")

    axes[0].legend(fontsize=8, fancybox=True, edgecolor='black', loc='lower right')
    return axes


def nn_distance_histogram():
    """Produces a histogram of nearest neighbor distances. Comes appropriately pre-scaled for optimum viewing.

    todo
    """
    pass
