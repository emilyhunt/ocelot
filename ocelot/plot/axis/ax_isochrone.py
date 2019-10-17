"""A set of functions for adding standardised things to an axis, specifically for isochrone plotting."""

from typing import Union

import numpy as np
import pandas as pd

from ocelot.isochrone import IsochroneInterpolator


def generate_line_label(front_matter: str, isochrone_arguments: Union[list, tuple], values: np.ndarray) -> str:
    """Generates a line label for use with isochrone plotting.

    Args:
        front_matter (str): the start of the label (e.g.: 'iso')
        isochrone_arguments (list, tuple): names of arguments that correspond to the literature_isochrones_to_plot.


    """
    # Loop over arguments and values, adding them to the string:
    for an_argument, a_value in zip(isochrone_arguments, values):
        front_matter += f" {an_argument}={a_value:.2f}"

    return front_matter


def literature_isochrone(axis,
                         data_isochrone: pd.DataFrame,
                         literature_isochrone_to_plot: np.ndarray,
                         isochrone_arguments: Union[list, tuple],
                         isochrone_parameters: Union[list, tuple],
                         color: tuple,
                         line_style: str):
    """Plots a literature isochrone onto an axis, given the axis and a CMD 3.3-style table. Performs no beautification,
    as this function is designed to be ran many times.

    Args:
        axis (matplotlib axis): the axis to plot the isochrone onto.
        data_isochrone (pd.DataFrame, optional): data for the isochrones, in the CMD 3.3 table format.
        literature_isochrone_to_plot (np.ndarray): a 1D array of values to find in data_isochrones, of shape
            (n_arguments).
        isochrone_arguments (list, tuple): names of arguments that correspond to the literature_isochrone_to_plot.
        isochrone_parameters (list, tuple): names of parameters that specify points for the literature isochrone.
        color (tuple): a tuple of length 4 of RGBA values, ala matplotlib. e.g.: (1., 0., 0., 1.) is red.
        line_style (str): style of the line. Must specify a valid matplotlib line style, WITHOUT a color. e.g.: '-'

    Returns:
        an edited matplotlib axis.

    """
    # Sum the number of parameter matches per row
    stars_to_use = np.sum(data_isochrone[isochrone_arguments] == literature_isochrone_to_plot, axis=1)

    # Only accept stars where all the parameters_are matched (the row-wise sum should equal n_parameters)
    stars_to_use = np.asarray(stars_to_use == len(isochrone_arguments))

    # Plot it all!
    axis.plot(data_isochrone.loc[stars_to_use, isochrone_parameters[0]],
              data_isochrone.loc[stars_to_use, isochrone_parameters[1]],
              line_style,
              color=color,
              label=generate_line_label("lit:", isochrone_arguments, literature_isochrone_to_plot))

    return axis


def interpolated_isochrone(axis,
                           isochrone_interpolator: IsochroneInterpolator,
                           interpolated_isochrone_to_plot: np.ndarray,
                           isochrone_arguments: Union[list, tuple],
                           color: tuple,
                           line_style: str,
                           resolution: int = 300):
    """Plots an interpolated isochrone onto an axis, given an ocelot.isochrone.IsochroneInterpolator instance. Performs
    no beautification, as this function is designed to be ran many times.

    Args:
        axis (matplotlib axis): the axis to plot the isochrone onto.
        isochrone_interpolator (ocelot.isochrone.IsochroneInterpolator): an isochrone interpolator to call values from.
        interpolated_isochrone_to_plot (np.ndarray): a 1D array of values to interpolate with isochrone_interpolator, of
            shape (n_arguments).
        isochrone_arguments (list, tuple): names of arguments that correspond to the interpolated_isochrone_to_plot.
        color (tuple): a tuple of length 4 of RGBA values, ala matplotlib. e.g.: (1., 0., 0., 1.) is red.
        line_style (str): style of the line. Must specify a valid matplotlib line style, WITHOUT a color. e.g.: '-'
        resolution (int): number of points to plot per line.
            Default: 300

    Returns:
        an edited matplotlib axis.

    """
    # Grab all the points to plot
    output_x, output_y = isochrone_interpolator(np.asarray([interpolated_isochrone_to_plot]), resolution=resolution)

    # Plot. It. ALL.
    axis.plot(output_x, output_y, line_style, color=color,
              label=generate_line_label("int:", isochrone_arguments, interpolated_isochrone_to_plot))

    return axis
