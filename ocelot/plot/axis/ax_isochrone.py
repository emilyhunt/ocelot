"""A set of functions for adding standardised things to an axis, specifically for isochrone plotting."""

from typing import Union

import numpy as np
import pandas as pd

from ocelot.isochrone import IsochroneInterpolator


def generate_line_label(front_matter: str, isochrone_arguments: Union[list, tuple], values: np.ndarray):
    """Generates a line label for use with isochrone plotting."""
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
    """Plots a literature isochrone onto an axis, given the axis and a CMD 3.3-style table."""
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
                           resolution: int = 500):
    """Plots an interpolated isochrone onto an axis, given an ocelot.isochrone.IsochroneInterpolator instance."""
    # Grab all the points to plot
    output_x, output_y = isochrone_interpolator(np.asarray([interpolated_isochrone_to_plot]), resolution=resolution)

    # Plot. It. ALL.
    axis.plot(output_x, output_y, line_style, color=color,
              label=generate_line_label("int:", isochrone_arguments, interpolated_isochrone_to_plot))

    return axis
