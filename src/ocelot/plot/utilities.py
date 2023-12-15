"""Some little friends to help with various odd jobs in OCELOT's plotting submod."""

from typing import Union, Optional
from scipy.stats import iqr

import numpy as np


def calculate_alpha(fig, ax, n_points: int, marker_size: float, desired_max_density: float = 0.1,
                    scale_with_dpi: bool = False,
                    max_alpha: float = 0.3,
                    min_alpha: float = 0.001,
                    scatter_plot: bool = False):
    """Calculates the optimum alpha value for points to make the plot not be
    over-saturated. Relies on the area value being correctly calculated
    prior to input, which should be the value in inches for the figure. It also
    assumes that all points are uniformly distributed across the figure area
    (which they often are not), so the input area number may need tweaking.

    Args:
        fig (matplotlib figure): the figure element. Required to calculate alpha values precisely.
        ax (matplotlib axis): the figure element. Required to calculate alpha values precisely.
        n_points (int): the number of points to plot.
        marker_size (px): the radius of the marker. Note that plt.scatter()
            specifies marker *area* but plt.plot() uses marker size, which is
            analogous to the marker radius. In the former case, ensure scatter_plot is True.
        scale_with_dpi (bool): whether or not to scale alpha values with the dpi of the figure.
            Default: False (correct for scatter plots)
        desired_max_density (float): tweakable parameter that was found to
            produce the best results. Simply mutliplying the area value is
            probably easier for normal function use, though.
            Default: 0.1.
        max_alpha (float): max alpha value to allow.
            Default: 0.3
        min_alpha (float): min alpha value to allow.
            Default: 0.001
        scatter_plot (str): whether to calculate alpha values for a scatter plot, which specifies the marker_size as an
            area, *not* a radius!

    Returns:
        float: A recommended alpha value to use on these axes.

    Todo: could still consider if the data are clumped or not

    """
    # We calculate the area of the figure with some fancy magic
    # See https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    area = bbox.width * bbox.height

    # Scatter plots are awkward (see above function docs)
    if scatter_plot:
        marker_area = n_points * marker_size  # In units of px
    else:
        marker_area = n_points * np.pi * marker_size ** 2  # In units of px

    # We aim to have a maximum density of uniform_max_density, assuming that
    # n_points are distributed evenly across the area.
    if scale_with_dpi:
        total_area = area * fig.dpi ** 2
    else:
        total_area = area * 100**2  # If not, we just use the default area
    current_max_density = marker_area / total_area

    # We only do the alpha calculation if current max density is non-zero, just to avoid a lil error
    if current_max_density != 0:
        return np.clip(desired_max_density / current_max_density, min_alpha, max_alpha)
    else:
        return max_alpha


def normalise_a_curve(x_values: np.ndarray, y_values: np.ndarray, normalisation_constant: Union[int, float]):
    """Wrapper for np.trapz intended for use with ocelot.plot.axis.nn_statistics.point_number_vs_nn_distance.

    Args:
        x_values (np.ndarray): x values of the curve.
        y_values (np.ndarray): y values of the curve.
        normalisation_constant (float): area to set the curve to have. If zero, the curve will not be normalised.

    Returns:
        normalised y_values (or unmodified y_values if normalisation_constant is 0)

    """
    if normalisation_constant == 0:
        return y_values
    else:
        return y_values * normalisation_constant / np.trapz(y_values, x=x_values)


def _good_points_plot_limits(target_points: np.ndarray, constraint_points: np.ndarray, constraint_limits: np.ndarray):
    """Finds appropriate plot limits for a target array of points based on the limits of a constraint set of points.
    Solves the issue with matplotlib limits being set on one dimension but not cropping out blank space on the other.

    Args:
        target_points (np.ndarray): points to find limits for
        constraint_points (np.ndarray): the points that already have a cut-off constraint
        constraint_limits (np.ndarray): the limits for the pre-constrained points.

    Returns:
        target_point_limits (np.ndarray): limits for the target points that will remove blank space.

    """
    good_points = np.logical_and(constraint_points >= constraint_limits[0], constraint_points <= constraint_limits[1])
    return np.asarray([target_points[good_points].min(), target_points[good_points].max()])


def percentile_based_plot_limits(x: np.ndarray,
                                 y: np.ndarray,
                                 x_percentiles: Optional[Union[tuple, list, np.ndarray]] = None,
                                 y_percentiles: Optional[Union[tuple, list, np.ndarray]] = None,
                                 range_padding: Optional[float] = 0.05):
    """Derives appropriate limits for a 2D plot based on percentiles of the data. May use just one or both of x or y
    data to infer plot limits. When only one is used, the unused axis will be cropped to remove blank space (which
    solves the issue with matplotlib limits being set on one dimension but not cropping out blank space on the other.)

    Args:
        x (np.ndarray): x points to base plotting limits from.
        y (np.ndarray): y points to base plotting limits from.
        x_percentiles (list-like): length two array of x percentiles to plot. Values within must be in the range
            [0, 100]. If None, x plot limits are inferred from the x range of valid y points.
        y_percentiles (np.ndarray, list): length two array of y percentiles to plot. Values within must be in the range
            [0, 100]. If None, y plot limits are inferred from the y range of valid x points.
        range_padding (list-like): padding factor to apply based on the valid range of the data. Stops plots from
            being cropped exactly on the points at hand. Remember padding is added individually around all ends of the
            data! May be set to "None" to be turned off.
            Default: 0.05

    """
    # Make sure the inputs are flat numpy arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # First, we make some actual limits
    if x_percentiles is not None:
        x_plot_limits = np.percentile(x, x_percentiles)

        # When x and y specified
        if y_percentiles is not None:
            y_plot_limits = np.percentile(y, y_percentiles)

        # When only x specified
        else:
            y_plot_limits = _good_points_plot_limits(y, x, x_plot_limits)

    # When only y specified
    elif y_percentiles is not None:
        y_plot_limits = np.percentile(y, y_percentiles)
        x_plot_limits = _good_points_plot_limits(x, y, y_plot_limits)

    # When neither specified: value error!
    else:
        raise ValueError("One or both of x_percentiles and y_percentiles must be specified.")

    # We can also add some padding if requested
    if range_padding is not None:
        # Grab the stars within the ranges
        good_x = np.logical_and(x >= x_plot_limits[0], x <= x_plot_limits[1])
        good_y = np.logical_and(y >= y_plot_limits[0], y <= y_plot_limits[1])

        # Find the statistical range/peak to peak (ptp) value of the good stars
        x_padding = np.ptp(x[good_x]) * range_padding
        y_padding = np.ptp(y[good_y]) * range_padding

        # Apply the padding to the limits
        x_plot_limits += np.asarray([-x_padding, x_padding])
        y_plot_limits += np.asarray([-y_padding, y_padding])

    return x_plot_limits, y_plot_limits
