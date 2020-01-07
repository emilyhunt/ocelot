"""The highest level within the plot sub-directory: a set of standard functions for making funky plots."""

from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ocelot.isochrone import IsochroneInterpolator
from .axis import ax_isochrone
from .axis import cluster


def location(data_gaia: pd.DataFrame,
             show_figure: bool = True,
             save_name: Optional[str] = None,
             figure_size: Union[list, np.ndarray] = (10, 10),
             figure_title: str = None,
             open_cluster_pm_to_mark: Optional[Union[list, np.ndarray]] = None,
             pmra_plot_limits: Optional[Union[list, np.ndarray]] = None,
             pmdec_plot_limits: Optional[Union[list, np.ndarray]] = None,
             plot_std_limit: float = 1.5,
             kde_resolution: int = 50,
             plotting_resolution: int = 50,
             kde_bandwidth_radec: float = 0.08,
             kde_bandwidth_pmotion: float = 0.08):
    """A figure showing the location of an open cluster in position and proper motion space. Has more parameters than
    there are grains of sand on all the beaches on the planet.

    Args:
        --- Function unique ---
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        open_cluster_pm_to_mark (list-like, optional): the co-ordinates
            (pmra, pmdec) of a point to mark on the proper motion diagram, such
            as a literature value.
        pmra_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the right ascension direction plot limits.
        pmdec_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the declination direction plot limits.
        plot_std_limit (float): standard deviation of proper motion to use
            to find plotting limits if none are explicitly specified.
            Default: 1.5
        kde_resolution (int): number of points to sample the KDE at when scoring
            (across a resolution x resolution-sized grid.)
            Default: 50.
        kde_bandwidth_radec (float): the bandwidth of the position kde.
            Default: 0.08
        kde_bandwidth_pmotion (float): the bandwidth of the proper motion kde.
            Default: 0.08
        plotting_resolution (int): number of levels to use in contour plots.
            Default: 50

        --- Module standard ---
        show_figure (bool): whether or not to show the figure at the end of plotting. Default: True
        save_name (string, optional): whether or not to save the figure at the end of plotting.
            Default: None (no figure is saved)
        figure_size (list-like): the 2D size of the figure in inches, ala Matplotlib. Default: (10, 10).
        figure_title (string, optional): the desired title of the figure.

    Returns:
        The generated figure.

    """
    # Initialise the figure and add stuff to the axes
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figure_size)

    ax[0, 0], ax[0, 1] = cluster.position_and_pmotion(
        ax[0, 0],
        ax[0, 1],
        data_gaia,
        open_cluster_pm_to_mark=open_cluster_pm_to_mark,
        pmra_plot_limits=pmra_plot_limits,
        pmdec_plot_limits=pmdec_plot_limits,
        plot_std_limit=plot_std_limit)

    ax[1, 0], ax[1, 1] = cluster.density_position_and_pmotion(
        ax[1, 0],
        ax[1, 1],
        data_gaia,
        kde_bandwidth_radec=kde_bandwidth_radec,
        kde_bandwidth_pmotion=kde_bandwidth_pmotion,
        pmra_plot_limits=pmra_plot_limits,
        pmdec_plot_limits=pmdec_plot_limits,
        plot_std_limit=plot_std_limit,
        kde_resolution=kde_resolution,
        plotting_resolution=plotting_resolution)

    # Beautifying
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    # Plot a title - we use text instead of the title UI so that it can be long and multi-line.
    if figure_title is not None:
        ax[0, 0].text(0., 1.10, transform=ax[0, 0].transAxes, va="bottom")

    # Output time
    if save_name is not None:
        fig.save(save_name, dpi=300)

    if show_figure is True:
        fig.show()

    return fig


def clustering_result(data_gaia: pd.DataFrame,
                      cluster_labels: np.ndarray = None,
                      cluster_indices: Optional[Union[list, np.ndarray]] = None,
                      show_figure: bool = True,
                      save_name: Optional[str] = None,
                      figure_size: Union[list, np.ndarray] = (10, 10),
                      figure_title: str = None,
                      open_cluster_pm_to_mark: Optional[Union[list, np.ndarray]] = None,
                      pmra_plot_limits: Optional[Union[list, np.ndarray]] = None,
                      pmdec_plot_limits: Optional[Union[list, np.ndarray]] = None,
                      cmd_plot_x_limits: Optional[Union[list, np.ndarray]] = None,
                      cmd_plot_y_limits: Optional[Union[list, np.ndarray]] = None,
                      plot_std_limit: float = 1.5,
                      cmd_plot_std_limit: float = 3.0):
    """A figure for evaluating the results of a clustering algorithm.

    Args:
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        cluster_labels (np.ndarray): the cluster membership labels for
            clustered stars. This should be the default sklearn.cluster output.
        cluster_indices (list-like, optional): the clusters to plot in
            cluster_labels. For instance, you may not want to plot '-1'
        open_cluster_pm_to_mark (list-like, optional): the co-ordinates
            (pmra, pmdec) of a point to mark on the proper motion diagram, such
            as a literature value.
        pmra_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the right ascension direction plot limits.
        pmdec_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the declination direction plot limits.
        cmd_plot_x_limits (list-like, optional): the minimum and maximum x (blue minus red) limits in the cmd plot.
        cmd_plot_y_limits (list-like, optional): the minimum and maximum y (apparent magnitude) limits in the cmd plot.
        plot_std_limit (float): standard deviation of proper motion to use to find plotting limits if none are
            explicitly specified.
            Default: 1.5
        cmd_plot_std_limit (float): standard deviation of proper motion to use to find plotting limits if none are
            explicitly specified, for the colour magnitude diagram plot. This is a separate parameter, as a higher
            value is often more suitable here.
            Default: 3.0

        --- Module standard ---
        show_figure (bool): whether or not to show the figure at the end of plotting. Default: True
        save_name (string, optional): whether or not to save the figure at the end of plotting.
            Default: None (no figure is saved)
        figure_size (list-like): the 2D size of the figure in inches, ala Matplotlib. Default: (10, 10).
        figure_title (string, optional): the desired title of the figure.

    Returns:
        The generated figure.

    """
    # Initialise the figure and add stuff to the axes
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figure_size)

    ax[0, 0], ax[0, 1] = cluster.position_and_pmotion(
        ax[0, 0], ax[0, 1], data_gaia, open_cluster_pm_to_mark=open_cluster_pm_to_mark,
        pmra_plot_limits=pmra_plot_limits, pmdec_plot_limits=pmdec_plot_limits, plot_std_limit=plot_std_limit)

    ax[1, 0] = cluster.color_magnitude_diagram(
        ax[1, 0], data_gaia, cluster_labels, cluster_indices,
        x_limits=cmd_plot_x_limits, y_limits=cmd_plot_y_limits, plot_std_limit=cmd_plot_std_limit)

    # Beautifying
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    # Plot a title - we use text instead of the title api so that it can be long and multi-line.
    if figure_title is not None:
        ax[0, 0].text(0., 1.10, figure_title, transform=ax[0, 0].transAxes, va="bottom")

    # Output time
    if save_name is not None:
        fig.save(save_name, dpi=300)

    if show_figure is True:
        fig.show()

    return fig


def isochrone(data_isochrone: Optional[pd.DataFrame] = None,
              literature_isochrones_to_plot: Optional[np.ndarray] = None,
              isochrone_interpolator: Optional[IsochroneInterpolator] = None,
              interpolated_isochrones_to_plot: Optional[np.ndarray] = None,
              isochrone_arguments: Union[list, tuple] = ('MH', 'logAge'),
              isochrone_parameters: Union[list, tuple] = ('G_BP-RP', 'Gmag'),
              literature_colors: str = 'same',
              interpolated_colors: str = 'same',
              literature_line_style: str = '-',
              interpolated_line_style: str = '--',
              legend: bool = True,
              show_figure: bool = True,
              save_name: Optional[str] = None,
              figure_size: Union[list, np.ndarray] = (6, 6),
              figure_title: str = None,):
    """A figure showing a number of different isochrones - both interpolated and not. Can be configured to plot as
    many different isochrones as required, working with CMD 3.3-style isochrone tables and the
    ocelot.isochrone.IsochroneInterpolator class.

    Notes:
        - The function will plot anything that matches the numerical & parameter arguments within the table. If you want
            to, for instance, only plot one metallicity, then please make sure data_isochrone is already cleaned of any
            different metallicites.
        - At least one of either [data_isochrone, literature_isochrones_to_plot] *or* [isochrone_interpolator,
            interpolated_isochrones_to_plot] must be specified, else nothing will be plotted!

    Args:
        --- Function unique: data & arguments ---
        data_isochrone (pd.DataFrame, optional): data for the isochrones, in the CMD 3.3 table format. If specified,
            literature_isochrones_to_plot must also be specified.
            Default: None
        literature_isochrones_to_plot (np.ndarray, optional): an array of values to find in data_isochrones, of shape
            (n_points, n_arguments.)
            Default: None
        isochrone_interpolator (ocelot.isochrone.IsochroneInterpolator, optional): an isochrone interpolator to call
            values from. If specified, interpolated_isochrones_to_plot must also be specified.
            Default: None
        interpolated_isochrones_to_plot (np.ndarray, optional): an array of values to interpolate from
            isochrone_interpolator, of shape (n_points, n_arguments.)
            Default: None
        isochrone_arguments (list, tuple): names of arguments that correspond to the literature_isochrones_to_plot.
            Default: ('MH', 'logAge')
        isochrone_parameters (list, tuple): names of parameters that specify points for literature isochrones.
            Default: ('G_BP-RP', 'Gmag')

        --- Function unique: plotting style ---
        literature_colors (str): colors for the literature isochrones. Must specify either a valid matplotlib colourmap
            or 'same', which plots them all as black lines.
            Default: same
        interpolated_colors (str): colors for the interpolated isochrones. Must specify either a valid matplotlib
            colourmap or 'same', which plots them all as red lines.
            Default: same
        literature_line_style (str): style of the literature lines. Must specify a valid matplotlib line style, WITHOUT
            a color.
            Default: '-'
        interpolated_line_style (str): style of the interpolated lines. Must specify a valid matplotlib line style,
            WITHOUT a color.
            Default: '--'
        legend (bool): whether or not to plot a legend, whose labels are auto_generated based on input arguments and
            parameters.
            Default: True

        --- Module standard ---
        show_figure (bool): whether or not to show the figure at the end of plotting.
            Default: True
        save_name (string, optional): whether or not to save the figure at the end of plotting.
            Default: None (no figure is saved)
        figure_size (list-like): the 2D size of the figure in inches, ala Matplotlib.
            Default: (6, 6).
        figure_title (string, optional): the desired title of the figure.

    Returns:
        a Matplotlib figure instance.
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figure_size)

    # Plot literature isochrones, if requested
    if data_isochrone is not None:
        # Raise an error if everything we need hasn't been given
        if literature_isochrones_to_plot is None:
            raise ValueError("data_isochrone specified but no corresponding literature_isochrones_to_plot provided.")

        # Colormap setup
        n_isochrones = literature_isochrones_to_plot.shape[0]
        if literature_colors == 'same':
            # If the same colour has been requested for all lines, we set the literature isochrones to be black
            colors = [(0.0, 0.0, 0.0, 1.0)] * n_isochrones
        else:
            colormap = plt.get_cmap(literature_colors)
            colors = [colormap(1. * i / n_isochrones) for i in range(n_isochrones)]

        # Cycle over the requested isochrones and plot them!
        for an_isochrone, a_color in zip(literature_isochrones_to_plot, colors):
            ax = ax_isochrone.literature_isochrone(ax, data_isochrone, an_isochrone,
                                                   isochrone_arguments, isochrone_parameters,
                                                   a_color, literature_line_style)

    # Raise an error if nothing was specified for plotting at all
    elif isochrone_interpolator is None:
        raise ValueError('Neither data_isochrone or isochrone_interpolator were specified. I have nothing to plot!')

    # Plot interpolated isochrones, if requested
    if isochrone_interpolator is not None:
        # Raise an error if everything we need hasn't been given
        if interpolated_isochrones_to_plot is None:
            raise ValueError("data_isochrone specified but no corresponding interpolated_isochrones_to_plot provided.")

        # Colormap setup
        n_isochrones = interpolated_isochrones_to_plot.shape[0]
        if interpolated_colors == 'same':
            # If the same colour has been requested for all lines, we set the literature isochrones to be black
            colors = [(1.0, 0.0, 0.0, 1.0)] * n_isochrones
        else:
            colormap = plt.get_cmap(interpolated_colors)
            colors = [colormap(1. * i / n_isochrones) for i in range(n_isochrones)]

        # Cycle over the requested isochrones and plot them!
        for an_isochrone, a_color in zip(interpolated_isochrones_to_plot, colors):
            ax = ax_isochrone.interpolated_isochrone(ax, isochrone_interpolator, an_isochrone,
                                                     isochrone_arguments, a_color, interpolated_line_style)

    # Make sure the axis is the right way up...
    ax.invert_yaxis()

    # Beautifying
    ax.set_xlabel(r'm_{bp} - m_{gp}')
    ax.set_ylabel(r'm_{G}')

    if legend is True:
        ax.legend(fontsize=8)

    # Plot a title - we use text instead of the title api so that it can be long and multi-line.
    if figure_title is not None:
        ax[0, 0].text(0., 1.10, figure_title, transform=ax[0, 0].transAxes, va="bottom")

    # Output time
    if save_name is not None:
        fig.save(save_name, dpi=300)

    if show_figure is True:
        fig.show()

    return fig
