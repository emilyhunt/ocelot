"""The highest level within the plot sub-directory: a set of standard functions for making funky plots."""

from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plot_on_axis


def location(data_gaia: pd.DataFrame,
             show_figure: bool = True,
             save_name: Optional[str] = None,
             figure_size: Union[list, np.ndarray] = (10, 10),
             figure_title: str = None,
             open_cluster_pm_to_mark: Optional[list, np.ndarray] = None,
             pmra_plot_limits: Optional[list, np.ndarray] = None,
             pmdec_plot_limits: Optional[list, np.ndarray] = None,
             plot_std_limit: float = 1.5,
             kde_resolution: int = 50,
             plotting_resolution: int = 50,
             kde_bandwidth_radec: float = 0.08,
             kde_bandwidth_pmotion: float = 0.08):
    """A figure showing the location of an open cluster in position and proper motion space. Has more parameters than
    there are grains of sand on all the beaches on the planet.

    Args:
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        show_figure (bool): whether or not to show the figure at the end of plotting. Default: True
        save_name (string, optional): whether or not to save the figure at the end of plotting.
            Default: None (no figure is saved)
        figure_size (list-like): the 2D size of the figure in inches, ala Matplotlib. Default: (10, 10).
        figure_title (string, optional): the desired title of the figure.
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
        plotting_resolution (int): number of levels to use in contour plots.
            Default: 50
        kde_bandwidth_radec (float): the bandwidth of the position kde.
            Default: 0.08
        kde_bandwidth_pmotion (float): the bandwidth of the proper motion kde.
            Default: 0.08

    Returns:
        The generated figure.

    """
    # Initialise the figure and add stuff to the axes
    fig, ax = plt.subplots(nrows=2, ncols=2, figure_size=figure_size)

    ax[0, 0], ax[0, 1] = plot_on_axis.position_and_pmotion(
        ax[0, 0],
        ax[0, 1],
        data_gaia,
        open_cluster_pm_to_mark=open_cluster_pm_to_mark,
        pmra_plot_limits=pmra_plot_limits,
        pmdec_plot_limits=pmdec_plot_limits,
        plot_std_limit=plot_std_limit)

    ax[1, 0], ax[1, 1] = plot_on_axis.density_position_and_pmotion(
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
                      cluster_indices: Optional[list, np.ndarray] = None,
                      show_figure: bool = True,
                      save_name: Optional[str] = None,
                      figure_size: Union[list, np.ndarray] = (10, 10),
                      figure_title: str = None,
                      open_cluster_pm_to_mark: Optional[list, np.ndarray] = None,
                      pmra_plot_limits: Optional[list, np.ndarray] = None,
                      pmdec_plot_limits: Optional[list, np.ndarray] = None,
                      cmd_plot_x_limits: Optional[list, np.ndarray] = None,
                      cmd_plot_y_limits: Optional[list, np.ndarray] = None,
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
        show_figure (bool): whether or not to show the figure at the end of plotting.
            Default: True
        save_name (string, optional): whether or not to save the figure at the end of plotting.
            Default: None (no figure is saved)
        figure_size (list-like): the 2D size of the figure in inches, ala Matplotlib.
            Default: (10, 10).
        figure_title (string, optional): the desired title of the figure.
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

    Returns:
        The generated figure.

    """
    # Initialise the figure and add stuff to the axes
    fig, ax = plt.subplots(nrows=2, ncols=2, figure_size=figure_size)

    ax[0, 0], ax[0, 1] = plot_on_axis.position_and_pmotion(
        ax[0, 0], ax[0, 1], data_gaia, open_cluster_pm_to_mark=open_cluster_pm_to_mark,
        pmra_plot_limits=pmra_plot_limits, pmdec_plot_limits=pmdec_plot_limits, plot_std_limit=plot_std_limit)

    ax[1, 0] = plot_on_axis.color_magnitude_diagram(
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


def isochrone():

    pass
