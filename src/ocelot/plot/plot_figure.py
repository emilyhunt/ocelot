"""The highest level within the plot sub-directory: a set of standard functions for making funky plots."""

from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .axis import cluster
from .axis import nn_statistics


def location(data_gaia: pd.DataFrame,
             show_figure: bool = True,
             save_name: Optional[str] = None,
             figure_size: Union[list, np.ndarray] = (10, 10),
             figure_title: str = None,
             dpi: int = 100,
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
        dpi (integer): dpi for display and saving. 300 is nice for high quality figures or when you need to see detail.
            Default: 100

    Returns:
        The generated figure.

    """
    # Initialise the figure and add stuff to the axes
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figure_size, dpi=dpi)

    ax[0, 0], ax[0, 1] = cluster.position_and_pmotion(
        fig,
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
        ax[0, 0].text(0., 1.10, figure_title, transform=ax[0, 0].transAxes, va="bottom")

    # Output time
    if save_name is not None:
        fig.savefig(save_name, dpi=dpi)

    if show_figure is True:
        fig.show()

    return fig, ax


_default_position_kwargs = {
    'x_limits': 'all_stars',  # one of: ['like_y', 'all_stars', 'all_clusters', [min, max], float_for_std_deviation_about_mean]
    'y_limits': 'x',  # one of: ['like_x', 'all_stars', 'all_clusters', [min, max], float_for_std_deviation_about_mean]
    'x_label': 'ra',  # String
    'y_label': 'dec',  # String
    'title': 'position',  # String
    'alpha': 'infer',  # Either 'infer' or a float between 0 and 1
    'legend': False,  # Whether or not to have a legend
    'legend_kwargs': {},  # Should be acceptable args of plt.legend()
    'extra_objects_to_plot': {},  # Dicts of format (n_samples, 2) to plot, where the key will be set to the label
    'colour_map': None,  # Which colour cycle to use when plotting clusters. None uses the default mpl style.
    'key_x': 'ra',  # Key into data_gaia
    'key_y': 'dec',  # Key into data_gaia. Ignored if type=histogram
    'marker_radius': 1.,  # Float
    'background_type': 'scatter',  # Type of plot. Allows 'scatter', 'line', 'kde', 'histogram', 'histogram_2d', or None to ignore.
    'cluster_type': 'scatter',  # As above
}


def _get_figure_shape(format: Union[tuple, list, np.ndarray]):
    """Uses a format list to get and check the shape of a figure."""
    # Cast to a numpy array


    pass


def clustering_result_new(data_gaia: pd.DataFrame,
                      cluster_labels: Optional[np.ndarray] = None,
                      cluster_indices: Optional[Union[list, np.ndarray]] = None,
                      cluster_shading: Optional[np.ndarray] = None,

                      format: Union[tuple, list, np.ndarray] = (('position', 'pmotion',),
                                                                ('cmd', 'blank')),

                      axis_kwargs: Optional[Union[list, tuple, np.ndarray]] = None,

                      show_figure: bool = True,
                      save_name: Optional[str] = None,
                      figure_size: Union[list, np.ndarray, tuple] = (10, 10),
                      figure_title: str = None,
                      dpi: int = 100,
                      ):
    """A figure for evaluating the results of a clustering algorithm.

    todo: could spec something that calls add_subplot() instead of using the format kwarg, making this much more
        flexible as a format for other plotting functions too

    Parameters
    ----------
    data_gaia : pd.DataFrame, shape (n_field_stars, n_features)
        the Gaia data read in to a DataFrame. Keys should be unchanged from default Gaia source table names.
    cluster_labels : np.ndarray, optional, shape (n_field_stars,)
        the cluster membership labels for clustered stars. This should be the default sklearn.cluster output (i.e.
        -1 is used to label field stars).
    cluster_indices : np.ndarray, optional, shape (n_clusters_to_plot,)
        the clusters to plot in cluster_labels. For instance, you may not want to plot '-1'.
    cluster_shading : np.ndarray, optional, shape (n_field_stars,)
        an array of floats in the range 0, 1 for all clusters to use to shade their colour (aka alpha value). Useful for
        displaying e.g. HDBSCAN soft clustering.
        Default: None
    format : list-like of str
        a 2D array where each entry contains the type of plot to make on each axis. The number of axes is inferred from
        the shape of the array. Current allowed axis types:
            'position'
            'proper_motion'
            'cmd' (colour magnitude diagram)
            'parallax_density' (a histogram of parallax values)
            'blank' (plots nothing)
        Default: (('position', 'pmotion',), ('cmd', 'parallax_density')), i.e. a position, pmotion, cmd, parallax_density plot.
    axis_kwargs : list-like of dicts, optional
        any parameters to overwrite for each axis. If specified, it should have the same shape as format.
        Default: None

    show_figure : bool
        whether or not to show the figure at the end of plotting.
        Default: True
    save_name : str, optional
        if specified, the name to save the file as. The file extension present (or .png) will be used.
        Default: None
    figure_size : list-like, shape (2,)
        the size of the figure (x, y) in inches.
        Default: 10, 10
    figure_title : str, optional
        supertitle for the entire figure object.
        Default: None
    dpi : int
        dots per inch for the figure object and for saving.
        Default: None

    Raises
    ------

    Returns
    -------
    fig : the matplotlib figure element
    ax : the matplotlib list of axes, with the same shape as format

    """
    # Initialise the figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figure_size, dpi=dpi)



    pass


def clustering_result(data_gaia: pd.DataFrame,
                      cluster_labels: Optional[np.ndarray] = None,
                      cluster_indices: Optional[Union[list, np.ndarray]] = None,
                      cluster_shading: Optional[np.ndarray] = None,
                      show_figure: bool = True,
                      save_name: Optional[str] = None,
                      figure_size: Union[list, np.ndarray] = (10, 10),
                      figure_title: str = None,
                      dpi: int = 100,
                      open_cluster_pm_to_mark: Optional[Union[list, np.ndarray]] = None,
                      pmra_plot_limits: Optional[Union[list, np.ndarray]] = None,
                      pmdec_plot_limits: Optional[Union[list, np.ndarray]] = None,
                      cmd_plot_x_limits: Optional[Union[list, np.ndarray]] = None,
                      cmd_plot_y_limits: Optional[Union[list, np.ndarray]] = None,
                      cmd_plot_color_key: bool = None,
                      parallax_plot_x_limits: Optional[Union[list, np.ndarray]] = None,
                      parallax_plot_y_limits: Optional[Union[list, np.ndarray]] = None,
                      plot_std_limit: float = 1.5,
                      cmd_plot_std_limit: float = 3.0,
                      cluster_marker_radius: Union[tuple, list] = (1., 1., 1., 1.),
                      clip_to_fit_clusters: bool = True,
                      make_parallax_plot: bool = False):
    """A figure for evaluating the results of a clustering algorithm.

    Args:
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        cluster_labels (np.ndarray): the cluster membership labels for
            clustered stars. This should be the default sklearn.cluster output.
        cluster_indices (list-like, optional): the clusters to plot in
            cluster_labels. For instance, you may not want to plot '-1'
        cluster_shading (np.ndarray, optional): an array of floats in the range 0, 1 for all clusters to use to shade
            their colour (aka alpha value). Useful for displaying e.g. HDBSCAN soft clustering.
            Default: None
        open_cluster_pm_to_mark (list-like, optional): the co-ordinates
            (pmra, pmdec) of a point to mark on the proper motion diagram, such
            as a literature value.
        pmra_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the right ascension direction plot limits.
        pmdec_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the declination direction plot limits.
        cmd_plot_x_limits (list-like, optional): the minimum and maximum x (color) limits in the cmd plot.
        cmd_plot_y_limits (list-like, optional): the minimum and maximum y (apparent magnitude) limits in the cmd plot.
        cmd_plot_color_key (string, optional): optional other color in data_gaia to use instead of calculating G - RP.
        parallax_plot_x_limits (list-like, optional): as above for ra on the parallax plot.
        parallax_plot_y_limits (list-like, optional): as above for parallax on the parallax plot.
        plot_std_limit (float): standard deviation of proper motion to use to find plotting limits if none are
            explicitly specified.
            Default: 1.5
        cmd_plot_std_limit (float): standard deviation of proper motion to use to find plotting limits if none are
            explicitly specified, for the colour magnitude diagram plot. This is a separate parameter, as a higher
            value is often more suitable here.
            Default: 3.0
        cluster_marker_radius (list-like): radius of the cluster markers. Useful to increase when clusters are hard to
            see against background points. Specified as a length 4 tuple, giving the radius for the position, pm, CMD
            and parallax plots.
            Default: (1., 1., 1.)
        clip_to_fit_clusters (bool): whether or not to ensure that the proper motion plot always includes all clusters.
            Default: True
        make_parallax_plot (bool): whether to make a parallax plot on the fourth axes or whether to leave them blank.
            Default: False (for backwards-compatibility)

        --- Module standard ---
        show_figure (bool): whether or not to show the figure at the end of plotting. Default: True
        save_name (string, optional): whether or not to save the figure at the end of plotting.
            Default: None (no figure is saved)
        figure_size (list-like): the 2D size of the figure in inches, ala Matplotlib. Default: (10, 10).
        figure_title (string, optional): the desired title of the figure.
        dpi (integer): dpi for display and saving. 300 is nice for high quality figures or when you need to see detail.
            Default: 100

    Returns:
        The generated figure.

    """
    # Initialise the figure and add stuff to the axes
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figure_size, dpi=dpi)

    ax[0, 0], ax[0, 1] = cluster.position_and_pmotion(
        fig, ax[0, 0], ax[0, 1], data_gaia, cluster_labels, cluster_indices, cluster_shading,
        open_cluster_pm_to_mark=open_cluster_pm_to_mark,
        pmra_plot_limits=pmra_plot_limits,
        pmdec_plot_limits=pmdec_plot_limits,
        plot_std_limit=plot_std_limit,
        cluster_marker_radius=cluster_marker_radius[0:2],
        clip_to_fit_clusters=clip_to_fit_clusters)

    ax[1, 0] = cluster.color_magnitude_diagram(
        fig, ax[1, 0], data_gaia, cluster_labels, cluster_indices, cluster_shading,
        x_limits=cmd_plot_x_limits,
        y_limits=cmd_plot_y_limits,
        plot_std_limit=cmd_plot_std_limit,
        cluster_marker_radius=cluster_marker_radius[2],
        cmd_plot_color_key=cmd_plot_color_key
    )

    if make_parallax_plot:
        ax[1, 1] = cluster.ra_versus_parallax(
            fig, ax[1, 1], data_gaia, cluster_labels, cluster_indices, cluster_shading,
            x_limits=parallax_plot_x_limits,
            y_limits=parallax_plot_y_limits,
            plot_std_limit=plot_std_limit,
            cluster_marker_radius=cluster_marker_radius[3]
        )

    # Beautifying
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    # Plot a title - we use text instead of the title api so that it can be long and multi-line.
    if figure_title is not None:
        ax[0, 0].text(0., 1.10, figure_title, transform=ax[0, 0].transAxes, va="bottom")

    # Output time
    if save_name is not None:
        fig.savefig(save_name, dpi=dpi)

    if show_figure is True:
        fig.show()

    return fig, ax


def nearest_neighbor_distances(distances: np.ndarray,
                               number_of_derivatives: int = 0,
                               show_figure: bool = True,
                               save_name: Optional[str] = None,
                               figure_size: Union[list, np.ndarray] = (6, 6),
                               figure_title: str = None,
                               dpi: int = 100,
                               **kwargs):
    """A function for plotting statistics about the kth nearest neighbor distance. Wraps the function
    ocelot.plot.axis.nn_statistics.point_number_vs_nn_distance.

    Args:
        --- Function unique: for plotting on the axes ---
        distances (np.ndarray): the kth nearest neighbor distance array, of shape (n_samples, n_neighbors_calculated.))
        number_of_derivatives (int): the number of derivatives to plot.
            Default: 0 (just plots the nn distances themselves)
        **kwargs: additional keyword arguments to pass to ocelot.plot.axis.nn_statistics.point_number_vs_nn_distance

        --- Module standard ---
        show_figure (bool): whether or not to show the figure at the end of plotting.
            Default: True
        save_name (string, optional): whether or not to save the figure at the end of plotting.
            Default: None (no figure is saved)
        figure_size (list-like): the 2D size of the figure in inches, ala Matplotlib.
            Default: (6, 6).
        figure_title (string, optional): the desired title of the figure.
        dpi (integer): dpi for display and saving. 300 is nice for high quality figures or when you need to see detail.
            Default: 100

    Returns:
        a list of the matplotlib figure, the matplotlib axis

    """

    fig, ax = plt.subplots(nrows=number_of_derivatives + 1, ncols=1, figsize=figure_size, sharex='all', dpi=dpi)

    # Do the plotty thing
    ax = nn_statistics.point_number_vs_nn_distance(ax, distances, **kwargs)

    # Plot a title - we use text instead of the title api so that it can be long and multi-line.
    if figure_title is not None:
        if type(ax) is np.ndarray:
            ax[0].text(0., 1.10, figure_title, transform=ax[0].transAxes, va="bottom")
        else:
            ax.text(0., 1.10, figure_title, transform=ax.transAxes, va="bottom")

    # Output time
    if save_name is not None:
        fig.savefig(save_name, dpi=dpi)

    if show_figure is True:
        fig.show()

    return fig, ax
