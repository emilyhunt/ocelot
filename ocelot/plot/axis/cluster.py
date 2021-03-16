"""A set of functions for adding standardised things to an axis, specifically for star cluster plotting."""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import process
from .. import utilities


def position_and_pmotion(
        fig,
        axis_1,
        axis_2,
        data_gaia: pd.DataFrame,
        cluster_labels: np.ndarray = None,
        cluster_indices: Optional[Union[list, np.ndarray]] = None,
        cluster_shading: Optional[np.ndarray] = None,
        open_cluster_pm_to_mark: Optional[Union[list, np.ndarray]] = None,
        pmra_plot_limits: Optional[Union[list, np.ndarray]] = None,
        pmdec_plot_limits: Optional[Union[list, np.ndarray]] = None,
        plot_std_limit: float = 1.5,
        cluster_marker_radius: Union[list, tuple] = (1., 1.),
        clip_to_fit_clusters: bool = True):
    """Makes a scatter plot of position and proper motion for a given cluster,
    using plot_helper_calculate_alpha to prevent over-saturation of the
    figures.

    Args:
        fig (matplotlib figure): the figure element. Required to calculate alpha values precisely.
        axis_1 (matplotlib axis): the positional (ra/dec) axis.
        axis_2 (matplotlib axis): the proper motion (pmra/pmdec) axis.
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        cluster_labels (np.ndarray): the cluster membership labels for
            clustered stars. This should be the default sklearn.cluster output.
        cluster_indices (list-like, optional): the clusters to plot in
            cluster_labels. For instance, you may not want to plot '-1'
            clusters (which are noise) produced by algorithm like DBSCAN.
        cluster_shading (np.ndarray, optional): an array of floats in the range 0, 1 for all clusters to use to shade
            their colour (aka alpha value). Useful for displaying e.g. HDBSCAN soft clustering.
            Default: None
        open_cluster_pm_to_mark (list-like, optional): the co-ordinates (pmra, pmdec) of a point to mark on the proper
            motion diagram, such as a literature value.
        pmra_plot_limits (list-like, optional): the minimum and maximum proper motion in the right ascension direction
            plot limits.
        pmdec_plot_limits (list-like, optional): the minimum and maximum proper motion in the declination direction
            plot limits.
        plot_std_limit (float): standard deviation of proper motion to use to find plotting limits if none are
            explicitly specified. Default: 1.5
        cluster_marker_radius (float): radius of the cluster markers. Useful to increase when clusters are hard to
            see against background points. Specified as a length 2 tuple, giving the radius for the position and pm
            plots.
            Default: (1., 1.)

    Returns:
        axis_1 (matplotlib axis): the modified position axis.
        axis_2 (matplotlib axis): the modified proper motion axis.
    """
    # Calculate our own plotting limits if none have been specified
    if pmra_plot_limits is None:
        pmra_plot_limits = (np.mean(data_gaia.pmra)
                            + np.std(data_gaia.pmra)
                            * np.array([-plot_std_limit, plot_std_limit]))

    if pmdec_plot_limits is None:
        pmdec_plot_limits = (np.mean(data_gaia.pmdec)
                             + np.std(data_gaia.pmdec)
                             * np.array([-plot_std_limit, plot_std_limit]))

    # We also clip the plot so that it will always include all found clusters
    if cluster_labels is not None and clip_to_fit_clusters:
        clustered_stars = np.isin(cluster_labels, cluster_indices)
        pmra_plot_limits = np.asarray(
            [np.clip(pmra_plot_limits[0], -np.inf, data_gaia.loc[clustered_stars, 'pmra'].min() - 1),
             np.clip(pmra_plot_limits[1], data_gaia.loc[clustered_stars, 'pmra'].max() + 1, np.inf)])
        pmdec_plot_limits = np.asarray(
            [np.clip(pmdec_plot_limits[0], -np.inf, data_gaia.loc[clustered_stars, 'pmdec'].min() - 1),
             np.clip(pmdec_plot_limits[1], data_gaia.loc[clustered_stars, 'pmdec'].max() + 1, np.inf)])

    if cluster_shading is None:
        cluster_shading = np.ones(data_gaia.shape[0])

    # Better notation
    pmra_range = pmra_plot_limits
    pmdec_range = pmdec_plot_limits

    # Work out how many points will fall in the pm range
    good_pmra = np.logical_and(data_gaia['pmra'] > pmra_range[0], data_gaia['pmra'] < pmra_range[1])
    good_pmdec = np.logical_and(data_gaia['pmdec'] > pmdec_range[0], data_gaia['pmdec'] < pmdec_range[1])
    pm_plot_points = np.count_nonzero(np.logical_and(good_pmra, good_pmdec))

    # Calculate a rough guess at a good alpha value
    alpha_estimate_1 = utilities.calculate_alpha(fig, axis_1, data_gaia.shape[0], 1)
    alpha_estimate_2 = utilities.calculate_alpha(fig, axis_1, pm_plot_points, 1)

    # Ra/dec plot
    axis_1.plot(data_gaia['ra'], data_gaia['dec'], '.', ms=1, alpha=alpha_estimate_1, c='k')
    axis_1.set_xlabel('ra')
    axis_1.set_ylabel('dec')
    axis_1.set_title('position')

    # Proper motion plot
    axis_2.plot(data_gaia['pmra'], data_gaia['pmdec'], '.', ms=1, alpha=alpha_estimate_2, c='k')
    axis_2.set_xlabel('pmra')
    axis_2.set_ylabel('pmdec')
    axis_2.set_xlim(pmra_range)
    axis_2.set_ylim(pmdec_range)
    axis_2.set_title('proper motion')

    # Plot clusters if labels have been specified
    if cluster_labels is not None:

        cmap = plt.get_cmap("tab10")  # Get the default matplotlib colourmap

        # We cycle over the clusters, grabbing their indices and plotting
        # them in a new colour erri time =)
        for i_color, a_cluster in enumerate(cluster_indices):
            stars_in_this_cluster = cluster_labels == a_cluster

            # Make colors and give them alpha values
            colors = np.tile(cmap(i_color % 10), (np.count_nonzero(stars_in_this_cluster), 1))
            colors[:, 3] = cluster_shading[stars_in_this_cluster]

            axis_1.scatter(data_gaia.loc[stars_in_this_cluster, 'ra'],
                           data_gaia.loc[stars_in_this_cluster, 'dec'],
                           marker='.',
                           s=cluster_marker_radius[0]**2,
                           c=colors,
                           zorder=100,
                           label=a_cluster)

            axis_2.scatter(data_gaia.loc[stars_in_this_cluster, 'pmra'],
                           data_gaia.loc[stars_in_this_cluster, 'pmdec'],
                           marker='.',
                           s=cluster_marker_radius[1]**2,
                           c=colors,
                           zorder=100,
                           label=a_cluster)

    # Add a red cross at a defined location on the pmra/pmdec plot if desired
    if open_cluster_pm_to_mark is not None:
        axis_2[0, 1].plot(open_cluster_pm_to_mark[0], open_cluster_pm_to_mark[1],
                          'rx', ms=6)

    return [axis_1, axis_2]


def density_position_and_pmotion(axis_1,
                                 axis_2,
                                 data_gaia,
                                 kde_bandwidth_radec,
                                 kde_bandwidth_pmotion,
                                 pmra_plot_limits=None,
                                 pmdec_plot_limits=None,
                                 plot_std_limit=1.5,
                                 kde_resolution: int = 50,
                                 plotting_resolution: int = 50):
    """Makes KDE plots of position density and proper motion density for a
    given cluster.

    Args:
        axis_1 (matplotlib axis): the positional (ra/dec) axis.
        axis_2 (matplotlib axis): the proper motion (pmra/pmdec) axis.
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        kde_bandwidth_radec (float): the bandwidth of the kde for position.
        kde_bandwidth_pmotion (float): the bandwidth of the proper motion
            kde.
        pmra_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the right ascension direction plot limits.
        pmdec_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the declination direction plot limits.
        plot_std_limit (float): standard deviation of proper motion to use
            to find plotting limits if none are explicitly specified.
            Default: 1.5
        kde_resolution (int): number of points to sample the KDE at when scoring
            (across a resolution x resolution-sized grid.) Default: 50.
        plotting_resolution (int): number of levels to use in contour plots.
            Default: 50

    Returns:
        axis_1 (matplotlib axis): the modified position axis.
        axis_2 (matplotlib axis): the modified proper motion axis.

    """
    # Calculate our own plotting limits if none have been specified
    if pmra_plot_limits is None:
        pmra_plot_limits = (np.mean(data_gaia['pmra'])
                            + np.std(data_gaia['pmra'])
                            * np.array([-plot_std_limit, plot_std_limit]))

    if pmdec_plot_limits is None:
        pmdec_plot_limits = (np.mean(data_gaia['pmdec'])
                             + np.std(data_gaia['pmdec'])
                             * np.array([-plot_std_limit, plot_std_limit]))

    # Better notation
    pmra_range = pmra_plot_limits
    pmdec_range = pmdec_plot_limits

    # Process the datasets - we may want to cut down the pmra/pmdec one
    data_position = data_gaia[['ra', 'dec']].to_numpy()

    data_proper_motion = data_gaia[['pmra', 'pmdec']]

    # Drop any bad proper motion points
    good_pmra = np.logical_and(data_gaia['pmra'] > pmra_range[0], data_gaia['pmra'] < pmra_range[1])
    good_pmdec = np.logical_and(data_gaia['pmdec'] > pmdec_range[0], data_gaia['pmdec'] < pmdec_range[1])
    good_stars = np.logical_and(good_pmra, good_pmdec)
    data_proper_motion = data_proper_motion.loc[good_stars, :].to_numpy()

    # Perform KDE fits to the data
    mesh_ra, mesh_dec, density_ra_dec = process.kde_fit_2d(data_position, kde_bandwidth_radec,
                                                           kde_resolution=kde_resolution)

    mesh_pmra, mesh_pmdec, density_pmotion = process.kde_fit_2d(data_proper_motion, kde_bandwidth_pmotion,
                                                                x_limits=pmra_range, y_limits=pmdec_range,
                                                                kde_resolution=kde_resolution)

    # Plot the results
    # Ra/dec
    axis_1.contourf(mesh_ra, mesh_dec,
                    density_ra_dec,
                    levels=plotting_resolution,
                    cmap=plt.get_cmap('Reds'))
    axis_1.set_xlabel('ra')
    axis_1.set_ylabel('dec')
    axis_1.text(0.02, 1.02, f'bw={kde_bandwidth_radec:.4f}',
                ha='left', va='bottom', transform=axis_1.transAxes,
                fontsize='small',
                bbox=dict(boxstyle='round', facecolor='w', alpha=0.8))

    # Pmra/pmdec
    axis_2.contourf(mesh_pmra, mesh_pmdec,
                    density_pmotion,
                    levels=plotting_resolution,
                    cmap=plt.get_cmap('Blues'))
    axis_2.set_xlabel('pmra')
    axis_2.set_ylabel('pmdec')
    axis_2.text(0.02, 1.02, f'bw={kde_bandwidth_pmotion:.4f}',
                ha='left', va='bottom', transform=axis_2.transAxes,
                fontsize='small',
                bbox=dict(boxstyle='round', facecolor='w', alpha=0.8))

    # Return stuff. Because I'm nice like that =)
    return [axis_1, axis_2]


def color_magnitude_diagram(fig,
                            axis,
                            data_gaia: pd.DataFrame,
                            cluster_labels: np.ndarray = None,
                            cluster_indices: Optional[Union[list, np.ndarray]] = None,
                            cluster_shading: Optional[np.ndarray] = None,
                            plot_std_limit: float = 3.0,
                            x_limits: Optional[Union[list, np.ndarray]] = None,
                            y_limits: Optional[Union[list, np.ndarray]] = None,
                            cluster_marker_radius: float = 1.0,
                            cmd_plot_color_key: bool = None):
    """Makes a colour magnitude diagram plot of a given star field, and may
    also overplot star clusters within that field.

    Notes:
        You can specify as few as one cluster_indices if cluster overplotting
        is too crowded.

    Args:
        fig (matplotlib figure): the figure element. Required to calculate alpha values precisely.
        axis (matplotlib axis): the colour magnitude diagram axis.
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        cluster_labels (np.ndarray): the cluster membership labels for
            clustered stars. This should be the default sklearn.cluster output.
        cluster_indices (list-like, optional): the clusters to plot in
            cluster_labels. For instance, you may not want to plot '-1'
            clusters (which are noise) produced by algorithm like DBSCAN.
        cluster_shading (np.ndarray, optional): an array of floats in the range 0, 1 for all clusters to use to shade
            their colour (aka alpha value). Useful for displaying e.g. HDBSCAN soft clustering.
            Default: None
        plot_std_limit (float): standard deviation of parameters to use
            to find plotting limits if none are explicitly specified.
            Default: 1.5
        x_limits (list-like, optional): the minimum and maximum blue minus red
            plotting limits.
        y_limits (list-like, optional): the minimum and maximum absolute
            magnitude plotting limits.
        cmd_plot_color_key (string, optional): optional other color in data_gaia to use instead of calculating G - RP.
        cluster_marker_radius (float): radius of the cluster marker. Useful to increase when clusters are hard to see
            against background points.
            Default: 1.0

    Returns:
        axis (matplotlib axis): the modified colour magnitude diagram axis.

    """

    # Calculate the stuff we need to plot
    # M = m - 5 * log10(d [pc]) + 5
    apparent_magnitude = data_gaia["phot_g_mean_mag"]

    if cmd_plot_color_key is None:
        color = (data_gaia["phot_bp_mean_mag"]
                 - data_gaia["phot_rp_mean_mag"])
    else:
        color = data_gaia[cmd_plot_color_key]

    # Calculate our own plotting limits if none have been specified
    if x_limits is None:
        x_limits = (np.mean(color)
                    + np.std(color)
                    * np.array([-plot_std_limit, plot_std_limit]))

    if y_limits is None:
        y_limits = (np.mean(apparent_magnitude)
                    + np.std(apparent_magnitude)
                    * np.array([-plot_std_limit, plot_std_limit]))

    if cluster_shading is None:
        cluster_shading = np.ones(data_gaia.shape[0])

    # Work out how many points will fall in the pm range
    good_x = np.logical_and(color > x_limits[0], color < x_limits[1])
    good_y = np.logical_and(apparent_magnitude > y_limits[0], apparent_magnitude < y_limits[1])
    n_points = np.count_nonzero(np.logical_and(good_x, good_y))

    # Calculate a rough guess at a good alpha value
    alpha_estimate = utilities.calculate_alpha(fig, axis, n_points, 1)

    # CMD plot
    axis.plot(color, apparent_magnitude,
              '.', ms=1, alpha=alpha_estimate, c='k')

    axis.set_xlabel(r'm_{bp} - m_{gp}')
    axis.set_ylabel(r'm_{G}')
    axis.set_title('colour magnitude diagram (Gaia photometry)')
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.invert_yaxis()

    # Plot clusters if labels have been specified
    if cluster_labels is not None:
        cmap = plt.get_cmap("tab10")  # Get the default matplotlib colourmap

        # We cycle over the clusters, grabbing their indices and plotting
        # them in a new colour erri time =)
        for i_color, a_cluster in enumerate(cluster_indices):
            stars_in_this_cluster = cluster_labels == a_cluster

            # Make colors and give them alpha values
            colors = np.tile(cmap(i_color % 10), (np.count_nonzero(stars_in_this_cluster), 1))
            colors[:, 3] = cluster_shading[stars_in_this_cluster]

            axis.scatter(color[stars_in_this_cluster],
                         apparent_magnitude[stars_in_this_cluster],
                         marker='.',
                         s=cluster_marker_radius**2,
                         c=colors,
                         zorder=100,
                         label=a_cluster)

    return axis


def ra_versus_parallax(fig,
                       axis,
                       data_gaia: pd.DataFrame,
                       cluster_labels: np.ndarray = None,
                       cluster_indices: Optional[Union[list, np.ndarray]] = None,
                       cluster_shading: Optional[np.ndarray] = None,
                       plot_std_limit: float = 3.0,
                       x_limits: Optional[Union[list, np.ndarray]] = None,
                       y_limits: Optional[Union[list, np.ndarray]] = None,
                       cluster_marker_radius: float = 1.0):
    """Makes a plot of right ascension vs parallax on an axis.

    Notes:
        You can specify as few as one cluster_indices if cluster overplotting
        is too crowded.

    Args:
        fig (matplotlib figure): the figure element. Required to calculate alpha values precisely.
        axis (matplotlib axis): the colour magnitude diagram axis.
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        cluster_labels (np.ndarray): the cluster membership labels for
            clustered stars. This should be the default sklearn.cluster output.
        cluster_indices (list-like, optional): the clusters to plot in
            cluster_labels. For instance, you may not want to plot '-1'
            clusters (which are noise) produced by algorithm like DBSCAN.
        cluster_shading (np.ndarray, optional): an array of floats in the range 0, 1 for all clusters to use to shade
            their colour (aka alpha value). Useful for displaying e.g. HDBSCAN soft clustering.
            Default: None
        plot_std_limit (float): standard deviation of parameters to use
            to find plotting limits if none are explicitly specified.
            Default: 1.5
        x_limits (list-like, optional): the minimum and maximum ra limits.
            Default: None
        y_limits (list-like, optional): the minimum and maximum parallax limits.
            Default: None
        cluster_marker_radius (float): radius of the cluster marker. Useful to increase when clusters are hard to see
            against background points.
            Default: 1.0

    Returns:
        axis (matplotlib axis): the modified colour magnitude diagram axis.

    """

    # Calculate the stuff we need to plot
    # M = m - 5 * log10(d [pc]) + 5
    ra = data_gaia['ra']
    parallax = data_gaia["parallax"]

    # Calculate our own plotting limits if none have been specified
    if x_limits is None:
        x_limits = np.asarray([ra.min(), ra.max()])

    if y_limits is None:
        y_limits = (np.mean(parallax)
                    + np.std(parallax)
                    * np.array([-plot_std_limit, plot_std_limit]))

    if cluster_shading is None:
        cluster_shading = np.ones(data_gaia.shape[0])

    # Work out how many points will fall in the pm range
    good_x = np.logical_and(ra > x_limits[0], ra < x_limits[1])
    good_y = np.logical_and(parallax > y_limits[0], parallax < y_limits[1])
    n_points = np.count_nonzero(np.logical_and(good_x, good_y))

    # Calculate a rough guess at a good alpha value
    alpha_estimate = utilities.calculate_alpha(fig, axis, n_points, 1)

    # CMD plot
    axis.plot(ra, parallax,
              '.', ms=1, alpha=alpha_estimate, c='k')

    axis.set_xlabel(r'ra')
    axis.set_ylabel(r'parallax')
    axis.set_title('parallax vs. ra')
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    # Plot clusters if labels have been specified
    if cluster_labels is not None:
        cmap = plt.get_cmap("tab10")  # Get the default matplotlib colourmap

        # We cycle over the clusters, grabbing their indices and plotting
        # them in a new colour erri time =)
        for i_color, a_cluster in enumerate(cluster_indices):
            stars_in_this_cluster = cluster_labels == a_cluster

            # Make colors and give them alpha values
            colors = np.tile(cmap(i_color % 10), (np.count_nonzero(stars_in_this_cluster), 1))
            colors[:, 3] = cluster_shading[stars_in_this_cluster]

            axis.scatter(ra[stars_in_this_cluster],
                         parallax[stars_in_this_cluster],
                         marker='.',
                         s=cluster_marker_radius**2,
                         c=colors,
                         zorder=100,
                         label=a_cluster)

    return axis
