"""A set of functions for adding standardised things to an axis."""

import numpy as np
import matplotlib.pyplot as plt
from . import utilities
from . import process


def position_and_pmotion(axis_1, axis_2, my_data_gaia,
                         open_cluster_pm_to_mark=None,
                         my_pmra_plot_limits=None,
                         my_pmdec_plot_limits=None,
                         my_plot_std_limit=1.5):
    """Makes a scatter plot of position and proper motion for a given cluster,
    using plot_helper_calculate_alpha to prevent over-saturation of the
    figures.

    Args:
        axis_1 (matplotlib axis): the positional (ra/dec) axis.
        axis_2 (matplotlib axis): the proper motion (pmra/pmdec) axis.
        my_data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        open_cluster_pm_to_mark (list-like, optional): the co-ordinates
            (pmra, pmdec) of a point to mark on the proper motion diagram, such
            as a literature value.
        my_pmra_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the right ascension direction plot limits.
        my_pmdec_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the declination direction plot limits.
        my_plot_std_limit (float): standard deviation of proper motion to use
            to find plotting limits if none are explicitly specified.
            Default: 1.5

    Returns:
        axis_1 (matplotlib axis): the modified position axis.
        axis_2 (matplotlib axis): the modified proper motion axis.
    """
    # Calculate our own plotting limits if none have been specified
    if my_pmra_plot_limits is None:
        my_pmra_plot_limits = (np.mean(my_data_gaia.pmra)
                               + np.std(my_data_gaia.pmra)
                               * np.array([-my_plot_std_limit, my_plot_std_limit]))

    if my_pmdec_plot_limits is None:
        my_pmdec_plot_limits = (np.mean(my_data_gaia.pmdec)
                                + np.std(my_data_gaia.pmdec)
                                * np.array([-my_plot_std_limit, my_plot_std_limit]))

    # Better notation
    pmra_range = my_pmra_plot_limits
    pmdec_range = my_pmdec_plot_limits

    # Calculate a rough guess at a good alpha value
    alpha_estimate = utilities.calculate_alpha(
        my_data_gaia.shape[0], np.pi * 1.2 ** 2, 1)

    # Ra/dec plot
    axis_1.plot(my_data_gaia.ra, my_data_gaia.dec,
                '.', ms=1, alpha=alpha_estimate, c='k')
    axis_1.set_xlabel('ra')
    axis_1.set_ylabel('dec')
    axis_1.set_title('position')

    # Proper motion plot
    axis_2.plot(my_data_gaia.pmra, my_data_gaia.pmdec,
                '.', ms=1, alpha=alpha_estimate * 0.5, c='k')
    axis_2.set_xlabel('pmra')
    axis_2.set_ylabel('pmdec')
    axis_2.set_xlim(pmra_range)
    axis_2.set_ylim(pmdec_range)

    axis_2.set_title('proper motion')

    # Add a red cross at a defined location on the pmra/pmdec plot if desired
    if open_cluster_pm_to_mark is not None:
        axis_2[0, 1].plot(open_cluster_pm_to_mark[0], open_cluster_pm_to_mark[1],
                      'rx', ms=6)

    return [axis_1, axis_2]


def density_position_and_pmotion(axis_1,
                                 axis_2,
                                 my_gaia_data,
                                 my_kde_bandwidth_radec,
                                 my_kde_bandwidth_pmotion,
                                 my_pmra_plot_limits=None,
                                 my_pmdec_plot_limits=None,
                                 my_plot_std_limit=1.5,
                                 plot_levels=50):
    """Makes KDE plots of position density and proper motion density for a
    given cluster.

    Args:
        axis_1 (matplotlib axis): the positional (ra/dec) axis.
        axis_2 (matplotlib axis): the proper motion (pmra/pmdec) axis.
        my_gaia_data (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        my_kde_bandwidth_radec (float): the bandwidth of the kde for position.
        my_kde_bandwidth_pmotion (float): the bandwidth of the proper motion
            kde.
        my_pmra_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the right ascension direction plot limits.
        my_pmdec_plot_limits (list-like, optional): the minimum and maximum
            proper motion in the declination direction plot limits.
        my_plot_std_limit (float): standard deviation of proper motion to use
            to find plotting limits if none are explicitly specified.
            Default: 1.5
        plot_levels (int): number of levels to use in contour plots.
            Default: 50

    Returns:
        axis_1 (matplotlib axis): the modified position axis.
        axis_2 (matplotlib axis): the modified proper motion axis.

    """
    # Calculate our own plotting limits if none have been specified
    if my_pmra_plot_limits is None:
        my_pmra_plot_limits = (np.mean(my_gaia_data.pmra)
                               + np.std(my_gaia_data.pmra)
                               * np.array([-my_plot_std_limit, my_plot_std_limit]))

    if my_pmdec_plot_limits is None:
        my_pmdec_plot_limits = (np.mean(my_gaia_data.pmdec)
                                + np.std(my_gaia_data.pmdec)
                                * np.array([-my_plot_std_limit, my_plot_std_limit]))

    # Better notation
    pmra_range = my_pmra_plot_limits
    pmdec_range = my_pmdec_plot_limits

    # Perform KDE fits to the data
    mesh_ra, mesh_dec, density_ra_dec = process.kde_fit_2d(
        my_gaia_data[['ra', 'dec']].to_numpy(),
        my_kde_bandwidth_radec)

    mesh_pmra, mesh_pmdec, density_pmotion = process.kde_fit_2d(
        my_gaia_data[['pmra', 'pmdec']].to_numpy(),
        my_kde_bandwidth_pmotion,
        x_limits=pmra_range,
        y_limits=pmdec_range)

    # Plot the results
    # Ra/dec
    axis_1.contourf(mesh_ra, mesh_dec,
                    density_ra_dec,
                    levels=plot_levels,
                    cmap=plt.cm.Reds)
    axis_1.set_xlabel('ra')
    axis_1.set_ylabel('dec')
    axis_1.text(0.02, 1.02, f'bw={my_kde_bandwidth_radec:.4f}',
                ha='left', va='bottom', transform=axis_1[1, 0].transAxes,
                fontsize='small',
                bbox=dict(boxstyle='round', facecolor='w', alpha=0.8))

    # Pmra/pmdec
    axis_2.contourf(mesh_pmra, mesh_pmdec,
                    density_pmotion,
                    levels=plot_levels,
                    cmap=plt.cm.Blues)
    axis_2.set_xlabel('pmra')
    axis_2.set_ylabel('pmdec')
    axis_2.text(0.02, 1.02, f'bw={my_kde_bandwidth_pmotion:.4f}',
                ha='left', va='bottom', transform=axis_2[1, 1].transAxes,
                fontsize='small',
                bbox=dict(boxstyle='round', facecolor='w', alpha=0.8))

    # Return stuff. Because I'm nice like that =)
    return [axis_1, axis_2]


def color_magnitude_diagram(axis, my_data_gaia, cluster_labels=None,
                                 cluster_indices=None, my_plot_std_limit=3.0,
                                 x_limits=None, y_limits=None):
    """Makes a colour magnitude diagram plot of a given star field, and may
    also overplot star clusters within that field.

    Notes:
        You can specify as few as one cluster_indices if cluster overplotting
        is too crowded.

    Args:
        axis (matplotlib axis): the colour magnitude diagram axis.
        my_data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        cluster_labels (list-like, optional): the cluster membership labels for
            clustered stars. This should be the default sklearn.cluster output.
        cluster_indices (list-like, optional): the clusters to plot in
            cluster_labels. For instance, you may not want to plot '-1'
            clusters (which are noise) produced by algorithsm like DBSCAN.
        my_plot_std_limit (float): standard deviation of parameters to use
            to find plotting limits if none are explicitly specified.
            Default: 1.5
        x_limits (list-like, optional): the minimum and maximum blue minus red
            plotting limits.
        y_limits (list-like, optional): the minimum and maximum absolute
            magnitude plotting limits.

    Returns:
        axis (matplotlib axis): the modified colour magnitude diagram axis.

    """

    # Calculate the stuff we need to plot
    # M = m - 5 * log10(d [pc]) + 5
    apparent_magnitude = my_data_gaia["phot_g_mean_mag"]

    bp_minus_rp_colour = (my_data_gaia["phot_bp_mean_mag"]
                          - my_data_gaia["phot_rp_mean_mag"])

    # Calculate our own plotting limits if none have been specified
    if x_limits is None:
        x_limits = (np.mean(bp_minus_rp_colour)
                    + np.std(bp_minus_rp_colour)
                    * np.array([-my_plot_std_limit, my_plot_std_limit]))

    if y_limits is None:
        y_limits = (np.mean(apparent_magnitude)
                    + np.std(apparent_magnitude)
                    * np.array([-my_plot_std_limit, my_plot_std_limit]))

    # Calculate a rough guess at a good alpha value
    alpha_estimate = utilities.calculate_alpha(
        my_data_gaia.shape[0], np.pi * 1.2 ** 2, 1)

    # CMD plot
    axis.plot(bp_minus_rp_colour, apparent_magnitude,
              '.', ms=1, alpha=alpha_estimate, c='k')
    axis.invert_yaxis()

    axis.set_xlabel(r'm_{bp} - m_{gp}')
    axis.set_ylabel(r'm_{G}')
    axis.set_title('colour magnitude diagram (Gaia photometry)')
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    # Plot clusters if labels have been specified
    if cluster_labels is not None:
        for a_cluster in cluster_indices:
            # We cycle over the clusters, grabbing their indices and plotting
            # them in a new colour erri time =)
            stars_in_this_cluster = np.where(cluster_labels == a_cluster)[0]
            axis.plot(bp_minus_rp_colour[stars_in_this_cluster],
                      apparent_magnitude[stars_in_this_cluster],
                      '.', ms=1, alpha=1.0)

    return axis
