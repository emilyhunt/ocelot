"""A set of functions for adding standardised things to an axis, specifically for star cluster plotting."""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import process
from .. import utilities


def position_and_pmotion(
        axis_1,
        axis_2,
        data_gaia: pd.DataFrame,
        open_cluster_pm_to_mark: Optional[Union[list, np.ndarray]] = None,
        pmra_plot_limits: Optional[Union[list, np.ndarray]] = None,
        pmdec_plot_limits: Optional[Union[list, np.ndarray]] = None,
        plot_std_limit: float = 1.5):
    """Makes a scatter plot of position and proper motion for a given cluster,
    using plot_helper_calculate_alpha to prevent over-saturation of the
    figures.

    Args:
        axis_1 (matplotlib axis): the positional (ra/dec) axis.
        axis_2 (matplotlib axis): the proper motion (pmra/pmdec) axis.
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        open_cluster_pm_to_mark (list-like, optional): the co-ordinates (pmra, pmdec) of a point to mark on the proper
            motion diagram, such as a literature value.
        pmra_plot_limits (list-like, optional): the minimum and maximum proper motion in the right ascension direction
            plot limits.
        pmdec_plot_limits (list-like, optional): the minimum and maximum proper motion in the declination direction
            plot limits.
        plot_std_limit (float): standard deviation of proper motion to use to find plotting limits if none are
            explicitly specified. Default: 1.5

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

    # Better notation
    pmra_range = pmra_plot_limits
    pmdec_range = pmdec_plot_limits

    # Calculate a rough guess at a good alpha value
    alpha_estimate = utilities.calculate_alpha(
        data_gaia.shape[0], np.pi * 1.2 ** 2, 1)

    # Ra/dec plot
    axis_1.plot(data_gaia.ra, data_gaia.dec,
                '.', ms=1, alpha=alpha_estimate, c='k')
    axis_1.set_xlabel('ra')
    axis_1.set_ylabel('dec')
    axis_1.set_title('position')

    # Proper motion plot
    axis_2.plot(data_gaia.pmra, data_gaia.pmdec,
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

    # Perform KDE fits to the data
    mesh_ra, mesh_dec, density_ra_dec = process.kde_fit_2d(
        data_gaia[['ra', 'dec']].to_numpy(),
        kde_bandwidth_radec, kde_resolution=kde_resolution)

    mesh_pmra, mesh_pmdec, density_pmotion = process.kde_fit_2d(
        data_gaia[['pmra', 'pmdec']].to_numpy(),
        kde_bandwidth_pmotion,
        x_limits=pmra_range,
        y_limits=pmdec_range,
        kde_resolution=kde_resolution)

    # Plot the results
    # Ra/dec
    axis_1.contourf(mesh_ra, mesh_dec,
                    density_ra_dec,
                    levels=plotting_resolution,
                    cmap=plt.cm.Reds)
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
                    cmap=plt.cm.Blues)
    axis_2.set_xlabel('pmra')
    axis_2.set_ylabel('pmdec')
    axis_2.text(0.02, 1.02, f'bw={kde_bandwidth_pmotion:.4f}',
                ha='left', va='bottom', transform=axis_2.transAxes,
                fontsize='small',
                bbox=dict(boxstyle='round', facecolor='w', alpha=0.8))

    # Return stuff. Because I'm nice like that =)
    return [axis_1, axis_2]


def color_magnitude_diagram(axis,
                            data_gaia: pd.DataFrame,
                            cluster_labels: np.ndarray = None,
                            cluster_indices: Optional[Union[list, np.ndarray]] = None,
                            plot_std_limit: float = 3.0,
                            x_limits: Optional[Union[list, np.ndarray]] = None,
                            y_limits: Optional[Union[list, np.ndarray]] = None):
    """Makes a colour magnitude diagram plot of a given star field, and may
    also overplot star clusters within that field.

    Notes:
        You can specify as few as one cluster_indices if cluster overplotting
        is too crowded.

    Args:
        axis (matplotlib axis): the colour magnitude diagram axis.
        data_gaia (pandas.DataFrame): the Gaia data read in to a DataFrame.
            Keys should be unchanged from default Gaia source table names.
        cluster_labels (np.ndarray): the cluster membership labels for
            clustered stars. This should be the default sklearn.cluster output.
        cluster_indices (list-like, optional): the clusters to plot in
            cluster_labels. For instance, you may not want to plot '-1'
            clusters (which are noise) produced by algorithm like DBSCAN.
        plot_std_limit (float): standard deviation of parameters to use
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
    apparent_magnitude = data_gaia["phot_g_mean_mag"]

    bp_minus_rp_colour = (data_gaia["phot_bp_mean_mag"]
                          - data_gaia["phot_rp_mean_mag"])

    # Calculate our own plotting limits if none have been specified
    if x_limits is None:
        x_limits = (np.mean(bp_minus_rp_colour)
                    + np.std(bp_minus_rp_colour)
                    * np.array([-plot_std_limit, plot_std_limit]))

    if y_limits is None:
        y_limits = (np.mean(apparent_magnitude)
                    + np.std(apparent_magnitude)
                    * np.array([-plot_std_limit, plot_std_limit]))

    # Calculate a rough guess at a good alpha value
    alpha_estimate = utilities.calculate_alpha(
        data_gaia.shape[0], np.pi * 1.2 ** 2, 1)

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
