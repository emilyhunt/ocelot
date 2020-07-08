"""Tools for performing likelihood ratio hypothesis tests on cluster candidates."""

import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import chi2, norm

from pathlib import Path

from typing import Union, Optional

from .epsilon import _kth_nn_distribution, _constrained_a


def get_field_stars_around_clusters(data_rescaled, labels, min_samples=10, overcalculation_factor=2.,
                                    min_field_stars=100, n_jobs=-1, nn_kwargs=None, max_iter=100):
    """Gets and returns a cloud of representative field stars around each reported cluster.

    Args:
        data_rescaled (np.ndarray): data to calculate SNRs for with shape (n_samples, n_features).
        labels (np.ndarray): labels of the data with shape (n_samples,), in the sklearn format for density-based
            clustering with at least field_sample stars with a noise label of -1.
        min_samples (int): the min_samples nearest neighbour will be returned.
        overcalculation_factor (float): min_samples*overcalculation_factor nearest neighbors of cluster stars will be
            searched to try and find compatible field stars to evaluate against for the field step.
        min_field_stars (int): minimum number of field stars to use in field calculations. If enough aren't found, then
            we'll traverse deeper into the k nearest neighbour graph to find more.
        n_jobs (int or None): number of jobs to use for calculating nn distances. -1 uses all cores. None uses 1.
        nn_kwargs (dict): a dict of kwargs to pass to sklearn.neighbors.NearestNeighbors when running on
            cluster or field stars.
        max_iter (int): maximum number of iterations to run when trying to search for neighbours for a single cluster
            before we raise a RuntimeError.

    Returns:
        a dict of cluster nn distances
        a dict of field nn distances
        (indices into labels of the field stars used for each field calculation)

    """
    if nn_kwargs is None:
        nn_kwargs = {}

    # Grab unique labels
    unique_labels, unique_label_counts = np.unique(labels, return_counts=True)

    # Drop any -1 "clusters" (these are in fact noise as reported by e.g. DBSCAN)
    good_clusters = unique_labels != -1
    n_field_stars = unique_label_counts[np.invert(good_clusters)]
    unique_labels = unique_labels[good_clusters]
    unique_label_counts = unique_label_counts[good_clusters]

    # Check that no biscuitry is in progress
    if np.any(unique_label_counts < min_samples):
        raise ValueError(f"one of the reported clusters is smaller than the value of min_samples "
                         f"of {min_samples}! Method will fail.")
    if n_field_stars < len(labels) * 2/3:
        warnings.warn("fewer than 2/3rds of points are field stars! This may be too few to accurately find neighbours "
                      "of cluster points.", RuntimeWarning)

    # Cycle over each cluster, grabbing stats about its own nearest neighbor distances
    cluster_nn_distances_dict = {}
    field_nn_distances_dict = {}

    field_star_indices_dict = {}

    # First off, let's make a KD tree for the entire field and fit it
    field_nn_classifier = NearestNeighbors(
        n_neighbors=min_samples, n_jobs=n_jobs, **nn_kwargs)
    field_nn_classifier.fit(data_rescaled)

    # Now, let's cycle over everything and find cluster nearest neighbour info
    for a_cluster in unique_labels:

        # Firstly, get the nearest neighbour distances for the cluster alone
        cluster_nn_classifier = NearestNeighbors(
            n_neighbors=min_samples, n_jobs=n_jobs, **nn_kwargs)
        cluster_stars = labels == a_cluster
        cluster_nn_classifier.fit(data_rescaled[cluster_stars])

        # And get the nn distances!
        cluster_nn_distances, cluster_nn_indices = cluster_nn_classifier.kneighbors(data_rescaled[cluster_stars])

        # Next off, let's look for at least min_field_stars stars
        n_field_stars = 0
        field_star_distances = []
        field_star_indices = []
        cluster_stars = cluster_stars.nonzero()[0]

        stars_to_check = cluster_stars
        already_done_stars = np.asarray([], dtype=int)  # i.e. an array of everything we already calculated a d for

        field_n_neighbors = int(np.round(min_samples * overcalculation_factor))
        i = 0

        while n_field_stars < min_field_stars:

            # Get some distances!
            field_nn_distances, field_nn_indices = field_nn_classifier.kneighbors(
                data_rescaled[stars_to_check], n_neighbors=field_n_neighbors)

            if i == 0:
                field_nn_indices = field_nn_indices[:, min_samples:]

            # Drop all objects that are connected to cluster stars and any non-unique objects
            # First, test if individual stars are good or bad
            valid_indices = np.isin(field_nn_indices, cluster_stars, invert=True)

            # Then, see if their row has enough non-cluster stars to be unpolluted
            # (we only do this if we aren't on the first step, as on the first step this doesn't matter since
            # our objective is just to move away)
            if i != 0:
                good_rows = np.all(valid_indices[:, :min_samples], axis=1)
                valid_indices = np.logical_and(good_rows.reshape(-1, 1), valid_indices)

            # Remove anything we've done before
            valid_indices[valid_indices] = np.isin(
                field_nn_indices[valid_indices], already_done_stars, invert=True)

            # Lastly, fuck off anything already done =p
            already_done_stars = np.append(already_done_stars, stars_to_check)

            # If we're on our first run, then we'll loop around again and get some new distances for the
            # now-found field stars. If not, then these distances are for non-cluster stars and we can start
            # to seriously consider stopping the loop
            if i != 0:
                # Grab the valid field star distances
                valid_field_stars_at_min_samples = np.unique(valid_indices[:, min_samples - 1].nonzero()[0])
                valid_field_star_distances = field_nn_distances[
                    valid_field_stars_at_min_samples, min_samples - 1]

                # Add them to the running order
                n_field_stars += len(valid_field_star_distances)
                field_star_distances.append(valid_field_star_distances)

                field_star_indices.append(field_nn_indices[
                                          valid_field_stars_at_min_samples, min_samples - 1])

            # Make sure that stars_to_check is now a flat array of indices into data_rescaled, aka the thing we need
            stars_to_check = np.unique(field_nn_indices[valid_indices.nonzero()])

            # Quick check that the while loop isn't the longest thing ever lol
            i += 1
            if i >= max_iter:
                raise RuntimeError(f"unable to traverse the graph of field stars in {max_iter} iterations!")

        # Convert field_stars_to_get_distances_for into a 1D array and not a Python list
        field_star_distances = np.hstack(field_star_distances)

        # Save these nearest neighbor distances
        cluster_nn_distances_dict[a_cluster] = cluster_nn_distances[:, min_samples - 1]
        field_nn_distances_dict[a_cluster] = field_star_distances
        field_star_indices_dict[a_cluster] = np.hstack(field_star_indices)

    # Return time!
    return cluster_nn_distances_dict, field_nn_distances_dict, field_star_indices_dict


class KthNNDistributionCDF:
    def __init__(self, a, d, k, grid_size=200, limits=(0, 1)):
        """Precalculates a grid of kth nn distribution CDF values for a given a, d and k."""
        # Precalculate cumulatively summed values!
        x_range = np.linspace(*limits, num=grid_size)
        y_range = np.cumsum(_kth_nn_distribution(x_range, a, d, k))

        # Normalise these values so that the max val is just 1!
        y_range /= np.max(y_range)

        # Interpolate this and set it to __call__!
        self.cdf_function = interp1d(x_range, y_range, kind="linear", fill_value=(0., 1.), bounds_error=False)

    def __call__(self, x):
        return self.cdf_function(x)


class KthNNDistributionPDF:
    def __init__(self, a, d, k, grid_size=200, limits=(0, 1)):
        """Precalculates a grid of kth nn distribution PDF values for a given a, d and k."""
        # Precalculate cumulatively summed values!
        x_range = np.linspace(*limits, num=grid_size)
        y_range = _kth_nn_distribution(x_range, a, d, k)

        # Normalise these values so that the max val is just 1!
        y_range /= np.trapz(y_range, x=x_range)

        # Interpolate this and set it to __call__!
        self.pdf_function = interp1d(x_range, y_range, kind="linear", fill_value=(0., 0.), bounds_error=False)

    def __call__(self, x):
        return self.pdf_function(x)


class _CurveToFit:
    def __init__(self, k):
        self.k = k

    def __call__(self, x_range, *params):
        """A version of the Chandrasekhar1943 curve for use with scipy's curve_fit function.

        Args:
            x_range: should *already* be cumulatively summed and sorted wrt point number!
            params: (a, dimension)

        Returns:
            y_range of points evaluated at x_range values!

        """
        y_func = np.cumsum(_kth_nn_distribution(x_range, params[0], params[1], self.k))
        y_func /= np.trapz(y_func, x=x_range)

        return y_func


def fit_kth_nn_distribution(nn_distances, min_samples, resolution=200):
    """Fits a kth nn distribution model to an array of nn distances."""
    # Sort nn distances and make an array of point numbers
    x_data = np.sort(nn_distances)
    y_data = np.arange(1, len(nn_distances) + 1)

    # Interpolate this function into something more valid for fitting
    y_interpolator = interp1d(x_data, y_data, kind="linear")

    x_range = np.linspace(x_data.min(), x_data.max(), num=resolution)
    y_range = y_interpolator(x_range)

    # Normalise
    y_range /= np.trapz(y_range, x=x_range)

    # Fit a function!
    # Get starting guesses
    k = min_samples
    d_0 = min_samples / 2  # Might want a better estimate sometimes!
    a_0 = _constrained_a(d_0, k, np.mean(nn_distances))

    # Minimize
    result, covariance = curve_fit(
        _CurveToFit(k), x_range, y_range, p0=np.asarray([a_0, d_0]), method="trf",
        bounds=([0, 0], [np.inf, np.inf]),
        verbose=False)

    # Return values
    return result[0], result[1], k


def _likelihood_of_object(x_data, membership_probabilities, a, d, k, cdf_resolution=200):
    # Get decent estimates of the range of the function we should work with
    limits = (0.0, x_data.max() * 2)

    # Make a cdf!
    func = KthNNDistributionCDF(a, d, k, limits=limits, grid_size=cdf_resolution)

    # Evaluate all the points at the cdf!
    return np.sum(np.log(func(x_data) * membership_probabilities))


def _likelihood_ratio_test(x_cluster, membership_probabilities, cluster_fit_params, field_fit_params,
                           cdf_resolution=200):
    """Finds the maximum likelihood of the cluser & field models and does a simple comparison between the two
    to see which one is a better fit.

    See: http://rnowling.github.io/machine/learning/2017/10/07/likelihood-ratio-test.html

    """
    # Grab likelihoods
    likelihood_cluster = _likelihood_of_object(x_cluster, membership_probabilities, *cluster_fit_params,
                                               cdf_resolution=cdf_resolution)
    likelihood_field = _likelihood_of_object(x_cluster, membership_probabilities, *field_fit_params,
                                             cdf_resolution=cdf_resolution)

    # Log and square the ratio!
    log_likelihood = 2 * (likelihood_cluster - likelihood_field)

    # Convert to a pval, then to a sigma significance
    p_val = chi2.sf(log_likelihood, (len(x_cluster) - 1) * (len(cluster_fit_params) - 1))
    significance = -norm.ppf(p_val / 2)

    # Stop -0.0 from happening. Only 0.0 is allowed!
    return np.abs(significance), log_likelihood


DEFAULT_DIAGNOSTIC_PLOT_SETTINGS = dict(
    dpi=100,
    ncols=3,
    inches_per_ax=4,
)


def _plot_points_cropped_on_ax(ax, x_data, y_data, cluster_star_bool_array, field_star_indices):
    """Function for plotting points on a diagnostic axis. Very generalised! Will make sure that the axis is as cropped
    as possible."""
    # Plot the cluster & field stars
    ax.scatter(x_data[cluster_star_bool_array], y_data[cluster_star_bool_array], s=2, c='r', zorder=200)
    ax.scatter(x_data[field_star_indices], y_data[field_star_indices], s=2, c='b', zorder=100)

    # Get the limits that work in this case and then also add other stars
    x_limits, y_limits = ax.get_xlim(), ax.get_ylim()

    good_field_stars = np.logical_and.reduce(
        (x_data > x_limits[0], x_data < x_limits[1], y_data > y_limits[0], y_data < y_limits[1]))
    ax.scatter(x_data[good_field_stars], y_data[good_field_stars], s=2, c=(0.5, 0.5, 0.5, 0.5), zorder=0)

    # Reset limits (so that the plot is as cropped as possible) and return
    ax.set(xlim=x_limits, ylim=y_limits)
    return ax


def _plot_cluster_probability_distributions(ax, nn_distances, fit_params, cluster_number):
    """Plots the nearest neighbour distributions of the cluster stars and the associated field stars as normed
    probability distributions."""
    # Grab lines of the model fits & normalise them
    x_range = np.linspace(0,
                          np.max(np.hstack((nn_distances[0][cluster_number], nn_distances[1][cluster_number]))),
                          num=100)
    y_cluster = _kth_nn_distribution(x_range, *fit_params[0][cluster_number])
    y_field = _kth_nn_distribution(x_range, *fit_params[1][cluster_number])

    y_cluster /= np.trapz(y_cluster, x=x_range)
    y_field /= np.trapz(y_field, x=x_range)

    # Plot some normalised histograms bebe!!
    ax.hist(nn_distances[0][cluster_number], bins='auto', histtype='step', color="r", lw=2,
               label='cluster', density=True)
    ax.hist(nn_distances[1][cluster_number], bins='auto', histtype='step', color="b", lw=2,
               label='local field', density=True)
    # ax.hist(true_field_nn_distances, bins='auto', histtype='step', color="k", ls="--", lw=2,
    #            label='true field', density=True)

    # Plot some models, bebe!!
    ax.plot(x_range, y_cluster, 'r:', lw=2, label="cluster fit")
    ax.plot(x_range, y_field, 'b:', lw=2, label="l. field fit")

    # Beautify and return
    ax.legend(edgecolor='k')
    ax.set(xlabel=f"{fit_params[0][-1]}th nearest neighbour distances",
           ylabel="probability")
    return ax


def _cluster_significance_diagnostic_plots(data_rescaled: np.ndarray, labels: np.ndarray,
                                           nn_distances: Union[tuple, list], fit_params: Union[tuple, list],
                                           cluster_significances: dict, field_star_indices: dict,
                                           plot_kwargs: Optional[dict] = None,
                                           output_dir: Optional[Union[Path, str]] = None):
    """Plotter for calculate_cluster_significance diagnostic plots! Will make a *lot* of plots. Turn on only when
    you're very stuck. =)"""
    # Setup of input args
    if output_dir is None:
        output_dir = Path("./cluster_significance_diagnostic_plots")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if plot_kwargs is not None:
        plot_settings = DEFAULT_DIAGNOSTIC_PLOT_SETTINGS.copy()
        plot_settings.update(plot_kwargs)
    else:
        plot_settings = DEFAULT_DIAGNOSTIC_PLOT_SETTINGS

    # Loop over every cluster, making a plot!
    for a_cluster in field_star_indices:
        # Numerical setup!
        cluster_stars = labels == a_cluster

        # First off, work out how many plots we'll need and the requisite figure geometry
        required_plots = int(np.ceil(data_rescaled.shape[1]/2) + 1)

        plot_settings['n_rows'] = int(np.ceil(required_plots / plot_settings['n_cols']))
        fig = plt.figure(figsize=(plot_settings['inches_per_ax'] * plot_settings['n_cols'],
                                  plot_settings['inches_per_ax'] * plot_settings['n_rows']),
                         dpi=plot_settings['dpi'])
        axes = []

        # Loop over, plotting the data dimensions
        i_plot = 0
        while i_plot < required_plots - 1:
            # Grab whichever dimension we're plotting on, being careful if we have an odd number of dimensions to loop
            # back around
            dim_1 = i_plot * 2
            if dim_1 + 1 >= data_rescaled.shape[1]:
                dim_2 = 0
            else:
                dim_2 = dim_1 + 1

            # Plot it!
            axes.append(fig.add_subplot(plot_settings['n_rows'], plot_settings['n_cols'], i_plot))
            axes[-1] = _plot_points_cropped_on_ax(
                axes[-1], data_rescaled[:, dim_1], data_rescaled[:, dim_2], cluster_stars,
                field_star_indices[a_cluster])
            axes[-1].set(xlabel=f"dim {dim_1}", ylabel=f"dim {dim_2}")

            i_plot += 1

        # Also plot the nearest neighbour probability distributions and model fits
        axes.append(fig.add_subplot(plot_settings['n_rows'], plot_settings['n_cols'], i_plot))
        axes[-1] = _plot_cluster_probability_distributions(axes[-1], nn_distances, fit_params, a_cluster)

        # Final beautification and output
        fig.suptitle(f"Cluster {a_cluster}: sig {cluster_significances[a_cluster]:.2f}\n"
                     f"  cluster/local field members: {np.count_nonzero(cluster_stars)}"
                     f" / {len(field_star_indices[a_cluster])}",
                     ha='left', fontsize="medium")
        fig.tight_layout()
        fig.savefig(output_dir / f"{a_cluster} significance.png", dpi=plot_settings['dpi'])

        plt.close(fig)


DEFAULT_KNN_KWARGS = dict(
    overcalculation_factor=3.,
    min_field_stars=100,
    n_jobs=-1,
    nn_kwargs=None,
    max_iter=100)


def calculate_cluster_significance(data_rescaled, labels, min_samples=10, membership_probabilities=None,
                                   knn_kwargs=None, sampling_resolution=200, return_field_star_indices=False,
                                   make_diagnostic_plots=False, plot_kwargs=None, plot_output_dir=None):
    """Calculates the significance of a cluster by looking at the nearest neighbour distribution of local field
    stars, fitting Chandrasekhar1943 models, and performing a likelihood ratio hypothesis test to evaluate
    whether or not cluster stars are more compatible with a cluster or the field.
    """
    if membership_probabilities is None:
        membership_probabilities = np.ones(labels.shape[0])

    if knn_kwargs is not None:
        knn_kwargs_to_use = DEFAULT_KNN_KWARGS.copy()
        knn_kwargs_to_use.update(knn_kwargs)
    else:
        knn_kwargs_to_use = DEFAULT_KNN_KWARGS

    # Get all nn distances
    cluster_nn_distances, field_nn_distances, field_star_indices = get_field_stars_around_clusters(
        data_rescaled, labels, min_samples=min_samples, **knn_kwargs_to_use)

    # Cycle over clusters, getting fit parameters and applying the test
    significances = {}
    log_likelihoods = {}
    cluster_fit_params = {}
    field_fit_params = {}

    for a_cluster in cluster_nn_distances:
        # Get fit parameters
        cluster_fit_params[a_cluster] = fit_kth_nn_distribution(
            cluster_nn_distances[a_cluster], min_samples, resolution=sampling_resolution)
        field_fit_params[a_cluster] = fit_kth_nn_distribution(
            field_nn_distances[a_cluster], min_samples, resolution=sampling_resolution)

        # Grab all probabilities
        a_probabilities = membership_probabilities[labels == a_cluster]

        # Perform the test!
        significances[a_cluster], log_likelihoods[a_cluster] = _likelihood_ratio_test(
            cluster_nn_distances[a_cluster], a_probabilities, cluster_fit_params[a_cluster],
            field_fit_params[a_cluster], cdf_resolution=sampling_resolution)

    if make_diagnostic_plots:
        _cluster_significance_diagnostic_plots(
            data_rescaled,
            labels,
            (cluster_nn_distances, field_nn_distances),
            (cluster_fit_params, field_fit_params),
            significances,
            field_star_indices,
            plot_kwargs=plot_kwargs,
            output_dir=plot_output_dir
        )

    # Return time!
    if return_field_star_indices:
        return significances, log_likelihoods, field_star_indices
    else:
        return significances, log_likelihoods
