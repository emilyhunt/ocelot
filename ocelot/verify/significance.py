"""Tools for performing likelihood ratio hypothesis tests on cluster candidates."""

import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import chi2, norm, kstest

from pathlib import Path

from typing import Union, Optional

from ..cluster.epsilon import kth_nn_distribution, constrained_a
from ..plot.utilities import calculate_alpha


def get_field_stars_around_clusters(data_rescaled: np.ndarray, labels, min_samples=10, overcalculation_factor=2.,
                                    min_field_stars=100, n_jobs=-1, nn_kwargs=None, max_iter=100):
    """Gets and returns a cloud of representative field stars around each reported cluster.

    Args:
        data_rescaled (np.ndarray): array of rescaled data, in shape (n_samples, n_features.)
        labels (np.ndarray): labels of the data with shape (n_samples,), in the sklearn format for density-based
            clustering with at least min_samples stars with a noise label of -1.
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
        y_range = np.cumsum(kth_nn_distribution(x_range, a, d, k))

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
        y_range = kth_nn_distribution(x_range, a, d, k)

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
        y_func = np.cumsum(kth_nn_distribution(x_range, params[0], params[1], self.k))
        y_func /= np.trapz(y_func, x=x_range)

        return y_func


def _fit_kth_nn_distribution(nn_distances: dict, min_samples: int, resolution: int = 200):
    """Fits a kth nn distribution model to a dict of arrays of nn distances."""
    result = {}

    for a_cluster in nn_distances:

        # Sort nn distances and make an array of point numbers
        x_data = np.sort(nn_distances[a_cluster])
        y_data = np.arange(1, len(nn_distances[a_cluster]) + 1)

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
        a_0 = constrained_a(d_0, k, np.mean(nn_distances[a_cluster]))

        # Minimize
        minimisation_result, covariance = curve_fit(
            _CurveToFit(k), x_range, y_range, p0=np.asarray([a_0, d_0]), method="trf",
            bounds=([0, 0], [np.inf, np.inf]),
            verbose=False)
        result[a_cluster] = (minimisation_result[0], minimisation_result[1], k)

    # Return values
    return result


def _likelihood_of_object(x_data, a, d, k, cdf_resolution=200):
    # Get decent estimates of the range of the function we should work with
    limits = (0.0, x_data.max() * 2)

    # Make a cdf!
    func = KthNNDistributionCDF(a, d, k, limits=limits, grid_size=cdf_resolution)

    # Evaluate all the points at the cdf!
    return np.sum(np.log(func(x_data)))


def _convert_one_sided_p_value_to_z_score(p_value):
    # np.abs is to stop -0.0 from happening. Only 0.0 is allowed!
    return np.abs(-norm.ppf(p_value / 2))


def _likelihood_ratio_test(nn_distances, cluster_fit_params, field_fit_params,
                           cdf_resolution=200):
    """Finds the maximum likelihood of the cluser & field models and does a simple comparison between the two
    to see which one is a better fit.

    See: http://rnowling.github.io/machine/learning/2017/10/07/likelihood-ratio-test.html

    """
    significances, log_likelihoods = {}, {}

    for a_cluster in nn_distances:

        # Grab likelihoods
        likelihood_cluster = _likelihood_of_object(nn_distances[a_cluster], *cluster_fit_params[a_cluster],
                                                   cdf_resolution=cdf_resolution)
        likelihood_field = _likelihood_of_object(nn_distances[a_cluster], *field_fit_params[a_cluster],
                                                 cdf_resolution=cdf_resolution)

        # Log and square the ratio!
        log_likelihoods[a_cluster] = 2 * (likelihood_cluster - likelihood_field)

        # Convert to a pval, then to a sigma significance
        p_val = chi2.sf(log_likelihoods[a_cluster], (len(nn_distances[a_cluster]) - 1) * (len(cluster_fit_params) - 1))
        significances[a_cluster] = _convert_one_sided_p_value_to_z_score(p_val)

    return significances, log_likelihoods


def _ks_test(cluster_nn_distances: dict, field_nn_distances: dict, one_sample: bool = True,
             cdf_resolution: int = 200):
    """Performs a KS test (one-sided) of cluster stars and field stars to see if they're compatible or not.

    Args:
        cluster_nn_distances (np.ndarray): dict of arrays of cluster nearest neighbour distances of shape (n_stars,)
        field_nn_distances (np.ndarray): dict of arrays of len(3) of field fit parameters if one_sample==True, or dict
            of arrays of of field nearest neighbour distances otherwise.
        one_sample (bool): whether or not to do the test in one sample mode. Can produce slightly more reliable results
            (since the number of field stars will stop being a factor) but requires that the fitting procedure was
            correct.
            Default: True
        cdf_resolution (int): resolution to sample the cdf at if one_sample=True.
            Default: 200

    Returns:
        significance value of the clusters and the KS test statistics in a dict.

    """
    significances, ks_test_statistics = {}, {}

    for a_cluster in cluster_nn_distances:
        # Turn the field_nn_distances into a callable function instead of a CDF if one_sample==True.
        if one_sample:
            limits = (0.0, np.max(cluster_nn_distances[a_cluster]) * 2)
            field_distribution = KthNNDistributionCDF(
                *field_nn_distances[a_cluster], grid_size=cdf_resolution, limits=limits)
        else:
            field_distribution = field_nn_distances[a_cluster]

        # Do the KS test, get a p-value, go home
        ks_test_statistics[a_cluster], p_value = kstest(
            cluster_nn_distances[a_cluster], field_distribution, alternative="greater")

        significances[a_cluster] = _convert_one_sided_p_value_to_z_score(p_value)

    return significances, ks_test_statistics


DEFAULT_DIAGNOSTIC_PLOT_SETTINGS = dict(
    dpi=100,
    n_cols=2,
    inches_per_ax=4,
    supertitle_y_coord=1.05,
    file_prefix="",
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
    ax.scatter(x_data[good_field_stars], y_data[good_field_stars], s=2, c='k', zorder=0,
               alpha=calculate_alpha(ax.figure, ax, np.count_nonzero(good_field_stars), 2, scatter_plot=True,
                                     max_alpha=0.1))

    # Reset limits (so that the plot is as cropped as possible) and return
    ax.set(xlim=x_limits, ylim=y_limits)
    return ax


def _plot_cluster_probability_distributions(ax, nn_distances, fit_params, cluster_number):
    """Plots the nearest neighbour distributions of the cluster stars and the associated field stars as normed
    probability distributions."""
    # Grab lines of the model fits & normalise them
    x_range = np.linspace(0, np.max(np.hstack((nn_distances[0][cluster_number], nn_distances[1][cluster_number]))),
                          num=100)

    # Plot some normalised histograms bebe!!
    ax.hist(nn_distances[0][cluster_number], bins='auto', histtype='step', color="r", lw=2,
            label='cluster', density=True)
    ax.hist(nn_distances[1][cluster_number], bins='auto', histtype='step', color="b", lw=2,
            label='local field', density=True)
    # ax.hist(true_field_nn_distances, bins='auto', histtype='step', color="k", ls="--", lw=2,
    #            label='true field', density=True)

    # Also plot the fits (if available)
    # For the cluster...
    if fit_params[0] is not None:
        y_cluster = kth_nn_distribution(x_range, *fit_params[0][cluster_number])
        y_cluster /= np.trapz(y_cluster, x=x_range)
        ax.plot(x_range, y_cluster, 'r:', lw=2, label="cluster fit")

    # ... and the field!
    if fit_params[1] is not None:
        y_field = kth_nn_distribution(x_range, *fit_params[1][cluster_number])
        y_field /= np.trapz(y_field, x=x_range)
        ax.plot(x_range, y_field, 'b:', lw=2, label="l. field fit")

    # Beautify and return
    ax.legend(edgecolor='k')
    ax.set(xlabel="kth nearest neighbour distance",
           ylabel="probability")
    return ax


def _cluster_significance_diagnostic_plots(data_rescaled: np.ndarray, labels: np.ndarray,
                                           nn_distances: Union[tuple, list], fit_params: Union[tuple, list],
                                           cluster_significances: dict, field_star_indices: dict, test_type: str,
                                           plot_kwargs: Optional[dict] = None,
                                           output_dir: Optional[Union[Path, str]] = None):
    """Plotter for cluster_significance_test diagnostic plots! Will make a *lot* of plots - one for every cluster!
    Turn on only when you're very stuck. =)

    Args:
        data_rescaled (np.ndarray): array of rescaled data, in shape (n_samples, n_features.)
        labels (np.ndarray): integer label produced by your clustering algorithm of choice for data_rescaled. Noise
            (unclustered stars) should have the label -1. Shape (n_samples,).
        nn_distances (tuple, list): length 2 list-like of nearest neighbour distance dicts for the cluster and the field
            stars respectively.
        fit_params (tuple, list): length 2 list-like of fit parameter dicts for the clusters and field stars
            respectively.
        cluster_significances (dict): a dict of cluster significances to include on the plots.
        field_star_indices (dict): a dict of the indices into data_rescaled of all the field stars selected for each
            cluster.
        test_type (str): the type of test the user requested. We need to know about this in case test_type=='all'!
        plot_kwargs (dict, optional): kwargs to pass to the plotting routine.
            Default: None, which reverts to:
                dpi=100
                ncols=3
                inches_per_ax=4
                supertitle_y_coord=1.05
                file_prefix: ''
        output_dir (str, pathlib.Path, optional): where to save diagnostic plots. If not specified, will revert to
            "./cluster_significance_diagnostic_plots".
            Default: None

    """
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

    # Work out how many plots we'll need and the requisite figure geometry
    required_plots = int(np.ceil(data_rescaled.shape[1]/2) + 1)
    plot_settings['n_rows'] = int(np.ceil(required_plots / plot_settings['n_cols']))

    # Loop over every cluster, making a plot!
    for a_cluster in field_star_indices:
        # Matplotlib setup!
        cluster_stars = labels == a_cluster
        fig = plt.figure(figsize=(plot_settings['inches_per_ax'] * plot_settings['n_cols'],
                                  plot_settings['inches_per_ax'] * plot_settings['n_rows']),
                         dpi=plot_settings['dpi'])
        axes = []

        # Loop over, plotting the data dimensions
        for i_plot in range(1, required_plots):  # 1 indexing since matplotlib's add_subplot is 1-indexed
            # Grab whichever dimension we're plotting on, being careful if we have an odd number of dimensions to loop
            # back around when we get to the last one
            dim_1 = (i_plot - 1) * 2
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

        # Also plot the nearest neighbour probability distributions and model fits
        axes.append(fig.add_subplot(plot_settings['n_rows'], plot_settings['n_cols'], required_plots))
        axes[-1] = _plot_cluster_probability_distributions(axes[-1], nn_distances, fit_params, a_cluster)

        # Process the title, since it can be a bit weird when doing all significances
        # Also, we don't use fig.suptitle (sadly) because that gets cropped by fig.tight_layout, but I can't get rid
        # of fig.tight_layout because it stops things being fucked (I think) when there are lots of plots
        if test_type == "all":
            significance_string = (f"lr: {cluster_significances['likelihood'][a_cluster]:.2f} / "
                                   f"ks1: {cluster_significances['ks_one'][a_cluster]:.2f} / "
                                   f"ks2: {cluster_significances['ks_two'][a_cluster]:.2f} ")
        else:
            significance_string = f"sig: {cluster_significances[a_cluster]}"

        axes[0].set_title(f"Cluster {a_cluster} -- {significance_string}\n"
                          f"cluster/local field members: {np.count_nonzero(cluster_stars)}"
                          f" / {len(field_star_indices[a_cluster])}", ha='left', x=0.00,
                          fontsize="medium")

        # Final beautification and output
        fig.tight_layout()
        fig.savefig(output_dir / f"{plot_settings['file_prefix']}{a_cluster}_significance.png",
                    dpi=plot_settings['dpi'])

        plt.close(fig)


DEFAULT_KNN_KWARGS = dict(
    overcalculation_factor=3.,
    min_field_stars=100,
    n_jobs=-1,
    nn_kwargs=None,
    max_iter=100)


def cluster_significance_test(data_rescaled: np.ndarray, labels: np.ndarray, min_samples: int = 10,
                              knn_kwargs: Optional[dict] = None, sampling_resolution: int = 200,
                              return_field_star_indices: bool = False,
                              make_diagnostic_plots: bool = False, plot_kwargs: Optional[dict] = None,
                              plot_output_dir: Optional[Union[Path, str]] = None,
                              test_type: str = 'all'):
    """Calculates the significance of a cluster by looking at the nearest neighbour distribution of local field
    stars, fitting Chandrasekhar1943 models, and performing a likelihood ratio hypothesis test to evaluate
    whether or not cluster stars are more compatible with a cluster or the field.

    # Todo: k = min_samples (current) or k = min_samples - 1 (ACG?)

    Args:
        data_rescaled (np.ndarray): array of rescaled data, in shape (n_samples, n_features.)
        labels (np.ndarray): integer label produced by your clustering algorithm of choice for data_rescaled. Noise
            (unclustered stars) should have the label -1. Shape (n_samples,).
        min_samples (int): min_samples used by your clustering algorithm, aka the kth nearest neighbour to
            look at.
            Default: 10
        knn_kwargs (dict, optional): extra kwargs for the kth nearest neighbour process in
            get_field_stars_around_clusters.
            Default: None, which reverts to:
                overcalculation_factor=3.
                min_field_stars=100
                n_jobs=-1
                nn_kwargs=None
                max_iter=100
        sampling_resolution (int): resolution to sample fitted kth nn models at.
            Default: 200
        return_field_star_indices (bool): whether or not to also return indices of field stars selected by the algorithm
            for your own analysis.
            Default: False
        make_diagnostic_plots (bool): whether or not to output diagnostic plots *for every cluster in labels* showing
            the cluster stars, selected field stars & model fits.
            Default: False
        plot_kwargs (dict, optional): kwargs to pass to the plotting routine.
            Default: None, which reverts to:
                dpi=100
                ncols=3
                inches_per_ax=4
        plot_output_dir (str, pathlib.Path, optional): where to save diagnostic plots. If not specified, will revert to
            "./cluster_significance_diagnostic_plots".
            Default: None

    Returns:
        dict of significances, with cluster labels as keys
        dict of log likelihoods (not normalised by the number of stars in each cluster!), mostly for diagnostics
        (a dict of indices into data_rescaled of selected field stars for each cluster, returned if
            return_field_star_indices is True)
    """

    if knn_kwargs is not None:
        knn_kwargs_to_use = DEFAULT_KNN_KWARGS.copy()
        knn_kwargs_to_use.update(knn_kwargs)
    else:
        knn_kwargs_to_use = DEFAULT_KNN_KWARGS

    # Get all nn distances
    cluster_nn_distances, field_nn_distances, field_star_indices = get_field_stars_around_clusters(
        data_rescaled, labels, min_samples=min_samples, **knn_kwargs_to_use)

    # Loop over clusters, doing the tests!!!
    # ALL TESTS
    if test_type == "all":
        # Get fit params
        cluster_fit_params = _fit_kth_nn_distribution(cluster_nn_distances, min_samples, resolution=sampling_resolution)
        field_fit_params = _fit_kth_nn_distribution(field_nn_distances, min_samples, resolution=sampling_resolution)

        # Do the tests
        significances, test_statistics = {}, {}

        significances['likelihood'], test_statistics['likelihood'] = _likelihood_ratio_test(
            cluster_nn_distances, cluster_fit_params, field_fit_params, cdf_resolution=sampling_resolution)

        significances['ks_one'], test_statistics['ks_one'] = _ks_test(
            cluster_nn_distances, field_fit_params, one_sample=True, cdf_resolution=sampling_resolution)
        significances['ks_two'], test_statistics['ks_two'] = _ks_test(
            cluster_nn_distances, field_nn_distances, one_sample=False)

    # LIKELIHOOD TEST
    elif test_type == "likelihood":
        cluster_fit_params = _fit_kth_nn_distribution(cluster_nn_distances, min_samples, resolution=sampling_resolution)
        field_fit_params = _fit_kth_nn_distribution(field_nn_distances, min_samples, resolution=sampling_resolution)

        significances, test_statistics = _likelihood_ratio_test(
            cluster_nn_distances, cluster_fit_params, field_fit_params, cdf_resolution=sampling_resolution)

    # ONE SAMPLE KS TEST
    elif test_type == "ks_one_sample":
        cluster_fit_params = None
        field_fit_params = _fit_kth_nn_distribution(field_nn_distances, min_samples, resolution=sampling_resolution)
        significances, test_statistics = _ks_test(
            cluster_nn_distances, field_fit_params, one_sample=True, cdf_resolution=sampling_resolution)

    # TWO SAMPLE KS TEST
    elif test_type == "ks_two_sample":
        cluster_fit_params, field_fit_params = None, None
        significances, test_statistics = _ks_test(
            cluster_nn_distances, field_nn_distances, one_sample=True)

    # BAD test_type VALUE
    else:
        raise ValueError(f"specified test_type {test_type} was not recognised! Please specify one of the following: "
                         "'all', 'likelihood', 'ks_one_sample' or 'ks_two_sample'.")

    # Diagnostic plot time!
    if make_diagnostic_plots:
        _cluster_significance_diagnostic_plots(
            data_rescaled,
            labels,
            (cluster_nn_distances, field_nn_distances),
            (cluster_fit_params, field_fit_params),
            significances,
            field_star_indices,
            test_type,
            plot_kwargs=plot_kwargs,
            output_dir=plot_output_dir
        )

    # Return time!
    if return_field_star_indices:
        return significances, test_statistics, field_star_indices
    else:
        return significances, test_statistics
