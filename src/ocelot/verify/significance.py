"""Tools for performing likelihood ratio hypothesis tests on cluster candidates."""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from typing import Union, Optional

from .find import get_field_stars_around_clusters
from .stats import _fit_kth_nn_distribution, _likelihood_ratio_test, _ks_test, _welch_t_test, _mann_whitney_rank_test
from ..cluster.epsilon import kth_nn_distribution
from ..plot.utilities import calculate_alpha

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
    ax.scatter(x_data[cluster_star_bool_array], y_data[cluster_star_bool_array], s=2, c="r", zorder=200)
    ax.scatter(x_data[field_star_indices], y_data[field_star_indices], s=2, c="b", zorder=100)

    # Get the limits that work in this case and then also add other stars
    x_limits, y_limits = ax.get_xlim(), ax.get_ylim()

    good_field_stars = np.logical_and.reduce(
        (x_data > x_limits[0], x_data < x_limits[1], y_data > y_limits[0], y_data < y_limits[1])
    )
    ax.scatter(
        x_data[good_field_stars],
        y_data[good_field_stars],
        s=2,
        c="k",
        zorder=0,
        alpha=calculate_alpha(ax.figure, ax, np.count_nonzero(good_field_stars), 2, scatter_plot=True, max_alpha=0.1),
    )

    # Reset limits (so that the plot is as cropped as possible) and return
    ax.set(xlim=x_limits, ylim=y_limits)
    return ax


def _plot_cluster_probability_distributions(ax, nn_distances, fit_params, cluster_number):
    """Plots the nearest neighbour distributions of the cluster stars and the associated field stars as normed
    probability distributions."""
    # Grab lines of the model fits & normalise them
    x_range = np.linspace(
        0, np.max(np.hstack((nn_distances[0][cluster_number], nn_distances[1][cluster_number]))), num=100
    )

    # Plot some normalised histograms bebe!!
    ax.hist(
        nn_distances[0][cluster_number], bins="auto", histtype="step", color="r", lw=2, label="cluster", density=True
    )
    ax.hist(
        nn_distances[1][cluster_number],
        bins="auto",
        histtype="step",
        color="b",
        lw=2,
        label="local field",
        density=True,
    )
    # ax.hist(true_field_nn_distances, bins='auto', histtype='step', color="k", ls="--", lw=2,
    #            label='true field', density=True)

    # Also plot the fits (if available)
    # For the cluster...
    if fit_params[0] is not None:
        y_cluster = kth_nn_distribution(x_range, *fit_params[0][cluster_number])
        y_cluster /= np.trapz(y_cluster, x=x_range)
        ax.plot(x_range, y_cluster, "r:", lw=2, label="cluster fit")

    # ... and the field!
    if fit_params[1] is not None:
        y_field = kth_nn_distribution(x_range, *fit_params[1][cluster_number])
        y_field /= np.trapz(y_field, x=x_range)
        ax.plot(x_range, y_field, "b:", lw=2, label="l. field fit")

    # Beautify and return
    ax.legend(edgecolor="k")
    ax.set(xlabel="kth nearest neighbour distance", ylabel="probability")
    return ax


def _cluster_significance_diagnostic_plots(
    data_rescaled: np.ndarray,
    labels: np.ndarray,
    nn_distances: Union[tuple, list],
    fit_params: Union[tuple, list],
    cluster_significances: dict,
    field_star_indices: dict,
    test_type: str,
    plot_kwargs: Optional[dict] = None,
    output_dir: Optional[Union[Path, str]] = None,
):
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
    required_plots = int(np.ceil(data_rescaled.shape[1] / 2) + 1)
    plot_settings["n_rows"] = int(np.ceil(required_plots / plot_settings["n_cols"]))

    # Loop over every cluster, making a plot!
    for a_cluster in field_star_indices:
        # Matplotlib setup!
        cluster_stars = labels == a_cluster
        fig = plt.figure(
            figsize=(
                plot_settings["inches_per_ax"] * plot_settings["n_cols"],
                plot_settings["inches_per_ax"] * plot_settings["n_rows"],
            ),
            dpi=plot_settings["dpi"],
        )
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
            axes.append(fig.add_subplot(plot_settings["n_rows"], plot_settings["n_cols"], i_plot))
            axes[-1] = _plot_points_cropped_on_ax(
                axes[-1], data_rescaled[:, dim_1], data_rescaled[:, dim_2], cluster_stars, field_star_indices[a_cluster]
            )
            axes[-1].set(xlabel=f"dim {dim_1}", ylabel=f"dim {dim_2}")

        # Also plot the nearest neighbour probability distributions and model fits
        axes.append(fig.add_subplot(plot_settings["n_rows"], plot_settings["n_cols"], required_plots))
        axes[-1] = _plot_cluster_probability_distributions(axes[-1], nn_distances, fit_params, a_cluster)

        # Process the title, since it can be a bit weird when doing all significances
        # Also, we don't use fig.suptitle (sadly) because that gets cropped by fig.tight_layout, but I can't get rid
        # of fig.tight_layout because it stops things being fucked (I think) when there are lots of plots
        if test_type == "all":
            significance_string = (
                f"lr: {cluster_significances['likelihood'][a_cluster]:.2f}  /  "
                f"t: {cluster_significances['welch_t'][a_cluster]:.2f}  /  "
                f"mw: {cluster_significances['mann_w'][a_cluster]:.2f} \n"
                f"k1+: {cluster_significances['ks_one+'][a_cluster]:.2f}  /  "
                f"k1-: {cluster_significances['ks_one-'][a_cluster]:.2f}  /  "
                f"diff: {cluster_significances['ks_one+'][a_cluster] - cluster_significances['ks_one-'][a_cluster]:.2f}"
                f"\nk2+: {cluster_significances['ks_two+'][a_cluster]:.2f}  /  "
                f"k2-: {cluster_significances['ks_two-'][a_cluster]:.2f}  /  "
                f"diff: {cluster_significances['ks_two+'][a_cluster] - cluster_significances['ks_two-'][a_cluster]:.2f}"
            )
        else:
            significance_string = f"sig: {cluster_significances[a_cluster]}"

        fig.suptitle(
            f"Cluster {a_cluster}\n{significance_string}\n"
            f"cluster/local field members: {np.count_nonzero(cluster_stars)}"
            f" / {len(field_star_indices[a_cluster])}",
            ha="left",
            x=0.10,
            fontsize="medium",
        )

        # Final beautification and output
        fig.subplots_adjust(hspace=0.25, wspace=0.3)
        fig.savefig(
            output_dir / f"{plot_settings['file_prefix']}{a_cluster}_significance.png", dpi=plot_settings["dpi"]
        )

        plt.close(fig)


DEFAULT_KNN_KWARGS = dict(
    overcalculation_factor=3.0,
    min_field_stars=100,
    max_field_stars=500,
    n_jobs=1,
    nn_kwargs=None,
    max_iter=100,
    kd_tree=None,
    cluster_nn_distance_type="internal",
    verbose=False,
)


def cluster_significance_test(
    data_rescaled: np.ndarray,
    labels: np.ndarray,
    min_samples: int = 10,
    knn_kwargs: Optional[dict] = None,
    sampling_resolution: int = 200,
    return_field_star_indices: bool = False,
    make_diagnostic_plots: bool = False,
    plot_kwargs: Optional[dict] = None,
    plot_output_dir: Optional[Union[Path, str]] = None,
    test_type: str = "mann_w",
):
    """Calculates the significance of a cluster by looking at the nearest neighbour distribution of local field
    stars, and performing a hypothesis test to evaluate whether or not cluster stars are more compatible with a cluster
    or the field.

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
                overcalculation_factor=3.,
                min_field_stars=100,
                max_field_stars=500,
                n_jobs=-1,
                nn_kwargs=None,
                max_iter=100,
                kd_tree=None,
                cluster_nn_distance_type="internal",
        sampling_resolution (int): resolution to sample fitted kth nn models at for likelihood ratio test.
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
        data_rescaled, labels, min_samples=min_samples, **knn_kwargs_to_use
    )

    # Loop over clusters, doing the tests!!!
    # ALL TESTS
    if test_type == "all":
        # Get fit params
        cluster_fit_params = _fit_kth_nn_distribution(cluster_nn_distances, min_samples, resolution=sampling_resolution)
        field_fit_params = _fit_kth_nn_distribution(field_nn_distances, min_samples, resolution=sampling_resolution)

        # Do the tests
        significances, test_statistics = {}, {}

        significances["likelihood"], test_statistics["likelihood"] = _likelihood_ratio_test(
            cluster_nn_distances, cluster_fit_params, field_fit_params, cdf_resolution=sampling_resolution
        )

        significances["ks_one+"], test_statistics["ks_one+"] = _ks_test(
            cluster_nn_distances, field_fit_params, one_sample=True, cdf_resolution=sampling_resolution
        )
        significances["ks_two+"], test_statistics["ks_two+"] = _ks_test(
            cluster_nn_distances, field_nn_distances, one_sample=False
        )

        significances["ks_one-"], test_statistics["ks_one-"] = _ks_test(
            cluster_nn_distances,
            field_fit_params,
            one_sample=True,
            cdf_resolution=sampling_resolution,
            alternative="less",
        )
        significances["ks_two-"], test_statistics["ks_two-"] = _ks_test(
            cluster_nn_distances, field_nn_distances, one_sample=False, alternative="less"
        )

        significances["welch_t"], test_statistics["welch_t"] = _welch_t_test(cluster_nn_distances, field_nn_distances)

        significances["mann_w"], test_statistics["mann_w"] = _mann_whitney_rank_test(
            cluster_nn_distances, field_nn_distances
        )

    # LIKELIHOOD TEST
    elif test_type == "likelihood":
        cluster_fit_params = _fit_kth_nn_distribution(cluster_nn_distances, min_samples, resolution=sampling_resolution)
        field_fit_params = _fit_kth_nn_distribution(field_nn_distances, min_samples, resolution=sampling_resolution)

        significances, test_statistics = _likelihood_ratio_test(
            cluster_nn_distances, cluster_fit_params, field_fit_params, cdf_resolution=sampling_resolution
        )

    # ONE SAMPLE KS TEST
    elif test_type == "ks_one_sample":
        cluster_fit_params = None
        field_fit_params = _fit_kth_nn_distribution(field_nn_distances, min_samples, resolution=sampling_resolution)
        significances, test_statistics = _ks_test(
            cluster_nn_distances, field_fit_params, one_sample=True, cdf_resolution=sampling_resolution
        )

    # TWO SAMPLE KS TEST
    elif test_type == "ks_two_sample":
        cluster_fit_params, field_fit_params = None, None
        significances, test_statistics = _ks_test(cluster_nn_distances, field_nn_distances, one_sample=True)

    # T TEST  (We do the Welch t-test - not the student's - which doesn't assume equal variance.)
    elif test_type == "t_test":
        cluster_fit_params, field_fit_params = None, None
        significances, test_statistics = _welch_t_test(cluster_nn_distances, field_nn_distances)

    # MANN WHITNEY
    elif test_type == "mann_w":
        cluster_fit_params, field_fit_params = None, None
        significances, test_statistics = _mann_whitney_rank_test(cluster_nn_distances, field_nn_distances)

    # BAD test_type VALUE
    else:
        raise ValueError(
            f"specified test_type {test_type} was not recognised! Please specify one of the following: "
            "'all', 'likelihood', 'ks_one_sample', 'ks_two_sample', 't_test', or 'mann_w'."
        )

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
            output_dir=plot_output_dir,
        )

    # Return time!
    if return_field_star_indices:
        return significances, test_statistics, field_star_indices, cluster_nn_distances, field_nn_distances
    else:
        return significances, test_statistics
