"""Some code from my Jupyter notebook that I wanted to run with PyCharm debugging!"""

import numpy as np
import matplotlib.pyplot as plt
import ocelot
import hdbscan
import pickle
import time

from pathlib import Path


# INITIAL SETUP
# Change this one
data_location = Path("/media/emily/Emilys External Disk1/data")

if not data_location.exists():
    raise ValueError("data location is invalid!")

# These should stay the same
location_mwsc = data_location / Path('sc_cats/mwsc_ii_catalogue')
location_dr2_10_ocs = data_location / Path('gaia_dr2_10ocs')
location_simulated_populations = data_location / Path('simulated_ocs/191122_logt_6_to_10_parsec_1.2s')

# A bit of automated handling to make a sorted list of locations
location_dr2_10_ocs_iterable = np.sort(list(location_dr2_10_ocs.glob('*.pickle')))


def read_oc(number: int):
    with open(location_dr2_10_ocs_iterable[number], 'rb') as handle:
        return pickle.load(handle)


# RUNNING CODE
# Disgusting code to reverse engineer the names
name_10_ocs = ["_".join(location_dr2_10_ocs_iterable[x].stem.rsplit("-")[0].rsplit("_")[1:]) for x in range(10)]

# Cycle over everyone!
start = 0
end = 5
output_dir_name = "500_stars_locked"

for i, a_name in enumerate(name_10_ocs[start:end], start=start):

    print(f"Working on {a_name}")

    # Get the data
    data_gaia = read_oc(i)
    data_gaia = ocelot.cluster.cut_dataset(data_gaia, parameter_cuts={'phot_g_mean_mag': [-np.inf, 18]})
    data_gaia = ocelot.cluster.recenter_dataset(
        data_gaia, center=(data_gaia['ra'].mean(), data_gaia['dec'].mean()))

    if np.any(data_gaia['ra'] > 270) and np.any(data_gaia['ra'] < 90):
        data_gaia['ra'] = np.where(data_gaia['ra'] < 180, data_gaia['ra'], data_gaia['ra'] - 360)

    # Rescale it
    data_rescaled = ocelot.cluster.rescale_dataset(
        data_gaia, columns_to_rescale=('lon', 'lat', 'pmlon', 'pmlat', 'parallax'))

    # Get labels and probs
    print("  doing clustering analysis")
    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10, cluster_selection_method='leaf')
    labels = clusterer.fit_predict(data_rescaled)
    probabilities = clusterer.probabilities_
    persistences = clusterer.cluster_persistence_
    cluster_time = time.time() - start_time

    del clusterer

    # Grab all of our cluster significances for every cluster
    print("  doing significance analysis")

    start_time = time.time()
    significances, log_likelihoods = ocelot.verify.cluster_significance_test(
        data_rescaled,
        labels,
        min_samples=10,
        make_diagnostic_plots=True,
        plot_output_dir=Path(f"./{output_dir_name}/{a_name}"),
        knn_kwargs={'overcalculation_factor': 5, 'cluster_nn_distance_type': 'internal', 'n_jobs': 1,
                    'min_field_stars': 500, 'max_field_stars': 500},
        test_type='all',
    )
    sig_time = time.time() - start_time

    print(f"  cluster: {cluster_time:.2f}s -- sig: {sig_time:.2f}s")

    # Also want unique labels without -1
    unique_labels, counts = np.unique(labels, return_counts=True)
    good_labels = unique_labels != -1
    unique_labels = unique_labels[good_labels]
    counts = counts[good_labels]

    for a_cluster in unique_labels:
        print(f"  plotting {a_cluster+1} of {len(unique_labels)}")

        # Get significance and log likelihood in a good format
        sig = (
            f"lr: {significances['likelihood'][a_cluster]:.2f}  /  "
            f"t: {significances['welch_t'][a_cluster]:.2f}  /  "
            f"mw: {significances['mann_w'][a_cluster]:.2f} \n"
            f"k1+: {significances['ks_one+'][a_cluster]:.2f}  /  "
            f"k1-: {significances['ks_one-'][a_cluster]:.2f}  /  "
            f"diff: {significances['ks_one+'][a_cluster] - significances['ks_one-'][a_cluster]:.2f}"
            f"\nk2+: {significances['ks_two+'][a_cluster]:.2f}  /  "
            f"k2-: {significances['ks_two-'][a_cluster]:.2f}  /  "
            f"diff: {significances['ks_two+'][a_cluster] - significances['ks_two-'][a_cluster]:.2f}"
        )

        fig, ax = ocelot.plot.clustering_result(
            data_gaia, labels, [-2, -2, -2] + [a_cluster], probabilities, make_parallax_plot=True,
            cmd_plot_y_limits=[8, 18], cluster_marker_radius=(3, 3, 3, 3), dpi=300, show_figure=False,
            figure_size=(6, 6),
            figure_title=f"Field around {a_name}, cluster {a_cluster}\n"
                         f"  detected by HDBSCAN w/ m_clSize=40, m_Pts=10\n"
                         f"  significance: {sig}\n"
                         f"  persistence: {persistences[a_cluster]:.4f}\n"
                         f"  n_stars: {counts[a_cluster]}"
        )
        ax[0, 0].set_title(None)
        ax[0, 1].set_title(None)
        ax[1, 0].set_title(None)
        ax[1, 1].set_title(None)

        fig.subplots_adjust(wspace=0.3)

        fig.savefig(f"./{output_dir_name}/{a_name}/{a_cluster}_ocelot.png", bbox_inches="tight")

        plt.close(fig)
