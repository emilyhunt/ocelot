"""A set of tests for use with the pytest module, covering ocelot.isochrone"""

from ocelot import verify, cluster
import numpy as np
import pickle
from pathlib import Path


test_data_path = Path(__file__).parent / "test_data"
path_blanco_1 = test_data_path / "blanco_1_gaia_dr2_gmag_18_cut.pickle"
path_blanco_1_labels = test_data_path / "blanco_1_gaia_dr2_gmag_18_cut_LABELS.pickle"


def test_cluster_significance_test():
    """Tests the cluster significance test from ocelot.verify.significance.py. Should be
    fun!
    """
    # Grab the data
    with open(path_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)
    with open(path_blanco_1_labels, "rb") as handle:
        labels = pickle.load(handle)

    # Re-scale and generally pre-process it!
    data_gaia = cluster.recenter_dataset(
        data_gaia, center=(data_gaia["ra"].mean(), data_gaia["dec"].mean())
    )
    data_rescaled = cluster.rescale_dataset(
        data_gaia, columns_to_rescale=("lon", "lat", "pmlon", "pmlat", "parallax")
    )

    # Insert two fake clusters: the first is a selection of non-overdense stars and the 
    # second is just some random stars
    # in the data that are totally unclustered!
    np.random.seed(42)
    unclustered_stars = labels == -1
    cluster_1 = np.logical_and.reduce(
        (
            unclustered_stars,
            data_gaia["lon"] > 0,
            data_gaia["lon"] < 1,
            data_gaia["lat"] > 0,
            data_gaia["lat"] < 1,
            data_gaia["pmlon"] > -5,
            data_gaia["pmlon"] < 5,
            data_gaia["pmlat"] > -5,
            data_gaia["pmlat"] < 5,
            data_gaia["parallax"] > 0.25,
            data_gaia["parallax"] < 3,
        )
    )
    cluster_2 = np.logical_and.reduce(
        (
            unclustered_stars,
            np.invert(cluster_1),
            np.random.uniform(size=len(labels)) > 0.975,
        )
    )
    labels[cluster_1] = 1
    labels[cluster_2] = 2

    # Check that the setup worked
    unique_labels, counts = np.unique(labels, return_counts=True)
    assert np.all(counts == np.asarray([14111, 262, 65, 347]))

    # Get us some significances!
    significances, log_likelihoods = verify.cluster_significance_test(
        data_rescaled,
        labels,
        min_samples=10,
        make_diagnostic_plots=True,
        plot_output_dir=Path("./test_results/cluster_significance_test3"),
        test_type="all",
    )
