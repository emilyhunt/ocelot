"""A set of tests for use with the pytest module, covering ocelot.isochrone"""

# FUCKING HATE PYTHON IMPORTS AAAA
# (the below fixes this though)
try:
    from .context import ocelot
except ModuleNotFoundError:
    print('Unable to find ocelot via .context! Trying to import from your python path instead...')
try:
    import ocelot
except ModuleNotFoundError:
    raise ModuleNotFoundError('Unable to find ocelot')

import numpy as np
import pickle
import pytest
from pathlib import Path

path_blanco_1 = Path("./test_data/blanco_1_gaia_dr2_gmag_18_cut.pickle")
path_blanco_1_labels = Path("./test_data/blanco_1_gaia_dr2_gmag_18_cut_LABELS.pickle")


def test_cluster_significance_test():
    """Tests the cluster significance test from ocelot.verify.significance.py. Should be fun!"""
    # Grab the data
    with open(path_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)
    with open(path_blanco_1_labels, 'rb') as handle:
        labels = pickle.load(handle)

    # Re-scale and generally pre-process it!
    data_gaia = ocelot.cluster.recenter_dataset(data_gaia, center=(data_gaia['ra'].mean(), data_gaia['dec'].mean()))
    data_rescaled = ocelot.cluster.rescale_dataset(
        data_gaia, columns_to_rescale=('lon', 'lat', 'pmlon', 'pmlat', 'parallax'))

    # Insert two fake clusters: the first is a selection of non-overdense stars and the second is just some random stars
    # in the data that are totally unclustered!
    np.random.seed(42)
    unclustered_stars = labels == -1
    cluster_1 = np.logical_and.reduce((unclustered_stars,
                                       data_gaia['lon'] > 0, data_gaia['lon'] < 1,
                                       data_gaia['lat'] > 0, data_gaia['lat'] < 1,
                                       data_gaia['pmlon'] > -5, data_gaia['pmlon'] < 5,
                                       data_gaia['pmlat'] > -5, data_gaia['pmlat'] < 5,
                                       data_gaia['parallax'] > 0.25, data_gaia['parallax'] < 3,))
    cluster_2 = np.logical_and.reduce((unclustered_stars, np.invert(cluster_1),
                                       np.random.uniform(size=len(labels)) > 0.975))
    labels[cluster_1] = 1
    labels[cluster_2] = 2

    # Check that the setup worked
    unique_labels, counts = np.unique(labels, return_counts=True)
    assert np.all(counts == np.asarray([14111, 262, 65, 347]))

    # Get us some significances!
    significances, log_likelihoods = ocelot.verify.cluster_significance_test(
        data_rescaled,
        labels,
        min_samples=10,
        make_diagnostic_plots=True,
        plot_output_dir=Path("./test_results/cluster_significance_test3"),
        test_type="all"
    )


if __name__ == "__main__":
    test_cluster_significance_test()

