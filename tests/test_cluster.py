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

import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.csr import csr_matrix

path_to_blanco_1 = Path('./test_data/blanco_1_gaia_dr2_gmag_18_cut.pickle')


def test_cut_dataset():
    """Tests the functionality of dataset cutting at ocelot.cluster.cut_dataset."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    # Define some chonky cuts to make
    cuts = {'phot_g_mean_mag': [6, 16],
            'r_est': [100, np.inf],
            'parallax_over_error': [10, np.inf]}

    data_cut = ocelot.cluster.cut_dataset(data_gaia, parameter_cuts=cuts)

    # Check that the correct number of cuts were performed
    assert data_cut.shape == (5039, 28)

    # Check that no values fall outside of the cut ranges for gmag (easy)
    assert np.invert(np.any(np.logical_or(data_cut['phot_g_mean_mag'] < cuts['phot_g_mean_mag'][0],
                                          data_cut['phot_g_mean_mag'] > cuts['phot_g_mean_mag'][1])))

    # Check that no values fall outside of the cut ranges for r_est (could indicate fucking up with np.inf)
    assert np.invert(np.any(np.logical_or(data_cut['r_est'] < cuts['r_est'][0],
                                          data_cut['r_est'] > cuts['r_est'][1])))

    return data_cut


def test_rescale_dataset():
    """Tests the functionality of the dataset re-scaling of ocelot.cluster.rescale_dataset."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    # Re-scale the data with standard scaling and check that it has zero mean, unit variance
    data_rescaled = ocelot.cluster.rescale_dataset(data_gaia, scaling_type='standard')

    # Check for zero mean, unit standard deviation
    means = np.mean(data_rescaled, axis=0)
    assert np.allclose(means, np.zeros(5), rtol=0.0, atol=1e-8)

    std_deviation = np.std(data_rescaled, axis=0)
    assert np.allclose(std_deviation, np.ones(5), rtol=0.0, atol=1e-8)

    # Check that calling an invalid scaler type throws an error
    with pytest.raises(ValueError, match="Selected scaling_type not supported!"):
        ocelot.cluster.rescale_dataset(data_gaia, scaling_type='an unsupported type of scaling, I guess')

    # Check that a cheeky nan value raises an error
    data_gaia.loc[100, 'ra'] = np.nan
    with pytest.raises(ValueError, match="At least one value in data_gaia is not finite! Unable to rescale the data."):
        ocelot.cluster.rescale_dataset(data_gaia, scaling_type='robust')

    return data_gaia, data_rescaled


def test_precalculate_nn_distances():
    """Tests ocelot.cluster.precalculate_nn_distances()"""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    # Re-scale the data with standard scaling and check that it has zero mean, unit variance
    data_rescaled = ocelot.cluster.rescale_dataset(data_gaia, scaling_type='robust')

    # Calculate some nearest neighbor distances baby!
    sparse_matrix, distance_matrix = ocelot.cluster.precalculate_nn_distances(
        data_rescaled, n_neighbors=10, return_sparse_matrix=True, return_knn_distance_array=True)

    # Check that specifying no return type throws an error
    with pytest.raises(ValueError, match="Nothing was specified for return. That's probably not intentional!"):
        ocelot.cluster.precalculate_nn_distances(data_rescaled, n_neighbors=10,
                                                 return_sparse_matrix=False, return_knn_distance_array=False)

    # Check that the sparse matrix is, in fact, sparse
    assert type(sparse_matrix) == csr_matrix

    # Check that the correct shapes are returned
    assert distance_matrix.shape == (14785, 10)
    assert sparse_matrix.shape == (14785, 14785)

    return sparse_matrix, distance_matrix


def test_acg18_epsilon():
    """Tests the OCELOT implementation of the Alfred Castro-Ginard+18 method of determining an optimum value for DBSCAN.
    """
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    # Define some chonky cuts to make to make this test faster
    cuts = {'phot_g_mean_mag': [-np.inf, 16],
            'r_est': [200, 300],
            'parallax_over_error': [5, np.inf]}
    cuts = {}

    data_cut = ocelot.cluster.cut_dataset(data_gaia, parameter_cuts=cuts)

    # Re-scale the data with standard scaling and check that it has zero mean, unit variance
    data_rescaled = ocelot.cluster.rescale_dataset(data_cut, scaling_type='robust')

    # Calculate some nearest neighbor distances baby!
    distance_matrix = ocelot.cluster.precalculate_nn_distances(
        data_rescaled, n_neighbors=10, return_sparse_matrix=False, return_knn_distance_array=True)

    # Epsilon time
    np.random.seed(42)
    epsilons, random_distance_matrix = ocelot.cluster.epsilon.acg18_epsilon(
        data_rescaled, distance_matrix, n_repeats=2, min_samples='all', return_last_random_distance=True)

    # Check that the correct shapes are returned
    assert distance_matrix.shape == random_distance_matrix.shape
    assert epsilons.shape == (10,)

    # Check the values against a target set that were correct at first implementation
    target_epsilons = np.asarray([0.00943362, 0.02696594, 0.04015817, 0.04884566, 0.05278521,
                                  0.05693874, 0.06477257, 0.06957070, 0.06899090, 0.07156333])
    assert np.allclose(epsilons, target_epsilons, rtol=0.0, atol=1e-8)

    # Check that we just get back one (correct) value if that's all we ask for
    np.random.seed(42)
    single_epsilon = ocelot.cluster.epsilon.acg18_epsilon(
        data_rescaled, distance_matrix, n_repeats=2, min_samples=10, return_last_random_distance=False)
    assert type(single_epsilon) == float or np.float
    assert np.allclose(single_epsilon, 0.07344272967449159, rtol=0.0, atol=1e-8)

    # Throw some errors by being a fuckwit with min_samples
    # Wrong string
    with pytest.raises(ValueError, match="Incompatible number or string of min_samples specified.\n"
                                         "Allowed values:\n"
                                         "- integer less than max_neighbors_to_calculate and greater than zero\n"
                                         "- 'all', which calculates all values upto max_neighbors_to_calculate\n"):
        ocelot.cluster.epsilon.acg18_epsilon(
            data_rescaled, distance_matrix, n_repeats=2, min_samples='your mum', return_last_random_distance=False)

    # Min_samples greater than the number of neighbours
    with pytest.raises(ValueError, match="min_samples may not be larger than max_neighbors_to_calculate"):
        ocelot.cluster.epsilon.acg18_epsilon(
            data_rescaled, distance_matrix, n_repeats=2, min_samples=100, return_last_random_distance=False)

    # Min_samples less than one
    with pytest.raises(ValueError, match="min_samples may not be larger than max_neighbors_to_calculate"):
        ocelot.cluster.epsilon.acg18_epsilon(
            data_rescaled, distance_matrix, n_repeats=2, min_samples=0, return_last_random_distance=False)

    return epsilons, random_distance_matrix



if __name__ == '__main__':
    #gaia, rescaled = test_rescale_dataset()
    #cut = test_cut_dataset()
    #spar, dist = test_precalculate_nn_distances()
    eps, ran = test_acg18_epsilon()
