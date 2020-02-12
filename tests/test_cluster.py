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
import pandas as pd
import pytest
from scipy.sparse.csr import csr_matrix

path_to_blanco_1 = Path('./test_data/blanco_1_gaia_dr2_gmag_18_cut.pickle')

path_to_one_simulated_population = Path('./test_data/simulated_population.dat')
path_to_all_simulated_populations = Path('./test_data')

path_to_simulated_population_test_clusters = Path('./test_data/simulated_population_test_clusters')


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


def test_acg18():
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
    epsilons, random_distance_matrix = ocelot.cluster.epsilon.acg18(
        data_rescaled, distance_matrix, n_repeats=2, min_samples='all', return_last_random_distance=True)

    # Check that the correct shapes are returned
    assert distance_matrix.shape == random_distance_matrix.shape
    assert epsilons.shape == (10,)

    # Check the values against a target set that were correct at first implementation
    target_epsilons = np.asarray([0.01594943, 0.06773323, 0.09449874, 0.10952761, 0.12000926, 0.13023932,
                                  0.14131321, 0.14809204, 0.15114417, 0.15699153])
    assert np.allclose(epsilons, target_epsilons, rtol=0.0, atol=1e-8)

    # Check that we just get back one (correct) value if that's all we ask for
    np.random.seed(42)
    single_epsilon = ocelot.cluster.epsilon.acg18(
        data_rescaled, distance_matrix, n_repeats=2, min_samples=10, return_last_random_distance=False)

    assert type(single_epsilon) == float or np.float
    assert np.allclose(single_epsilon, 0.1588709230676416, rtol=0.0, atol=1e-8)

    # Throw some errors by being a fuckwit with min_samples
    # Wrong string
    with pytest.raises(ValueError, match="Incompatible number or string of min_samples specified.\n"
                                         "Allowed values:\n"
                                         "- integer less than max_neighbors_to_calculate and greater than zero\n"
                                         "- 'all', which calculates all values upto max_neighbors_to_calculate\n"):
        ocelot.cluster.epsilon.acg18(
            data_rescaled, distance_matrix, n_repeats=2, min_samples='your mum', return_last_random_distance=False)

    # Min_samples greater than the number of neighbours
    with pytest.raises(ValueError, match="min_samples may not be larger than max_neighbors_to_calculate"):
        ocelot.cluster.epsilon.acg18(
            data_rescaled, distance_matrix, n_repeats=2, min_samples=100, return_last_random_distance=False)

    # Min_samples less than one
    with pytest.raises(ValueError, match="min_samples may not be larger than max_neighbors_to_calculate"):
        ocelot.cluster.epsilon.acg18(
            data_rescaled, distance_matrix, n_repeats=2, min_samples=0, return_last_random_distance=False)

    return epsilons, random_distance_matrix


def test__summed_kth_nn_distribution_one_cluster():
    """Tests ocelot.cluster.epsilon_summed_kth_nn_distribution_one_cluster (and by extension, _kth_nn_distribution)."""
    # Set some parameters to play with
    field_constant = 0.3
    field_dimension = 5
    cluster_constant = 0.05
    cluster_dimension = 3
    cluster_fraction = 0.01
    parameters = np.asarray([field_constant, field_dimension, cluster_constant, cluster_dimension, cluster_fraction])

    # Calculate some residuals
    residual_inf = ocelot.cluster.epsilon._summed_kth_nn_distribution_one_cluster(
        parameters, 10, np.linspace(0.0, 1, num=50), y_range=np.linspace(1, 1000, num=50) ** 2, minimisation_mode=True)

    residual_good = ocelot.cluster.epsilon._summed_kth_nn_distribution_one_cluster(
        parameters, 10, np.linspace(0.1, 1, num=50), y_range=np.linspace(1, 1000, num=50) ** 2, minimisation_mode=True)

    # Check that the bad residual is inf
    assert residual_inf == np.inf

    # Check that the good residual is... good
    assert np.allclose(residual_good, 10316599967493.814, rtol=0.001, atol=1e-8)

    # Calculate a field
    y_fields = ocelot.cluster.epsilon._summed_kth_nn_distribution_one_cluster(
        parameters, 10, np.linspace(0, 1, num=50), minimisation_mode=False)

    # Check the shape
    assert y_fields.shape == (3, 50)

    # Check that we get np.inf on the first value, showing that my log comprehension works
    assert np.allclose(y_fields[:, 0], np.asarray([-np.inf, -np.inf, -np.inf]), rtol=0.0, atol=1e-8)

    # Check that the mean of everything else is right, and by extension it's... probably right
    assert np.allclose(np.mean(y_fields[:, 1:]), -1.4524090935710814, rtol=0.0, atol=1e-8)

    return 0


def test__find_sign_change_epsilons():
    """Tests ocelot.cluster.epsilon._find_sign_change_epsilons, a function that finds the biggest second derivative
    bump.

    Diagnostic code to plot what the function sees:

    import matplotlib.pyplot as plt
    x = np.log10(x)
    plt.plot(x, np.gradient(np.gradient(y, x), x), 'r-')
    plt.show()

    print(epsilon_values, all_sign_changes)

    """

    # Make some fake data - a squared sine wave that gets exponentially smaller
    x = np.geomspace(0.1, 40 * np.pi, num=200)
    y = (np.sin(np.log10(x)) * np.exp(-x / 10)) ** 2 - 0.2

    # Call the function and hope for a good result lol
    epsilon_values, all_sign_changes = ocelot.cluster.epsilon._find_sign_change_epsilons(
        x, y, return_all_sign_changes=True)

    # Check that it got the right number of sign changes
    assert all_sign_changes.shape == (3,)

    # Check that the main sign changes are right (could be sensitive to setting changes in the resolution of x)
    target = [0.15938995421984642, 0.6934124282589442, 2.2642842494315762]
    assert np.allclose(epsilon_values, target, rtol=0.0, atol=1e-8)


def test__find_curve_absolute_maximum_epsilons():
    """Tests ocelot.cluster.epsilon._find_curve_absolute_maximum_epsilons

    Test plotting code that shows what the function sees::

    import matplotlib.pyplot as plt
    x = np.log10(x)
    plt.plot(x, np.abs(np.gradient(np.gradient(y, x, ), x, )))
    plt.show()

    print(maximum_x)

    """
    # Make some fake data
    x = np.linspace(0, 10, num=200)
    y = x**2 * np.exp(-x) - 1

    # Call time
    maximum_x = ocelot.cluster.epsilon._find_curve_absolute_maximum_epsilons(x, y)

    assert np.allclose(maximum_x, 2.361809045226131, rtol=0.0, atol=1e-8)


def test_field_model(show_figure=False):
    """Tests ocelot.cluster.epsilon.field_model using data of Blanco 1. It can be worth testing this manually too,
    as it does quite a lot of funky stuff that's worth looking at."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    # Re-scale the data with standard scaling and check that it has zero mean, unit variance
    data_rescaled = ocelot.cluster.rescale_dataset(data_gaia, scaling_type='robust')

    # Calculate some nearest neighbor distances baby!
    distance_matrix = ocelot.cluster.precalculate_nn_distances(
        data_rescaled, n_neighbors=10, return_sparse_matrix=False, return_knn_distance_array=True)

    # Specify some plot options
    plot_options = {'number_of_derivatives': 2,
                    'figure_size': (6, 8),
                    'show_figure': show_figure,
                    'figure_title': 'Unit test of ocelot.cluster.epsilon.field_model',
                    }

    # See what the field model fit thinks of this
    success, epsilon_values, parameters, n_cluster_members = ocelot.cluster.epsilon.field_model(
        distance_matrix, min_samples=10, min_cluster_size=1, make_diagnostic_plot=True,
        **plot_options)

    # Test that the results are good
    assert success is True

    target_epsilon = [0.13765954, 0.1809968, 0.22433406, 0.25747432, 0.28968726]
    assert np.allclose(epsilon_values, target_epsilon, rtol=0.0, atol=1e-8)

    target_parameters = [0.26971331, 9.40067326, 0.08953837, 3.36263488, 0.02072574]
    assert np.allclose(parameters, target_parameters, rtol=0.0, atol=1e-8)

    assert n_cluster_members == 291


def test_read_cmd_simulated_populations():
    """Tests the ability of ocelot.cluster.SimulatedPopulations to read in CMD simulated populations in bulk."""
    target_columns = ['Z', 'age', 'Mini', 'Mass', 'logL', 'logTe', 'logg', 'cmd_label', 'mbolmag',
                      'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'log_age', 'log_Z']

    # Check that we can read in one population ok
    data_one = ocelot.cluster.SimulatedPopulations(path_to_one_simulated_population).data

    assert data_one.shape == (6116, 14)
    assert np.allclose(data_one.iloc[1500, 5], 3.7996)
    assert list(data_one.columns) == target_columns

    # Check that we can read in multiple
    data_all = ocelot.cluster.SimulatedPopulations(path_to_all_simulated_populations, search_pattern="*.dat").data

    assert data_all.shape == (2 * 6116, 14)
    assert np.allclose(data_all.iloc[6116 + 1500, 5], 3.7996)
    assert list(data_all.columns) == target_columns

    return data_one, data_all


def test_simulated_populations():
    """Given that ocelot.cluster.SimulatedPopulations reading works (see test_read_cmd_simulated_populations),
    we next test whether or not it can make some simulated populations!

    Data generated with:
        data_smol = data_mwsc.loc[clusters_to_plot_mwsc_table_numbers, :]
        data_smol['extinction_v'] = 3.1 * data_smol['E(B-V)']
        data_smol['v_int'] = 500
        data_smol['[Fe/H]'] = 0.
        data_smol['mass'] = 1e3
        new_data = pd.DataFrame()
        new_data[['ra', 'dec', 'l', 'b', 'distance', 'extinction_v', 'pmra', 'pmdec',
                  'age', 'Z', 'mass', 'radius_c', 'radius_t', 'v_int']] = data_smol[[
            'RAJ2000', 'DEJ2000', 'GLON', 'GLAT', 'd', 'extinction_v', 'pmRA', 'pmDE',
            'logt', '[Fe/H]', 'mass', 'rc', 'rt', 'v_int'
        ]]
        new_data.to_csv('test_clusters', index=False)

    """
    test_clusters = pd.read_csv(path_to_simulated_population_test_clusters)

    simulated_populations = ocelot.cluster.SimulatedPopulations(path_to_one_simulated_population)

    simulated_clusters = simulated_populations.get_clusters(test_clusters, concatenate=False, error_on_invalid_request=False)

    return simulated_populations, test_clusters, simulated_clusters


if __name__ == '__main__':
    print('uncomment something ya frikin jabroni')
    # gaia, rescaled = test_rescale_dataset()
    # cut = test_cut_dataset()
    # spar, dist = test_precalculate_nn_distances()
    # eps, ran = test_acg18()
    # test__summed_kth_nn_distribution_one_cluster()
    # test__find_sign_change_epsilons()
    # test__find_curve_absolute_maximum_epsilons()
    # test_field_model(show_figure=True)
    # one, all = test_read_cmd_simulated_populations()
    simpop, test_clusters, simcl = test_simulated_populations()
