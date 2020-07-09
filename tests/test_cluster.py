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
import matplotlib.pyplot as plt
from scipy.sparse.csr import csr_matrix

path_to_blanco_1 = Path('./test_data/blanco_1_gaia_dr2_gmag_18_cut.pickle')
path_to_healpix_pixel = Path('./test_data/healpix_12237.csv')
path_to_healpix_pixels = Path('./test_data/healpix_pixel/')

path_to_one_simulated_population = Path('./test_data/simulated_populations/small/1.dat')
path_to_all_simulated_populations = Path('./test_data/simulated_populations/small')
path_to_big_simulated_population = Path('./test_data/simulated_populations/large/2.dat')

path_to_simulated_population_test_clusters = Path('./test_data/simulated_populations/test_clusters')


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


def _plot_for_recenter_dataset(data_gaia: pd.DataFrame, super_title: str = ''):
    """Cheeky plotting function for manual testing of the dataset recentering functionality."""
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    marker_radius = 1
    alpha = 0.3

    ax[0, 0].scatter(data_gaia['ra'], data_gaia['dec'], s=marker_radius**2, alpha=alpha)
    ax[0, 1].scatter(data_gaia['pmra'], data_gaia['pmdec'], s=marker_radius**2, alpha=alpha)
    ax[1, 0].scatter(data_gaia['lon'], data_gaia['lat'], s=marker_radius**2, alpha=alpha)
    ax[1, 1].scatter(data_gaia['pmlon'], data_gaia['pmlat'], s=marker_radius**2, alpha=alpha)

    ax[0, 0].set_title('ra vs dec')
    ax[0, 1].set_title('pmra vs pmdec')
    ax[1, 0].set_title('lon vs lat')
    ax[1, 1].set_title('pmlon vs pmlat')

    fig.suptitle(super_title)

    fig.show()
    plt.close(fig)


def test_recenter_dataset(show_figure=False):
    """Tests that position recentering in ocelot.cluster.recenter_dataset works as intended."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    center = [data_gaia['ra'].median(), data_gaia['dec'].median()]

    data_gaia = ocelot.cluster.recenter_dataset(data_gaia, center=center)

    if show_figure:
        _plot_for_recenter_dataset(data_gaia, super_title="test_recenter_dataset")

    # A quick check that the median values are about 0, 0 (won't be exact due to distortions)
    assert np.allclose(0.0, data_gaia['lon'].median(), rtol=0.0, atol=0.05)
    assert np.allclose(0.0, data_gaia['lat'].median(), rtol=0.0, atol=0.05)

    return data_gaia


def test_recenter_dataset_healpix(show_figure=False):
    """Tests that position recentering in ocelot.cluster.recenter_dataset works as intended, but for when using a
    healpix pixel."""
    # Read in data for the pixel
    data_gaia = pd.read_csv(path_to_healpix_pixel)

    data_gaia = ocelot.cluster.recenter_dataset(data_gaia, pixel_id=12237, rotate_frame=True)

    if show_figure:
        _plot_for_recenter_dataset(data_gaia, super_title="test_recenter_dataset_healpix")

    # A quick check that the median values are about 0, 0 (won't be exact due to distortions)
    assert np.allclose(0.0, data_gaia['lon'].median(), rtol=0.0, atol=0.05)
    assert np.allclose(0.0, data_gaia['lat'].median(), rtol=0.0, atol=0.05)

    return data_gaia


def test_rescale_dataset():
    """Tests the functionality of the dataset re-scaling of ocelot.cluster.rescale_dataset."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    # Re-scale the data with standard scaling and check that it has zero mean, unit variance
    data_rescaled, data_rescaled_2 = ocelot.cluster.rescale_dataset(data_gaia, data_gaia, scaling_type='standard',
                                                                    concatenate=False)
    data_rescaled_3 = ocelot.cluster.rescale_dataset(data_gaia, scaling_type='standard')

    # Check for zero mean, unit standard deviation
    means = np.mean(data_rescaled, axis=0)
    assert np.allclose(means, np.zeros(5), rtol=0.0, atol=1e-8)

    std_deviation = np.std(data_rescaled, axis=0)
    assert np.allclose(std_deviation, np.ones(5), rtol=0.0, atol=1e-8)

    # Check that the additional data frame was correctly re-scaled too - they should be identical
    assert np.allclose(data_rescaled, data_rescaled_2, rtol=1e-6, atol=1e-8)

    # And check that doing it just once has the same result too
    assert np.allclose(data_rescaled, data_rescaled_3, rtol=1e-6, atol=1e-8)

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


def test_find_largest_cluster():
    """Tests ocelot.cluster.partition.find_largest_cluster_in_pixel."""
    largest_cluster = ocelot.cluster.partition.find_largest_cluster_in_pixel([0], 5, 1)

    assert np.allclose(largest_cluster, 15.886934673445522, atol=1e-3, rtol=1e-3)

    return largest_cluster


def test_data_partition(show_figure=False):
    """Tests all elements of the data partition class, ocelot.cluster.DataPartition."""
    # Read in the data
    to_concatenate = []
    for a_file in path_to_healpix_pixels.glob('*.csv'):
        to_concatenate.append(pd.read_csv(a_file))

    data_gaia = pd.concat(to_concatenate, ignore_index=True)
    del to_concatenate
    
    # We want the data to have lon, lat values for our upcoming unit test use
    data_gaia = ocelot.cluster.recenter_dataset(data_gaia, pixel_id=12238)

    # Define what our partition will look like
    constraints = [
        [None, None, 0],
        [5, 7, 700],
        [6, 8, 3000],
    ]
    
    size_threshold = 4

    # Let's go!
    partitioner = ocelot.cluster.DataPartition(data_gaia,
                                               12238,
                                               constraints=constraints,
                                               final_distance=np.inf,
                                               parallax_sigma_threshold=2.,
                                               minimum_size=size_threshold,
                                               n_stars_per_component=[800, 500, 200],
                                               verbose=True)

    # Check the total number of partitions
    # assert partitioner.total_partitions == 11

    # Test that we can check whether or not something is internal to the partitions properly
    # First off, check a single star in the first partition (easy)
    partitioner.get_partition(0)
    assert partitioner.test_if_in_current_partition(0, 0, 20)

    # Check that getting of n_components works
    assert partitioner.get_n_components() == 11
    
    # Now, let's check some more values. The very last one is the only one with a bad parallax.
    lons = np.asarray([-0.608355,  0.991817,  1.991817, 0.581063, 2.3,  0.])
    lats = np.asarray([-2.737361, -0.958997, -0.958997, -2.1,     2.2, -2.])
    parallaxes = np.asarray([1.3, 1.2, 1.1, 1.0, 0.9, 1.7])

    partitioner.get_partition(8)
    target_array = np.asarray([True, True, False, True, False, False])
    result_array = partitioner.test_if_in_current_partition(lons, lats, parallaxes)

    assert np.all(target_array == result_array)

    # Run the plotting code (integration test alert!!!! Call the bad testing police on me why don't you)
    partitioner.plot_partition_bar_chart(figure_title='test of the partitioner', show_figure=show_figure,
                                         desired_size=size_threshold, base_n_stars_per_component=800)
    partitioner.plot_partitions(figure_title='testing of the partitioner', show_figure=show_figure,
                                cmd_plot_y_limits=[9, 16])

    return data_gaia, partitioner


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
        data_rescaled, distance_matrix, n_repeats=[1, 2], min_samples='all', return_last_random_distance=True,
        return_std_deviation=True)

    # Check that the correct shapes are returned
    assert distance_matrix.shape == random_distance_matrix.shape
    assert epsilons.shape == (10, 5)
    assert epsilons.columns.to_list() == ['min_samples', 'acg_1', 'acg_1_std', 'acg_2', 'acg_2_std']

    # Check the values against a target set that were correct at first implementation
    target_epsilons = np.asarray([[0.01767441, 0.01594943],
                                  [0.06125835, 0.06773323],
                                  [0.09002014, 0.09449874],
                                  [0.10619648, 0.10952761],
                                  [0.12325615, 0.12000926],
                                  [0.12975783, 0.13023932],
                                  [0.14481370, 0.14131321],
                                  [0.14834746, 0.14809204],
                                  [0.15089530, 0.15114417],
                                  [0.15616676, 0.15699153]])
    assert np.allclose(epsilons[['acg_1', 'acg_2']].values, target_epsilons, rtol=0.0, atol=1e-8)
    
    # Also check the standard deviations
    target_epsilons = np.asarray([[0., 0.00344996],
                                  [0., 0.01294977],
                                  [0., 0.00895719],
                                  [0., 0.00666226],
                                  [0., 0.00649378],
                                  [0., 0.00096298],
                                  [0., 0.00700098],
                                  [0., 0.00051084],
                                  [0., 0.00049773],
                                  [0., 0.00164952]])
    assert np.allclose(epsilons[['acg_1_std', 'acg_2_std']].values, target_epsilons, rtol=0.0, atol=1e-8)
    
    # And check the range of min_samples. It should go from 2 to 11 when provided with values of k from 0 to 10.
    assert np.all(epsilons['min_samples'].values == np.arange(10) + 2)

    # Check that we just get back one (correct) value if that's all we ask for
    np.random.seed(42)
    single_epsilon = ocelot.cluster.epsilon.acg18(
        data_rescaled, distance_matrix, n_repeats=2, min_samples=10, return_last_random_distance=False)
    single_epsilon = single_epsilon['acg_2'].values[0]

    assert type(single_epsilon) == float or np.float
    assert np.allclose(single_epsilon, 0.15144678787864096, rtol=0.0, atol=1e-8)

    # Throw some errors by being a fuckwit with min_samples
    # Wrong string
    with pytest.raises(ValueError, match="Incompatible number or string of min_samples specified."):
        ocelot.cluster.epsilon.acg18(
            data_rescaled, distance_matrix, n_repeats=2, min_samples='not the droid you were looking for', 
            return_last_random_distance=False)

    # Min_samples greater than the number of neighbours
    with pytest.raises(ValueError,
                       match="min_samples may not be larger than max_neighbors_to_calculate"):
        ocelot.cluster.epsilon.acg18(
            data_rescaled, distance_matrix, n_repeats=2, min_samples=100, return_last_random_distance=False)

    # Min_samples less than one
    with pytest.raises(ValueError,
                       match="min_samples may not be larger than max_neighbors_to_calculate"):
        ocelot.cluster.epsilon.acg18(
            data_rescaled, distance_matrix, n_repeats=2, min_samples=0, return_last_random_distance=False)

    return epsilons, random_distance_matrix


def test__summed_kth_nn_distribution_one_cluster():
    """Tests ocelot.cluster.epsilon_summed_kth_nn_distribution_one_cluster (and by extension, kth_nn_distribution)."""
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


def test_simulated_populations(plot_clusters=False):
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

    Todo: expand this unit test to be more than just a visual one

    """
    np.random.seed(42)
    test_clusters = pd.read_csv(path_to_simulated_population_test_clusters)

    simulated_populations = ocelot.cluster.SimulatedPopulations(path_to_big_simulated_population,
                                                                mass_tolerance=0.01)

    simulated_clusters = simulated_populations.get_clusters(test_clusters, concatenate=False,
                                                            error_on_invalid_request=False,)

    # Plot some clustery friends for the user, if requested
    if plot_clusters:

        # Plot all the clusters
        for number, a_cluster in enumerate(simulated_clusters):
            ocelot.plot.clustering_result(a_cluster, plot_std_limit=4., figure_title=f"test cluster {number}")

        # Grab Blanco 1 and correct for its RAs (which straddle 0)
        test_cluster = simulated_clusters[0]
        with open(Path('./test_data/blanco_1_gaia_dr2_gmag_18_cut_cluster_only.pickle'), 'rb') as handle:
            data_gaia = pickle.load(handle)
        data_gaia['ra'] = np.where(data_gaia['ra'] > 180, data_gaia['ra'] - 360, data_gaia['ra'])

        # Plotting time of the King profile - Blanco 1 vs a simulated version!
        plt.hist(np.sqrt((test_cluster['ra'] - np.mean(test_cluster['ra'])) ** 2
                         + (test_cluster['dec'] - np.mean(test_cluster['dec'])) ** 2),
                 bins='auto', density=True, color='k', alpha=0.5, label='generated', cumulative=False)
        plt.hist(np.sqrt(
            (data_gaia['ra'] - np.mean(data_gaia['ra'])) ** 2 + (data_gaia['dec'] - np.mean(data_gaia['dec'])) ** 2),
                 bins='auto', density=True, color='b', alpha=0.5, label='blanco 1', cumulative=False)
        plt.yscale('log')
        plt.legend()
        plt.xlabel('r')
        plt.ylabel('log normalised density')
        plt.show()

    return simulated_populations, test_clusters, simulated_clusters


def test__setup_cluster_parameter():
    """Tests ocelot.cluster.synthetic._setup_cluster_parameter, a function for handling cluster parameter stuff."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    # Constants
    parameter_name = 'ra'

    # Test identical parameters, for both int and float specified - both cases should return an array of floats
    target = np.ones(5, dtype=float) * 2.

    float_like = ocelot.cluster.synthetic._setup_cluster_parameter(data_gaia, 'ra', 2., 5)
    assert np.allclose(target, float_like)

    int_like = ocelot.cluster.synthetic._setup_cluster_parameter(data_gaia, 'ra', 2, 5)
    assert np.allclose(target, int_like)

    # Test sampling of parameters
    np.random.seed(42)
    target = np.asarray([0.75, 1., 0.5])

    draw_from = np.linspace(0, 1, num=5)
    array_like = ocelot.cluster.synthetic._setup_cluster_parameter(data_gaia, 'ra', draw_from, 3)

    assert np.allclose(target, array_like)

    # Test callable function use
    target = np.asarray([-3.98379464, 5.76513088])

    callable_like = ocelot.cluster.synthetic._setup_cluster_parameter(
        data_gaia, 'ra', ocelot.cluster.synthetic._c_position_limits_plus_minus_two, 2)

    assert np.allclose(target, callable_like)


def test__find_nearest_magnitude_star():
    """Tests ocelot.cluster.synthetic._find_nearest_magnitude_star, which basically just does some array maths"""
    real = np.linspace(0, 10, num=1000)
    synthetic = np.linspace(0, 10, num=6)
    np.random.seed(42)

    matches = ocelot.cluster.synthetic._find_nearest_magnitude_star(real, synthetic)

    # Check that we get the correct number of matches
    assert synthetic.shape == matches.shape

    # Check that the values returned are about the same
    assert np.allclose(synthetic, real[matches], atol=0.01)


def test_generate_synthetic_clusters(plot_clusters=True):
    """Tests ocelot.cluster.generate_synthetic_clusters, a fancy wrapper to the SimulatedPopulations class that also
    handles adding things like error."""
    # Read in data for Blanco 1 and make a simulated populations object
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    # Add some random errors onto the data because Emily was a doofus and didn't download Gaia data with flux errors
    data_gaia['phot_g_mean_flux_error'] = 4.
    data_gaia['phot_bp_mean_flux_error'] = 30.
    data_gaia['phot_rp_mean_flux_error'] = 30.

    simulated_populations = ocelot.cluster.SimulatedPopulations(path_to_big_simulated_population,
                                                                mass_tolerance=0.01)

    # Make some clusters in augmentation input_mode
    data_augmented = ocelot.cluster.generate_synthetic_clusters(
        simulated_populations,
        data_gaia,
        mode='clustering_augmentation',
        cluster_parameters_to_overwrite={'age': 9.5},
        shuffle=False,)

    data_augmented = ocelot.cluster.cut_dataset(data_augmented, parameter_cuts={'phot_g_mean_mag': [-np.inf, 18]})
    data_augmented['ra'] = np.where(data_augmented['ra'] > 180, data_augmented['ra'] - 360, data_augmented['ra'])

    # Make some clusters in generator input_mode
    import time
    start = time.time()
    data_generated = ocelot.cluster.generate_synthetic_clusters(
        simulated_populations,
        data_gaia,
        mode='generator',
        cluster_parameters_to_overwrite={'distance': ocelot.cluster.synthetic._c_random_cbj_distance,
                                         'age': np.linspace(6., 9.5, num=50),
                                         'v_int': np.random.normal(loc=500, scale=50, size=50),
                                         'mass': np.random.exponential(scale=200, size=50) + 50,
                                         'extinction_v': np.random.exponential(scale=1., size=20)},
        shuffle=True,
        concatenate=False,
        n_clusters=10)
    print(time.time() - start)

    data_generated = ocelot.cluster.cut_dataset(data_generated, parameter_cuts={'phot_g_mean_mag': [-np.inf, 21]})
    data_generated['ra'] = np.where(data_generated['ra'] > 180, data_generated['ra'] - 360, data_generated['ra'])

    if plot_clusters:
        ocelot.plot.clustering_result(data_augmented,
                                      data_augmented['cluster_label'],
                                      [0, 1],
                                      plot_std_limit=5.,
                                      figure_title='augmented clustering',
                                      cmd_plot_y_limits=[8, 18])

        ocelot.plot.clustering_result(data_generated,
                                      data_generated['cluster_label'],
                                      np.unique(data_generated['cluster_label']),
                                      plot_std_limit=1.5,
                                      figure_title='generated clustering',
                                      cmd_plot_y_limits=[8, 21])


if __name__ == '__main__':
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # cut = test_cut_dataset()
    # gaia, rescaled = test_rescale_dataset()
    # spar, dist = test_precalculate_nn_distances()
    # eps, ran = test_acg18()
    # test__summed_kth_nn_distribution_one_cluster()
    # test__find_sign_change_epsilons()
    # test__find_curve_absolute_maximum_epsilons()
    # test_field_model(show_figure=True)
    # one, all = test_read_cmd_simulated_populations()
    # simpop, test_clusters, simcl = test_simulated_populations(plot_clusters=False)
    # test__setup_cluster_parameter()
    # test__find_nearest_magnitude_star()
    # test_generate_synthetic_clusters(plot_clusters=True)
    # gaia = test_recenter_dataset(show_figure=True)
    # gaia = test_recenter_dataset_healpix(show_figure=True).
    # largest = test_find_largest_cluster()
    data, parter = test_data_partition(show_figure=True)
