"""A set of tests for use with the pytest module, covering ocelot.isochrone"""

import ocelot.cluster
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

test_data_path = Path(__file__).parent / "test_data"

path_to_blanco_1 = test_data_path / "blanco_1_gaia_dr2_gmag_18_cut.pickle"
path_to_healpix_pixel = test_data_path / "healpix_12237.csv"
path_to_healpix_pixels = test_data_path / "healpix_pixel"

path_to_one_simulated_population = test_data_path / "simulated_populations/small/1.dat"
path_to_all_simulated_populations = test_data_path / "simulated_populations/small"
path_to_big_simulated_population = test_data_path / "simulated_populations/large/2.dat"

path_to_simulated_population_test_clusters = (
    test_data_path / "simulated_populations/test_clusters"
)


def test_cut_dataset():
    """Tests the functionality of dataset cutting at ocelot.cluster.cut_dataset."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)

    # Define some chonky cuts to make
    cuts = {
        "phot_g_mean_mag": [6, 16],
        "r_est": [100, np.inf],
        "parallax_over_error": [10, np.inf],
    }

    data_cut = ocelot.cluster.cut_dataset(data_gaia, parameter_cuts=cuts)

    # Check that the correct number of cuts were performed
    assert data_cut.shape == (5039, 28)

    # Check that no values fall outside of the cut ranges for gmag (easy)
    assert np.invert(
        np.any(
            np.logical_or(
                data_cut["phot_g_mean_mag"] < cuts["phot_g_mean_mag"][0],
                data_cut["phot_g_mean_mag"] > cuts["phot_g_mean_mag"][1],
            )
        )
    )

    # Check that no values fall outside of the cut ranges for r_est (could indicate fucking up with np.inf)
    assert np.invert(
        np.any(
            np.logical_or(
                data_cut["r_est"] < cuts["r_est"][0],
                data_cut["r_est"] > cuts["r_est"][1],
            )
        )
    )


def _plot_for_recenter_dataset(data_gaia: pd.DataFrame, super_title: str = ""):
    """Cheeky plotting function for manual testing of the dataset recentering functionality."""
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    marker_radius = 1
    alpha = 0.3

    ax[0, 0].scatter(data_gaia["ra"], data_gaia["dec"], s=marker_radius**2, alpha=alpha)
    ax[0, 1].scatter(
        data_gaia["pmra"], data_gaia["pmdec"], s=marker_radius**2, alpha=alpha
    )
    ax[1, 0].scatter(
        data_gaia["lon"], data_gaia["lat"], s=marker_radius**2, alpha=alpha
    )
    ax[1, 1].scatter(
        data_gaia["pmlon"], data_gaia["pmlat"], s=marker_radius**2, alpha=alpha
    )

    ax[0, 0].set_title("ra vs dec")
    ax[0, 1].set_title("pmra vs pmdec")
    ax[1, 0].set_title("lon vs lat")
    ax[1, 1].set_title("pmlon vs pmlat")

    fig.suptitle(super_title)

    fig.show()
    plt.close(fig)


def test_recenter_dataset(show_figure=False):
    """Tests that position recentering in ocelot.cluster.recenter_dataset works as intended."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)

    center = [data_gaia["ra"].median(), data_gaia["dec"].median()]

    data_gaia = ocelot.cluster.recenter_dataset(data_gaia, center=center)

    if show_figure:
        _plot_for_recenter_dataset(data_gaia, super_title="test_recenter_dataset")

    # A quick check that the median values are about 0, 0 (won't be exact due to distortions)
    assert np.allclose(0.0, data_gaia["lon"].median(), rtol=0.0, atol=0.05)
    assert np.allclose(0.0, data_gaia["lat"].median(), rtol=0.0, atol=0.05)


def test_recenter_dataset_healpix(show_figure=False):
    """Tests that position recentering in ocelot.cluster.recenter_dataset works as intended, but for when using a
    healpix pixel."""
    # Read in data for the pixel
    data_gaia = pd.read_csv(path_to_healpix_pixel)

    data_gaia = ocelot.cluster.recenter_dataset(
        data_gaia, pixel_id=12237, rotate_frame=True
    )

    if show_figure:
        _plot_for_recenter_dataset(
            data_gaia, super_title="test_recenter_dataset_healpix"
        )

    # A quick check that the median values are about 0, 0 (won't be exact due to distortions)
    assert np.allclose(0.0, data_gaia["lon"].median(), rtol=0.0, atol=0.05)
    assert np.allclose(0.0, data_gaia["lat"].median(), rtol=0.0, atol=0.05)


def test_rescale_dataset():
    """Tests the functionality of the dataset re-scaling of ocelot.cluster.rescale_dataset."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)

    # Re-scale the data with standard scaling and check that it has zero mean, unit variance
    data_rescaled, data_rescaled_2 = ocelot.cluster.rescale_dataset(
        data_gaia, data_gaia, scaling_type="standard", concatenate=False
    )
    data_rescaled_3 = ocelot.cluster.rescale_dataset(data_gaia, scaling_type="standard")

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
        ocelot.cluster.rescale_dataset(
            data_gaia, scaling_type="an unsupported type of scaling, I guess"
        )

    # Check that a cheeky nan value raises an error
    data_gaia.loc[100, "ra"] = np.nan
    with pytest.raises(
        ValueError,
        match="At least one value in data_gaia is not finite! Unable to rescale the data.",
    ):
        ocelot.cluster.rescale_dataset(data_gaia, scaling_type="robust")


def test_precalculate_nn_distances():
    """Tests ocelot.cluster.precalculate_nn_distances()"""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)

    # Re-scale the data with standard scaling and check that it has zero mean, unit variance
    data_rescaled = ocelot.cluster.rescale_dataset(data_gaia, scaling_type="robust")

    # Calculate some nearest neighbor distances baby!
    sparse_matrix, distance_matrix = ocelot.cluster.precalculate_nn_distances(
        data_rescaled,
        n_neighbors=10,
        return_sparse_matrix=True,
        return_knn_distance_array=True,
    )

    # Check that specifying no return type throws an error
    with pytest.raises(
        ValueError,
        match="Nothing was specified for return. That's probably not intentional!",
    ):
        ocelot.cluster.precalculate_nn_distances(
            data_rescaled,
            n_neighbors=10,
            return_sparse_matrix=False,
            return_knn_distance_array=False,
        )

    # Check that the sparse matrix is, in fact, sparse
    assert isinstance(sparse_matrix, csr_matrix)

    # Check that the correct shapes are returned
    assert distance_matrix.shape == (14785, 10)
    assert sparse_matrix.shape == (14785, 14785)


def test_acg18():
    """Tests the OCELOT implementation of the Alfred Castro-Ginard+18 method of determining an optimum value for DBSCAN."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)

    # Define some chonky cuts to make to make this test faster
    cuts = {
        "phot_g_mean_mag": [-np.inf, 16],
        "r_est": [200, 300],
        "parallax_over_error": [5, np.inf],
    }
    cuts = {}

    data_cut = ocelot.cluster.cut_dataset(data_gaia, parameter_cuts=cuts)

    # Re-scale the data with standard scaling and check that it has zero mean, unit variance
    data_rescaled = ocelot.cluster.rescale_dataset(data_cut, scaling_type="robust")

    # Calculate some nearest neighbor distances
    distance_matrix = ocelot.cluster.precalculate_nn_distances(
        data_rescaled,
        n_neighbors=10,
        return_sparse_matrix=False,
        return_knn_distance_array=True,
    )

    # Epsilon time
    np.random.seed(42)
    epsilons, random_distance_matrix = ocelot.cluster.epsilon.castro_ginard(
        data_rescaled,
        distance_matrix,
        n_repeats=[1, 2],
        min_samples="all",
        return_last_random_distance=True,
        return_std_deviation=True,
    )

    # Check that the correct shapes are returned
    assert distance_matrix.shape == random_distance_matrix.shape
    assert epsilons.shape == (10, 5)
    assert epsilons.columns.to_list() == [
        "min_samples",
        "acg_1",
        "acg_1_std",
        "acg_2",
        "acg_2_std",
    ]

    # Check the values against a target set that were correct at first implementation
    target_epsilons = np.asarray(
        [
            [0.01767441, 0.01594943],
            [0.06125835, 0.06773323],
            [0.09002014, 0.09449874],
            [0.10619648, 0.10952761],
            [0.12325615, 0.12000926],
            [0.12975783, 0.13023932],
            [0.14481370, 0.14131321],
            [0.14834746, 0.14809204],
            [0.15089530, 0.15114417],
            [0.15616676, 0.15699153],
        ]
    )
    assert np.allclose(
        epsilons[["acg_1", "acg_2"]].values, target_epsilons, rtol=0.0, atol=1e-8
    )

    # Also check the standard deviations
    target_epsilons = np.asarray(
        [
            [0.0, 0.00344996],
            [0.0, 0.01294977],
            [0.0, 0.00895719],
            [0.0, 0.00666226],
            [0.0, 0.00649378],
            [0.0, 0.00096298],
            [0.0, 0.00700098],
            [0.0, 0.00051084],
            [0.0, 0.00049773],
            [0.0, 0.00164952],
        ]
    )
    assert np.allclose(
        epsilons[["acg_1_std", "acg_2_std"]].values,
        target_epsilons,
        rtol=0.0,
        atol=1e-8,
    )

    # And check the range of min_samples. It should go from 2 to 11 when provided with values of k from 0 to 10.
    assert np.all(epsilons["min_samples"].values == np.arange(10) + 2)

    # Check that we just get back one (correct) value if that's all we ask for
    np.random.seed(42)
    single_epsilon = ocelot.cluster.epsilon.castro_ginard(
        data_rescaled,
        distance_matrix,
        n_repeats=2,
        min_samples=10,
        return_last_random_distance=False,
    )
    single_epsilon = single_epsilon["acg_2"].values[0]

    assert isinstance(single_epsilon, float) or isinstance(single_epsilon, np.float)
    assert np.allclose(single_epsilon, 0.15144678787864096, rtol=0.0, atol=1e-8)

    # Throw some errors by being a fuckwit with min_samples
    # Wrong string
    with pytest.raises(
        ValueError, match="Incompatible number or string of min_samples specified"
    ):
        ocelot.cluster.epsilon.castro_ginard(
            data_rescaled,
            distance_matrix,
            n_repeats=2,
            min_samples="not the droid you were looking for",
            return_last_random_distance=False,
        )

    # Min_samples greater than the number of neighbours
    with pytest.raises(
        ValueError,
        match="min_samples may not be larger than max_neighbors_to_calculate",
    ):
        ocelot.cluster.epsilon.castro_ginard(
            data_rescaled,
            distance_matrix,
            n_repeats=2,
            min_samples=100,
            return_last_random_distance=False,
        )

    # Min_samples less than one
    with pytest.raises(
        ValueError,
        match="min_samples may not be larger than max_neighbors_to_calculate",
    ):
        ocelot.cluster.epsilon.castro_ginard(
            data_rescaled,
            distance_matrix,
            n_repeats=2,
            min_samples=0,
            return_last_random_distance=False,
        )


def test__summed_kth_nn_distribution_one_cluster():
    """Tests ocelot.cluster.epsilon_summed_kth_nn_distribution_one_cluster (and by extension, kth_nn_distribution)."""
    # Set some parameters to play with
    field_constant = 0.3
    field_dimension = 5
    cluster_constant = 0.05
    cluster_dimension = 3
    cluster_fraction = 0.01
    parameters = np.asarray(
        [
            field_constant,
            field_dimension,
            cluster_constant,
            cluster_dimension,
            cluster_fraction,
        ]
    )

    # Calculate a field
    y_fields = ocelot.cluster.epsilon._summed_kth_nn_distribution_one_cluster(
        parameters, 10, np.linspace(0, 1, num=50), minimisation_mode=False
    )

    # Check the shape
    assert y_fields.shape == (3, 50)

    # Check that we get np.inf on the first value, showing that my log comprehension works
    assert np.allclose(
        y_fields[:, 0], np.asarray([-np.inf, -np.inf, -np.inf]), rtol=0.0, atol=1e-8
    )

    # Check that the mean of everything else is right, and by extension it's... probably right
    assert np.allclose(
        np.mean(y_fields[:, 1:]), -1.4524090935710814, rtol=0.0, atol=1e-8
    )


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
    epsilon_values, all_sign_changes = (
        ocelot.cluster.epsilon._find_sign_change_epsilons(
            x, y, return_all_sign_changes=True
        )
    )

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
    with open(path_to_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)

    # Re-scale the data with standard scaling and check that it has zero mean, unit variance
    data_rescaled = ocelot.cluster.rescale_dataset(data_gaia, scaling_type="robust")

    # Calculate some nearest neighbor distances baby!
    distance_matrix = ocelot.cluster.precalculate_nn_distances(
        data_rescaled,
        n_neighbors=10,
        return_sparse_matrix=False,
        return_knn_distance_array=True,
    )

    # Specify some plot options
    plot_options = {
        "number_of_derivatives": 2,
        "figure_size": (6, 8),
        "show_figure": show_figure,
        "figure_title": "Unit test of ocelot.cluster.epsilon.field_model",
    }

    # See what the field model fit thinks of this
    success, epsilon_values, parameters, n_cluster_members = (
        ocelot.cluster.epsilon.field_model(
            distance_matrix,
            min_samples=10,
            min_cluster_size=1,
            make_diagnostic_plot=False,
            **plot_options,
        )
    )

    # Test that the results are good
    assert success is True

    target_epsilon = [0.13765954, 0.1809968, 0.22433406, 0.25747432, 0.28968726]
    assert np.allclose(epsilon_values, target_epsilon, rtol=1e-5, atol=1e-8)

    target_parameters = [0.26971331, 9.40067326, 0.08953837, 3.36263488, 0.02072574]
    assert np.allclose(parameters, target_parameters, rtol=1e-5, atol=1e-8)

    assert n_cluster_members == 291
