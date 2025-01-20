"""A set of tests for use with the pytest module, covering ocelot.plot"""

from ocelot import plot
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import pytest

# Path towards the test isochrones
test_data_path = Path(__file__).parent / "test_data"

max_label = 0
path_to_test_isochrones = test_data_path / "isochrones/isochrones.dat"

# Path towards blanco 1 data and clustering produced by DBSCAN
path_to_blanco_1 = test_data_path / "blanco_1_gaia_dr2_gmag_18_cut.pickle"
path_to_blanco_1_cluster_labels = (
    test_data_path / "blanco_1_gaia_dr2_gmag_18_cut_LABELS.pickle"
)


def test_curve_normalisation():
    """Tests the curve normalisation function ocelot.plot.utilities.normalise_a_curve."""
    # Make a shitty little sine wave
    x_range = np.linspace(0, 10, num=100)
    y_range = np.sin(x_range)

    # Calculate some areas
    true_area = np.trapz(y_range, x=x_range)
    normalised_area = np.trapz(
        plot.utilities.normalise_a_curve(x_range, y_range, 2.5), x=x_range
    )
    unnormalised_area = np.trapz(
        plot.utilities.normalise_a_curve(x_range, y_range, 0.0), x=x_range
    )

    # Test that normalisation works to an arbitrary number
    assert np.allclose(2.5, normalised_area, rtol=0.0, atol=1e-8)

    # Test that normalisation can be turned off
    assert np.allclose(true_area, unnormalised_area, rtol=0.0, atol=1e-8)


def test_percentile_based_plot_limits():
    """Tests the function ocelot.plot.utilities.percentile_based_plot_limits, also
    testing the function ocelot.plot.utilities.good_points_plot_limits at the same time.
    """
    # Make some test data
    x = np.linspace(0, 100, num=101)
    y = x**2

    # -------------
    # Set some limits based on x, also test the padding
    x_percentiles = [0, 30]
    target_x = np.asarray([-1.5, 31.5])
    target_y = np.asarray([-45.0, 945.0])
    x_limits, y_limits = plot.utilities.percentile_based_plot_limits(
        x, y, x_percentiles=x_percentiles, range_padding=0.05
    )

    # Check x and y
    assert np.allclose(x_limits, target_x, rtol=0.0, atol=1e-8)
    assert np.allclose(y_limits, target_y, rtol=0.0, atol=1e-8)

    # -------------
    # Set some limits based on y
    y_percentiles = [50, 100]
    target_x = np.asarray([50.0, 100.0])
    target_y = np.asarray([2500.0, 10000.0])
    x_limits, y_limits = plot.utilities.percentile_based_plot_limits(
        x, y, y_percentiles=y_percentiles, range_padding=None
    )

    # Check x and y
    assert np.allclose(x_limits, target_x, rtol=0.0, atol=1e-8)
    assert np.allclose(y_limits, target_y, rtol=0.0, atol=1e-8)

    # -------------
    # Set some limits based on x and y at the same time
    x_percentiles = [0, 50]
    y_percentiles = [0, 10]
    target_x = np.asarray([0.0, 50.0])
    target_y = np.asarray([0.0, 100.0])
    x_limits, y_limits = plot.utilities.percentile_based_plot_limits(
        x,
        y,
        x_percentiles=x_percentiles,
        y_percentiles=y_percentiles,
        range_padding=None,
    )

    # Check x and y
    assert np.allclose(x_limits, target_x, rtol=0.0, atol=1e-8)
    assert np.allclose(y_limits, target_y, rtol=0.0, atol=1e-8)

    # -------------
    # Check that specifying nothing raises a value error
    with pytest.raises(
        ValueError,
        match="One or both of x_percentiles and y_percentiles must be specified.",
    ):
        plot.utilities.percentile_based_plot_limits(x, y)


def test_clustering_result_plotting(show_figure=False):
    """Integration test for the function ocelot.plot.clustering_result."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)

    with open(path_to_blanco_1_cluster_labels, "rb") as handle:
        labels = pickle.load(handle)

    # Grab the unique labels
    unique_labels = np.unique(labels)  # The first value is -1, which is noise points

    cluster_shading = 1 - np.clip(
        1.0
        * np.sqrt(
            (data_gaia["ra"] - np.mean(data_gaia["ra"])) ** 2
            + (data_gaia["dec"] - np.mean(data_gaia["dec"])) ** 2
        ),
        0.0,
        1.0,
    )

    # Plot time!
    fig, ax = plot.clustering_result(
        data_gaia,
        labels,
        unique_labels[1:],
        cluster_shading,
        show_figure=show_figure,
        figure_title="Blanco 1 should be clearly visible! \nCut at G=18, clustered by DBSCAN",
        pmra_plot_limits=[14, 24],
        pmdec_plot_limits=[-2, 8],
        cmd_plot_y_limits=[6, 18],
        cluster_marker_radius=(
            1.0,
            3.0,
            1.0,
        ),
    )

    # Friendship ended with excessive memory use,,, now "del" is my friend
    del data_gaia
    del labels


def test_location_plotting(show_figure=False):
    """Integration test for the function ocelot.plot.location."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)

    # Plot time!
    fig, ax = plot.location(
        data_gaia,
        show_figure=show_figure,
        figure_title="Blanco 1 should be clearly visible! \nCut at G=18",
        pmra_plot_limits=[14, 24],
        pmdec_plot_limits=[-2, 8],
    )

    # Friendship ended with excessive memory use,,, now "del" is my friend
    del data_gaia


def test_nearest_neighbour_plotting(show_figure=True):
    """Tests the functionality of ocelot.plot.nearest_neighbor_distances."""
    # Create some bullshit data that looks about right with a beta distribution
    distances = (np.sort(np.random.beta(4, 4, size=1000)) + 0.1).reshape(-1, 1)

    # Also make a thing to fit this with using an analytical beta distribution
    x_range = distances[::10, 0]
    y_range = np.linspace(0, 999, num=x_range.shape[0]) + 1

    fitting_func = {
        "x": x_range,
        "y": y_range,
        "style": "r-",
        "label": "fit",
        "differentiate": True,
    }

    line = {
        "x": np.asarray([0.5, 0.5]),
        "y": np.asarray([0.00001, 1000]),
        "style": "b--",
        "label": "nn = 0.5",
        "differentiate": False,
    }

    normalisation_constants = [1.0, 1.0, 0]

    plot.nearest_neighbor_distances(
        distances,
        number_of_derivatives=2,
        figure_size=[6, 8],
        show_figure=show_figure,
        figure_title="messing with a beta function",
        normalisation_constants=normalisation_constants,
        functions_to_overplot=[fitting_func, line],
        show_numerical_derivatives=False,
    )


def test_gaia_explorer():
    """Tests the Gaia explorer figure thing of fun"""
    # Firstly, we're gonna need a Blanco 1 to test with
    with open(path_to_blanco_1, "rb") as handle:
        data_gaia = pickle.load(handle)

    # Let's guess at where the cluster is
    cluster_location = dict(
        name="Blanco 1",
        ra=1.0,
        dec=-30.0,
        pmra=18.7,
        pmdec=2.9,
        parallax=4.25,
        ang_radius_t=2,
        pmra_error=1.0,
        pmdec_error=1.0,
        parallax_error=0.25,
    )

    # Let's also make a cheeky little dataframe of fake locations to test things
    catalogue = pd.DataFrame(
        dict(
            name=["Blanco 1", "not Blanco", "ocelots r cute", "emily", "lauren"],
            ra=[1.0, 1.5, 1.7, 0.5, 0.0],
            dec=[-30.0, -30.5, -29.5, -30.5, -30],
            pmra=[18.7, 22, 18, 16, 0],
            pmdec=[2.9, 5, 7, -3, 0],
            parallax=[4.25, 2, 3, 1, 0],
            source=["TCG+20", "bob", "bob", "me", "me"],
        )
    )

    # And plot it!
    # ocelot.plot.ion()
    gaia_explorer = plot.GaiaExplorer(
        data_gaia,
        cluster_location,
        debug=True,
        extra_catalogue_to_plot=catalogue,
        extra_catalogue_cmap={"TCG+20": "b", "bob": "c", "me": "m"},
        error_regions_multiplier=5.0,
    )
    gaia_explorer()
