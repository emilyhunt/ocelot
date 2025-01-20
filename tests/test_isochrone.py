"""A set of tests for use with the pytest module, covering ocelot.isochrone"""

from ocelot import isochrone
from pathlib import Path
import numpy as np
import pytest

# Path towards the test isochrones
test_data_path = Path(__file__).parent / "test_data"
max_label = 7
path_to_test_isochrone = test_data_path / "isochrones/isochrones.dat"
path_to_test_isochrones = test_data_path / "isochrones/"
list_of_paths_to_test_isochrones = [
    test_data_path / "isochrones/isochrones.dat",
    test_data_path / "isochrones/isochrones_2.dat",
]
path_to_simulated_population = test_data_path / "simulated_population.dat"


def test_read_parsec():
    """Tests the input-output functionality of the isochrone module."""
    my_isochrones = isochrone.read_parsec(path_to_test_isochrone, max_label=max_label)

    # Check that we've read in the right shape of file
    assert my_isochrones.shape == (2878, 15)

    # Test that the headers were read in correctly
    assert list(my_isochrones.keys()) == [
        "Zini",
        "MH",
        "logAge",
        "Mini",
        "int_IMF",
        "Mass",
        "logL",
        "logTe",
        "logg",
        "label",
        "mbolmag",
        "Gmag",
        "G_BPmag",
        "G_RPmag",
        "G_BP-RP",
    ]

    # Check that there aren't any hidden headers in there (CMD 3.3 hides them in some really annoying spots)
    rows_with_another_header = np.where(my_isochrones["Zini"] == "#")[0]
    assert rows_with_another_header.size == 0

    # Check that all the rows have the right max_label
    rows_with_a_bad_label = np.where(my_isochrones["label"] > max_label)[0]
    assert rows_with_a_bad_label.size == 0

    # Check some random values (by extension checking the typing too)
    assert my_isochrones.loc[0, "Zini"] == 0.0048313
    assert my_isochrones.loc[1000, "Gmag"] == 7.681
    assert my_isochrones.loc[2877, "label"] == 3


def test_read_parsec_multiple_isochrones():
    """Specifically tests the ability of ocelot.isochrone.read_parsec to read multiple files."""
    # Read in isochrones two ways
    isochrones_read_as_directory = isochrone.read_parsec(
        path_to_test_isochrones, max_label=max_label
    )
    isochrones_read_as_list = isochrone.read_parsec(
        list_of_paths_to_test_isochrones, max_label=max_label
    )

    # Check their shapes
    assert isochrones_read_as_directory.shape == (5756, 15)
    assert isochrones_read_as_list.shape == (5756, 15)

    # Check two values near to the end - if indexing went wrong, these will have been scrambled
    assert isochrones_read_as_directory.loc[5000, "Gmag"] == -0.866
    assert isochrones_read_as_directory.loc[5001, "Gmag"] == -0.908

    assert isochrones_read_as_list.loc[5000, "Gmag"] == -0.866
    assert isochrones_read_as_list.loc[5001, "Gmag"] == -0.908


def test_isochrone_interpolation():
    """Tests the isochrone interpolation functionality of ocelot."""
    my_isochrones = isochrone.read_parsec(path_to_test_isochrone, max_label=max_label)

    # Cut some stars for speed purposes
    stars_to_cut = np.asarray(my_isochrones["MH"] != 0.0).nonzero()[0]
    my_isochrones = my_isochrones.drop(stars_to_cut).reset_index(drop=True)

    isochrones_for_fun_and_profit = isochrone.IsochroneInterpolator(
        my_isochrones, parameters_as_arguments=["logAge"], interpolation_type="LinearND"
    )

    # Test the output, where logAge=6.25 is *not* sampled by the input and should hence give interesting results.
    test_points = np.asarray([[6.0], [6.25], [6.5]])
    output_x, output_y = isochrones_for_fun_and_profit(test_points, resolution=100)

    # Check the shapes
    assert output_x.shape == (300,)
    assert output_y.shape == (300,)

    # Check that there aren't any nans or infs
    assert np.count_nonzero(np.isfinite(output_x)) == 300
    assert np.count_nonzero(np.isfinite(output_y)) == 300

    # Check some random numbers
    assert np.allclose(
        output_x[[0, 100, 200]], [3.551, 3.678, 3.805], rtol=0.0, atol=1e-8
    )
    assert np.allclose(
        output_y[[0, 100, 200]], [9.753, 10.0945, 10.436], rtol=0.0, atol=1e-8
    )


def test_find_nearest_point():
    """Tests a small function that attempts to find the nearest point on a line to a set of data."""
    # Setup some input data
    y = np.repeat(np.arange(5), 2).reshape(5, 2)  # 5 points spaced equally along y=x
    x = (y**2)[
        :3
    ]  # The squaring makes it so the first and second points match, but then the last matches to y[4]
    desired_result = np.array([0, 1, 4])
    desired_distances = np.array([[0, 0], [0, 0], [0, 0]])

    # Call the function, both with and without raw distances
    result = isochrone.interpolate.find_nearest_point(x, y)
    result_distances, distances = isochrone.interpolate.find_nearest_point(
        x, y, return_raw_distances=True
    )

    # Test that result is right
    np.testing.assert_array_equal(result, desired_result)

    # Test that result is right when distances are returned
    np.testing.assert_array_equal(result_distances, desired_result)

    # Test that the distances are correct
    np.testing.assert_array_equal(distances, desired_distances)

    # Test that giving it an array with the wrong amount of points raises an error
    with pytest.raises(ValueError, match="Input arrays must be two dimensional."):
        isochrone.interpolate.find_nearest_point(x[0], y)

    # Test that giving an array with the wrong number of features raises an error
    with pytest.raises(
        ValueError,
        match="Number of features mismatch between points_to_match and points_on_line.",
    ):
        isochrone.interpolate.find_nearest_point(x[:, 0:1], y)


def test_proximity_to_line_sort():
    """Tests the proximity to line sorting (which is mostly just a wrapper to
    ocelot.isochrone.interpolate.find_nearest_point)."""
    # Setup some input data
    y = np.repeat(np.arange(5), 2).reshape(5, 2)  # 5 points spaced equally along y=x
    x = np.array([[0, 0], [1.1, 1.1], [0.9, 0.8], [4.5, 3.8], [4.0, 3.7]])
    desired_result = np.array([0, 2, 1, 4, 3])

    # Run the function
    result = isochrone.interpolate.proximity_to_line_sort(x, y)

    # Test that result is right
    np.testing.assert_array_equal(result, desired_result)
