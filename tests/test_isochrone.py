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

from pathlib import Path

# And now... everything else
import numpy as np

# Path towards the test isochrones
max_label = 7
path_to_test_isochrones = Path('./test_data/isochrones.dat')


def test_read_cmd_isochrone():
    """Tests the input-output functionality of the isochrone module."""
    my_isochrones = ocelot.isochrone.read_cmd_isochrone(path_to_test_isochrones, max_label=max_label)

    # Check that we've read in the right shape of file
    assert my_isochrones.shape == (2878, 15)

    # Test that the headers were read in correctly
    assert list(my_isochrones.keys()) == \
        ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe',
         'logg', 'label', 'mbolmag', 'Gmag', 'G_BPmag', 'G_RPmag', 'G_BP-RP']

    # Check that there aren't any hidden headers in there (CMD 3.3 hides them in some really annoying spots)
    rows_with_another_header = np.where(my_isochrones['Zini'] == '#')[0]
    assert rows_with_another_header.size == 0

    # Check that all the rows have the right max_label
    rows_with_a_bad_label = np.where(my_isochrones['label'] > max_label)[0]
    assert rows_with_a_bad_label.size == 0

    # Check some random values (by extension checking the typing too)
    assert my_isochrones.loc[0, 'Zini'] == 0.0048313
    assert my_isochrones.loc[1000, 'Gmag'] == 7.681
    assert my_isochrones.loc[2877, 'label'] == 3

    return my_isochrones


def test_isochrone_interpolation():
    """Tests the isochrone interpolation functionality of ocelot."""
    my_isochrones = ocelot.isochrone.read_cmd_isochrone(path_to_test_isochrones, max_label=max_label)

    # Cut some stars for speed purposes
    stars_to_cut = np.asarray(my_isochrones['MH'] != 0.0).nonzero()[0]
    my_isochrones = my_isochrones.drop(stars_to_cut).reset_index(drop=True)

    isochrones_for_fun_and_profit = ocelot.isochrone.IsochroneInterpolator(my_isochrones,
                                                                           parameters_as_arguments=['logAge'],
                                                                           parameters_to_infer=None,
                                                                           interpolation_type='LinearND')

    # Test the output, where logAge=6.25 is *not* sampled by the input and should hence give interesting results.
    test_points = np.asarray([[6.], [6.25], [6.5]])
    output_x, output_y = isochrones_for_fun_and_profit(test_points, resolution=100)

    # Check the shapes
    assert output_x.shape == (300,)
    assert output_y.shape == (300,)

    # Check that there aren't any nans or infs
    assert np.count_nonzero(np.isfinite(output_x)) == 300
    assert np.count_nonzero(np.isfinite(output_y)) == 300

    # Check some random numbers
    assert np.allclose(output_x[[0, 100, 200]], [3.551, 3.678, 3.805], rtol=0.0, atol=1e-8)
    assert np.allclose(output_y[[0, 100, 200]], [9.753, 10.0945, 10.436], rtol=0.0, atol=1e-8)

    return [isochrones_for_fun_and_profit, output_x, output_y]


# Run tests manually if the file is ran
if __name__ == '__main__':
    isochrones = test_read_cmd_isochrone()
    interpolator, interpolator_output_x, interpolator_output_y = test_isochrone_interpolation()

    """
    # Some plotting code just for my own sanity check
    import matplotlib.pyplot as plt

    good_stars = np.asarray(np.logical_and(isochrones['MH'] == 0.0,
                                           np.logical_or(isochrones['logAge'] == 6., isochrones['logAge'] == 6.5)))
    plt.plot(isochrones['G_BP-RP'].iloc[good_stars], isochrones['Gmag'].iloc[good_stars], 'k-', label='original')

    plt.plot(interpolator_output[0], interpolator_output[1], 'r.', label='interpolated')

    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'Isochrones: interpolated vs original')
    plt.show()
    """