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
# Todo: we should have a small isochrone table test packaged with the module instead, preferably in the tests folder.
max_label = 7
path_to_test_isochrones = Path('../../../data/isochrones/191008_isochrones_wide_logz_and_logt.dat')


def test_read_cmd_isochrone():
    """Tests the input-output functionality of the isochrone module."""
    my_isochrones = ocelot.isochrone.read_cmd_isochrone(path_to_test_isochrones, max_label=max_label)

    # Check that we've read in the right shape of file
    assert my_isochrones.shape == (72116, 30)

    # Test that the headers were read in correctly
    assert list(my_isochrones.keys()) == \
        ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe',
         'logg', 'label', 'McoreTP', 'C_O', 'period0', 'period1', 'pmode',
         'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag',
         'Gmag', 'G_BPmag', 'G_RPmag', 'G_BP-RP', 'logZini']

    # Check that there aren't any hidden rows in there (CMD 3.3 hides them in some spots
    rows_with_another_header = np.where(my_isochrones['Zini'] == '#')[0]
    assert rows_with_another_header.size == 0

    # Check that all the rows have the right max_label
    rows_with_a_bad_label = np.where(my_isochrones['label'] > max_label)[0]
    assert rows_with_a_bad_label.size == 0

    # Check some random values
    assert my_isochrones.loc[0, 'Zini'] == 0.00015547
    assert my_isochrones.loc[10000, 'Gmag'] == 9.308
    assert my_isochrones.loc[70000, 'Mini'] == 2.1509203911

    return my_isochrones


def test_isochrone_interpolation():
    """Tests the isochrone interpolation functionality of ocelot."""
    my_isochrones = ocelot.isochrone.read_cmd_isochrone(path_to_test_isochrones, max_label=max_label)

    stars_to_cut = np.asarray(my_isochrones['Zini'] != 0.0094713).nonzero()[0]

    my_isochrones = my_isochrones.drop(stars_to_cut).reset_index(drop=True)

    isochrones_for_fun_and_profit = ocelot.isochrone.IsochroneInterpolator(my_isochrones,
                                                                           parameters_as_arguments=['logAge'],
                                                                           parameters_to_infer=None,
                                                                           interpolation_type='LinearND')

    test_points = np.asarray([[6.]])

    isochrone_output = isochrones_for_fun_and_profit(test_points)

    return [isochrones_for_fun_and_profit, isochrone_output]


# Run tests manually if the file is ran
if __name__ == '__main__':
    isochrones = test_read_cmd_isochrone()

    interpolator, interpolator_output = test_isochrone_interpolation()

    # Some plotting code just for my own sanity check
    import matplotlib.pyplot as plt

    good_stars = np.asarray(np.logical_and(isochrones['Zini'] == 0.0094713,
                                           np.logical_or(isochrones['logAge'] == 6., isochrones['logAge'] == 6.)))
    plt.plot(isochrones['G_BP-RP'].iloc[good_stars], isochrones['Gmag'].iloc[good_stars], 'k-', label='original')

    plt.plot(interpolator_output[0], interpolator_output[1], 'r-', label='interpolated')

    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'Isochrones: interpolated vs original')
    plt.show()
