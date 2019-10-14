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


# Run tests manually if the file is ran
if __name__ == '__main__':
    iso = test_read_cmd_isochrone()

    """
    # Some plotting code just for my own sanity check
    import matplotlib.pyplot as plt

    age = 6
    max_label = 3  # See CMD website for what stage of stellar evolution these correspond to

    hello = iso.drop(np.where(iso['logAge'] != age)[0], axis=0).reset_index()
    unique_z = np.unique(hello['Zini'])

    for a_z in unique_z:
        where_z_met = np.logical_and(np.where(hello['Zini'] == a_z, True, False),
                                     np.where(hello['label'] <= max_label, True, False))
        x = hello.loc[where_z_met, 'G_BP-RP']
        y = hello.loc[where_z_met, 'Gmag']
        plt.plot(x, y, '-', label=f'[M/H]={np.log10(a_z):.2}')

    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'Isochrones for max_label={max_label} and logAge={age}')
    plt.show()
    """
