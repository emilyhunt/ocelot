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


def test_read_cmd_isochrone():
    """Tests the input-output functionality of the isochrone module."""
    path_to_test_isochrones = Path('../../data/isochrones/191008_isochrones_wide_logz_and_logt.dat')
    my_isochrones = ocelot.isochrone.read_cmd_isochrone(path_to_test_isochrones)

    # Check that we've read in the right shape of file
    assert my_isochrones.shape == (95919, 28)

    # Test that the headers were read in correctly
    assert list(my_isochrones.keys()) == \
        ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe',
         'logg', 'label', 'McoreTP', 'C_O', 'period0', 'period1', 'pmode',
         'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag',
         'Gmag', 'G_BPmag', 'G_RPmag']

    # Check that there aren't any hidden rows in there (CMD 3.3 hides them in some spots
    rows_with_another_header = np.where(my_isochrones['Zini'] == '#')[0]
    assert rows_with_another_header.size == 0

    # Check some random values
    assert my_isochrones.loc[0, 'Zini'] == 0.00015547
    assert my_isochrones.loc[10000, 'Gmag'] == 6.197
    assert my_isochrones.loc[94000, 'Mini'] == 1.6531767845


# Run tests manually if the file is ran
if __name__ == '__main__':
    test_read_cmd_isochrone()
