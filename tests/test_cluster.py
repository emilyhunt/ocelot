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


if __name__ == '__main__':
    gaia, rescaled = test_rescale_dataset()
    cut = test_cut_dataset()
