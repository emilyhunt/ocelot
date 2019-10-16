"""A set of tests for use with the pytest module, covering ocelot.plot"""

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

import numpy as np

# Path towards the test isochrones
max_label = 0
path_to_test_isochrones = Path('./test_data/isochrones.dat')
#path_to_test_isochrones = Path('../../../data/isochrones/191015_isochrones_z-2_8_to_1_2/z_-2.8_1.2_1young_age.dat')


def test_isochrone_plotting(show_figure=False):
    """Tests the function ocelot.plot.isochrone."""
    my_isochrones = ocelot.isochrone.read_cmd_isochrone(path_to_test_isochrones, max_label=max_label)

    # Cut some stars for speed purposes
    stars_to_cut = np.asarray(my_isochrones['MH'] != 0.0).nonzero()[0]
    my_isochrones = my_isochrones.drop(stars_to_cut).reset_index(drop=True)

    # Define constants
    test_points_literature = np.asarray([[6.0], [6.5]])
    test_points_interpolated = np.asarray(np.expand_dims(np.linspace(6.0, 6.50, 4), axis=1))
    arguments = ['logAge']

    # Interpolate, so that we have something to use
    isochrones_for_fun_and_profit = ocelot.isochrone.IsochroneInterpolator(my_isochrones,
                                                                           parameters_as_arguments=['logAge'],
                                                                           parameters_to_infer=None,
                                                                           interpolation_type='LinearND')

    # Check that the plotting code runs
    ocelot.plot.isochrone(data_isochrone=my_isochrones,
                          literature_isochrones_to_plot=test_points_literature,
                          isochrone_interpolator=isochrones_for_fun_and_profit,
                          interpolated_isochrones_to_plot=test_points_interpolated,
                          isochrone_arguments=arguments,
                          literature_colors='same',
                          interpolated_colors='autumn',
                          show_figure=show_figure,
                          figure_size=[20, 20])

    return my_isochrones


# Run tests manually if the file is ran
if __name__ == '__main__':
    iso = test_isochrone_plotting(show_figure=True)
