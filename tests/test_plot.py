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

import pickle
from pathlib import Path

import numpy as np

# Path towards the test isochrones
max_label = 0
path_to_test_isochrones = Path('./test_data/isochrones.dat')
#path_to_test_isochrone = Path('../../../data/isochrones/191015_isochrones_z-2_8_to_1_2/z_-2.8_1.2_1young_age.dat')

# Path towards blanco 1 data and clustering produced by DBSCAN
path_to_blanco_1 = Path('./test_data/blanco_1_gaia_dr2_gmag_18_cut.pickle')
path_to_blanco_1_cluster_labels = Path('./test_data/blanco_1_gaia_dr2_gmag_18_cut_LABELS.pickle')


def test_isochrone_plotting(show_figure=False):
    """Integration test for the function ocelot.plot.isochrone."""
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
                          figure_size=[10, 10])

    return my_isochrones


def test_clustering_result_plotting(show_figure=False):
    """Integration test for the function ocelot.plot.clustering_result."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    with open(path_to_blanco_1_cluster_labels, 'rb') as handle:
        labels = pickle.load(handle)

    # Grab the unique labels
    unique_labels = np.unique(labels)  # The first value is -1, which is noise points

    # Plot time!
    fig, ax = ocelot.plot.clustering_result(
        data_gaia, labels, unique_labels[1:], show_figure=show_figure,
        figure_title="Blanco 1 should be clearly visible! \nCut at G=18, clustered by DBSCAN",
        pmra_plot_limits=[14, 24], pmdec_plot_limits=[-2, 8], cmd_plot_y_limits=[6, 18])

    # Friendship ended with excessive memory use,,, now "del" is my friend
    del data_gaia
    del labels

    return fig, ax


def test_location_plotting(show_figure=False):
    """Integration test for the function ocelot.plot.location."""
    # Read in data for Blanco 1
    with open(path_to_blanco_1, 'rb') as handle:
        data_gaia = pickle.load(handle)

    # Plot time!
    fig, ax = ocelot.plot.location(
        data_gaia, show_figure=show_figure, figure_title="Blanco 1 should be clearly visible! \nCut at G=18",
        pmra_plot_limits=[14, 24], pmdec_plot_limits=[-2, 8])

    # Friendship ended with excessive memory use,,, now "del" is my friend
    del data_gaia

    return fig, ax


# Run tests manually if the file is ran
if __name__ == '__main__':
    iso = test_isochrone_plotting(show_figure=True)
    clustering_result = test_clustering_result_plotting(show_figure=True)
    location = test_location_plotting(show_figure=True)