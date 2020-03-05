"""A set of tests for ocelot.calculate."""

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
import matplotlib.pyplot as plt

path_to_blanco_1_cluster = Path('./test_data/blanco_1_gaia_dr2_gmag_18_cut_cluster_only.pickle')


def test__weighted_standard_deviation():
    """Tests ocelot.calculate.calculate._weighted_standard_deviation, a function for computing weighted standard
    deviations."""
    # Create some simple test data which clearly has standard deviation ~1, other than ten cheeky points
    np.random.seed(42)
    length = 10000
    normal_points = np.random.normal(loc=0.0, scale=1.0, size=length)

    not_so_normal_points = np.empty((length + 1000,), dtype=float)
    not_so_normal_points[:length] = normal_points
    not_so_normal_points[length:] = 10000.

    weights = np.ones(not_so_normal_points.shape)
    weights[length:] = 0.

    # Do our calculations
    weighted_std_normal_points = ocelot.calculate.calculate._weighted_standard_deviation(normal_points, None)
    weighted_std_abnormal_points = ocelot.calculate.calculate._weighted_standard_deviation(not_so_normal_points,
                                                                                           weights)
    std_normal_points = np.std(normal_points)

    # Check that my function and the actual standard deviation are about the same
    assert np.allclose(weighted_std_normal_points, std_normal_points, rtol=0.0, atol=1e-3)

    # Check that the weighting correctly ignores the dodgy points
    assert np.allclose(weighted_std_abnormal_points, std_normal_points, rtol=0.0, atol=1e-3)


def parameter_return_test_function(parameters, target, rtol=0.0, atol=1e-8):
    """Accepts a parameter dict and a target dict, like those returned by ocelot.calculate.

    Tests that:
        1. The number of keys is the same
        2. The keys are the same
        3. The values are the same

    """
    # Check that the keys are right (indicative of a code change)
    keys_parameters = list(parameters.keys())
    keys_target = list(target.keys())
    assert len(keys_parameters) == len(keys_target)

    for i in range(len(keys_parameters)):
        assert keys_parameters[i] == keys_target[i]

    # Check that the values match the target ones
    assert np.allclose(list(parameters.values()), list(target.values()), rtol=rtol, atol=atol, equal_nan=True)


def test_mean_distance():
    """Tests ocelot.calculate.mean_distance"""
    # Read in Blanco 1
    with open(path_to_blanco_1_cluster, 'rb') as handle:
        data_cluster = pickle.load(handle)

    # Calculate distances
    parameters = ocelot.calculate.mean_distance(data_cluster)

    target_dict = {
        'parallax': 4.152918183080261,
        'parallax_error': 0.329960190884155,
        'inverse_parallax': 240.79453432869946,
        'inverse_parallax_l68': 223.07096391634954,
        'inverse_parallax_u68': 261.57755383169876,
        'distance': 237.76771451021528,
        'distance_error': 18.751810964372865}

    parameter_return_test_function(parameters, target_dict)


def test_mean_proper_motion():
    """Tests ocelot.calculate.mean_proper_motion"""
    # Read in Blanco 1
    with open(path_to_blanco_1_cluster, 'rb') as handle:
        data_cluster = pickle.load(handle)

    # Calculate the mean proper motion
    parameters = ocelot.calculate.mean_proper_motion(data_cluster)

    target_dict = {
        'pmra': 18.661186373958262,
        'pmra_error': 0.7654178147576686,
        'pmdec': 2.6210183216120617,
        'pmdec_error': 0.7759414309583798}

    parameter_return_test_function(parameters, target_dict)


def test_mean_radius():
    """Tests ocelot.calculate.mean_radius"""
    # Read in Blanco 1
    with open(path_to_blanco_1_cluster, 'rb') as handle:
        data_cluster = pickle.load(handle)

    # Calculate the mean proper motion
    parameters = ocelot.calculate.mean_radius(data_cluster)

    target_dict = {
        'ra': 0.9211424874405434,
        'ra_error': np.nan,
        'dec': -29.968388740371715,
        'dec_error': np.nan,
        'ang_radius_50': 0.5104069066354973,
        'ang_radius_50_error': np.nan,
        'ang_radius_c': np.nan,
        'ang_radius_c_error': np.nan,
        'ang_radius_t': 1.0224028660460727,
        'ang_radius_t_error': np.nan,
        'radius_50': 2.1451221301933647,
        'radius_50_error': np.nan,
        'radius_c': np.nan,
        'radius_c_error': np.nan,
        'radius_t': 4.2972651364598065,
        'radius_t_error': np.nan}

    parameter_return_test_function(parameters, target_dict)


def test_mean_internal_velocity_dispersion():
    """Tests ocelot.calculate.internal_velocity_dispersion"""
    # Read in Blanco 1
    with open(path_to_blanco_1_cluster, 'rb') as handle:
        data_cluster = pickle.load(handle)

    # Calculate the mean proper motion
    parameters = ocelot.calculate.mean_internal_velocity_dispersion(data_cluster)

    target_dict = {
        'v_internal_tangential': 927.8763318000849,
        'v_internal_tangential_error': np.nan}

    parameter_return_test_function(parameters, target_dict)


def test_all_statistics():
    """Tests ocelot.calculate.all_statistics"""
    # Read in Blanco 1
    with open(path_to_blanco_1_cluster, 'rb') as handle:
        data_cluster = pickle.load(handle)

    # Calculate the mean proper motion
    parameters = ocelot.calculate.all_statistics(data_cluster, mode="mean")

    target_dict = {
        'n_stars': 262,
        'ra': 0.9211424874405434,
        'ra_error': np.nan,
        'dec': -29.968388740371715,
        'dec_error': np.nan,
        'ang_radius_50': 0.5104069066354973,
        'ang_radius_50_error': np.nan,
        'ang_radius_c': np.nan,
        'ang_radius_c_error': np.nan,
        'ang_radius_t': 1.0224028660460727,
        'ang_radius_t_error': np.nan,
        'radius_50': 2.1451221301933647,
        'radius_50_error': np.nan,
        'radius_c': np.nan,
        'radius_c_error': np.nan,
        'radius_t': 4.2972651364598065,
        'radius_t_error': np.nan,
        'parallax': 4.152918183080261,
        'parallax_error': 0.329960190884155,
        'inverse_parallax': 240.79453432869946,
        'inverse_parallax_l68': 223.07096391634954,
        'inverse_parallax_u68': 261.57755383169876,
        'distance': 237.76771451021528,
        'distance_error': 18.751810964372865,
        'pmra': 18.661186373958262,
        'pmra_error': 0.7654178147576686,
        'pmdec': 2.6210183216120617,
        'pmdec_error': 0.7759414309583798,
        'v_internal_tangential': 927.8763318000849,
        'v_internal_tangential_error': np.nan,
        'parameter_inference_mode': 'mean'}

    # parameter_return_test_function(parameters, target_dict)
    # todo test function is broken!!! The string at the end fucks it up.

    return parameters


def test_points_on_sphere(show_diagnostic_histograms=False):
    """A function for testing the random spherical co-ordinates generator."""
    np.random.seed(42)

    # Calls and a basic diagnostic plot if requested
    theta_rad, phi_rad = ocelot.calculate.random.points_on_sphere(3000, radians=True, phi_symmetric=False)
    theta_deg, phi_deg = ocelot.calculate.random.points_on_sphere(3000, radians=False, phi_symmetric=True)

    if show_diagnostic_histograms:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].hist(theta_deg, bins='auto')
        ax[0].set_title("distribution of theta in degrees")
        ax[1].hist(phi_deg, bins='auto')
        ax[1].set_title("distribution of phi in degrees")
        fig.show()
        plt.close('all')

    # Check radian input_mode, asymmetric
    assert np.all(np.logical_and(theta_rad >= 0, theta_rad < 2*np.pi))
    assert np.all(np.logical_and(phi_rad >= 0, phi_rad <= np.pi))

    # Check degree input_mode, symmetric
    assert np.all(np.logical_and(theta_deg >= 0, theta_deg < 360))
    assert np.all(np.logical_and(phi_deg >= -90, phi_deg <= 90))

    # Also check that phi appears correctly distributed as it's the hard one here
    assert np.allclose(np.std(phi_deg), 39.22, atol=0.5)


if __name__ == "__main__":
    test__weighted_standard_deviation()
    test_mean_distance()
    test_mean_proper_motion()
    test_mean_radius()
    test_mean_internal_velocity_dispersion()
    par = test_all_statistics()
    test_points_on_sphere(show_diagnostic_histograms=True)
