"""A set of tests for ocelot.calculate."""

from ocelot.calculate.generic import _weighted_standard_deviation
# from pathlib import Path
import numpy as np

# path_to_blanco_1_cluster = Path(
#     "./test_data/blanco_1_gaia_dr2_gmag_18_cut_cluster_only.pickle"
# )


def test__weighted_standard_deviation():
    """Tests ocelot.calculate.generic._weighted_standard_deviation, a function for
    computing weighted standard deviations."""
    # Create some simple test data which clearly has standard deviation ~1, other than 
    # ten cheeky points
    np.random.seed(42)
    length = 10000
    normal_points = np.random.normal(loc=0.0, scale=1.0, size=length)

    not_so_normal_points = np.empty((length + 1000,), dtype=float)
    not_so_normal_points[:length] = normal_points
    not_so_normal_points[length:] = 10000.0

    weights = np.ones(not_so_normal_points.shape)
    weights[length:] = 0.0

    # Do our calculations
    weighted_std_normal_points = _weighted_standard_deviation(normal_points, None)
    weighted_std_abnormal_points = _weighted_standard_deviation(
        not_so_normal_points, weights
    )
    std_normal_points = np.std(normal_points)

    # Check that my function and the actual standard deviation are about the same
    assert np.allclose(
        weighted_std_normal_points, std_normal_points, rtol=0.0, atol=1e-3
    )

    # Check that the weighting correctly ignores the dodgy points
    assert np.allclose(
        weighted_std_abnormal_points, std_normal_points, rtol=0.0, atol=1e-3
    )


def parameter_return_test_function(parameters, target, rtol=0.0, atol=1e-8):
    """Accepts a parameter dict and a target dict, like those returned by 
    ocelot.calculate.

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
        # Check the keys are the same, in the same location
        assert keys_parameters[i] == keys_target[i]

        # Check the value (done differently depending on if it is or isn't a string)
        if isinstance(target[keys_parameters[i]], str):
            assert parameters[keys_parameters[i]] == target[keys_parameters[i]]
        else:
            assert np.allclose(
                parameters[keys_parameters[i]],
                target[keys_parameters[i]],
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )


# def test_mean_distance():
#     """Tests ocelot.calculate.mean_distance"""
#     # Read in Blanco 1
#     with open(path_to_blanco_1_cluster, "rb") as handle:
#         data_cluster = pickle.load(handle)

#     # Make the ra values non-corrected
#     data_cluster["ra"] = np.where(
#         data_cluster["ra"] < 0, data_cluster["ra"] + 360, data_cluster["ra"]
#     )

#     # Calculate distances
#     parameters = ocelot.calculate.mean_distance(data_cluster)

#     target_dict = {
#         "parallax": 4.124794980897457,
#         "parallax_std": 0.3915133864674593,
#         "parallax_error": 0.024187777793597237,
#         "inverse_parallax": 242.4362918959972,
#         "inverse_parallax_l68": 241.02293457480792,
#         "inverse_parallax_u68": 243.86632283215837,
#         "distance": 244.3519932235079,
#         "distance_std": 36.51808584564543,
#         "distance_error": 2.25609487801101,
#     }

#     parameter_return_test_function(parameters, target_dict)


# def test_mean_proper_motion():
#     """Tests ocelot.calculate.mean_proper_motion"""
#     # Read in Blanco 1
#     with open(path_to_blanco_1_cluster, "rb") as handle:
#         data_cluster = pickle.load(handle)

#     # Make the ra values non-corrected
#     data_cluster["ra"] = np.where(
#         data_cluster["ra"] < 0, data_cluster["ra"] + 360, data_cluster["ra"]
#     )

#     # Calculate the mean proper motion
#     parameters = ocelot.calculate.mean_proper_motion(data_cluster)

#     target_dict = {
#         "pmra": 18.691800985496148,
#         "pmra_std": 0.7782083402189158,
#         "pmra_error": 0.04807787181985345,
#         "pmdec": 2.6059267710228995,
#         "pmdec_std": 0.7849144024486323,
#         "pmdec_error": 0.0484921737280103,
#         "pm_dispersion": 1.1053048629032505,
#     }

#     parameter_return_test_function(parameters, target_dict)


# def test_mean_radius():
#     """Tests ocelot.calculate.mean_radius"""
#     # Read in Blanco 1
#     with open(path_to_blanco_1_cluster, "rb") as handle:
#         data_cluster = pickle.load(handle)

#     # Make the ra values non-corrected
#     data_cluster["ra"] = np.where(
#         data_cluster["ra"] < 0, data_cluster["ra"] + 360, data_cluster["ra"]
#     )

#     # Calculate the mean proper motion
#     parameters = ocelot.calculate.mean_radius(data_cluster)

#     target_dict = {
#         "ra": 0.9211424874405434,
#         "ra_std": 0.42919239850712043,
#         "ra_error": 0.026515594931398594,
#         "dec": -29.968388740371715,
#         "dec_std": 0.37079687952364343,
#         "dec_error": 0.022907907720347058,
#         "ang_dispersion": 0.5671828988966136,
#         "ang_radius_50": 0.4704416634430194,
#         "ang_radius_50_error": np.nan,
#         "ang_radius_c": np.nan,
#         "ang_radius_c_error": np.nan,
#         "ang_radius_t": 0.9014932764089681,
#         "ang_radius_t_error": np.nan,
#         "radius_50": 1.9906299639509861,
#         "radius_50_error": np.nan,
#         "radius_c": np.nan,
#         "radius_c_error": np.nan,
#         "radius_t": 3.814813688665708,
#         "radius_t_error": np.nan,
#     }

#     parameter_return_test_function(parameters, target_dict)


# def test_mean_internal_velocity_dispersion():
#     """Tests ocelot.calculate.internal_velocity_dispersion"""
#     # Read in Blanco 1
#     with open(path_to_blanco_1_cluster, "rb") as handle:
#         data_cluster = pickle.load(handle)

#     # Make the ra values non-corrected
#     data_cluster["ra"] = np.where(
#         data_cluster["ra"] < 0, data_cluster["ra"] + 360, data_cluster["ra"]
#     )

#     # Calculate the mean proper motion
#     parameters = ocelot.calculate.mean_internal_velocity_dispersion(data_cluster)

#     target_dict = {
#         "v_internal_tangential": 931.3613254058669,
#         "v_internal_tangential_error": np.nan,
#     }

#     parameter_return_test_function(parameters, target_dict)


# def test_all_statistics():
#     """Tests ocelot.calculate.all_statistics"""
#     # Read in Blanco 1
#     with open(path_to_blanco_1_cluster, "rb") as handle:
#         data_cluster = pickle.load(handle)

#     # Make the ra values non-corrected
#     data_cluster["ra"] = np.where(
#         data_cluster["ra"] < 0, data_cluster["ra"] + 360, data_cluster["ra"]
#     )

#     # Calculate the mean proper motion
#     parameters = ocelot.calculate.all_statistics(data_cluster, mode="mean")

#     target_dict = {
#         "n_stars": 262,
#         "ra": 0.9211424874405434,
#         "ra_std": 0.42919239850712043,
#         "ra_error": 0.026515594931398594,
#         "dec": -29.968388740371715,
#         "dec_std": 0.37079687952364343,
#         "dec_error": 0.022907907720347058,
#         "ang_dispersion": 0.5671828988966136,
#         "ang_radius_50": 0.4704416634430194,
#         "ang_radius_50_error": np.nan,
#         "ang_radius_c": np.nan,
#         "ang_radius_c_error": np.nan,
#         "ang_radius_t": 0.9014932764089681,
#         "ang_radius_t_error": np.nan,
#         "radius_50": 1.9906299639509861,
#         "radius_50_error": np.nan,
#         "radius_c": np.nan,
#         "radius_c_error": np.nan,
#         "radius_t": 3.814813688665708,
#         "radius_t_error": np.nan,
#         "parallax": 4.124794980897457,
#         "parallax_std": 0.3915133864674593,
#         "parallax_error": 0.024187777793597237,
#         "inverse_parallax": 242.4362918959972,
#         "inverse_parallax_l68": 241.02293457480792,
#         "inverse_parallax_u68": 243.86632283215837,
#         "distance": 244.3519932235079,
#         "distance_std": 36.51808584564543,
#         "distance_error": 2.25609487801101,
#         "pmra": 18.691800985496148,
#         "pmra_std": 0.7782083402189158,
#         "pmra_error": 0.04807787181985345,
#         "pmdec": 2.6059267710228995,
#         "pmdec_std": 0.7849144024486323,
#         "pmdec_error": 0.0484921737280103,
#         "pm_dispersion": 1.1053048629032505,
#         "v_internal_tangential": 931.3613254058669,
#         "v_internal_tangential_error": np.nan,
#         "parameter_inference_mode": "mean",
#     }

#     parameter_return_test_function(parameters, target_dict)

#     return parameters


if __name__ == "__main__":
    test__weighted_standard_deviation()
    # test_mean_distance()
    # test_mean_proper_motion()
    # test_mean_radius()
    # test_mean_internal_velocity_dispersion()
    # par = test_all_statistics()
    # test_points_on_sphere(show_diagnostic_histograms=True)
