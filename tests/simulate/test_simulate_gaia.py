"""Utilities to test a simulation of a Gaia observation."""

from ocelot.simulate import (
    SimulatedCluster,
    SimulatedClusterModels,
)
from ocelot.simulate.cluster import SimulatedClusterFeatures
from ocelot.model.observation import (
    GaiaDR3ObservationModel,
    GenericSubsampleSelectionFunction,
)
from .test_simulate import _get_default_parameters
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy import units as u


def _get_gaia_test_data():
    location = (
        Path(__file__).parent.parent
        / "test_data/gaia/dr3/ra=45,dec=0,radius=0.5.parquet"
    )
    return pd.read_parquet(location)


def _get_gaia_model(subsample=False):
    gaia_test_data = _get_gaia_test_data()
    subsample_sf = []
    if subsample:
        gaia_data_subsample = gaia_test_data.loc[
            np.logical_and.reduce(
                (
                    gaia_test_data["astrometric_params_solved"] >= 31,
                    gaia_test_data["phot_g_mean_mag"].notna(),
                    gaia_test_data["phot_bp_mean_mag"].notna(),
                    gaia_test_data["phot_rp_mean_mag"].notna(),
                    gaia_test_data["ruwe"] < 1.4,
                    gaia_test_data["phot_g_mean_mag"] < 19,
                )
            )
        ]
        subsample_sf = [
            GenericSubsampleSelectionFunction(
                gaia_test_data,
                gaia_data_subsample,
                column="gaia_dr3_g",
                column_in_data="phot_g_mean_mag",
            )
        ]
    model = SimulatedClusterModels(
        observations=[
            GaiaDR3ObservationModel(
                representative_stars=gaia_test_data,
                subsample_selection_functions=subsample_sf,
            )
        ]
    )
    return model


def test_basic_gaia_observation():
    features = SimulatedClusterFeatures(
        selection_effects=False,
        astrometric_uncertainties=False,
        photometric_uncertainties=False,
    )
    parameters = _get_default_parameters()
    models = _get_gaia_model(subsample=False)
    cluster = SimulatedCluster(
        parameters=parameters, features=features, models=models, random_seed=42
    )
    cluster.make()

    # Basic checks of the observation
    observation = cluster.observations["gaia_dr3"]

    # Check that we made unresolved binaries
    assert len(observation) < len(cluster.cluster)

    # Check that the number of unresolved companions is correct
    assert observation["unresolved_companions"].sum() == len(cluster.cluster) - len(
        observation
    )

    # Check that we have additional columns
    expected_cols = pd.Index(
        [
            "simulated_id",
            "cluster_id",
            "simulated_star",
            "mass_initial",
            "mass",
            "temperature",
            "luminosity",
            "log_g",
            "gaia_dr3_g_true",
            "gaia_dr3_bp_true",
            "gaia_dr3_rp_true",
            "companions",
            "mass_ratio",
            "period",
            "eccentricity",
            "simulated_id_primary",
            "ra",
            "dec",
            "l",
            "b",
            "pmra",
            "pmdec",
            "parallax",
            "pmra_true",
            "pmdec_true",
            "parallax_true",
            "radial_velocity_true",
            "extinction",
            "extinction_gaia_dr3_g",
            "extinction_gaia_dr3_bp",
            "extinction_gaia_dr3_rp",
            "gaia_dr3_g",
            "gaia_dr3_bp",
            "gaia_dr3_rp",
            "unresolved_companions",
        ],
        dtype="object",
    )
    pd.testing.assert_index_equal(observation.columns, expected_cols)


def test_gaia_observation_uncertainties():
    features = SimulatedClusterFeatures(
        selection_effects=False,
    )
    parameters = _get_default_parameters()
    models = _get_gaia_model(subsample=False)
    cluster = SimulatedCluster(
        parameters=parameters, features=features, models=models, random_seed=42
    )
    cluster.make()

    # Basic checks of the observation
    observation = cluster.observations["gaia_dr3"]

    # Check that we have additional columns
    expected_cols = pd.Index(
        [
            "simulated_id",
            "cluster_id",
            "simulated_star",
            "mass_initial",
            "mass",
            "temperature",
            "luminosity",
            "log_g",
            "gaia_dr3_g_true",
            "gaia_dr3_bp_true",
            "gaia_dr3_rp_true",
            "companions",
            "mass_ratio",
            "period",
            "eccentricity",
            "simulated_id_primary",
            "ra",
            "dec",
            "l",
            "b",
            "pmra",
            "pmdec",
            "parallax",
            "pmra_true",
            "pmdec_true",
            "parallax_true",
            "radial_velocity_true",
            "extinction",
            "extinction_gaia_dr3_g",
            "extinction_gaia_dr3_bp",
            "extinction_gaia_dr3_rp",
            "gaia_dr3_g",
            "gaia_dr3_bp",
            "gaia_dr3_rp",
            "unresolved_companions",
            "matching_gaia_dr3_source_id",
            "gaia_dr3_g_flux_error",
            "gaia_dr3_bp_flux_error",
            "gaia_dr3_rp_flux_error",
            "pmra_error",
            "pmdec_error",
            "parallax_error",
        ],
        dtype="object",
    )
    pd.testing.assert_index_equal(observation.columns, expected_cols)

    # Check that uncertainties did something
    assert (observation["pmra"] != observation["pmra_true"]).all()
    assert (observation["pmdec"] != observation["pmdec_true"]).all()
    assert (observation["parallax"] != observation["parallax_true"]).all()

    assert (observation["gaia_dr3_g"] != observation["gaia_dr3_g_true"]).all()
    assert (observation["gaia_dr3_bp"] != observation["gaia_dr3_bp_true"]).all()
    assert (observation["gaia_dr3_rp"] != observation["gaia_dr3_rp_true"]).all()

    # Check that the order of magnitude is about right
    observation_with_mag = observation.loc[observation["gaia_dr3_g_true"].notna()]
    assert (
        (
            np.abs(observation_with_mag["pmra"] - observation_with_mag["pmra_true"])
            / observation_with_mag["pmra_error"]
        )
        < 5.0  # 5 sigma feels reasonable - 1 in a million to be above this!
    ).all()


def test_gaia_observation_selection():
    parameters = _get_default_parameters()

    models = _get_gaia_model(subsample=False)
    features = SimulatedClusterFeatures(
        selection_effects=False,
    )
    cluster_no_selection = SimulatedCluster(
        parameters=parameters, features=features, models=models, random_seed=42
    )
    cluster_no_selection.make()

    models = _get_gaia_model()
    cluster = SimulatedCluster(parameters=parameters, models=models, random_seed=42)
    cluster.make()

    # Basic checks of the observation
    observation_no_selection = cluster_no_selection.observations["gaia_dr3"]
    observation = cluster.observations["gaia_dr3"]

    # Check that we have fewer stars
    assert len(observation) < len(observation_no_selection)

    # Check that we have additional columns
    expected_cols = pd.Index(
        [
            "simulated_id",
            "cluster_id",
            "simulated_star",
            "mass_initial",
            "mass",
            "temperature",
            "luminosity",
            "log_g",
            "gaia_dr3_g_true",
            "gaia_dr3_bp_true",
            "gaia_dr3_rp_true",
            "companions",
            "mass_ratio",
            "period",
            "eccentricity",
            "simulated_id_primary",
            "ra",
            "dec",
            "l",
            "b",
            "pmra",
            "pmdec",
            "parallax",
            "pmra_true",
            "pmdec_true",
            "parallax_true",
            "radial_velocity_true",
            "extinction",
            "extinction_gaia_dr3_g",
            "extinction_gaia_dr3_bp",
            "extinction_gaia_dr3_rp",
            "gaia_dr3_g",
            "gaia_dr3_bp",
            "gaia_dr3_rp",
            "unresolved_companions",
            "matching_gaia_dr3_source_id",
            "gaia_dr3_g_flux_error",
            "gaia_dr3_bp_flux_error",
            "gaia_dr3_rp_flux_error",
            "pmra_error",
            "pmdec_error",
            "parallax_error",
            "selection_probability_GaiaDR3SelectionFunction",
            "selection_probability",
        ],
        dtype="object",
    )
    pd.testing.assert_index_equal(observation.columns, expected_cols)


def test_gaia_observation_cannot_be_reused():
    parameters = _get_default_parameters()
    models = _get_gaia_model(subsample=False)
    SimulatedCluster(parameters=parameters, models=models, random_seed=42).make()

    cluster_two = SimulatedCluster(
        parameters=parameters, models=models, random_seed=42
    ).make_cluster()
    with pytest.raises(RuntimeError):
        cluster_two.make_observations()


def test_gaia_observation_selection_with_subsample():
    params = _get_default_parameters()

    models = _get_gaia_model(subsample=False)
    cluster_no_subsample = SimulatedCluster(
        parameters=params, models=models, random_seed=42
    ).make()

    models = _get_gaia_model(subsample=True)
    cluster = SimulatedCluster(parameters=params, models=models, random_seed=42).make()

    # Basic checks of the observation
    observation_no_subsample = cluster_no_subsample.observations["gaia_dr3"]
    observation = cluster.observations["gaia_dr3"]

    # Check that we have fewer stars
    assert len(observation) < len(observation_no_subsample)

    # Check that we have additional columns
    expected_cols = pd.Index(
        [
            "simulated_id",
            "cluster_id",
            "simulated_star",
            "mass_initial",
            "mass",
            "temperature",
            "luminosity",
            "log_g",
            "gaia_dr3_g_true",
            "gaia_dr3_bp_true",
            "gaia_dr3_rp_true",
            "companions",
            "mass_ratio",
            "period",
            "eccentricity",
            "simulated_id_primary",
            "ra",
            "dec",
            "l",
            "b",
            "pmra",
            "pmdec",
            "parallax",
            "pmra_true",
            "pmdec_true",
            "parallax_true",
            "radial_velocity_true",
            "extinction",
            "extinction_gaia_dr3_g",
            "extinction_gaia_dr3_bp",
            "extinction_gaia_dr3_rp",
            "gaia_dr3_g",
            "gaia_dr3_bp",
            "gaia_dr3_rp",
            "unresolved_companions",
            "matching_gaia_dr3_source_id",
            "gaia_dr3_g_flux_error",
            "gaia_dr3_bp_flux_error",
            "gaia_dr3_rp_flux_error",
            "pmra_error",
            "pmdec_error",
            "parallax_error",
            "selection_probability_GaiaDR3SelectionFunction",
            "selection_probability_GenericSubsampleSelectionFunction",
            "selection_probability",
        ],
        dtype="object",
    )
    pd.testing.assert_index_equal(observation.columns, expected_cols)


def test_cluster_with_zero_stars():
    """Checks that generating a cluster with zero stars works."""
    params = _get_default_parameters(distance=100000)
    params.mass = 10
    models = _get_gaia_model(subsample=True)
    cluster = SimulatedCluster(parameters=params, random_seed=42, models=models).make()

    assert len(cluster.observations["gaia_dr3"]) == 0

    # Check that we have additional columns
    expected_cols = pd.Index(
        [
            "simulated_id",
            "cluster_id",
            "simulated_star",
            "mass_initial",
            "mass",
            "temperature",
            "luminosity",
            "log_g",
            "gaia_dr3_g_true",
            "gaia_dr3_bp_true",
            "gaia_dr3_rp_true",
            "companions",
            "mass_ratio",
            "period",
            "eccentricity",
            "simulated_id_primary",
            "ra",
            "dec",
            "l",
            "b",
            "pmra",
            "pmdec",
            "parallax",
            "pmra_true",
            "pmdec_true",
            "parallax_true",
            "radial_velocity_true",
            "extinction",
            "extinction_gaia_dr3_g",
            "extinction_gaia_dr3_bp",
            "extinction_gaia_dr3_rp",
            "gaia_dr3_g",
            "gaia_dr3_bp",
            "gaia_dr3_rp",
            "unresolved_companions",
            "matching_gaia_dr3_source_id",
            "gaia_dr3_g_flux_error",
            "gaia_dr3_bp_flux_error",
            "gaia_dr3_rp_flux_error",
            "pmra_error",
            "pmdec_error",
            "parallax_error",
            "selection_probability_GaiaDR3SelectionFunction",
            "selection_probability_GenericSubsampleSelectionFunction",
            "selection_probability",
        ],
        dtype="object",
    )
    pd.testing.assert_index_equal(
        cluster.observations["gaia_dr3"].columns, expected_cols
    )


def test_cluster_with_one_star():
    """Checks that generating a cluster with one star works.

    N.B. this took a lot of messing around to generate just one star. May break easily.
    """
    params = _get_default_parameters(distance=10000)
    params.mass = 15
    models = _get_gaia_model(subsample=True)
    cluster = SimulatedCluster(parameters=params, random_seed=42, models=models).make()

    assert len(cluster.observations["gaia_dr3"]) == 1

    # Check that we have additional columns
    expected_cols = pd.Index(
        [
            "simulated_id",
            "cluster_id",
            "simulated_star",
            "mass_initial",
            "mass",
            "temperature",
            "luminosity",
            "log_g",
            "gaia_dr3_g_true",
            "gaia_dr3_bp_true",
            "gaia_dr3_rp_true",
            "companions",
            "mass_ratio",
            "period",
            "eccentricity",
            "simulated_id_primary",
            "ra",
            "dec",
            "l",
            "b",
            "pmra",
            "pmdec",
            "parallax",
            "pmra_true",
            "pmdec_true",
            "parallax_true",
            "radial_velocity_true",
            "extinction",
            "extinction_gaia_dr3_g",
            "extinction_gaia_dr3_bp",
            "extinction_gaia_dr3_rp",
            "gaia_dr3_g",
            "gaia_dr3_bp",
            "gaia_dr3_rp",
            "unresolved_companions",
            "matching_gaia_dr3_source_id",
            "gaia_dr3_g_flux_error",
            "gaia_dr3_bp_flux_error",
            "gaia_dr3_rp_flux_error",
            "pmra_error",
            "pmdec_error",
            "parallax_error",
            "selection_probability_GaiaDR3SelectionFunction",
            "selection_probability_GenericSubsampleSelectionFunction",
            "selection_probability",
        ],
        dtype="object",
    )
    pd.testing.assert_index_equal(
        cluster.observations["gaia_dr3"].columns, expected_cols
    )


def test_tiny_nearby_cluster():
    """Checks we can cope with tiny distances too!"""
    params = _get_default_parameters(distance=1)
    params.mass = 10
    models = _get_gaia_model(subsample=True)
    SimulatedCluster(parameters=params, random_seed=42, models=models).make()


def test_cluster_parameters():
    """Re-measures cluster parameters and checks that EVERYTHING works out correct."""
    params = _get_default_parameters(distance=100, r_core=5)
    params.mass = 1000
    models = _get_gaia_model(subsample=True)
    cluster = SimulatedCluster(parameters=params, random_seed=42, models=models).make()

    # Check total mass is sensible
    simulated = cluster.cluster
    np.testing.assert_allclose(
        simulated["mass"].sum(), params.mass, atol=0.0, rtol=1e-3
    )

    # Check things about the observation
    observation = cluster.observations["gaia_dr3"]

    # Check bulk parameters
    np.testing.assert_allclose(observation["ra"].mean(), params.ra, atol=0.1)
    np.testing.assert_allclose(observation["dec"].mean(), params.dec, atol=0.1)
    np.testing.assert_allclose(observation["pmra"].mean(), params.pmra, atol=0.1)
    np.testing.assert_allclose(observation["pmdec"].mean(), params.pmdec, atol=0.1)
    np.testing.assert_allclose(
        1000 / observation["parallax"].mean(), params.distance, atol=0.1
    )
    np.testing.assert_allclose(
        observation["radial_velocity_true"].mean(), params.radial_velocity, atol=0.1
    )

    # Check radius
    coords = SkyCoord(observation['ra'], observation['dec'], unit="deg")
    separations = coords.separation(params.position)
    separations_pc = np.tan(separations.to(u.rad).value) * params.distance

    np.testing.assert_allclose(np.max(separations_pc), params.r_tidal, atol=0, rtol=0.1)
    np.testing.assert_allclose(np.median(separations_pc), params.r_50, atol=0, rtol=0.1)

    # Todo test parallax and pmra distributions. Hard to do as they include errors.
