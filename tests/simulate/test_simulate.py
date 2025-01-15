from ocelot.simulate import (
    SimulatedCluster,
    SimulatedClusterModels,
    SimulatedClusterParameters,
)
from ocelot.simulate.cluster import SimulatedClusterFeatures
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
import numpy as np


def _get_default_parameters():
    return SimulatedClusterParameters(
        position=SkyCoord(
            ra=45 * u.deg,
            dec=0 * u.deg,
            distance=1000 * u.pc,
            pm_ra_cosdec=10 * u.mas / u.yr,
            pm_dec=0 * u.mas / u.yr,
            radial_velocity=0 * u.km / u.s,
            frame="icrs",
        ),
        mass=1000,
        log_age=8.0,
        metallicity=0.2,
        extinction=1.0,
        differential_extinction=0.3,
        r_core=2,
        r_tidal=10,
    )


def test_simulation_no_features():
    features = SimulatedClusterFeatures(
        binary_stars=False,
        differential_extinction=False,
        selection_effects=False,
        astrometric_uncertainties=False,
        photometric_uncertainties=False,
    )
    parameters = _get_default_parameters()
    cluster = SimulatedCluster(parameters=parameters, features=features)
    cluster.make()
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
        ],
        dtype="object",
    )

    # Check that we aren't missing any expected columns
    pd.testing.assert_index_equal(cluster.cluster.columns, expected_cols)


def test_simulation_no_features_with_random_seed():
    features = SimulatedClusterFeatures(
        binary_stars=False,
        differential_extinction=False,
        selection_effects=False,
        astrometric_uncertainties=False,
        photometric_uncertainties=False,
    )
    parameters = _get_default_parameters()
    cluster = SimulatedCluster(parameters=parameters, features=features, random_seed=42)
    cluster.make()

    cluster_two = SimulatedCluster(
        parameters=parameters, features=features, random_seed=42
    )
    cluster_two.make()

    # Check that both dataframes are identical
    pd.testing.assert_frame_equal(cluster.cluster, cluster_two.cluster)

    # Check that we have the expected number of stars
    assert len(cluster.cluster) == 3065

    # Make sure they all have masses
    assert cluster.cluster["mass"].notna().all()

    # Make sure that some (the not-brown-dwarfs) have temperatures, etc
    assert cluster.cluster["temperature"].notna().any()
    assert cluster.cluster["log_g"].notna().any()
    assert cluster.cluster["luminosity"].notna().any()


def test_simulation_with_binaries():
    features = SimulatedClusterFeatures(
        binary_stars=True,
        differential_extinction=False,
        selection_effects=False,
        astrometric_uncertainties=False,
        photometric_uncertainties=False,
    )
    parameters = _get_default_parameters()
    cluster = SimulatedCluster(parameters=parameters, features=features, random_seed=42)
    cluster.make()

    # Check that we have the expected number of stars still
    assert len(cluster.cluster) == 3065

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
            "index_primary",
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
        ],
        dtype="object",
    )

    # Check that we aren't missing any expected columns
    pd.testing.assert_index_equal(cluster.cluster.columns, expected_cols)

    # Make sure we have expected number of binaries with this seed
    is_secondary = cluster.cluster["index_primary"] != -1
    n_secondaries = (is_secondary).sum()
    assert n_secondaries == 741

    # Do some checks of our binaries
    secondaries = cluster.cluster.loc[is_secondary]

    # Check that indexing makes sense
    primaries_by_index = cluster.cluster.loc[secondaries["index_primary"]].reset_index(
        drop=True
    )
    primaries_by_simulated_star = (
        cluster.cluster.loc[
            np.isin(
                cluster.cluster["simulated_id"], secondaries["simulated_id_primary"]
            )
        ]
        .sort_values("mass")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(primaries_by_index, primaries_by_simulated_star)

    # Check for no NaN values - especially important as we use numba
    assert secondaries["mass_ratio"].notna().all()
    assert secondaries["period"].notna().all()
    assert secondaries["eccentricity"].notna().all()
    assert secondaries["simulated_id_primary"].notna().all()

    # Check parameters in expected ranges
    assert (secondaries["mass_ratio"] <= 1.0).all()
    assert (secondaries["mass_ratio"] > 0.0).all()
    assert (secondaries["period"] > 0.0).all()
    assert (secondaries["eccentricity"] >= 0.0).all()
    assert (secondaries["eccentricity"] < 1.0).all()
