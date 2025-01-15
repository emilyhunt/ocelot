"""Functions for dealing with cluster photometry, including account for the Gaia and
subsample selection functions.
"""

from __future__ import annotations  # Necessary to type hint without cyclic import
import warnings
import numpy as np
import pandas as pd
import ocelot.simulate.cluster
from scipy.interpolate import interp1d
import imf
from ocelot import DATA_PATH
from astropy.coordinates import SkyCoord


IMF = imf.Kroupa
ISOCHRONES_DIRECTORY = DATA_PATH / "isochrones/PARSEC_v1.2S"
if not ISOCHRONES_DIRECTORY.exists():
    warnings.warn(
        f"Unable to find directory of isochrones at {str(ISOCHRONES_DIRECTORY)}. "
        "Will be unable to generate clusters."
    )


AVAILABLE_METALLICITIES = np.asarray(
    [
        float(name.stem.split("=")[1])
        for name in ISOCHRONES_DIRECTORY.glob("mh=*.parquet")
    ]
)
MINIMUM_METALLICITY = AVAILABLE_METALLICITIES.min()
MAXIMUM_METALLICITY = AVAILABLE_METALLICITIES.max()


def load_isochrone(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Loads a simulated stellar population at a given age."""
    # Todo refactor to an Isochrone model class or similar. Also add interpolation probably. Also.also link up better with observation models
    # Check that the requested metallicity is valid
    metallicity = cluster.parameters.metallicity

    if metallicity < MINIMUM_METALLICITY:
        warnings.warn(
            f"Desired metallicity of [M/H]={metallicity:.5f} is less than the minimum "
            "available of {MINIMUM_METALLICITY}!"
        )
    elif metallicity > MAXIMUM_METALLICITY:
        warnings.warn(
            f"Desired metallicity of [M/H]={metallicity:.5f} is greater than the "
            "maximum available of {MAXIMUM_METALLICITY}!"
        )

    # Get the closest available metallicity & read in
    metallicity_to_use = AVAILABLE_METALLICITIES[
        np.abs(AVAILABLE_METALLICITIES - metallicity).argmin()
    ]
    isochrones = pd.read_parquet(
        ISOCHRONES_DIRECTORY / f"mh={metallicity_to_use:.3f}.parquet"
    )

    # Filter by age
    ages = np.unique(isochrones["logAge"].to_numpy())
    best_age = ages[(np.abs(ages - cluster.parameters.log_age)).argmin()]
    isochrone = (
        isochrones.loc[isochrones["logAge"] == best_age]
        .sort_values("Mass")
        .reset_index(drop=True)
    )

    # Read in & return best one
    return isochrone


def _interpolated_parameter(parameter, isochrone, masses):
    return interp1d(isochrone["Mini"], isochrone[parameter], bounds_error=False)(masses)


def create_population(
    cluster: ocelot.simulate.cluster.SimulatedCluster, minimum_mass=0.03
):
    """Samples from a pre-simulated stellar population until the sample population has
    the correct mass.
    """
    isochrone = load_isochrone(cluster)

    # Make initial stars
    selected_imf = IMF(mmin=minimum_mass, mmax=isochrone["Mini"].max())
    masses = imf.make_cluster(
        cluster.parameters.mass, massfunc=selected_imf, silent=True
    )
    if len(masses) == 0:
        raise RuntimeError(
            "Generated cluster contains zero stars! Consider increasing the mass of "
            "your cluster."
        )

    ids = np.arange(len(masses))
    cluster.cluster = pd.DataFrame.from_dict(
        {
            "simulated_id": ids,
            "cluster_id": cluster.parameters.id,
            "simulated_star": True,
            "mass_initial": masses,
        }
    )

    # Add stuff from PARSEC
    distance = cluster.parameters.distance
    distance_modulus = 5 * np.log10(distance) - 5

    cluster.cluster["mass"] = _interpolated_parameter("Mass", isochrone, masses)
    cluster.cluster["temperature"] = 10 ** (
        _interpolated_parameter("logTe", isochrone, masses)
    )
    cluster.cluster["luminosity"] = 10 ** (
        _interpolated_parameter("logL", isochrone, masses)
    )
    cluster.cluster["log_g"] = _interpolated_parameter("logg", isochrone, masses)
    cluster.cluster["gaia_dr3_g_true"] = (
        _interpolated_parameter("Gmag", isochrone, masses) + distance_modulus
    )
    cluster.cluster["gaia_dr3_bp_true"] = (
        _interpolated_parameter("G_BPmag", isochrone, masses) + distance_modulus
    )
    cluster.cluster["gaia_dr3_rp_true"] = (
        _interpolated_parameter("G_RPmag", isochrone, masses) + distance_modulus
    )

    # Make sure brown dwarfs still have a mass (we just assume no mass loss)
    nan_mass = cluster.cluster["mass"].isna()
    cluster.cluster.loc[nan_mass, "mass"] = cluster.cluster.loc[
        nan_mass, "mass_initial"
    ]

    # Optionally also prune the cluster
    if len(cluster.prune_simulated_cluster) > 0:
        cluster.cluster = cluster.cluster.query(
            cluster.prune_simulated_cluster
        ).reset_index(drop=True)


def apply_extinction(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Applies extinction across a cluster."""
    # Easy cases: when extinction is 0 or differential extinction is 0
    if (
        cluster.parameters.extinction == 0.0
        and cluster.parameters.differential_extinction == 0.0
    ):
        cluster.cluster["extinction"] = 0.0
        return
    if (
        cluster.parameters.differential_extinction == 0.0
        or not cluster.features.differential_extinction
        or len(cluster.cluster) == 1  # Can't do diff A_V if there's just one star!
    ):
        cluster.cluster["extinction"] = cluster.parameters.extinction
        return

    # Harder cases: when we need to differentially extinguish
    # We need to change coordinate frame to be centred on the cluster
    # Todo: may be a bad choice when dealing with a big cluster? I don't know
    center = SkyCoord(cluster.parameters.ra, cluster.parameters.dec, unit="deg")
    coords = SkyCoord(cluster.cluster["ra"], cluster.cluster["dec"], unit="deg")
    coords_transformed = coords.transform_to(center.skyoffset_frame())

    # Then, we just query the model!
    cluster.cluster["extinction"] = cluster.models.differential_reddening.extinction(
        coords_transformed.lon.value,
        coords_transformed.lat.value,
        mean=cluster.parameters.extinction,
        width=cluster.parameters.differential_extinction,
    )
