"""Functions for dealing with cluster photometry, including account for the Gaia and
subsample selection functions.
"""

from __future__ import annotations  # Necessary to type hint without cyclic import
import warnings
import numpy as np
from ocelot.simulate.selection import apply_selection_functions
import pandas as pd
import ocelot.simulate.cluster
from ocelot.simulate.binaries import make_binaries
from ocelot.simulate import photutils
from ocelot.simulate.uncertainties import apply_gaia_photometric_uncertainties
from scipy.interpolate import interp1d
import imf


IMF = imf.Kroupa
ISOCHRONES_DIRECTORY = DATA_DIRECTORY / "PARSEC"
if not ISOCHRONES_DIRECTORY.exists():
    warnings.warn(
        f"Unable to find directory of isochrones at {str(ISOCHRONES_DIRECTORY)}. "
        "Will be unable to generate clusters."
    )


def load_isochrone(cluster: ocelot.simulate.cluster.Cluster):
    """Loads a simulated stellar population at a given age."""
    # Check that we have isochrones at that metallicity
    isochrones_folder = (
        ISOCHRONES_DIRECTORY / f"FeH_{cluster.parameters.metallicity:.1f}"
    )
    if not isochrones_folder.exists():
        raise ValueError("Isochrones at selected metallicity not found.")

    # Pick best isochrone to read in
    all_isochrones = list(isochrones_folder.glob("*.csv"))
    ages = np.asarray([float(x.stem) for x in isochrones_folder.glob("*.csv")])
    id_nearest_age = (np.abs(ages - cluster.parameters.log_age)).argmin()

    # Read in & return best one
    return pd.read_csv(all_isochrones[id_nearest_age])


def _interpolated_parameter(parameter, isochrone, masses):
    return interp1d(isochrone["Mini"], isochrone[parameter], bounds_error=False)(masses)


def create_population(cluster: ocelot.simulate.cluster.Cluster, minimum_mass=0.03):
    """Samples from a pre-simulated stellar population until the sample population has
    the correct mass.
    """
    isochrone = load_isochrone(cluster)

    # Make initial stars
    selected_imf = IMF(mmin=minimum_mass, mmax=isochrone["Mini"].max())
    masses = imf.make_cluster(cluster.parameters.mass, massfunc=selected_imf, silent=True)
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
    cluster.cluster["t_eff"] = 10 ** (
        _interpolated_parameter("logTe", isochrone, masses)
    )
    cluster.cluster["log_g"] = (
        _interpolated_parameter("logTe", isochrone, masses)
    )
    cluster.cluster["g_true"] = (
        _interpolated_parameter("G", isochrone, masses) + distance_modulus
    )
    cluster.cluster["bp_true"] = (
        _interpolated_parameter("BP", isochrone, masses) + distance_modulus
    )
    cluster.cluster["rp_true"] = (
        _interpolated_parameter("RP", isochrone, masses) + distance_modulus
    )

    # Make sure brown dwarfs still have a mass (we just assume no mass loss)
    nan_mass = cluster.cluster["mass"].isna()
    cluster.cluster.loc[nan_mass, "mass"] = cluster.cluster.loc[
        nan_mass, "mass_initial"
    ]


def apply_extinction(cluster: ocelot.simulate.cluster.Cluster):
    """Applies extinction to a simulated cluster."""
    if cluster.parameters.extinction == 0.0:
        cluster.cluster["a_g"] = 0.0
        cluster.cluster["a_bp"] = 0.0
        cluster.cluster["a_rp"] = 0.0
        return

    extinction_repeated = np.repeat(cluster.parameters.extinction, len(cluster.cluster))

    # Calculate G, BP, and RP band extinctions from extinction in A_V
    cluster.cluster["a_g"] = photutils.AG(
        extinction_repeated, cluster.cluster["t_eff"].to_numpy()
    )
    cluster.cluster["a_bp"] = photutils.ABP(
        extinction_repeated, cluster.cluster["t_eff"].to_numpy()
    )
    cluster.cluster["a_rp"] = photutils.ARP(
        extinction_repeated, cluster.cluster["t_eff"].to_numpy()
    )

    # Assign to cluster stars
    cluster.cluster["g_true"] = cluster.cluster["g_true"] + cluster.cluster["a_g"]
    cluster.cluster["bp_true"] = cluster.cluster["bp_true"] + cluster.cluster["a_bp"]
    cluster.cluster["rp_true"] = cluster.cluster["rp_true"] + cluster.cluster["a_rp"]


def generate_cluster_photometry(
    cluster: ocelot.simulate.cluster.Cluster, field: None | pd.DataFrame = None
):
    """Generates a star cluster of given photometry at a given age and extinction."""
    create_population(cluster)
    apply_extinction(cluster)
    make_binaries(cluster)
    apply_selection_functions(cluster, field)
    apply_gaia_photometric_uncertainties(cluster, field)