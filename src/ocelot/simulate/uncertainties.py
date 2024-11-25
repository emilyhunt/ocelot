"""Functions for applying representative Gaia uncertainties to a simulated cluster."""

from __future__ import annotations
import numpy as np  # Necessary to type hint without cyclic import
import ocelot.simulate.cluster
import pandas as pd
import warnings
from .binaries import G_ZP, BP_ZP, RP_ZP, _flux_to_mag


def _closest_gaia_star(
    cluster: ocelot.simulate.cluster.SimulatedCluster, field: pd.DataFrame
):
    """Finds the nearest star in G-band magnitude to a given star."""
    field_magnitudes = field["phot_g_mean_mag"].to_numpy()
    stars_to_assign = cluster.cluster["g_true"].notna()
    cluster_magnitudes = cluster.cluster.loc[stars_to_assign, "g_true"].to_numpy()

    # Search a sorted version of the field magnitudes array to find closest real star
    sort_args = np.argsort(field_magnitudes)
    field_magnitudes_sorted = field_magnitudes[sort_args]
    best_matching_stars_in_sorted = np.searchsorted(
        field_magnitudes_sorted, cluster_magnitudes
    )

    # Any simulated stars with magnitudes higher than observed in the field are given
    # the closest available star
    beyond_allowed_values = best_matching_stars_in_sorted == sort_args.size
    best_matching_stars_in_sorted[beyond_allowed_values] -= 1

    # Find indices back into the field dataframe
    best_matching_stars = sort_args[best_matching_stars_in_sorted]

    return best_matching_stars, stars_to_assign


def _assign_field_uncertainties_to_cluster(
    cluster: ocelot.simulate.cluster.SimulatedCluster, field: None | pd.DataFrame = None
):
    if (
        not cluster.parameters.photometric_errors
        and not cluster.parameters.astrometric_errors
    ):
        return

    if field is None:
        raise ValueError(
            "field containing stars was not specified, meaning it is not possible to "
            "generate uncertainties for this cluster!"
        )
    field_good = field.loc[
        np.logical_and.reduce(
            (
                field["phot_g_mean_mag"].notna(),
                field["phot_bp_mean_mag"].notna(),
                field["phot_rp_mean_mag"].notna(),
                field["parallax_error"].notna(),
                # field["pmra_error"].notna(),  # Probably not needed
                # field["pmdec_error"].notna(),  # Probably not needed
            )
        )
    ]

    best_matching_stars, stars_to_assign = _closest_gaia_star(cluster, field_good)
    matching_stars = field_good.iloc[best_matching_stars]
    cluster.cluster["matching_star_id"] = -1
    cluster.cluster.loc[stars_to_assign, "matching_star_id"] = matching_stars[
        "source_id"
    ].to_numpy()

    if cluster.parameters.photometric_errors:
        cluster.cluster.loc[stars_to_assign, "phot_g_mean_flux_error"] = matching_stars[
            "phot_g_mean_flux_error"
        ].to_numpy()
        cluster.cluster.loc[stars_to_assign, "phot_bp_mean_flux_error"] = (
            matching_stars["phot_bp_mean_flux_error"].to_numpy()
        )
        cluster.cluster.loc[stars_to_assign, "phot_rp_mean_flux_error"] = (
            matching_stars["phot_rp_mean_flux_error"].to_numpy()
        )
    if cluster.parameters.astrometric_errors:
        cluster.cluster.loc[stars_to_assign, "pmra_error"] = matching_stars[
            "pmra_error"
        ].to_numpy()
        cluster.cluster.loc[stars_to_assign, "pmdec_error"] = matching_stars[
            "pmdec_error"
        ].to_numpy()
        cluster.cluster.loc[stars_to_assign, "parallax_error"] = matching_stars[
            "parallax_error"
        ].to_numpy()


def apply_gaia_photometric_uncertainties(
    cluster: ocelot.simulate.cluster.SimulatedCluster, field: None | pd.DataFrame = None
):
    """Applies representative Gaia photometric uncertainties to photometry in a cluster."""
    _assign_field_uncertainties_to_cluster(cluster, field)

    if not cluster.parameters.photometric_errors:
        cluster.cluster["phot_g_mean_mag"] = cluster.cluster["g_true"]
        cluster.cluster["phot_bp_mean_mag"] = cluster.cluster["bp_true"]
        cluster.cluster["phot_rp_mean_mag"] = cluster.cluster["rp_true"]
        return
    
    # Cycle over pmra, pmdec, and parallax, updating their uncertainty value
    for photometric_band, zero_point in zip(("g", "bp", "rp"), (G_ZP, BP_ZP, RP_ZP)):
        error_column = f"phot_{photometric_band}_mean_flux_error"
        flux_column = f"{photometric_band}_flux"
        mag_column = f"phot_{photometric_band}_mean_mag"

        to_assign = cluster.cluster[error_column].notna()
        std = cluster.cluster.loc[to_assign, error_column]

        cluster.cluster.loc[to_assign, mag_column] = _flux_to_mag(
            cluster.cluster.loc[to_assign, flux_column] + 
            cluster.random_generator.normal(loc=0, scale=std),
            zero_point
        )

    # Todo: BP and RP flux issues for Gaia DR3 not added


def apply_gaia_astrometric_uncertainties(
    cluster: ocelot.simulate.cluster.SimulatedCluster,
):
    """Applies representative Gaia astrometric uncertainties to the astrometry of a
    cluster. Assumes that photometric errors have already been applied (which also adds
    astrometric uncertainties for each star.)
    """
    if not cluster.parameters.astrometric_errors:
        cluster.cluster["pmra_plus_anomaly"] = cluster.cluster["pmra_true"]
        cluster.cluster["pmdec_plus_anomaly"] = cluster.cluster["pmdec_true"]
        cluster.cluster["parallax"] = cluster.cluster["parallax_true"]
        return

    # Cycle over pmra, pmdec, and parallax, updating their uncertainty value
    for dimension in ("pmra", "pmdec", "parallax"):
        to_assign = cluster.cluster[f"{dimension}_error"].notna()
        std = (
            cluster.cluster.loc[to_assign, f"{dimension}_error"]
            * cluster.parameters.astrometric_errors_scale_factor
        )
        cluster.cluster.loc[to_assign, f"{dimension}"] += (
            cluster.random_generator.normal(loc=0, scale=std)
            # + cluster.cluster.loc[to_assign, f"{dim}_true"]
        )

    # Todo: Addition of systematic uncertainties (e.g. parallax) not yet done
