"""Functions common to many different observation classes."""

from __future__ import annotations
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from ocelot.model.observation import BaseObservation
import ocelot.simulate.cluster


def calculate_separation(primary: pd.DataFrame, secondary: pd.DataFrame) -> Quantity:
    """Calculate the separation between a primary and a secondary list of stars.

    Both dataframes must contain the keys 'ra' and 'dec'.

    Parameters
    ----------
    primary : pd.DataFrame
        The primary dataframe of stars. Must contain 'ra' and 'dec'.
    secondary : pd.DataFrame
        The secondary dataframe of stars. Must contain 'ra' and 'dec'. Must have the
        same length as 'primary'

    Returns
    -------
    separations: Quantity
        astropy Quantity array containing separations between stars in the two
        specified dataframes.

    Raises
    ------
    ValueError
        If 'ra' or 'dec' not in the columns of primary or secondary, or if there is a
        length mismatch.
    """
    # Checks
    if "ra" not in primary.columns or "dec" not in primary.columns:
        raise ValueError(
            "separation not specified, and will instead be calculated manually;"
            " however, required columns 'ra' and 'dec' are not in the columns "
            "of 'primary'."
        )
    if "ra" not in secondary.columns or "dec" not in secondary.columns:
        raise ValueError(
            "separation not specified, and will instead be calculated manually;"
            " however, required columns 'ra' and 'dec' are not in the columns "
            "of 'secondary'."
        )
    if len(primary) != len(secondary):
        raise ValueError(
            "primary and secondary star dataframes must have equal length."
        )

    # Create skycoords & calculate the sep
    coord_primary = SkyCoord(
        primary["ra"].to_numpy(), primary["dec"].to_numpy(), unit="deg"
    )
    coord_secondary = SkyCoord(
        secondary["ra"].to_numpy(), secondary["dec"].to_numpy(), unit="deg"
    )
    return coord_primary.separation(coord_secondary)


def apply_astrometric_errors_simple_gaussian(
    cluster: ocelot.simulate.cluster.SimulatedCluster,
    model: BaseObservation,
    columns: None | list[str] | tuple[str] = None,
):
    """Calculates astrometry sampled from a Gaussian error distribution and adds it
    as a column in the relevant observation.

    Parameters
    ----------
    cluster : ocelot.simulate.cluster.SimulatedCluster
        Simulated cluster to apply to.
    model : BaseObservation
        Current model being used.
    columns : None | list[str] | tuple[str], optional
        List or tuple of columns to apply the errors to. Default: None, in which case
        proper motion and parallax columns (if present) will have errors applied.
    """
    observation = cluster.observations[model.name]

    if columns is None:
        columns = []
        if model.has_parallaxes:
            columns.append("parallax")
        if model.has_proper_motions:
            columns.extend(["pmra", "pmdec"])

    for column in columns:
        observation[column] = cluster.random_generator.normal(
            loc=observation[column].to_numpy(),
            scale=observation[f"{column}_error"].to_numpy(),
        )


def apply_photometric_errors_simple_gaussian(
    cluster: ocelot.simulate.cluster.SimulatedCluster,
    model: BaseObservation,
    bands: None | list[str] | tuple[str] = None,
):
    """Calculates photometry sampled from a Gaussian error distribution and adds it
    as a column in the relevant observation.

    Parameters
    ----------
    cluster : ocelot.simulate.cluster.SimulatedCluster
        Simulated cluster to apply to.
    model : BaseObservation
        Current model being used.
    bands : None | list[str] | tuple[str], optional
        List or tuple of bands to apply the errors to. Default: None, in which case all
        bands in model.photometric_band_names have error applied.
    """
    if bands is None:
        bands = model.photometric_band_names

    observation = cluster.observations[model.name]

    for band in bands:
        new_fluxes = cluster.random_generator.normal(
            loc=model.mag_to_flux(observation[band].to_numpy(), band),
            scale=observation[f"{band}_flux_error"].to_numpy(),
        )
        observation[band] = model.flux_to_mag(new_fluxes, band)
