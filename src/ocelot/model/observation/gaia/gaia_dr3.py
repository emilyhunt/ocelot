"""Main class defining an observation made with Gaia DR3."""

from __future__ import annotations
from ocelot.model.observation._base import BaseObservation
from ocelot.simulate import SimulatedCluster
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from .photutils import AG, ABP, ARP


class GaiaDR3ObservationModel(BaseObservation):
    # Zeropoints in the Vegamag system (see documentation table 5.2)
    # These are for Gaia DR3!
    ZEROPOINTS = dict(G_ZP=25.6874, BP_ZP=25.3385, RP_ZP=24.7479)

    def __init__(
        self,
        representative_stars: pd.DataFrame | None = None,
        subsample_selection_function: str = "",
    ):
        """A model for an observation made with Gaia DR3."""
        self.representative_stars = representative_stars
        self.subsample_selection_function = subsample_selection_function
        self.matching_stars = None
        self.stars_to_assign = None

        # Todo support other error models, like Anthony Brown's package
        # Todo refactor this way of doing errors to be less tied to this one class
        if self.representative_stars is None:
            raise ValueError(
                "Must set 'representative_stars' parameter of this class to apply photometric errors."
            )

        self.representative_stars = self.representative_stars.loc[
            np.logical_and.reduce(
                (
                    self.representative_stars["phot_g_mean_mag"].notna(),
                    self.representative_stars["phot_bp_mean_mag"].notna(),
                    self.representative_stars["phot_rp_mean_mag"].notna(),
                    self.representative_stars["parallax_error"].notna(),
                    # field["pmra_error"].notna(),  # Probably not needed
                    # field["pmdec_error"].notna(),  # Probably not needed
                )
            )
        ]

    @property
    def name(self) -> str:
        """Type of observation modelled by this class.

        Should return a lowercase string, like 'gaia_dr3'.
        """
        return "gaia_dr3"

    @property
    def photometric_band_names(self) -> list[str]:
        """Names of photometric bands modelled by this system."""
        return ["gaia_dr3_g", "gaia_dr3_bp", "gaia_dr3_rp"]

    @property
    def has_proper_motions(self) -> bool:
        return True

    @property
    def has_parallaxes(self) -> bool:
        return True

    def apply_photometric_errors(self, cluster: SimulatedCluster):
        """Apply photometric errors to a simulated cluster."""
        if self.matching_stars is None:
            self.matching_stars, self.stars_to_assign = _closest_gaia_star(
                cluster.observations["gaia_dr3"], self.representative_stars
            )
        observation = cluster.observations["gaia_dr3"]
        observation.loc[self.stars_to_assign, "gaia_dr3_g_flux_error"] = (
            self.matching_stars["phot_g_mean_flux_error"].to_numpy()
        )
        observation.loc[self.stars_to_assign, "gaia_dr3_bp_flux_error"] = (
            self.matching_stars["phot_bp_mean_flux_error"].to_numpy()
        )
        observation.loc[self.stars_to_assign, "gaia_dr3_rp_flux_error"] = (
            self.matching_stars["phot_rp_mean_flux_error"].to_numpy()
        )

    def apply_astrometric_errors(self, cluster: SimulatedCluster):
        """Apply astrometric errors to a simulated cluster."""
        if self.matching_stars is None:
            self.matching_stars, self.stars_to_assign = _closest_gaia_star(
                cluster.observations["gaia_dr3"], self.representative_stars
            )
        observation = cluster.observations["gaia_dr3"]
        observation.loc[self.stars_to_assign, "pmra_error"] = (
            self.matching_stars["pmra_error"].to_numpy()
        )
        observation.loc[self.stars_to_assign, "pmdec_error"] = (
            self.matching_stars["pmdec_error"].to_numpy()
        )
        observation.loc[self.stars_to_assign, "parallax_error"] = (
            self.matching_stars["parallax_error"].to_numpy()
        )

    def apply_selection_function(self, cluster: SimulatedCluster):
        """Apply a selection function to a simulated cluster."""
        pass

    def apply_extinction(self, cluster: SimulatedCluster):
        """Applies extinction in a given photometric band observed in this dataset."""
        for band, func in zip(self.photometric_band_names, (AG, ABP, ARP)):
            cluster.cluster[f"extinction_{band}"] = func(
                cluster.cluster["extinction"], cluster.cluster["Teff"]
            )

    def mag_to_flux(
        self, magnitude: int | float | ArrayLike, band: str
    ) -> int | float | ArrayLike:
        """Convert a magnitude in some band into a flux in some band."""
        self._check_band_name(band)
        magnitude = np.atleast_1d(magnitude)
        return 10 ** (
            (self.ZEROPOINTS[band] - magnitude) / 2.5
        )  # todo actually always returns a np.ndarray

    def flux_to_mag(
        self, flux: int | float | ArrayLike, band: str
    ) -> int | float | ArrayLike:
        """Convert a flux in some band into a magnitude in some band."""
        self._check_band_name(band)
        flux = np.atleast_1d(flux)
        # We safely handle negative fluxes - they're set to inf
        good_fluxes = flux > 0
        magnitude = -2.5 * np.log10(flux, where=good_fluxes) + self.ZEROPOINTS[band]
        magnitude[np.invert(good_fluxes)] = np.inf
        return magnitude  # todo actually always returns a np.ndarray

    def _check_band_name(self, band: str):
        if band not in self.ZEROPOINTS:
            raise ValueError(
                f"band {band} is not the correct name of a photometric band modelled "
                "in this observation."
            )


def _closest_gaia_star(observation: pd.DataFrame, field: pd.DataFrame):
    """Finds the nearest star in G-band magnitude to a given star."""
    field_magnitudes = field["phot_g_mean_mag"].to_numpy()
    stars_to_assign = observation["gaia_dr3_g_true"].notna()
    cluster_magnitudes = observation.loc[stars_to_assign, "gaia_dr3_g_true"].to_numpy()

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
    matching_stars = field.iloc[best_matching_stars]

    # Also record source ids
    observation["matching_gaia_dr3_source_id"] = -1
    observation.loc[stars_to_assign, "matching_gaia_dr3_source_id"] = matching_stars[
        "source_id"
    ].to_numpy()
    return matching_stars, stars_to_assign
