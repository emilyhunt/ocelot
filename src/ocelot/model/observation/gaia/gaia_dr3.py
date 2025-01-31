"""Main class defining an observation made with Gaia DR3."""

from __future__ import annotations
from ocelot.model.observation._base import (
    BaseObservation,
    BaseSelectionFunction,
)
import ocelot.simulate.cluster
from scipy.interpolate import interp1d
from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from .photutils import AG, ABP, ARP
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy import units as u


class GaiaDR3ObservationModel(BaseObservation):
    # Zeropoints in the Vegamag system (see documentation table 5.2)
    # These are for Gaia DR3!
    ZEROPOINTS = dict(gaia_dr3_g=25.6874, gaia_dr3_bp=25.3385, gaia_dr3_rp=24.7479)

    def __init__(
        self,
        representative_stars: pd.DataFrame | None = None,
        subsample_selection_functions: tuple[BaseSelectionFunction] = tuple(),
        overestimate_bp_rp_fluxes: bool = True,
    ):
        """A model for an observation made with Gaia DR3."""
        self.representative_stars = representative_stars
        self.subsample_selection_functions = subsample_selection_functions
        self.overestimate_bp_rp_fluxes = overestimate_bp_rp_fluxes
        self.matching_stars = None
        self.stars_to_assign = None
        self.simulated_cluster = None  # To prevent it being removed # Todo: somehow stop issues with model not being re-assignable

        # Todo support other error models, like Anthony Brown's package
        # Todo refactor this way of doing errors to be less tied to this one class
        if self.representative_stars is None:
            raise ValueError(
                "Must set 'representative_stars' parameter of this class to apply photometric errors."
            )

        # Todo should raise an error if there are missing columns that we need
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

    def calculate_photometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Calculate photometric errors for a simulated cluster."""
        self._assert_simulated_cluster_not_reused(cluster)
        if self.matching_stars is None:
            self.matching_stars, self.stars_to_assign = _closest_gaia_star(
                cluster.observations["gaia_dr3"], self.representative_stars
            )

        for band in ("g", "bp", "rp"):
            cluster.observations["gaia_dr3"].loc[
                self.stars_to_assign, f"gaia_dr3_{band}_flux_error"
            ] = self.matching_stars[f"phot_{band}_mean_flux_error"].to_numpy()

    def apply_photometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Custom method to apply photometric errors to a simulated cluster.

        Method incorporates the underestimated BP and RP flux measurement issue in DR3.

        Follows things discussed in Riello+21, section 8.1.
        """
        observation = cluster.observations["gaia_dr3"]

        # Calculate true flux
        fluxes = {}
        for band in self.photometric_band_names:
            fluxes[band] = self.mag_to_flux(observation[band].to_numpy(), band)

        # For BP and RP, count how many times Gaia would have observed the star, and
        # then reverse-apply the flux calculation mistake in Gaia DR3
        if self.overestimate_bp_rp_fluxes:
            for band in ["gaia_dr3_bp", "gaia_dr3_rp"]:
                fluxes = self._apply_incorrect_flux_summing_to_flux(
                    observation, fluxes, band
                )

        # Now, finally, we can apply photometric errors from other sources & move on!
        for band in self.photometric_band_names:
            new_fluxes = cluster.random_generator.normal(
                loc=fluxes[band],
                scale=observation[f"{band}_flux_error"].to_numpy(),
            )
            observation[band] = self.flux_to_mag(new_fluxes, band)

    def _apply_incorrect_flux_summing_to_flux(self, observation, fluxes, band):
        """Method incorporates the underestimated BP and RP flux measurement issue in
        DR3. Follows things discussed in Riello+21, section 8.1, using values in
        Table 4.
        """
        faint_stars_with_potential_issue = np.logical_and(
            observation[band].to_numpy() >= 19, np.isfinite(fluxes[band])
        )
        if faint_stars_with_potential_issue.sum() == 0:
            return fluxes

        # Riello+21 Table 4
        # First values are calculated - last values are a guess lol
        _true_flux = np.asarray([343.08, 139.90, 56.28, 23.34, 10.15, 5.60, 1.0])
        _wrong_flux = np.asarray([343.08, 140.74, 62.36, 40.02, 34.70, 31.25, 29.5])

        # Update fluxes with interpolated values
        interpolator = interp1d(
            _true_flux,
            _wrong_flux,
            kind="cubic",
            bounds_error=False,
            fill_value=(_wrong_flux[-1], _wrong_flux[0]),
        )
        fluxes[band][faint_stars_with_potential_issue] = interpolator(
            fluxes[band][faint_stars_with_potential_issue]
        )

        return fluxes

    def calculate_astrometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Calculate astrometric errors for a simulated cluster."""
        self._assert_simulated_cluster_not_reused(cluster)
        if self.matching_stars is None:
            self.matching_stars, self.stars_to_assign = _closest_gaia_star(
                cluster.observations["gaia_dr3"], self.representative_stars
            )

        for column in ("pmra_error", "pmdec_error", "parallax_error"):
            cluster.observations["gaia_dr3"].loc[self.stars_to_assign, column] = (
                self.matching_stars[column].to_numpy()
            )

    def get_selection_functions(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Get an initialized GaiaDR3SelectionFunction in addition to any subsample
        selection functions defined by the user.
        """
        self._assert_simulated_cluster_not_reused(cluster)
        gaia = GaiaDR3SelectionFunction(
            SkyCoord(
                cluster.parameters.ra, cluster.parameters.dec, frame="icrs", unit="deg"
            )
        )
        return [gaia] + list(self.subsample_selection_functions)

    def calculate_extinction(self, cluster: ocelot.simulate.cluster.SimulatedCluster):
        """Applies extinction in a given photometric band observed in this dataset."""
        observation = cluster.observations["gaia_dr3"]

        for band, func in zip(self.photometric_band_names, (AG, ABP, ARP)):
            observation[f"extinction_{band}"] = func(
                observation["extinction"], observation["temperature"]
            )

    def calculate_resolving_power(
        self,
        primary: pd.DataFrame,
        secondary: pd.DataFrame,
        separation: Quantity,
    ) -> np.ndarray[float]:
        """Calculates the probability that a given pair of stars would be separately
        resolved."""
        # Todo currently very simplistic
        separation = separation.to(u.arcsec).value
        return np.where(separation >= 0.6, 1.0, 0.0)

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

    def _assert_simulated_cluster_not_reused(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        # Todo this is bad and should be improved lol - model should be cluster-agnostic
        if self.simulated_cluster is None:
            self.simulated_cluster = cluster
            return
        if cluster is not self.simulated_cluster:
            raise RuntimeError(
                "Gaia DR3 model may not be reused on different clusters!"
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


class GaiaDR3SelectionFunction(BaseSelectionFunction):
    def __init__(
        self,
        coordinate: SkyCoord,
        resolution: int = 500,
        g_range: tuple | list = (2, 22),
    ):
        """Gaia DR3 selection function. Based on

        Parameters
        ----------
        coordinate : astropy.coordinates.SkyCoord
            Coordinate to query the selection function at. Must have length one!
        resolution : int, optional
            Resolution of the selection function interpolator. Default: 500
        g_range : tuple or list, optional
            Range of values in G magnitude to sample, from min to max. Default: (2, 22).
        """
        self._selection_function = DR3SelectionFunctionTCG()
        self._coodinate = coordinate
        if coordinate.size > 1:
            raise ValueError(
                "You must specify exactly one coordinate to sample the selection "
                "function at!"
            )

        # Repeat resolution times
        coordinates = SkyCoord(
            np.repeat(coordinate.ra.value, resolution),
            np.repeat(coordinate.dec.value, resolution),
            frame="icrs",
            unit="deg",
        )

        # Query & setup
        # Todo check that values 0 and 22 don't give stupid results
        self._magnitudes = np.linspace(g_range[0], g_range[1], num=resolution)
        self._probability = self._selection_function.query(
            coordinates, self._magnitudes
        )
        self._interpolator = interp1d(
            self._magnitudes,
            self._probability,
            bounds_error=False,
            fill_value=(1.0, 0.0),  # Since this sf is always 1.0 at high mags
        )

    def _query(self, observation: pd.DataFrame) -> np.ndarray:
        return self._interpolator(observation["gaia_dr3_g"].to_numpy())
