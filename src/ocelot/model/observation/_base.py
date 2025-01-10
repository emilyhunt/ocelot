from __future__ import annotations  # Necessary to not get circular import on type hints
from ocelot.simulate import SimulatedCluster

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike


class BaseObservation(ABC):
    """Class defining a model to simulate an observation by some telescope."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Type of observation modelled by this class.

        Should return a lowercase string, like 'gaia_dr3'.
        """
        pass

    @property
    @abstractmethod
    def photometric_band_names(self) -> list[str]:
        """Names of photometric bands modelled by this system.

        Should return a list of strings, like ['gaia_dr3_g', 'gaia_dr3_bp'].
        """
        pass

    @property
    @abstractmethod
    def has_proper_motions(self) -> bool:
        """Boolean flag indicating whether or not a given dataset includes proper
        motions.
        """
        pass

    @property
    @abstractmethod
    def has_parallaxes(self) -> bool:
        """Boolean flag indicating whether or not a given dataset includes parallaxes."""
        pass

    @abstractmethod
    def apply_photometric_errors(self, cluster: SimulatedCluster):
        """Apply photometric errors to a simulated cluster."""
        pass

    @abstractmethod
    def apply_astrometric_errors(self, cluster: SimulatedCluster):
        """Apply astrometric errors to a simulated cluster."""
        pass

    @abstractmethod
    def apply_selection_function(self, cluster: SimulatedCluster):
        """Apply a selection function to a simulated cluster."""
        pass

    @abstractmethod
    def apply_extinction(self, cluster: SimulatedCluster):
        """Applies extinction in a given photometric band observed in this dataset."""
        pass

    @abstractmethod
    def mag_to_flux(
        self, magnitude: int | float | ArrayLike, band: str
    ) -> int | float | ArrayLike:
        """Convert a magnitude in some band into a flux in some band."""
        pass

    @abstractmethod
    def flux_to_mag(
        self, flux: int | float | ArrayLike, band: str
    ) -> int | float | ArrayLike:
        """Convert a flux in some band into a magnitude in some band."""
        pass
