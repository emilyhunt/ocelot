from __future__ import annotations  # Necessary to not get circular import on type hints
import ocelot.simulate.cluster
from ocelot.model.observation.common import (
    apply_astrometric_errors_simple_gaussian,
    apply_photometric_errors_simple_gaussian,
)

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from astropy.units import Quantity


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
    def calculate_photometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Calculate photometric errors and save them to the observation."""
        pass

    @abstractmethod
    def calculate_astrometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Calculate astrometric errors and save them to the observation."""
        pass

    def apply_photometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Apply photometric errors to the cluster and save observed photometry to the
        cluster. 
        
        calculate_photometric_errors should be called before this method.
        
        By default, this is just a simple Gaussian; it may be overwritten if desired.
        """
        apply_photometric_errors_simple_gaussian(
            cluster, self, self.photometric_band_names
        )

    def apply_astrometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Apply astrometric errors to the cluster and save observed astrometry to the
        cluster. 
        
        calculate_astrometric_errors should be called before this method.
        
        By default, this is just a simple UNCORRELATED Gaussian; it may be overwritten 
        if desired. In addition, this method only works on proper motions and
        parallaxes; if you're writing an observation class that also needs to simulate
        positional errors, then you should overwrite this function.
        """
        apply_astrometric_errors_simple_gaussian(
            cluster, self, self.photometric_band_names
        )

    @abstractmethod
    def get_selection_functions(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ) -> list[BaseSelectionFunction]:
        """Fetch all selection functions associated with this observation."""
        pass

    @abstractmethod
    def calculate_resolving_power(
        self,
        primary: pd.DataFrame,
        secondary: pd.DataFrame,
        separation: Quantity,
    ) -> np.ndarray:
        """Calculates the probability that a given pair of stars would be separately
        resolved. Overwrite me please!"""
        pass

    @abstractmethod
    def calculate_extinction(self, cluster: ocelot.simulate.cluster.SimulatedCluster):
        """Calculate extinction in all photometric bands and saves them to the
        observation.
        """
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


class BaseSelectionFunction(ABC):
    def query(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster, observation: str
    ) -> str:
        """Query a selection function. Assigns a column called
        'selection_probability_NAME' to the dataframe, along with the column name.
        """
        observation_df = cluster.observations[observation]

        # Think of a name for this column, and ensure that it's unique if we have
        # multiple with the same name
        original_column = f"selection_probability_{type(self).__name__}"
        column = original_column
        i = 1
        while column in observation_df.columns:
            column = f"{original_column}_{i}"
            i += 1

        # Calculate sf & assign
        observation_df[column] = self._query(observation_df)
        return column

    @abstractmethod
    def _query(self, observation: pd.DataFrame) -> np.ndarray:
        """Query a selection function. Returns a numpy array containing the probability
        of detecting a given star.
        """
        pass
