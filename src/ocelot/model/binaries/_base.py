import numpy as np
from abc import ABC, abstractmethod


class BaseBinaryStarModel(ABC):
    """A binary star model abstract base class for models that implement MF, CSF, and q."""

    @abstractmethod
    def multiplicity_fraction(self, masses: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def companion_star_frequency(self, masses: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def random_mass_ratio(self, masses: np.ndarray, seed=None) -> np.ndarray:
        """Returns a random binary star mass ratio q."""
        pass


class BaseBinaryStarModelWithPeriods(BaseBinaryStarModel):
    """An extension of BinaryStarModel for models that also include period modelling,
    allowing for binaries to be classed as resolved or unresolved.
    """

    @abstractmethod
    def random_binary(
        self, masses: np.ndarray, seed=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a random binary star mass ratio q and a period in days."""
        pass


class BaseBinaryStarModelWithEccentricities(BaseBinaryStarModelWithPeriods):
    """An extension of BinaryStarModelWithPeriods for models that also include
    eccentricities, allowing for binaries to be classed as resolved or unresolved.
    """

    @abstractmethod
    def random_binary(
        self, masses: np.ndarray, seed=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns a random binary star mass ratio q, a period in days, and an 
        eccentricity.
        """
        pass
