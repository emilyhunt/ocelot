import numpy as np
from abc import ABC, abstractmethod


class BinaryStarModel(ABC):
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


class BinaryStarModelWithPeriods(BinaryStarModel):
    """An extension of BinaryStarModel for models that also include period modelling,
    allowing for binaries to be classed as resolved or unresolved.
    """

    @abstractmethod
    def random_binary(self, masses: np.ndarray, seed=None) -> tuple[np.ndarray]:
        """Returns a random binary star mass ratio q and a period in days."""
        pass
