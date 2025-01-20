import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod


class BaseDifferentialReddeningModel(ABC):
    accepts_random_seed = False  # Class attribute for whether or not a given model needs a seed in __init__

    def extinction(
        self,
        x: ArrayLike,
        y: ArrayLike,
        mean: float,
        width: float,
    ) -> np.ndarray:
        """Calculates differential extinction across a cluster as a function of the
        star's position in X and Y (arbitrary units) as viewed by the observer. Note
        that not all versions of this class will produce spatially correlated
        differential extinction.
        """
        x, y = _normalize_within_0_and_1(x), _normalize_within_0_and_1(y)
        extinctions = self._differential_extinction(x, y, width)
        return np.clip(extinctions + mean, 0, np.inf)

    @abstractmethod
    def _differential_extinction(
        self, x: np.ndarray, y: np.ndarray, width: float
    ) -> np.ndarray:
        """Calculates differential extinction across a cluster as a function of the
        star's position in X and Y (arrays in the range [0, 1] as viewed by the
        observer. Returns values scaled by width.
        """
        pass


def _normalize_within_0_and_1(array: ArrayLike) -> np.ndarray:
    """Noramlizes values within an array to be between 0 and 1, as well as flattening."""
    array = np.atleast_1d(array).flatten()
    return (array - np.nanmin(array)) / np.ptp(array)
