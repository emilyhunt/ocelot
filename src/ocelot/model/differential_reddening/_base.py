import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod


class BaseDifferentialReddeningModel(ABC):
    @abstractmethod
    def extinction(
        self, mean: float, width: float, x: ArrayLike, y: ArrayLike, rng=None
    ) -> np.ndarray:
        """Calculates differential extinction across a cluster as a function of the
        star's position in X and Y (arbitrary units) as viewed by the observer. Note
        that not all versions of this class will produce spatially correlated
        differential extinction.
        """
        pass
