import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from astropy.units import Quantity, Unit


class BaseClusterDistributionModel(ABC):
    def _input_is_0d(self, input):
        if isinstance(input, float) or isinstance(input, int):
            return True
        return False

    def _convert_quantity_to_float(self, input):
        unit = None
        if isinstance(input, Quantity):
            unit = input.unit
            input = input.value
        return input, unit

    def _convert_float_to_quantity(self, output, unit=None):
        if unit is None:
            unit = self.base_unit
        return output * unit

    def pdf(self, x: int | float | ArrayLike) -> np.ndarray | float:
        """Probability density function of the model."""
        should_be_1d = self._input_is_0d(x)
        x, unit = self._convert_quantity_to_float(x)
        x = np.atleast_1d(x).astype(float)

        result = self._pdf(x)

        result = self._convert_float_to_quantity(result, unit)
        if should_be_1d:
            return result[0]
        return result

    def cdf(self, x: int | float | ArrayLike) -> np.ndarray | float:
        """Cumulative density function of the model."""
        should_be_1d = self._input_is_0d(x)
        x, unit = self._convert_quantity_to_float(x)
        x = np.atleast_1d(x).astype(float)

        result = self._cdf(x)

        result = self._convert_float_to_quantity(result, unit)
        if should_be_1d:
            return result[0]
        return result

    def rvs(
        self, size: int | np.int_ = 1, seed: None | int | np.random.Generator = None
    ) -> np.ndarray | float:
        """Sample a random variate from the model. Returns an array of shape
        (n_stars, n_dims), or a single float if size is 1.
        """
        if not (isinstance(size, int) or isinstance(size, np.int_)):
            raise ValueError("Input argument size must be an integer.")
        if size <= 0:
            raise ValueError(
                "Cannot specify a negative or zero number of stars to sample."
            )

        result = self._rvs(size, seed=seed)

        result = self._convert_float_to_quantity(result)
        if size == 1:
            return result[0]
        return result

    @property
    @abstractmethod
    def base_unit(self) -> Unit:
        """Property defining the basic unit of the model. Returned whenever possible."""
        pass

    @abstractmethod
    def _pdf(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """PDF method. Intended to be overwritten."""
        pass

    @abstractmethod
    def _cdf(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """PDF method. Intended to be overwritten."""
        pass

    @abstractmethod
    def _rvs(self, size: int, seed=None) -> np.ndarray[float]:
        """PDF method. Intended to be overwritten."""
        pass


class Implements1DMethods(ABC):
    """Empty interface that serves as an indicator that the methods of a class implement
    methods in 1D (i.e. with a single r coordinate.)
    """

    pass


class Implements2DMethods(ABC):
    """Empty interface that serves as an indicator that the methods of a class implement
    methods in 1D (i.e. with x,y coordinates.)
    """

    pass


class Implements3DMethods(ABC):
    """Empty interface that serves as an indicator that the methods of a class implement
    methods in 1D (i.e. with x,y,z coordinates.)
    """

    pass
