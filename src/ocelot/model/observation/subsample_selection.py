"""A generic subsample selection function class that can be used on many different
surveys.
"""

import numpy as np
import pandas as pd
from ocelot.model.observation._base import BaseSelectionFunction
from scipy.interpolate import interp1d
from ocelot.util.stats import variable_bin_histogram
from ocelot.util.stats import calculate_bin_centers


class GenericSubsampleSelectionFunction(BaseSelectionFunction):
    def __init__(
        self,
        data_all: pd.DataFrame,
        data_cut: pd.DataFrame,
        column: str,
        minimum_bin_width: float = 0.2,
        minimum_bin_size: int = 10,
        bounds_value: float = 0.0,
        range: tuple | None = None,
        column_in_data: str | None = None,
    ):
        """A generic subsample selection function, following the prescription from
        [1, 2] and using binomial statistics.

        By default, it uses a variable bin width histogram - set minimum_bin_size to
        zero to prevent this.

        Parameters
        ----------
        data_all : pd.DataFrame
            A dataset in the region of interest.
        data_cut : pd.DataFrame
            A dataset in the region of interest that
        column : str
            Column to create the selection function on.
        minimum_bin_width : float, optional
            Minimum width of each bin. Set higher to reduce the granularity of the
            selection function. Default: 0.2
        minimum_bin_size : int, optional
            Minimum number of stars in each bin. Default: 10
        bounds_value : float, optional
            Value of the selection function outside of the range of observed data.
            Default: 0.0
        range: array-like or None, optional
            Length-2 array of the minimum and maximum values of 'column' in data_all.
            Default: None, meaning that the range of data values is inferred from the
            data itself. This may be inappropriate with small datasets.
        column_in_data: string or None, optional
            Alternative column name to use for the datasets. If not specified, 'column'
            is used. Default: None

        References
        ----------
        [1] https://ui.adsabs.harvard.edu/abs/2021AJ....162..142R/abstract
        [2] https://ui.adsabs.harvard.edu/abs/2023A%26A...677A..37C/abstract
        """
        # Initial setup
        self.column = column
        self.column_in_data = self.column
        if column_in_data is not None:
            self.column_in_data = column_in_data
        self.bounds_value = bounds_value

        # Computations
        self._count, self._count_cut, self._bins = self._compute_histograms(
            data_all[self.column_in_data].to_numpy(),
            data_cut[self.column_in_data].to_numpy(),
            minimum_bin_width=minimum_bin_width,
            minimum_bin_size=minimum_bin_size,
            range=range,
        )
        self._probability, self._standard_deviation = (
            self._compute_binomial_probabilities()
        )
        self._bin_centers = calculate_bin_centers(self._bins)
        self._interpolator = self._setup_interpolator()

    def _compute_histograms(
        self,
        all: np.ndarray,
        cut: np.ndarray,
        minimum_bin_width: float = 0.2,
        minimum_bin_size: int = 10,
        range: tuple | None = None,
    ):
        if range is None:
            range = np.nanmin(all), np.nanmax(all)

        count, bins = variable_bin_histogram(
            all, range[0], range[1], minimum_bin_width, minimum_size=minimum_bin_size
        )
        count_cut, _ = np.histogram(cut, bins=bins)

        return count, count_cut, bins

    def _compute_binomial_probabilities(self):
        """Follows definition in Castro-Ginard+23 (see references in __init__.)"""
        probability = (self._count_cut + 1) / (self._count + 2)
        standard_deviation = np.sqrt(
            (self._count_cut + 1)
            * (self._count - self._count_cut + 1)
            / ((self._count + 2) ** 2 * (self._count + 3))
        )
        return probability, standard_deviation

    def _setup_interpolator(self):
        """Sets up the interpolator used by _query."""
        return interp1d(
            self._bin_centers,
            self._probability,
            bounds_error=False,
            fill_value=self.bounds_value,
        )

    def _query(self, observation: pd.DataFrame) -> np.ndarray:
        return self._interpolator(observation[self.column].to_numpy())
