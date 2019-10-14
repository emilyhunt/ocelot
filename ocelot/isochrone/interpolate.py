"""Classes and systems to handle interpolation of isochrones, given a grid of input points."""

from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf as RBFInterpolator


class IsochroneInterpolator:
    def __init__(self, data_isochrone: pd.DataFrame, parameters: Union[str, list, tuple] = 'Default',
                 parameters_to_infer: str = 'Default', interpolation_type: str = 'RadialBasisFunction'):
        """Given an input """
        # Use the default hard-coded set of parameter labels, if laziness is desired by the user
        if parameters == 'Default':
            self.parameters = ['Zini', 'logAge']
        if parameters_to_infer == 'Default':
            self.parameters_to_infer = ['Gmag', 'G_BP-RP']

        # Grab some lists of the parameters we need as arguments and the parameters we need to infer separately
        parameters_as_arguments = parameters[:-2]
        parameters_to_infer = parameters[-2:]

        # I may need something like:
        # args = [np.arange(10), np.arange(10, 20)]
        # np.meshgrid(*args)

        # Set the interpolation type based on user input
        if interpolation_type == 'RadialBasisFunction':
            interpolator = RBFInterpolator
        elif interpolation_type == 'LinearND':
            interpolator = LinearNDInterpolator
        else:
            raise ValueError('Specified interpolation method not recognised!')

    @staticmethod
    def sum_along_curve(x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series],
                        normalise: bool = True) -> Union[np.ndarray, pd.Series]:
        """Sums values along a curve and returns the cumulative sum!

        Args:
            x (np.ndarray, pd.Series): x data for the curve.
            y (np.ndarray, pd.Series): y data for the curve.
            normalise (bool): whether or not to normalise the resulting cumulative sum.
                Default: True

        Returns:
            a np.ndarray or a pd.Series, depending on the input type.
        """

        # Calculate the difference between consecutive points
        x_difference = np.diff(x)
        y_difference = np.diff(y)

        # Calculate the Euclidean distance between each point
        distance_between_points = np.sqrt(x_difference**2 + y_difference**2)
        cumulative_distance_between_points = np.cumsum(distance_between_points)

        # Add a zero at the start because diff reduces the length by 1
        cumulative_distance_between_points = np.insert(cumulative_distance_between_points, 0, 0)

        # Normalise the distances to the range [0, 1] if desired (this makes later use a lot easier)
        if normalise:
            cumulative_distance_between_points = (cumulative_distance_between_points
                                                  / np.max(cumulative_distance_between_points))

        return cumulative_distance_between_points

    def calculate_normalised_curve_distances(self, data_isochrone: pd.DataFrame):
        """Calculates cumulative distances along an arbitrary curve and normalises them."""

        # Create a dictionary of all unique parameter values we've been given, and populate the dictionary with a loop
        # over all parameters

        if len(self.parameters) != 0:
            unique_parameter_combos = np.unique(data_isochrone[self.parameters], axis=0)

            for


    def evaluate(self):
        # Todo: this could be __call__ instead?
        pass


class IsochroneDistanceInterpolator:
    def __init__(self):
        print(1)

    def evaluate(self):
        pass
