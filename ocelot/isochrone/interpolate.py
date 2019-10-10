"""Classes and systems to handle interpolation of isochrones, given a grid of input points."""

from typing import Union

import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf as RBFInterpolator


def calculate_normalised_curve_distances(x, y):
    """Calculates cumulative distances along an arbitrary curve and normalises them."""


class IsochroneInterpolator:
    def __init__(self, data_isochrone: pd.DataFrame, parameters: Union[str, list, tuple] = 'Default',
                 parameters_to_infer: int = -2, interpolation_type: str = 'RadialBasisFunction'):
        """Given an input """
        # Use the default hard-coded set of parameter labels, if laziness is desired by the user
        if parameters == 'Default':
            parameters = ['Zini', 'logAge', 'Gmag', 'G_BP-RP']

        # Grab some lists of the parameters we need as arguments and the parameters we need to infer separately
        parameters_as_arguments = parameters[:-2]
        parameters_to_infer = parameters[-2:]

        # I may need something like:
        # args = [np.arange(10), np.arange(10, 20)]
        # np.meshgrid(*args)

        # Set the interpolation type based on user input
        if interpolation_type == 'RadialBasisFunction':
            interpolator = RBFInterpolator
        elif interpolation_type == 'LinearNDInterpolator':
            interpolator = LinearNDInterpolator
        else:
            raise ValueError('Specified interpolation method not recognised!')


    def evaluate(self):
        pass


class IsochroneDistanceInterpolator:
    def __init__(self):
        print(1)

    def evaluate(self):
        pass
