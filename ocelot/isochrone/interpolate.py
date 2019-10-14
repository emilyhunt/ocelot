"""Classes and systems to handle interpolation of isochrones, given a grid of input points."""

from typing import Union, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf as RBFInterpolator


class IsochroneInterpolator:
    def __init__(self, data_isochrone: pd.DataFrame, parameters_as_arguments: Optional[list] = None,
                 parameters_to_infer: Optional[list] = None, interpolation_type: str = 'LinearND'):
        """Given an input in the format outputted by the CMD 3.3 web interface, this class will produce interpolated
        isochrones, marginalised over the specified argument_parameters. Yay!

        Todo: consider removing RadialBasisFunction interpolation, as it can't use duplicated co-ordinates of any kind
        which is... pretty crappy.
        See: https://stackoverflow.com/questions/27295853/rbf-interpolation-fails-linalgerror-singular-matrix

        Args:
            data_isochrone (pd.DataFrame): the data for the isochrones, in the same format as CMD 3.3 (so: a long
                fuckoff list of isochrones, all joined together.)
            parameters_as_arguments (list, optional): parameters to use as arguments in the interpolator (e.g. logAge
                for each isochrone).
                Default: None, which uses ['logZini', 'logAge'] as default.
            parameters_to_infer (str, optional): parameters to infer with the interpolator. There must be two due to the
                design of this implementation!
                Default: None, which uses ['G_BP-RP', 'Gmag'] as default.
            interpolation_type (str): type of interpolation to use.
                Implemented: 'LinearND' : scipy.interpolate.LinearNDInterpolator
                             'RadialBasisFunction' : scipy.interpolate.Rbf
                Default: 'LinearND'

        Returns:
            absolutely... NOTHING
            (call the class if you want to evaluate it)

        """
        # Use the default hard-coded set of parameter labels, if laziness is desired by the user
        if parameters_as_arguments is None:
            self.argument_parameters = ['logZini', 'logAge']
        else:
            self.argument_parameters = parameters_as_arguments
        if parameters_to_infer is None:
            self.inferred_parameters = ['G_BP-RP', 'Gmag']
        else:
            self.inferred_parameters = parameters_to_infer

        # Calculate the normalised distances along the curve
        data_isochrone = self.calculate_normalised_cumulative_distances_along_curve(data_isochrone)

        # Hence, grab the input data we'll need for interpolation
        input_data = data_isochrone[self.argument_parameters + ['cumulative_distance_along_curve']].to_numpy()

        # Interpolate each of the inferred_parameters as a function of the normalised distance parameter.
        if interpolation_type == 'LinearND':
            # Interpolate!!!
            self.x_interpolator = LinearNDInterpolator(input_data, data_isochrone[self.inferred_parameters[0]])
            self.y_interpolator = LinearNDInterpolator(input_data, data_isochrone[self.inferred_parameters[1]])

            # Set __call__ to use this function!
            self.evaluate = self.evaluate_linear_nd_interpolator

        elif interpolation_type == 'RadialBasisFunction':
            # Prepare the input for use with the * way of unpacking numpy axes
            input_data = input_data.T
            input_data_x = np.vstack((input_data, data_isochrone[self.inferred_parameters[0]].to_numpy()))
            input_data_y = np.vstack((input_data, data_isochrone[self.inferred_parameters[1]].to_numpy()))

            # Interpolate!!!
            self.x_interpolator = RBFInterpolator(*input_data_x)
            self.y_interpolator = RBFInterpolator(*input_data_y)

            # Set __call__ to use this function!
            self.evaluate = self.evaluate_rbf_interpolator

        else:
            raise ValueError('Specified interpolation method not recognised!')

    @staticmethod
    def sum_along_curve(x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series],
                        normalise: bool = True) -> Union[np.ndarray, pd.Series]:
        """Calculates differences between consecutive values along a curve and returns the cumulative sum! Assumes
        use of the Euclidean metric.

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

    def calculate_normalised_cumulative_distances_along_curve(self, data_isochrone: pd.DataFrame):
        """Calculates cumulative distances along an arbitrary curve and normalises them.

        Args:
            data_isochrone (pd.DataFrame): the data describing the isochrone(s) in the same output format as CMD 3.3.

        Returns:
            the modified data_isochrone, with cumulative distances under the key 'cumulative_distance_along_curve'.

        """

        # Grab the number of argument_parameters (this comment is self-evident)
        n_parameters = len(self.argument_parameters)

        if n_parameters != 0:

            # Initialise the new column as nans, so that an error will be thrown if any are missed
            data_isochrone['cumulative_distance_along_curve'] = np.nan

            # Grab all possible unique parameter combinations
            unique_parameter_combos = np.unique(data_isochrone[self.argument_parameters], axis=0)

            # Cycle over all possible parameter combinations
            for a_parameter_combination in unique_parameter_combos:
                stars_to_use = np.sum(data_isochrone[self.argument_parameters] == a_parameter_combination, axis=1)

                # We only want stars where all the parameters_are matched (the row-wise sum should equal n_parameters
                stars_to_use = np.asarray(stars_to_use == n_parameters).nonzero()

                # Hence, calculate the distance
                x = data_isochrone[self.inferred_parameters[0]].iloc[stars_to_use]
                y = data_isochrone[self.inferred_parameters[1]].iloc[stars_to_use]
                data_isochrone['cumulative_distance_along_curve'].iloc[stars_to_use] = self.sum_along_curve(x, y)

        # If there aren't any other parameters, then we just find the cumulative distance for the whole damn table
        else:
            x = data_isochrone[self.inferred_parameters[0]]
            y = data_isochrone[self.inferred_parameters[1]]
            data_isochrone['cumulative_distance_along_curve'] = self.sum_along_curve(x, y)

        return data_isochrone

    def evaluate_linear_nd_interpolator(self, input_data: np.ndarray) -> list:
        """Evaluation function in the case that the class's interpolator is a scipy.interpolate.LinearNDInterpolator.

        Args:
            input_data (np.ndarray): input in shape (n_points, n_dims)

        Returns:
            a list containing [x_data, y_data] for the requested points.

        """
        return [self.x_interpolator(input_data), self.y_interpolator(input_data)]

    def evaluate_rbf_interpolator(self, input_data: np.ndarray) -> list:
        """Evaluation function in the case that the class's interpolator is a scipy.interpolate.Rbf (radial basis
        function interpolator.)

        Args:
            input_data (np.ndarray): input in shape (n_points, n_dims)

        Returns:
            a list containing [x_data, y_data] for the requested points.

        """
        input_data = input_data.T
        return [self.x_interpolator(*input_data), self.y_interpolator(*input_data)]

    def __call__(self, input_data: Union[np.ndarray, pd.DataFrame],
                 range_to_evaluate: Union[str, np.ndarray] = 'full isochrone',
                 resolution: int = 100):
        """Evaluate the interpolator against a specific set of points. Contains systems to automate a lot of the process
        for you, because it's kind like that. =)

        Args:
            input_data (np.ndarray, pd.DataFrame): input arguments for isochrones to be evaluated at, with shape
                (n_points, n_dims). Every n_points will generate a full isochrone.
            range_to_evaluate (str, np.ndarray): array of values in the correct range (typically [0, 1]) to evaluate the
                isochrone at.
                Default: 'full isochrone', which computes a set of points in the range [0, 1] automatically.
            resolution (int): if range_to_evaluate is set to 'full isochrone', this controls how many points will be
                returned.

        Returns:
            a list containing [x_data, y_data] for the requested points.

        """
        if range_to_evaluate == 'full isochrone':
            range_to_evaluate = np.linspace(0, 1, resolution)

        points_per_isochrone = range_to_evaluate.size

        # Tile the range_to_evaluate to match the length of input_data's number of points
        range_to_evaluate = np.tile(range_to_evaluate, [input_data.shape[0]])

        # Process the function arguments into something feedable to the interpolator
        input_data = np.repeat(np.asarray(input_data), points_per_isochrone, axis=0)
        input_data = np.hstack((input_data, np.expand_dims(range_to_evaluate, axis=1)))

        return self.evaluate(input_data)


class IsochroneDistanceInterpolator:
    def __init__(self):
        print(1)

    def __call__(self):
        pass
