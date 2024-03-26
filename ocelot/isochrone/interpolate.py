"""Classes and systems to handle interpolation of isochrones, given a grid of input points."""

from typing import Union, Optional, List

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf as RBFInterpolator
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from sklearn.neighbors import NearestNeighbors


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
    distance_between_points = np.sqrt(x_difference ** 2 + y_difference ** 2)
    cumulative_distance_between_points = np.cumsum(distance_between_points)

    # Add a zero at the start because diff reduces the length by 1
    cumulative_distance_between_points = np.insert(cumulative_distance_between_points, 0, 0)

    # Normalise the distances to the range [0, 1] if desired (this makes later use a lot easier)
    if normalise:
        cumulative_distance_between_points = (cumulative_distance_between_points
                                              / np.max(cumulative_distance_between_points))

    return cumulative_distance_between_points


def find_nearest_point(points_to_match: np.ndarray, points_on_line: np.ndarray,
                       return_raw_distances: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
    """Given a first and a second set of points, this function will find which second point is nearest to each first
    point, and return an array detailing these findings with the same shape as the first set of points.

    Args:
        points_to_match (np.ndarray): points to match to the line. Should have shape (n_match_points, n_features).
        points_on_line (np.ndarray): points to cross-match the points_to_match with. Should have shape
            (n_line_points, n_features).
        return_raw_distances (bool): whether to return the raw distances to each point as well or not.

    Returns:
        return_raw_distances = False:
            An array of length n_match_points where each value is the nearest point in points_on_line to that
            point_to_match.
        return_raw_distances = True:
            Same as above, but also the raw distances too (tagged on as the second item in a list).

    """
    # Raise an error if one of the arrays isn't 2D
    if len(points_to_match.shape) != 2 or len(points_on_line.shape) != 2:
        raise ValueError('Input arrays must be two dimensional.')

    # Raise an error if the last shape of the arrays isn't the same
    if points_to_match.shape[1] != points_on_line.shape[1]:
        raise ValueError('Number of features mismatch between points_to_match and points_on_line.')

    # Given that points_to_match has shape (a, n) and points_on_line has shape (b, n), we tile both arrays to have shape
    # (b, a, n) so that they can be acted on together.
    points_on_line_tiled = np.tile(points_on_line, (points_to_match.shape[0], 1, 1))
    points_to_match_tiled = (np.repeat(points_to_match, points_on_line.shape[0], axis=0)
                             .reshape(points_on_line_tiled.shape))

    # Then, we find the Euclidean distance between each point, and use the smallest one to match points to the line.
    raw_distances = points_to_match_tiled - points_on_line_tiled
    raw_distances_squared = np.sum(raw_distances**2, axis=2)
    closest_point = np.argmin(raw_distances_squared, axis=1)

    # Return raw distances to each closest point too if desired
    # Todo: is this still needed? It isn't really if I still sort
    if return_raw_distances:
        return [closest_point, raw_distances[np.arange(points_to_match.shape[0]), closest_point, :]]
    else:
        return closest_point


def proximity_to_line_sort(points_to_match: np.ndarray, points_on_line: np.ndarray) -> np.ndarray:
    """Given a field of unorganised points and an ordered list of points defining a line, this function returns the
    field of points sorted by order of how close to the points on the line they are. The final step is to sort matches
    to the same point based on their last axis distance from the point (in 2D, that will be the y axis.)

    Todo: is it ok to just oversample this, or should I improve the final sorting scoring step? This isn't an issue if
        the line has enough points, but that may often not be the case...
        Instead, it could use the derivative of the points on the line to work out which way the line is going, and then
        use that to work out if points_to_match are above or below the direction of travel. Fucking hard though!

    Notes:
        - This function was made to sort unordered CMDs.
        - Make sure the line has enough points, as the final sorting step between points matched to the same line point
            is ONLY done on y values, which will cause icky discontinuities if the line doesn't have a high enough
            resolution.

    Args:
        points_to_match (np.ndarray): points to match to the line. Should have shape (n_match_points, n_features).
        points_on_line (np.ndarray): points to cross-match the points_to_match with. Should have shape
            (n_line_points, n_features).

    Returns:
        The optimum set of arguments to sort the set of points along the line.

    """
    # Get both the nearest point to the line *and* how far away from that point it is
    closest_point, raw_distances = find_nearest_point(points_to_match, points_on_line, return_raw_distances=True)

    # Grab all the y distances from the points, and normalise them to be in the range [-0.495, 0.495]
    y_distances = raw_distances[:, -1]
    min_y = np.min(y_distances)
    max_y = np.max(y_distances)
    normalised_y_distances = ((y_distances - min_y) / (max_y - min_y) - 0.5) * 0.99

    # Add the y distances to the closest point ordering, and then sort the points given that we're sorting on both point
    # along the line (integer steps) and how above/below the point each one is (float steps)
    sort_args = np.argsort(closest_point + normalised_y_distances)

    return sort_args


def nn_graph_sort(x, y):
    """Sorts data based on a nearest neighbour graph.

    # Todo: still needed?

    Note:
        - Currently always starts at the lowest value.
        - A solution to my CMDInterpolator sorting problem from:
            https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
        - MAY NOT RETURN A GRAPH INCLUDING ALL POINTS - this solution is quite basic unfortunately and does NOT work for
            noisy data.

    Args:
        x (np.ndarray): x of points to sort.
        y (np.ndarray): y of points to sort.

    Returns:
        The optimum set of arguments to sort the path.


    """
    import networkx as nx

    # Order stuff into a 2D array
    points = np.column_stack([x, y])

    # Create a graph (as a sparse matrix), where each node is connected to its nearest neighbours:
    clf = NearestNeighbors(2).fit(points)
    graph = clf.kneighbors_graph()
    nx_graph = nx.from_scipy_sparse_matrix(graph)

    # Extract the best paths between all the points
    # Todo optimise this steaming pile of for-loop dogshit!!!1!

    list_of_all_paths = [list(nx.dfs_preorder_nodes(nx_graph, source=i)) for i in range(x.size)]

    # Cycle over all paths and find the one with the smallest cost
    minimum_distance = np.inf
    path_with_minimum_distance = None

    for i in range(len(points)):
        a_path = list_of_all_paths[i]  # order of nodes
        ordered = points[a_path]  # ordered nodes
        # Find cost of that order by the sum of euclidean distances between points (i) and (i+1),
        cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
        if cost < minimum_distance:
            minimum_distance = cost
            path_with_minimum_distance = i

    return list_of_all_paths[path_with_minimum_distance]


class IsochroneInterpolator:
    def __init__(self, data_isochrone: pd.DataFrame, parameters_as_arguments: Union[list, tuple] = ('MH', 'logAge'),
                 parameters_to_infer: Union[list, tuple] = ('G_BP-RP', 'Gmag'), interpolation_type: str = 'LinearND'):
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
                Default: ('G_BP-RP', 'Gmag')
            interpolation_type (str): type of interpolation to use.
                Implemented: 'LinearND' : scipy.interpolate.LinearNDInterpolator
                             'RadialBasisFunction' : scipy.interpolate.Rbf
                Default: 'LinearND'

        Returns:
            absolutely... NOTHING
            (call the class if you want to evaluate it)
        """
        # Set all the parameter names to the class
        self.argument_parameters = parameters_as_arguments
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
                # Sum the number of parameter matches per row
                stars_to_use = np.sum(data_isochrone[self.argument_parameters] == a_parameter_combination, axis=1)

                # Only accept stars where all the parameters_are matched (the row-wise sum should equal n_parameters)
                stars_to_use = np.asarray(stars_to_use == n_parameters)

                # Hence, calculate the distance
                x = data_isochrone.loc[stars_to_use, self.inferred_parameters[0]]
                y = data_isochrone.loc[stars_to_use, self.inferred_parameters[1]]
                data_isochrone.loc[stars_to_use, 'cumulative_distance_along_curve'] = sum_along_curve(x, y)

        # If there aren't any other parameters, then we just find the cumulative distance for the whole damn table
        else:
            x = data_isochrone[self.inferred_parameters[0]]
            y = data_isochrone[self.inferred_parameters[1]]
            data_isochrone['cumulative_distance_along_curve'] = sum_along_curve(x, y)

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
                 resolution: int = 100) -> list:
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


class CMDInterpolator:
    def __init__(self,
                 parameters: Union[list, tuple] = ('B-R', 'G'),
                 curve_parameterisation_type: str = 'summed',
                 data_sorted_on: str = 'y_values',
                 filter_input: bool = False,
                 filter_window_size: int = 5,
                 filter_order: int = 3,
                 interp_type: str = 'UnivariateSpline',
                 interp_weights: Optional[str] = None,
                 interp_smoothing: Optional[float] = 0.05,
                 interp_order: int = 3):
        """Interpolates data from a potential cluster, allowing its scatter from the best fit spline to be evaluated.

        Notes:
            - Call the object later to evaluate the interpolated spline!
            - Will get slow if an especially big cluster is passed to the method.

        Args:
            parameters (list, tuple): the parameters in data_gaia that define the blue-red and total magnitudes.
            curve_parameterisation_type (string): how to parameterise the curve. Options:
                linear: parameterise with a sequence from 0 to 1 for all data points. Easiest to do,
                    but may produce sharp changes in gradient in what has to be fit.
                summed:  parameterise by mapping the sequence from 0 to 1 to a weighted sum of the distance along the
                    curve. Will generally produce better results, but assumes that the sorting was largely successful.
                Default: 'summed'
            data_sorted_on (string): how to sort the data (since the interpolator has to receive points in a certain order.
                y_values: sort all data on the magnitude parameter.
                x_values: sort all data on the colour parameter.
                Default: 'y_values'
            Todo add filtering to the docstr
            interp_type (string): type of interpolator to use. Currently, only 'UnivariateSpline' is implemented.
            interp_weights (string, optional): name of the column in data_gaia that gives parameter weights. If None, no
                weights are passed to the spline fit.
                Default: None
            interp_smoothing (float, optional): amount of smoothing to apply. If None, the SciPy intepolator will
                try to calculate an optimum value itself.
                Default: 0.05
            interp_order (int): order of the polynomial to fit.
                Default: 3
        """
        # Define some useful stuff to keep class-side
        self.fits_completed = 0

        # Set the parameter names to the class
        self.parameters = parameters
        self.weight_parameter = interp_weights

        # Write Useful stuff to the class
        self.sort_method = data_sorted_on
        self.curve_parameterisation_type = curve_parameterisation_type
        self.curve_parameter = None
        self.filter_input = filter_input
        self.filter_window_size = filter_window_size
        self.filter_order = filter_order

        # Stuff we'll need to set later to do with interpolation
        self.x_interpolator = None
        self.y_interpolator = None
        self.interpolation_type = interp_type

        if self.interpolation_type == 'UnivariateSpline':
            self.interpolation_arguments = {'s': interp_smoothing, 'k': interp_order}

            # Set __call__ to use this function!
            self.evaluate = self.evaluate_univariate_spline

        else:
            raise ValueError('Specified interpolation method not recognised!')

    def fit(self, data_gaia, max_repeats: int = 1, print_current_step: bool = False):
        """Fits a spline to the CMD data. Depending on the sort method chosen, this may be done iteratively to improve
        the quality of the fit.

        Args:
            data_gaia (pd.DataFrame): the Gaia data, which ought to contain magnitude information (including a blue -
                red column.)
            max_repeats (int): how many times to re-run the fitting.


        """
        # Kill The Panda, It's Time For X and Y Arrays
        data_colour = np.asarray(data_gaia[self.parameters[0]])
        data_magnitude = np.asarray(data_gaia[self.parameters[1]])

        # Decide on what the weights for the curve fitting will be
        if self.weight_parameter is not None:
            star_weights = data_gaia[self.weight_parameter]
        else:
            star_weights = None  # Aka no weights will be passed to functions

        while self.fits_completed < max_repeats:
            # Sort the data
            data_colour, data_magnitude = self._input_sort(data_colour, data_magnitude)

            # Make a parameter to describe the 2D curve with
            self.curve_parameter = self._input_parameterisation(data_colour, data_magnitude)

            # Filter the input (assuming that filtering was turned on during __init__)
            filtered_data_colour, filtered_data_magnitude = self._input_filter(data_colour, data_magnitude)

            # Interpolate!
            self._input_interpolate(filtered_data_colour, filtered_data_magnitude, star_weights)

            self.fits_completed += 1

            if print_current_step:
                print(self.fits_completed)

    def _input_sort(self, data_colour, data_magnitude):
        """Sorts input values to the class."""

        # Decide on how we're gonna initially sort the data
        # Sort x and y based on y values
        if self.sort_method == 'y_values':
            sort_args = np.argsort(data_magnitude)
            self.fits_completed = np.inf  # This sort method is non-iterable!

        # Sort x and y based on x values
        elif self.sort_method == 'x_values':
            sort_args = np.argsort(data_colour)
            self.fits_completed = np.inf  # This sort method is non-iterable!

        # Create a nearest neighbour graph and sort using it
        elif self.sort_method == 'nearest_neighbour_graph':
            sort_args = nn_graph_sort(data_colour, data_magnitude)
            self.fits_completed = np.inf  # This sort method is non-iterable!

        # Sort based on proximity to a pre-established line
        elif self.sort_method == 'proximity_to_line':
            # If this is our first run, then we have to sort on y values first instead
            if self.fits_completed == 0:
                sort_args = np.argsort(data_magnitude)
            else:
                # Grab data to feed to the sorter
                points_to_match = np.vstack([data_colour, data_magnitude]).T

                # Use a resoltuion of 100 or twice the length of the array & grab interpolated points
                resolution = np.max([1 * data_colour.size, 100])
                points_on_line = np.vstack(self(resolution=resolution)).T

                sort_args = proximity_to_line_sort(points_to_match, points_on_line)
        else:
            raise ValueError("Selected sort_method not recognised!")

        data_colour = data_colour[sort_args]
        data_magnitude = data_magnitude[sort_args]

        return data_colour, data_magnitude

    def _input_parameterisation(self, data_colour, data_magnitude):
        """Parameterise the CMD."""
        # Decide on how we're gonna parameterise the curve
        if self.curve_parameterisation_type == 'linear':
            curve_parameter = np.linspace(0, 1, data_magnitude.size)
        elif self.curve_parameterisation_type == 'summed':
            curve_parameter = sum_along_curve(data_colour, data_magnitude, normalise=True)
        else:
            raise ValueError('Specified method for curve_parameterisation_type of the curve not recognised!')

        return curve_parameter

    def _input_filter(self, data_colour, data_magnitude):
        """Filters the input data."""

        if self.filter_input:

            # # Adapt the filtering strength based on how many fits we've already made
            # self.filter_window_fraction /= self.fits_completed + 1

            # Set the filter window length to be some fraction of the dataset size, making sure that it's an odd number
            # filter_window_length = int(data_colour.size * self.filter_window_fraction)
            #
            # # Make sure it isn't less than the polyorder
            # if filter_window_length <= self.filter_order:
            #     filter_window_length = self.filter_order + 1
            #
            # # Make sure it's odd
            # filter_window_length += (1 - filter_window_length % 2)
            # filter_window_size = filter_window_length

            # Apply the filter!
            data_colour = savgol_filter(data_colour, self.filter_window_size, self.filter_order)
            data_magnitude = savgol_filter(data_magnitude, self.filter_window_size, self.filter_order)

        return data_colour, data_magnitude

    def _input_interpolate(self, data_colour, data_magnitude, star_weights):
        """Interpolates, given settings in the class"""
        if self.interpolation_type == 'UnivariateSpline':
            self.x_interpolator = UnivariateSpline(self.curve_parameter, data_colour, w=star_weights,
                                                   **self.interpolation_arguments)
            self.y_interpolator = UnivariateSpline(self.curve_parameter, data_magnitude, w=star_weights,
                                                   **self.interpolation_arguments)

        else:
            raise ValueError('Specified interpolation method not recognised!')

        return data_colour, data_magnitude

    def evaluate_univariate_spline(self, input_data: np.ndarray) -> tuple:
        """Evaluation function in the case that the class's interpolator is a scipy.interpolate.UnivariateSpline.

        Args:
            input_data (np.ndarray): input parameter data at whatever resolution is desired.

        Returns:
            a list containing [x_data, y_data] for the requested points.
        """
        return self.x_interpolator(input_data), self.y_interpolator(input_data)

    def __call__(self, range_to_evaluate: Union[str, np.ndarray] = 'full isochrone',
                 resolution: int = 100) -> tuple:
        """Evaluate the interpolator and re-produce all (or part of) the interpolated isochrone. Contains systems to
        automate a lot of the process for you, because it's kind like that. =)

        Args:
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

        return self.evaluate(range_to_evaluate)


class IsochroneDistanceInterpolator:
    def __init__(self):
        raise NotImplementedError("What are you doing, this isn't implemented, stop")

    def __call__(self):
        pass

