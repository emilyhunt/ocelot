"""A number of functions for pre-processing Gaia data before clustering can begin."""

from typing import Optional, Union, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def cut_dataset(data_gaia: pd.DataFrame, parameter_cuts: Optional[dict] = None, geometric_cuts: Optional[dict] = None) \
        -> pd.DataFrame:
    """A function for cutting a dataset based on certain requirements: either on allowed parameter ranges or based on
    geometric cuts (such as selecting a circle from the data.)

    Todo: this function should be able to move clusters that are on an ra boundary.
        E.g. Blanco 1, which always gets messed up!

    Notes:
        - This function will do nothing if parameter_cuts and geometric_cuts are both None (the default settings!).
        - Currently, geometric_cuts is applied first, as this could drastically reduce the dataset size (and speed up
            later applications of parameter_cuts).
        - This could be done a lot more neatly, but is written for speed instead =)

    Args:
        data_gaia (pd.DataFrame): the data to apply the cuts to.
        parameter_cuts (dict, optional): a dictionary of lists or np.ndarrays of allowed [minimum, maximum] parameter
            ranges, in a style like: {"phot_g_mag": [-np.inf, 18]}, where np.inf can be used to ignore limits.
            Default: None
        geometric_cuts (dict, optional): a parameter dictionary for a geometric cut to apply to the dataset. Implemented
            cuts: NONE todo this
            Example: {"type":"great_circle", "sky_position"=[125.429, -16.743]}

    Returns:
        a cut data_gaia with reset indexing.

    todo this should return the original indices of whatever was dropped, which also need saving so that crossmatching
        can be done later

    """

    if parameter_cuts is not None:
        # Grab the names of the parameters we want to cut based on the dictionary keys.
        list_of_parameters_to_cut = parameter_cuts.keys()

        # Cycle over the parameters to cut and find the stars that satisfy all cuts. It's assumed that data_gaia is
        # oppressively huge, and that dropping & copying the array many times in a row would be inefficient. Instead,
        # we check all parameter matches first, and change data_gaia at the end.
        good_stars = np.ones(data_gaia.shape[0], dtype=bool)

        for a_parameter in list_of_parameters_to_cut:
            a_parameter_cut = parameter_cuts[a_parameter]

            # Check that the range is a list or np.ndarray
            type_to_check = type(a_parameter_cut)
            if type_to_check != list and type_to_check != np.ndarray:
                raise ValueError("Incorrect type of range found in parameter_cuts! All ranges must be specified as "
                                 "Python lists or np.ndarray arrays.")

            # Find the matches to the parameter in data_gaia (fyi: it's a lot faster to convert to numpy first. lol)
            # We then only update the good_stars indexes that are already True, so that we don't overwrite existing
            # fails and also only do the minimum amount of calculation.
            column_to_test = data_gaia.loc[good_stars, a_parameter].to_numpy()
            good_stars[good_stars] = np.logical_and(column_to_test > a_parameter_cut[0],
                                                    column_to_test < a_parameter_cut[1])

        # Once we have all the cuts, only *now* do we make data_gaia smaller.
        data_gaia = data_gaia.loc[good_stars, :].reset_index(drop=True)

    if geometric_cuts is not None:
        raise NotImplementedError("Sorry, geometric cutting isn't yet implemented here. Poke Emily to do something.")

    return data_gaia


def rescale_dataset(data_gaia: pd.DataFrame,
                    *args,
                    columns_to_rescale: Union[list, tuple] = ('ra', 'dec', 'pmra', 'pmdec', 'parallax'),
                    column_weights: Union[list, tuple] = (1., 1., 1., 1., 1.),
                    scaling_type: str = 'robust',
                    concatenate: bool = True,
                    **kwargs_for_scaler) -> Union[np.ndarray, List[np.ndarray]]:
    """A wrapper for sklearn's scaling methods that can automatically handle re-scaling data a number of different ways.

    Args:
        data_gaia (pd.DataFrame): the data to apply the scaling to.
        *args (pd.DataFrame): other data frames to apply the same scaling to.
        columns_to_rescale (list, tuple): keys of the columns to re-scale. Should be in the final desired order!
            Default: ('ra', 'dec', 'pmra', 'pmdec', 'parallax')
        column_weights (list, tuple): linear weights to apply to each column after re-scaling, which will reduce their
            impact in nearest-neighbor calculations (see: Liu+19 for an example of this done with parallax)
            Default: (1., 1., 1., 1., 1.)
        scaling_type (str): type of scaler to use, choose from: 'robust' (sklearn.preprocessing.RobustScaler) or
            'standard' (sklearn.preprocessing.StandardScaler). Robust scaling is more appropriate for data with outliers
            Default: 'robust'
        concatenate (bool): whether or not to join all rescaled arrays (assuming multiple args were specified) into one.
            Default: True, so only one np.array is returned.
        **kwargs_for_scaler: additional keyword arguments will be passed to the selected scaler.

    Returns:
        a np.ndarray of shape (n_stars, n_columns) of the re-scaled data, or a list of separate arrays if
            concatenate=False and at least one arg was specified.

    """
    # Make sure that the columns_to_rescale are a list (pandas don't like tuples #fact) and column weights is np.ndarray
    columns_to_rescale = list(columns_to_rescale)
    column_weights = np.asarray(column_weights)

    # Select the scaler to use
    if scaling_type == 'robust':
        scaler = RobustScaler(**kwargs_for_scaler)
    elif scaling_type == 'standard':
        scaler = StandardScaler(**kwargs_for_scaler)
    else:
        raise ValueError("Selected scaling_type not supported!")

    # Check that all values are finite
    if not np.isfinite(data_gaia[columns_to_rescale].to_numpy()).all():
        raise ValueError("At least one value in data_gaia is not finite! Unable to rescale the data.")

    # Apply the scaling & multiply by column weights
    to_return = [scaler.fit_transform(data_gaia[columns_to_rescale].to_numpy()) * column_weights]

    # Apply the same scaling scaling to any other data frames, if others have been specified
    for an_extra_cluster in args:

        # Check that all values are finite
        if not np.isfinite(an_extra_cluster[columns_to_rescale].to_numpy()).all():
            raise ValueError("At least one value in args is not finite! Unable to rescale the data.")

        # Re-scale!
        to_return.append(scaler.transform(an_extra_cluster[columns_to_rescale].to_numpy()) * column_weights)

    # Return a concatenated array
    if concatenate:
        return np.concatenate(to_return, axis=0)

    # Or, just return the one array if no args were specified (slightly ugly code so my API doesn't break lololol)
    elif len(to_return) == 1:
        return to_return[0]

    # Or, return all arrays separately if we aren't concatenating and have multiple arrays
    else:
        return to_return
