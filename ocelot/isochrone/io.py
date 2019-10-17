"""Set of functions for reading in isochrones from their downloaded formats and processing them into useful spaces."""

from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
from astropy.io import ascii


def read_cmd_isochrone(file_location: Union[Path, List[Path]],
                       max_label: int = 7,
                       column_names: Optional[str] = None) -> pd.DataFrame:
    """Reads input CMD 3.3 isochrone(s). In the plural case, they're combined into one DataFrame.

    Notes:
        - Only verified for use when reading in isochrones from the CMD v3.3 & PARSEC v1.2S web interface at
            http://stev.oapd.inaf.it/cgi-bin/cmd
        - This function uses the pathlib module so that it's os-independent. If you've never used it before, then you
            can find docs at: https://docs.python.org/3.6/library/pathlib.html
        - You can make a new path with pathlib with something like:
            from pathlib import Path
            path_to_isochrones = Path('<location_of_the_isochrones>')

    Args:
        file_location (pathlib.Path or list of pathlib.Path): you have three options:
            - Location of a file containing CMD 3.3 / PARSEC v1.2s isochrones.
            - A list of files as the above - they will all be joined.
            - A directory of .dat files containing isochrones
        max_label (int): maximum label to read in. This corresponds to:
            0 = PMS, pre main sequence
            1 = MS, main sequence
            2 = SGB, subgiant branch, or Hertzsprung gap for more intermediate+massive stars
            3 = RGB, red giant branch, or the quick stage of red giant for intermediate+massive stars
            4 = CHEB, core He-burning for low mass stars, or the initial stage of CHeB for intermediate+massive stars
            5 = still CHEB, the blueward part of the Cepheid loop of intermediate+massive stars
            6 = still CHEB, the redward part of the Cepheid loop of intermediate+massive stars
            7 = EAGB, the early asymptotic giant branch, or a quick stage of red giant for massive stars
            8 = TPAGB, the thermally pulsing asymptotic giant branch
            9 = post-AGB (in preparation!)
            default: 7
        column_names (str, optional): alternative column names to use.
            Default: None, which uses column names:
                ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label',
                 'mbolmag', 'Gmag', 'G_BPmag', 'G_RPmag']
            as from the CMD 3.3 output in the Gaia DR2 Evans+2018 photometric system.

    Returns:
        a pd.DataFrame of the read-in isochrone. It's worth checking manually that this worked, as the tables are in
            a format that requires some cleaning (that should hopefully be done automatically.) It will also have had
            the colour calculated, which will be labelled 'G_BP-RP'.

    """
    # Use default column names if none are specified
    if column_names is None:
        column_names = ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label',
                        'mbolmag', 'Gmag', 'G_BPmag', 'G_RPmag']

    # See if file_location is a directory - only works if one path is specified
    try:
        file_location_is_a_directory = file_location.is_dir()
    except AttributeError:  # Catches if file_location is a list of paths instead, and does not have an is_dir() method
        file_location_is_a_directory = False

    # Iterate over all files if a list of files or a directory has been specified
    if type(file_location) is list or file_location_is_a_directory:

        # If a directory was specified, we want to first make a list of all its .dat files.
        if file_location_is_a_directory:
            file_location = list(file_location.glob('*.dat'))

        # Cycle over all the isochrones, combining them into a file
        list_of_isochrones = []
        for a_file in file_location:
            current_isochrones = ascii.read(str(a_file.resolve())).to_pandas()
            current_isochrones.columns = column_names
            list_of_isochrones.append(current_isochrones)

        # Combine all the isochrones into one and reset the indexing
        isochrones = pd.concat(list_of_isochrones, ignore_index=True)

    # Otherwise, just read in the one file
    else:
        isochrones = ascii.read(str(file_location.resolve())).to_pandas()
        isochrones.columns = column_names

    # Use default column names if none are specified
    if column_names is None:
        isochrones.columns = ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label',
                              'mbolmag', 'Gmag', 'G_BPmag', 'G_RPmag']

    # Drop anything that isn't the right type of star
    rows_with_bad_label = np.where(isochrones['label'] > max_label)[0]
    isochrones = isochrones.drop(labels=rows_with_bad_label, axis='index').reset_index(drop=True)

    # Add some extra useful things
    isochrones['G_BP-RP'] = isochrones['G_BPmag'] - isochrones['G_RPmag']

    return isochrones


def convert_cmd_table_to_array(data_isochrone: pd.DataFrame,
                               dimensions: Union[list, tuple, np.ndarray, str] = 'infer',
                               axes: Union[list, tuple] = None,
                               equally_sized_axes: int = 2) -> np.ndarray:
    """Converts an input CMD 3.3 table into an array for use in interpolation.

    Args:
        data_isochrone (pd.DataFrame): a table of the isochrones, read in via read_cmd_isochrone.
        dimensions (list-like or str): the dimensions to split the table along, or 'infer' to try and find this
            automatically.
            Default: 'infer'
        axes (list-like of str): the axis labels, in order, to convert to an array on.
            Default: ['Zini', 'logAge', 'Gmag', 'G_BP-RP']

    Returns:
        a np.ndarray of the specified shape.

    """
    # Set the axes to default values if None were specified:
    if axes is None:
        axes = ['Zini', 'logAge', 'Gmag', 'G_BP-RP']

    # Infer the dimensions of the table if necessary
    if dimensions == 'infer':
        # Grab some constants we need and re-assign dimensions to be blank
        number_of_axes = len(axes)
        dimensions = np.zeros(number_of_axes)

        # Loop over each axis and find how many unique values there are, stopping before the equally sized axes
        for an_axis, i in enumerate(axes[:-equally_sized_axes]):
            dimensions[i] = np.unique(data_isochrone[an_axis]).size

    # Todo: probably not actually needed. oops lol

    pass

