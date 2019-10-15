"""Set of functions for reading in isochrones from their downloaded formats and processing them into useful spaces."""

from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from astropy.io import ascii


def read_cmd_isochrone(file_location: Path,
                       max_label: int = 7,
                       column_names: Optional[str]=None,
                       solar_metallicity: float = 0.0207) -> pd.DataFrame:
    """Reads an input CMD 3.3 isochrone.

    Notes:
        - Only verified for use when reading in isochrones from the CMD v3.3 & PARSEC v1.2S web interface at
            http://stev.oapd.inaf.it/cgi-bin/cmd
        - This function uses the pathlib module so that it's os-independent. If you've never used it before, then you
            can find docs at: https://docs.python.org/3.6/library/pathlib.html
        - You can make a new path with pathlib with something like:
            from pathlib import Path
            path_to_isochrones = Path('<location_of_the_isochrones>')

    Args:
        file_location (pathlib.Path): location of the file containing CMD 3.3 / PARSEC v1.2s isochrones.
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
        solar_metallicity (float): value of the solar metallicity Z/X to use when computing log initial metallicities.
            Default: 0.0207

    Returns:
        a pd.DataFrame of the read-in isochrone. It's worth checking manually that this worked, as the tables are in
            a format that requires some cleaning (that should hopefully be done automatically.)

    """
    isochrones = ascii.read(str(file_location.resolve())).to_pandas()

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

