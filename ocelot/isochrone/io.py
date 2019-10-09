"""Set of functions for reading in isochrones from their downloaded formats and processing them into useful spaces."""

from pathlib import Path

import numpy as np
import pandas as pd


def read_cmd_isochrone(file_location: Path, skiprows: int = 11) -> pd.DataFrame:
    """Reads an input CMD 3.3 isochrone.

    # Todo: more sophisticated would be to cycle over the first few rows (outside of pandas) until the first one not starting with a # is found.

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
        skiprows (int): how many rows to skip (the output header from CMD 3.3).
            Default: 11. Will need changing if CMD change anything on their end!

    Returns:
        a pd.DataFrame of the read-in isochrone. It's worth checking manually that this worked, as the tables are in
            a format that requires some cleaning (that should hopefully be done automatically.)

    """
    isochrones = pd.read_csv(file_location, skiprows=skiprows, delim_whitespace=True)

    # Deal with the fact that the header row starts with a #, which adds an extra empty column & fucks with the header
    # names
    cleaned_headers = list(isochrones.keys())[1:]
    isochrones = isochrones.drop(labels=cleaned_headers[-1], axis='columns')
    isochrones.columns = cleaned_headers

    # Drop rows that contain a repeat of the header
    rows_with_another_header = np.where(isochrones['Zini'] == '#')[0]
    isochrones = isochrones.drop(labels=rows_with_another_header, axis='index').reset_index(drop=True)

    # Reset so that everything is a float
    isochrones = isochrones.astype(np.float)

    return isochrones
