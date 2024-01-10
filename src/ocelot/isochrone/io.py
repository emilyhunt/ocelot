"""Set of functions for reading in isochrones from their downloaded formats and processing them into useful spaces."""

from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
from astropy.io import ascii


def read_parsec(
    file_location: Union[Path, List[Path], str, List[str]],
    max_label: int = 9,
    column_names: Optional[str] = None,
) -> pd.DataFrame:
    """Reads input CMD 3.3 isochrone(s). In the plural case, they're combined into one 
    DataFrame.

    Only verified for use when reading in isochrones from the CMD v3.3 & PARSEC v1.2S 
    web interface at http://stev.oapd.inaf.it/cgi-bin/cmd.

    Parameters
    ----------
    file_location : pathlib.Path or list of pathlib.Path or str
        You can specify three things:
        - Location of a file containing CMD 3.3 / PARSEC v1.2s isochrones.
        - A list of files as the above - they will all be joined.
        - A directory of .dat files containing isochrones
    max_label : int
        Maximum label to read in, which defaults to 9. The upper end of PARSEC tracks 
        can be a little bit experimental, so you may not want all labels (7 is a good 
        choice.) Default is 9 (the highest). See notes for more details.  
    column_names : ArrayLike, optional
        Alternative column names to use. Default: None, which uses column names:
        ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 
        'label', 'mbolmag', 'Gmag', 'G_BPmag', 'G_RPmag'] as from the CMD 3.3 output in 
        the Gaia DR2 Evans+2018 photometric system.

    Returns
    -------
    isochrone : pd.Dataframe
        a pd.DataFrame of the read-in isochrone. It's worth checking manually that this 
        worked, as the tables are in a format that requires some cleaning (that should 
        hopefully be done automatically.) It will also have had the colour calculated, 
        which will be labelled 'G_BP-RP'.

    Notes
    -----
    PARSEC v1.2s labels correspond to...
    - 0 = PMS, pre main sequence
    - 1 = MS, main sequence
    - 2 = SGB, subgiant branch, or Hertzsprung gap for more intermediate+massive stars
    - 3 = RGB, red giant branch, or the quick stage of red giant for 
        intermediate+massive stars
    - 4 = CHEB, core He-burning for low mass stars, or the initial stage of CHeB for 
        intermediate+massive stars
    - 5 = still CHEB, the blueward part of the Cepheid loop of intermediate+massive 
        stars
    - 6 = still CHEB, the redward part of the Cepheid loop of intermediate+massive stars
    - 7 = EAGB, the early asymptotic giant branch, or a quick stage of red giant for 
        massive stars
    - 8 = TPAGB, the thermally pulsing asymptotic giant branch
    - 9 = post-AGB (in preparation!)
    """
    # Todo refactor to be more cross-compatible with other PARSEC versions (and be generally better)
    # Use default column names if none are specified
    if column_names is None:
        column_names = [
            "Zini",
            "MH",
            "logAge",
            "Mini",
            "int_IMF",
            "Mass",
            "logL",
            "logTe",
            "logg",
            "label",
            "mbolmag",
            "Gmag",
            "G_BPmag",
            "G_RPmag",
        ]

    if isinstance(file_location, str):
        file_location = Path(file_location)

    # See if file_location is a directory - only works if one path is specified
    try:
        file_location_is_a_directory = file_location.is_dir()
    except AttributeError:  # Catches if file_location is a list of paths instead, and does not have an is_dir() method
        file_location_is_a_directory = False

    # Iterate over all files if a list of files or a directory has been specified
    if isinstance(file_location, list) or file_location_is_a_directory:
        # If a directory was specified, we want to first make a list of all its .dat files.
        if file_location_is_a_directory:
            file_location = list(file_location.glob("*.dat"))

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
        isochrones.columns = [
            "Zini",
            "MH",
            "logAge",
            "Mini",
            "int_IMF",
            "Mass",
            "logL",
            "logTe",
            "logg",
            "label",
            "mbolmag",
            "Gmag",
            "G_BPmag",
            "G_RPmag",
        ]

    # Drop anything that isn't the right type of star
    rows_with_bad_label = np.where(isochrones["label"] > max_label)[0]
    isochrones = isochrones.drop(labels=rows_with_bad_label, axis="index").reset_index(
        drop=True
    )

    # Add some extra useful things
    isochrones["G_BP-RP"] = isochrones["G_BPmag"] - isochrones["G_RPmag"]

    return isochrones


def read_mist():
    # Todo
    pass