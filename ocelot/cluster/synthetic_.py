"""Functions for generating synthetic clusters convolved with Gaia measurement errors."""

import numpy as np
import pandas as pd
from astropy.io import ascii

from pathlib import Path
from typing import Union


def _calculate_m_over_h(z, z_over_x_solar: float = 0.0207, y_constant: float = 0.2485, z_multiplier: float = 1.78):
    """Convenience function for calculating [M/H] given a parsec isochrone.

    See "Ages/metallicities" at: http://stev.oapd.inaf.it/cgi-bin/cmd (last retrieved Feb 2020)

    """
    y = y_constant + z_multiplier * z
    x = 1 - y - z
    return np.log10(z / x) - np.log10(z_over_x_solar)


class SimulatedPopulations:
    def __init__(self, location: Union[Path, str], search_pattern: str = "*.dat"):
        """Convenience class to read in simulated stellar population output from the CMD 3.3 tool into a DataFrame. Also
        stores useful information about the simulated populations (e.g. available unique values) and provides methods to
        access the population.

        Clusters are read in using astropy.io.ascii.

        Note: it is assumed that simulated populations are on a *grid of values*, and this class will fail later if they
        aren't! Aka, there must be the same different ages for every different metallicity.

        Args:
            location (Path, str): location of the single file or directory to cycle over.
            search_pattern (str): if location is a directory, only read in file names that match this pattern. Note that
                * is treated as a wildcard.
                Default: "*.dat" (default extension of CMD 3.3 files)

        """
        # Cast the path as a pathlib.Path and check that it's real
        location = Path(location)

        if location.exists() is False:
            raise ValueError("specified path does not seem to exist!")

        # Make a list of all files to read in
        if location.is_dir():
            files_to_read = list(location.glob(search_pattern))
        else:
            files_to_read = [location]

        # Cycle over files... reading them in!!!
        separate_dataframes = []
        for a_file in files_to_read:
            separate_dataframes.append(ascii.read(str(a_file.resolve())).to_pandas())

        # Join all the data frames together and be super paranoid about the indexes being correct
        self.data = pd.concat(separate_dataframes, ignore_index=True).reset_index(drop=True)

        self.data.columns = ['Z', 'age', 'Mini', 'Mass', 'logL', 'logTe', 'logg', 'cmd_label',
                            'mbolmag', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']

        self.data["log_age"] = np.log10(self.data["age"])
        self.data["log_Z"] = _calculate_m_over_h(self.data["Z"])

        # We also want to keep a record of the unique values that we have, rounded to avoid silly floating point errors
        self.unique_log_ages = np.round(np.unique(self.data["log_ages"].to_numpy()), decimals=2)
        self.unique_log_metallicities = np.round(np.unique(self.data["log_Z"].to_numpy()), decimals=2)

    def get_stars(self, log_ages: Union[np.ndarray, list, tuple],
                  log_metallicities: Union[np.ndarray, list, tuple],
                  n_stars: Union[np.ndarray, list, tuple],
                  error_on_distant_request: bool = True,
                  concatenate: bool = True):
        """Creates random samples of stars given an input log_ages and log_metallicities.

        Warning: will use the closest available population, which means you ought to check that the input age and
        metallicity given are close to what was given when the class was made!

        Args:
            log_ages (list-like of floats): log ages to look for.
            log_metallicities (list-like of floats): log metallicities to look for.
            n_stars (list-like of ints): total numbers of stars in the clusters to retrieve.
            error_on_distant_request (bool): whether or not to raise an error if the requested age and metallicity
                values are not within the range of the catalogue.
                Default: True
            concatenate (bool): whether or not to join all the clusters together into one DataFrame.
                Default: True

        Returns:
            a single pd.DataFrame or a list of pd.DataFrame

        """
        # Get the nearest ages and metallicities of the required stars vs the input stars
        age_differences = np.abs(np.asarray(log_ages).reshape(-1, 1) - self.unique_log_ages)
        metallicity_differences = np.abs(np.asarray(log_metallicities).reshape(-1, 1) - self.unique_log_metallicities)

        # Grab the best values for each
        best_log_ages = self.unique_log_ages[np.argmin(age_differences, axis=1)]
        best_log_metallicities = self.unique_log_metallicities[np.argmin(metallicity_differences, axis=1)]

        # Check the user isn't being a numpty
        if error_on_distant_request:
            if np.any(np.logical_or(best_log_ages > 1, best_log_metallicities > 1)):
                raise ValueError("input age or metallicity values are too different (more than 1) from the values "
                                 "available in the catalogue! Set error_on_distant_request=False to disable this "
                                 "warning, albeit at your own risk...")

        # Loop over all of the requested ages and metallicities, and make clusters of about the same mass
        clusters_to_return = []
        for cluster_label, (an_age, a_metallicity, a_n_stars) in enumerate(zip(
                best_log_ages, best_log_metallicities, n_stars)):
            # Grab the stars that be good
            good_stars = np.logical_and(self.data["log_age"] == an_age, self.data["log_Z"] == a_metallicity)

            # Check that something hasn't gone wrong
            if np.count_nonzero(good_stars) == 0:
                raise ValueError(f"specified age {an_age} and metallicity {a_metallicity} combination was not found "
                                 "in the simulated population!")

            # Turn this into a new DataFrame!
            new_population = (self.data.loc[good_stars, :]
                              .copy()
                              .sample(n=a_n_stars, replace=True)
                              .reset_index(drop=True))
            new_population["cluster_label"] = cluster_label

            # And add it to our list of shit to concatenate
            clusters_to_return.append(new_population)

        # Concatenate if requested and return
        if concatenate:
            return pd.concat(clusters_to_return, ignore_index=True).reset_index(drop=True)
        else:
            return clusters_to_return


def generate_synthetic_clusters(simulated_populations: SimulatedPopulations,
                                data_gaia: pd.DataFrame,
                                age_examples: np.ndarray,
                                metallicity_examples: np.ndarray,
                                n_stars: Union[np.ndarray, list, tuple],
                                internal_velocity_dispersion: float = 500, ):
    """Generates synthetic star clusters based on generated simulated populations (e.g. via CMD 3.3) and real examples
    of Gaia data (and its associated errors.)"""

    # Load default










    pass
