"""Functions for generating synthetic clusters convolved with Gaia measurement errors."""

import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.coordinates import SkyCoord, CartesianDifferential, CartesianRepresentation
import astropy.units as u
from ..calculate.dust import gaia_dr2_a_lambda_over_a_v
from ..calculate import king_surface_density, king_surface_density_fast
from scipy.stats import norm

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
    def __init__(self,
                 location: Union[Path, str],
                 search_pattern: str = "*.dat",
                 mass_tolerance: float = 0.05,
                 max_mass_iterations: int = 1000,
                 position_oversampling_factor: Union[float, int] = 1.,
                 limiting_magnitude: float = 22.):
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
            mass_tolerance (float): fractional tolerance within which to allow a drawn clusters' mass to deviate from
                the target mass. If set too low, it may not be possible to find a solution for the cluster!
                Default: 0.05
            max_mass_iterations (int): maximum number of times to try and re-draw stars to get a cluster of the correct
                mass before giving up and raising a ValueError.
                Default: 1000
            position_oversampling_factor (float, int): by what factor too many position values do we draw. We calculate
                the expectation value of successes for a given core & tidal radius, then draw this factor too many
                to reduce the number of times we have to call again.
                Set this lower if you're having memory issues!
                Default: 1 (aka about half of the time we'll already have drawn enough random values to move on.)
            limiting_magnitude (float, int): what limiting magnitude we'll immediately drop all stars below. Should be
                set conservatively (and below the actual limiting magnitude) else the far end of clusters will be
                underpopulated, as stars below the limit could scatter back within the bounds of the limit. Set to
                np.inf to turn this off.
                Default: 22., which is very conservative (few stars in Gaia DR2 have photometric errors of more than 0.1
                    even)

        """
        # Store some class stuff
        self.mass_tolerance = mass_tolerance
        self.max_mass_iterations = max_mass_iterations
        self.required_info_on_clusters = ['ra', 'dec', 'distance', 'extinction_v', 'pmra', 'pmdec',
                                          'age', 'Z', 'mass', 'tidal_radius', 'v_int']
        self.position_oversampling_factor = position_oversampling_factor
        self.limiting_magnitude = limiting_magnitude

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

        # We also want to keep a record of the unique values that we have
        self.unique_log_ages = np.unique(self.data["log_age"].to_numpy())
        self.unique_log_metallicities = np.unique(self.data["log_Z"].to_numpy())

    def _add_positions_to_cluster(self, new_population: pd.DataFrame, data_cluster: pd.Series):
        """Gives every star in a cluster a position, randomly assigned by drawing from a King profile in a brute-force
        Monte-Carlo way. Uses self.position_oversampling_factor to control how much we over or under-draw vs. the
        expected number of cluster member stars.

        Args:
            new_population (pd.DataFrame): the population of stars to work on, so far.
            data_cluster (pd.Series): the data for the cluster to work on (see .get_clusters() for full docs), but just
                for the one cluster.

        Returns:
            new_population (pd.DataFrame) but with new photometry appended.

        """
        # Find the maximum value of a King profile given these parameters, making our MC in a moment more efficient
        # This is *NOT* done with the fast version, as the slow version has checks to ensure the values are correct.
        max_king = king_surface_density(0, data_cluster['radius_c'], data_cluster['radius_t'])

        # Calculate the expected acceptance fraction with the area under the curve (aka the inv. normalisation constant)
        # (since 1/A = (max) / (A * max) = area under the curve)
        inverse_normalisation_constant = (
            max_king / king_surface_density(0, data_cluster['radius_c'], data_cluster['radius_t'], normalise=True))

        # The expected acceptance fraction is the ratio of the King curve area to the area of the uniform deviates
        # square, then cubed as we're in 3D.
        expected_acceptance_fraction = (inverse_normalisation_constant / (data_cluster['radius_t'] * max_king))**3

        # Grab some stats about how many stars we're working with
        total_stars = new_population.shape[0]
        remaining_stars = total_stars
        completed_stars = 0

        """ OLD WAY - valid only for 1D!
                # Keep an array of final r values to write to and an array of bools for which r values still need to be re-drawn
                # (initialised as an array of only True as we have to begin)
                r_values = np.empty(total_stars, dtype=float)
                to_draw = np.ones(total_stars, dtype=bool)

                # Loop time! Draw random r values and a number between 0,max_king to decide whether or not to accept the r value
                while remaining_stars > 0:
                    # Pull some random deviates from a square
                    r_values[to_draw] = np.random.rand(remaining_stars) * data_cluster['radius_t']
                    test_values = np.random.rand(remaining_stars) * max_king

                    # Evaluate the King profile at these points
                    king_values = king_surface_density(
                        r_values[to_draw], data_cluster['radius_c'], data_cluster['radius_t'])

                    # Tell the loop whether or not to keep drawing these values or whether to accept them
                    to_draw[to_draw] = test_values > king_values

                    remaining_stars = np.count_nonzero(to_draw)

                # Draw random angles for all the stars
                theta, phi = points_on_sphere(total_stars, radians=False, phi_symmetric=True)
                """

        """
        # VALID FOR 3D - but slow as fuck!
        # Keep an array of final r values to write to and an array of bools for which r values still need to be re-drawn
        # (initialised as an array of only True as we have to begin)
        r_values = np.empty((total_stars, 3), dtype=float)
        to_draw = np.ones(total_stars, dtype=bool)

        # Loop time! Draw random r values and a number between 0,max_king to decide whether or not to accept the r value
        while remaining_stars > 0:
            # Pull some random deviates from a square
            r_values[to_draw, :] = (np.random.rand(remaining_stars, 3) * 2 - 1) * data_cluster['radius_t']
            test_values = np.random.rand(remaining_stars, 3) * max_king

            # Evaluate the King profile at these points
            king_values = king_surface_density_fast(
                np.abs(r_values[to_draw, :]), data_cluster['radius_c'], data_cluster['radius_t'])

            # Tell the loop whether or not to keep drawing these values or whether to accept them
            to_draw[to_draw] = np.any(test_values > king_values, axis=1)

            # print(np.any(test_values > king_values, axis=1))

            remaining_stars = np.count_nonzero(to_draw)
            

        """

        # Keep an array of final r values to write to and an array of bools for which r values still need to be re-drawn
        # (initialised as an array of only True as we have to begin)
        r_values = np.empty((total_stars, 3), dtype=float)

        # Loop time! Draw random r values and a number between 0,max_king to decide whether or not to accept the r value
        while remaining_stars > 0:
            # Pull some random deviates from a square - we pull out a lot more than we need so we can rejection sample
            # in bulk
            stars_to_draw = int(remaining_stars * self.position_oversampling_factor / expected_acceptance_fraction)
            test_r_values = ((np.random.rand(stars_to_draw, 3) * 2 - 1)
                             * data_cluster['radius_t'])
            test_values = np.random.rand(*test_r_values.shape) * max_king

            # Evaluate the King profile at these points
            king_values = king_surface_density_fast(
                np.abs(test_r_values), data_cluster['radius_c'], data_cluster['radius_t'])

            # Find the good values
            good_test_values = np.all(test_values < king_values, axis=1)
            n_good_test_values = np.count_nonzero(good_test_values)

            # Clip the number to write, as we don't want to try and write too many!
            if n_good_test_values > remaining_stars:
                values_to_write = remaining_stars
                index_when_we_have_enough_values = ((np.cumsum(good_test_values) == values_to_write).nonzero()[0])[0]
                good_test_values[index_when_we_have_enough_values + 1:] = False
            else:
                values_to_write = n_good_test_values

            # Write as many values as we can
            r_values[completed_stars: completed_stars + values_to_write] = (
                test_r_values[good_test_values])

            # Update with however many stars we wrote
            remaining_stars -= values_to_write
            completed_stars = total_stars - remaining_stars

        # Convert the cluster's stars and the cluster's own position into cartesian coords (we assume that the cluster
        # stars are all around the sun at first, hijacking astropy as a way to go from 3D sphericals to cartesian coords
        cluster_location = SkyCoord(
            ra=data_cluster['ra'] << u.deg, dec=data_cluster['dec'] << u.deg,
            distance=data_cluster['distance'] << u.pc).cartesian

        cluster_stars = CartesianRepresentation(
            r_values << u.pc, xyz_axis=1)

        cluster_stars = SkyCoord(cluster_location + cluster_stars, frame='icrs')

        # Pop this all back in the DataFrame
        new_population['ra'] = cluster_stars.ra.value
        new_population['dec'] = cluster_stars.dec.value

        # Also make l, b co-ordinates, distances and parallaxes in mas
        cluster_stars = cluster_stars.transform_to('galactic')
        new_population['l'] = cluster_stars.l.value
        new_population['b'] = cluster_stars.b.value
        new_population['distance'] = cluster_stars.distance.value
        new_population['parallax'] = 1000 / new_population['distance']

        return new_population

    @staticmethod
    def _add_proper_motions_to_cluster(new_population: pd.DataFrame, data_cluster: pd.Series):
        """Gives every star in a cluster a proper motion. Requires that they already have positions!

        Args:
            new_population (pd.DataFrame): the population of stars to work on, so far. Must contain 'ra', 'dec' and
                'distance' values!
            data_cluster (pd.Series): the data for the cluster to work on (see .get_clusters() for full docs), but just
                for the one cluster.

        Returns:
            new_population (pd.DataFrame) but with new photometry appended.

        """
        # Convert the cluster location into cartesians.
        cluster_location = SkyCoord(
            ra=data_cluster['ra'] << u.deg, dec=data_cluster['dec'] << u.deg,
            distance=data_cluster['distance'] << u.pc,
            pm_ra_cosdec=data_cluster['pmra'] << u.mas / u.yr,
            pm_dec=data_cluster['pmdec'] << u.mas / u.yr,
            radial_velocity=0. * u.m / u.s,
            frame='icrs')
        cluster_velocity = cluster_location.velocity

        # Draw Gaussian internal velocities based on the requested internal velocity dispersion.
        v_x, v_y, v_z = norm.rvs(loc=0.0, scale=data_cluster['v_int'], size=(3, new_population.shape[0]))

        # Add these velocities to the cluster stars' locations to get proper motions
        cluster_stars = SkyCoord(
            ra=new_population['ra'] << u.deg, dec=new_population['dec'] << u.deg,
            distance=new_population['distance'] << u.pc, frame='icrs',).cartesian

        cluster_star_velocities = CartesianDifferential(
            d_x=(v_x << u.m / u.s) + cluster_velocity.d_x,
            d_y=(v_y << u.m / u.s) + cluster_velocity.d_y,
            d_z=(v_z << u.m / u.s) + cluster_velocity.d_z,
            unit=u.m / u.s)

        cluster_stars = SkyCoord(cluster_stars.with_differentials(cluster_star_velocities), frame='icrs')

        # Add the proper motions to the DataFrame
        new_population['pmra'] = cluster_stars.pm_ra_cosdec.value
        new_population['pmdec'] = cluster_stars.pm_dec.value

        return new_population

    @staticmethod
    def _add_photometry_to_cluster(new_population: pd.DataFrame, data_cluster: pd.Series):
        """Adds Gaia photometry to a cluster's member stars.

        Args:
            new_population (pd.DataFrame): the population of stars to work on, so far.
            data_cluster (pd.Series): the data for the cluster to work on (see .get_clusters() for full docs), but just
                for the one cluster.

        Returns:
            new_population (pd.DataFrame) but with new photometry appended.

        """
        new_population['phot_g_mean_mag'] += (
                data_cluster['distance_modulus'] + data_cluster['extinction_v'] * gaia_dr2_a_lambda_over_a_v["G"])
        new_population['phot_bp_mean_mag'] += (
                data_cluster['distance_modulus'] + data_cluster['extinction_v'] * gaia_dr2_a_lambda_over_a_v["G_BP"])
        new_population['phot_rp_mean_mag'] += (
                data_cluster['distance_modulus'] + data_cluster['extinction_v'] * gaia_dr2_a_lambda_over_a_v["G_RP"])
        new_population['bp_rp'] = new_population['phot_bp_mean_mag'] - new_population['phot_rp_mean_mag']

        return new_population

    def _draw_cluster_of_correct_mass(self, simulated_population: pd.DataFrame, target_mass: float):
        """Draws random samples from a simulated population and creates a cluster of the desired mass. Uses
        self.mass_tolerance and self.max_mass_iterations to govern the random sampling. These can be set during class
        initialisation.

        Args:
            simulated_population (pd.DataFrame): a simulated stellar population at the required Z, age.
            target_mass (float): the final mass to aim for when generating a cluster.

        Returns:
            new_population (pd.DataFrame): a data frame of all the details of the newly generated cluster.

        """
        # Some stuff we'll need
        n_simulated_stars = simulated_population.shape[0]
        total_simulated_mass = np.sum(simulated_population['Mass'])
        typical_mass = np.mean(simulated_population['Mass'])

        # First, we try to get close by randomly selecting stars in the simulated population
        fraction_of_stars_to_draw = target_mass / total_simulated_mass
        stars_to_draw = int(target_mass / total_simulated_mass * n_simulated_stars)
        new_indices = np.random.randint(
            0, high=n_simulated_stars, size=stars_to_draw)

        # Now, we'll try to re-sample the cluster until the new mass is within the mass_tolerance of the target mass
        new_mass = np.sum(simulated_population.loc[new_indices, 'Mass'])
        mass_difference = target_mass - new_mass
        iterations = 0
        while np.abs(mass_difference) > target_mass * self.mass_tolerance:

            # Decide whether to add a star or take one away
            n_stars_to_draw = int(np.clip(mass_difference / typical_mass, 1, np.inf))
            if mass_difference > 0:
                new_indices = np.append(
                    new_indices, np.random.randint(0, high=n_simulated_stars, size=n_stars_to_draw))
            else:
                new_indices = np.delete(
                    new_indices, np.random.randint(0, high=new_indices.shape[0], size=n_stars_to_draw))

            new_mass = np.sum(simulated_population.loc[new_indices, 'Mass'])
            mass_difference = target_mass - new_mass

            iterations += 1
            if iterations > self.max_mass_iterations:
                raise ValueError(f"unable to select a cluster of the correct mass after {iterations} iterations! Try "
                                 f"reducing the tolerance of the desired cluster mass.")

        return simulated_population.loc[new_indices, :].reset_index(drop=True)

    def _make_cluster(self, data_cluster: pd.Series, cluster_label: int):
        """The actual function in the class that iteratively makes clusters one by one! Calls all the other methods
        around these parts and returns a whole cluster once done.

        Args:
            data_cluster (pd.Series): the data for the cluster to work on (see .get_clusters() for full docs), but just
                for the one cluster.
            cluster_label (int): the label to append to cluster stars. Useful to keep track of which star belongs
                where if the overall output of .get_clusters() is concatenated later.

        Returns:
            new_population (pd.DataFrame): a data frame of all the details of the newly generated cluster.

        """
        # Grab the stars that be good from the simulated populations
        good_stars = np.logical_and(self.data["log_age"] == data_cluster['age'],
                                    self.data["log_Z"] == data_cluster['Z'])

        # Check that something hasn't gone wrong and that at least one star matches this combination
        if np.count_nonzero(good_stars) == 0:
            raise ValueError(f"specified age {data_cluster['age']} and metallicity {data_cluster['Z']} combination "
                             f"was not found in the simulated population!")

        # Turn this into a new DataFrame! (I don't want to mess up the old one)
        simulated_population = self.data.loc[good_stars, :].copy().reset_index(drop=True)

        # Shuffle the DataFrame and create a stellar sample of roughly the required mass
        new_population = self._draw_cluster_of_correct_mass(simulated_population, data_cluster['mass'])

        # Add parameters to the cluster & perform dropping if requested
        new_population = self._add_photometry_to_cluster(new_population, data_cluster)
        new_population = (
            new_population.loc[new_population['phot_g_mean_mag'] < self.limiting_magnitude, :].reset_index(drop=True))
        new_population = self._add_positions_to_cluster(new_population, data_cluster)
        new_population = self._add_proper_motions_to_cluster(new_population, data_cluster)
        new_population["cluster_label"] = cluster_label

        return new_population

    def get_clusters(self, data_clusters: pd.DataFrame,
                     error_on_invalid_request: bool = True,
                     concatenate: bool = True):
        """Creates random samples of stars given an input log_ages and log_metallicities.

        Warning: will use the closest available population, which means you ought to check that the input age and
        metallicity given are close to what was given when the class was made!

        Args:
            data_clusters (pd.DataFrame): information on all clusters to return, including:
                Spatial parameters
                    'ra': right ascension of the cluster. (deg)
                    'dec': declination of the cluster. (deg)
                    'distance': distance to the cluster. (parsecs)
                    'extinction_v': v-band extinction towards the cluster.
                    'pmra': proper motion in right ascension of the cluster. (mas/yr)
                    'pmdec': proper motion in declination of the cluster. (mas/yr)
                Internal parameters
                    'age': logarithmic age of the cluster.
                    'Z': metallicity of the cluster, in units of dex (aka [Fe/H] or Z/X).
                    'mass': underlying mass of the cluster, in solar masses.
                    'radius_c': King's core radius of the cluster, in parsecs.
                    'radius_t': King's tidal radius of the cluster, in parsecs.
                    'v_int': internal velocity dispersion of the cluster, in m/s.
            error_on_invalid_request (bool): whether or not to raise an error if the requested age and metallicity
                values are not within the range of the catalogue.
                Default: True
            concatenate (bool): whether or not to join all the clusters together into one DataFrame.
                Default: True

        Returns:
            a single pd.DataFrame or a list of pd.DataFrame

        """
        # Get the nearest ages and metallicities of the required stars vs the input stars
        age_differences = np.abs(np.asarray(data_clusters['age']).reshape(-1, 1) - self.unique_log_ages)
        metallicity_differences = np.abs(np.asarray(data_clusters['Z']).reshape(-1, 1) - self.unique_log_metallicities)
        indices_best_age = np.argmin(age_differences, axis=1)
        indices_best_metallicity = np.argmin(metallicity_differences, axis=1)

        # Check the user isn't being a numpty
        if error_on_invalid_request:
            if np.any(np.logical_or(age_differences[indices_best_age] > 1,
                                    metallicity_differences[indices_best_metallicity] > 1)):
                raise ValueError("an input age or metallicity value is too different (more than 1) from the values "
                                 "available in the catalogue! Set error_on_invalid_request=False to disable this "
                                 "warning, albeit at your own risk as output clusters will not match your "
                                 "specifications in age or metallicity.")

        # Grab the best values for each based on what's near to the simulated population
        data_clusters['age'] = self.unique_log_ages[indices_best_age]
        data_clusters['Z'] = self.unique_log_metallicities[indices_best_metallicity]

        # Grab some stuff we'll need in a moment: distance modulus and limiting magnitude
        data_clusters['distance_modulus'] = 5 * np.log10(data_clusters['distance']) - 5

        # Loop over all of the requested ages and metallicities, and make clusters of about the same mass
        n_clusters = data_clusters.shape[0]
        clusters_to_return = []
        i_cluster = 0
        while i_cluster < n_clusters:
            clusters_to_return.append(self._make_cluster(data_clusters.loc[i_cluster, :], i_cluster))
            i_cluster += 1

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
