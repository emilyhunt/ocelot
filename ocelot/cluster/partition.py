"""Classes that will partition a dataset into multiple pieces."""

from typing import Optional, Union

import numpy as np
import healpy as hp
import pandas as pd
import gc
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt
from ..plot import clustering_result
from .preprocess import _get_healpix_frame, recenter_dataset
from pathlib import Path
from astropy.coordinates import SkyCoord
from scipy.optimize import minimize


def _tidal_radius_of_cluster(distance, tidal_radius):
    """Returns degrees from parsecs."""
    return np.arctan(tidal_radius / distance) * 180 / np.pi


def _sky_area_of_cluster(distance, tidal_radius):
    """Unit is in square degrees!"""
    tidal_radius_in_degrees = _tidal_radius_of_cluster(distance, tidal_radius)
    return np.pi * tidal_radius_in_degrees ** 2


def _convert_pixels_to_polygon(current_pixels, nside=32, step=5):
    """Converts a list of pixels into a polygon object, which gets re-centered too for free!"""
    # Work out a center numerically from the mean center of all the pixels
    ra, dec = hp.pix2ang(nside, current_pixels, nest=True, lonlat=True)
    center = (ra.mean(), dec.mean())

    # Extract the boundaries of the pixels
    pixel_data = []
    for a_pixel in current_pixels:
        pixel_ra, pixel_dec = hp.vec2ang(hp.boundaries(nside, a_pixel, step=step, nest=True).T, lonlat=True)
        pixel_data.append(pd.DataFrame({'ra': pixel_ra, 'dec': pixel_dec}))

    # Recenter them all
    pixel_data_recentered = recenter_dataset(*pixel_data, center=center, proper_motion=False,
                                             always_return_list=True)

    # Turn them all into one big shape of happiness and fun
    polygon = Polygon()
    for a_pixel_data in pixel_data_recentered:
        new_poly = Polygon(a_pixel_data[['lon', 'lat']].values)
        polygon = polygon.union(new_poly)

    return polygon


def _cluster_contained_within_pixel(parameters, polygon: Polygon, parallax: float):
    cluster_radius = parameters[0]

    if cluster_radius < 0:
        return np.inf

    cluster = Point(0, 0).buffer(np.arctan(cluster_radius * parallax / 1000) * 180 / np.pi)

    # Return a negative number if the cluster doesn't fit entirely within the pixel
    if polygon.contains(cluster):
        to_return = - cluster_radius
    else:
        to_return = + cluster_radius

    return to_return


def find_largest_cluster_in_pixel(current_pixels: Union[list, np.ndarray, tuple],
                                  level: int,
                                  parallax: float,
                                  step: int = 5):
    """A minimisation scheme for finding the largest possible cluster that could live at the center of a HEALPix pixel.

    Args:
        current_pixels (list-like): list-like of current pixels that make up this field. May simply be length 1.
        level (int): the HEALPix level the pixel is specified with, where nside = 2 ** level.
        parallax (float): the parallax (in mas) of the start of the pixel's range. Aka the nearest possible cluster
            distance to consider.
        step (int): resolution of the shapely instance of a pixel. Higher values mean more points per side. Minimimum
            is 1, which just returns the corners only.
            Default: 5

    Returns:
        the largest possible radius of cluster that can fit within the pixel. It's a float!
        Raises a RuntimeError if minimisation was unsuccessful.

    """
    # Do our setup
    polygon = _convert_pixels_to_polygon(current_pixels, nside=2**level, step=step)

    # Minimise the function and find the smallest area
    result = minimize(_cluster_contained_within_pixel, np.atleast_1d(1.), args=(polygon, parallax), method='powell')

    if result.success:
        return np.atleast_1d(result.x)[0]
    else:
        raise RuntimeError(f"unable to calculate largest possible cluster that could be contained in pixel list "
                           f"{current_pixels} at HP level {level}. "
                           f"Minimisation failed!")


def check_constraints(constraints, sky_area, minimum_area_fraction=2., tidal_radius=10.):
    """Function for checking the validity of constraints.

    Args:
        constraints (list-like): the constraints array, of shape (n_distances, 2), where the first entries are the
            number of times to tile this partition, and the second number is the final distance of the bin. np.inf
            specifies a bin that's endlessly long.
        sky_area (float): the total area of the field being studied.
        minimum_area_fraction (float): minimum tile_area / cluster_area allowed at each distance. If any tiles have
            area fractions below this, this function will raise a ValueError.
            Default: 2.
        tidal_radius (float): tidal radius of the test cluster to consider.
            Default: 10. (a decent average number for open clusters based on MWSC.)

    Returns:
        if no issues found:
            - first_valid_distance (float), i.e. the first distance at which a whole cluster will satisfy
              minimum_area_fraction in bin 0
            - the start excess area fractions of bins
            - the end excess area fractions of bins

    """
    pass


default_partition = [
    [None, None, 0.],
    [None, None, 700.],
    [5, 6, 2000.]
]


class DataPartition:
    def __init__(self,
                 data: pd.DataFrame,
                 central_pixel: int,
                 constraints: Optional[Union[list, tuple, np.ndarray]] = None,
                 final_distance: float = np.inf,
                 parallax_sigma_threshold: float = 2.,
                 minimum_size: int = 10,
                 n_stars_per_component: Optional[Union[list, tuple, np.ndarray, int]] = None,
                 verbose: bool = False):
        """A class for creating Gaia dataset partitions. Checks the quality of the constraints and writes them to the
        class after some processing.

        WARNING: Currently only supports squares! Rectangular fields may have unwanted effects.

        Args:
            central_pixel (int): healpix level 5 id of the central pixel.
            constraints (list-like, optional): the constraints array, of shape (n_distances, 3), where each entry looks
                like:
                [healpix_level or None, overlap healpix_level or None, start_distance in pc]
                Default: ocelot.cluster.preprocess.default_partition
            final_distance (float): end distance of the last set of constraints.
                Default: np.inf
            parallax_sigma_threshold (float): how much of the parallax error to consider when deciding whether or not to
                include stars. If zero, error is not considered at all and there won't be any overlap. If large, then
                most stars will be in every parallax bin.
                Default: 2. (i.e. ~90% of stars actually in a bin but outside of it within error will end up in the bin)
            minimum_size (int): special option when working with GMMs only! Allows a minimum size to be specified as a
                multiplier of the n_stars_per_component of this bin.
                Default: 10
            n_stars_per_component (list-like or int, optional): n_components to use, if partitioning for a Gaussian Mixture
                Model. May either be a list-like (of length n_distances) or a single int to use for all.
                Default: None (i.e. this functionality is turned off)
            verbose (bool): whether or not to print a couple of info things to the console upon creation.
                Default: False

        """
        self.verbose = verbose

        if self.verbose:
            print("Creating a dataset partitioning scheme...")

        # Check that the input constraints aren't total BS
        if constraints is None:
            constraints = default_partition

        constraints = np.asarray(constraints, dtype=object)
        if constraints.shape[1] != 3:
            raise ValueError("input constraints must have shape (n_levels, 3)!")

        # Turn the constraints into numpy arrays of the info we need
        healpix_levels = constraints[:, 0]
        healpix_overlaps = constraints[:, 1]

        # For the distances, we grab them and then safely turn them into parallaxes
        start_distances = np.asarray(constraints[:, 2], dtype=float)
        end_distances = np.append(start_distances[1:], final_distance)
        start_parallaxes, end_parallaxes = self._safe_distance_to_parallax(start_distances, end_distances)

        # Let's also do some potential setup of the n_components functionality
        if n_stars_per_component is not None:
            # Cast an int into a longer list
            if isinstance(n_stars_per_component, int):
                n_stars_per_component = np.repeat(n_stars_per_component, len(start_parallaxes))

            # Cycle over all these n_components, adding them to a dictionary indexed with parallax range tuples
            self.n_stars_per_component = {}
            for a_components, a_start_parallax, a_end_parallax in zip(
                    n_stars_per_component, start_parallaxes, end_parallaxes):
                parallax_range = (a_start_parallax, a_end_parallax)
                self.n_stars_per_component[parallax_range] = a_components

        else:
            self.n_stars_per_component = None

        # Grab the level 5 healpix pixels and create a map that includes them
        self.level_5_pixels = np.append(central_pixel,
                                        hp.get_all_neighbours(2 ** 5, central_pixel, nest=True, lonlat=True))
        self.level_5_pixels = self.level_5_pixels[self.level_5_pixels != -1]

        self.base_map = np.zeros(12288, dtype=int)
        self.base_map[self.level_5_pixels] = 1.

        # Set the data and calculate any missing HEALPix levels
        self.data = data
        self._check_data(healpix_levels, healpix_overlaps)

        # Some final attributes we want the class to have
        self.central_pixel = central_pixel
        self._central_pixel_astropy_frame = _get_healpix_frame(central_pixel)

        self.total_partitions = 0
        self._stored_partitions = []

        self.current_partition = 0

        self.parallax_sigma_threshold = parallax_sigma_threshold

        self.current_core_pixels = None
        self.current_healpix_level = None
        self.current_healpix_string = None
        self.current_parallax_range = None
        self.current_n_stars = None

        self._stored_parallax_partitions = {}

        # Now, let's cycle over every partition and work out which healpix pixels it needs
        self.partitions = []
        self._calculate_partitions(healpix_levels, healpix_overlaps, start_parallaxes, end_parallaxes, minimum_size)
        self.minimum_partition_size = minimum_size

        if self.verbose:
            print(f"Created a dataset partitioning scheme!")

    def _calculate_partitions(self, healpix_levels, healpix_overlaps, start_parallaxes, end_parallaxes,
                              minimum_size: int = 10):
        """Calculates all partitions! Will size them up if necessary to make them bigger."""
        if self.verbose:
            print("Calculating the partitions themselves!")

        n_partitions = len(healpix_levels)
        i_partitions = 0
        first_run = True
        while i_partitions < n_partitions:
            if self.verbose and first_run:
                print(f"  attempting to calculate parallax range {i_partitions+1} of {n_partitions}")

            # Calculate this sub-partition
            partitions_to_append = self._calculate_sub_partition(
                healpix_levels[i_partitions], healpix_overlaps[i_partitions],
                start_parallaxes[i_partitions], end_parallaxes[i_partitions])

            # Temporarily spoof the partitions the class has
            n_new_partitions = len(partitions_to_append)
            n_old_partitions = self.total_partitions
            self.partitions += partitions_to_append
            self.total_partitions += n_new_partitions
            self._stored_partitions += [None] * self.total_partitions

            # Calculate the new ones
            counts = np.zeros(n_new_partitions)
            for i in range(n_new_partitions):
                a_partition = self.get_partition(n_old_partitions + i, return_data=False)
                counts[i] = np.count_nonzero(a_partition)

            # Calculate the minimum number of stars per component
            if self.n_stars_per_component is not None:
                minimum_stars = minimum_size * self.n_stars_per_component[self.current_parallax_range]
            else:
                minimum_stars = -1

            # See if this partitioning fails
            if np.any(counts < minimum_stars) and healpix_levels[i_partitions] is not None:
                first_run = False
                healpix_levels[i_partitions] -= 1

                if self.verbose:
                    print(f"    too few stars! Reducing HEALPix level to {healpix_levels[i_partitions]}")

                # See if we already have too few a HEALPix level and should move on
                if healpix_levels[i_partitions] < 5:
                    healpix_levels[i_partitions] = None

                # Otherwise, using one of the worst pieces of code I've ever written, we remove the stuff we just
                # calculated and decide to try again
                self.partitions = self.partitions[:n_old_partitions]
                self.total_partitions = n_old_partitions
                self._stored_partitions = self._stored_partitions[:n_old_partitions]

            else:
                first_run = True
                i_partitions += 1

        # Reset the partition counter - also making sure that the current information in the class is removed now that
        # setup is complete
        self.reset_partition_counter()

    def _calculate_sub_partition(self, core_level: Optional[int], overlap_level: Optional[int],
                                 start: float, end: float):
        """Calculates and returns the required sub-partitions for a certain partition as defined by input constraints.
        """
        parallax_range = (start, end)
        to_return = []

        # Work out how many separate sub-partitions this partition will have, the requisite HEALPix level to use,
        # and whether or not we need to also think about making an overlap
        # Case 1: we run over the whole field, within the parallax bin.
        if core_level is None:
            pixel_level = None
            to_return.append([5, None, None, parallax_range])

        # Cases 2 to 4 can be ran together.
        else:
            if core_level < 5:
                raise ValueError("the HEALPix level of core pixels must be greater than or equal to 5!")

            # Case 2: we run over a certain number of pixels individually, but without an overlap.
            if overlap_level is None:
                pixel_level = core_level
                overlap = False

            # Case 3: the user is a doofus
            elif overlap_level < core_level or overlap_level < 5:
                raise ValueError("the overlap level must be greater than or equal to the HEALPix level itself, and "
                                 "must be greater than or equal to 5.")

            # Case 4: we run over a certain number of pixels individually, *with* an overlap.
            else:
                pixel_level = overlap_level
                overlap = True

            # Get the pixels for cases 2/4, by...
            # Using our normal level 5 pixels for this field, or
            if core_level == 5:
                core_pixels_at_core_level = np.asarray(self.level_5_pixels)
            # Upgrading the map resolution to work out which pixels are a part of this
            else:
                core_pixels_at_core_level = np.nonzero(
                    hp.ud_grade(self.base_map, nside_out=2**core_level,
                                order_in='NESTED', order_out='NESTED', dtype=int))[0]

            # Then, downsample these core pixels if necessary
            if core_level != pixel_level:
                # We loop over all of the pixels, adding them in
                core_pixels_at_pixel_level = []
                for a_pixel in core_pixels_at_core_level:
                    test_map = np.zeros(hp.nside2npix(2**core_level), dtype=int)
                    test_map[a_pixel] = 1

                    core_pixels_at_pixel_level.append(np.nonzero(
                        hp.ud_grade(test_map, nside_out=2**pixel_level,
                                    order_in='NESTED', order_out='NESTED', dtype=int))[0])

            else:
                core_pixels_at_pixel_level = list(np.asarray(core_pixels_at_core_level).reshape(-1, 1))

            n_core_pixels = len(core_pixels_at_core_level)

            # Get all the overlap pixels if requested, by cycling over all core pixels
            if overlap:
                neighbor_pixels = []

                for i_core_pixel in range(n_core_pixels):
                    # Grab all neighbors for the core pixel in question, working at the underlying minimum pixel level
                    current_neighbors = np.asarray(
                        [hp.get_all_neighbours(2 ** pixel_level, a_pixel, nest=True, lonlat=True)
                         for a_pixel in core_pixels_at_pixel_level[i_core_pixel]]).flatten()

                    # Remove any invalid neighbors and make sure that the list is unique
                    current_neighbors = np.unique(current_neighbors[current_neighbors != -1])

                    # Remove any pixels that are in the list of core pixels already
                    neighbor_pixels.append(current_neighbors[
                        np.isin(current_neighbors, core_pixels_at_pixel_level[i_core_pixel], invert=True)])
            else:
                neighbor_pixels = [None] * n_core_pixels

            # FINALLY, append all this to the partition array
            for a_core, a_neighbor in zip(core_pixels_at_pixel_level, neighbor_pixels):
                to_return.append([pixel_level, a_core, a_neighbor, parallax_range])

        return to_return

    @staticmethod
    def _safe_distance_to_parallax(*args, milliarcseconds: bool=True):
        """Safely converts distances to parallax values and casts whatever input there is into numpy arrays."""
        if milliarcseconds:
            numerator = 1000.
        else:
            numerator = 1.

        # Grab the length and convert to a list to unlock mutability
        n_args = len(args)
        args = list(args)

        for i in range(n_args):
            # Ensure we can make it into an array of floats
            an_arg = np.asarray(args[i], dtype=float)

            # Make sure all distances are positive and not np.nan
            if np.any(an_arg) < 0:
                raise ValueError("input distances cannot be negative!")
            if np.any(np.isnan(an_arg)):
                raise ValueError("at least one input distance was not a number. Input distances "
                                 "must be 0 or positive (np.inf is allowed)")

            # Find where we have zeros or infs
            zeros = an_arg == 0.
            infs = an_arg == np.inf
            safe_values = np.invert(np.logical_or(zeros, infs))

            # Apply different operations to the different extremes
            an_arg[safe_values] = numerator / an_arg[safe_values]
            an_arg[zeros] = np.inf
            an_arg[infs] = -np.inf

            # Assign it right back at cha!
            args[i] = an_arg

        return args

    def _check_data(self, *healpix_levels):
        """Sets the data currently associated with the class and resets various internals.

        Args:
            *args: arrays of HEALPix levels that we need to calculate.

        """
        # Get unique HEALPix levels to calculate (removing any None values) from the input
        healpix_levels_to_calculate = np.atleast_1d([])
        for an_arg in healpix_levels:
            an_arg_numpy = np.atleast_1d(an_arg)
            good_values = an_arg_numpy != None
            healpix_levels_to_calculate = np.append(healpix_levels_to_calculate, an_arg_numpy[good_values])

        healpix_levels_to_calculate = np.unique(healpix_levels_to_calculate)

        # Calculate any requisite healpix data
        for a_level in healpix_levels_to_calculate:

            test_string = f"gaia_healpix_{a_level}"

            if test_string not in self.data.keys():
                self.data[test_string] = hp.ang2pix(2**a_level, self.data['ra'], self.data['dec'],
                                                    nest=True, lonlat=True)

        if self.verbose:
            print("  successfully set data to the class!")

    def reset_partition_counter(self):
        """Resets the information displaying which partition we're currently on."""
        self.current_partition = -1
        self.current_core_pixels = None
        self.current_healpix_level = None
        self.current_healpix_string = None
        self.current_parallax_range = None

    def next_partition(self, return_data: bool = True, reset_index: bool = True):
        """Gets the next data partition and checks that there's even a next one to get.

        Args:
            return_data (bool): whether or not to return a DataFrame (True) or a numpy array of bools for stars in
                self.data that are or aren't in this partition.
                Default: True
            reset_index (bool): if returning data, whether or not to reset the index for this new DataFrame view.
                Default: True

        Returns:
            the next partition, as a pd.DataFrame (if return_data is True) or a np.ndarray (if False).

        """
        self.current_partition += 1

        partition_number = self.current_partition

        if partition_number >= self.total_partitions or partition_number < 0:
            raise ValueError(f"Partition {partition_number} is out of range. Have you iterated this method too many "
                             f"times?")

        return self.get_partition(partition_number, return_data=return_data, reset_index=reset_index)

    def get_partition(self, partition_number: int, return_data: bool = True, reset_index: bool = True):
        """Returns a specific partition. It won't re-calculate the indexes of the partition if not necessary!

        Args:
            partition_number (int): number of the partition to return.
            return_data (bool): whether or not to return a DataFrame (True) or a numpy array of bools for stars in
                self.data that are or aren't in this partition.
                Default: True
            reset_index (bool): if returning data, whether or not to reset the index for this new DataFrame view.
                Default: True

        Returns:
            the next partition, as a pd.DataFrame (if return_data is True) or a np.ndarray (if False).

        """
        if partition_number >= self.total_partitions or partition_number < 0:
            raise ValueError(f"Cannot return partition {partition_number} as it is out of range of the total number of "
                             f"partitions, {self.total_partitions}.")

        self.current_healpix_level, self.current_core_pixels, neighbor_pixels, self.current_parallax_range = \
            self.partitions[partition_number]
        self.current_healpix_string = f"gaia_healpix_{self.current_healpix_level}"
        self.current_partition = partition_number

        if self._stored_partitions[partition_number] is None:

            # Only calculate good parallaxes if needed
            if self.current_parallax_range not in self._stored_parallax_partitions.keys():
                # Grab stuff we need
                parallax = self.data['parallax'].to_numpy()
                parallax_error = self.data['parallax_error'].to_numpy() * self.parallax_sigma_threshold

                # Test if it's in by default (easy)
                good_upper = parallax < self.current_parallax_range[0]
                good_lower = parallax > self.current_parallax_range[1]
                bad_upper = np.invert(good_upper)
                bad_lower = np.invert(good_lower)

                good_parallax = np.logical_and(good_upper, good_lower)

                # See if any of our friends ruined by the upper or lower limit are within it with error
                good_parallax[bad_upper] = np.logical_and(
                    parallax[bad_upper] - parallax_error[bad_upper]
                    < self.current_parallax_range[0], good_lower[bad_upper])

                good_parallax[bad_lower] = np.logical_and(
                    parallax[bad_lower] + parallax_error[bad_lower]
                    > self.current_parallax_range[1], good_upper[bad_lower])

                # Save this result!
                self._stored_parallax_partitions[self.current_parallax_range] = good_parallax.copy()

            else:
                good_parallax = self._stored_parallax_partitions[self.current_parallax_range].copy()

            # Now, calculate which stars are within the correct HEALPix pixels given this parallax result and
            # store it all!
            # Case when all stars are allowed
            if self.current_core_pixels is None:
                self._stored_partitions[partition_number] = good_parallax

            # Case when we don't have an overlap
            elif neighbor_pixels is None:
                good_parallax[good_parallax] = np.isin(
                    self.data.loc[good_parallax, self.current_healpix_string].to_numpy(),
                    self.current_core_pixels)
                self._stored_partitions[partition_number] = good_parallax

            # Or, the case when we have overlap pixels too
            else:
                good_parallax[good_parallax] = np.isin(
                    self.data.loc[good_parallax, self.current_healpix_string].to_numpy(),
                    np.append(self.current_core_pixels, neighbor_pixels))
                self._stored_partitions[partition_number] = good_parallax

        # Get & set the number of current stars in this partition
        self.current_n_stars = np.count_nonzero(self._stored_partitions[partition_number])

        # Return time!
        if return_data:
            if reset_index:
                return self.data.loc[self._stored_partitions[partition_number]].reset_index(drop=True)
            else:
                return self.data.loc[self._stored_partitions[partition_number]]
        else:
            return self._stored_partitions[partition_number]

    def delete_data(self):
        """Deletes the data after init for memory efficiency!"""
        # Say bye to the data (not using del, because that would delete the actual data too)
        self.data = None
        gc.collect()

    def test_if_in_current_partition(self,
                                     lon: Union[int, float, np.ndarray],
                                     lat: Union[int, float, np.ndarray],
                                     parallax: Union[int, float, np.ndarray],
                                     return_ra_dec: bool = False):
        """Tests whether or not clusters are within the bounds of the current partition. If not, then it's game over
        for this punk.

        Args:
            lon (int, float or np.ndarray): the longitude values (post-recentering.) Should have shape (n,).
            lat (int, float or np.ndarray): the latitude values (post-recentering.) Should have shape (n,).
            parallax (int, float or np.ndarray): the parallax values. Should have shape (n,).
            return_ra_dec: whether or not to also return arrays of the ra, dec values this function calculated
                internally by reversing the transform.
                Default: False

        Returns:
            an array of bools of shape (n,) as to whether or not the cluster is valid.
            if return_ra_dec:
                two more arrays, of the ra, dec respectively.

        """
        if self.current_partition < 0 or self.current_partition > self.total_partitions:
            raise ValueError("current partition number is invalid! Has get_partition() been called yet with this "
                             "partitioner? test_if_in_current_partition uses internals that are set by get_partition() "
                             "or next_partition().")

        # GOOD POSITIONS
        # Put all of the stars in the lon/lat frame and transform them
        coords = SkyCoord(lon, lat, unit='deg', frame=self._central_pixel_astropy_frame)
        coords = coords.transform_to('icrs')

        ra = coords.ra.value
        dec = coords.dec.value

        # Grab the current pixels we need to test against (
        if self.current_core_pixels is None:
            pixels_to_test_against = self.level_5_pixels
        else:
            pixels_to_test_against = self.current_core_pixels

        # Calculate the pixel id of each and if it's in the core pixel range
        pixels = hp.ang2pix(2**self.current_healpix_level, ra, dec, nest=True, lonlat=True)
        good_locations = np.isin(pixels, pixels_to_test_against)

        # GOOD PARALLAXES
        # Pretty nice. Just work out what is or isn't in the correct range.
        good_upper = parallax < self.current_parallax_range[0]
        good_lower = parallax > self.current_parallax_range[1]

        good_parallax = np.logical_and(good_upper, good_lower)

        # FNAL JOIN AND RETURN
        good_stars = np.logical_and(good_locations, good_parallax)

        if return_ra_dec:
            return good_stars, ra, dec
        else:
            return good_stars

    def get_n_components(self, minimum_n_components: int = 1) -> int:
        """Gets the value of n_components for the current partition, based on the current number of stars in the
        partition. Rounds to the nearest integer."""
        if self.n_stars_per_component is None:
            raise ValueError("n_components was not set during initialisation of this class so cannot be used!")
        elif self.current_parallax_range is None:
            raise ValueError("get_partition must be run first before this function, as it uses internals produced by "
                             "that function to calculate other things.")
        else:
            n_components = np.round(self.current_n_stars / self.n_stars_per_component[self.current_parallax_range])
            return int(np.clip(n_components, minimum_n_components, np.inf))

    def plot_partition_bar_chart(self, figure_title: Optional[str] = None, save_name: Optional[str] = None,
                                 show_figure: bool = True, dpi: int = 100, y_log: bool = True,
                                 maximum_parallax_for_cluster_radii: float = 10.,
                                 desired_radius: float = 10.,
                                 desired_size: float = 10,
                                 base_n_stars_per_component: int = 600):
        """Makes histograms showing the number of members of different constraint bins.

        Args:
            show_figure (bool): whether or not to show the figure at the end of plotting.
                Default: True
            save_name (string, optional): whether or not to save the figure at the end of plotting.
                Default: None (no figure is saved)
            figure_title (string, optional): the desired title of the figure.
                Default: None
            dpi (int): dpi of the figure.
                Default: 100
            y_log (bool): whether or not to make the y scale of the bar chart logarithmic.
                Default: True
            maximum_parallax_for_cluster_radii (str): the maximum start parallax to consider calculating a largest valid
                cluster radii for. Stops the plot from tending to inf for the first partition, basically.
                Default: 10. (aka 100pc away)
            desired_radius (float): point to mark on plot, in pc.
                Default: 10
            desired_size (float): point to mark on plot. I.e. the desired size of a bin as a multiplier of the number of
                stars per component of each bin.
                Default: 10
            base_n_stars_per_component (int): when calculating GMM runtimes, this is the assumed value that would
                otherwise be used if running on whole segments at once.
                Default: 600

        Returns:
            fig, ax (i.e. the figure and axis elements for you to mess with if desired)

        """
        if self.verbose:
            print("Plotting a bar chart of the number of members in each partition!")

        # Make a cute little bar chart with info of what's gone on
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=dpi)

        # We also grab the current default colourmap so that we can colour things by parallax cut
        total_count = 0
        all_counts = np.zeros(self.total_partitions)
        desired_counts = np.zeros(self.total_partitions)
        all_cluster_radii = np.zeros(self.total_partitions)
        all_n_components = np.ones(self.total_partitions)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = len(colors)
        i_color = -1

        # Get all the different partitions and group them by parallax
        last_parallax_cut = np.asarray([-10, -10])

        if self.verbose:
            print("  iterating over partitions...")

        for i_partition, a_partition in enumerate(self.partitions):

            # Check if it has the same parallax cut as the last partition
            if np.allclose(a_partition[3], last_parallax_cut):
                label = None

            else:
                label = f"{a_partition[3][0]:.2f} to {a_partition[3][1]:.2f}"
                last_parallax_cut = a_partition[3]

                # Increment the color
                if i_color < n_colors - 1:
                    i_color += 1
                else:
                    i_color = 0

            # Call get_partition and get count information
            count = np.count_nonzero(self.get_partition(i_partition, return_data=False))
            all_counts[i_partition] = count
            total_count += count

            desired_counts[i_partition] = self.n_stars_per_component[self.current_parallax_range] * desired_size

            # Also get information about the number of components, assuming that we're even using that functionality
            if self.n_stars_per_component is not None:
                all_n_components[i_partition] = self.get_n_components()

            # Also get information about the largest valid cluster radius of this pixel
            if a_partition[3][0] < maximum_parallax_for_cluster_radii:
                # Get the current pixels, dealing with if it's None
                current_pixels = a_partition[1]
                if current_pixels is None:
                    current_pixels = self.level_5_pixels

                all_cluster_radii[i_partition] = find_largest_cluster_in_pixel(
                    current_pixels, a_partition[0], a_partition[3][0])

            else:
                all_cluster_radii[i_partition] = np.nan

            # Plot the friend
            ax1.bar(i_partition, count, label=label, color=colors[i_color], zorder=0)

        # Also make a second axis so we can plot the cluster radii too
        ax2 = ax1.twinx()
        ax2.plot(np.arange(self.total_partitions), all_cluster_radii, 'ks', ms=3, zorder=100)

        # And add the desired values as horizontal lines
        desired_counts_repeated = np.repeat(desired_counts, 2)
        counts_x = np.repeat(np.arange(self.total_partitions), 2) + 0.5 * (-1)**(np.arange(self.total_partitions*2) + 1)
        ax1.plot(counts_x, desired_counts_repeated, 'k-', lw=3, zorder=100, alpha=0.5)

        fraction_of_total = self.total_partitions / 10
        ax2.plot([self.total_partitions - fraction_of_total, self.total_partitions + 1], [desired_radius] * 2,
                 'r-', lw=3, zorder=50)

        # Beautification
        ax1.legend(edgecolor='k', fontsize=8, title='parallaxes (mas)',
                   loc='lower left', bbox_to_anchor=(1.0, 1.0))
        ax1.set_xlabel("Partition number")
        ax1.set_ylabel("Bin count")
        ax2.set_ylabel("Maximum cluster radius at center (pc)")

        ax1.set_xlim(0 - 0.4, self.total_partitions - 0.6)

        if y_log:
            ax1.set_yscale('log')

        # Plot a title - we use text instead of the title api so that it can be long and multi-line.
        if figure_title is None:
            figure_title = ""

        initial_count = len(self.get_partition(0, return_data=False))
        runtime_fraction_nlogn = initial_count * np.log(initial_count) / np.sum(all_counts * np.log(all_counts))
        runtime_fraction_nsquared = initial_count**2 / np.sum(all_counts**2)
        runtime_fraction_gmm = (initial_count**2 / base_n_stars_per_component) / np.sum(all_counts * all_n_components)

        memory_fraction_n = np.max(all_counts) / initial_count
        memory_fraction_nsquared = np.max(all_counts)**2 / initial_count**2
        memory_fraction_gmm = np.max(all_counts * all_n_components) / (initial_count**2 / base_n_stars_per_component)

        count_passes = np.count_nonzero(all_counts > desired_counts)
        radius_passes = np.count_nonzero(all_cluster_radii[np.isfinite(all_cluster_radii)] > desired_radius)

        ax1.text(0., 1.05,
                 figure_title
                 + f"\ninitial stars:         {initial_count}"
                 + f"\npartitioned stars:     {total_count}"
                 + f"\ncount passes:          {count_passes:3d}  {count_passes / self.total_partitions:5.1%}"
                 + f"\nradius passes:         {radius_passes:3d}  {radius_passes / self.total_partitions:5.1%}\n"
                 + f"\ntime saving for nlogn: {runtime_fraction_nlogn:.2f}x faster"
                 + f"\ntime saving for n^2:   {runtime_fraction_nsquared:.2f}x faster"
                 + f"\ntime saving for GMM:   {runtime_fraction_gmm:.2f}x faster"
                 + f"\nRAM use for m ~ n:     {1 / memory_fraction_n:.2f}x less"
                 + f"\nRAM use for m ~ n^2:   {1 / memory_fraction_nsquared:.2f}x less"
                 + f"\nRAM use for GMM:       {1 / memory_fraction_gmm:.2f}x less"
                 + f"\n(assuming base GMM has {base_n_stars_per_component} stars / component)",
                 transform=ax1.transAxes, va="bottom",
                 family='monospace')

        # Output time
        if save_name is not None:
            fig.savefig(save_name, dpi=dpi, bbox_inches='tight')

        if show_figure is True:
            fig.show()

        if self.verbose:
            print("  done plotting the bar chart")

        return fig, ax1

    def plot_partitions(self, figure_title: Optional[str] = None, save_name: Optional[str] = None,
                        show_figure: bool = True, dpi: int = 100, **extra_args_for_plot):
        """Plots the individual parallax levels of partitions in a number of separate plots.

        Args:
            show_figure (bool): whether or not to show the figure at the end of plotting.
                Default: True
            save_name (string, optional): whether or not to save the figure at the end of plotting.
                Default: None (no figure is saved)
            figure_title (string, optional): the desired title of the figure.
                Default: None
            dpi (int): dpi of the figure.
                Default: 100
            extra_args_for_plot: extra arguments to pass to ocelot.plot.clustering_result.

        Returns:
            fig, ax (i.e. the figure and axis elements for you to mess with if desired)

        """
        if self.verbose:
            print("Making plots of each individual partition.")

        default_plot_args = {
            'cmd_plot_y_limits': [8, 18],
            'make_parallax_plot': True,
            'clip_to_fit_clusters': False,
            'plot_std_limit': 2.
        }
        default_plot_args.update(**extra_args_for_plot)

        # Make a mock labels array and a mock shades one
        labels = np.full((self.total_partitions, self.data.shape[0]), -1)

        # Update the labels array with the results of every partition
        if self.verbose:
            print("  iterating over partitions...")

        for i_partition in range(self.total_partitions):
            a_partition = self.get_partition(i_partition, return_data=False)
            labels[i_partition, a_partition] = i_partition

        # Cycle over all partitions, working out how many are in each parallax level. We make lists of their indices
        # for every different parallax entry in partition_numbers.
        partition_numbers = {}
        for i_partition, a_partition in enumerate(self.partitions):
            a_parallax_partition = a_partition[-1]

            # Update the partition number dict with this friend
            if a_parallax_partition in partition_numbers.keys():
                partition_numbers[a_parallax_partition].append(i_partition)
            else:
                partition_numbers[a_parallax_partition] = [i_partition]

        # Process the figure title and save name
        n_partitions = len(partition_numbers.keys())
        if save_name is not None:
            save_name = Path(save_name)
            prefix = save_name.parent / save_name.stem
            suffix = save_name.suffix
            save_name_list = [f"{prefix}_{x}{suffix}" for x in range(n_partitions)]
        else:
            save_name_list = [None] * n_partitions

        if figure_title is None:
            figure_title = ''
        else:
            figure_title = f'{figure_title}\n'

        # AND NOW, THE FUN REALLY BEGINS
        # Let's make some plots!
        for i_parallax_set, (a_parallax_set, a_save_name) in enumerate(zip(partition_numbers.keys(), save_name_list)):
            if self.verbose:
                print(f"  plotting partition {i_parallax_set + 1} of {n_partitions}")

            current_parallax_indices = partition_numbers[a_parallax_set]

            # Draw random values for every non-field label to decide which cluster to assign them to
            random_vals = np.where(labels[current_parallax_indices] == -1, -1,
                                   np.random.rand(len(current_parallax_indices), labels.shape[1]))

            labels_view = labels[current_parallax_indices]
            a_labels = labels_view[np.argmax(random_vals, axis=0), np.arange(labels.shape[1])]

            fig, ax = clustering_result(
                self.data,
                a_labels,
                current_parallax_indices,
                figure_title=
                f"{figure_title}Partitions for parallax set ({a_parallax_set[0]:.2f}, {a_parallax_set[0]:.2f})",
                save_name=a_save_name,
                dpi=dpi,
                show_figure=show_figure,
                **default_plot_args)

            if not show_figure:
                plt.close(fig)

        if self.verbose:
            print("  done with plotting individual partitions!")
