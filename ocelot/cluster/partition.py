"""Classes that will partition a dataset into multiple pieces."""

from typing import Optional, Union

import numpy as np
import healpy as hp
import pandas as pd
from matplotlib import pyplot as plt


def _tidal_radius_of_cluster(distance, tidal_radius):
    """Returns degrees from parsecs."""
    return np.arctan(tidal_radius / distance) * 180 / np.pi


def _sky_area_of_cluster(distance, tidal_radius):
    """Unit is in square degrees!"""
    tidal_radius_in_degrees = _tidal_radius_of_cluster(distance, tidal_radius)
    return np.pi * tidal_radius_in_degrees ** 2


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
                 central_pixel: int,
                 constraints: Optional[Union[list, tuple, np.ndarray]] = None,
                 final_distance: float = np.inf,
                 parallax_sigma_threshold: float = 2.,
                 verbose: bool = True):
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
            verbose (bool): whether or not to print a couple of info things to the console upon creation.
                Default: True

        """
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

        # Grab the level 5 healpix pixels and create a map that includes them
        self.level_5_pixels = np.append(central_pixel,
                                        hp.get_all_neighbours(2 ** 5, central_pixel, nest=True, lonlat=True))
        self.level_5_pixels = self.level_5_pixels[self.level_5_pixels != -1]

        self.base_map = np.zeros(12288, dtype=int)
        self.base_map[self.level_5_pixels] = 1.

        # Now, let's cycle over every partition and work out which healpix pixels it needs
        self.partitions = []
        self.healpix_levels_to_calculate = []

        for a_level, an_overlap, a_start, an_end in zip(
                healpix_levels, healpix_overlaps, start_parallaxes, end_parallaxes):
            self._calculate_sub_partition(a_level, an_overlap, a_start, an_end)

        # Some final attributes we want the class to have
        self.total_partitions = len(self.partitions)
        self._stored_partitions = [None] * self.total_partitions
        self.current_partition = 0
        self.data = None
        self.parallax_sigma_threshold = parallax_sigma_threshold

        self.current_core_pixels = None
        self.current_healpix_string = None

        self._stored_parallax_partitions = {}

        if verbose:
            print(f"Created a dataset partitioning scheme!")

    def _calculate_sub_partition(self, core_level: Optional[int], overlap_level: Optional[int],
                                 start: float, end: float):
        """Calculates and returns the required sub-partitions for a certain partition as defined by input constraints.
        """
        parallax_range = (start, end)

        # Work out how many separate sub-partitions this partition will have, the requisite HEALPix level to use,
        # and whether or not we need to also think about making an overlap
        # Case 1: we run over the whole field, within the parallax bin.
        if core_level is None:
            pixel_level = None
            self.partitions.append([5, None, None, parallax_range])

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
                self.partitions.append([pixel_level, a_core, a_neighbor, parallax_range])

        # Make sure that this HEALPix level will be calculated in the data
        if pixel_level not in self.healpix_levels_to_calculate and pixel_level is not None:
            self.healpix_levels_to_calculate.append(pixel_level)

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

    def plot_partitions(self, figure_title: Optional[str] = None, save_name: Optional[str] = None,
                        show_figure: bool = True, dpi: int = 100, y_log: bool = True,):
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
            y_log (bool): whether or not to make the y scale logarithmic.
                Default: True

        Returns:
            fig, ax (i.e. the figure and axis elements for you to mess with if desired)

        """
        # Check that the user isn't a Total Biscuit
        if self.data is None:
            raise ValueError("The class currently has no data so there's nothing to plot! Please call the method "
                             "set_data first and assign me a DataFrame to work with.")

        # Make a cute little bar chart with info of what's gone on
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=dpi)

        # We also grab the current default colourmap so that we can colour things by parallax cut
        total_count = 0
        all_counts = np.zeros(self.total_partitions)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = len(colors)
        i_color = -1

        # Get all the different partitions and group them by parallax
        last_parallax_cut = np.asarray([-10, -10])

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

            # Plot the friend
            count = np.count_nonzero(self.get_partition(i_partition, return_data=False))
            all_counts[i_partition] = count
            total_count += count
            ax.bar(i_partition, count, label=label, color=colors[i_color])

        # Beautification
        ax.legend(edgecolor='k', fontsize=8, title='parallaxes (mas)',
                  loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_xlabel("Partition number")
        ax.set_ylabel("Bin count")

        if y_log:
            ax.set_yscale('log')

        # Plot a title - we use text instead of the title api so that it can be long and multi-line.
        if figure_title is None:
            figure_title = ""

        initial_count = self.data.shape[0]
        runtime_fraction_nlogn = initial_count * np.log(initial_count) / np.sum(all_counts * np.log(all_counts))
        runtime_fraction_nsquared = initial_count**2 / np.sum(all_counts**2)

        ax.text(0., 1.10,
                figure_title + f"\ninitial stars: {initial_count}\npartitioned stars: {total_count}"
                + f"\ntime saving for nlogn algorithm: {runtime_fraction_nlogn:.2f}x faster"
                + f"\ntime saving for n^2 algorithm: {runtime_fraction_nsquared:.2f}x faster",
                transform=ax.transAxes, va="bottom")

        # Output time
        if save_name is not None:
            fig.savefig(save_name, dpi=dpi)

        if show_figure is True:
            fig.show()

        return fig, ax

    def set_data(self, data: pd.DataFrame):
        """Sets the data currently associated with the class and resets various internals.

        Args:
            data (pd.DataFrame): the dataframe to set. *Must* have arguments 'lat', 'lon', 'parallax' and
                'parallax_error'.
        """
        self.reset_partition_counter()
        self._stored_partitions = [None] * self.total_partitions
        self.data = data

        # Calculate any requisite healpix data
        for a_level in self.healpix_levels_to_calculate:

            test_string = f"gaia_healpix_{a_level}"

            if test_string not in self.data.keys():
                self.data[test_string] = hp.ang2pix(2**a_level, self.data['ra'], self.data['dec'],
                                                    nest=True, lonlat=True)

    def reset_partition_counter(self):
        """Resets the counter displaying which partition we're currently on."""
        self.current_partition = 0

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

        partition_number = self.current_partition

        if partition_number >= self.total_partitions:
            raise ValueError(f"Partition {partition_number} is out of range. Have you iterated this method too many "
                             f"times?")

        self.current_partition += 1

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

        healpix_level, self.current_core_pixels, neighbor_pixels, par_range = self.partitions[partition_number]
        self.current_healpix_string = f"gaia_healpix_{healpix_level}"

        if self._stored_partitions[partition_number] is None:

            # Only calculate good parallaxes if needed
            if par_range not in self._stored_parallax_partitions.keys():
                # Grab stuff we need
                parallax = self.data['parallax'].to_numpy()
                parallax_error = self.data['parallax_error'].to_numpy() * self.parallax_sigma_threshold

                # Test if it's in by default (easy)
                good_upper = parallax < par_range[0]
                good_lower = parallax > par_range[1]
                bad_upper = np.invert(good_upper)
                bad_lower = np.invert(good_lower)

                good_parallax = np.logical_and(good_upper, good_lower)

                # See if any of our friends ruined by the upper or lower limit are within it with error
                good_parallax[bad_upper] = np.logical_and(
                    parallax[bad_upper] - parallax_error[bad_upper] < par_range[0], good_lower[bad_upper])

                good_parallax[bad_lower] = np.logical_and(
                    parallax[bad_lower] + parallax_error[bad_lower] > par_range[1], good_upper[bad_lower])

                # Save this result!
                self._stored_parallax_partitions[par_range] = good_parallax.copy()

            else:
                good_parallax = self._stored_parallax_partitions[par_range].copy()

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

        if return_data:
            if reset_index:
                return self.data.loc[self._stored_partitions[partition_number]].reset_index(drop=True)
            else:
                return self.data.loc[self._stored_partitions[partition_number]]
        else:
            return self._stored_partitions[partition_number]

    def generate_all_partitions(self, delete_data: bool = False):
        """Pre-generates all partitions for the current data and resets the partition counter.

        Args:
            delete_data (bool): whether or not to also remove the class's local version of the input data. This is
                useful for memory-efficient clustering analysis.
                Default: False

        """
        # Pre-get all partitions
        for i in range(self.total_partitions):
            self.get_partition(i, return_data=False)

        # Say bye to the data (not using del, because that would delete the attribute too)
        if delete_data:
            self.data = None

        # We also reset the partition counter. We were never here!
        self.reset_partition_counter()
