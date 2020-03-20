"""Classes that will partition a dataset into multiple pieces."""

from typing import Optional, Union

import numpy as np
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
    # Set constraints into a numpy array
    constraints = np.asarray(constraints)

    # Error checking
    if constraints.ndim != 2:
        raise ValueError("constraints must have shape (n_distances, 2)")
    if constraints.shape[1] != 2:
        raise ValueError("constraints must have shape (n_distances, 2)")

    # Check that all the distances are strictly increasing and that the last one is inf
    if np.any(constraints[1:, 1] <= constraints[:-1, 1]):
        raise ValueError("constraints must have strictly increasing distances.")
    if constraints[-1, 1] != np.inf:
        raise ValueError("the last tile must have an infinite end distance, aka np.inf.")

    # Check that all numbers of bins are valid
    if np.any(constraints[:, 0] < 1):
        raise ValueError("sky areas must be divided into at least one box.")

    n_distances = constraints.shape[0]

    # Calculate the area at all points and check them, in a way that can handle divisions by zero
    cluster_areas = _sky_area_of_cluster(constraints[:-1, 1], tidal_radius=tidal_radius)
    tile_areas = sky_area / constraints[:, 0]
    nonzero_areas = cluster_areas != 0

    start_fractions = np.zeros(n_distances)
    start_fractions[0] = np.inf  # We ignore the first start, as we always start at 0 distance
    start_fractions[1:] = tile_areas[1:] / cluster_areas[:]

    end_fractions = np.zeros(n_distances)
    end_fractions = tile_areas[:-1] / cluster_areas[:]
    end_fractions[-1] = np.inf  # We ignore the last bin too, as it's always np.inf

    # Area checking time!
    bad_starts = start_fractions < minimum_area_fraction
    bad_ends = end_fractions < minimum_area_fraction
    if np.any(bad_ends) or np.any(bad_starts):
        raise ValueError(f"constraints {bad_starts.nonzero()[0]} have start area excesses below the "
                         f"minimum_area_fraction, and {bad_ends.nonzero()[0]} also have final area excesses below "
                         f"the minimum_area_fraction. All excess areas: \n"
                         f"start: {start_fractions}\nend: {end_fractions}")

    # Calculate the first valid distance
    first_valid_distance = tidal_radius / np.tan(np.sqrt(tile_areas[0] * np.pi / 180 ** 2 / minimum_area_fraction))

    return first_valid_distance, start_fractions, end_fractions


default_partition = [
    [1, 800],
    [1, 1500],
    [9, np.inf]
]


class DataPartition:
    def __init__(self, sky_area: float,
                 partitions,
                 constraints: Union[list, tuple, np.ndarray],
                 parallax_sigma_threshold: float = 2.,
                 minimum_area_fraction: float = 2.,
                 tidal_radius: float = 10.,
                 verbose: bool = True):
        """Superclass for data partitions. Does a bare minimum of error checking!

        Args:
            sky_area (float): the total area of the data on the sky.
            partitions (list): an array of the partitions
            tile_overlap (float): overlap (in parsecs) to make between the start of tiles. You will need to set this
                lower if you have a large number of partitions or want to minimise the number of bin members.
                Default: 10. (should prevent edge effects on all but the most... edgey clusters)
            parallax_sigma_threshold (float): how much of the parallax error to consider when deciding whether or not to
                include stars. If zero, error is not considered at all and there won't be any overlap. If large, then
                most stars will be in every parallax bin.
                Default: 2. (i.e. ~90% of stars actually in a bin but outside of it within error will end up in the bin)
            minimum_area_fraction (float): minimum tile_area / cluster_area allowed at each distance. If any tiles have
                area fractions below this, this class will raise a ValueError. Will need to be set lower for small
                fields or if you don't care about preventing edge effects.
                Default: 2.
            tidal_radius (float): tidal radius of the test cluster to consider.
                Default: 10. (a decent average number for open clusters based on MWSC.)
            verbose (bool): whether or not to print a couple of info things to the console upon creation.
                Default: True

        """
        # Error checking!
        self.first_valid_distance = check_constraints(
            constraints, sky_area, minimum_area_fraction=minimum_area_fraction, tidal_radius=tidal_radius)

        # Turn the partitions list into a numpy array for easier indexing later
        # Where indexes go as [partition_number, [x, y, parallax], [start, end]]
        # i.e. shape (total_partitions, 3, 2)
        self.partitions = np.asarray(partitions)

        # Some final attributes we want the class to have
        self.total_partitions = self.partitions.shape[0]
        self._stored_partitions = [None] * self.total_partitions
        self.current_partition = 0
        self.data = None
        self.parallax_sigma_threshold = parallax_sigma_threshold

        if verbose:
            print(f"Created a dataset partitioning scheme. Its first valid distance is at "
                  f"{self.first_valid_distance:.2f}pc.\n Clusters nearer this are liable to suffer edge effects as the "
                  f"data region isn't large enough!")

    @staticmethod
    def _safe_square_root(number, precision=8):
        """Safely takes a square root and checks that the original number was indeed a square."""
        sqrt_number = np.sqrt(number)

        if np.round(sqrt_number % 1, decimals=precision) != 0.0:
            raise ValueError("numbers of partitions must be square numbers, as this function currently only "
                             "supports square partitions.")
        else:
            return int(sqrt_number)

    def plot_partitions(self, figure_title: Optional[str] = None, save_name: Optional[str] = None,
                        show_figure: bool = True, dpi: int = 100):
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
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = len(colors)
        i_color = -1

        # Get all the different partitions and group them by parallax
        last_parallax_cut = np.asarray([-10, -10])

        for i_partition, a_partition in enumerate(self.partitions):

            # Check if it has the same parallax cut as the last partition
            if np.allclose(a_partition[2], last_parallax_cut):
                label = None

            else:
                label = f"{a_partition[2, 0]:.2f} to {a_partition[2, 1]:.2f}"
                last_parallax_cut = a_partition[2]

                # Increment the color
                if i_color < n_colors - 1:
                    i_color += 1
                else:
                    i_color = 0

            # Plot the friend
            count = np.count_nonzero(self.get_partition(i_partition, return_data=False))
            total_count += count
            ax.bar(i_partition, count, label=label, color=colors[i_color])

        # Beautification
        ax.legend(edgecolor='k', fontsize=8, title='parallaxes (mas)',
                  loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_xlabel("Partition number")
        ax.set_ylabel("Bin count")

        # Plot a title - we use text instead of the title api so that it can be long and multi-line.
        if figure_title is None:
            figure_title = ""

        ax.text(0., 1.10,
                figure_title + f"\ninitial stars: {self.data.shape[0]}\npartitioned stars: {total_count}",
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
        """Blank function. Should be over-written by subclasses."""
        return np.asarray([True])

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


class SquareDataPartition(DataPartition):
    def __init__(self, constraints: Optional[Union[list, tuple, np.ndarray]] = None,
                 shape: Union[list, tuple, np.ndarray] = (5, 5),
                 tile_overlap: float = 10.,
                 parallax_sigma_threshold: float = 2.,
                 minimum_area_fraction: float = 2.,
                 tidal_radius: float = 10.,
                 verbose: bool = True):
        """A class for creating Gaia dataset partitions. Checks the quality of the constraints and writes them to the
        class after some processing. Makes square partitions!

        Args:
            constraints (list-like, optional): the constraints array, of shape (n_distances, 2), where the first entries
                are the number of times to tile this partition, and the second number is the final distance of the bin.
                np.inf specifies a bin that's endlessly long.
                Default: ocelot.cluster.preprocess.default_partition
            shape (list-like): shape of the field, length 2. Must be square:
                Default: (5, 5)
            tile_overlap (float): overlap (in parsecs) to make between the start of tiles. You will need to set this
                lower if you have a large number of partitions or want to minimise the number of bin members.
                Default: 10. (should prevent edge effects on all but the most... edgey clusters)
            parallax_sigma_threshold (float): how much of the parallax error to consider when deciding whether or not to
                include stars. If zero, error is not considered at all and there won't be any overlap. If large, then
                most stars will be in every parallax bin.
                Default: 2. (i.e. ~90% of stars actually in a bin but outside of it within error will end up in the bin)
            minimum_area_fraction (float): minimum tile_area / cluster_area allowed at each distance. If any tiles have
                area fractions below this, this class will raise a ValueError. Will need to be set lower for small
                fields or if you don't care about preventing edge effects.
                Default: 2.
            tidal_radius (float): tidal radius of the test cluster to consider.
                Default: 10. (a decent average number for open clusters based on MWSC.)
            verbose (bool): whether or not to print a couple of info things to the console upon creation.
                Default: True

        """
        # Check that the input constraints aren't total BS
        if constraints is None:
            constraints = default_partition

        sky_area = shape[0] * shape[1]

        # Calculate co-ordinates defining the maximum vertices of the partition
        half_x = shape[0] / 2
        half_y = shape[1] / 2

        # Cycle over all the constraints, making us some new partitions in the form of cut dicts
        partitions = []

        start_distance = 0.0001
        start_parallax = np.inf
        for a_constraint in constraints:

            # Get stats we need
            n = a_constraint[0]
            sqrt_n = self._safe_square_root(n)
            safety_factor = _tidal_radius_of_cluster(start_distance, tile_overlap)

            # If the end distance is np.inf, then we should take the hint and make the end distance -np.inf to allow
            # for negative parallaxes
            if a_constraint[1] != np.inf:
                end_parallax = 1000 / a_constraint[1]  # In mas
            else:
                end_parallax = -np.inf

            # Make ranges of partition bins and turn these into starts and ends with some clever indexing
            x_range = np.linspace(-half_x, half_x, num=sqrt_n + 1)
            y_range = np.linspace(-half_y, half_y, num=sqrt_n + 1)

            start_x, start_y = np.meshgrid(x_range[:-1], y_range[:-1])
            end_x, end_y = np.meshgrid(x_range[1:], y_range[1:])

            # Join up the starts and ends into neater arrays
            x_partition_cuts = np.vstack([start_x.flatten(), end_x.flatten()]).T
            y_partition_cuts = np.vstack([start_y.flatten(), end_y.flatten()]).T

            # Work out whichever ones aren't edges (as we don't want to be larger than these) to add the safety factor
            x_not_an_edge = np.abs(x_partition_cuts) != half_x
            y_not_an_edge = np.abs(y_partition_cuts) != half_y
            x_partition_cuts[:, 0] -= safety_factor * x_not_an_edge[:, 0]
            x_partition_cuts[:, 1] += safety_factor * x_not_an_edge[:, 1]
            y_partition_cuts[:, 0] -= safety_factor * y_not_an_edge[:, 0]
            y_partition_cuts[:, 1] += safety_factor * y_not_an_edge[:, 1]

            # Clip the partitions + safety to not be larger than desired (sometimes large safety margins can make 'em
            # larger than the field itself)
            x_partition_cuts = np.clip(x_partition_cuts, -half_x, half_x)
            y_partition_cuts = np.clip(y_partition_cuts, -half_y, half_y)

            # Finally, add this partition to the class
            for a_x, a_y in zip(x_partition_cuts, y_partition_cuts):
                partitions.append([a_x, a_y, [start_parallax, end_parallax]])

            start_distance = a_constraint[1]
            start_parallax = end_parallax

        # And lastly, let's do a super call
        super().__init__(sky_area,
                         partitions,
                         constraints,
                         parallax_sigma_threshold=parallax_sigma_threshold,
                         minimum_area_fraction=minimum_area_fraction,
                         tidal_radius=tidal_radius,
                         verbose=verbose)

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

        lon_range, lat_range, par_range = self.partitions[partition_number]

        if self._stored_partitions[partition_number] is None:
            good_positions = np.logical_and(
                np.logical_and(self.data['lon'] > lon_range[0], self.data['lon'] < lon_range[1]),
                np.logical_and(self.data['lat'] > lat_range[0], self.data['lat'] < lat_range[1]))

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

            # Store time!
            self._stored_partitions[partition_number] = np.logical_and(good_parallax, good_positions)

        if return_data:
            if reset_index:
                return self.data.loc[self._stored_partitions[partition_number]].reset_index(drop=True)
            else:
                return self.data.loc[self._stored_partitions[partition_number]]
        else:
            return self._stored_partitions[partition_number]
