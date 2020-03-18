"""A number of functions for pre-processing Gaia data before clustering can begin."""

from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy

from sklearn.preprocessing import RobustScaler, StandardScaler
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.optimize import minimize


def cut_dataset(data_gaia: pd.DataFrame, parameter_cuts: Optional[dict] = None, geometric_cuts: Optional[dict] = None,
                return_cut_stars: bool = False, reset_index: bool = True) \
        -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
        return_cut_stars (bool): whether or not to also return a DataFrame containing *only* the stars that have
            been cut.
            Default: False
        reset_index (bool): whether or not to reset the indexes on the data frames to be returned. May not be intended
            behaviour if you're planning on re-combining the two DataFrames later.
            Default: True

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
        data_gaia_cut = data_gaia.loc[good_stars, :]

        if reset_index:
            data_gaia_cut = data_gaia_cut.reset_index(drop=True)

        if return_cut_stars:
            data_gaia_dropped_stars = data_gaia.loc[np.invert(good_stars), :]

            if reset_index:
                data_gaia_dropped_stars = data_gaia_dropped_stars.reset_index(drop=True)
        else:
            data_gaia_dropped_stars = None

    else:
        raise ValueError("You must specify parameter_cuts!")

    if geometric_cuts is not None:
        raise NotImplementedError("Sorry, geometric cutting isn't yet implemented here. Poke Emily to do something.")

    if return_cut_stars:
        return data_gaia_cut, data_gaia_dropped_stars
    else:
        return data_gaia_cut


def _rotation_function_lat(rotation: np.ndarray, coords: SkyCoord, center_coords: SkyCoord):
    """Returns the least squares value of the 0th/1st and 2nd/3rd latitudes for a given rotation."""
    # The astropy bits
    center_frame = center_coords.skyoffset_frame(rotation=rotation[0] * u.deg)
    coords = coords.transform_to(center_frame)
    lat = coords.lat.value

    # Grab the least squares differences
    return (lat[0] - lat[1]) ** 2 + (lat[2] - lat[3]) ** 2


default_healpy_kwargs = {
    'nest': True,
    'lonlat': True,
    'nside': 32  # i.e. 2**5
}


def _get_healpix_frame(pixel_id: int, rotate_frame: bool = True, **user_healpy_kwargs):
    """Returns an astropy SkyOffsetFrame given a healpix pixel id denoting the central pixel of the frame.

    Args:
        pixel_id (int): id of the healpix pixel.
        rotate_frame (bool): whether or not to rotate the frame so that the boundaries of the quadrilateral are roughly
            parallel to the latitude, longitude co-ordinate axes.
            Default: True
        **user_healpy_kwargs: kwargs for the healpy methods. You may change the following (shown with their defaults):
            'nest': True
            'lonlat': True
            'nside': 32  # i.e. 2**5

    Returns:
        an astropy SkyOffsetFrame centered on the center of the pixel.

    """
    # Grab arguments for healpy
    healpy_kwargs = default_healpy_kwargs
    healpy_kwargs.update(user_healpy_kwargs)

    # Get the center from the number of sides and the pixel id
    center = healpy.pix2ang(healpy_kwargs['nside'], pixel_id, nest=True, lonlat=True)
    center_coords = SkyCoord(center[0], center[1], frame='icrs', unit='deg')

    # Rotate the frame if requested
    if rotate_frame:
        # Get the four corners of the pixel
        corners_icrs = healpy.vec2ang(
            healpy.boundaries(healpy_kwargs['nside'], pixel_id, step=1, nest=healpy_kwargs['nest']).T,
            lonlat=healpy_kwargs['lonlat'])

        # Make some astropy SkyCoords to handle various bits and bobs and get the rotation angle
        coords = SkyCoord(ra=corners_icrs[0], dec=corners_icrs[1], unit='deg')
        result = minimize(_rotation_function_lat, np.asarray([0]), args=(coords, center_coords))

        # Check the result and cast the rotation angle correctly
        if not result.success:
            raise RuntimeError(f"failed to converge on an optimum rotation angle for pixel {pixel_id}!")
        rotation_angle = result.x[0] * u.deg

    else:
        rotation_angle = 0 * u.deg

    return center_coords.skyoffset_frame(rotation=rotation_angle)


def recenter_dataset(data_gaia: pd.DataFrame,
                     center: Optional[Union[tuple, list, np.ndarray]] = None,
                     center_type: str = 'icrs',
                     pixel_id: Optional[int] = None,
                     rotate_frame: bool = True,
                     proper_motion: bool = True,
                     **user_healpy_kwargs) -> pd.DataFrame:
    """Creates new arbitrary co-ordinate axes centred on centre, allowing for clustering analysis that doesn't get
    affected by distortions. N.B.: currently only able to use ra, dec from data_gaia!

    Args:
        data_gaia (pd.DataFrame): Gaia data, with the standard column names.
        center (list-like): array of length 2 with the ra, dec co-ordinates of the new centre. Must be specified if
            pixel_id is not specified.
            Default: None
        center_type (str): type of frame center is defined in. Must be acceptable by astropy.coordinates.SkyCoord. E.g.
            could be 'icrs' or 'galactic'.
            Default: 'icrs', i.e. center should be (ra, dec).
        pixel_id (int): if working with healpix pixels, this is the id of the central healpix pixel. Must be specified
            if center is not specified.
            Default: None
        rotate_frame (bool): if pixel_id is not None, then you can also ask to have the frame rotated. This will rotate
            the frame so that the boundaries of the quadrilateral pixel are roughly parallel to the latitude, longitude
            co-ordinate axes.
            Default: True
        proper_motion (bool): whether or not to also make transformed proper motions.
            Default: True
        user_healpy_kwargs: kwargs to change for passing to healpy. Will only do something if you're using a pixel_id
            instead of a field center. You may change the following (shown with their defaults):
            'nest': True
            'lonlat': True
            'nside': 32  # i.e. 2**5

    Returns:
        data_gaia, but now with lat, lon, (pmlat, pmlon) keys for the centered data.

    """
    # Deal with if we're using a healpix pixel and hence specify the center as a pixel number
    if center is None and pixel_id is not None:
        center_frame = _get_healpix_frame(pixel_id, rotate_frame=rotate_frame, **user_healpy_kwargs)

    # Otherwise, deal with the frame being user-specified
    elif center is not None and pixel_id is None:
        center_frame = SkyCoord(
            center[0], center[1], frame=center_type, unit=u.deg).skyoffset_frame()

    else:
        raise ValueError("you must specify one or the other of center and pixel_id. Not both or neither!")

    if proper_motion:
        coords = SkyCoord(ra=data_gaia['ra'].to_numpy() << u.deg,
                          dec=data_gaia['dec'].to_numpy() << u.deg,
                          pm_ra_cosdec=data_gaia['pmra'].to_numpy() << u.mas / u.yr,
                          pm_dec=data_gaia['pmdec'].to_numpy() << u.mas / u.yr)
    else:
        coords = SkyCoord(ra=data_gaia['ra'].to_numpy() << u.deg,
                          dec=data_gaia['dec'].to_numpy() << u.deg)

    # Apply the transform and save it
    coords = coords.transform_to(center_frame)

    data_gaia['lon'] = coords.lon.value
    data_gaia['lat'] = coords.lat.value

    if proper_motion:
        data_gaia['pmlon'] = coords.pm_lon_coslat.value
        data_gaia['pmlat'] = coords.pm_lat.value

    # Correct all values above 180 to simply be negative numbers
    data_gaia['lon'] = np.where(data_gaia['lon'] > 180, data_gaia['lon'] - 360, data_gaia['lon'])

    return data_gaia


def rescale_dataset(data_gaia: pd.DataFrame,
                    *args,
                    columns_to_rescale: Union[list, tuple] = ('ra', 'dec', 'pmra', 'pmdec', 'parallax'),
                    column_weights: Union[list, tuple] = (1., 1., 1., 1., 1.),
                    scaling_type: str = 'robust',
                    concatenate: bool = True,
                    return_scaler: bool = False,
                    **kwargs_for_scaler):
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
        return_scaler (bool): whether or not to also return the scaler object for future use.
            Default: False
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
        to_return = np.concatenate(to_return, axis=0)

    # Or, just return the one array if no args were specified (slightly ugly code so my API doesn't break lololol)
    elif len(to_return) == 1:
        to_return = to_return[0]

    # Or, return all arrays separately if we aren't concatenating and have multiple arrays (keeps to_return as is)
    # Now we just need to decide on whether or not to also return the scaler.
    if return_scaler:
        return to_return, scaler
    else:
        return to_return


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
    [4, 1500],
    [9, np.inf]
]


class DataPartition:
    def __init__(self, constraints: Optional[Union[list, tuple, np.ndarray]] = None,
                 shape: Union[list, tuple, np.ndarray] = (5, 5),
                 tile_overlap: float = 10.,
                 parallax_sigma_threshold: float = 2.,
                 minimum_area_fraction: float = 2.,
                 tidal_radius: float = 10.,
                 verbose: bool = True):
        """A class for creating Gaia dataset partitions. Checks the quality of the constraints and writes them to the
        class after some processing.

        WARNING: Currently only supports squares! Rectangular fields may have unwanted effects.

        Args:
            constraints (list-like, optional): the constraints array, of shape (n_distances, 2), where the first entries are the
                number of times to tile this partition, and the second number is the final distance of the bin. np.inf
                specifies a bin that's endlessly long.
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

        self.first_valid_distance = check_constraints(
            constraints, sky_area, minimum_area_fraction=minimum_area_fraction, tidal_radius=tidal_radius)

        # Calculate co-ordinates defining the maximum vertices of the partition
        half_x = shape[0] / 2
        half_y = shape[1] / 2

        # Cycle over all the constraints, making us some new partitions in the form of cut dicts
        self.partitions = []

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
                self.partitions.append([a_x, a_y, [start_parallax, end_parallax]])

            start_distance = a_constraint[1]
            start_parallax = end_parallax

        # Turn the partitions list into a numpy array for easier indexing later
        # Where indexes go as [partition_number, [x, y, parallax], [start, end]]
        # i.e. shape (total_partitions, 3, 2)
        self.partitions = np.asarray(self.partitions)

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
