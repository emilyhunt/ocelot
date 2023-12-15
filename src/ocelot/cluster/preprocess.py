"""A number of functions for pre-processing data before clustering can begin."""

from typing import Optional, Union, Tuple, List

import numpy as np
import pandas as pd
import healpy

from sklearn.preprocessing import RobustScaler, StandardScaler
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.optimize import minimize


def cut_dataset(
    data_gaia: pd.DataFrame,
    parameter_cuts: Optional[dict] = None,
    geometric_cuts: Optional[dict] = None,
    return_cut_stars: bool = False,
    reset_index: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
                raise ValueError(
                    "Incorrect type of range found in parameter_cuts! All ranges must be specified as "
                    "Python lists or np.ndarray arrays."
                )

            # Find the matches to the parameter in data_gaia (fyi: it's a lot faster to convert to numpy first. lol)
            # We then only update the good_stars indexes that are already True, so that we don't overwrite existing
            # fails and also only do the minimum amount of calculation.
            column_to_test = data_gaia.loc[good_stars, a_parameter].to_numpy()
            good_stars[good_stars] = np.logical_and(
                column_to_test > a_parameter_cut[0], column_to_test < a_parameter_cut[1]
            )

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
        raise NotImplementedError(
            "Sorry, geometric cutting isn't yet implemented here. Poke Emily to do something."
        )

    if return_cut_stars:
        return data_gaia_cut, data_gaia_dropped_stars
    else:
        return data_gaia_cut


def _rotation_function_lat(
    rotation: np.ndarray, coords: SkyCoord, center_coords: SkyCoord
):
    """Returns the least squares value of the 0th/1st and 2nd/3rd latitudes for a given rotation."""
    # The astropy bits
    center_frame = center_coords.skyoffset_frame(rotation=rotation[0] * u.deg)
    coords = coords.transform_to(center_frame)
    lat = coords.lat.value

    # Grab the least squares differences
    return (lat[0] - lat[1]) ** 2 + (lat[2] - lat[3]) ** 2


default_healpy_kwargs = {
    "nest": True,
    "lonlat": True,
    "nside": 32,  # i.e. 2**5
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
    center = healpy.pix2ang(healpy_kwargs["nside"], pixel_id, nest=True, lonlat=True)
    center_coords = SkyCoord(center[0], center[1], frame="icrs", unit="deg")

    # Rotate the frame if requested
    if rotate_frame:
        # Get the four corners of the pixel
        corners_icrs = healpy.vec2ang(
            healpy.boundaries(
                healpy_kwargs["nside"], pixel_id, step=1, nest=healpy_kwargs["nest"]
            ).T,
            lonlat=healpy_kwargs["lonlat"],
        )

        # Make some astropy SkyCoords to handle various bits and bobs and get the rotation angle
        coords = SkyCoord(ra=corners_icrs[0], dec=corners_icrs[1], unit="deg")
        result = minimize(
            _rotation_function_lat, np.asarray([0]), args=(coords, center_coords)
        )

        # Check the result and cast the rotation angle correctly
        if not result.success:
            raise RuntimeError(
                f"failed to converge on an optimum rotation angle for pixel {pixel_id}!"
            )
        rotation_angle = result.x[0] * u.deg

    else:
        rotation_angle = 0 * u.deg

    return center_coords.skyoffset_frame(rotation=rotation_angle)


def recenter_dataset(
    *args,
    center: Optional[Union[tuple, list, np.ndarray]] = None,
    center_type: str = "icrs",
    pixel_id: Optional[int] = None,
    rotate_frame: bool = True,
    proper_motion: bool = True,
    always_return_list: bool = False,
    **user_healpy_kwargs,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Creates new arbitrary co-ordinate axes centred on centre, allowing for clustering analysis that doesn't get
    affected by distortions. N.B.: currently only able to use ra, dec from data_gaia!

    Args:
        *args (pd.DataFrame): Gaia dataframes to apply the transform to.
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
        always_return_list (bool): for backwards-compatibility, we always return one DataFrame when only passed one.
            However, setting this to true will make it a one-element list of DataFrames.
            Default: False
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
        center_frame = _get_healpix_frame(
            pixel_id, rotate_frame=rotate_frame, **user_healpy_kwargs
        )

    # Otherwise, deal with the frame being user-specified
    elif center is not None and pixel_id is None:
        center_frame = SkyCoord(
            center[0], center[1], frame=center_type, unit=u.deg
        ).skyoffset_frame()

    else:
        raise ValueError(
            "you must specify one or the other of center and pixel_id. Not both or neither!"
        )

    to_return = []

    for data_gaia in args:
        if proper_motion:
            coords = SkyCoord(
                ra=data_gaia["ra"].to_numpy() << u.deg,
                dec=data_gaia["dec"].to_numpy() << u.deg,
                pm_ra_cosdec=data_gaia["pmra"].to_numpy() << u.mas / u.yr,
                pm_dec=data_gaia["pmdec"].to_numpy() << u.mas / u.yr,
            )
        else:
            coords = SkyCoord(
                ra=data_gaia["ra"].to_numpy() << u.deg,
                dec=data_gaia["dec"].to_numpy() << u.deg,
            )

        # Apply the transform and save it
        coords = coords.transform_to(center_frame)

        data_gaia["lon"] = coords.lon.value
        data_gaia["lat"] = coords.lat.value

        if proper_motion:
            data_gaia["pmlon"] = coords.pm_lon_coslat.value
            data_gaia["pmlat"] = coords.pm_lat.value

        # Correct all values above 180 to simply be negative numbers
        data_gaia["lon"] = np.where(
            data_gaia["lon"] > 180, data_gaia["lon"] - 360, data_gaia["lon"]
        )

        to_return.append(data_gaia)

    if len(to_return) == 1 and always_return_list is False:
        return to_return[0]
    else:
        return to_return


def rescale_dataset(
    data_gaia: pd.DataFrame,
    *args,
    columns_to_rescale: Union[list, tuple] = ("ra", "dec", "pmra", "pmdec", "parallax"),
    column_weights: Union[list, tuple] = (1.0, 1.0, 1.0, 1.0, 1.0),
    scaling_type: str = "robust",
    concatenate: bool = True,
    return_scaler: bool = False,
    **kwargs_for_scaler,
):
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
    if scaling_type == "robust":
        scaler = RobustScaler(**kwargs_for_scaler)
    elif scaling_type == "standard":
        scaler = StandardScaler(**kwargs_for_scaler)
    else:
        raise ValueError("Selected scaling_type not supported!")

    # Check that all values are finite
    if not np.isfinite(data_gaia[columns_to_rescale].to_numpy()).all():
        raise ValueError(
            "At least one value in data_gaia is not finite! Unable to rescale the data."
        )

    # Apply the scaling & multiply by column weights
    to_return = [
        scaler.fit_transform(data_gaia[columns_to_rescale].to_numpy()) * column_weights
    ]

    # Apply the same scaling scaling to any other data frames, if others have been specified
    for an_extra_cluster in args:
        # Check that all values are finite
        if not np.isfinite(an_extra_cluster[columns_to_rescale].to_numpy()).all():
            raise ValueError(
                "At least one value in args is not finite! Unable to rescale the data."
            )

        # Re-scale!
        to_return.append(
            scaler.transform(an_extra_cluster[columns_to_rescale].to_numpy())
            * column_weights
        )

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
