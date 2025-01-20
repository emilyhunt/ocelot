"""A class for crossmatching clusters with literature catalogues."""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import norm
from typing import Optional, Union, List, Tuple


class Catalogue:
    # Todo this class needs refactoring and overhauling. It's... messy...
    def __init__(
        self,
        data: pd.DataFrame,
        catalogue_name: str,
        key_name: str = "Name",
        key_ra: str = "ra",
        key_dec: str = "dec",
        key_tidal_radius: str = "ang_radius_t",
        extra_axes: Optional[Union[list, tuple, np.ndarray]] = None,
        assumed_position_systematic_error: float = 0.0,
        match_tidal_radius: bool = False,
    ):
        """Storage system for existing catalogues that handles cross-matching too.

        Args:
            data (pd.DataFrame or pd.Series if just one cluster): data for the catalogue to match against.
            catalogue_name (str): name of the catalogue. This is stored and used to
            key_name (str): name of the name column in data.
                Default: "Name"
            key_ra (str): name of the ra column in data.
                Default: "ra"
            key_dec (str): name of the dec column in data.
                Default: "dec"
            extra_axes (list-like, optional): array of extra axes to match against in shape (n_axes, 2), where the
                array elements give the following names:
                0: name of the column in data
                1: name of the error column in data - may simply be "None"
            assumed_position_systematic_error (float): assumed Gaussian systematic error on the position of stars in the
                catalogue. Useful when catalogues have positional errors to weight the closeness between matched and
                catalogue clusters.
                Default: 0.0, although this is *rarely zero* as many catalogues will do a shitty job of inferring the
                center of clusters!
            match_tidal_radius (bool): whether or not to also match with tidal radii. We'll specify the angular
                separation in terms of tidal radius between literature & found, too.
                Default: False

        """
        # Initialise the catalogue's name, cluster names and co-ordinates
        self.catalogue_name = catalogue_name
        self.names = data[key_name].to_numpy()
        self.coords = SkyCoord(
            ra=data[key_ra].to_numpy() << u.deg, dec=data[key_dec].to_numpy() << u.deg
        )
        self.assumed_position_systematic_error = assumed_position_systematic_error

        self.match_tidal_radius = match_tidal_radius
        if match_tidal_radius:
            self.tidal_radius_data = data[key_tidal_radius].to_numpy()

        # Read in any extra axes
        if extra_axes is None:
            self.n_extra_features = 0
            self.extra_axes_catalogue_column_names = None
            self.extra_axes_data = None
            self.extra_axes_data_errors = None
        else:
            self.n_extra_features = len(extra_axes)
            self.extra_axes_catalogue_column_names = [None] * self.n_extra_features
            self.extra_axes_data = [None] * self.n_extra_features
            self.extra_axes_data_errors = [None] * self.n_extra_features

            # Cycle over the axes, adding the data to the lists in np.ndarray format
            for i, an_axis in enumerate(extra_axes):
                self.extra_axes_catalogue_column_names[i] = an_axis[0]
                self.extra_axes_data[i] = np.asarray(data[an_axis[0]])

                # Deal with a possible lack of specified error information
                if an_axis[1] is None:
                    self.extra_axes_data_errors[i] = np.zeros(data.shape[0])
                else:
                    self.extra_axes_data_errors[i] = np.asarray(data[an_axis[1]])

        # Write the usual package-wide keys for clusters to the class
        self.ocelot_key_names = default_ocelot_key_names

    def override_default_ocelot_parameter_names(
        self, parameters_to_override: dict
    ) -> None:
        """A function for overriding the default parameter names assigned to open clusters by ocelot.calculate.
        Necessary for crossmatching with a catalogue produced by another module.

        Args:
            parameters_to_override (dict of str): parameters to override, in the format "parameter": "new_name". See
                ocelot.calculate.constants.default_ocelot_key_names for a list of all parameters that can be
                overwriten.

        Returns:
            None

        """
        for a_parameter in parameters_to_override:
            self.ocelot_key_names[a_parameter] = parameters_to_override[a_parameter]

    def _two_parameter_position_crossmatch(
        self, clusters: SkyCoord, maximum_separation: float
    ):
        """Todo fstring"""

        # Perform the matching
        (
            id_clusters,
            id_catalog,
            distances_2d,
            distances_3d,
        ) = self.coords.search_around_sky(clusters, maximum_separation * u.deg)

        return id_clusters, id_catalog, distances_2d

    @staticmethod
    def _calculate_sigma(
        crossmatched_value: float,
        catalog_value: float,
        random_error: float,
        systematic_error: float = 0.0,
    ):
        """A function for calculating sigma difference between a literature object and a crossmatching object. Can
        also consider uniformly distributed systematic errors."""
        # We calculate the difference, then reduce that difference by the systematic error, make sure the sigma is
        # still >= 0.0 (sigma=0.0 means difference is systematic-dominated) and then divide by the random error toooo
        return (
            np.clip(
                np.abs(catalog_value - crossmatched_value) - systematic_error,
                0.0,
                np.inf,
            )
            / random_error
        )

    def crossmatch(
        self,
        data_cluster: pd.DataFrame,
        extra_axes_keys: Union[Tuple[str], List[str]] = (),
        max_separation: float = 1.0,
        best_match_on: str = "mean_sigma",
        tidal_radius_mode: str = "max",
        matches_to_record: int = 2,
        max_sigma_threshold: float = 5,
        position_systematics: float = 0.0,
        extra_axes_systematics: Optional[Union[list, tuple, np.ndarray]] = None,
    ):
        """Crossmatches the stored catalogue with one/multiple clusters depending on what was specified by the
        user.

        Notes:
            - Typically, matches should be ordered best based on their mean sigma (e.g. a cluster with sigmas 0, 2 is a
                better match than one with 1.5, 1.5 sigmas), but should be decided matched or not based on their max
                sigma (a cluster with a large max sigma simply isn't a match.)

        Args:
            data_cluster (pd.DataFrame): data on the clusters to match against, in the shape (n_clusters, n_axes).
            extra_axes_keys (list, optional): a list of the names of extra axes to match against. Should be keys that
                can be found in ocelot.calculate.constants.default_ocelot_key_names. Default names can be overriden
                or added to with override_default_ocelot_parameter_names.
            max_separation (float in degrees): maximum separation on the sky to report matches for.
                Default: 1 degree (which should be ok most of the time)
            best_match_on (str): what to make the best match on. May use "just_position" or may use "max_sigma" or
                "mean_sigma" of all axes.
                Default: "mean_sigma"
            tidal_radius_mode (str): which tidal radius value to use or how. Allowed values:
                - "literature": uses the literature value only
                - "data":       uses the data value only
                - "max":        uses the max value of the two
                - "mean":       uses the mean value of the two
                Default: "max"
            matches_to_record (int): the maximum number of matches to report back for a given cluster.
                Default: 2
            max_sigma_threshold (float): the threshold maximum sigma level above which matches will not be considered
                a valid match.
                Default: 5.

        Returns:
            matches: a pd.DataFrame of all found matches, including data on the separations and probabilities of
                matches being true, of shape (n_matches, ? - the number of stats per match depends on extra_axes)
            match_statistics: a pd.DataFrame of all input clusters' total numbers of matches, useful for quick
                diagnostics of a crossmatch run, of shape (n_input_clusters, ?)

        """
        # -------------------------------------
        # INITIAL SETUP AND READING IN OF DATA
        # Check that the correct number of extra_axes_keys have been specified based on the initialisation
        if len(extra_axes_keys) != self.n_extra_features:
            raise ValueError(
                "length of extra_axes_keys does not match the number of extra features specified at "
                "initialisation of the catalogue."
            )

        # Setup the extra axes systematics frames
        if extra_axes_systematics is None:
            extra_axes_systematics = np.zeros(self.n_extra_features)

        # Grab a list of extra axes key names from the class' ocelot parameter dict
        # We'll store main parameters in even indices (0, 2, 4) and their errors at the next odd one (1, 3, 5)...
        extra_axes_ocelot_key_names = []
        for a_parameter in extra_axes_keys:
            extra_axes_ocelot_key_names.append(self.ocelot_key_names[a_parameter])
            extra_axes_ocelot_key_names.append(
                self.ocelot_key_names[a_parameter + "_error"]
            )

        # Grab all the data we need from data_cluster
        names = np.asarray(data_cluster[self.ocelot_key_names["name"]])
        ra = np.asarray(data_cluster[self.ocelot_key_names["ra"]])
        dec = np.asarray(data_cluster[self.ocelot_key_names["dec"]])
        extra_axes = np.asarray(data_cluster[extra_axes_ocelot_key_names])

        # -------------------------------------
        # 2D ON-SKY CROSSMATCHING
        # Gather all possible 2D matches below the threshold max_separation:
        cluster_skycoords = SkyCoord(ra=ra << u.deg, dec=dec << u.deg)
        id_clusters, id_catalog, distances_2d = self._two_parameter_position_crossmatch(
            cluster_skycoords, max_separation
        )

        # -------------------------------------
        # CREATION OF MATCH DATA DATAFRAME FOR POSITION
        # Turn this into a fancy DataFrame thing
        match_data = pd.DataFrame(
            {
                "name": names[id_clusters],
                "name_match": self.names[id_catalog],
                "angular_sep": distances_2d,
            }
        )

        # Calculate the error in the position estimate - firstly, find the positional errors on potential clusters in
        # ra, dec
        ra_error = np.sqrt(
            np.asarray(data_cluster.loc[id_clusters, self.ocelot_key_names["ra_error"]])
            ** 2
            + self.assumed_position_systematic_error**2
        )
        dec_error = np.sqrt(
            np.asarray(
                data_cluster.loc[id_clusters, self.ocelot_key_names["dec_error"]]
            )
            ** 2
            + self.assumed_position_systematic_error**2
        )

        # We only find the sigma values where the error isn't zero - else we get division by 0 =(
        # This should never happen, but we do the check here anyway!
        good_ra = ra_error != 0
        ra_sigma = np.empty(ra_error.shape)
        ra_sigma[good_ra] = self._calculate_sigma(
            ra[id_clusters],
            self.coords.ra.value[id_catalog],
            ra_error,
            systematic_error=position_systematics,
        )
        ra_sigma[np.invert(good_ra)] = 0.0

        good_dec = dec_error != 0
        dec_sigma = np.empty(dec_error.shape)
        dec_sigma[good_dec] = self._calculate_sigma(
            dec[id_clusters],
            self.coords.dec.value[id_catalog],
            dec_error,
            systematic_error=position_systematics,
        )
        dec_sigma[np.invert(good_dec)] = 0.0

        # Lastly, we can grab the total sigma values and convert this into a probability
        match_data["angular_sep_sigma"] = np.sqrt(ra_sigma**2 + dec_sigma**2)
        match_data["angular_sep_prob"] = 2 * norm.cdf(
            -match_data["angular_sep_sigma"], loc=0.0, scale=1.0
        )

        # And also do some stuff with the tidal radii
        if self.match_tidal_radius:
            radius_literature = self.tidal_radius_data[id_catalog]
            radius_data = data_cluster[
                self.ocelot_key_names["ang_radius_t"]
            ].to_numpy()[id_clusters]

            # Decide on which form to use
            if tidal_radius_mode == "literature":
                tidal_radii = radius_literature
            elif tidal_radius_mode == "data":
                tidal_radii = radius_data
            elif tidal_radius_mode == "mean":
                tidal_radii = np.mean(
                    np.vstack([radius_literature, radius_data]), axis=0
                )
            elif tidal_radius_mode == "max":
                tidal_radii = np.maximum(radius_literature, radius_data)
            elif tidal_radius_mode == "min":
                tidal_radii = np.minimum(radius_literature, radius_data)
            else:
                raise ValueError(
                    f"specified tidal_radius_mode '{tidal_radius_mode}' not recognised/supported!"
                )

            match_data["tidal_sep_ratio"] = match_data["angular_sep"] / tidal_radii
            match_data["tidal_sep_ratio_literature"] = (
                match_data["angular_sep"] / radius_literature
            )
            match_data["tidal_sep_ratio_data"] = match_data["angular_sep"] / radius_data

        # -------------------------------------
        # EXTENSION OF MATCH DATA FRAME TO THE EXTRA AXES
        # Incrementally work through the extra axes
        list_of_sigmas = []
        i_axis = 0
        while i_axis < self.n_extra_features:
            # Get the combined error with quadrature
            combined_error = np.sqrt(
                (self.extra_axes_data_errors[i_axis])[id_catalog] ** 2
                + extra_axes[id_clusters, 2 * i_axis + 1] ** 2
            )

            # Make axis names
            an_axis_name = extra_axes_keys[i_axis] + "_sep"
            an_axis_name_sigma = an_axis_name + "_sigma"
            list_of_sigmas.append(an_axis_name_sigma)

            # Calculate and store the separation
            match_data[an_axis_name] = np.abs(
                (self.extra_axes_data[i_axis])[id_catalog]
                - extra_axes[id_clusters, 2 * i_axis]
            )

            # Quantify the separation in terms of the error
            match_data[an_axis_name_sigma] = self._calculate_sigma(
                extra_axes[id_clusters, 2 * i_axis],
                (self.extra_axes_data[i_axis])[id_catalog],
                combined_error,
                systematic_error=extra_axes_systematics[i_axis],
            )

            i_axis += 1

        # Make some summary statistics that use all of the above
        if self.n_extra_features != 0 and best_match_on != "just_position":
            # angular_sep_sigma is irrelevant when working instead with tidal constraints, so we aren't interesting in
            # them.
            if best_match_on == "just_tidal_separation":
                match_data["max_sigma"] = np.max(match_data[list_of_sigmas], axis=1)
                match_data["mean_sigma"] = np.mean(match_data[list_of_sigmas], axis=1)

            else:
                match_data["max_sigma"] = np.max(
                    match_data[list_of_sigmas + ["angular_sep_sigma"]], axis=1
                )
                match_data["mean_sigma"] = np.mean(
                    match_data[list_of_sigmas + ["angular_sep_sigma"]], axis=1
                )
        else:
            match_data["max_sigma"] = match_data["angular_sep_sigma"]
            match_data["mean_sigma"] = match_data["angular_sep_sigma"]

        # -------------------------------------
        # CREATION OF SUMMARY DATAFRAME
        # See which clusters have matches in the matches returned
        cluster_has_a_match = pd.Series(names).isin(match_data["name"])
        names_of_clusters_with_matches = pd.Series(names)[cluster_has_a_match]

        # Also make an array of the total number of matches per cluster
        matches_per_cluster = np.zeros(names.shape, dtype=int)
        matches_per_cluster[cluster_has_a_match] = (match_data["name"].value_counts())[
            names_of_clusters_with_matches
        ]

        # Set which column we're gonna use to pick best matches
        # NOTE TO FUTURE ME: this should always be done with sigma values - *not* probabilities - as the best match is
        # then found later by sorting in *ascending* order. I.e. the "best thing" in a column must be the minimum thing.
        if best_match_on == "max_sigma":
            best_match_column = "max_sigma"
        elif best_match_on == "mean_sigma":
            best_match_column = "mean_sigma"
        elif best_match_on == "just_position":
            best_match_column = "angular_sep_sigma"
        elif best_match_on == "just_tidal_separation":
            best_match_column = "tidal_sep_ratio"
        else:
            raise ValueError(
                "specified best_match_on invalid: may only be one of 'max_sigma', 'mean_sigma', "
                "'just_position' or 'just_tidal_separation'."
            )

        # Create a summary DataFrame with the best matches for each cluster
        summary_match_data = pd.DataFrame(
            {
                "name": names,
                "valid_matches": np.zeros(matches_per_cluster.shape, dtype=int),
                "total_matches": matches_per_cluster,
            }
        )
        i_match = 0
        while i_match < matches_to_record:
            summary_match_data[f"match_{i_match}"] = np.nan
            summary_match_data[f"match_{i_match}_angular_sep"] = np.nan
            if self.match_tidal_radius:
                summary_match_data[f"match_{i_match}_tidal_sep_ratio"] = np.nan
            summary_match_data[f"match_{i_match}_max_sigma"] = np.nan
            summary_match_data[f"match_{i_match}_mean_sigma"] = np.nan
            i_match += 1

        # Cycle over clusters with matches, storing things about their best matches
        if self.match_tidal_radius:
            columns_to_read = [
                "name_match",
                "angular_sep",
                "tidal_sep_ratio",
                "max_sigma",
                "mean_sigma",
            ]
        else:
            columns_to_read = ["name_match", "angular_sep", "max_sigma", "mean_sigma"]

        for i_cluster in names_of_clusters_with_matches.index:
            # Make a new DataFrame of the current matches and get a sorted list of the best IDs
            current_matches = match_data.loc[
                match_data["name"] == names_of_clusters_with_matches[i_cluster], :
            ].reset_index(drop=True)
            best_match_ids = (
                current_matches[best_match_column].sort_values(ascending=True).index
            )

            # Count the number of matches below the sigma threshold
            summary_match_data.loc[i_cluster, "valid_matches"] = np.count_nonzero(
                np.asarray(current_matches["max_sigma"] < max_sigma_threshold)
            )

            # Add these matches to the main DataFrame
            i_match = 0
            while i_match < np.min([matches_to_record, len(best_match_ids)]):
                # Boilerplate to get the ID of the match and the prefix of the columns to write to
                an_id = best_match_ids[i_match]
                match = f"match_{i_match}"

                # And now, write the data to the overall DataFrame
                if self.match_tidal_radius:
                    columns_to_write = [
                        match,
                        match + "_angular_sep",
                        match + "_tidal_sep_ratio",
                        match + "_max_sigma",
                        match + "_mean_sigma",
                    ]
                else:
                    columns_to_write = [
                        match,
                        match + "_angular_sep",
                        match + "_max_sigma",
                        match + "_mean_sigma",
                    ]

                summary_match_data.loc[
                    i_cluster, columns_to_write
                ] = current_matches.loc[an_id, columns_to_read].values
                i_match += 1

        return match_data, summary_match_data


default_ocelot_key_names = {
    # Position
    "name": "name",
    "ra": "ra",
    "ra_error": "ra_error",
    "dec": "dec",
    "dec_error": "dec_error",

    # Angular size
    "ang_radius_50": "ang_radius_50",
    "ang_radius_50_error": "ang_radius_50_error",
    "ang_radius_c": "ang_radius_c",
    "ang_radius_c_error": "ang_radius_c_error",
    "ang_radius_t": "ang_radius_t",
    "ang_radius_t_error": "ang_radius_t_error",

    # Physical size
    "radius_50": "ang_radius_50",
    "radius_50_error": "ang_radius_50_error",
    "radius_c": "ang_radius_c",
    "radius_c_error": "ang_radius_c_error",
    "radius_t": "ang_radius_t",
    "radius_t_error": "ang_radius_t_error",

    # Distance
    "parallax": "parallax",
    "parallax_error": "parallax_error",
    "inverse_parallax": "inverse_parallax",
    "inverse_parallax_l68": "inverse_parallax_l68",
    "inverse_parallax_u68": "inverse_parallax_u68",
    "distance": "distance",
    "distance_error": "distance_error",

    # Proper motion and velocity
    "pmra": "pmra",
    "pmra_error": "pmra_error",
    "pmdec": "pmdec",
    "pmdec_error": "pmdec_error",
    "v_internal_tangential": "v_internal_tangential",
    "v_internal_tangential_error": "v_internal_tangential_error",

    # Diagnostics
    "parameter_inference_mode": "parameter_inference_mode"
}
