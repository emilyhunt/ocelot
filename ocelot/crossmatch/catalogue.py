"""A class for crossmatching clusters with literature catalogues."""

import numpy as np
import pandas as pd

from .crossmatch import _three_parameter_position_crossmatch
from .crossmatch import _backpropagate_cluster_epoch
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import norm
from typing import Optional, Union


class Catalogue:

    def __init__(self,
                 data: Union[pd.DataFrame],
                 catalogue_name: str,
                 key_name: str = "Name",
                 key_ra: str = "ra",
                 key_dec: str = "dec",
                 extra_axes: Optional[Union[list, tuple, np.ndarray]] = None,
                 store_kd_tree: bool = True):
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
            extra_axes (list-like, optional): array of extra axes to match against in shape (n_axes, 3), where the
                array elements give the following names:
                0: name of the column to put on the final crossmatch report
                1: name of the column in data
                2: name of the error column in data - may simply be "None"
            store_kd_tree (bool): whether or not to store a kdtree for the catalogue with the name catalogue_name. Will
                greatly speed up new calls with the crossmatch function/later calls with this catalogue, but it requires
                that no new catalogue with this name is created!
                Default: True

        """
        # Initialise the catalogue's name, cluster names and co-ordinates
        self.catalogue_name = catalogue_name
        self.names = data[key_name].to_numpy()
        self.coords = SkyCoord(ra=data[key_ra].to_numpy() << u.deg, dec=data[key_dec].to_numpy() << u.deg)
        self.store_kd_tree = store_kd_tree

        # Read in any extra axes
        if extra_axes is None:
            self.n_extra_features = 0
            self.extra_axes_names = None
            self.extra_axes_data = None
            self.extra_axes_data_errors = None
        else:
            self.n_extra_features = len(extra_axes)
            self.extra_axes_names = [None] * self.n_extra_features
            self.extra_axes_data = [None] * self.n_extra_features
            self.extra_axes_data_errors = [None] * self.n_extra_features

            # Cycle over the axes, adding the data to the lists in np.ndarray format
            for i, an_axis in enumerate(extra_axes):
                self.extra_axes_names[i] = an_axis[0]
                self.extra_axes_data[i] = data[an_axis[1]]

                # Deal with a possible lack of specified error information
                if an_axis[2] is None:
                    self.extra_axes_data_errors[i] = np.zeros(data.shape[0])
                else:
                    self.extra_axes_data_errors[i] = data[an_axis[2]]

    def _two_parameter_position_crossmatch(self, clusters: SkyCoord, maximum_separation: float):
        """Todo fstring
        """

        # Perform the matching
        id_clusters, id_catalog, distances_2d, distances_3d = self.coords.search_around_sky(
            clusters, maximum_separation * u.deg)

        return id_clusters, id_catalog, distances_2d

    def crossmatch(self,
                   names: Union[np.ndarray, pd.Series],
                   ra: Union[np.ndarray, pd.Series],
                   dec: Union[np.ndarray, pd.Series],
                   tidal_radii: Optional[Union[np.ndarray, pd.Series]] = None,
                   extra_axes: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                   max_separation: float = 1.,
                   best_match_on: str = "all",
                   matches_to_record: int = 2):
        """Crossmatches the stored catalogue with one/multiple clusters depending on what was specified by the
        user.

        Todo: add a way to have a "minimum error" in catalogue ra, dec position, so that very small clusters/partially
            detected clusters don't end up with stupidly small match probabilities.

        Args:
            names (np.ndarray, pd.Series): names of the possible clusters to match against.
            ra (np.ndarray, pd.Series): ras of the possible clusters to match against.
            dec (np.ndarray, pd.Series): declinations of the possible clusters to match against.
            tidal_radii (np.ndarray, pd.Series, optional): tidal radii of the clusters to match against in degrees.
                When specified, the distance from the match will also be given in dimensionless units of the tidal
                radius.
                Default: None
            extra_axes (np.ndarray, pd.DataFrame): array of data of all the extra axes, in the order originally
                specified upon class creation, in the shape (n_samples, 2 * n_features), where every second axis is the
                error on the prior axis (which may simply be an array of zeros).
                Default: None
            max_separation (float in degrees): maximum separation on the sky to report matches for.
                Default: 1 degree (which should be ok most of the time)
            best_match_on (str): what to make the best match on. May use "just_position" or may use "all" extra axes.
                Default: "all"
            matches_to_record (int): the maximum number of matches to report back for a given cluster.
                Default: 2

        Returns:
            matches: a pd.DataFrame of all found matches, including data on the separations and probabilities of
                matches being true, of shape (n_matches, ? - the number of stats per match depends on extra_axes)
            match_statistics: a pd.DataFrame of all input clusters' total numbers of matches, useful for quick
                diagnostics of a crossmatch run, of shape (n_input_clusters, 2)

        """
        # Todo: is this necessary? I think not, Series objects don't need .iloc. Only extra_axes needs this
        # Turn any remaining pd.Series into numpy arrays, meaning we can do indexing without needing .loc or .iloc
        # names = np.asarray(names)
        # ra = np.asarray(ra)
        # dec = np.asarray(dec)
        # tidal_radii = np.asarray(tidal_radii)
        extra_axes = np.asarray(extra_axes)

        # Gather all possible 2D matches below the threshold max_separation:
        cluster_skycoords = SkyCoord(ra=ra << u.deg, dec=dec << u.deg)
        id_clusters, id_catalog, distances_2d = self._two_parameter_position_crossmatch(
            cluster_skycoords, max_separation)

        # -------------------------------------
        # CREATION OF MATCH DATA DATAFRAME
        # Turn this into a fancy DataFrame thing
        match_data = pd.DataFrame({"name": names[id_clusters],
                                   "name_match": self.names[id_catalog],
                                   "angular_sep": distances_2d})

        # Add the separation in units of tidal radius as a proxy for a sigma
        match_data["angular_sep_tidal"] = match_data["angular_sep"] / tidal_radii[id_clusters]
        match_data["angular_sep_tidal_prob"] = 2 * norm.cdf(match_data["angular_sep_tidal"], loc=0.0, scale=1.0)

        # Incrementally work through the extra axes
        list_of_probabilities = []
        i_axis = 0
        while i_axis < self.n_extra_features:
            # Get the combined error with quadrature
            combined_error = np.sqrt(
                (self.extra_axes_data_errors[i_axis])[id_catalog]**2 + extra_axes[id_clusters, 2*i_axis + 1])

            # Make axis names
            an_axis_name = self.extra_axes_names[i_axis] + "_sep"
            an_axis_name_sigma = an_axis_name + "_sigma"
            an_axis_name_prob = an_axis_name + "_prob"
            list_of_probabilities.append(an_axis_name_prob)

            # Calculate and store the separation
            match_data[an_axis_name] = np.abs(
                (self.extra_axes_data[i_axis])[id_catalog] - extra_axes[id_clusters, 2*i_axis])

            # Quantify the separation in terms of the error
            match_data[an_axis_name_sigma] = match_data[an_axis_name] / combined_error

            # Also express this as a probability based on the value of a normal CDF
            match_data[an_axis_name_prob] = 2 * norm.cdf(-match_data[an_axis_name + "_sigma"], loc=0.0, scale=1.0)

            i_axis += 1

        # Make some summary statistics that use all of the above
        if self.n_extra_features != 0:
            match_data["mean_extra_probability"] = np.mean(match_data[list_of_probabilities], axis=1)
            match_data["mean_total_probability"] = np.mean(
                match_data[list_of_probabilities + ["angular_sep_tidal_prob"]], axis=1)
        else:
            match_data["mean_total_probability"] = match_data["angular_sep_tidal_prob"]

        # -------------------------------------
        # CREATION OF SUMMARY DATAFRAME
        # See which clusters have matches in the matches returned
        cluster_has_a_match = pd.Series(names).isin(match_data["name"])
        names_of_clusters_with_matches = pd.Series(names)[cluster_has_a_match]
        n_clusters_with_matches = np.asarray(cluster_has_a_match).nonzero()[0]

        # Also make an array of the total number of matches per cluster
        matches_per_cluster = np.zeros(names.shape)
        matches_per_cluster[cluster_has_a_match] = (match_data["name"].value_counts())[names_of_clusters_with_matches]

        # Create a summary DataFrame with the best matches for each cluster
        summary_match_data = pd.DataFrame({"name": names,
                                           "n_matches": matches_per_cluster})
        i_match = 0
        while i_match < matches_to_record:
            summary_match_data[f"match_{i_match}"] = ""
            summary_match_data[f"match_{i_match}_angular_sep"] = np.nan
            summary_match_data[f"match_{i_match}_total_prob"] = np.nan
            i_match += 1

        # Set which column we're gonna use to pick best matches
        if best_match_on == "all":
            best_match_column = "mean_total_probability"
        elif best_match_on == "just_position":
            best_match_column = "angular_sep_tidal_prob"
        else:
            raise ValueError("specified best_match_on invalid: may only be one of 'all' or 'just_position'.")

        # Cycle over clusters with matches, storing things about their best matches
        i_cluster = 0
        for i_cluster in cluster_has_a_match.index:
            # Make a new DataFrame of the current matches and get a sorted list of the best IDs
            current_matches = match_data.loc[match_data["name"] == names_of_clusters_with_matches[i_cluster], :]
            best_match_ids = current_matches[best_match_column].sort_values().index

            # Add these matches to the main DataFrame
            i_match = 0
            while i_match < np.min([matches_to_record, len(best_match_ids)]):
                # Boilerplate to get the ID of the match and the prefix of the columns to write to
                an_id = best_match_ids[i_match]
                match = f"match_{i_match}"

                # And now, write the data to the overall DataFrame
                summary_match_data.loc[i_cluster, [match, match + "_angular_sep", match + "_total_prob"]] = [
                    current_matches.loc[an_id, ["name", "angular_sep", best_match_column]]]
                i_match += 1

            i_cluster += 1

        return match_data, summary_match_data
