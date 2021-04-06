"""Tools to find field stars around a list of cluster stars."""

import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_field_stars_around_clusters(data_rescaled: np.ndarray, labels, min_samples=10, overcalculation_factor=2.,
                                    min_field_stars=100, max_field_stars=500, n_jobs=1, nn_kwargs=None, max_iter=100,
                                    kd_tree=None, minimum_next_stars_to_check=10, cluster_nn_distance_type="internal",
                                    verbose=False):
    """Gets and returns a cloud of representative field stars around each reported cluster.

    Args:
        data_rescaled (np.ndarray): array of rescaled data, in shape (n_samples, n_features.)
        labels (np.ndarray): labels of the data with shape (n_samples,), in the sklearn format for density-based
            clustering with at least min_samples stars with a noise label of -1.
        min_samples (int): the min_samples nearest neighbour will be returned.
            Default: 10
        overcalculation_factor (float): min_samples*overcalculation_factor nearest neighbors of cluster stars will be
            searched to try and find compatible field stars to evaluate against for the field step.
            Default: 2.
        min_field_stars (int): minimum number of field stars to use in field calculations. If enough aren't found, then
            we'll traverse deeper into the k nearest neighbour graph to find more.
            Default: 100
        max_field_stars (int): as for min_field_stars but a max instead! We won't add more than this many stars.
            Must be less than min_field_stars.
            Default: 500
        n_jobs (int or None): number of jobs to use for calculating nn distances. -1 uses all cores. None uses 1.
            In general, unless the numbers of stars invovled are very large, it can actually be a lot quicker just to
            use one core.
            Default: 1
        nn_kwargs (dict): a dict of kwargs to pass to sklearn.neighbors.NearestNeighbors when running on
            cluster or field stars.
            Default: None
        max_iter (int): maximum number of iterations to run when trying to search for neighbours for a single cluster
            before we raise a RuntimeError.
            Default: 100
        kd_tree (sklearn.neighbors.NearestNeighbors): pre-initialised nearest neighbor searcher (a KD tree unless you
            mad) to look for field stars with. MUST have already been fit to data_rescaled!
            Default: None
        minimum_next_stars_to_check (int): minimum number of stars from a loop that we're checking next time around
            before we automatically increase the overcalculation_factor to ensure that we don't run out of field
            stars to look at.
            Default: 10
        cluster_nn_distance_type (str): type of cluster internal nearest neighbour distance to calculate. If "internal",
            then cluster nn distances will only be made for cluster stars. If "external", these nn distances will also
            include any nearby field stars, which may or may not produce more accurate tails of the cluster nn
            distribution. Depends on what you're going for, though.
            Default: 'internal'
        verbose (bool): whether or not to print to the console about the current cluster we're working on.
            Default: False

    Returns:
        a dict of cluster nn distances
        a dict of field nn distances
        (indices into labels of the field stars used for each field calculation)

    """
    # Process the default nn kwargs
    if nn_kwargs is None:
        nn_kwargs = {}
    nn_kwargs.update(n_neighbors=min_samples, n_jobs=n_jobs)

    # Grab unique labels
    unique_labels, unique_label_counts = np.unique(labels, return_counts=True)

    # Drop any -1 "clusters" (these are in fact noise as reported by e.g. DBSCAN)
    good_clusters = unique_labels != -1
    n_field_stars = unique_label_counts[np.invert(good_clusters)]
    unique_labels = unique_labels[good_clusters]
    unique_label_counts = unique_label_counts[good_clusters]

    # Check that no biscuitry is in progress
    if np.any(unique_label_counts < min_samples):
        raise ValueError(f"one of the reported clusters is smaller than the value of min_samples "
                         f"of {min_samples}! Method will fail.")
    if n_field_stars < len(labels) * 2/3:
        warnings.warn("fewer than 2/3rds of points are field stars! This may be too few to accurately find neighbours "
                      "of cluster points.", RuntimeWarning)
    if min_field_stars > max_field_stars:
        raise ValueError("min_field_stars may not be greater than max_field_stars!")
    if cluster_nn_distance_type == "internal":
        internal_cluster_nn_distances = True
    elif cluster_nn_distance_type == "external":
        internal_cluster_nn_distances = False
    else:
        raise ValueError("selected nearest neighbor distance mode for clusters not recognised! Must be one of "
                         "'internal' or 'external'.")

    # Let's make a KD tree for the entire field and fit it
    if kd_tree is None:
        field_nn_classifier = NearestNeighbors(**nn_kwargs)
        field_nn_classifier.fit(data_rescaled)
    else:
        field_nn_classifier = kd_tree

    # Cycle over each cluster, grabbing stats about its own nearest neighbor distances
    cluster_nn_distances_dict = {}
    field_nn_distances_dict = {}

    field_star_indices_dict = {}

    # Now, let's cycle over everything and find cluster nearest neighbour info
    for a_cluster in unique_labels:
        if verbose:
            print(f"\r    working on cluster {a_cluster}", end="")

        cluster_stars = labels == a_cluster

        # Grab nearest neighbor distances for the cluster & field
        if internal_cluster_nn_distances:
            cluster_nn_distances = _calculate_cluster_nn_distances_internal(data_rescaled, cluster_stars, nn_kwargs)
        else:
            cluster_nn_distances = _calculate_cluster_nn_distances_external(data_rescaled, cluster_stars,
                                                                            field_nn_classifier)

        # We have a lot of error handling for the field to make sure edge cases don't ruin us!
        current_overcalculation_factor = overcalculation_factor
        n_attempts = 0
        while n_attempts < 10:
            try:
                field_star_distances, field_star_indices = _calculate_field_nn_distances(
                    data_rescaled,
                    field_nn_classifier,
                    cluster_stars,
                    max_iter=max_iter,
                    min_field_stars=min_field_stars,
                    max_field_stars=max_field_stars,
                    min_samples=min_samples,
                    minimum_next_stars_to_check=minimum_next_stars_to_check,
                    overcalculation_factor=current_overcalculation_factor)
                n_attempts = 100

            except RuntimeError:
                n_attempts += 1
                if n_attempts >= 10:
                    raise RuntimeError("unable to find neighbors for cluster despite 10 attempts at doing so.")

                warnings.warn(f"failed to find enough neighbors for cluster {a_cluster}! "
                              f"Doubling the overcalculation factor and starting again.", RuntimeWarning)
                current_overcalculation_factor *= 2

        # Save these nearest neighbor distances
        cluster_nn_distances_dict[a_cluster] = cluster_nn_distances[:, min_samples - 1]
        field_nn_distances_dict[a_cluster] = field_star_distances
        field_star_indices_dict[a_cluster] = field_star_indices

    if verbose:
        print("\r    finished nn distance calculations!")

    # Return time!
    return cluster_nn_distances_dict, field_nn_distances_dict, field_star_indices_dict


def _calculate_field_nn_distances(data_rescaled, field_nn_classifier, cluster_stars, max_iter=100, min_field_stars=100,
                                  max_field_stars=500, min_samples=10, minimum_next_stars_to_check=10,
                                  overcalculation_factor=2.):
    """Function for getting the nearest neighbour distribution of field stars around a cluster."""
    # Setup for our loop'in
    current_overcalculation_factor = overcalculation_factor
    n_field_stars = 0
    field_star_distances = []
    field_star_indices = []

    cluster_stars = cluster_stars.nonzero()[0]
    next_stars_to_check = cluster_stars
    already_done_stars = np.asarray([], dtype=int)  # i.e. an array of everything we already calculated a d for

    i = 0
    while n_field_stars < min_field_stars:
        field_n_neighbors = int(np.round(min_samples * current_overcalculation_factor))
        stars_to_check = next_stars_to_check

        # Get some distances!
        field_nn_distances, field_nn_indices = field_nn_classifier.kneighbors(
            data_rescaled[stars_to_check], n_neighbors=field_n_neighbors)

        # Automatically yeet away from the cluster a bit, since most things within min_samples are likely to be
        # cluster stars.
        # if i == 0:
        #     field_nn_indices = field_nn_indices[:, min_samples:]

        # ------ FIND VALID NON-CLUSTER OBJECTS
        # Drop all objects that are connected to cluster stars and any non-unique objects
        # First, test if individual stars are good or bad
        valid_indices = np.isin(field_nn_indices, cluster_stars, invert=True)

        # Then, see if their row has enough non-cluster stars to be unpolluted
        # (we only do this if we aren't on the first step, as on the first step this doesn't matter since
        # our objective is just to move away)
        if i != 0:
            good_rows = np.all(valid_indices[:, :min_samples], axis=1)
            valid_indices = np.logical_and(good_rows.reshape(-1, 1), valid_indices)

        # Remove anything we've done before
        valid_indices[valid_indices] = np.isin(
            field_nn_indices[valid_indices], already_done_stars, invert=True)

        # ------ RECORD VALID STARS
        # If we're on our first run, then we'll loop around again and get some new distances for the
        # now-found field stars. If not, then these distances are for non-cluster stars and we can start
        # to seriously consider stopping the loop
        if i != 0:
            # Get the indices of the valid field stars and get rid of any that would push us over max_field_stars
            remaining_stars = max_field_stars - n_field_stars
            valid_field_stars_at_min_samples = \
                np.unique(valid_indices[:, min_samples - 1].nonzero()[0])[:remaining_stars]

            # Grab the valid field star distances
            valid_field_star_distances = field_nn_distances[
                valid_field_stars_at_min_samples, min_samples - 1]

            # Add them to the running order
            n_field_stars += len(valid_field_star_distances)
            field_star_distances.append(valid_field_star_distances)

            # field_star_indices.append(field_nn_indices[
            #                               valid_field_stars_at_min_samples, min_samples - 1])
            field_star_indices.append(stars_to_check[valid_field_stars_at_min_samples])

        # ------ PREP FOR NEXT LOOP
        # Look for indices of stars we can check next
        next_stars_to_check = np.unique(field_nn_indices[valid_indices.nonzero()])

        # If there aren't enough stars to check next, then use our old list and double the overcalculation factor
        # so that we can hopefully find more next time
        if len(next_stars_to_check) < minimum_next_stars_to_check:
            next_stars_to_check = stars_to_check
            current_overcalculation_factor *= 2

        if current_overcalculation_factor * min_samples > len(data_rescaled):
            raise RuntimeError("unable to find enough neighbors for this cluster, since n_neighbors > n_samples!")

        # Otherwise, we can record these stars as being already done!
        else:
            already_done_stars = np.append(already_done_stars, stars_to_check)

        # Quick check that the while loop isn't the longest thing ever lol
        i += 1
        if i >= max_iter:
            raise RuntimeError(f"unable to traverse the graph of field stars in {max_iter} iterations!")

    # Convert field_star_distances into a 1D array and not a Python list of arrays
    return np.hstack(field_star_distances), np.hstack(field_star_indices)


def _calculate_cluster_nn_distances_internal(data_rescaled, cluster_stars, nn_kwargs):
    """Function for getting the internal nearest neighbour distribution of stars within a cluster.
    (I.e. the distribution but only of cluster stars)
    """
    # Firstly, get the nearest neighbour distances for the cluster alone
    cluster_nn_classifier = NearestNeighbors(**nn_kwargs)
    cluster_nn_classifier.fit(data_rescaled[cluster_stars])

    # And get the nn distances!
    cluster_nn_distances, cluster_nn_indices = cluster_nn_classifier.kneighbors(data_rescaled[cluster_stars])

    return cluster_nn_distances


def _calculate_cluster_nn_distances_external(data_rescaled, cluster_stars, field_nn_classifier):
    """Function for getting the external nearest neighbour distribution of stars within a cluster.
    (I.e. the distribution of both cluster and field stars, may produce more or less accurate distribution tails)
    """
    cluster_nn_distances, cluster_nn_indices = field_nn_classifier.kneighbors(data_rescaled[cluster_stars])

    return cluster_nn_distances
