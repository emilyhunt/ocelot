"""Functions for computing the Gaia DR3 selection function and subsample selection
function.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import ocelot.simulate.cluster
from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d


DR3_SELECTION_FUNCTION = DR3SelectionFunctionTCG()


def gaia_selection_function(cluster, g_values):
    """Calculates the Gaia selection function of a cluster at cluster_info at the values
    g_values.
    """
    n_points = len(g_values)
    coords = SkyCoord(
        np.repeat(cluster.parameters.l, n_points),
        np.repeat(cluster.parameters.b, n_points),
        unit="deg",
        frame="galactic",
    ).transform_to("icrs")
    return {
        "prob": DR3_SELECTION_FUNCTION.query(coords, g_values),
        "std": np.zeros(len(g_values), dtype=float),
    }


def calculate_bin_centers(bin_edges):
    """Calculates the centers of a binned histogram."""
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def variable_bin_histogram(values, min, max, minimum_width, minimum_size=5):
    """Computes a variably binned histogram."""
    # First pass
    n_bins = int(np.round((max - min) / minimum_width)) + 1
    bins = np.linspace(min, max, num=n_bins)

    count, _ = np.histogram(values, bins=bins)

    # Early return condition if all bins are fine / minimum_size is zero
    if minimum_size == 0 or np.all(count >= minimum_size):
        return count, bins

    # Error check that should stop any bins from ever not being filled
    if np.sum(count) < minimum_size:
        raise ValueError(
            "Unable to fill bins due to fewer than minimum_size items in total"
        )

    # Otherwise, loop over all values, removing items until we get the desired bin
    # occupancies
    index = 0
    bins, count = list(bins), list(count)
    while index < len(count) - 2:
        if count[index] < minimum_size:
            count[index] += count[index + 1]
            count.pop(index + 1)
            bins.pop(index + 1)
        else:
            index += 1

    # Handle the final bin as a special case (since we have to go in reverse)
    while count[-1] < minimum_size:
        index = len(count) - 1
        count[index] += count[index - 1]
        count.pop(index - 1)
        bins.pop(index - 1)

    return np.asarray(count), np.asarray(bins)


def subsample_selection_function(
    data_gaia, g_min=2.0, g_max=21.0, bin_width=0.5, minimum_size=5
):
    """Estimates sf of a subsample. Uses method in https://arxiv.org/abs/2303.17738
    (Castro-Ginard+23)
    """
    # # Bin sample vs. subsample
    # n_bins = int(np.round((g_max - g_min) / bin_width)) + 1
    # bins = np.linspace(g_min, g_max, num=n_bins)
    # count, _ = np.histogram(data_gaia['phot_g_mean_mag'], bins=bins)

    count, bins = variable_bin_histogram(
        data_gaia["phot_g_mean_mag"], g_min, g_max, bin_width, minimum_size=minimum_size
    )
    count_cut, _ = np.histogram(
        data_gaia.loc[data_gaia["passes_cuts"], "phot_g_mean_mag"], bins=bins
    )

    # Do binomial probabilities
    probability = (count_cut + 1) / (count + 2)
    standard_deviation = np.sqrt(
        (count_cut + 1) * (count - count_cut + 1) / ((count + 2) ** 2 * (count + 3))
    )

    return {
        "prob": probability,
        "std": standard_deviation,
        "bins": bins,
        "count": count,
        "count_cut": count_cut,
    }


def calculate_selection_function(
    cluster: ocelot.simulate.cluster.SimulatedCluster,
    field: pd.DataFrame,
    g_min: int | float = 2,
    g_max: int | float = 21,
    subsample_bin_width=0.2,
    subsample_minimum_bin_size=10,
):
    """Calculates selection function of a given cluster."""
    subsample_sf = subsample_selection_function(
        field,
        g_min=g_min,
        g_max=g_max,
        bin_width=subsample_bin_width,
        minimum_size=subsample_minimum_bin_size,
    )
    g_bin_centers = calculate_bin_centers(subsample_sf["bins"])
    gaia_sf = gaia_selection_function(cluster, g_bin_centers)
    combined_sf = {
        "prob": gaia_sf["prob"] * subsample_sf["prob"],
        "std": np.sqrt(gaia_sf["std"] ** 2 + subsample_sf["std"] ** 2),
        "bins": subsample_sf["bins"],
        "bin_centers": g_bin_centers,
        "gaia_sf": gaia_sf,
        "subsample_sf": subsample_sf,
        "region_size": len(field),
    }
    return combined_sf


def interpolate_selection_function(
    g_magnitudes, probabilities, bounds_error=False, fill_value=0.0
):
    """Sets up an interpolator for a selection function"""
    return interp1d(
        g_magnitudes, probabilities, bounds_error=bounds_error, fill_value=fill_value
    )


def apply_selection_functions(
    cluster: ocelot.simulate.cluster.SimulatedCluster, field: None | pd.DataFrame = None
):
    """Applies selection functions to a cluster."""
    if not cluster.parameters.selection_effects:
        cluster.stars = len(cluster.cluster)
        cluster.parameters.n_stars = cluster.stars
        return
    
    if field is None:
        raise ValueError(
            "field containing stars was not specified, meaning it is not possible to "
            "estimate selection function for this cluster!"
        )

    # Setup the selection function
    combined_sf = calculate_selection_function(cluster, field)
    sf_interpolator = interpolate_selection_function(
        combined_sf["bin_centers"], combined_sf["prob"]
    )

    # Probabilistically calculate if stars are visible
    samples = cluster.random_generator.uniform(size=len(cluster.cluster))
    star_is_visible = samples < sf_interpolator(cluster.cluster["g_true"])

    if cluster.parameters.visible_stars_only:
        cluster.cluster = cluster.cluster.loc[star_is_visible].reset_index(drop=True)
        cluster.stars = len(cluster.cluster)
        cluster.parameters.n_stars = cluster.stars
        return
    
    cluster.cluster["visible"] = star_is_visible
    cluster.stars = np.sum(star_is_visible)
    cluster.parameters.n_stars = cluster.stars
