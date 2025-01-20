"""Methods to pair stars randomly picked from the IMF into binaries."""

from __future__ import annotations
import numpy as np
from ocelot.model.binaries import (
    BaseBinaryStarModelWithPeriods,
    BaseBinaryStarModelWithEccentricities,
)
import ocelot.simulate.cluster
from numba import jit


def make_binaries(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Assigns certain stars in a cluster as binaries."""
    if not cluster.features.binary_stars:
        return

    _assign_number_of_companions(cluster)

    # Cycle over every star, giving it companions
    _convert_singles_to_systems(cluster)


def _assign_number_of_companions(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Assigns a number of companions to each star probabilistically."""
    expected_companions = cluster.models.binaries.companion_star_frequency(
        cluster.cluster["mass"]
    )
    expected_multiplicity = cluster.models.binaries.multiplicity_fraction(
        cluster.cluster["mass"]
    )
    n_stars = len(cluster.cluster)
    is_multiple = cluster.random_generator.uniform(size=n_stars) < expected_multiplicity
    total_multiples = is_multiple.sum()

    cluster.cluster["companions"] = np.zeros(n_stars, dtype=int)
    cluster.cluster.loc[is_multiple, "companions"] = (
        cluster.random_generator.poisson(
            (expected_companions[is_multiple] / expected_multiplicity[is_multiple]) - 1,
            size=total_multiples,
        )
        + 1
    ).astype(int)


def _convert_singles_to_systems(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Converts single stars to systems, based on precomputed appropriate numbers of
    companions to make for each one.
    """
    indices_to_go_over = _filter_cluster(cluster)

    # Do some setup of arrays
    masses = cluster.cluster["mass"].to_numpy()
    companions = cluster.cluster.loc[indices_to_go_over, "companions"].to_numpy()
    masses_repeated = np.repeat(masses[indices_to_go_over], companions)

    # Query binary star relation to get ideal binary parameters for each primary
    mass_ratios, periods, eccentricites = _get_binary_parameters(
        cluster, masses_repeated
    )
    secondary_masses = masses_repeated * mass_ratios

    # Expensive (optimized) step to find the best secondaries for each primary
    cluster.cluster["index_primary"], index_into_parameters = _convert_singles_numba(
        masses, secondary_masses, companions, indices_to_go_over
    )

    # Also save orbital parameters of each primary
    _save_orbital_parameters(
        cluster, mass_ratios, periods, eccentricites, index_into_parameters
    )


def _save_orbital_parameters(
    cluster: ocelot.simulate.cluster.SimulatedCluster,
    mass_ratios: np.ndarray,
    periods: np.ndarray,
    eccentricites: np.ndarray,
    index_into_parameters: np.ndarray,
):
    # Some extra indexing faff as index_into_parameters includes many stars with '-1'
    # (i.e. not a binary)
    star_is_secondary = index_into_parameters > -1
    i_secondary = star_is_secondary.nonzero()[0]
    i_parameters = index_into_parameters[star_is_secondary]

    # Save!
    (
        cluster.cluster["mass_ratio"],
        cluster.cluster["period"],
        cluster.cluster["eccentricity"],
    ) = np.nan, np.nan, np.nan
    cluster.cluster.loc[i_secondary, "mass_ratio"] = mass_ratios[i_parameters]
    cluster.cluster.loc[i_secondary, "period"] = periods[i_parameters]
    cluster.cluster.loc[i_secondary, "eccentricity"] = eccentricites[i_parameters]

    # Finally, also save the ID of each primary - this is a bit safer.
    cluster.cluster["simulated_id_primary"] = -1
    cluster.cluster.loc[star_is_secondary, "simulated_id_primary"] = (
        cluster.cluster.loc[
            cluster.cluster.loc[star_is_secondary, "index_primary"].tolist(),
            "simulated_id",
        ].to_numpy()
    )


def _filter_cluster(cluster: ocelot.simulate.SimulatedCluster):
    """Filters cluster to only stars we're interested in, and returns a numba-friendly
    set of indices for them.
    """
    # Ensure we start with the most massive star and go down
    cluster.cluster = cluster.cluster.sort_values("mass", ignore_index=True)

    query = "companions > 0"
    indices_to_go_over = cluster.cluster.query(query).index.to_numpy()[::-1]
    return indices_to_go_over


def _get_binary_parameters(
    cluster: ocelot.simulate.SimulatedCluster, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetches parameters of the binary stars to simulate.

    Parameters
    ----------
    cluster : ocelot.simulate.SimulatedCluster
        Simulated cluster to work with.
    masses : np.ndarray
        Array of primary star masses.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Array of mass ratios
        Array of periods (measured in days)
        Array of eccentricities in range [0, 1)
    """
    if isinstance(cluster.models.binaries, BaseBinaryStarModelWithEccentricities):
        return cluster.models.binaries.random_binary(masses, seed=cluster.random_seed)

    eccentricities = np.zeros_like(masses)
    if isinstance(cluster.models.binaries, BaseBinaryStarModelWithPeriods):
        mass_ratios, periods = cluster.models.binaries.random_binary(
            masses, seed=cluster.random_seed
        )
        return mass_ratios, periods, eccentricities

    periods = np.zeros_like(masses)
    mass_ratios = cluster.models.binaries.random_mass_ratio(
        masses, seed=cluster.random_seed
    )
    return mass_ratios, periods, eccentricities


@jit(nopython=True, cache=True)
def _convert_singles_numba(
    masses,
    desired_secondary_masses,
    companions_per_primary,
    indices_of_stars_with_companions,
):
    """Fast inner loop for convert_singles_to_systems.

    It's very weirdly written so that it works with numba (I had to really wrestle with
    making it optimisation-friendly), sorry :D
    """
    # Initialisation
    starting_index_secondary_masses = 0

    # Helper array used to get original index after boolean indexing
    all_star_indices = np.arange(masses.size)

    # Boolean array saying which masses (stars) are available to be assigned
    valid_masses = np.ones(masses.size, dtype=np.bool_)
    n_valid_masses = masses.size

    # Index into masses that is the index of the primary star
    index_primary = np.full_like(all_star_indices, -1)

    # Index into q_values that is the index of the parameters of the binary
    index_into_parameters = np.full_like(all_star_indices, -1)

    for i_primary, n_companions in zip(
        indices_of_stars_with_companions, companions_per_primary
    ):
        # Skip stars that are already binaries
        if index_primary[i_primary] >= 0:
            starting_index_secondary_masses += n_companions
            continue
        # Can't run on the lowest-mass star
        if i_primary == 0:
            continue

        # Prevent this star from being picked as a binary
        valid_masses[i_primary] = False
        n_valid_masses -= 1

        # Calculate the value of the masses that we'd like to find
        masses_to_look_for = _get_mass_values(
            desired_secondary_masses,
            starting_index_secondary_masses,
            n_companions,
        )

        # Cycle over every companion and select best stars
        for i_parameters, a_mass in enumerate(masses_to_look_for):
            # Stop if we've ran out of valid potential companions
            if n_valid_masses == 0:
                break

            # Look for best potential companion
            i_secondary = _get_index_of_best_star(
                masses, all_star_indices, valid_masses, a_mass
            )

            # Save this result
            valid_masses[i_secondary] = False
            n_valid_masses -= 1
            index_primary[i_secondary] = i_primary
            index_into_parameters[i_secondary] = (
                i_parameters + starting_index_secondary_masses
            )

        # Setup for the next loop
        starting_index_secondary_masses += n_companions

        # Stop if we've ran out of valid potential companions
        if n_valid_masses == 0:
            break

    return index_primary, index_into_parameters


@jit(nopython=True, cache=True)
def _get_index_of_best_star(
    masses, all_star_indices, valid_masses, closest_mass_to_find
):
    """Finds the index of the valid star closest in mass to the one"""
    i_secondary_on_valid_array = np.searchsorted(
        masses[valid_masses], closest_mass_to_find
    )

    # Now we need to convert this to an index back in the main index space
    # (i.e. 0 to n_stars, not 0 to n_valid_stars)
    # We start by preventing issues if the best star is at the end of the array, i.e.
    # all stars are smaller than closest_mass_to_find - in that case, we pick the
    # smallest available star (the last one)
    indices_to_search = all_star_indices[valid_masses]
    if i_secondary_on_valid_array >= len(indices_to_search):
        i_secondary_on_valid_array = len(indices_to_search) - 1

    # Now, finally, we can grab the index of the star!
    return indices_to_search[i_secondary_on_valid_array]


@jit(nopython=True, cache=True)
def _get_mass_values(
    desired_secondary_masses, starting_index_secondary_masses, n_companions
):
    """Calculates mass values we need to search for based on the mass ratios."""
    return desired_secondary_masses[
        starting_index_secondary_masses : starting_index_secondary_masses + n_companions
    ]
