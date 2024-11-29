"""Binary star relations for clusters."""

from __future__ import annotations
import numpy as np
import pickle
import ocelot.simulate.cluster
from scipy.interpolate import interp1d
from numba import jit
from ocelot import DATA_PATH


def make_binaries(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Assigns certain stars in a cluster as binaries."""
    if not cluster.parameters.binary_stars:
        return

    multiplicity_relation_all, multiplicity_relation_resolved = (
        _get_multiplicity_relations(cluster)
    )

    _assign_number_of_companions(cluster, multiplicity_relation_all)

    # No need to go any further if nobody is going to be a binary
    if cluster.cluster["companions"].sum() == 0:
        return

    # Cycle over every star, giving it companions
    _convert_singles_to_systems(
        cluster,
        multiplicity_relation_all,
        multiplicity_relation_resolved,
    )


def _get_multiplicity_relations(cluster):
    multiplicity_relation_all_binaries = cluster.parameters.binary_star_relation(
        make_resolved_stars=False
    )
    multiplicity_relation_resolved_binaries = cluster.parameters.binary_star_relation(
        make_resolved_stars=True, distance=cluster.parameters.distance
    )
    return multiplicity_relation_all_binaries, multiplicity_relation_resolved_binaries


def _assign_number_of_companions(cluster, multiplicity_relation_all_binaries):
    """Assigns a number of companions to each star probabilistically."""
    expected_companions = multiplicity_relation_all_binaries.companion_star_fraction(
        cluster.cluster["mass"]
    )
    expected_multiplicity = multiplicity_relation_all_binaries.multiplicity_fraction(
        cluster.cluster["mass"]
    )
    n_stars = len(cluster.cluster)
    is_multiple = cluster.random_generator.uniform(size=n_stars) < expected_multiplicity
    total_multiples = is_multiple.sum()

    cluster.cluster["companions"] = np.zeros(n_stars, dtype=int)
    cluster.cluster.loc[is_multiple, "companions"] = (
        np.random.poisson(
            (expected_companions[is_multiple] / expected_multiplicity[is_multiple]) - 1,
            size=total_multiples,
        )
        + 1
    ).astype(int)


def _convert_singles_to_systems(
    cluster, multiplicity_relation_all_binaries, multiplicity_relation_resolved_binaries
):
    """Converts single stars to systems, based on precomputed appropriate numbers of
    companions to make for each one.

    # Todo tidy this function please it's a mess
    """
    # Ensure we start with the most massive star and go down
    cluster.cluster = cluster.cluster.sort_values("mass", ignore_index=True)

    query = "companions > 0"
    # Todo: add new limiting optimization here
    # if cluster.parameters.selection_effects:
    #     query += " and g_true < 21"
    indices_to_go_over = cluster.cluster.query(query).index.to_numpy()[::-1]

    masses = cluster.cluster["mass"].to_numpy()
    companions = cluster.cluster.loc[indices_to_go_over, "companions"].to_numpy()
    index_primary = np.full(len(cluster.cluster), -2)

    masses_repeated = np.repeat(masses[indices_to_go_over], companions)
    q_values = multiplicity_relation_all_binaries.random_q(
        masses_repeated, cluster.random_generator
    )

    # Todo: Whether or not stars are resolved is currently done independent of mass. This is wrong! Minimal impact on distant clusters but bad in other cases.
    q_resolved = multiplicity_relation_resolved_binaries.random_q(
        masses_repeated, cluster.random_generator
    )
    is_resolved = q_resolved < 0.1  # Weird thing with how the interp is specced

    cluster.cluster["index_primary"] = _convert_singles_numba(
        masses, companions, indices_to_go_over, index_primary, q_values, is_resolved
    )


@jit(nopython=True, cache=True)
def _convert_singles_numba(
    masses, companions, indices_to_go_over, index_primary, q_values, is_resolved
):
    """Fast inner loop for convert_singles_to_systems.

    It's very weirdly written so that it works with numba (I had to really wrestle with
    making it optimisation-friendly), sorry :D

    # Todo: make this function non-shit
    """
    index_into_companions = 0
    all_star_indices = np.arange(masses.size)
    valid_masses = np.ones(masses.size, dtype=np.bool_)

    for i_primary, n_companions in zip(indices_to_go_over, companions):
        # -------------
        # CHECKS
        # -------------
        # Skip stars that are already binaries
        if index_primary[i_primary] > -2:
            index_into_companions += n_companions
            continue
        # Can't run on the lowest-mass star
        if i_primary == 0:
            continue

        # -------------
        # GET DATA FROM OVERALL ARRAYS INTO SHORTER ONES
        # -------------
        primary_mass = masses[i_primary]
        valid_masses[i_primary] = False

        # Stop if we've ran out of valid potential companions
        if np.sum(valid_masses) == 0:
            return index_primary

        mass_companions = (
            q_values[index_into_companions : index_into_companions + n_companions]
            * primary_mass
        )
        resolved_companions = is_resolved[
            index_into_companions : index_into_companions + n_companions
        ]

        # Increment for next loop
        index_into_companions += n_companions

        # -------------
        # CYCLE OVER EVERY COMPANION AND SELECT
        # -------------
        # Get (unique) indices of companions
        for a_mass, a_resolved in zip(mass_companions, resolved_companions):
            # Find star closest in mass to the one required by q
            i_secondary = np.searchsorted(masses[valid_masses], a_mass)

            # Get a valid index for this star
            indices_to_search = all_star_indices[valid_masses]

            # Prevent issues if the best star is at the end of the array
            if i_secondary >= len(indices_to_search):
                i_secondary = len(indices_to_search) - 1

            # Now, finally, we can grab the index of the star and set that it's now a
            # secondary!
            i_secondary_converted = indices_to_search[i_secondary]
            valid_masses[i_secondary_converted] = False

            # If the star is resolved (randomly sampled for now), then assign it as
            # such; otherwise, set it to contribute to the primary star's magnitude
            # (i.e. be unresolved), meaning it later gets removed from the star list
            if a_resolved:
                index_primary[i_secondary_converted] = -1
            else:
                index_primary[i_secondary_converted] = i_primary

            # Stop if we've ran out of valid potential companions
            if np.sum(valid_masses) == 0:
                return index_primary

    return index_primary


# Zeropoints in the Vegamag system (see documentation table 5.2)
# These are for Gaia DR3!
G_ZP = 25.6874
BP_ZP = 25.3385
RP_ZP = 24.7479


def _mag_to_flux(magnitudes, zero_point):
    return 10 ** ((zero_point - magnitudes) / 2.5)


def _flux_to_mag(fluxes, zero_point):
    # We also handle negative fluxes here - in that case, it should just be inf
    good_fluxes = np.atleast_1d(fluxes > 0).flatten()
    magnitudes = (
        -2.5 * np.log10(np.atleast_1d(fluxes).flatten(), where=good_fluxes) + zero_point
    )
    magnitudes[np.invert(good_fluxes)] = np.inf
    return magnitudes


def _add_two_magnitudes(magnitude_1, magnitude_2):
    """Correct (simplified) equation to add two magnitudes.
    Source: https://www.astro.keele.ac.uk/jkt/pubs/JKTeq-fluxsum.pdf
    """
    return -2.5 * np.log10(10 ** (-magnitude_1 / 2.5) + 10 ** (-magnitude_2 / 2.5))



